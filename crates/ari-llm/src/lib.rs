//! On-device LLM fallback for Ari.
//!
//! When no skill matches the user's input, the engine can optionally hand it
//! to a small on-device language model that answers general-knowledge
//! questions directly.
//!
//! The model uses a **lazy lifecycle**: it stays on disk until a query
//! actually misses all skills, loads on demand (~1-2 s cold start), then
//! unloads after 60 seconds of idle to free RAM.
//!
//! LiteRT-LM runs it, from a `.litertlm` or `.task` file (see [`litert`]).
//! llama.cpp, which ran GGUF files, went in ari-android#52.

mod cpu;
mod litert;
pub use cpu::cpu_features;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ── Public types ────────────────────────────────────────────────────────

/// Metadata the engine passes about each registered skill so the LLM can
/// decide whether to reroute.
pub struct SkillInfo {
    pub id: String,
    pub description: String,
}

/// What the LLM decided to do with the unmatched input.
pub enum FallbackResult {
    /// The LLM answered the question directly.
    DirectAnswer { text: String },
    /// The LLM wrote [`HANDOFF_MARKER`]: it doesn't know, or the question
    /// needs live information. `said` is whatever it wrote before the
    /// marker (usually nothing), `query` the question it wrote after it.
    DoesNotKnow { said: String, query: String },
}

/// What the on-device model writes, followed by the question, when it
/// can't answer and a cloud assistant could be offered instead. Only the
/// handoff prompts ask for it, in every language, so it stays English.
pub const HANDOFF_MARKER: &str = "QUERY:";

/// The label the prompts' examples put before each reply, which Gemma 3 1B
/// copies into its own now and then.
const RESPONSE_LABEL: &str = "Response:";

/// The part of the model's output so far that comes before
/// [`HANDOFF_MARKER`]: the marker and the question after it are for the
/// engine, never to be shown or spoken.
pub fn before_handoff(raw: &str) -> &str {
    let body = raw.trim_start();
    let body = body.strip_prefix(RESPONSE_LABEL).map_or(body, str::trim_start);
    let lead = raw.len() - body.len();
    handoff_at(body).map_or(raw, |at| &raw[..lead + at])
}

/// Where [`HANDOFF_MARKER`] starts in `text`: in capitals anywhere, or in
/// any case at the very start, where Gemma 3 1B writes "Query:" now and
/// then. A lowercase "query:" further in is just a word in an answer.
fn handoff_at(text: &str) -> Option<usize> {
    let leads = text
        .get(..HANDOFF_MARKER.len())
        .is_some_and(|head| head.eq_ignore_ascii_case(HANDOFF_MARKER));
    if leads { Some(0) } else { text.find(HANDOFF_MARKER) }
}

/// Trait so the engine can use a mock in tests.
pub trait Fallback: Send + Sync {
    /// Try answering the user's input using the on-device LLM.
    /// `locale` is the user's currently-active language (ISO 639-1
    /// lowercase) — the impl threads it into the system prompt's
    /// "respond in <language>" fence so the model doesn't bleed
    /// non-target-language tokens into the answer. Returns `None`
    /// if the model declined to answer or the call failed.
    ///
    /// `tier` picks the system prompt: Small keeps its own.
    ///
    /// `handoff` asks the model to write [`HANDOFF_MARKER`] rather than
    /// refuse when it doesn't know, because the engine can offer a cloud
    /// assistant instead.
    ///
    /// `on_text` sees the raw output so far, reasoning included, after
    /// every generated token, so the engine can speak finished sentences
    /// while the model writes the rest (ari-android#37).
    #[allow(clippy::too_many_arguments)]
    fn try_answer(
        &self,
        input: &str,
        skills: &[SkillInfo],
        locale: &str,
        tier: BuiltinTier,
        handoff: bool,
        history: &[(String, String)],
        facts: &[String],
        personality: &str,
        on_text: &mut dyn FnMut(&str),
    ) -> Option<FallbackResult>;

    /// Run an arbitrary prompt and return the raw stripped output. Used
    /// by Layer C to run on-device assistant consultation when the user
    /// has chosen the built-in LLM at medium or large tier. Default impl
    /// returns an error so test mocks and impls without a real model
    /// don't have to override.
    fn run_prompt(&self, _prompt: &str) -> Result<String, LlmError> {
        Err(LlmError::Backend(
            "run_prompt not supported by this Fallback".into(),
        ))
    }

    /// Last error from `try_answer` or `run_prompt`, if the most recent
    /// call failed. Used by the engine to surface model-load /
    /// inference failures to logcat — `try_answer` returns `None` on
    /// failure with no other signal, which is impossible to debug
    /// without this side channel. Default impl returns `None`.
    fn last_error(&self) -> Option<String> {
        None
    }

    /// What runs the loaded model, for logs: "LiteRT-LM GPU", "LiteRT-LM
    /// CPU", or the CPU with the reason the GPU didn't load.
    /// None while nothing is loaded. Default `None`, for impls with no model.
    fn runtime(&self) -> Option<String> {
        None
    }

    /// Drop the model's weights from memory but stay ready: the next
    /// question loads them again. For the phone running short of memory,
    /// where removing the fallback would switch the assistant off.
    /// Default does nothing, for impls that hold no model.
    fn release_memory(&self) {}
}

/// Where LiteRT-LM runs the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Accelerator {
    Cpu,
    /// Falls back to the CPU when the GPU won't load the model.
    Gpu,
}

/// Size classification of the loaded built-in model. Layer C uses this to
/// gate consultation: small is too dim for structured JSON, medium and
/// large are eligible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinTier {
    Small,
    Medium,
    Large,
}

impl BuiltinTier {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "small" => Some(Self::Small),
            "medium" => Some(Self::Medium),
            "large" => Some(Self::Large),
            _ => None,
        }
    }
}

// ── Errors ──────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum LlmError {
    Backend(String),
    Model(String),
    Context(String),
}

impl std::fmt::Display for LlmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LlmError::Backend(e) => write!(f, "llm backend init failed: {e}"),
            LlmError::Model(e) => write!(f, "llm model load failed: {e}"),
            LlmError::Context(e) => write!(f, "llm context creation failed: {e}"),
        }
    }
}

impl std::error::Error for LlmError {}

// ── Loaded model (internal) ─────────────────────────────────────────────

/// Maximum number of tokens we allow the model to generate per call.
/// Generous enough to fit Gemma 4's optional thinking mode preamble
/// plus the structured JSON output Layer C asks for.
const MAX_GENERATION_TOKENS: usize = 1024;

/// Context window the on-device model runs with: prompt, conversation and
/// answer together. A conversation that outgrows it loses its oldest turns
/// (see [`fit_history`]).
const CONTEXT_TOKENS: usize = 2048;

/// Most tokens an on-device answer may run to: a five-item list with a
/// line on each fits, and it matches the cloud assistants' default
/// `max_tokens`. An answer is spoken, so a runaway one costs the user
/// time twice, once generating and once listening.
const ANSWER_MAX_TOKENS: usize = 256;

/// What a generation produced, and whether it ran out of tokens before
/// the model ended its turn.
struct Generation {
    text: String,
    cut_off: bool,
}

/// How long the model stays loaded after the last query.
const IDLE_TIMEOUT_SECS: u64 = 60;

/// Load the model at `model_path` on `accelerator`. Only `.litertlm` and
/// `.task` files: a GGUF from before ari-android#52 has to be replaced.
fn load_model(model_path: &Path, accelerator: Accelerator) -> Result<litert::LoadedModel, LlmError> {
    if !litert::handles(model_path) {
        return Err(LlmError::Model(format!(
            "{} isn't a .litertlm or .task file, the only models Ari runs: update the model",
            model_path.display()
        )));
    }
    litert::LoadedModel::load(model_path, CONTEXT_TOKENS, accelerator).map_err(LlmError::Model)
}

/// The answer to `user` after as much of `history` as fits the context,
/// with the system prompt `system_for` gives for whether any history is
/// kept. The error is for `last_error`.
fn answer(
    model: &litert::LoadedModel,
    system_for: &dyn Fn(bool) -> String,
    history: &[(String, String)],
    user: &str,
    on_text: &mut dyn FnMut(&str),
) -> Result<String, String> {
    let history = fit_history(history, |h| {
        model
            .prompt_tokens(&system_for(!h.is_empty()), h, user)
            .is_ok_and(|n| n + ANSWER_MAX_TOKENS <= CONTEXT_TOKENS)
    });
    let generation = model.generate(&system_for(!history.is_empty()), history, user, ANSWER_MAX_TOKENS, on_text)?;
    Ok(answer_text(&generation))
}

// ── Lazy LLM fallback ───────────────────────────────────────────────────

/// Lazy-loading LLM fallback. The model stays on disk until a query
/// actually misses all skills, then loads on demand. After
/// [`IDLE_TIMEOUT_SECS`] of inactivity the model is dropped and RAM is
/// freed. The next miss reloads it (cold start ~1-2 s on phone).
pub struct LazyLlmFallback {
    model_path: PathBuf,
    accelerator: Accelerator,
    /// Shared with the idle-eviction timers, which hold it weakly.
    inner: Arc<Mutex<LazyState>>,
    /// Last error from a `try_answer` or `run_prompt` failure, surfaced
    /// via [`Fallback::last_error`] so the engine can log model-load /
    /// inference failures that the trait's `Option<FallbackResult>`
    /// shape would otherwise swallow.
    last_error: Mutex<Option<String>>,
}

struct LazyState {
    loaded: Option<litert::LoadedModel>,
    last_used: Option<Instant>,
}

// SAFETY: the loaded model's LiteRT-LM handles may be used from any thread
// but not from two at once; all access is serialised through the Mutex.
unsafe impl Send for LazyLlmFallback {}
unsafe impl Sync for LazyLlmFallback {}
// SAFETY: as above. The eviction timer moves a handle to the state to its
// own thread, and only ever touches it with the Mutex held.
unsafe impl Send for LazyState {}

impl LazyLlmFallback {
    /// Create a lazy fallback that will load from `model_path` on first use,
    /// on `accelerator`. This is cheap — no model loading happens here.
    pub fn new(model_path: &Path, accelerator: Accelerator) -> Self {
        LazyLlmFallback {
            model_path: model_path.to_path_buf(),
            accelerator,
            inner: Arc::new(Mutex::new(LazyState {
                loaded: None,
                last_used: None,
            })),
            last_error: Mutex::new(None),
        }
    }

    fn record_error(&self, msg: impl Into<String>) {
        if let Ok(mut g) = self.last_error.lock() {
            *g = Some(msg.into());
        }
    }

    fn clear_error(&self) {
        if let Ok(mut g) = self.last_error.lock() {
            *g = None;
        }
    }

    /// Force-unload the model, freeing RAM immediately.
    pub fn unload(&self) {
        if let Ok(mut state) = self.inner.lock() {
            state.loaded = None;
            state.last_used = None;
        }
    }

    /// Returns true if the model is currently loaded in RAM.
    pub fn is_loaded(&self) -> bool {
        self.inner
            .lock()
            .map(|s| s.loaded.is_some())
            .unwrap_or(false)
    }

    /// Drops the model `after` the use at `used_at` ([`IDLE_TIMEOUT_SECS`]
    /// in production), unless it was used again in the meantime. The timer holds the state weakly: once
    /// this fallback is gone (another model picked, or the assistant
    /// removed), it ends without touching anything. It used to hold a raw
    /// pointer, which a fallback dropped within the minute left dangling.
    fn schedule_eviction(&self, used_at: Instant, after: Duration) -> std::thread::JoinHandle<()> {
        let state = Arc::downgrade(&self.inner);
        std::thread::spawn(move || {
            std::thread::sleep(after);
            let Some(shared) = state.upgrade() else { return };
            let Ok(mut state) = shared.lock() else { return };
            if state.last_used == Some(used_at) {
                state.loaded = None;
                state.last_used = None;
            }
        })
    }
}

impl Fallback for LazyLlmFallback {
    /// Run an arbitrary prompt through the loaded model and return the
    /// raw output. The prompt goes in as a single user turn (no system
    /// prompt), which the model's own chat template wraps.
    /// Mirrors the lazy lifecycle of [`Self::try_answer`]: evicts on
    /// idle, loads on demand, schedules a 60-second eviction timer after
    /// each call. Serialised through the same mutex — concurrent callers
    /// queue.
    fn run_prompt(&self, prompt: &str) -> Result<String, LlmError> {
        let mut state = self
            .inner
            .lock()
            .map_err(|_| LlmError::Backend("inner mutex poisoned".into()))?;

        if let Some(last) = state.last_used {
            if last.elapsed().as_secs() >= IDLE_TIMEOUT_SECS {
                state.loaded = None;
            }
        }

        if state.loaded.is_none() {
            let loaded = load_model(&self.model_path, self.accelerator)?;
            state.loaded = Some(loaded);
        }

        let now = Instant::now();
        state.last_used = Some(now);

        let output = state
            .loaded
            .as_ref()
            .unwrap()
            .generate("", &[], prompt, MAX_GENERATION_TOKENS, &mut |_| {})
            .map_err(LlmError::Context)?
            .text;
        if output.is_empty() {
            return Err(LlmError::Backend("model emitted zero tokens".into()));
        }

        let last_used_at = Instant::now();
        state.last_used = Some(last_used_at);
        drop(state);
        self.schedule_eviction(last_used_at, Duration::from_secs(IDLE_TIMEOUT_SECS));

        Ok(output)
    }

    fn try_answer(
        &self,
        input: &str,
        skills: &[SkillInfo],
        locale: &str,
        tier: BuiltinTier,
        handoff: bool,
        history: &[(String, String)],
        facts: &[String],
        personality: &str,
        on_text: &mut dyn FnMut(&str),
    ) -> Option<FallbackResult> {
        self.clear_error();
        let mut state = match self.inner.lock() {
            Ok(g) => g,
            Err(_) => {
                self.record_error("inner mutex poisoned");
                return None;
            }
        };

        // Evict if idle too long.
        if let Some(last) = state.last_used {
            if last.elapsed().as_secs() >= IDLE_TIMEOUT_SECS {
                state.loaded = None;
            }
        }

        // Load on demand.
        if state.loaded.is_none() {
            match load_model(&self.model_path, self.accelerator) {
                Ok(m) => state.loaded = Some(m),
                Err(e) => {
                    self.record_error(format!(
                        "model load failed for {:?}: {e}",
                        self.model_path
                    ));
                    return None;
                }
            }
        }

        // Record usage time before borrowing the model.
        let now = Instant::now();
        state.last_used = Some(now);

        let user_prompt = build_user_prompt(input);
        let model = state.loaded.as_ref().unwrap();
        let system_for = |has_history| build_system_prompt(skills, locale, tier, handoff, has_history, facts, personality);
        let output = match answer(model, &system_for, history, &user_prompt, on_text) {
            Ok(answer) => answer,
            Err(e) => {
                self.record_error(e);
                return None;
            }
        };

        if parse_output(&output, skills).is_none() {
            // Record diagnostic about why parse_output rejected the
            // model's response. Common reasons: empty / "NONE" /
            // nothing but punctuation after trimming.
            let preview: String = output.chars().take(120).collect();
            self.record_error(format!(
                "parse_output rejected response (output_len={}, first_120={preview:?})",
                output.len()
            ));
        }

        // Update last_used after inference (could have taken a while), then
        // schedule idle eviction with the lock released.
        let last_used_at = Instant::now();
        state.last_used = Some(last_used_at);
        drop(state);
        self.schedule_eviction(last_used_at, Duration::from_secs(IDLE_TIMEOUT_SECS));

        parse_output(&output, skills)
    }

    fn last_error(&self) -> Option<String> {
        self.last_error.lock().ok().and_then(|g| g.clone())
    }

    fn runtime(&self) -> Option<String> {
        self.inner.lock().ok()?.loaded.as_ref().map(|m| m.runtime().to_string())
    }

    fn release_memory(&self) {
        self.unload();
    }
}

// ── Eager LLM fallback (kept for CLI / tests) ──────────────────────────

/// Eagerly-loaded LLM fallback. Loads the model immediately and keeps it
/// in RAM until dropped. Used by the CLI where lazy lifecycle isn't needed.
pub struct LlmFallback {
    loaded: litert::LoadedModel,
    inference_lock: Mutex<()>,
}

unsafe impl Send for LlmFallback {}
unsafe impl Sync for LlmFallback {}

impl LlmFallback {
    pub fn load(model_path: &Path, accelerator: Accelerator) -> Result<Self, LlmError> {
        Ok(LlmFallback {
            loaded: load_model(model_path, accelerator)?,
            inference_lock: Mutex::new(()),
        })
    }
}

impl Fallback for LlmFallback {
    fn runtime(&self) -> Option<String> {
        Some(self.loaded.runtime().to_string())
    }

    fn run_prompt(&self, prompt: &str) -> Result<String, LlmError> {
        let _guard = self
            .inference_lock
            .lock()
            .map_err(|_| LlmError::Backend("inference mutex poisoned".into()))?;

        self.loaded
            .generate("", &[], prompt, MAX_GENERATION_TOKENS, &mut |_| {})
            .map(|g| g.text)
            .map_err(LlmError::Context)
    }

    fn try_answer(
        &self,
        input: &str,
        skills: &[SkillInfo],
        locale: &str,
        tier: BuiltinTier,
        handoff: bool,
        _history: &[(String, String)],
        // Intentionally dropped here: this eager `LlmFallback` is only the
        // ari-cli dev-harness path. Production uses `LazyLlmFallback`, which
        // injects `_facts` into the system prompt. Passing `&[]` below is a
        // deliberate no-op for facts recall in the harness, not a parity bug.
        _facts: &[String],
        _personality: &str,
        on_text: &mut dyn FnMut(&str),
    ) -> Option<FallbackResult> {
        let _guard = self.inference_lock.lock().ok()?;

        // `&[]`: facts recall is a no-op in this dev-harness impl (see `_facts`).
        let system_for = |_| build_system_prompt(skills, locale, tier, handoff, false, &[], "");
        let user_prompt = build_user_prompt(input);
        let output = answer(&self.loaded, &system_for, &[], &user_prompt, on_text).ok()?;

        parse_output(&output, skills)
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Remove reasoning from a model's raw output, leaving only what the user
/// should hear: `<think>…</think>` blocks, and Gemma 4's
/// `<|channel>thought…<channel|>`. The QA path always strips, but Layer C
/// calls this explicitly so it can also log the raw output for diagnostics.
pub fn strip_thinking(raw: &str) -> String {
    let result = strip_reasoning(raw, "<|channel>", "<channel|>");
    strip_reasoning(&result, "<think>", "</think>").trim().to_string()
}

/// Strip `open…close` blocks from `text`. A closer with no opener before it
/// means the prompt opened the block (LFM3's template ends the generation
/// prompt in `<think>`, or `<think></think>` with reasoning off, and the
/// model still writes a draft answer, a stray `</think>`, then the answer
/// again), so everything before it goes. An opener that never closes is
/// reasoning still in progress, or cut off, so everything after it goes.
fn strip_reasoning(text: &str, open: &str, close: &str) -> String {
    let mut out = text.to_string();
    while let Some(close_at) = out.find(close) {
        let start = out[..close_at].rfind(open).unwrap_or(0);
        out.replace_range(start..close_at + close.len(), "");
    }
    if let Some(open_at) = out.find(open) {
        out.truncate(open_at);
    }
    out
}

/// Per-locale system prompt baked into the binary at compile time
/// via `include_str!`. Adding a new locale is a `prompts/{locale}/`
/// directory + an arm in [`build_system_prompt`].
const PROMPT_EN_SYSTEM: &str = include_str!("../prompts/en/system.md");
const PROMPT_IT_SYSTEM: &str = include_str!("../prompts/it/system.md");

/// The Small tier's prompt: one short sentence, and no word about answering
/// "best" questions from what it knows. Gemma 3 1B answered worse with the
/// Medium and Large prompt (Sydney for Australia's capital, "Response:"
/// read out before an answer), so it keeps this one (ari-android#37).
const PROMPT_EN_SYSTEM_SMALL: &str = include_str!("../prompts/en/system-small.md");
const PROMPT_IT_SYSTEM_SMALL: &str = include_str!("../prompts/it/system-small.md");

/// The same prompts for when a cloud assistant can take over: a question
/// needing live information gets [`HANDOFF_MARKER`] rather than a refusal,
/// and the weather example shows it. Device control still gets the
/// refusal, since no cloud assistant can do that either.
const PROMPT_EN_SYSTEM_HANDOFF: &str = include_str!("../prompts/en/system-handoff.md");
const PROMPT_IT_SYSTEM_HANDOFF: &str = include_str!("../prompts/it/system-handoff.md");
const PROMPT_EN_SYSTEM_SMALL_HANDOFF: &str = include_str!("../prompts/en/system-small-handoff.md");
const PROMPT_IT_SYSTEM_SMALL_HANDOFF: &str = include_str!("../prompts/it/system-small-handoff.md");

/// The QA-style system prompt for the on-device LLM fallback. Picks
/// the locale-specific template for the model's tier and trims trailing
/// whitespace from the file. Unknown locales fall back to English with a
/// warn-log (engine catches it via the LogSink). The locale fence is
/// critical for small models (Gemma 4 E2B occasionally bleeds in
/// non-target-language tokens like `不` for "no" when the prompt isn't
/// language-scoped).
fn build_system_prompt(
    _skills: &[SkillInfo],
    locale: &str,
    tier: BuiltinTier,
    handoff: bool,
    has_history: bool,
    facts: &[String],
    personality: &str,
) -> String {
    let small = tier == BuiltinTier::Small;
    let template = match (locale, small, handoff) {
        ("it", true, false) => PROMPT_IT_SYSTEM_SMALL,
        ("it", false, false) => PROMPT_IT_SYSTEM,
        ("it", true, true) => PROMPT_IT_SYSTEM_SMALL_HANDOFF,
        ("it", false, true) => PROMPT_IT_SYSTEM_HANDOFF,
        // No template for other locales — silently fall through to
        // English. Engine-level logging tracks the active locale
        // so the dev can see this is happening; we don't surface
        // it as a model error.
        (_, true, false) => PROMPT_EN_SYSTEM_SMALL,
        (_, false, false) => PROMPT_EN_SYSTEM,
        (_, true, true) => PROMPT_EN_SYSTEM_SMALL_HANDOFF,
        (_, false, true) => PROMPT_EN_SYSTEM_HANDOFF,
    };
    let mut prompt = template.trim().to_string();
    if has_history {
        prompt.push_str("\n\n");
        prompt.push_str(ari_core::HISTORY_INSTRUCTION);
    }
    if let Some(block) = ari_core::personality_block(personality) {
        prompt.push_str("\n\n");
        prompt.push_str(&block);
    }
    if let Some(block) = ari_core::remembered_facts_block(facts) {
        prompt.push_str("\n\n");
        prompt.push_str(&block);
    }
    prompt
}

fn build_user_prompt(input: &str) -> String {
    input.to_string()
}

/// The most recent part of `history` that `fits`, dropping the oldest
/// exchange (a user message and its reply) at a time. Long answers, or a
/// long conversation setting, can outgrow the on-device context, and a
/// prompt that overflows it fails outright.
fn fit_history<'a>(
    history: &'a [(String, String)],
    fits: impl Fn(&[(String, String)]) -> bool,
) -> &'a [(String, String)] {
    let mut history = history;
    while !history.is_empty() && !fits(history) {
        history = &history[history.len().min(2)..];
    }
    history
}

/// The user-facing text of an answer: reasoning stripped and, when the
/// token cap cut it off, the unfinished tail dropped.
fn answer_text(generation: &Generation) -> String {
    let text = strip_thinking(&generation.text);
    if generation.cut_off {
        drop_unfinished_tail(&text).to_string()
    } else {
        text
    }
}

/// Cut a truncated answer back to its last complete line, or failing
/// that its last complete sentence, so a list loses its half-written
/// last item rather than ending mid-word. Text with neither is kept whole.
fn drop_unfinished_tail(text: &str) -> &str {
    let text = text.trim_end();
    if let Some(at) = text.rfind('\n') {
        return text[..at].trim_end();
    }
    match text.rfind(['.', '!', '?']) {
        Some(at) => &text[..=at],
        None => text,
    }
}

/// The whole answer, every line of it: a list is a heading line and then
/// its items, so keeping only the first line kept only the heading
/// (ari-android#30).
///
/// Any answer with a word or number in it counts, however short. A
/// minimum of 11 characters, left from an older answer format, threw away
/// "Paris.", "1945." and "12." and told the user it hadn't understood
/// (ari-android#32).
///
/// [`HANDOFF_MARKER`] anywhere in it means the model doesn't know. Only
/// the marker itself counts: the models copy the prompt's capitals, and a
/// lowercase "query:" turns up in real answers about databases.
fn parse_output(output: &str, _skills: &[SkillInfo]) -> Option<FallbackResult> {
    let output = output.trim();
    let text = output.strip_prefix("ANSWER:").unwrap_or(output).trim();
    let text = text.strip_prefix(RESPONSE_LABEL).unwrap_or(text).trim();

    if let Some(at) = handoff_at(text) {
        // Small models wrap the marker in bold now and then.
        let markup = |c: char| c == '*' || c == '_' || c.is_whitespace();
        return Some(FallbackResult::DoesNotKnow {
            said: text[..at].trim_matches(markup).to_string(),
            query: text[at + HANDOFF_MARKER.len()..].trim_matches(markup).to_string(),
        });
    }

    if text == "NONE" || !text.chars().any(char::is_alphanumeric) {
        return None;
    }

    Some(FallbackResult::DirectAnswer {
        text: text.to_string(),
    })
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;


    fn test_skills() -> Vec<SkillInfo> {
        vec![
            SkillInfo { id: "current_time".into(), description: "Tells the current time.".into() },
            SkillInfo { id: "open".into(), description: "Opens apps by name.".into() },
            SkillInfo { id: "calculator".into(), description: "Evaluates math expressions.".into() },
        ]
    }

    #[test]
    fn system_prompt_english_is_concise_and_locale_fenced() {
        let prompt = build_system_prompt(&test_skills(), "en", BuiltinTier::Medium, false, false, &[], "");
        assert!(prompt.contains("Ari"));
        // Locale fence — the explicit "English" word matters for
        // small models that occasionally bleed in non-English tokens.
        assert!(prompt.contains("English"));
        // Brevity hint stays loud, with a list covering everything asked about.
        assert!(prompt.contains("briefly"));
        assert!(prompt.contains("a short list that covers every one of them"));
    }

    #[test]
    fn small_tier_keeps_its_one_sentence_prompt() {
        for (locale, brevity) in [("en", "one short English sentence"), ("it", "una sola frase breve")] {
            let small = build_system_prompt(&[], locale, BuiltinTier::Small, false, false, &[], "");
            assert!(small.contains(brevity), "{small}");
            let medium = build_system_prompt(&[], locale, BuiltinTier::Medium, false, false, &[], "");
            assert!(!medium.contains(brevity), "{medium}");
            assert_eq!(medium, build_system_prompt(&[], locale, BuiltinTier::Large, false, false, &[], ""));
        }
    }

    #[test]
    fn only_the_handoff_prompts_ask_for_the_marker() {
        for locale in ["en", "it"] {
            for tier in [BuiltinTier::Small, BuiltinTier::Medium, BuiltinTier::Large] {
                let without = build_system_prompt(&[], locale, tier, false, false, &[], "");
                assert!(!without.contains(HANDOFF_MARKER), "{without}");
                let with = build_system_prompt(&[], locale, tier, true, false, &[], "");
                assert!(with.contains("Response: QUERY: "), "{with}");
            }
        }
    }

    #[test]
    fn handoff_prompts_keep_each_tiers_brevity() {
        for (locale, brevity) in [("en", "one short English sentence"), ("it", "una sola frase breve")] {
            let small = build_system_prompt(&[], locale, BuiltinTier::Small, true, false, &[], "");
            assert!(small.contains(brevity), "{small}");
            let medium = build_system_prompt(&[], locale, BuiltinTier::Medium, true, false, &[], "");
            assert!(!medium.contains(brevity), "{medium}");
            assert_eq!(medium, build_system_prompt(&[], locale, BuiltinTier::Large, true, false, &[], ""));
        }
    }

    #[test]
    fn the_larger_models_handoff_prompt_still_refuses_device_control() {
        // No cloud assistant can switch a light on either, so the marker
        // would only lead to a second refusal.
        for (locale, refusal) in [
            ("en", "Response: I don't have a skill installed that can do that"),
            ("it", "Response: Non ho una skill installata per farlo"),
        ] {
            let medium = build_system_prompt(&[], locale, BuiltinTier::Medium, true, false, &[], "");
            assert_eq!(medium.matches(refusal).count(), 1, "{medium}");
        }
    }

    #[test]
    fn every_prompt_answers_figures_from_what_it_knows() {
        // A population counted as live information once: "I don't have live
        // population data" for Australia. A figure and an example settle it.
        for (locale, line, example) in [
            ("en", "populations, sizes and other figures", "About 68 million people live in France."),
            ("it", "popolazioni, dimensioni e altri dati", "In Francia vivono circa 68 milioni di persone."),
        ] {
            for tier in [BuiltinTier::Small, BuiltinTier::Medium] {
                for handoff in [false, true] {
                    let prompt = build_system_prompt(&[], locale, tier, handoff, false, &[], "");
                    assert!(prompt.contains(line), "{prompt}");
                    assert!(prompt.contains(example), "{prompt}");
                }
            }
        }
    }

    #[test]
    fn the_marker_alone_is_a_handoff_with_its_question() {
        match parse_output("QUERY: what's the weather in Rome?", &test_skills()) {
            Some(FallbackResult::DoesNotKnow { said, query }) => {
                assert_eq!(said, "");
                assert_eq!(query, "what's the weather in Rome?");
            }
            _ => panic!("expected a handoff"),
        }
    }

    #[test]
    fn words_before_the_marker_are_what_the_model_said() {
        match parse_output("I don't have live data. QUERY: population of Australia", &test_skills()) {
            Some(FallbackResult::DoesNotKnow { said, query }) => {
                assert_eq!(said, "I don't have live data.");
                assert_eq!(query, "population of Australia");
            }
            _ => panic!("expected a handoff"),
        }
    }

    #[test]
    fn an_echoed_label_and_bold_marker_are_not_part_of_what_was_said() {
        match parse_output("Response: **QUERY:** latest news", &test_skills()) {
            Some(FallbackResult::DoesNotKnow { said, query }) => {
                assert_eq!(said, "");
                assert_eq!(query, "latest news");
            }
            _ => panic!("expected a handoff"),
        }
    }

    #[test]
    fn a_capitalised_marker_counts_only_at_the_start() {
        match parse_output("Query: What's the latest news?", &test_skills()) {
            Some(FallbackResult::DoesNotKnow { said, query }) => {
                assert_eq!(said, "");
                assert_eq!(query, "What's the latest news?");
            }
            _ => panic!("expected a handoff"),
        }
        assert_eq!(before_handoff("Response: Query: news"), "Response: ");
        assert_eq!(before_handoff("Quer"), "Quer");
    }

    #[test]
    fn a_copied_response_label_is_not_said() {
        match parse_output("Response: The capital of Australia is Canberra.", &test_skills()) {
            Some(FallbackResult::DirectAnswer { text }) => assert_eq!(text, "The capital of Australia is Canberra."),
            _ => panic!("expected an answer"),
        }
    }

    #[test]
    fn a_lowercase_query_is_an_answer() {
        match parse_output("A SQL query: SELECT * FROM users.", &test_skills()) {
            Some(FallbackResult::DirectAnswer { text }) => assert_eq!(text, "A SQL query: SELECT * FROM users."),
            _ => panic!("expected an answer"),
        }
    }

    #[test]
    fn nothing_from_the_marker_on_is_released() {
        assert_eq!(before_handoff("QUERY: weather in Rome"), "");
        assert_eq!(before_handoff("Not sure. QUERY: weather"), "Not sure. ");
        assert_eq!(before_handoff("Paris is the capital."), "Paris is the capital.");
    }

    #[test]
    fn build_system_prompt_injects_facts_block() {
        let facts = vec!["i live in valletta".to_string()];
        let prompt = build_system_prompt(&[], "en", BuiltinTier::Medium, false, false, &facts, "");
        assert!(prompt.contains("Things you know about the user:\n- i live in valletta"));
    }

    #[test]
    fn personality_comes_after_the_rules_and_before_the_facts() {
        let facts = vec!["i live in valletta".to_string()];
        let prompt = build_system_prompt(&[], "en", BuiltinTier::Medium, false, true, &facts, "Dry and witty.");
        let rules = prompt.find("You cannot access live information").unwrap();
        let personality = prompt.find("Dry and witty.").unwrap();
        let known = prompt.find("Things you know about the user").unwrap();
        assert!(rules < personality && personality < known, "{prompt}");
    }

    #[test]
    fn no_personality_block_when_blank() {
        assert!(!build_system_prompt(&[], "en", BuiltinTier::Medium, false, false, &[], " ").contains("personality"));
    }

    #[test]
    fn history_loses_whole_exchanges_oldest_first_until_it_fits() {
        let turn = |role: &str, text: &str| (role.to_string(), text.to_string());
        let history = vec![
            turn("user", "one"),
            turn("assistant", "1"),
            turn("user", "two"),
            turn("assistant", "2"),
            turn("user", "three"),
            turn("assistant", "3"),
        ];
        let kept = fit_history(&history, |h| h.len() <= 3);
        assert_eq!(kept, &history[4..]);
        assert_eq!(fit_history(&history, |_| true), &history[..]);
        assert!(fit_history(&history, |_| false).is_empty());
    }

    #[test]
    fn build_system_prompt_no_facts_block_when_empty() {
        let prompt = build_system_prompt(&[], "en", BuiltinTier::Medium, false, false, &[], "");
        assert!(!prompt.contains("Things you know about the user"));
    }

    #[test]
    fn system_prompt_forbids_bluffing_about_live_data_and_devices() {
        for tier in [BuiltinTier::Small, BuiltinTier::Medium] {
            let p = build_system_prompt(&[], "en", tier, false, false, &[], "");
            assert!(
                p.contains("You cannot access live information"),
                "capability-honesty line missing from EN system prompt: {p}"
            );
            // A rule alone doesn't bind a small on-device Gemma (device
            // evidence: it still hallucinated a Rome temperature with the
            // rule-only copy) — the few-shot refusal + answerable pair is
            // the actual fix, so pin both examples down.
            assert!(
                p.contains("Response: I don't have a skill installed that can do that"),
                "few-shot refusal example missing from EN system prompt: {p}"
            );
            assert!(
                p.contains("Response: The sky is blue."),
                "few-shot answerable example missing from EN system prompt: {p}"
            );

            let it = build_system_prompt(&[], "it", tier, false, false, &[], "");
            assert!(
                it.contains("Non puoi accedere a informazioni in tempo reale"),
                "capability-honesty line missing from IT system prompt: {it}"
            );
            // A single memorised refusal example didn't generalise on-device
            // (gemma3-1b-q4 still hallucinated for a differently-phrased
            // weather question) — two refusal examples across different
            // domains (weather, device control) are needed to demonstrate
            // the REFUSAL PATTERN rather than one fixed sentence to match.
            // The weather example uses the apostrophe-free phrasing the
            // engine's normaliser actually feeds the model ("com e il tempo",
            // not "com'è il tempo").
            assert_eq!(
                it.matches("Response: Non ho una skill installata per farlo").count(),
                2,
                "expected two refusal examples (weather + device control) in IT system prompt: {it}"
            );
            assert!(
                it.contains("User: com e il tempo a Roma?"),
                "normalised-phrasing weather refusal example missing from IT system prompt: {it}"
            );
            assert!(
                it.contains("User: accendi le luci del soggiorno"),
                "device-control refusal example missing from IT system prompt: {it}"
            );
            assert!(
                it.contains("Response: Il cielo è blu."),
                "few-shot answerable example missing from IT system prompt: {it}"
            );
        }
    }

    #[test]
    fn system_prompt_italian_is_localized() {
        let prompt = build_system_prompt(&test_skills(), "it", BuiltinTier::Medium, false, false, &[], "");
        assert!(prompt.contains("Ari"));
        assert!(prompt.contains("italiano"));
        assert!(prompt.contains("un breve elenco che le copra tutte"));
        // Italian template must NOT carry the English fence — that
        // would confuse the model into mixing languages.
        assert!(!prompt.contains("English"));
    }

    #[test]
    fn system_prompt_unknown_locale_falls_back_to_english() {
        let prompt = build_system_prompt(&test_skills(), "ja", BuiltinTier::Medium, false, false, &[], "");
        // "Ja" isn't shipped — should silently get the en template.
        assert!(prompt.contains("Ari"));
        assert!(prompt.contains("English"));
    }

    #[test]
    fn parses_direct_answer() {
        let result = parse_output("ANSWER:Paris is the capital of France.", &test_skills());
        match result {
            Some(FallbackResult::DirectAnswer { text }) => {
                assert_eq!(text, "Paris is the capital of France.");
            }
            _ => panic!("expected DirectAnswer"),
        }
    }

    #[test]
    fn parses_none() {
        assert!(parse_output("NONE", &test_skills()).is_none());
    }

    #[test]
    fn parses_answer_without_prefix() {
        let result = parse_output("The capital of France is Paris.", &test_skills());
        match result {
            Some(FallbackResult::DirectAnswer { text }) => {
                assert_eq!(text, "The capital of France is Paris.");
            }
            _ => panic!("expected DirectAnswer"),
        }
    }

    #[test]
    fn rejects_empty_answer() {
        assert!(parse_output("ANSWER:", &test_skills()).is_none());
        assert!(parse_output("ANSWER:   ", &test_skills()).is_none());
    }

    #[test]
    fn keeps_short_answers() {
        for answer in ["Paris.", "1945.", "12.", "Yes.", "Tokyo"] {
            match parse_output(answer, &test_skills()) {
                Some(FallbackResult::DirectAnswer { text }) => assert_eq!(text, answer),
                _ => panic!("{answer:?} is a whole answer"),
            }
        }
    }

    #[test]
    fn rejects_output_with_no_words() {
        assert!(parse_output("", &test_skills()).is_none());
        assert!(parse_output("  \n ", &test_skills()).is_none());
        assert!(parse_output("...", &test_skills()).is_none());
    }

    #[test]
    fn keeps_every_line_of_a_list() {
        // ari-android#30: keeping the first line kept only the heading.
        let list = "Here are the top 5 fastest animals:\n1. Peregrine falcon\n2. Cheetah";
        match parse_output(list, &test_skills()) {
            Some(FallbackResult::DirectAnswer { text }) => assert_eq!(text, list),
            _ => panic!("expected DirectAnswer"),
        }
    }

    #[test]
    fn a_finished_answer_is_kept_whole() {
        let generation = Generation { text: "Heading:\n1. One\n2. Two".into(), cut_off: false };
        assert_eq!(answer_text(&generation), "Heading:\n1. One\n2. Two");
    }

    #[test]
    fn a_cut_off_list_loses_its_unfinished_item() {
        let generation = Generation { text: "Heading:\n1. One\n2. Two\n3. Th".into(), cut_off: true };
        assert_eq!(answer_text(&generation), "Heading:\n1. One\n2. Two");
    }

    #[test]
    fn a_cut_off_paragraph_ends_at_its_last_full_sentence() {
        assert_eq!(drop_unfinished_tail("It is blue. Light scatt"), "It is blue.");
        assert_eq!(drop_unfinished_tail("no sentence end at all"), "no sentence end at all");
    }

    #[test]
    fn strip_thinking_removes_think_blocks() {
        assert_eq!(
            strip_thinking("<think>They want a city.</think>Canberra is the capital."),
            "Canberra is the capital."
        );
    }

    #[test]
    fn strip_thinking_drops_the_draft_before_a_stray_closer() {
        // LFM3 with reasoning off: the prompt already ends in
        // <think></think>, yet the model writes the answer, a stray closer,
        // then the answer again. Seen in 11 of 20 answers on 2026-11-14.
        assert_eq!(
            strip_thinking("The Berlin Wall fell in 1989.</think>The Berlin Wall fell in 1989."),
            "The Berlin Wall fell in 1989."
        );
    }

    #[test]
    fn strip_thinking_cuts_reasoning_that_never_closes() {
        assert_eq!(strip_thinking("Sure.\n<think>still working"), "Sure.");
        assert_eq!(strip_thinking("<think>never finished"), "");
    }

    #[test]
    fn strip_thinking_removes_gemma4_thought_channel() {
        assert_eq!(
            strip_thinking("<|channel>thought\nThey ask about colour.<channel|>The sky is blue because of Rayleigh scattering."),
            "The sky is blue because of Rayleigh scattering."
        );
        assert_eq!(strip_thinking("<|channel>thought\nstill thinking"), "");
    }

    #[test]
    fn strip_thinking_leaves_plain_answers_alone() {
        assert_eq!(strip_thinking("  Jane Austen wrote it.  "), "Jane Austen wrote it.");
    }

    fn lazy_marked_used() -> (LazyLlmFallback, Instant) {
        let lazy = LazyLlmFallback::new(Path::new("/nonexistent/model.gguf"), Accelerator::Cpu);
        let used_at = Instant::now();
        lazy.inner.lock().unwrap().last_used = Some(used_at);
        (lazy, used_at)
    }

    #[test]
    fn eviction_timer_clears_an_unused_model() {
        let (lazy, used_at) = lazy_marked_used();
        lazy.schedule_eviction(used_at, Duration::from_millis(10)).join().unwrap();
        assert_eq!(lazy.inner.lock().unwrap().last_used, None);
    }

    #[test]
    fn eviction_timer_spares_a_model_used_since() {
        let (lazy, used_at) = lazy_marked_used();
        let timer = lazy.schedule_eviction(used_at, Duration::from_millis(10));
        let again = Instant::now();
        lazy.inner.lock().unwrap().last_used = Some(again);
        timer.join().unwrap();
        assert_eq!(lazy.inner.lock().unwrap().last_used, Some(again));
    }

    #[test]
    fn eviction_timer_outlives_a_dropped_fallback() {
        // The timer used to hold a raw pointer to the state, so dropping the
        // fallback inside the minute (switching models) left it dangling.
        let (lazy, used_at) = lazy_marked_used();
        let timer = lazy.schedule_eviction(used_at, Duration::from_millis(10));
        drop(lazy);
        timer.join().unwrap();
    }

    #[test]
    fn release_memory_with_nothing_loaded_is_harmless() {
        // Answering again after a release is covered, with a real model, by
        // real_model_release_memory_then_answer_again.
        let (lazy, _) = lazy_marked_used();
        lazy.release_memory();
        assert!(!lazy.is_loaded());
        assert_eq!(lazy.inner.lock().unwrap().last_used, None);
    }

    #[test]
    #[ignore]
    fn real_model_release_memory_then_answer_again() {
        let model_path = std::env::var("LLM_TEST_MODEL")
            .unwrap_or_else(|_| "/tmp/gemma3-1b-q4.gguf".to_string());
        let path = std::path::Path::new(&model_path);
        if !path.exists() {
            eprintln!("Model not found at {model_path}, skipping");
            return;
        }
        let lazy = LazyLlmFallback::new(path, Accelerator::Cpu);
        let ask = || lazy.try_answer("What's the capital of France?", &[], "en", BuiltinTier::Small, false, &[], &[], "", &mut |_| {});
        assert!(ask().is_some(), "{:?}", lazy.last_error());
        assert!(lazy.is_loaded());
        lazy.release_memory();
        assert!(!lazy.is_loaded());
        assert!(ask().is_some(), "{:?}", lazy.last_error());
        assert!(lazy.is_loaded());
    }

    /// A lazy fallback over `LLM_TEST_MODEL` (a GGUF, or a `.litertlm` or
    /// `.task` file, which also needs liblitert-lm.so on LD_LIBRARY_PATH)
    /// and the tier `LLM_TEST_TIER` names, or `None` with no model there.
    fn real_model() -> Option<(LazyLlmFallback, BuiltinTier)> {
        let model_path = std::env::var("LLM_TEST_MODEL").unwrap_or_else(|_| "/tmp/gemma3-1b-q4.gguf".to_string());
        if !Path::new(&model_path).exists() {
            eprintln!("Model not found at {model_path}, skipping");
            return None;
        }
        let tier = std::env::var("LLM_TEST_TIER").unwrap_or_else(|_| "small".to_string());
        let accelerator = if std::env::var("LLM_TEST_GPU").is_ok() { Accelerator::Gpu } else { Accelerator::Cpu };
        Some((LazyLlmFallback::new(Path::new(&model_path), accelerator), BuiltinTier::parse(&tier).expect("LLM_TEST_TIER")))
    }

    fn answer_of(result: Option<FallbackResult>, lazy: &LazyLlmFallback) -> String {
        match result {
            Some(FallbackResult::DirectAnswer { text }) => text,
            Some(FallbackResult::DoesNotKnow { said, query }) => panic!("handed off: said {said:?}, query {query:?}"),
            None => panic!("no answer: {:?}", lazy.last_error()),
        }
    }

    #[test]
    #[ignore]
    fn real_model_hands_off_a_live_question() {
        let Some((lazy, tier)) = real_model() else { return };
        let result = lazy.try_answer("What's the weather in Paris right now?", &[], "en", tier, true, &[], &[], "", &mut |_| {});
        match result {
            Some(FallbackResult::DoesNotKnow { .. }) => {}
            Some(FallbackResult::DirectAnswer { text }) => panic!("answered instead: {text}"),
            None => panic!("no answer: {:?}", lazy.last_error()),
        }
    }

    #[test]
    #[ignore]
    fn real_model_answers_a_slow_fact_even_when_it_could_hand_off() {
        let Some((lazy, tier)) = real_model() else { return };
        let result = lazy.try_answer("What is the population of Australia?", &[], "en", tier, true, &[], &[], "", &mut |_| {});
        let answer = answer_of(result, &lazy);
        assert!(answer.contains("million"), "{answer}");
    }

    #[test]
    #[ignore]
    fn real_model_answers_and_streams() {
        let Some((lazy, tier)) = real_model() else { return };
        let mut streamed = String::new();
        let result = lazy.try_answer("What's the capital of Australia?", &[], "en", tier, false, &[], &[], "", &mut |text| {
            streamed = text.to_string()
        });
        let answer = answer_of(result, &lazy);
        assert!(answer.contains("Canberra"), "{answer}");
        assert_eq!(strip_thinking(&streamed), answer);
    }

    #[test]
    #[ignore]
    fn real_model_answers_from_the_history() {
        let Some((lazy, tier)) = real_model() else { return };
        let history = [
            ("user".to_string(), "Please remember the number 4817.".to_string()),
            ("assistant".to_string(), "Okay, I'll remember 4817.".to_string()),
        ];
        let result = lazy.try_answer("What number did I ask you to remember?", &[], "en", tier, false, &history, &[], "", &mut |_| {});
        let answer = answer_of(result, &lazy);
        assert!(answer.contains("4817"), "{answer}");
    }

    #[test]
    #[ignore]
    fn real_model_runs_a_bare_prompt() {
        let Some((lazy, _)) = real_model() else { return };
        let output = lazy.run_prompt("Reply with the single word yes, in lower case.").expect("run_prompt");
        assert!(strip_thinking(&output).to_lowercase().contains("yes"), "{output}");
    }
}
