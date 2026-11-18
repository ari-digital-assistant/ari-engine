//! On-device LLM fallback for Ari.
//!
//! When no skill matches the user's input, the engine can optionally hand it
//! to a small on-device language model that answers general-knowledge
//! questions directly.
//!
//! The model uses a **lazy lifecycle**: it stays on disk until a query
//! actually misses all skills, loads on demand (~1-2 s cold start), then
//! unloads after 60 seconds of idle to free RAM.

mod cpu_build;
pub use cpu_build::{cpu_build, cpu_features, CpuBuild};

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{LlamaChatMessage, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::vocab::LlamaVocab;

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

    /// Drop the model's weights from memory but stay ready: the next
    /// question loads them again. For the phone running short of memory,
    /// where removing the fallback would switch the assistant off.
    /// Default does nothing, for impls that hold no model.
    fn release_memory(&self) {}
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

/// A loaded GGUF model ready for inference. Held transiently by
/// `LazyLlmFallback` and dropped when the idle timer fires.
struct LoadedModel {
    model: LlamaModel,
}

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

/// Process-wide shared `LlamaBackend`. llama.cpp's backend is a global
/// singleton — calling `LlamaBackend::init()` more than once fails
/// with `BackendAlreadyInitialized`, which previously broke any path
/// that loaded a second model after the first (e.g. the QA fallback's
/// first invocation when another model was already resident). Shared
/// access via `OnceLock` ensures every loader sees the same
/// already-initialized backend.
static SHARED_BACKEND: std::sync::OnceLock<LlamaBackend> = std::sync::OnceLock::new();

fn shared_backend() -> Result<&'static LlamaBackend, LlmError> {
    // Double-checked locking. `OnceLock::get_or_try_init` is still
    // nightly-only, so we fall back to an explicit mutex-guarded
    // init: fast path = lock-free read; slow path = serialise the
    // first init across threads so we don't race two LlamaBackend
    // creations.
    static INIT_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());
    if let Some(b) = SHARED_BACKEND.get() {
        return Ok(b);
    }
    let _guard = INIT_MUTEX
        .lock()
        .map_err(|_| LlmError::Backend("backend init mutex poisoned".into()))?;
    if let Some(b) = SHARED_BACKEND.get() {
        return Ok(b);
    }
    // The CPU backend first: without one, llama.cpp has nothing to run on.
    if cpu_build::ensure_loaded().is_none() {
        return Err(LlmError::Backend("no llama.cpp CPU build would load on this device".into()));
    }
    let backend = LlamaBackend::init().map_err(|e| LlmError::Backend(e.to_string()))?;
    let _ = SHARED_BACKEND.set(backend);
    SHARED_BACKEND
        .get()
        .ok_or_else(|| LlmError::Backend("backend set race lost".into()))
}

impl LoadedModel {
    fn load(model_path: &Path) -> Result<Self, LlmError> {
        let backend = shared_backend()?;
        let model_params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(backend, model_path, &model_params)
            .map_err(|e| LlmError::Model(e.to_string()))?;
        Ok(LoadedModel { model })
    }

    /// Wrap the conversation in the model's own chat template, rendered with
    /// minijinja. llama.cpp's built-in matcher only knows a fixed list of
    /// formats and Gemma 4 (Medium, Large) isn't on it: left to the matcher,
    /// those tiers get an unwrapped prompt and answer with nothing. The
    /// matcher stays as a second try for templates minijinja can't render.
    /// The error names both failures so a caller can log why it fell back.
    fn build_chat_prompt(&self, system: &str, history: &[(String, String)], user: &str) -> Result<String, String> {
        let tmpl = self.model.chat_template(None).map_err(|e| format!("no chat template: {e}"))?;
        let mut turns: Vec<(&str, &str)> = Vec::with_capacity(history.len() + 2);
        // Some Gemma chat templates raise on the system role outright
        // ("System role not supported") — drop the system message when
        // it's empty so the template doesn't reject the whole prompt.
        if !system.is_empty() {
            turns.push(("system", system));
        }
        turns.extend(history.iter().map(|(role, content)| (role.as_str(), content.as_str())));
        turns.push(("user", user));

        let vocab = self.model.vocab();
        let bos = token_text(&vocab, vocab.bos());
        let rendered = tmpl.to_str().map_err(|e| e.to_string()).and_then(|source| {
            render_chat_template(source, &turns, &bos, &token_text(&vocab, vocab.eos()))
        });
        let jinja_err = match rendered {
            // run_inference's tokenizer adds BOS when the model wants one, so a
            // template that prints it too would double it.
            Ok(prompt) if vocab.should_add_bos() => {
                return Ok(prompt.strip_prefix(bos.as_str()).map(str::to_string).unwrap_or(prompt))
            }
            Ok(prompt) => return Ok(prompt),
            Err(e) => e,
        };

        let messages = turns
            .iter()
            .map(|(role, content)| LlamaChatMessage::new(role.to_string(), content.to_string()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("template: {jinja_err}; matcher: {e}"))?;
        self.model
            .apply_chat_template(&tmpl, &messages, true)
            .map_err(|e| format!("template: {jinja_err}; matcher: {e}"))
    }

    fn token_count(&self, prompt: &str) -> usize {
        self.model.vocab().tokenize(prompt.as_bytes(), true, true).len()
    }

    /// Generate until the model ends its turn or [`max_tokens`] run out,
    /// showing `on_text` the output so far after every token.
    /// `stop_on_newline` also ends it at the first complete line, for a
    /// prompt sent without the model's chat template: left to run on, a
    /// model writes the next "User:" turn itself.
    fn run_inference(
        &self,
        prompt: &str,
        max_tokens: usize,
        stop_on_newline: bool,
        on_text: &mut dyn FnMut(&str),
    ) -> Result<Generation, LlmError> {
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(CONTEXT_TOKENS as u32));

        let mut ctx = self
            .model
            .new_context(shared_backend()?, ctx_params)
            .map_err(|e| LlmError::Context(e.to_string()))?;

        let tokens = self
            .model
            .vocab()
            .tokenize(prompt.as_bytes(), true, true);

        let mut batch = LlamaBatch::new(tokens.len() + max_tokens, 1);
        for (i, &token) in tokens.iter().enumerate() {
            let is_last = i == tokens.len() - 1;
            batch
                .add(token, i as i32, &[0], is_last)
                .map_err(|e| LlmError::Context(format!("batch add: {e}")))?;
        }

        ctx.decode(&mut batch)
            .map_err(|e| LlmError::Context(format!("decode prompt: {e}")))?;

        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::temp(0.7),
            LlamaSampler::greedy(),
        ]);

        let mut output = String::new();
        let mut n_cur = tokens.len() as i32;
        let mut cut_off = true;

        for _ in 0..max_tokens {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);

            if self.model.vocab().is_eog(token) {
                cut_off = false;
                break;
            }

            let bytes = self
                .model
                .vocab()
                .token_to_piece(token, false, None);
            let piece = String::from_utf8_lossy(&bytes);
            output.push_str(&piece);
            on_text(&output);

            if stop_on_newline {
                let cleaned = strip_thinking(&output);
                if !cleaned.is_empty() && cleaned.contains('\n') {
                    cut_off = false;
                    break;
                }
            }

            batch.clear();
            batch
                .add(token, n_cur, &[0], true)
                .map_err(|e| LlmError::Context(format!("batch add gen: {e}")))?;
            n_cur += 1;

            ctx.decode(&mut batch)
                .map_err(|e| LlmError::Context(format!("decode gen: {e}")))?;
        }

        // Return raw output — callers strip thinking blocks themselves
        // so Layer C can log both raw and stripped for diagnostics.
        Ok(Generation { text: output, cut_off })
    }
}

// ── Lazy LLM fallback ───────────────────────────────────────────────────

/// Lazy-loading LLM fallback. The model stays on disk until a query
/// actually misses all skills, then loads on demand. After
/// [`IDLE_TIMEOUT_SECS`] of inactivity the model is dropped and RAM is
/// freed. The next miss reloads it (cold start ~1-2 s on phone).
pub struct LazyLlmFallback {
    model_path: PathBuf,
    /// Shared with the idle-eviction timers, which hold it weakly.
    inner: Arc<Mutex<LazyState>>,
    /// Last error from a `try_answer` or `run_prompt` failure, surfaced
    /// via [`Fallback::last_error`] so the engine can log model-load /
    /// inference failures that the trait's `Option<FallbackResult>`
    /// shape would otherwise swallow.
    last_error: Mutex<Option<String>>,
}

struct LazyState {
    loaded: Option<LoadedModel>,
    last_used: Option<Instant>,
}

// SAFETY: LoadedModel fields (LlamaModel) are Send once
// loaded. All access is serialised through the Mutex.
unsafe impl Send for LazyLlmFallback {}
unsafe impl Sync for LazyLlmFallback {}
// SAFETY: as above. The eviction timer moves a handle to the state to its
// own thread, and only ever touches it with the Mutex held.
unsafe impl Send for LazyState {}

impl LazyLlmFallback {
    /// Create a lazy fallback that will load from `model_path` on first use.
    /// This is cheap — no model loading happens here.
    pub fn new(model_path: &Path) -> Self {
        LazyLlmFallback {
            model_path: model_path.to_path_buf(),
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
    /// raw stripped output. The prompt is wrapped in the model's chat
    /// template as a single user turn (no system prompt) so
    /// instruction-tuned models get the turn markers they expect.
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
            let loaded = LoadedModel::load(&self.model_path)?;
            state.loaded = Some(loaded);
        }

        let now = Instant::now();
        state.last_used = Some(now);

        let model = state.loaded.as_ref().unwrap();
        let (wrapped, wrapping) = match model.build_chat_prompt("", &[], prompt) {
            Ok(p) => (p, "native".to_string()),
            Err(e) => {
                // Neither minijinja nor llama.cpp's matcher could apply the
                // GGUF's chat template. Last resort: Gemma 3's turn markers,
                // the format of the Small tier. The tokenizer adds <bos>.
                (
                    format!("<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"),
                    format!("manual_gemma ({e})"),
                )
            }
        };

        let output = model.run_inference(&wrapped, MAX_GENERATION_TOKENS, false, &mut |_| {})?.text;

        // Surface a diagnostic when the model produces zero tokens.
        // Fold in whether the chat template applied and a head sample of
        // the wrapped prompt so logcat can tell us which path is broken.
        if output.is_empty() {
            let head: String = wrapped.chars().take(120).collect();
            return Err(LlmError::Backend(format!(
                "model emitted zero tokens; wrapping={wrapping}, wrapped_len={}, wrapped_head={head:?}",
                wrapped.len()
            )));
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
            match LoadedModel::load(&self.model_path) {
                Ok(m) => state.loaded = Some(m),
                Err(e) => {
                    self.record_error(format!(
                        "LoadedModel::load failed for {:?}: {e}",
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
        let wrap = |history: &[(String, String)]| {
            let system_prompt = build_system_prompt(skills, locale, tier, !history.is_empty(), facts, personality);
            match model.build_chat_prompt(&system_prompt, history, &user_prompt) {
                Ok(p) => (p, None),
                Err(e) => (format!("{system_prompt}\n\nUser: {user_prompt}\n\nResponse: "), Some(e)),
            }
        };
        let history = fit_history(history, |h| {
            model.token_count(&wrap(h).0) + ANSWER_MAX_TOKENS <= CONTEXT_TOKENS
        });
        let (prompt, template_err) = wrap(history);

        let output = match model.run_inference(&prompt, ANSWER_MAX_TOKENS, template_err.is_some(), on_text) {
            Ok(generation) => answer_text(&generation),
            Err(e) => {
                self.record_error(format!("run_inference failed: {e}"));
                return None;
            }
        };

        if parse_output(&output, skills).is_none() {
            // Record diagnostic about why parse_output rejected the
            // model's response. Common reasons: empty / "NONE" /
            // nothing but punctuation after trimming.
            let preview: String = output.chars().take(120).collect();
            let template = template_err
                .map(|e| format!("; chat template failed, prompt sent unwrapped: {e}"))
                .unwrap_or_default();
            self.record_error(format!(
                "parse_output rejected response (output_len={}, first_120={preview:?}){template}",
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

    fn release_memory(&self) {
        self.unload();
    }
}

// ── Eager LLM fallback (kept for CLI / tests) ──────────────────────────

/// Eagerly-loaded LLM fallback. Loads the model immediately and keeps it
/// in RAM until dropped. Used by the CLI where lazy lifecycle isn't needed.
pub struct LlmFallback {
    loaded: LoadedModel,
    inference_lock: Mutex<()>,
}

unsafe impl Send for LlmFallback {}
unsafe impl Sync for LlmFallback {}

impl LlmFallback {
    pub fn load(model_path: &Path) -> Result<Self, LlmError> {
        Ok(LlmFallback {
            loaded: LoadedModel::load(model_path)?,
            inference_lock: Mutex::new(()),
        })
    }

    fn build_chat_prompt(&self, system: &str, history: &[(String, String)], user: &str) -> Result<String, String> {
        self.loaded.build_chat_prompt(system, history, user)
    }

    fn run_inference(
        &self,
        prompt: &str,
        max_tokens: usize,
        stop_on_newline: bool,
        on_text: &mut dyn FnMut(&str),
    ) -> Result<Generation, LlmError> {
        self.loaded.run_inference(prompt, max_tokens, stop_on_newline, on_text)
    }
}

impl Fallback for LlmFallback {
    fn run_prompt(&self, prompt: &str) -> Result<String, LlmError> {
        let _guard = self
            .inference_lock
            .lock()
            .map_err(|_| LlmError::Backend("inference mutex poisoned".into()))?;

        let wrapped = self
            .build_chat_prompt("", &[], prompt)
            .unwrap_or_else(|_| prompt.to_string());

        self.run_inference(&wrapped, MAX_GENERATION_TOKENS, false, &mut |_| {}).map(|g| g.text)
    }

    fn try_answer(
        &self,
        input: &str,
        skills: &[SkillInfo],
        locale: &str,
        tier: BuiltinTier,
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
        let system_prompt = build_system_prompt(skills, locale, tier, false, &[], "");
        let user_prompt = build_user_prompt(input);

        let (prompt, templated) = match self.build_chat_prompt(&system_prompt, &[], &user_prompt) {
            Ok(p) => (p, true),
            Err(_) => (format!("{system_prompt}\n\nUser: {user_prompt}\n\nResponse: "), false),
        };

        let output = match self.run_inference(&prompt, ANSWER_MAX_TOKENS, !templated, on_text) {
            Ok(generation) => answer_text(&generation),
            Err(_) => return None,
        };

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

/// A special token's text, for templates that print `bos_token` or
/// `eos_token` themselves.
fn token_text(vocab: &LlamaVocab<'_>, token: LlamaToken) -> String {
    String::from_utf8_lossy(&vocab.token_to_piece(token, true, None)).into_owned()
}

/// Render a GGUF's Hugging Face chat template the way `transformers` does
/// (trim_blocks, lstrip_blocks, Python string and dict methods), ending in
/// the generation prompt with reasoning switched off. An on-device answer
/// is one spoken sentence: hidden reasoning costs seconds per answer and
/// adds nothing the user hears.
fn render_chat_template(source: &str, turns: &[(&str, &str)], bos: &str, eos: &str) -> Result<String, String> {
    let mut env = minijinja::Environment::new();
    env.set_syntax(
        minijinja::syntax::SyntaxConfig::builder()
            .trim_blocks(true)
            .lstrip_blocks(true)
            .build()
            .map_err(|e| e.to_string())?,
    );
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_function("raise_exception", |msg: String| -> Result<minijinja::Value, minijinja::Error> {
        Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, msg))
    });
    // `{% generation %}…{% endgeneration %}` is a transformers extension that
    // marks assistant text for training loss masks and renders nothing of its
    // own; LFM3's template uses it and minijinja has no such tag. An
    // always-true block keeps the tags' whitespace control.
    static GENERATION_TAG: std::sync::LazyLock<regex::Regex> =
        std::sync::LazyLock::new(|| regex::Regex::new(r"\{%(-?)\s*(end)?generation\s*(-?)%\}").unwrap());
    let source = GENERATION_TAG.replace_all(source, |caps: &regex::Captures| {
        let tag = if caps.get(2).is_some() { "endif" } else { "if true" };
        format!("{{%{} {tag} {}%}}", &caps[1], &caps[3])
    });
    let template = env.template_from_str(&source).map_err(|e| e.to_string())?;
    let messages: Vec<minijinja::Value> = turns
        .iter()
        .map(|(role, content)| minijinja::context! { role => *role, content => *content })
        .collect();
    template
        .render(minijinja::context! {
            messages => messages,
            add_generation_prompt => true,
            enable_thinking => false,
            bos_token => bos,
            eos_token => eos,
        })
        .map_err(|e| e.to_string())
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
    has_history: bool,
    facts: &[String],
    personality: &str,
) -> String {
    let small = tier == BuiltinTier::Small;
    let template = match (locale, small) {
        ("it", true) => PROMPT_IT_SYSTEM_SMALL,
        ("it", false) => PROMPT_IT_SYSTEM,
        // No template for other locales — silently fall through to
        // English. Engine-level logging tracks the active locale
        // so the dev can see this is happening; we don't surface
        // it as a model error.
        (_, true) => PROMPT_EN_SYSTEM_SMALL,
        (_, false) => PROMPT_EN_SYSTEM,
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
fn parse_output(output: &str, _skills: &[SkillInfo]) -> Option<FallbackResult> {
    let output = output.trim();
    let text = output.strip_prefix("ANSWER:").unwrap_or(output).trim();

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
        let prompt = build_system_prompt(&test_skills(), "en", BuiltinTier::Medium, false, &[], "");
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
            let small = build_system_prompt(&[], locale, BuiltinTier::Small, false, &[], "");
            assert!(small.contains(brevity), "{small}");
            let medium = build_system_prompt(&[], locale, BuiltinTier::Medium, false, &[], "");
            assert!(!medium.contains(brevity), "{medium}");
            assert_eq!(medium, build_system_prompt(&[], locale, BuiltinTier::Large, false, &[], ""));
        }
    }

    #[test]
    fn build_system_prompt_injects_facts_block() {
        let facts = vec!["i live in valletta".to_string()];
        let prompt = build_system_prompt(&[], "en", BuiltinTier::Medium, false, &facts, "");
        assert!(prompt.contains("Things you know about the user:\n- i live in valletta"));
    }

    #[test]
    fn personality_comes_after_the_rules_and_before_the_facts() {
        let facts = vec!["i live in valletta".to_string()];
        let prompt = build_system_prompt(&[], "en", BuiltinTier::Medium, true, &facts, "Dry and witty.");
        let rules = prompt.find("You cannot access live information").unwrap();
        let personality = prompt.find("Dry and witty.").unwrap();
        let known = prompt.find("Things you know about the user").unwrap();
        assert!(rules < personality && personality < known, "{prompt}");
    }

    #[test]
    fn no_personality_block_when_blank() {
        assert!(!build_system_prompt(&[], "en", BuiltinTier::Medium, false, &[], " ").contains("personality"));
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
        let prompt = build_system_prompt(&[], "en", BuiltinTier::Medium, false, &[], "");
        assert!(!prompt.contains("Things you know about the user"));
    }

    #[test]
    fn system_prompt_forbids_bluffing_about_live_data_and_devices() {
        for tier in [BuiltinTier::Small, BuiltinTier::Medium] {
            let p = build_system_prompt(&[], "en", tier, false, &[], "");
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

            let it = build_system_prompt(&[], "it", tier, false, &[], "");
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
        let prompt = build_system_prompt(&test_skills(), "it", BuiltinTier::Medium, false, &[], "");
        assert!(prompt.contains("Ari"));
        assert!(prompt.contains("italiano"));
        assert!(prompt.contains("un breve elenco che le copra tutte"));
        // Italian template must NOT carry the English fence — that
        // would confuse the model into mixing languages.
        assert!(!prompt.contains("English"));
    }

    #[test]
    fn system_prompt_unknown_locale_falls_back_to_english() {
        let prompt = build_system_prompt(&test_skills(), "ja", BuiltinTier::Medium, false, &[], "");
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

    #[test]
    fn chat_template_renders_generation_prompt_with_reasoning_off() {
        // The generation-prompt end of LFM3's template, cut down to what a
        // regression would break: the thinking switch and bos_token.
        let template = "{{- bos_token -}}\
            {%- for message in messages -%}\
            {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' -}}\
            {%- endfor -%}\
            {%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' -}}\
            {%- if enable_thinking -%}{{- '<think>' -}}{%- else -%}{{- '<think></think>' -}}{%- endif -%}\
            {%- endif -%}";
        let out = render_chat_template(template, &[("system", "Be brief."), ("user", "Hi")], "<s>", "</s>").unwrap();
        assert_eq!(
            out,
            "<s><|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think></think>"
        );
    }

    #[test]
    fn chat_template_trims_block_lines_like_transformers() {
        // Hugging Face templates are written for trim_blocks + lstrip_blocks:
        // lines holding only block tags must vanish entirely.
        let template = "{% for message in messages %}\n  {% if message.role == 'user' %}\n[{{ message.content }}]\n  {% endif %}\n{% endfor %}\n";
        let out = render_chat_template(template, &[("system", "S"), ("user", "Hi")], "", "").unwrap();
        assert_eq!(out, "[Hi]\n");
    }

    #[test]
    fn chat_template_supports_python_methods() {
        // Gemma 4's template calls dict.get and str.split, which plain
        // minijinja doesn't provide.
        let template = "{%- for message in messages -%}\
            {{- '<|turn>' + message['role'] + '\\n' + (message.get('content', '').split('<channel|>')[-1] | trim) + '<turn|>\\n' -}}\
            {%- endfor -%}\
            {%- if add_generation_prompt -%}{{- '<|turn>model\\n' -}}{%- endif -%}";
        let out = render_chat_template(
            template,
            &[("assistant", "<|channel>x<channel|>Old answer"), ("user", "Hi")],
            "",
            "",
        )
        .unwrap();
        assert_eq!(out, "<|turn>assistant\nOld answer<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n");
    }

    #[test]
    fn chat_template_accepts_transformers_generation_tag() {
        // LFM3's template wraps assistant turns in {% generation %}; without
        // handling, the whole template fails and reasoning can't be switched off.
        let template = "{%- for message in messages -%}\
            {%- if message.role == 'assistant' -%}{%- generation -%}{{- '[' + message.content + ']' -}}{%- endgeneration -%}\
            {%- else -%}{{- message.content -}}{%- endif -%}\
            {%- endfor -%}";
        let out = render_chat_template(template, &[("user", "Hi"), ("assistant", "Hello")], "", "").unwrap();
        assert_eq!(out, "Hi[Hello]");
    }

    #[test]
    fn chat_template_raise_exception_becomes_an_error() {
        let err = render_chat_template("{{ raise_exception('System role not supported') }}", &[("user", "Hi")], "", "")
            .unwrap_err();
        assert!(err.contains("System role not supported"), "{err}");
    }

    fn lazy_marked_used() -> (LazyLlmFallback, Instant) {
        let lazy = LazyLlmFallback::new(Path::new("/nonexistent/model.gguf"));
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
        let lazy = LazyLlmFallback::new(path);
        let ask = || lazy.try_answer("What's the capital of France?", &[], "en", BuiltinTier::Small, &[], &[], "", &mut |_| {});
        assert!(ask().is_some(), "{:?}", lazy.last_error());
        assert!(lazy.is_loaded());
        lazy.release_memory();
        assert!(!lazy.is_loaded());
        assert!(ask().is_some(), "{:?}", lazy.last_error());
        assert!(lazy.is_loaded());
    }

    #[test]
    #[ignore]
    fn real_model_debug_output() {
        let model_path = std::env::var("LLM_TEST_MODEL")
            .unwrap_or_else(|_| "/tmp/gemma3-1b-q4.gguf".to_string());
        let path = std::path::Path::new(&model_path);
        if !path.exists() {
            eprintln!("Model not found at {model_path}, skipping");
            return;
        }

        eprintln!("Loading model from {model_path}...");
        let fallback = LlmFallback::load(path).expect("failed to load model");
        eprintln!("Model loaded.");

        let skills = test_skills();
        let system = build_system_prompt(&skills, "en", BuiltinTier::Small, false, &[], "");
        let user = build_user_prompt("what is the capital of australia");

        eprintln!("--- System prompt ---");
        eprintln!("{system}");
        eprintln!("--- User prompt ---");
        eprintln!("{user}");

        let prompt = match fallback.build_chat_prompt(&system, &[], &user) {
            Ok(p) => {
                eprintln!("--- Chat template applied ---");
                eprintln!("{p}");
                p
            }
            Err(e) => {
                eprintln!("--- No chat template ({e}), using raw ---");
                let raw = format!("{system}\n\nUser: {user}\n\nResponse: ");
                eprintln!("{raw}");
                raw
            }
        };

        eprintln!("--- Running inference ---");
        let generation = fallback
            .run_inference(&prompt, ANSWER_MAX_TOKENS, false, &mut |_| {})
            .expect("inference failed");
        eprintln!("--- Raw output (cut off: {}) ---", generation.cut_off);
        eprintln!("[{}]", generation.text);
        let output = answer_text(&generation);

        let result = parse_output(&output, &skills);
        eprintln!("--- Parsed result ---");
        match &result {
            Some(FallbackResult::DirectAnswer { text }) => eprintln!("DirectAnswer: {text}"),
            None => eprintln!("None (no match)"),
        }
    }
}
