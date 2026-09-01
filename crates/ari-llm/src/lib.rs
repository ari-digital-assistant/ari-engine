//! On-device LLM fallback for Ari.
//!
//! When no skill matches the user's input, the engine can optionally hand it
//! to a small on-device language model that answers general-knowledge
//! questions directly.
//!
//! The model uses a **lazy lifecycle**: it stays on disk until a query
//! actually misses all skills, loads on demand (~1-2 s cold start), then
//! unloads after 60 seconds of idle to free RAM.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{LlamaChatMessage, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;

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
    fn try_answer(
        &self,
        input: &str,
        skills: &[SkillInfo],
        locale: &str,
        history: &[(String, String)],
        facts: &[String],
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

    fn build_chat_prompt(&self, system: &str, history: &[(String, String)], user: &str) -> Option<String> {
        let tmpl = self.model.chat_template(None).ok()?;
        let mut messages = Vec::with_capacity(history.len() + 2);
        // Some Gemma chat templates raise on the system role outright
        // ("System role not supported") — drop the system message when
        // it's empty so apply_chat_template doesn't fall through to the
        // None branch and leave us sending an unwrapped prompt.
        if !system.is_empty() {
            messages.push(LlamaChatMessage::new("system".to_string(), system.to_string()).ok()?);
        }
        for (role, content) in history {
            messages.push(LlamaChatMessage::new(role.clone(), content.clone()).ok()?);
        }
        messages.push(LlamaChatMessage::new("user".to_string(), user.to_string()).ok()?);
        self.model.apply_chat_template(&tmpl, &messages, true).ok()
    }

    fn run_inference(&self, prompt: &str, stop_on_newline: bool) -> Result<String, LlmError> {
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(2048));

        let mut ctx = self
            .model
            .new_context(shared_backend()?, ctx_params)
            .map_err(|e| LlmError::Context(e.to_string()))?;

        let tokens = self
            .model
            .str_to_token(prompt, llama_cpp_2::model::AddBos::Always)
            .map_err(|e| LlmError::Context(e.to_string()))?;

        let mut batch = LlamaBatch::new(tokens.len() + MAX_GENERATION_TOKENS, 1);
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

        for _ in 0..MAX_GENERATION_TOKENS {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);

            if self.model.is_eog_token(token) {
                break;
            }

            let bytes = self
                .model
                .token_to_piece_bytes(token, 128, false, None)
                .unwrap_or_default();
            let piece = String::from_utf8_lossy(&bytes);
            output.push_str(&piece);

            if stop_on_newline {
                let cleaned = strip_thinking(&output);
                if !cleaned.is_empty() && cleaned.contains('\n') {
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
        Ok(output)
    }
}

// ── Lazy LLM fallback ───────────────────────────────────────────────────

/// Lazy-loading LLM fallback. The model stays on disk until a query
/// actually misses all skills, then loads on demand. After
/// [`IDLE_TIMEOUT_SECS`] of inactivity the model is dropped and RAM is
/// freed. The next miss reloads it (cold start ~1-2 s on phone).
pub struct LazyLlmFallback {
    model_path: PathBuf,
    inner: Mutex<LazyState>,
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

impl LazyLlmFallback {
    /// Create a lazy fallback that will load from `model_path` on first use.
    /// This is cheap — no model loading happens here.
    pub fn new(model_path: &Path) -> Self {
        LazyLlmFallback {
            model_path: model_path.to_path_buf(),
            inner: Mutex::new(LazyState {
                loaded: None,
                last_used: None,
            }),
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
        let wrapping;
        let wrapped = match model.build_chat_prompt("", &[], prompt) {
            Some(p) => {
                wrapping = "native";
                p
            }
            None => {
                // llama-cpp-2 couldn't apply the GGUF's embedded chat
                // template (Gemma 4's Jinja can be too rich for minja).
                // All three tiers we ship are Gemma, so fall back to the
                // well-known Gemma turn-marker format manually. <bos> is
                // already prepended by AddBos::Always in str_to_token.
                wrapping = "manual_gemma";
                format!(
                    "<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
                )
            }
        };

        let output = model.run_inference(&wrapped, false)?;

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

        let idle_timeout = std::time::Duration::from_secs(IDLE_TIMEOUT_SECS);
        drop(state);

        let inner = &self.inner as *const Mutex<LazyState> as usize;
        std::thread::spawn(move || {
            std::thread::sleep(idle_timeout);
            // SAFETY: same as try_answer's eviction thread — the
            // LazyLlmFallback outlives any spawned timer.
            let mutex = unsafe { &*(inner as *const Mutex<LazyState>) };
            if let Ok(mut state) = mutex.lock() {
                if let Some(last) = state.last_used {
                    if last == last_used_at {
                        state.loaded = None;
                        state.last_used = None;
                    }
                }
            }
        });

        Ok(output)
    }

    fn try_answer(
        &self,
        input: &str,
        skills: &[SkillInfo],
        locale: &str,
        history: &[(String, String)],
        facts: &[String],
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

        let system_prompt = build_system_prompt(skills, locale, !history.is_empty(), facts);
        let user_prompt = build_user_prompt(input);

        let model = state.loaded.as_ref().unwrap();
        let prompt = match model.build_chat_prompt(&system_prompt, history, &user_prompt) {
            Some(p) => p,
            None => format!("{system_prompt}\n\nUser: {user_prompt}\n\nResponse: "),
        };

        let output = match model.run_inference(&prompt, true) {
            Ok(text) => strip_thinking(&text),
            Err(e) => {
                self.record_error(format!("run_inference failed: {e}"));
                return None;
            }
        };

        if parse_output(&output, skills).is_none() {
            // Record diagnostic about why parse_output rejected the
            // model's response. Common reasons: empty / "NONE" /
            // ≤10 chars after first-line trim.
            let preview: String = output.chars().take(120).collect();
            self.record_error(format!(
                "parse_output rejected response (output_len={}, first_120={preview:?})",
                output.len()
            ));
        }

        // Update last_used after inference (could have taken a while).
        let last_used_at = Instant::now();
        state.last_used = Some(last_used_at);

        // Schedule idle eviction.
        let idle_timeout = std::time::Duration::from_secs(IDLE_TIMEOUT_SECS);

        // Drop the lock before spawning the eviction thread.
        drop(state);

        let inner = &self.inner as *const Mutex<LazyState> as usize;
        std::thread::spawn(move || {
            std::thread::sleep(idle_timeout);
            // SAFETY: the LazyLlmFallback (and thus the Mutex) lives as long as
            // the engine, which outlives any eviction thread. The pointer is
            // only used to re-acquire the lock.
            let mutex = unsafe { &*(inner as *const Mutex<LazyState>) };
            if let Ok(mut state) = mutex.lock() {
                if let Some(last) = state.last_used {
                    if last == last_used_at {
                        // No query since we started the timer — evict.
                        state.loaded = None;
                        state.last_used = None;
                    }
                }
            }
        });

        parse_output(&output, skills)
    }

    fn last_error(&self) -> Option<String> {
        self.last_error.lock().ok().and_then(|g| g.clone())
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

    fn build_chat_prompt(&self, system: &str, history: &[(String, String)], user: &str) -> Option<String> {
        self.loaded.build_chat_prompt(system, history, user)
    }

    fn run_inference(&self, prompt: &str, stop_on_newline: bool) -> Result<String, LlmError> {
        self.loaded.run_inference(prompt, stop_on_newline)
    }
}

impl Fallback for LlmFallback {
    fn run_prompt(&self, prompt: &str) -> Result<String, LlmError> {
        let _guard = self
            .inference_lock
            .lock()
            .map_err(|_| LlmError::Backend("inference mutex poisoned".into()))?;

        let wrapped = match self.build_chat_prompt("", &[], prompt) {
            Some(p) => p,
            None => prompt.to_string(),
        };

        self.run_inference(&wrapped, false)
    }

    fn try_answer(
        &self,
        input: &str,
        skills: &[SkillInfo],
        locale: &str,
        _history: &[(String, String)],
        // Intentionally dropped here: this eager `LlmFallback` is only the
        // ari-cli dev-harness path. Production uses `LazyLlmFallback`, which
        // injects `_facts` into the system prompt. Passing `&[]` below is a
        // deliberate no-op for facts recall in the harness, not a parity bug.
        _facts: &[String],
    ) -> Option<FallbackResult> {
        let _guard = self.inference_lock.lock().ok()?;

        // `&[]`: facts recall is a no-op in this dev-harness impl (see `_facts`).
        let system_prompt = build_system_prompt(skills, locale, false, &[]);
        let user_prompt = build_user_prompt(input);

        let prompt = match self.build_chat_prompt(&system_prompt, &[], &user_prompt) {
            Some(p) => p,
            None => format!("{system_prompt}\n\nUser: {user_prompt}\n\nResponse: "),
        };

        let output = match self.run_inference(&prompt, true) {
            Ok(text) => strip_thinking(&text),
            Err(_) => return None,
        };

        parse_output(&output, skills)
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Remove `<think>…</think>` blocks from a model's raw output. Gemma 4
/// can emit a reasoning preamble before its real answer; the QA path
/// always strips, but Layer C calls this explicitly so it can also log
/// the raw output for diagnostics.
pub fn strip_thinking(raw: &str) -> String {
    let mut result = raw.to_string();
    while let Some(start) = result.find("<think>") {
        if let Some(end) = result.find("</think>") {
            let block_end = end + "</think>".len();
            result = format!("{}{}", &result[..start], &result[block_end..]);
        } else {
            result = result[..start].to_string();
            break;
        }
    }
    result.trim().to_string()
}

/// Per-locale system prompt baked into the binary at compile time
/// via `include_str!`. Adding a new locale is a `prompts/{locale}/`
/// directory + an arm in [`build_system_prompt`].
const PROMPT_EN_SYSTEM: &str = include_str!("../prompts/en/system.md");
const PROMPT_IT_SYSTEM: &str = include_str!("../prompts/it/system.md");

/// The QA-style system prompt for the on-device LLM fallback. Picks
/// the locale-specific template and trims trailing whitespace from
/// the file. Unknown locales fall back to English with a warn-log
/// (engine catches it via the LogSink). The locale fence is critical
/// for small models (Gemma 4 E2B occasionally bleeds in non-target-
/// language tokens like `不` for "no" when the prompt isn't language-
/// scoped).
fn build_system_prompt(_skills: &[SkillInfo], locale: &str, has_history: bool, facts: &[String]) -> String {
    let template = match locale {
        "en" => PROMPT_EN_SYSTEM,
        "it" => PROMPT_IT_SYSTEM,
        _ => {
            // No template for this locale — silently fall through to
            // English. Engine-level logging tracks the active locale
            // so the dev can see this is happening; we don't surface
            // it as a model error.
            PROMPT_EN_SYSTEM
        }
    };
    let mut prompt = template.trim().to_string();
    if has_history {
        prompt.push_str("\n\n");
        prompt.push_str(ari_core::CONTINUATION_INSTRUCTION);
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

fn parse_output(output: &str, _skills: &[SkillInfo]) -> Option<FallbackResult> {
    let line = output.lines().next()?.trim();

    if line.is_empty() || line == "NONE" {
        return None;
    }

    let text = line.strip_prefix("ANSWER:").unwrap_or(line).trim();

    if text.is_empty() || text == "NONE" || text.len() <= 10 {
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
        let prompt = build_system_prompt(&test_skills(), "en", false, &[]);
        assert!(prompt.contains("Ari"));
        // Locale fence — the explicit "English" word matters for
        // small models that occasionally bleed in non-English tokens.
        assert!(prompt.contains("English"));
        // Brevity hint stays loud.
        assert!(prompt.contains("one short"));
        assert!(prompt.contains("sentence"));
    }

    #[test]
    fn build_system_prompt_injects_facts_block() {
        let facts = vec!["i live in valletta".to_string()];
        let prompt = build_system_prompt(&[], "en", false, &facts);
        assert!(prompt.contains("Things you know about the user:\n- i live in valletta"));
    }

    #[test]
    fn build_system_prompt_no_facts_block_when_empty() {
        let prompt = build_system_prompt(&[], "en", false, &[]);
        assert!(!prompt.contains("Things you know about the user"));
    }

    #[test]
    fn system_prompt_forbids_bluffing_about_live_data_and_devices() {
        let p = build_system_prompt(&[], "en", false, &[]);
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

        let it = build_system_prompt(&[], "it", false, &[]);
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

    #[test]
    fn system_prompt_italian_is_localized() {
        let prompt = build_system_prompt(&test_skills(), "it", false, &[]);
        assert!(prompt.contains("Ari"));
        assert!(prompt.contains("italiano"));
        assert!(prompt.contains("frase breve"));
        // Italian template must NOT carry the English fence — that
        // would confuse the model into mixing languages.
        assert!(!prompt.contains("English"));
    }

    #[test]
    fn system_prompt_unknown_locale_falls_back_to_english() {
        let prompt = build_system_prompt(&test_skills(), "ja", false, &[]);
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
    fn rejects_short_output() {
        assert!(parse_output("lol what", &test_skills()).is_none());
        assert!(parse_output("", &test_skills()).is_none());
        assert!(parse_output("ok", &test_skills()).is_none());
    }

    #[test]
    fn takes_first_line_only() {
        let result = parse_output(
            "ANSWER:The answer.\nSome extra garbage the model spat out.",
            &test_skills(),
        );
        match result {
            Some(FallbackResult::DirectAnswer { text }) => {
                assert_eq!(text, "The answer.");
            }
            _ => panic!("expected DirectAnswer from first line"),
        }
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
        let system = build_system_prompt(&skills, "en", false, &[]);
        let user = build_user_prompt("what is the capital of australia");

        eprintln!("--- System prompt ---");
        eprintln!("{system}");
        eprintln!("--- User prompt ---");
        eprintln!("{user}");

        let prompt = match fallback.build_chat_prompt(&system, &[], &user) {
            Some(p) => {
                eprintln!("--- Chat template applied ---");
                eprintln!("{p}");
                p
            }
            None => {
                eprintln!("--- No chat template, using raw ---");
                let raw = format!("{system}\n\nUser: {user}\n\nResponse: ");
                eprintln!("{raw}");
                raw
            }
        };

        eprintln!("--- Running inference ---");
        let output = fallback.run_inference(&prompt, true).expect("inference failed");
        eprintln!("--- Raw output ---");
        eprintln!("[{output}]");

        let result = parse_output(&output, &skills);
        eprintln!("--- Parsed result ---");
        match &result {
            Some(FallbackResult::DirectAnswer { text }) => eprintln!("DirectAnswer: {text}"),
            None => eprintln!("None (no match)"),
        }
    }
}
