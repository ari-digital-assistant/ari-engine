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
    fn try_answer(&self, input: &str, skills: &[SkillInfo]) -> Option<FallbackResult>;

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
    backend: LlamaBackend,
    model: LlamaModel,
}

/// Maximum number of tokens we allow the model to generate per call.
/// Generous enough to fit Gemma 4's optional thinking mode preamble
/// plus the structured JSON output Layer C asks for.
const MAX_GENERATION_TOKENS: usize = 1024;

/// How long the model stays loaded after the last query.
const IDLE_TIMEOUT_SECS: u64 = 60;

impl LoadedModel {
    fn load(model_path: &Path) -> Result<Self, LlmError> {
        let backend =
            LlamaBackend::init().map_err(|e| LlmError::Backend(e.to_string()))?;
        let model_params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
            .map_err(|e| LlmError::Model(e.to_string()))?;
        Ok(LoadedModel { backend, model })
    }

    fn build_chat_prompt(&self, system: &str, user: &str) -> Option<String> {
        let tmpl = self.model.chat_template(None).ok()?;
        // Some Gemma chat templates raise on the system role outright
        // ("System role not supported") — drop the system message when
        // it's empty so apply_chat_template doesn't fall through to the
        // None branch and leave us sending an unwrapped prompt.
        let messages = if system.is_empty() {
            vec![LlamaChatMessage::new("user".to_string(), user.to_string()).ok()?]
        } else {
            vec![
                LlamaChatMessage::new("system".to_string(), system.to_string()).ok()?,
                LlamaChatMessage::new("user".to_string(), user.to_string()).ok()?,
            ]
        };
        self.model.apply_chat_template(&tmpl, &messages, true).ok()
    }

    fn run_inference(&self, prompt: &str, stop_on_newline: bool) -> Result<String, LlmError> {
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(2048));

        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
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
}

struct LazyState {
    loaded: Option<LoadedModel>,
    last_used: Option<Instant>,
}

// SAFETY: LoadedModel fields (LlamaBackend, LlamaModel) are Send once
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
        let wrapped = match model.build_chat_prompt("", prompt) {
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

    fn try_answer(&self, input: &str, skills: &[SkillInfo]) -> Option<FallbackResult> {
        let mut state = self.inner.lock().ok()?;

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
                Err(_) => return None,
            }
        }

        // Record usage time before borrowing the model.
        let now = Instant::now();
        state.last_used = Some(now);

        let system_prompt = build_system_prompt(skills);
        let user_prompt = build_user_prompt(input);

        let model = state.loaded.as_ref().unwrap();
        let prompt = match model.build_chat_prompt(&system_prompt, &user_prompt) {
            Some(p) => p,
            None => format!("{system_prompt}\n\nUser: {user_prompt}\n\nResponse: "),
        };

        let output = match model.run_inference(&prompt, true) {
            Ok(text) => strip_thinking(&text),
            Err(_) => return None,
        };

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

    fn build_chat_prompt(&self, system: &str, user: &str) -> Option<String> {
        self.loaded.build_chat_prompt(system, user)
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

        let wrapped = match self.build_chat_prompt("", prompt) {
            Some(p) => p,
            None => prompt.to_string(),
        };

        self.run_inference(&wrapped, false)
    }

    fn try_answer(&self, input: &str, skills: &[SkillInfo]) -> Option<FallbackResult> {
        let _guard = self.inference_lock.lock().ok()?;

        let system_prompt = build_system_prompt(skills);
        let user_prompt = build_user_prompt(input);

        let prompt = match self.build_chat_prompt(&system_prompt, &user_prompt) {
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

fn build_system_prompt(_skills: &[SkillInfo]) -> String {
    "You are Ari, a helpful voice assistant. Answer the user's question in one short sentence."
        .to_string()
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

// ── FunctionGemma skill router ─────────────────────────────────────────

use ari_core::{RouteResult, SkillRouter};

const ROUTER_MAX_TOKENS: usize = 60;

/// Lazy-loading FunctionGemma router. Same lifecycle pattern as
/// `LazyLlmFallback`: stays on disk until the first keyword-miss,
/// loads on demand, unloads after idle timeout.
pub struct FunctionGemmaRouter {
    model_path: PathBuf,
    inner: Mutex<LazyState>,
    /// Raw output of the most recent `route()` call, exposed via
    /// [`SkillRouter::last_raw_output`] so the engine can log what the
    /// model actually emitted (function name + args block + stop
    /// tokens). Useful when investigating whether the router is
    /// producing typed-args we can consume downstream.
    last_raw: Mutex<Option<String>>,
}

unsafe impl Send for FunctionGemmaRouter {}
unsafe impl Sync for FunctionGemmaRouter {}

impl FunctionGemmaRouter {
    pub fn new(model_path: &Path) -> Self {
        FunctionGemmaRouter {
            model_path: model_path.to_path_buf(),
            inner: Mutex::new(LazyState {
                loaded: None,
                last_used: None,
            }),
            last_raw: Mutex::new(None),
        }
    }

    pub fn unload(&self) {
        if let Ok(mut state) = self.inner.lock() {
            state.loaded = None;
            state.last_used = None;
        }
    }
}

impl SkillRouter for FunctionGemmaRouter {
    fn route(
        &self,
        input: &str,
        skills: &[(String, String, String)],
    ) -> RouteResult {
        let mut state = match self.inner.lock() {
            Ok(s) => s,
            Err(_) => return RouteResult::NoMatch,
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
                Err(_) => return RouteResult::NoMatch,
            }
        }

        let now = Instant::now();
        state.last_used = Some(now);

        let prompt = build_router_prompt(input, skills);
        let model = state.loaded.as_ref().unwrap();

        let output = match model.run_router_inference(&prompt) {
            Ok(text) => text,
            Err(_) => return RouteResult::NoMatch,
        };

        // Stash the raw output so the engine can log it after the call
        // returns. The whole point of the diagnostic is seeing what the
        // model actually emits — function name, args block, stop tokens,
        // anything weird — so we can decide whether typed args are
        // already viable or whether the prompt / training needs work.
        if let Ok(mut last) = self.last_raw.lock() {
            *last = Some(output.clone());
        }

        state.last_used = Some(Instant::now());

        // Schedule idle eviction (same pattern as LazyLlmFallback).
        let last_used_at = Instant::now();
        let idle_timeout = std::time::Duration::from_secs(IDLE_TIMEOUT_SECS);
        drop(state);

        let inner = &self.inner as *const Mutex<LazyState> as usize;
        std::thread::spawn(move || {
            std::thread::sleep(idle_timeout);
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

        parse_router_output(&output, skills)
    }

    fn last_raw_output(&self) -> Option<String> {
        self.last_raw.lock().ok().and_then(|g| g.clone())
    }
}

impl LoadedModel {
    fn run_router_inference(&self, prompt: &str) -> Result<String, LlmError> {
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(2048));

        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| LlmError::Context(e.to_string()))?;

        let tokens = self
            .model
            .str_to_token(prompt, llama_cpp_2::model::AddBos::Always)
            .map_err(|e| LlmError::Context(e.to_string()))?;

        let mut batch = LlamaBatch::new(tokens.len() + ROUTER_MAX_TOKENS, 1);
        for (i, &token) in tokens.iter().enumerate() {
            let is_last = i == tokens.len() - 1;
            batch
                .add(token, i as i32, &[0], is_last)
                .map_err(|e| LlmError::Context(format!("batch add: {e}")))?;
        }

        ctx.decode(&mut batch)
            .map_err(|e| LlmError::Context(format!("decode prompt: {e}")))?;

        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::temp(0.0),
            LlamaSampler::greedy(),
        ]);

        let mut output = String::new();
        let mut n_cur = tokens.len() as i32;

        for _ in 0..ROUTER_MAX_TOKENS {
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

            // Stop at end_of_turn or start_function_response (FunctionGemma stop tokens)
            if output.contains("<end_of_turn>") || output.contains("<start_function_response>") {
                break;
            }

            // Stop after the first complete function call
            if output.contains("<end_function_call>") {
                break;
            }

            batch.clear();
            batch
                .add(token, n_cur, &[0], true)
                .map_err(|e| LlmError::Context(format!("batch add gen: {e}")))?;
            n_cur += 1;

            ctx.decode(&mut batch)
                .map_err(|e| LlmError::Context(format!("decode gen: {e}")))?;
        }

        Ok(output)
    }
}

/// Build the FunctionGemma prompt with tool declarations.
///
/// Each tool declaration embeds the skill's actual parameter schema
/// (from [`Skill::parameters_schema`]) instead of the previous
/// `parameters:{type:OBJECT}` placeholder — without the real schema,
/// the model has no slot names to fill in and dutifully emits empty
/// args even for parameterised skills. With the schema present, it
/// can produce typed args matching what the training data taught it.
fn build_router_prompt(input: &str, skills: &[(String, String, String)]) -> String {
    let e = "<escape>";
    let mut declarations = String::new();
    for (id, description, schema_json) in skills {
        // The schema is JSON like `{"type":"object","properties":{...}}`.
        // FunctionGemma's tool format wants Python-dict-ish keys
        // wrapped in <escape>...<escape> rather than JSON quotes —
        // close enough that we can transform a clean schema by simple
        // substitution. The training pipeline does the equivalent via
        // HuggingFace's apply_chat_template, but we reach the same
        // place by shape here.
        let schema_inline = json_schema_to_funcgemma(schema_json, e);
        declarations.push_str(&format!(
            "<start_function_declaration>declaration:{id}{{description:{e}{description}{e},parameters:{schema_inline}}}<end_function_declaration>"
        ));
    }
    format!(
        "<start_of_turn>developer\n\
         You are a model that can do function calling with the following functions\
         {declarations}<end_of_turn>\n\
         <start_of_turn>user\n\
         {input}<end_of_turn>\n\
         <start_of_turn>model\n"
    )
}

/// Translate a JSON-format parameter schema into FunctionGemma's
/// declaration syntax. The format swaps JSON's `"key"` for bare
/// identifiers and wraps string values in `<escape>...<escape>`.
/// Falls back to `{type:<escape>OBJECT<escape>}` if the schema
/// can't be parsed — matching the previous placeholder, so a
/// malformed schema isn't worse than the old behaviour.
fn json_schema_to_funcgemma(schema_json: &str, e: &str) -> String {
    let value: serde_json::Value = match serde_json::from_str(schema_json) {
        Ok(v) => v,
        Err(_) => return format!("{{type:{e}OBJECT{e}}}"),
    };
    render_funcgemma_value(&value, e)
}

fn render_funcgemma_value(v: &serde_json::Value, e: &str) -> String {
    match v {
        serde_json::Value::Object(obj) => {
            let mut out = String::from("{");
            let mut first = true;
            for (k, val) in obj {
                if !first {
                    out.push(',');
                }
                first = false;
                // Bare identifier when the key is a typical schema key
                // (alphanumeric + underscore); fall back to escape-
                // wrapped if it has anything unusual.
                if k.chars().all(|c| c.is_alphanumeric() || c == '_') {
                    out.push_str(k);
                } else {
                    out.push_str(&format!("{e}{k}{e}"));
                }
                out.push(':');
                out.push_str(&render_funcgemma_value(val, e));
            }
            out.push('}');
            out
        }
        serde_json::Value::Array(arr) => {
            let mut out = String::from("[");
            let mut first = true;
            for item in arr {
                if !first {
                    out.push(',');
                }
                first = false;
                out.push_str(&render_funcgemma_value(item, e));
            }
            out.push(']');
            out
        }
        serde_json::Value::String(s) => {
            // Schema values like "object", "string", "number" are upper-
            // cased by FunctionGemma's convention — mirror what training
            // emitted via HF's tool template.
            let upper = s.to_uppercase();
            format!("{e}{upper}{e}")
        }
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Null => format!("{e}NULL{e}"),
    }
}

/// Parse FunctionGemma's output into a RouteResult.
///
/// The model emits
/// `<start_function_call>call:NAME{key:<escape>value<escape>,...}<end_function_call>`
/// where `NAME` is the skill id and the brace block holds args in
/// FunctionGemma's declaration syntax (bare keys, escape-wrapped
/// strings, uppercased type sentinels). This converts the brace block
/// into JSON via [`funcgemma_to_json`] so downstream layers can read
/// the args as standard JSON. Empty / parameterless calls fall back
/// to [`RouteResult::Skill`]; calls with non-empty args emit
/// [`RouteResult::SkillWithArgs`].
fn parse_router_output(output: &str, skills: &[(String, String, String)]) -> RouteResult {
    let skill_names: std::collections::HashSet<&str> =
        skills.iter().map(|(id, _, _)| id.as_str()).collect();

    let Some((name, args_block)) = extract_call_block(output) else {
        // No function call — model declined.
        return RouteResult::NoMatch;
    };

    if !skill_names.contains(name.as_str()) {
        // Function name not in our skill list — could be a mobile action
        // or a hallucination. For now, treat unknown names as NoMatch.
        // TODO: map known mobile action names to RouteResult::Action
        return RouteResult::NoMatch;
    }

    // No args block (or model emitted just `name{}`) — pre-typed-args
    // shape, equivalent to RouteResult::Skill.
    let args_json = funcgemma_to_json(args_block.as_str()).unwrap_or_default();
    if args_json.is_empty() || args_json == "{}" {
        return RouteResult::Skill(name);
    }

    RouteResult::SkillWithArgs { id: name, args_json }
}

/// Extract the function name and the inner brace contents from a
/// FunctionGemma call. Returns `None` if no call is present in the
/// output. The brace block is matched by counting braces / respecting
/// the `<escape>...<escape>` string delimiter, so nested objects work.
fn extract_call_block(output: &str) -> Option<(String, String)> {
    let head_re = regex::Regex::new(r"<start_function_call>call:(\w+)\{").unwrap();
    let caps = head_re.captures(output)?;
    let name = caps.get(1)?.as_str().to_string();
    let head_match = caps.get(0)?;
    // Inner block starts right after the opening `{`.
    let inner_start = head_match.end();
    let bytes = output.as_bytes();
    let mut depth: i32 = 1;
    let mut in_escape = false;
    let mut i = inner_start;
    while i < bytes.len() {
        // FunctionGemma's string delimiter is the literal token
        // "<escape>" — toggle in/out on each occurrence so braces
        // inside strings don't perturb the depth counter.
        if !in_escape && output[i..].starts_with("<escape>") {
            in_escape = true;
            i += "<escape>".len();
            continue;
        }
        if in_escape && output[i..].starts_with("<escape>") {
            in_escape = false;
            i += "<escape>".len();
            continue;
        }
        if !in_escape {
            match bytes[i] {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some((name, output[inner_start..i].to_string()));
                    }
                }
                _ => {}
            }
        }
        i += 1;
    }
    // Unbalanced braces — likely truncated output.
    None
}

/// Translate FunctionGemma's brace-block syntax into a JSON object
/// string. The input looks like
/// `key:<escape>value<escape>,key2:42,key3:{nested:<escape>x<escape>}`
/// with bare alphanumeric/underscore keys, `<escape>...<escape>`-wrapped
/// string values, bare numbers/booleans, and nested object/array
/// braces. The output is conventional JSON: `{"key":"value",...}`.
///
/// Returns `None` on malformed input (mismatched escape/braces/etc.).
fn funcgemma_to_json(input: &str) -> Option<String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Some(String::new());
    }
    let (rendered, consumed) = render_object_body(trimmed)?;
    // Allow trailing whitespace but no other tokens after the body.
    if trimmed[consumed..].chars().all(|c| c.is_whitespace()) {
        Some(rendered)
    } else {
        None
    }
}

/// Render the body of an object — `key:value, key:value, ...` — into a
/// JSON object string. Returns the JSON plus the number of bytes
/// consumed from `input`.
fn render_object_body(input: &str) -> Option<(String, usize)> {
    let mut out = String::from("{");
    let mut first = true;
    let mut cursor = 0;
    loop {
        cursor += skip_whitespace(&input[cursor..]);
        if cursor >= input.len() {
            break;
        }
        // End of containing object — caller handles the closing brace.
        if input.as_bytes()[cursor] == b'}' {
            break;
        }
        if !first {
            // Expect a separator before the next pair.
            if input.as_bytes()[cursor] != b',' {
                return None;
            }
            cursor += 1;
            cursor += skip_whitespace(&input[cursor..]);
        }
        // Key — bare identifier.
        let key_end = input[cursor..]
            .find(|c: char| !c.is_alphanumeric() && c != '_')
            .map(|p| cursor + p)
            .unwrap_or(input.len());
        if key_end == cursor {
            return None;
        }
        let key = &input[cursor..key_end];
        cursor = key_end;
        cursor += skip_whitespace(&input[cursor..]);
        if cursor >= input.len() || input.as_bytes()[cursor] != b':' {
            return None;
        }
        cursor += 1;
        cursor += skip_whitespace(&input[cursor..]);
        let (value_json, value_consumed) = render_value(&input[cursor..])?;
        cursor += value_consumed;
        if !first {
            out.push(',');
        }
        first = false;
        out.push('"');
        out.push_str(&escape_json_string(key));
        out.push_str("\":");
        out.push_str(&value_json);
    }
    out.push('}');
    Some((out, cursor))
}

/// Render a single value — string, number, bool, null, object, or
/// array — into JSON. Returns the JSON plus bytes consumed.
fn render_value(input: &str) -> Option<(String, usize)> {
    let bytes = input.as_bytes();
    if input.is_empty() {
        return None;
    }
    // String: <escape>...<escape>
    if input.starts_with("<escape>") {
        let after = &input["<escape>".len()..];
        let close = after.find("<escape>")?;
        let raw = &after[..close];
        let consumed = "<escape>".len() + close + "<escape>".len();
        // Type sentinels (OBJECT/STRING/NUMBER/etc.) are always
        // uppercased values; preserve case verbatim. Real string
        // values are passed through with JSON-string escaping.
        let json = format!("\"{}\"", escape_json_string(raw));
        return Some((json, consumed));
    }
    // Object: { key:value, ... }
    if bytes[0] == b'{' {
        let (body, body_consumed) = render_object_body(&input[1..])?;
        let after = 1 + body_consumed;
        if after >= input.len() || input.as_bytes()[after] != b'}' {
            return None;
        }
        return Some((body, after + 1));
    }
    // Array: [ value, value, ... ]
    if bytes[0] == b'[' {
        let mut out = String::from("[");
        let mut first = true;
        let mut cursor = 1;
        loop {
            cursor += skip_whitespace(&input[cursor..]);
            if cursor >= input.len() {
                return None;
            }
            if input.as_bytes()[cursor] == b']' {
                cursor += 1;
                break;
            }
            if !first {
                if input.as_bytes()[cursor] != b',' {
                    return None;
                }
                cursor += 1;
                cursor += skip_whitespace(&input[cursor..]);
            }
            let (item_json, item_consumed) = render_value(&input[cursor..])?;
            cursor += item_consumed;
            if !first {
                out.push(',');
            }
            first = false;
            out.push_str(&item_json);
        }
        out.push(']');
        return Some((out, cursor));
    }
    // Bare token: number, boolean, null, or unquoted identifier
    // (e.g. an enum-ish value). Stop at the first separator.
    let token_end = input
        .find(|c: char| c == ',' || c == '}' || c == ']' || c.is_whitespace())
        .unwrap_or(input.len());
    let token = &input[..token_end];
    if token.is_empty() {
        return None;
    }
    let json = if token == "true" || token == "false" || token == "null" {
        token.to_string()
    } else if token.parse::<f64>().is_ok() {
        token.to_string()
    } else {
        // Bare unquoted identifier — fall back to string. Safer than
        // emitting an invalid JSON token.
        format!("\"{}\"", escape_json_string(token))
    };
    Some((json, token_end))
}

fn skip_whitespace(s: &str) -> usize {
    s.bytes().take_while(|b| b.is_ascii_whitespace()).count()
}

fn escape_json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
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
    fn system_prompt_is_concise() {
        let prompt = build_system_prompt(&test_skills());
        assert!(prompt.contains("Ari"));
        assert!(prompt.contains("one short sentence"));
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

    fn router_skill_catalog() -> Vec<(String, String, String)> {
        vec![
            (
                "current_time".to_string(),
                "Tells the current time.".to_string(),
                r#"{"type":"object","properties":{}}"#.to_string(),
            ),
            (
                "open".to_string(),
                "Opens apps by name.".to_string(),
                r#"{"type":"object","properties":{"app_name":{"type":"string"}}}"#.to_string(),
            ),
        ]
    }

    #[test]
    fn parse_router_output_parameterless_returns_skill() {
        // Empty brace block — parameterless skill, behave as before.
        let out = "<start_function_call>call:current_time{}<end_function_call>";
        match parse_router_output(out, &router_skill_catalog()) {
            RouteResult::Skill(id) => assert_eq!(id, "current_time"),
            other => panic!("expected Skill, got {:?}", std::mem::discriminant(&other)),
        }
    }

    #[test]
    fn parse_router_output_typed_args_returns_skill_with_args() {
        // Real shape we observed from Gemma 4 E2B for "fire up spotify".
        let out = "<start_function_call>call:open{app_name:<escape>Spotify<escape>}<end_function_call>";
        match parse_router_output(out, &router_skill_catalog()) {
            RouteResult::SkillWithArgs { id, args_json } => {
                assert_eq!(id, "open");
                assert_eq!(args_json, r#"{"app_name":"Spotify"}"#);
            }
            other => panic!("expected SkillWithArgs, got {:?}", std::mem::discriminant(&other)),
        }
    }

    #[test]
    fn parse_router_output_unknown_skill_returns_nomatch() {
        let out = "<start_function_call>call:not_a_skill{}<end_function_call>";
        assert!(matches!(
            parse_router_output(out, &router_skill_catalog()),
            RouteResult::NoMatch
        ));
    }

    #[test]
    fn parse_router_output_no_call_returns_nomatch() {
        // Model declined; emitted no function call at all.
        assert!(matches!(
            parse_router_output("Some prose with no call.", &router_skill_catalog()),
            RouteResult::NoMatch
        ));
    }

    #[test]
    fn funcgemma_to_json_handles_strings_numbers_bools_nested() {
        // String + number + bool + nested object + escaped quote in value.
        let input = r#"name:<escape>foo "bar"<escape>,count:42,enabled:true,meta:{kind:<escape>tool<escape>}"#;
        let json = funcgemma_to_json(input).expect("parse should succeed");
        // serde_json roundtrip ensures the output is valid JSON.
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("emit valid JSON");
        assert_eq!(parsed["name"], "foo \"bar\"");
        assert_eq!(parsed["count"], 42);
        assert_eq!(parsed["enabled"], true);
        assert_eq!(parsed["meta"]["kind"], "tool");
    }

    #[test]
    fn funcgemma_to_json_handles_braces_inside_escape_strings() {
        // Brace counter must respect <escape> delimiters — a `}` inside
        // a string mustn't close the surrounding object.
        let input = r#"title:<escape>fix {bug}<escape>,count:1"#;
        let json = funcgemma_to_json(input).expect("parse should succeed");
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["title"], "fix {bug}");
        assert_eq!(parsed["count"], 1);
    }

    #[test]
    fn funcgemma_to_json_empty_input_yields_empty_string() {
        // Empty body → empty string sentinel so the caller can short-
        // circuit to RouteResult::Skill instead of SkillWithArgs.
        assert_eq!(funcgemma_to_json("").as_deref(), Some(""));
        assert_eq!(funcgemma_to_json("   ").as_deref(), Some(""));
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
        let system = build_system_prompt(&skills);
        let user = build_user_prompt("what is the capital of australia");

        eprintln!("--- System prompt ---");
        eprintln!("{system}");
        eprintln!("--- User prompt ---");
        eprintln!("{user}");

        let prompt = match fallback.build_chat_prompt(&system, &user) {
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
