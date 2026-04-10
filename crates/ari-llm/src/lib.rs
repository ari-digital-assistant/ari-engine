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
const MAX_GENERATION_TOKENS: usize = 512;

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
        let messages = vec![
            LlamaChatMessage::new("system".to_string(), system.to_string()).ok()?,
            LlamaChatMessage::new("user".to_string(), user.to_string()).ok()?,
        ];
        self.model.apply_chat_template(&tmpl, &messages, true).ok()
    }

    fn run_inference(&self, prompt: &str) -> Result<String, LlmError> {
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

            let cleaned = strip_thinking(&output);
            if !cleaned.is_empty() && cleaned.contains('\n') {
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

        Ok(strip_thinking(&output))
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

        let output = match model.run_inference(&prompt) {
            Ok(text) => text,
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

    fn run_inference(&self, prompt: &str) -> Result<String, LlmError> {
        self.loaded.run_inference(prompt)
    }
}

impl Fallback for LlmFallback {
    fn try_answer(&self, input: &str, skills: &[SkillInfo]) -> Option<FallbackResult> {
        let _guard = self.inference_lock.lock().ok()?;

        let system_prompt = build_system_prompt(skills);
        let user_prompt = build_user_prompt(input);

        let prompt = match self.build_chat_prompt(&system_prompt, &user_prompt) {
            Some(p) => p,
            None => format!("{system_prompt}\n\nUser: {user_prompt}\n\nResponse: "),
        };

        let output = match self.run_inference(&prompt) {
            Ok(text) => text,
            Err(_) => return None,
        };

        parse_output(&output, skills)
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

fn strip_thinking(raw: &str) -> String {
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
        let output = fallback.run_inference(&prompt).expect("inference failed");
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
