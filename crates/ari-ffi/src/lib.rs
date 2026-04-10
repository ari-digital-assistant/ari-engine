#![allow(clippy::new_without_default)]

use ari_engine::{Engine, FALLBACK_RESPONSE};
use ari_skill_loader::{
    load_skill_directory_with, Capability, HostCapabilities, HttpConfig, LoadOptions,
    NullLogSink, StorageConfig,
};
use ari_skills::{
    CalculatorSkill, CurrentTimeSkill, DateSkill, GreetingSkill, OpenSkill, SearchSkill,
};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

mod assistant_registry;
mod skill_registry;

pub use assistant_registry::{
    AssistantRegistry, FfiAssistantEntry, FfiConfigField, FfiSelectOption,
};
pub use skill_registry::{
    FfiBrowseEntry, FfiInstalledSkill, FfiRegistryError, FfiSkillUpdate, SkillRegistry,
};

/// Build the [`LoadOptions`] the Android host uses for every install and
/// every reload. Grants `pure_frontend` caps (frontend-mediated actions),
/// `http` (backed by reqwest with bundled webpki-roots — see `tls.rs`),
/// and `storage_kv` (backed by per-skill JSON files under `storage_dir`).
///
/// Keep this in one place so every loader entry point in the FFI crate
/// sees the same grants. A mismatch — e.g. install granting `http` but
/// reload not — would let a skill install cleanly and then silently drop
/// off the conversation engine on the next app start.
pub(crate) fn android_load_options(storage_dir: &str) -> LoadOptions {
    let host_caps = HostCapabilities::pure_frontend()
        .with(Capability::Http)
        .with(Capability::StorageKv);
    LoadOptions {
        log_sink: Arc::new(NullLogSink),
        host_capabilities: host_caps,
        http_config: HttpConfig::strict(),
        storage_config: StorageConfig::new(PathBuf::from(storage_dir)),
    }
}

uniffi::setup_scaffolding!();

#[derive(uniffi::Enum)]
pub enum FfiResponse {
    Text { body: String },
    Action { json: String },
    Binary { mime: String, data: Vec<u8> },
    /// The engine couldn't match any skill to the input. The host can use
    /// this signal to retry the upstream STT (e.g. with a fresh sherpa
    /// stream on the buffered audio) before falling back to the apology.
    /// `body` carries the apology text the host should say if the retry
    /// also fails — kept here so the host doesn't have to hardcode it.
    NotUnderstood { body: String },
}

#[derive(uniffi::Object)]
pub struct AriEngine {
    // Wrapped in Mutex because `reload_community_skills` mutates the
    // skill set after construction. `process_input` only needs a shared
    // lock in practice but the Engine trait takes `&self` anyway.
    pub(crate) inner: Mutex<Engine>,
}

fn build_engine_with_builtins() -> Engine {
    let mut engine = Engine::new();
    engine.register_skill(Box::new(CurrentTimeSkill::new()));
    engine.register_skill(Box::new(DateSkill::new()));
    engine.register_skill(Box::new(CalculatorSkill::new()));
    engine.register_skill(Box::new(GreetingSkill::new()));
    engine.register_skill(Box::new(OpenSkill::new()));
    engine.register_skill(Box::new(SearchSkill::new()));
    engine
}

#[uniffi::export]
impl AriEngine {
    #[uniffi::constructor]
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(build_engine_with_builtins()),
        }
    }

    pub fn process_input(&self, input: String) -> FfiResponse {
        let engine = self.inner.lock().expect("engine mutex poisoned");
        match engine.process_input(&input) {
            ari_core::Response::Text(s) => {
                if s == FALLBACK_RESPONSE {
                    FfiResponse::NotUnderstood { body: s }
                } else {
                    FfiResponse::Text { body: s }
                }
            }
            ari_core::Response::Action(v) => FfiResponse::Action {
                json: serde_json::to_string(&v).unwrap_or_default(),
            },
            ari_core::Response::Binary { mime, data } => FfiResponse::Binary { mime, data },
        }
    }

    /// Set the GGUF model path for the LLM fallback. The model is NOT
    /// loaded immediately — it loads on demand when the first unmatched
    /// query arrives, and unloads after 60 seconds of idle to free RAM.
    ///
    /// Returns `true` if the path exists, `false` otherwise.
    /// Call at app startup if a model file is available on disk.
    #[cfg(feature = "llm")]
    pub fn load_llm_model(&self, model_path: String) -> bool {
        let path = std::path::Path::new(&model_path);
        if !path.is_file() {
            return false;
        }
        let lazy = ari_llm::LazyLlmFallback::new(path);
        let mut engine = self.inner.lock().expect("engine mutex poisoned");
        engine.set_llm(Box::new(lazy));
        true
    }

    /// Remove the LLM fallback. If a model is currently loaded in RAM,
    /// it is dropped and the memory is freed.
    #[cfg(feature = "llm")]
    pub fn unload_llm_model(&self) {
        let mut engine = self.inner.lock().expect("engine mutex poisoned");
        engine.set_llm_none();
    }

    /// Rebuild the engine's skill set from scratch: the 6 built-in Rust
    /// skills plus every community skill on disk under `skill_store_dir`.
    ///
    /// `storage_dir` is where per-skill `storage_kv` JSON files live —
    /// must match what `SkillRegistry` was constructed with, otherwise a
    /// skill's installed state (on-disk JSON) will be invisible at
    /// conversation time. Both dirs should sit under the app's private
    /// files directory on Android (`context.filesDir`).
    ///
    /// Call once at app startup (after constructing `SkillRegistry` so the
    /// store dir exists) and again after every successful install / update
    /// / uninstall so the next `process_input` can see the new state.
    ///
    /// Silently ignores skills that fail to load — individual failures are
    /// recorded in the loader's `LoadReport.failures`, which we currently
    /// discard at this boundary. A broken skill should not take the
    /// conversation engine down with it. Returns the number of community
    /// skills successfully registered so the caller can log / surface it.
    pub fn reload_community_skills(
        &self,
        skill_store_dir: String,
        storage_dir: String,
    ) -> u32 {
        let mut fresh = build_engine_with_builtins();
        let options = android_load_options(&storage_dir);
        let loaded: u32 =
            match load_skill_directory_with(&PathBuf::from(&skill_store_dir), &options) {
                Ok(report) => {
                    let n = report.skills.len() as u32;
                    for skill in report.skills {
                        fresh.register_skill(skill);
                    }
                    n
                }
                Err(_) => 0,
            };
        *self.inner.lock().expect("engine mutex poisoned") = fresh;
        loaded
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_creates_and_responds_to_greeting() {
        let engine = AriEngine::new();
        let resp = engine.process_input("hello".to_string());
        match resp {
            FfiResponse::Text { body } => {
                assert!(!body.is_empty());
                assert_ne!(body, "Sorry, I didn't understand that.");
            }
            _ => panic!("expected Text response for greeting"),
        }
    }

    #[test]
    fn engine_returns_time() {
        let engine = AriEngine::new();
        let resp = engine.process_input("what time is it".to_string());
        match resp {
            FfiResponse::Text { body } => {
                assert!(body.starts_with("It's "), "response was: {body}");
            }
            _ => panic!("expected Text response for time"),
        }
    }

    #[test]
    fn engine_returns_calculation() {
        let engine = AriEngine::new();
        let resp = engine.process_input("calculate 5 + 3".to_string());
        match resp {
            FfiResponse::Text { body } => assert_eq!(body, "8"),
            _ => panic!("expected Text response for calculation"),
        }
    }

    #[test]
    fn engine_returns_action_for_open() {
        let engine = AriEngine::new();
        let resp = engine.process_input("open spotify".to_string());
        match resp {
            FfiResponse::Action { json } => {
                let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
                assert_eq!(parsed["action"], "open");
                assert_eq!(parsed["target"], "spotify");
            }
            _ => panic!("expected Action response for open"),
        }
    }

    #[test]
    fn engine_returns_not_understood_for_gibberish() {
        let engine = AriEngine::new();
        let resp = engine.process_input("asdfghjkl".to_string());
        match resp {
            FfiResponse::NotUnderstood { body } => {
                assert_eq!(body, "Sorry, I didn't understand that.");
            }
            _ => panic!("expected NotUnderstood fallback"),
        }
    }
}
