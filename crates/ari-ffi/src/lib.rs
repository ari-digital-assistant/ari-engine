#![allow(clippy::new_without_default)]

use ari_engine::Engine;
use ari_skills::{
    CalculatorSkill, CurrentTimeSkill, DateSkill, GreetingSkill, OpenSkill, SearchSkill,
};

mod skill_registry;

pub use skill_registry::{FfiInstalledSkill, FfiRegistryError, FfiSkillUpdate, SkillRegistry};

uniffi::setup_scaffolding!();

#[derive(uniffi::Enum)]
pub enum FfiResponse {
    Text { body: String },
    Action { json: String },
    Binary { mime: String, data: Vec<u8> },
}

#[derive(uniffi::Object)]
pub struct AriEngine {
    inner: Engine,
}

#[uniffi::export]
impl AriEngine {
    #[uniffi::constructor]
    pub fn new() -> Self {
        let mut engine = Engine::new();
        engine.register_skill(Box::new(CurrentTimeSkill::new()));
        engine.register_skill(Box::new(DateSkill::new()));
        engine.register_skill(Box::new(CalculatorSkill::new()));
        engine.register_skill(Box::new(GreetingSkill::new()));
        engine.register_skill(Box::new(OpenSkill::new()));
        engine.register_skill(Box::new(SearchSkill::new()));
        Self { inner: engine }
    }

    pub fn process_input(&self, input: String) -> FfiResponse {
        match self.inner.process_input(&input) {
            ari_core::Response::Text(s) => FfiResponse::Text { body: s },
            ari_core::Response::Action(v) => FfiResponse::Action {
                json: serde_json::to_string(&v).unwrap_or_default(),
            },
            ari_core::Response::Binary { mime, data } => FfiResponse::Binary { mime, data },
        }
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
    fn engine_returns_fallback_for_gibberish() {
        let engine = AriEngine::new();
        let resp = engine.process_input("asdfghjkl".to_string());
        match resp {
            FfiResponse::Text { body } => {
                assert_eq!(body, "Sorry, I didn't understand that.");
            }
            _ => panic!("expected Text fallback"),
        }
    }
}
