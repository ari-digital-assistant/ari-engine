use ari_core::{Response, Skill, SkillContext, Specificity};

const TRIGGER_WORDS: &[&str] = &["open", "launch", "start", "run"];

pub struct OpenSkill;

impl OpenSkill {
    pub fn new() -> Self {
        Self
    }
}

impl Default for OpenSkill {
    fn default() -> Self {
        Self::new()
    }
}

fn extract_target(input: &str) -> Option<String> {
    let words: Vec<&str> = input.split_whitespace().collect();

    for (i, word) in words.iter().enumerate() {
        if TRIGGER_WORDS.contains(word) {
            let target: Vec<&str> = words[i + 1..].to_vec();
            if !target.is_empty() {
                return Some(target.join(" "));
            }
        }
    }

    None
}

impl Skill for OpenSkill {
    fn id(&self) -> &str {
        "open"
    }

    fn description(&self) -> &str {
        "Opens or launches apps by name. Use when the user asks to open, launch, start, run, or fire up an application or app."
    }

    fn specificity(&self) -> Specificity {
        Specificity::Medium
    }

    fn score(&self, input: &str, _ctx: &SkillContext) -> f32 {
        let words: Vec<&str> = input.split_whitespace().collect();

        let has_trigger = words.iter().any(|w| TRIGGER_WORDS.contains(w));
        if !has_trigger {
            return 0.0;
        }

        if extract_target(input).is_some() {
            0.9
        } else {
            0.3
        }
    }

    fn execute(&self, input: &str, _ctx: &SkillContext) -> Response {
        match extract_target(input) {
            Some(target) => Response::Action(serde_json::json!({
                "action": "open",
                "target": target,
            })),
            None => Response::Text("What would you like me to open?".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> SkillContext {
        SkillContext::default()
    }

    // Scoring: trigger + target = 0.9, trigger alone = 0.3, no trigger = 0.0

    #[test]
    fn score_with_target() {
        let skill = OpenSkill::new();
        assert_eq!(skill.score("open spotify", &ctx()), 0.9);
        assert_eq!(skill.score("launch the camera", &ctx()), 0.9);
        assert_eq!(skill.score("start firefox", &ctx()), 0.9);
        assert_eq!(skill.score("run my app", &ctx()), 0.9);
    }

    #[test]
    fn score_trigger_without_target() {
        let skill = OpenSkill::new();
        assert_eq!(skill.score("open", &ctx()), 0.3);
        assert_eq!(skill.score("launch", &ctx()), 0.3);
    }

    #[test]
    fn score_zero_no_trigger() {
        let skill = OpenSkill::new();
        assert_eq!(skill.score("what time is it", &ctx()), 0.0);
        assert_eq!(skill.score("hello", &ctx()), 0.0);
        assert_eq!(skill.score("spotify", &ctx()), 0.0);
    }

    #[test]
    fn execute_single_word_target() {
        let skill = OpenSkill::new();
        match skill.execute("open spotify", &ctx()) {
            Response::Action(v) => {
                assert_eq!(v["action"], "open");
                assert_eq!(v["target"], "spotify");
            }
            other => panic!("expected Action, got {other:?}"),
        }
    }

    #[test]
    fn execute_multi_word_target() {
        let skill = OpenSkill::new();
        match skill.execute("open file manager", &ctx()) {
            Response::Action(v) => {
                assert_eq!(v["action"], "open");
                assert_eq!(v["target"], "file manager");
            }
            other => panic!("expected Action, got {other:?}"),
        }
    }

    #[test]
    fn execute_takes_everything_after_trigger() {
        let skill = OpenSkill::new();
        match skill.execute("launch the camera app", &ctx()) {
            Response::Action(v) => assert_eq!(v["target"], "the camera app"),
            other => panic!("expected Action, got {other:?}"),
        }
    }

    #[test]
    fn execute_no_target_asks_for_clarification() {
        let skill = OpenSkill::new();
        match skill.execute("open", &ctx()) {
            Response::Text(s) => assert_eq!(s, "What would you like me to open?"),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn extract_target_picks_first_trigger() {
        // "please open the app" — "open" is at index 1, target = "the app"
        assert_eq!(extract_target("please open the app"), Some("the app".to_string()));
    }

    #[test]
    fn extract_target_returns_none_for_no_trigger() {
        assert_eq!(extract_target("spotify please"), None);
    }

    #[test]
    fn specificity_is_medium() {
        assert_eq!(OpenSkill::new().specificity(), Specificity::Medium);
    }
}
