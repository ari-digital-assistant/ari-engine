use ari_core::{ExampleUtterance, Response, Skill, SkillContext, Specificity};

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

    fn parameters_schema(&self) -> &'static str {
        r#"{"type": "object", "properties": {"app_name": {"type": "string", "description": "Name of the app to open."}}, "required": ["app_name"]}"#
    }

    fn example_utterances(&self) -> &[ExampleUtterance] {
        &[
            ExampleUtterance { text: "open spotify", args: r#"{"app_name": "Spotify"}"# },
            ExampleUtterance { text: "launch the camera", args: r#"{"app_name": "Camera"}"# },
            ExampleUtterance { text: "start the browser", args: r#"{"app_name": "Browser"}"# },
            ExampleUtterance { text: "open youtube", args: r#"{"app_name": "YouTube"}"# },
            ExampleUtterance { text: "can you open settings", args: r#"{"app_name": "Settings"}"# },
            ExampleUtterance { text: "launch maps", args: r#"{"app_name": "Maps"}"# },
            ExampleUtterance { text: "fire up the music player", args: r#"{"app_name": "Music Player"}"# },
            ExampleUtterance { text: "run chrome", args: r#"{"app_name": "Chrome"}"# },
            ExampleUtterance { text: "open my email", args: r#"{"app_name": "Email"}"# },
            ExampleUtterance { text: "start whatsapp", args: r#"{"app_name": "WhatsApp"}"# },
            ExampleUtterance { text: "open the calculator app", args: r#"{"app_name": "Calculator"}"# },
            ExampleUtterance { text: "launch instagram", args: r#"{"app_name": "Instagram"}"# },
            ExampleUtterance { text: "can you start the camera app", args: r#"{"app_name": "Camera"}"# },
            ExampleUtterance { text: "open netflix", args: r#"{"app_name": "Netflix"}"# },
            ExampleUtterance { text: "fire up spotify", args: r#"{"app_name": "Spotify"}"# },
            ExampleUtterance { text: "launch my music player", args: r#"{"app_name": "Music Player"}"# },
            ExampleUtterance { text: "run the gallery", args: r#"{"app_name": "Gallery"}"# },
            ExampleUtterance { text: "open telegram", args: r#"{"app_name": "Telegram"}"# },
            ExampleUtterance { text: "start firefox", args: r#"{"app_name": "Firefox"}"# },
            ExampleUtterance { text: "open the clock app", args: r#"{"app_name": "Clock"}"# },
            ExampleUtterance { text: "launch the phone app", args: r#"{"app_name": "Phone"}"# },
            ExampleUtterance { text: "open messages", args: r#"{"app_name": "Messages"}"# },
            ExampleUtterance { text: "can you open twitter", args: r#"{"app_name": "Twitter"}"# },
            ExampleUtterance { text: "start the notes app", args: r#"{"app_name": "Notes"}"# },
            ExampleUtterance { text: "open slack", args: r#"{"app_name": "Slack"}"# },
            ExampleUtterance { text: "launch the calendar", args: r#"{"app_name": "Calendar"}"# },
            ExampleUtterance { text: "fire up the weather app", args: r#"{"app_name": "Weather"}"# },
            ExampleUtterance { text: "open reddit", args: r#"{"app_name": "Reddit"}"# },
            ExampleUtterance { text: "run discord", args: r#"{"app_name": "Discord"}"# },
            ExampleUtterance { text: "open the files app", args: r#"{"app_name": "Files"}"# },
        ]
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
            // `speak` is omitted deliberately — the frontend owns the
            // platform-appropriate phrasing ("Opening Spotify" on Android,
            // possibly a different verb on Linux) and can override with a
            // failure message if the launch doesn't work.
            Some(target) => Response::Action(serde_json::json!({
                "v": 1,
                "launch_app": target,
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
                assert_eq!(v["v"], 1);
                assert_eq!(v["launch_app"], "spotify");
                // speak is intentionally absent — frontend produces the text.
                assert!(v.get("speak").is_none());
            }
            other => panic!("expected Action, got {other:?}"),
        }
    }

    #[test]
    fn execute_multi_word_target() {
        let skill = OpenSkill::new();
        match skill.execute("open file manager", &ctx()) {
            Response::Action(v) => {
                assert_eq!(v["v"], 1);
                assert_eq!(v["launch_app"], "file manager");
            }
            other => panic!("expected Action, got {other:?}"),
        }
    }

    #[test]
    fn execute_takes_everything_after_trigger() {
        let skill = OpenSkill::new();
        match skill.execute("launch the camera app", &ctx()) {
            Response::Action(v) => assert_eq!(v["launch_app"], "the camera app"),
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
