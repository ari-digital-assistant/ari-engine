use ari_core::{ExampleUtterance, Response, Skill, SkillContext, Specificity};

// English + Italian trigger verbs. Same union-dictionary pattern as
// the other built-ins — words don't collide across these languages so
// a single contains-check disambiguates without a locale parameter.
const TRIGGER_WORDS: &[&str] = &[
    // English
    "open", "launch", "start", "run",
    // Italian: apri (open), avvia (start/launch), lancia (launch),
    // esegui (run)
    "apri", "avvia", "lancia", "esegui",
    // Italian imperative + clitic: "aprimi spotify" (open-me spotify),
    // "aprila"/"aprilo" (open-it). Natural spoken Italian that the bare
    // `apri` trigger misses because the clitic fuses onto the verb.
    "aprimi", "aprila", "aprilo", "avviami", "avviala", "avvialo",
];

// Router training examples. Natural raw text as a user would actually
// say it, paired with the canonical app name the router should emit.
//
// The canonical value stays English in every locale: `execute_with_args`
// hands `app_name` straight to the frontend's `launch_app` slot, which
// resolves it against installed apps' labels and package names. It feeds
// resolution, not display — so it isn't translated.
const OPEN_EXAMPLES_EN: &[ExampleUtterance] = &[
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
    // Paraphrases without explicit open/launch/start/run/fire-up triggers —
    // teach the router that any "I want to use / get to / show me X" maps
    // to opening that app.
    ExampleUtterance { text: "I want to use spotify", args: r#"{"app_name": "Spotify"}"# },
    ExampleUtterance { text: "show me the camera", args: r#"{"app_name": "Camera"}"# },
    ExampleUtterance { text: "bring up the calculator", args: r#"{"app_name": "Calculator"}"# },
    ExampleUtterance { text: "I need maps", args: r#"{"app_name": "Maps"}"# },
    ExampleUtterance { text: "jump into telegram", args: r#"{"app_name": "Telegram"}"# },
    ExampleUtterance { text: "get me to the weather app", args: r#"{"app_name": "Weather"}"# },
    ExampleUtterance { text: "switch to chrome", args: r#"{"app_name": "Chrome"}"# },
    ExampleUtterance { text: "I want to check whatsapp", args: r#"{"app_name": "WhatsApp"}"# },
    ExampleUtterance { text: "boot up the music player", args: r#"{"app_name": "Music Player"}"# },
    ExampleUtterance { text: "take me to settings", args: r#"{"app_name": "Settings"}"# },
];

// The same 40 intents in natural Italian — same app spread and the same
// phrasing variety (apri / avvia / lancia / fai partire / esegui, article
// and "l'app X" variants, plus the no-launch-verb paraphrases), rather
// than a line-by-line translation of the English.
const OPEN_EXAMPLES_IT: &[ExampleUtterance] = &[
    ExampleUtterance { text: "apri spotify", args: r#"{"app_name": "Spotify"}"# },
    ExampleUtterance { text: "apri la fotocamera", args: r#"{"app_name": "Camera"}"# },
    ExampleUtterance { text: "avvia il browser", args: r#"{"app_name": "Browser"}"# },
    ExampleUtterance { text: "apri youtube", args: r#"{"app_name": "YouTube"}"# },
    ExampleUtterance { text: "puoi aprire le impostazioni", args: r#"{"app_name": "Settings"}"# },
    ExampleUtterance { text: "avvia mappe", args: r#"{"app_name": "Maps"}"# },
    ExampleUtterance { text: "fai partire il lettore musicale", args: r#"{"app_name": "Music Player"}"# },
    ExampleUtterance { text: "esegui chrome", args: r#"{"app_name": "Chrome"}"# },
    ExampleUtterance { text: "apri la posta", args: r#"{"app_name": "Email"}"# },
    ExampleUtterance { text: "avvia whatsapp", args: r#"{"app_name": "WhatsApp"}"# },
    ExampleUtterance { text: "apri l'app calcolatrice", args: r#"{"app_name": "Calculator"}"# },
    ExampleUtterance { text: "lancia instagram", args: r#"{"app_name": "Instagram"}"# },
    ExampleUtterance { text: "puoi avviare l'app fotocamera", args: r#"{"app_name": "Camera"}"# },
    ExampleUtterance { text: "apri netflix", args: r#"{"app_name": "Netflix"}"# },
    ExampleUtterance { text: "fai partire spotify", args: r#"{"app_name": "Spotify"}"# },
    ExampleUtterance { text: "lancia il mio lettore musicale", args: r#"{"app_name": "Music Player"}"# },
    ExampleUtterance { text: "apri la galleria", args: r#"{"app_name": "Gallery"}"# },
    ExampleUtterance { text: "apri telegram", args: r#"{"app_name": "Telegram"}"# },
    ExampleUtterance { text: "avvia firefox", args: r#"{"app_name": "Firefox"}"# },
    ExampleUtterance { text: "apri l'app orologio", args: r#"{"app_name": "Clock"}"# },
    ExampleUtterance { text: "avvia l'app telefono", args: r#"{"app_name": "Phone"}"# },
    ExampleUtterance { text: "apri messaggi", args: r#"{"app_name": "Messages"}"# },
    ExampleUtterance { text: "puoi aprire twitter", args: r#"{"app_name": "Twitter"}"# },
    ExampleUtterance { text: "avvia l'app note", args: r#"{"app_name": "Notes"}"# },
    ExampleUtterance { text: "apri slack", args: r#"{"app_name": "Slack"}"# },
    ExampleUtterance { text: "avvia il calendario", args: r#"{"app_name": "Calendar"}"# },
    ExampleUtterance { text: "fai partire l'app meteo", args: r#"{"app_name": "Weather"}"# },
    ExampleUtterance { text: "apri reddit", args: r#"{"app_name": "Reddit"}"# },
    ExampleUtterance { text: "esegui discord", args: r#"{"app_name": "Discord"}"# },
    ExampleUtterance { text: "apri il gestore dei file", args: r#"{"app_name": "Files"}"# },
    // Paraphrases without an explicit launch verb — teach the router that
    // "voglio usare / portami a / mostrami X" ("I want to use / take me
    // to / show me X") means opening X.
    ExampleUtterance { text: "voglio usare spotify", args: r#"{"app_name": "Spotify"}"# },
    ExampleUtterance { text: "mostrami la fotocamera", args: r#"{"app_name": "Camera"}"# },
    ExampleUtterance { text: "fammi vedere la calcolatrice", args: r#"{"app_name": "Calculator"}"# },
    ExampleUtterance { text: "mi servono le mappe", args: r#"{"app_name": "Maps"}"# },
    ExampleUtterance { text: "vai su telegram", args: r#"{"app_name": "Telegram"}"# },
    ExampleUtterance { text: "portami all'app meteo", args: r#"{"app_name": "Weather"}"# },
    ExampleUtterance { text: "passa a chrome", args: r#"{"app_name": "Chrome"}"# },
    ExampleUtterance { text: "voglio controllare whatsapp", args: r#"{"app_name": "WhatsApp"}"# },
    ExampleUtterance { text: "apri il lettore musicale", args: r#"{"app_name": "Music Player"}"# },
    ExampleUtterance { text: "portami alle impostazioni", args: r#"{"app_name": "Settings"}"# },
];

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
        "Opens or launches apps by name. Use when the user asks to open, launch, start, run, fire up, or boot up an application — but ALSO when they want to use, get to, bring up, or jump into an app without naming a launch verb. Phrases like 'I want to use spotify', 'show me the camera', 'bring up the calculator', 'I need maps', 'jump into telegram', 'get me to the weather app', 'switch to chrome' all belong here. The app_name parameter is the app the user wants to interact with."
    }

    fn specificity(&self) -> Specificity {
        Specificity::Medium
    }

    fn parameters_schema(&self) -> &str {
        r#"{"type": "object", "properties": {"app_name": {"type": "string", "description": "Name of the app to open."}}, "required": ["app_name"]}"#
    }

    fn example_utterances(&self) -> &[ExampleUtterance] {
        OPEN_EXAMPLES_EN
    }

    fn example_utterances_for(&self, locale: &str) -> &[ExampleUtterance] {
        match locale {
            "it" => OPEN_EXAMPLES_IT,
            _ => OPEN_EXAMPLES_EN,
        }
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

    fn execute(&self, input: &str, ctx: &SkillContext) -> Response {
        match extract_target(input) {
            // `speak` is omitted deliberately — the frontend owns the
            // platform-appropriate phrasing ("Opening Spotify" on Android,
            // possibly a different verb on Linux) and can override with a
            // failure message if the launch doesn't work.
            Some(target) => Response::Action(serde_json::json!({
                "v": 1,
                "launch_app": target,
            })),
            None => Response::Text(
                match ctx.locale.as_str() {
                    "it" => "Cosa vuoi che apra?",
                    _ => "What would you like me to open?",
                }
                .to_string(),
            ),
        }
    }

    /// Typed-args path. The FunctionGemma router extracts the
    /// `app_name` slot directly so we skip `extract_target`'s trigger-
    /// word scan and use the model's value verbatim. Falls back to
    /// `execute` when the args JSON is missing the field or the model
    /// emitted something unparseable.
    fn execute_with_args(
        &self,
        input: &str,
        args_json: &str,
        ctx: &SkillContext,
    ) -> Response {
        let app_name = serde_json::from_str::<serde_json::Value>(args_json)
            .ok()
            .and_then(|v| v.get("app_name").and_then(|n| n.as_str()).map(String::from))
            .filter(|s| !s.trim().is_empty());

        match app_name {
            Some(name) => Response::Action(serde_json::json!({
                "v": 1,
                "launch_app": name,
            })),
            None => self.execute(input, ctx),
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

    #[test]
    fn execute_with_args_uses_app_name_directly() {
        // Router-extracted args bypass the trigger-word scan entirely;
        // even an utterance with no "open"/"launch"/etc. keyword gets
        // routed to the right app when the model picked the slot.
        let skill = OpenSkill::new();
        let response = skill.execute_with_args(
            "fire up spotify",
            r#"{"app_name":"Spotify"}"#,
            &ctx(),
        );
        match response {
            Response::Action(v) => {
                assert_eq!(v["launch_app"], "Spotify");
                assert_eq!(v["v"], 1);
            }
            other => panic!("expected Action, got {other:?}"),
        }
    }

    #[test]
    fn execute_with_args_falls_back_when_app_name_missing() {
        // Empty args object, malformed JSON, or whitespace-only
        // app_name should fall through to the keyword-scan path so the
        // skill behaves as if the router had emitted no args.
        let skill = OpenSkill::new();
        let response = skill.execute_with_args("open the camera", "{}", &ctx());
        match response {
            Response::Action(v) => assert_eq!(v["launch_app"], "the camera"),
            other => panic!("expected Action via fallback, got {other:?}"),
        }
    }

    #[test]
    fn execute_with_args_falls_back_on_blank_app_name() {
        let skill = OpenSkill::new();
        let response = skill.execute_with_args(
            "launch firefox",
            r#"{"app_name":"   "}"#,
            &ctx(),
        );
        match response {
            Response::Action(v) => assert_eq!(v["launch_app"], "firefox"),
            other => panic!("expected Action via fallback, got {other:?}"),
        }
    }

    #[test]
    fn italian_router_examples() {
        let skill = OpenSkill::new();
        let it = skill.example_utterances_for("it");
        let en = skill.example_utterances_for("en");
        assert_eq!(it.len(), en.len(), "Italian example count matches English");
        assert_ne!(it, en, "Italian examples are distinct from English");
        assert!(it.iter().any(|e| e.text == "apri spotify" && e.args == r#"{"app_name": "Spotify"}"#),
            "Italian utterance carries the canonical English app_name");
        assert!(it.iter().all(|e| e.args.contains("app_name")), "every open example supplies app_name");
        // Every Italian example must reuse a canonical English app_name —
        // the value feeds locale-agnostic app resolution, not display.
        assert!(
            it.iter().all(|e| en.iter().any(|x| x.args == e.args)),
            "an Italian example carries an app_name English never uses"
        );
        assert!(en.iter().any(|e| e.text == "open spotify"), "English arm unchanged");
        assert_eq!(skill.example_utterances_for("fr"), en, "unknown locale falls back to English");
    }

    #[test]
    fn execute_no_target_spanish_falls_back_to_english() {
        let skill = OpenSkill::new();
        let mut es = SkillContext::default();
        es.locale = "es".to_string();
        match skill.execute("open", &es) {
            Response::Text(s) => assert_eq!(s, "What would you like me to open?"),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn italian_imperative_clitic_forms_trigger_open() {
        let skill = OpenSkill::new();
        let mut ctx = SkillContext::default();
        ctx.locale = "it".to_string();
        // `aprimi X` = imperative `apri` + clitic `mi`. Natural Italian that
        // the plain `apri` trigger misses entirely.
        assert!(skill.score("aprimi duolingo", &ctx) >= 0.8, "aprimi must trigger open");
        assert!(skill.score("avviami spotify", &ctx) >= 0.8, "avviami must trigger open");
    }
}
