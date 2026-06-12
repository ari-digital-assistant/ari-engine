use ari_core::{ExampleUtterance, Response, Skill, SkillContext, Specificity};
use chrono::Local;

// English + Italian. The scorer is locale-agnostic; the words don't
// collide across languages, so a single union table keeps Stage 1
// keyword routing fast (no cloud round-trip needed for "che ora è").
//
// Italian trigger shapes after `normalize_input` (lowercases, strips
// elisions like `l'ora` → `l ora`):
//   - "che ora è" / "a che ora" → ["che", "ora"]
//   - "che ore sono" / "che ore" → ["che", "ore"]
//   - "dimmi l'ora" → "dimmi l ora" → ["dimmi", "ora"]
//   - "ora attuale" → ["ora", "attuale"]
const TRIGGER_PHRASES: &[&[&str]] = &[
    // English
    &["what", "time"],
    &["current", "time"],
    &["tell", "time"],
    &["what is", "time"],
    // Italian
    &["che", "ora"],
    &["che", "ore"],
    &["dimmi", "ora"],
    &["ora", "attuale"],
];

pub struct CurrentTimeSkill;

impl CurrentTimeSkill {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CurrentTimeSkill {
    fn default() -> Self {
        Self::new()
    }
}

impl Skill for CurrentTimeSkill {
    fn id(&self) -> &str {
        "current_time"
    }

    fn description(&self) -> &str {
        "Tells the current time. Use when the user asks what time it is, what hour it is, whether it is morning or afternoon, or anything about the current time of day."
    }

    fn specificity(&self) -> Specificity {
        Specificity::High
    }

    fn example_utterances(&self) -> &[ExampleUtterance] {
        &[
            ExampleUtterance { text: "what time is it", args: "{}" },
            ExampleUtterance { text: "what's the time", args: "{}" },
            ExampleUtterance { text: "tell me the time", args: "{}" },
            ExampleUtterance { text: "what time do you have", args: "{}" },
            ExampleUtterance { text: "do you know what time it is", args: "{}" },
            ExampleUtterance { text: "what hour is it", args: "{}" },
            ExampleUtterance { text: "can you tell me the time", args: "{}" },
            ExampleUtterance { text: "what's the current time", args: "{}" },
            ExampleUtterance { text: "is it morning or afternoon", args: "{}" },
            ExampleUtterance { text: "how late is it", args: "{}" },
            ExampleUtterance { text: "what time is it right now", args: "{}" },
            ExampleUtterance { text: "got the time", args: "{}" },
            ExampleUtterance { text: "what's the time now", args: "{}" },
            ExampleUtterance { text: "could you tell me the time please", args: "{}" },
            ExampleUtterance { text: "I need to know what time it is", args: "{}" },
            ExampleUtterance { text: "time please", args: "{}" },
            ExampleUtterance { text: "what time have you got", args: "{}" },
            ExampleUtterance { text: "is it late", args: "{}" },
            ExampleUtterance { text: "am or pm right now", args: "{}" },
            ExampleUtterance { text: "check the time for me", args: "{}" },
            ExampleUtterance { text: "I wonder what time it is", args: "{}" },
            ExampleUtterance { text: "any idea what time it is", args: "{}" },
            ExampleUtterance { text: "do you have the time", args: "{}" },
            ExampleUtterance { text: "quick, what time is it", args: "{}" },
            ExampleUtterance { text: "is it still early", args: "{}" },
            ExampleUtterance { text: "how early is it", args: "{}" },
            ExampleUtterance { text: "tell me the current time", args: "{}" },
            ExampleUtterance { text: "what's the clock say", args: "{}" },
            ExampleUtterance { text: "current time please", args: "{}" },
        ]
    }

    fn score(&self, input: &str, _ctx: &SkillContext) -> f32 {
        let words: Vec<&str> = input.split_whitespace().collect();

        let mut best_score: f32 = 0.0;

        for phrase in TRIGGER_PHRASES {
            let matched = phrase
                .iter()
                .filter(|keyword| words.iter().any(|w| w == *keyword))
                .count();

            if matched == phrase.len() {
                let coverage = matched as f32 / words.len().max(1) as f32;
                let phrase_score = 0.5 + (coverage * 0.5);
                best_score = best_score.max(phrase_score);
            }
        }

        best_score
    }

    fn execute(&self, _input: &str, ctx: &SkillContext) -> Response {
        let now = Local::now();
        // Locale-aware time format. English keeps 12-hour with AM/PM
        // ("It's 3:25 PM."). Other shipped locales use 24-hour ("alle
        // 15:25") — that's the conventional written form in IT/ES/FR/DE
        // and avoids translating the AM/PM tokens.
        let response = match ctx.locale.as_str() {
            "it" => format!("Sono le {}.", now.format("%H:%M")),
            _ => format!("It's {}.", now.format("%-I:%M %p")),
        };
        Response::Text(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> SkillContext {
        SkillContext::default()
    }

    // Score formula: 0.5 + (matched_keywords / total_words * 0.5)
    // Triggers: ["what","time"], ["current","time"], ["tell","time"], ["what is","time"]

    #[test]
    fn score_what_time_is_it() {
        let skill = CurrentTimeSkill::new();
        // "what time is it" = 4 words, ["what","time"] matches 2 keywords
        // coverage = 2/4 = 0.5, score = 0.5 + 0.5*0.5 = 0.75
        assert_eq!(skill.score("what time is it", &ctx()), 0.75);
    }

    #[test]
    fn score_current_time() {
        let skill = CurrentTimeSkill::new();
        // "current time" = 2 words, 2 keywords match, coverage = 1.0
        // score = 0.5 + 1.0*0.5 = 1.0
        assert_eq!(skill.score("current time", &ctx()), 1.0);
    }

    #[test]
    fn score_tell_me_the_time() {
        let skill = CurrentTimeSkill::new();
        // "tell me the time" = 4 words, ["tell","time"] = 2 match
        // coverage = 2/4 = 0.5, score = 0.75
        assert_eq!(skill.score("tell me the time", &ctx()), 0.75);
    }

    #[test]
    fn score_diluted_by_extra_words() {
        let skill = CurrentTimeSkill::new();
        // "can you please tell me the time right now" = 9 words, 2 match
        // coverage = 2/9 ≈ 0.222, score = 0.5 + 0.222*0.5 ≈ 0.611
        let score = skill.score("can you please tell me the time right now", &ctx());
        assert!((score - 0.611).abs() < 0.01, "score was {score}");
    }

    #[test]
    fn score_zero_on_no_keyword_match() {
        let skill = CurrentTimeSkill::new();
        assert_eq!(skill.score("hello there", &ctx()), 0.0);
        assert_eq!(skill.score("what is the weather", &ctx()), 0.0);
    }

    #[test]
    fn score_zero_on_partial_keyword() {
        let skill = CurrentTimeSkill::new();
        // "what" alone doesn't trigger — needs "what" AND "time"
        assert_eq!(skill.score("what is up", &ctx()), 0.0);
    }

    #[test]
    fn score_zero_when_keyword_is_substring_of_other_word() {
        // Regression: scorer used `w.contains(**keyword)` which
        // false-positived on words containing the keyword as a
        // substring — "runtimes" tripped "time", "lifetime" likewise.
        // Word-equality is the right test.
        let skill = CurrentTimeSkill::new();
        assert_eq!(
            skill.score("what does the internet say about async runtimes in rust", &ctx()),
            0.0,
        );
        assert_eq!(skill.score("what is my lifetime achievement", &ctx()), 0.0);
        assert_eq!(skill.score("what about overtime pay", &ctx()), 0.0);
    }

    #[test]
    fn execute_format_matches_12hr_with_am_pm() {
        let skill = CurrentTimeSkill::new();
        let resp = skill.execute("what time is it", &ctx());
        match resp {
            Response::Text(s) => {
                // Format: "It's H:MM AM." or "It's HH:MM PM."
                let inner = s.strip_prefix("It's ").expect("should start with 'It's '");
                let inner = inner.strip_suffix('.').expect("should end with '.'");
                // Must contain a colon and end with AM or PM
                assert!(inner.contains(':'), "no colon in time: {inner}");
                assert!(
                    inner.ends_with("AM") || inner.ends_with("PM"),
                    "no AM/PM in time: {inner}"
                );
                // Hour part should be 1-12
                let hour: u32 = inner.split(':').next().unwrap().parse().unwrap();
                assert!((1..=12).contains(&hour), "hour out of range: {hour}");
                // Minute part should be 00-59
                let min_str = &inner.split(':').nth(1).unwrap()[..2];
                let min: u32 = min_str.parse().unwrap();
                assert!(min <= 59, "minute out of range: {min}");
            }
            _ => panic!("expected Text response"),
        }
    }

    #[test]
    fn specificity_is_high() {
        assert_eq!(CurrentTimeSkill::new().specificity(), Specificity::High);
    }

    #[test]
    fn execute_italian_uses_24h_and_italian_text() {
        let skill = CurrentTimeSkill::new();
        let mut italian = SkillContext::default();
        italian.locale = "it".to_string();
        let resp = skill.execute("che ora e", &italian);
        match resp {
            Response::Text(s) => {
                // Italian: "Sono le HH:MM." — 24-hour, no AM/PM, leading
                // "Sono le". Don't pin the exact time; just shape.
                assert!(
                    s.starts_with("Sono le "),
                    "Italian response should start with 'Sono le ': {s}"
                );
                assert!(s.ends_with('.'));
                assert!(!s.contains("AM"));
                assert!(!s.contains("PM"));
                // Pull the HH:MM out of "Sono le HH:MM."
                let inner = s
                    .strip_prefix("Sono le ")
                    .and_then(|s| s.strip_suffix('.'))
                    .expect("expected 'Sono le HH:MM.' shape");
                assert!(inner.contains(':'), "expected HH:MM, got {inner}");
            }
            _ => panic!("expected Text response"),
        }
    }

    #[test]
    fn score_italian_che_ora() {
        let skill = CurrentTimeSkill::new();
        // "che ora è" — the canonical Italian "what time is it"
        // After normalize_input("che ora è", "it") the input stays as
        // "che ora è" (lowercase, è preserved as alphanumeric). Words
        // = ["che", "ora", "è"]. Phrase ["che", "ora"] matches:
        // coverage = 2/3, score = 0.5 + 0.667*0.5 ≈ 0.833.
        let score = skill.score("che ora è", &ctx());
        assert!(score > 0.5, "expected score > 0.5, got {score}");
    }

    #[test]
    fn score_italian_che_ore_sono() {
        let skill = CurrentTimeSkill::new();
        // "che ore sono" — the other common Italian time query
        let score = skill.score("che ore sono", &ctx());
        assert!(score > 0.5, "expected score > 0.5, got {score}");
    }

    #[test]
    fn score_italian_dimmi_lora_after_normalisation() {
        let skill = CurrentTimeSkill::new();
        // "dimmi l'ora" → after `strip_italian_elisions` becomes
        // "dimmi l ora" (3 tokens). Phrase ["dimmi", "ora"] matches.
        let score = skill.score("dimmi l ora", &ctx());
        assert!(score > 0.5, "expected score > 0.5, got {score}");
    }

    #[test]
    fn execute_unknown_locale_falls_back_to_english() {
        let skill = CurrentTimeSkill::new();
        let mut other = SkillContext::default();
        other.locale = "ja".to_string();
        let resp = skill.execute("what time is it", &other);
        match resp {
            Response::Text(s) => assert!(s.starts_with("It's "), "fallback to English: {s}"),
            _ => panic!("expected Text response"),
        }
    }

    #[test]
    fn execute_spanish_falls_back_to_english_format() {
        let skill = CurrentTimeSkill::new();
        let mut es = SkillContext::default();
        es.locale = "es".to_string();
        let resp = skill.execute("what time is it", &es);
        match resp {
            Response::Text(s) => assert!(s.starts_with("It's "), "expected English fallback: {s}"),
            _ => panic!("expected Text response"),
        }
    }
}
