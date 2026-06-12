use ari_core::{ExampleUtterance, Response, Skill, SkillContext, Specificity};
use chrono::{Datelike, Local};

// English + Italian. Same union-dictionary pattern as `current_time` —
// keeps Stage 1 keyword routing fast for both languages, no cloud
// round-trip needed. Italian shapes after normalize_input:
//   - "che giorno è" / "che giorno è oggi" → ["che", "giorno"]
//   - "che data è" / "data di oggi" → ["che", "data"] / ["oggi", "data"]
//   - "che data abbiamo" → ["che", "data"]
//   - "in che giorno siamo" → ["che", "giorno"]
const TRIGGER_PHRASES: &[&[&str]] = &[
    // English
    &["what", "date"],
    &["today", "date"],
    &["current", "date"],
    &["what", "day"],
    &["which", "day"],
    // Italian
    &["che", "giorno"],
    &["che", "data"],
    &["data", "oggi"],
    &["data", "attuale"],
];

pub struct DateSkill;

impl DateSkill {
    pub fn new() -> Self {
        Self
    }
}

impl Default for DateSkill {
    fn default() -> Self {
        Self::new()
    }
}

impl Skill for DateSkill {
    fn id(&self) -> &str {
        "current_date"
    }

    fn description(&self) -> &str {
        "Tells today's date. Use when the user asks what day it is, what date it is, which day of the week it is, or anything about today's date."
    }

    fn specificity(&self) -> Specificity {
        Specificity::High
    }

    fn example_utterances(&self) -> &[ExampleUtterance] {
        &[
            ExampleUtterance { text: "what's the date today", args: "{}" },
            ExampleUtterance { text: "what day is it", args: "{}" },
            ExampleUtterance { text: "what's today's date", args: "{}" },
            ExampleUtterance { text: "which day of the week is it", args: "{}" },
            ExampleUtterance { text: "what date is it", args: "{}" },
            ExampleUtterance { text: "tell me today's date", args: "{}" },
            ExampleUtterance { text: "what day are we on", args: "{}" },
            ExampleUtterance { text: "is it Monday today", args: "{}" },
            ExampleUtterance { text: "what's the date", args: "{}" },
            ExampleUtterance { text: "do you know today's date", args: "{}" },
            ExampleUtterance { text: "can you tell me the date", args: "{}" },
            ExampleUtterance { text: "what day of the week is it today", args: "{}" },
            ExampleUtterance { text: "I need to know the date", args: "{}" },
            ExampleUtterance { text: "the date please", args: "{}" },
            ExampleUtterance { text: "is today a weekday", args: "{}" },
            ExampleUtterance { text: "what's today", args: "{}" },
            ExampleUtterance { text: "which day is today", args: "{}" },
            ExampleUtterance { text: "tell me what day it is", args: "{}" },
            ExampleUtterance { text: "date please", args: "{}" },
            ExampleUtterance { text: "current date", args: "{}" },
            ExampleUtterance { text: "what is today's date", args: "{}" },
            ExampleUtterance { text: "is it the weekend", args: "{}" },
            ExampleUtterance { text: "what day is today", args: "{}" },
            ExampleUtterance { text: "do you know what day it is", args: "{}" },
            ExampleUtterance { text: "I forgot what day it is", args: "{}" },
            ExampleUtterance { text: "is it still Tuesday", args: "{}" },
            ExampleUtterance { text: "what's the day today", args: "{}" },
            ExampleUtterance { text: "today's date please", args: "{}" },
            ExampleUtterance { text: "check the date for me", args: "{}" },
            ExampleUtterance { text: "could you tell me the date", args: "{}" },
        ]
    }

    fn score(&self, input: &str, _ctx: &SkillContext) -> f32 {
        let words: Vec<&str> = input.split_whitespace().collect();

        // English "time" / Italian "ora"/"ore" — any of these in the
        // input means the user wants the time skill, not date. Mirrors
        // the English-only guard that was here before; same intent.
        const TIME_WORDS: &[&str] = &["time", "ora", "ore"];
        if words.iter().any(|w| TIME_WORDS.contains(w)) {
            return 0.0;
        }

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
        // chrono's `%A` (weekday) and `%B` (month) format using the
        // system's C-locale by default — that's English-only on most
        // builds. Hand-roll the locale-specific tables so the response
        // doesn't depend on what locales the host happens to have
        // installed at the OS level.
        let weekday_idx = now.weekday().num_days_from_monday() as usize; // 0..=6
        let month_idx = now.month() as usize; // 1..=12
        let day = now.day();
        let year = now.year();
        let response = match ctx.locale.as_str() {
            "it" => {
                let weekday = ITALIAN_WEEKDAYS[weekday_idx];
                let month = ITALIAN_MONTHS[month_idx];
                format!("Oggi è {} {} {} {}.", weekday, day, month, year)
            }
            _ => {
                let formatted = now.format("%A, %B %-d, %Y").to_string();
                format!("Today is {}.", formatted)
            }
        };
        Response::Text(response)
    }
}

// Index 0 = Monday (chrono's `num_days_from_monday`).
const ITALIAN_WEEKDAYS: [&str; 7] = [
    "lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica",
];

// Index 0 unused — months are 1..=12.
const ITALIAN_MONTHS: [&str; 13] = [
    "", "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio",
    "agosto", "settembre", "ottobre", "novembre", "dicembre",
];

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> SkillContext {
        SkillContext::default()
    }

    // Score formula: same as CurrentTimeSkill — 0.5 + (matched/total * 0.5)
    // But returns 0.0 if "time" is in the input

    #[test]
    fn score_what_date() {
        let skill = DateSkill::new();
        // "what is the date" = 4 words, ["what","date"] match 2
        // coverage = 2/4 = 0.5, score = 0.75
        assert_eq!(skill.score("what is the date", &ctx()), 0.75);
    }

    #[test]
    fn score_what_day() {
        let skill = DateSkill::new();
        // "what day is it" = 4 words, ["what","day"] match 2
        assert_eq!(skill.score("what day is it", &ctx()), 0.75);
    }

    #[test]
    fn score_current_date() {
        let skill = DateSkill::new();
        // "current date" = 2 words, 2 match, coverage = 1.0
        assert_eq!(skill.score("current date", &ctx()), 1.0);
    }

    #[test]
    fn score_zero_when_time_present() {
        let skill = DateSkill::new();
        // Disambiguation: "time" in input → 0.0
        assert_eq!(skill.score("what time is it", &ctx()), 0.0);
        assert_eq!(skill.score("date and time", &ctx()), 0.0);
    }

    #[test]
    fn score_zero_on_unrelated() {
        let skill = DateSkill::new();
        assert_eq!(skill.score("hello there", &ctx()), 0.0);
        assert_eq!(skill.score("open spotify", &ctx()), 0.0);
    }

    #[test]
    fn score_zero_when_keyword_is_substring_of_other_word() {
        // Regression: scorer used `w.contains(**keyword)` which
        // false-positived on words containing the keyword as a
        // substring — "today" tripped "today" inside "todays" etc.
        let skill = DateSkill::new();
        assert_eq!(
            skill.score("what is sundays special at the deli", &ctx()),
            0.0,
        );
        assert_eq!(skill.score("what is the holiday discount", &ctx()), 0.0);
        // "what" and "today" both as standalone words still trigger.
        assert!(skill.score("what is the date today", &ctx()) > 0.0);
    }

    #[test]
    fn score_italian_che_giorno() {
        let skill = DateSkill::new();
        // "che giorno è" — the canonical Italian "what day is it"
        let score = skill.score("che giorno è", &ctx());
        assert!(score > 0.5, "expected score > 0.5, got {score}");
    }

    #[test]
    fn score_italian_che_data() {
        let skill = DateSkill::new();
        // "che data è oggi" — Italian "what date is it today"
        let score = skill.score("che data è oggi", &ctx());
        assert!(score > 0.5, "expected score > 0.5, got {score}");
    }

    #[test]
    fn score_italian_zero_when_ora_present() {
        // "ora" (Italian for "hour") in the input means the user wants
        // the time skill, not date — same logic as the existing
        // English "time" guard. Without this, "che ora è" would match
        // both date (no — actually no Italian date phrase, fine) and
        // current_time. Just a sanity check that date doesn't
        // false-positive on Italian time queries.
        let skill = DateSkill::new();
        assert_eq!(skill.score("che ora è", &ctx()), 0.0);
    }

    #[test]
    fn execute_italian_uses_italian_weekday_and_month() {
        let skill = DateSkill::new();
        let mut italian = SkillContext::default();
        italian.locale = "it".to_string();
        let resp = skill.execute("che giorno è oggi", &italian);
        match resp {
            Response::Text(s) => {
                // Shape: "Oggi è <weekday> <day> <month> <year>."
                assert!(s.starts_with("Oggi è "));
                assert!(s.ends_with('.'));
                // At least one Italian weekday or month must be present —
                // the exact ones depend on test-run date, so we check
                // membership rather than equality.
                let italian_weekday = ITALIAN_WEEKDAYS.iter().any(|w| s.contains(*w));
                let italian_month = ITALIAN_MONTHS.iter().any(|m| !m.is_empty() && s.contains(*m));
                assert!(italian_weekday, "no Italian weekday in: {s}");
                assert!(italian_month, "no Italian month in: {s}");
                // No English months should appear
                assert!(!s.contains("January"));
                assert!(!s.contains("Today is"));
            }
            _ => panic!("expected Text response"),
        }
    }

    #[test]
    fn execute_format_weekday_month_day_year() {
        let skill = DateSkill::new();
        let resp = skill.execute("what date is it", &ctx());
        match resp {
            Response::Text(s) => {
                // Format: "Today is Wednesday, April 6, 2026."
                let inner = s
                    .strip_prefix("Today is ")
                    .expect("should start with 'Today is '");
                let inner = inner.strip_suffix('.').expect("should end with '.'");
                let parts: Vec<&str> = inner.splitn(2, ", ").collect();
                assert_eq!(parts.len(), 2, "expected 'Weekday, Month Day, Year' got: {inner}");
                let weekdays = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"];
                assert!(weekdays.contains(&parts[0]), "bad weekday: {}", parts[0]);
                // Rest should be "Month Day, Year"
                assert!(parts[1].contains(", "), "missing year separator in: {}", parts[1]);
            }
            _ => panic!("expected Text"),
        }
    }

    #[test]
    fn specificity_is_high() {
        assert_eq!(DateSkill::new().specificity(), Specificity::High);
    }

    #[test]
    fn execute_spanish_falls_back_to_english() {
        let skill = DateSkill::new();
        let mut es = SkillContext::default();
        es.locale = "es".to_string();
        let resp = skill.execute("what date is it", &es);
        match resp {
            Response::Text(s) => assert!(s.starts_with("Today is "), "expected English fallback: {s}"),
            _ => panic!("expected Text"),
        }
    }
}
