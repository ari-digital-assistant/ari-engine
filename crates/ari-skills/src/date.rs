use ari_core::{Response, Skill, SkillContext, Specificity};
use chrono::Local;

const TRIGGER_PHRASES: &[&[&str]] = &[
    &["what", "date"],
    &["today", "date"],
    &["current", "date"],
    &["what", "day"],
    &["which", "day"],
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
        "Tells today's date. Use when the user asks what day or date it is."
    }

    fn specificity(&self) -> Specificity {
        Specificity::High
    }

    fn score(&self, input: &str, _ctx: &SkillContext) -> f32 {
        let words: Vec<&str> = input.split_whitespace().collect();

        // "time" in the input likely means the user wants the time skill, not date
        if words.contains(&"time") {
            return 0.0;
        }

        let mut best_score: f32 = 0.0;

        for phrase in TRIGGER_PHRASES {
            let matched = phrase
                .iter()
                .filter(|keyword| words.iter().any(|w| w.contains(**keyword)))
                .count();

            if matched == phrase.len() {
                let coverage = matched as f32 / words.len().max(1) as f32;
                let phrase_score = 0.5 + (coverage * 0.5);
                best_score = best_score.max(phrase_score);
            }
        }

        best_score
    }

    fn execute(&self, _input: &str, _ctx: &SkillContext) -> Response {
        let now = Local::now();
        let formatted = now.format("%A, %B %-d, %Y").to_string();
        Response::Text(format!("Today is {}.", formatted))
    }
}

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
}
