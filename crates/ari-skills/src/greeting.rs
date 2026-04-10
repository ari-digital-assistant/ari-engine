use ari_core::{Response, Skill, SkillContext, Specificity};

const GREETINGS: &[&str] = &[
    "hello", "hi", "hey", "heya", "howdy", "greetings", "good morning",
    "good afternoon", "good evening", "yo", "sup", "hiya", "ello",
    "hey ari", "hi ari", "hello ari",
];

const HOW_ARE_YOU: &[&[&str]] = &[
    &["how", "are", "you"],
    &["how", "you", "doing"],
    &["how", "is", "it", "going"],
    &["what", "is", "up"],
    &["what", "up"],
];

const RESPONSES: &[&str] = &[
    "Hey there! What can I do for you?",
    "Hello! How can I help?",
    "Hi! What's on your mind?",
    "Hey! Ready when you are.",
];

pub struct GreetingSkill;

impl GreetingSkill {
    pub fn new() -> Self {
        Self
    }
}

impl Default for GreetingSkill {
    fn default() -> Self {
        Self::new()
    }
}

impl Skill for GreetingSkill {
    fn id(&self) -> &str {
        "greeting"
    }

    fn description(&self) -> &str {
        "Responds to greetings. Use when the user says hello, hi, or asks how Ari is doing."
    }

    fn specificity(&self) -> Specificity {
        Specificity::Low
    }

    fn score(&self, input: &str, _ctx: &SkillContext) -> f32 {
        let words: Vec<&str> = input.split_whitespace().collect();

        for phrase in HOW_ARE_YOU {
            let matched = phrase
                .iter()
                .filter(|kw| words.contains(kw))
                .count();
            if matched == phrase.len() {
                return 0.9;
            }
        }

        for greeting in GREETINGS {
            let greeting_words: Vec<&str> = greeting.split_whitespace().collect();
            let matched = greeting_words
                .iter()
                .filter(|kw| words.contains(kw))
                .count();
            if matched == greeting_words.len() {
                let coverage = matched as f32 / words.len().max(1) as f32;
                return 0.6 + (coverage * 0.4);
            }
        }

        0.0
    }

    fn execute(&self, input: &str, _ctx: &SkillContext) -> Response {
        let words: Vec<&str> = input.split_whitespace().collect();
        let is_how_are_you = HOW_ARE_YOU.iter().any(|phrase| {
            phrase.iter().all(|kw| words.contains(kw))
        });

        if is_how_are_you {
            return Response::Text("I'm doing great, thanks for asking! How can I help you?".to_string());
        }

        let idx = input.len() % RESPONSES.len();
        Response::Text(RESPONSES[idx].to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> SkillContext {
        SkillContext::default()
    }

    // Score for HOW_ARE_YOU phrases: always 0.9
    // Score for GREETINGS: 0.6 + (matched/total_words * 0.4)

    #[test]
    fn score_single_word_greeting() {
        let skill = GreetingSkill::new();
        // "hello" = 1 word, 1 match, coverage = 1.0
        // score = 0.6 + 1.0*0.4 = 1.0
        assert_eq!(skill.score("hello", &ctx()), 1.0);
        assert_eq!(skill.score("hi", &ctx()), 1.0);
        assert_eq!(skill.score("hey", &ctx()), 1.0);
        assert_eq!(skill.score("heya", &ctx()), 1.0);
        assert_eq!(skill.score("yo", &ctx()), 1.0);
    }

    #[test]
    fn score_greeting_diluted_by_extra_words() {
        let skill = GreetingSkill::new();
        // "hello there" = 2 words, "hello" matches, coverage = 1/2
        // score = 0.6 + 0.5*0.4 = 0.8
        assert_eq!(skill.score("hello there", &ctx()), 0.8);
    }

    #[test]
    fn score_multi_word_greeting() {
        let skill = GreetingSkill::new();
        // "good morning" = 2 words, both match the GREETINGS entry, coverage = 2/2 = 1.0
        // score = 0.6 + 1.0*0.4 = 1.0
        assert_eq!(skill.score("good morning", &ctx()), 1.0);
    }

    #[test]
    fn score_how_are_you_always_09() {
        let skill = GreetingSkill::new();
        assert_eq!(skill.score("how are you", &ctx()), 0.9);
        assert_eq!(skill.score("how are you doing today", &ctx()), 0.9);
    }

    #[test]
    fn score_what_is_up() {
        let skill = GreetingSkill::new();
        assert_eq!(skill.score("what is up", &ctx()), 0.9);
    }

    #[test]
    fn score_zero_on_unrelated() {
        let skill = GreetingSkill::new();
        assert_eq!(skill.score("what time is it", &ctx()), 0.0);
        assert_eq!(skill.score("calculate 2 plus 2", &ctx()), 0.0);
    }

    #[test]
    fn execute_how_are_you_returns_specific_response() {
        let skill = GreetingSkill::new();
        let resp = skill.execute("how are you", &ctx());
        assert_eq!(
            matches!(resp, Response::Text(ref s) if s == "I'm doing great, thanks for asking! How can I help you?"),
            true
        );
    }

    #[test]
    fn execute_what_is_up_returns_specific_response() {
        let skill = GreetingSkill::new();
        let resp = skill.execute("what is up", &ctx());
        match resp {
            Response::Text(s) => assert_eq!(s, "I'm doing great, thanks for asking! How can I help you?"),
            _ => panic!("expected Text"),
        }
    }

    #[test]
    fn execute_regular_greeting_picks_from_responses() {
        let skill = GreetingSkill::new();
        // Response selection: input.len() % RESPONSES.len()
        // "hello" = 5 chars, 5 % 4 = 1 → RESPONSES[1]
        let resp = skill.execute("hello", &ctx());
        match resp {
            Response::Text(s) => assert_eq!(s, "Hello! How can I help?"),
            _ => panic!("expected Text"),
        }
    }

    #[test]
    fn execute_different_input_different_response() {
        let skill = GreetingSkill::new();
        // "hi" = 2 chars, 2 % 4 = 2 → RESPONSES[2]
        let resp = skill.execute("hi", &ctx());
        match resp {
            Response::Text(s) => assert_eq!(s, "Hi! What's on your mind?"),
            _ => panic!("expected Text"),
        }
    }

    #[test]
    fn specificity_is_low() {
        assert_eq!(GreetingSkill::new().specificity(), Specificity::Low);
    }
}
