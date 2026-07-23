use ari_core::{ExampleUtterance, Response, Skill, SkillContext, Specificity};

// English + Italian trigger words. Same union-dictionary approach as
// the reminder skill's parser — words don't collide across these
// languages, so a single contains-check disambiguates.
const GREETINGS: &[&str] = &[
    // English
    "hello", "hi", "hey", "heya", "howdy", "greetings", "good morning",
    "good afternoon", "good evening", "yo", "sup", "hiya", "ello",
    "hey ari", "hi ari", "hello ari",
    // Italian. No "buonanotte" — it's a farewell, and English doesn't
    // list "good night" either. Answering it with "Ciao! Cosa posso fare
    // per te?" is the wrong end of the conversation; let it fall through
    // to the assistant, which is what the Italian router examples assume.
    "ciao", "salve", "buongiorno", "buonasera",
    "ciao ari", "salve ari",
];

const HOW_ARE_YOU: &[&[&str]] = &[
    // English
    &["how", "are", "you"],
    &["how", "you", "doing"],
    &["how", "is", "it", "going"],
    &["what", "is", "up"],
    &["what", "up"],
    // Italian
    &["come", "stai"],
    &["come", "va"],
];

const RESPONSES_EN: &[&str] = &[
    "Hey there! What can I do for you?",
    "Hello! How can I help?",
    "Hi! What's on your mind?",
    "Hey! Ready when you are.",
];

const RESPONSES_IT: &[&str] = &[
    "Ciao! Cosa posso fare per te?",
    "Ciao! Come posso aiutarti?",
    "Ciao! A cosa stai pensando?",
    "Ciao! Sono qui quando vuoi.",
];

fn responses_for_locale(locale: &str) -> &'static [&'static str] {
    match locale {
        "it" => RESPONSES_IT,
        _ => RESPONSES_EN,
    }
}

fn how_are_you_response(locale: &str) -> &'static str {
    match locale {
        "it" => "Sto benissimo, grazie! Come posso aiutarti?",
        _ => "I'm doing great, thanks for asking! How can I help you?",
    }
}

// Router training examples. Natural raw text as a user would actually say it.
// (Whether the generator should normalise these to match inference is the
// parity spike's question — not this file's.)
const GREETING_EXAMPLES_EN: &[ExampleUtterance] = &[
    ExampleUtterance { text: "hello", args: "{}" },
    ExampleUtterance { text: "hi", args: "{}" },
    ExampleUtterance { text: "hey", args: "{}" },
    ExampleUtterance { text: "hey there", args: "{}" },
    ExampleUtterance { text: "howdy", args: "{}" },
    ExampleUtterance { text: "good morning", args: "{}" },
    ExampleUtterance { text: "good afternoon", args: "{}" },
    ExampleUtterance { text: "good evening", args: "{}" },
    ExampleUtterance { text: "yo", args: "{}" },
    ExampleUtterance { text: "sup", args: "{}" },
    ExampleUtterance { text: "what's up", args: "{}" },
    ExampleUtterance { text: "hiya", args: "{}" },
    ExampleUtterance { text: "heya", args: "{}" },
    ExampleUtterance { text: "hello ari", args: "{}" },
    ExampleUtterance { text: "hi ari", args: "{}" },
    ExampleUtterance { text: "hey ari", args: "{}" },
    ExampleUtterance { text: "good morning ari", args: "{}" },
    ExampleUtterance { text: "greetings", args: "{}" },
    ExampleUtterance { text: "how are you", args: "{}" },
    ExampleUtterance { text: "how are you doing", args: "{}" },
    ExampleUtterance { text: "how's it going", args: "{}" },
    ExampleUtterance { text: "what's going on", args: "{}" },
    ExampleUtterance { text: "how do you do", args: "{}" },
    ExampleUtterance { text: "nice to meet you", args: "{}" },
    ExampleUtterance { text: "hey there ari", args: "{}" },
    ExampleUtterance { text: "morning", args: "{}" },
    ExampleUtterance { text: "evening", args: "{}" },
    ExampleUtterance { text: "how are things", args: "{}" },
    ExampleUtterance { text: "how you doing", args: "{}" },
    ExampleUtterance { text: "what's happening", args: "{}" },
];

const GREETING_EXAMPLES_IT: &[ExampleUtterance] = &[
    ExampleUtterance { text: "ciao", args: "{}" },
    ExampleUtterance { text: "salve", args: "{}" },
    ExampleUtterance { text: "buongiorno", args: "{}" },
    ExampleUtterance { text: "buonasera", args: "{}" },
    ExampleUtterance { text: "buon pomeriggio", args: "{}" },
    ExampleUtterance { text: "ehi", args: "{}" },
    ExampleUtterance { text: "ehilà", args: "{}" },
    ExampleUtterance { text: "buondì", args: "{}" },
    ExampleUtterance { text: "ciao ari", args: "{}" },
    ExampleUtterance { text: "salve ari", args: "{}" },
    ExampleUtterance { text: "buongiorno ari", args: "{}" },
    ExampleUtterance { text: "buonasera ari", args: "{}" },
    ExampleUtterance { text: "ehi ari", args: "{}" },
    ExampleUtterance { text: "come stai", args: "{}" },
    ExampleUtterance { text: "come va", args: "{}" },
    ExampleUtterance { text: "come sta", args: "{}" },
    ExampleUtterance { text: "tutto bene", args: "{}" },
    ExampleUtterance { text: "tutto a posto", args: "{}" },
    ExampleUtterance { text: "come butta", args: "{}" },
    ExampleUtterance { text: "come te la passi", args: "{}" },
    ExampleUtterance { text: "come vanno le cose", args: "{}" },
    ExampleUtterance { text: "come procede", args: "{}" },
    ExampleUtterance { text: "che si dice", args: "{}" },
    ExampleUtterance { text: "che mi racconti", args: "{}" },
    ExampleUtterance { text: "novità", args: "{}" },
    ExampleUtterance { text: "piacere di conoscerti", args: "{}" },
    ExampleUtterance { text: "come stai oggi", args: "{}" },
    ExampleUtterance { text: "ciao come stai", args: "{}" },
    ExampleUtterance { text: "salve come sta", args: "{}" },
    ExampleUtterance { text: "come va la vita", args: "{}" },
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
        "Responds to greetings. Use when the user says hello, hi, hey, good morning, good evening, howdy, what's up, or asks how Ari is doing."
    }

    fn specificity(&self) -> Specificity {
        Specificity::Low
    }

    fn example_utterances(&self) -> &[ExampleUtterance] {
        GREETING_EXAMPLES_EN
    }

    fn example_utterances_for(&self, locale: &str) -> &[ExampleUtterance] {
        match locale {
            "it" => GREETING_EXAMPLES_IT,
            _ => GREETING_EXAMPLES_EN,
        }
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

    fn execute(&self, input: &str, ctx: &SkillContext) -> Response {
        let words: Vec<&str> = input.split_whitespace().collect();
        let is_how_are_you = HOW_ARE_YOU.iter().any(|phrase| {
            phrase.iter().all(|kw| words.contains(kw))
        });

        if is_how_are_you {
            return Response::Text(how_are_you_response(ctx.locale.as_str()).to_string());
        }

        let responses = responses_for_locale(ctx.locale.as_str());
        let idx = input.len() % responses.len();
        Response::Text(responses[idx].to_string())
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
        // Response selection: input.len() % RESPONSES_EN.len()
        // "hello" = 5 chars, 5 % 4 = 1 → RESPONSES_EN[1]
        let resp = skill.execute("hello", &ctx());
        match resp {
            Response::Text(s) => assert_eq!(s, "Hello! How can I help?"),
            _ => panic!("expected Text"),
        }
    }

    #[test]
    fn execute_italian_how_are_you() {
        let skill = GreetingSkill::new();
        let mut italian = SkillContext::default();
        italian.locale = "it".to_string();
        let resp = skill.execute("come stai", &italian);
        match resp {
            Response::Text(s) => assert_eq!(
                s,
                "Sto benissimo, grazie! Come posso aiutarti?"
            ),
            _ => panic!("expected Text"),
        }
    }

    #[test]
    fn execute_italian_regular_greeting_picks_from_italian_responses() {
        let skill = GreetingSkill::new();
        let mut italian = SkillContext::default();
        italian.locale = "it".to_string();
        // "ciao" = 4 chars, 4 % 4 = 0 → RESPONSES_IT[0]
        let resp = skill.execute("ciao", &italian);
        match resp {
            Response::Text(s) => assert_eq!(s, "Ciao! Cosa posso fare per te?"),
            _ => panic!("expected Text"),
        }
    }

    #[test]
    fn score_italian_greeting_triggers() {
        let skill = GreetingSkill::new();
        // Italian greeting "ciao" must score above 0 — the union
        // dictionary lets the same scorer recognise both languages.
        assert_eq!(skill.score("ciao", &ctx()), 1.0);
        assert_eq!(skill.score("buongiorno", &ctx()), 1.0);
    }

    #[test]
    fn farewells_are_not_greetings() {
        let skill = GreetingSkill::new();
        // Both languages agree: saying good night is leaving, not arriving.
        assert_eq!(skill.score("buonanotte", &ctx()), 0.0);
        assert_eq!(skill.score("good night", &ctx()), 0.0);
        // The evening greeting it rhymes with is still a greeting.
        assert_eq!(skill.score("buonasera", &ctx()), 1.0);
        assert_eq!(skill.score("good evening", &ctx()), 1.0);
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

    #[test]
    fn italian_router_examples() {
        let skill = GreetingSkill::new();
        let it = skill.example_utterances_for("it");
        let en = skill.example_utterances_for("en");
        assert_eq!(it.len(), en.len(), "Italian example count matches English");
        assert_ne!(it, en, "Italian examples are distinct from English");
        assert!(it.iter().any(|e| e.text == "ciao"), "canonical Italian greeting present");
        assert!(it.iter().all(|e| e.args == "{}"), "greeting is parameterless");
        assert!(en.iter().any(|e| e.text == "hello"), "English arm unchanged");
        assert_eq!(skill.example_utterances_for("fr"), en, "unknown locale falls back to English");
    }

    #[test]
    fn execute_spanish_locale_falls_back_to_english() {
        let skill = GreetingSkill::new();
        let mut es = SkillContext::default();
        es.locale = "es".to_string();
        // After the strip, "es" is no longer special-cased -> English responses.
        // "hello" = 5 chars, 5 % 4 = 1 -> RESPONSES_EN[1].
        let resp = skill.execute("hello", &es);
        match resp {
            Response::Text(s) => assert_eq!(s, "Hello! How can I help?"),
            _ => panic!("expected Text"),
        }
    }
}
