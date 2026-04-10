use ari_core::{Response, Skill, SkillContext, Specificity};

const TRIGGER_PHRASES: &[&[&str]] = &[
    &["search", "for"],
    &["look", "up"],
    &["google"],
    &["search"],
    &["find"],
];

const QUESTION_STARTERS: &[&[&str]] = &[
    &["where", "can"],
    &["where", "do"],
    &["where", "is"],
    &["where", "are"],
    &["how", "do"],
    &["how", "to"],
    &["how", "can"],
    &["who", "is"],
    &["who", "are"],
    &["who", "was"],
    &["why", "is"],
    &["why", "do"],
    &["why", "does"],
    &["why", "are"],
];

pub struct SearchSkill;

impl SearchSkill {
    pub fn new() -> Self {
        Self
    }
}

impl Default for SearchSkill {
    fn default() -> Self {
        Self::new()
    }
}

fn extract_query_explicit(input: &str) -> Option<String> {
    let skip_words = ["search", "for", "look", "up", "google", "find", "please", "can", "you", "me"];

    let words: Vec<&str> = input.split_whitespace().collect();
    let query_words: Vec<&&str> = words.iter().filter(|w| !skip_words.contains(w)).collect();

    if query_words.is_empty() {
        return None;
    }

    Some(query_words.iter().map(|w| **w).collect::<Vec<&str>>().join(" "))
}

fn is_question(input: &str) -> bool {
    let words: Vec<&str> = input.split_whitespace().collect();

    QUESTION_STARTERS.iter().any(|starter| {
        starter.iter().all(|kw| words.contains(kw))
            && words.iter().position(|w| w == &starter[0]).unwrap_or(usize::MAX) < 3
    })
}

impl Skill for SearchSkill {
    fn id(&self) -> &str {
        "search"
    }

    fn description(&self) -> &str {
        "Searches the web. Use when the user asks to search for something."
    }

    fn specificity(&self) -> Specificity {
        Specificity::Low
    }

    fn score(&self, input: &str, _ctx: &SkillContext) -> f32 {
        let words: Vec<&str> = input.split_whitespace().collect();

        for phrase in TRIGGER_PHRASES {
            let matched = phrase
                .iter()
                .filter(|kw| words.contains(kw))
                .count();

            if matched == phrase.len() {
                if extract_query_explicit(input).is_some() {
                    return 0.90;
                }
                return 0.4;
            }
        }

        if is_question(input) && words.len() >= 4 {
            return 0.85;
        }

        0.0
    }

    fn execute(&self, input: &str, _ctx: &SkillContext) -> Response {
        let words: Vec<&str> = input.split_whitespace().collect();
        let has_trigger = TRIGGER_PHRASES.iter().any(|phrase| {
            phrase.iter().all(|kw| words.contains(kw))
        });

        let query = if has_trigger {
            extract_query_explicit(input).unwrap_or_else(|| input.to_string())
        } else {
            input.to_string()
        };

        Response::Action(serde_json::json!({
            "action": "search",
            "query": query,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> SkillContext {
        SkillContext::default()
    }

    // Scoring: trigger + query = 0.90, trigger alone = 0.4,
    //          question (>=4 words) = 0.85, nothing = 0.0

    #[test]
    fn score_explicit_trigger_with_query() {
        let skill = SearchSkill::new();
        assert_eq!(skill.score("search for cats", &ctx()), 0.90);
        assert_eq!(skill.score("google rust programming", &ctx()), 0.90);
        assert_eq!(skill.score("find nearby restaurants", &ctx()), 0.90);
    }

    #[test]
    fn score_look_up_trigger() {
        let skill = SearchSkill::new();
        // "look up" is a multi-word trigger
        assert_eq!(skill.score("look up the weather in london", &ctx()), 0.90);
    }

    #[test]
    fn score_trigger_without_query_content() {
        let skill = SearchSkill::new();
        // "search for" — after stripping skip_words, nothing remains
        assert_eq!(skill.score("search for", &ctx()), 0.4);
    }

    #[test]
    fn score_question_patterns() {
        let skill = SearchSkill::new();
        assert_eq!(skill.score("where can i get pizza in malta", &ctx()), 0.85);
        assert_eq!(skill.score("how do i cook pasta", &ctx()), 0.85);
        assert_eq!(skill.score("who is the president of france", &ctx()), 0.85);
        assert_eq!(skill.score("why is the sky blue", &ctx()), 0.85);
    }

    #[test]
    fn score_short_question_rejected() {
        let skill = SearchSkill::new();
        // "who is bob" = 3 words, below the 4-word minimum for questions
        assert_eq!(skill.score("who is bob", &ctx()), 0.0);
    }

    #[test]
    fn score_zero_on_unrelated() {
        let skill = SearchSkill::new();
        assert_eq!(skill.score("hello there", &ctx()), 0.0);
        assert_eq!(skill.score("open spotify", &ctx()), 0.0);
    }

    // --- is_question ---

    #[test]
    fn is_question_detects_starters() {
        assert!(is_question("where can i find food"));
        assert!(is_question("how do i reset my password"));
        assert!(is_question("who is that person"));
        assert!(is_question("why does this happen"));
    }

    #[test]
    fn is_question_rejects_non_questions() {
        assert!(!is_question("hello there"));
        assert!(!is_question("open spotify"));
    }

    #[test]
    fn is_question_starter_must_be_near_beginning() {
        // "i wonder where can i find food" — "where" is at index 2, still < 3
        assert!(is_question("i wonder where can i find food"));
        // but at index 3 or beyond it should not match
        assert!(!is_question("tell me please now where can i find food"));
    }

    // --- extract_query_explicit ---

    #[test]
    fn extract_strips_trigger_and_skip_words() {
        assert_eq!(
            extract_query_explicit("search for best rust crates"),
            Some("best rust crates".to_string())
        );
    }

    #[test]
    fn extract_strips_all_skip_words() {
        assert_eq!(
            extract_query_explicit("can you please search for cats"),
            Some("cats".to_string())
        );
    }

    #[test]
    fn extract_returns_none_when_only_skip_words() {
        assert_eq!(extract_query_explicit("search for"), None);
    }

    // --- execute ---

    #[test]
    fn execute_explicit_trigger_strips_skip_words() {
        let skill = SearchSkill::new();
        match skill.execute("search for best rust crates", &ctx()) {
            Response::Action(v) => {
                assert_eq!(v["action"], "search");
                assert_eq!(v["query"], "best rust crates");
            }
            other => panic!("expected Action, got {other:?}"),
        }
    }

    #[test]
    fn execute_question_preserves_full_input() {
        let skill = SearchSkill::new();
        match skill.execute("where can i get pizza in malta", &ctx()) {
            Response::Action(v) => {
                assert_eq!(v["action"], "search");
                assert_eq!(v["query"], "where can i get pizza in malta");
            }
            other => panic!("expected Action, got {other:?}"),
        }
    }

    #[test]
    fn execute_google_strips_trigger() {
        let skill = SearchSkill::new();
        match skill.execute("google cats", &ctx()) {
            Response::Action(v) => assert_eq!(v["query"], "cats"),
            other => panic!("expected Action, got {other:?}"),
        }
    }

    #[test]
    fn specificity_is_low() {
        assert_eq!(SearchSkill::new().specificity(), Specificity::Low);
    }
}
