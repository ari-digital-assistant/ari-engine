use ari_core::{ExampleUtterance, Response, Skill, SkillContext, Specificity};

const TRIGGER_PHRASES: &[&[&str]] = &[
    // English
    &["search", "for"],
    &["look", "up"],
    &["google"],
    &["search"],
    &["find"],
    // Italian — cerca (search), trova (find), cercare (to search)
    &["cerca"],
    &["cercare"],
    &["trova"],
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
    let skip_words = [
        // English — command/intent words and polite filler only (never articles)
        "search", "for", "look", "up", "google", "find", "please",
        "can", "you", "me",
        // Italian — command verbs and polite filler only; deliberately NOT
        // articles/prepositions (la/il/lo/le/di/su/per), which are load-bearing
        // parts of proper nouns and titles.
        "cerca", "cercare", "trova", "favore", "mi", "puoi", "dimmi",
    ];

    let words: Vec<&str> = input.split_whitespace().collect();
    let query_words: Vec<&&str> = words.iter().filter(|w| !skip_words.contains(w)).collect();

    if query_words.is_empty() {
        return None;
    }

    Some(query_words.iter().map(|w| **w).collect::<Vec<&str>>().join(" "))
}

impl Skill for SearchSkill {
    fn id(&self) -> &str {
        "search"
    }

    fn description(&self) -> &str {
        "Searches the web. Use only when the user explicitly asks to search, look up, google, or find something on the web — phrases like 'search for X', 'look up Y', 'google Z', 'find information about W', 'what does the internet say about X'. Plain questions the assistant can answer ('what is X', 'tell me about Y') do NOT belong here. The query parameter captures what to search for."
    }

    fn router_eligible(&self) -> bool {
        false
    }

    fn specificity(&self) -> Specificity {
        Specificity::Low
    }

    fn parameters_schema(&self) -> &str {
        r#"{"type": "object", "properties": {"query": {"type": "string", "description": "The search query."}}, "required": ["query"]}"#
    }

    fn example_utterances(&self) -> &[ExampleUtterance] {
        &[
            ExampleUtterance { text: "search for python tutorials", args: r#"{"query": "python tutorials"}"# },
            ExampleUtterance { text: "look up the weather in London", args: r#"{"query": "weather in London"}"# },
            ExampleUtterance { text: "google how to make pasta", args: r#"{"query": "how to make pasta"}"# },
            ExampleUtterance { text: "find information about black holes", args: r#"{"query": "black holes"}"# },
            ExampleUtterance { text: "search for nearby restaurants", args: r#"{"query": "nearby restaurants"}"# },
            ExampleUtterance { text: "look up who won the world cup", args: r#"{"query": "who won the world cup"}"# },
            ExampleUtterance { text: "find me a recipe for brownies", args: r#"{"query": "recipe for brownies"}"# },
            ExampleUtterance { text: "search how tall is mount everest", args: r#"{"query": "how tall is mount everest"}"# },
            ExampleUtterance { text: "google the latest news", args: r#"{"query": "latest news"}"# },
            ExampleUtterance { text: "look up train times to London", args: r#"{"query": "train times to London"}"# },
            ExampleUtterance { text: "find out about the Mars rover", args: r#"{"query": "Mars rover"}"# },
            ExampleUtterance { text: "search for cheap flights to Tokyo", args: r#"{"query": "cheap flights to Tokyo"}"# },
            ExampleUtterance { text: "I need to look something up about batteries", args: r#"{"query": "batteries"}"# },
            ExampleUtterance { text: "can you google that for me", args: r#"{"query": "that"}"# },
            ExampleUtterance { text: "search the web for Ari digital assistant", args: r#"{"query": "Ari digital assistant"}"# },
            ExampleUtterance { text: "find directions to the airport", args: r#"{"query": "directions to the airport"}"# },
            ExampleUtterance { text: "look up symptoms of a cold", args: r#"{"query": "symptoms of a cold"}"# },
            ExampleUtterance { text: "google how to change a tyre", args: r#"{"query": "how to change a tyre"}"# },
            ExampleUtterance { text: "search for best programming languages 2026", args: r#"{"query": "best programming languages 2026"}"# },
            ExampleUtterance { text: "find reviews for the pixel phone", args: r#"{"query": "reviews for the pixel phone"}"# },
            ExampleUtterance { text: "look up the population of Malta", args: r#"{"query": "population of Malta"}"# },
            ExampleUtterance { text: "search for hiking trails near me", args: r#"{"query": "hiking trails near me"}"# },
            ExampleUtterance { text: "google what time the shops close", args: r#"{"query": "what time the shops close"}"# },
            ExampleUtterance { text: "find out when the next bus is", args: r#"{"query": "when the next bus is"}"# },
            ExampleUtterance { text: "search for the meaning of serendipity", args: r#"{"query": "meaning of serendipity"}"# },
            ExampleUtterance { text: "look up how to tie a tie", args: r#"{"query": "how to tie a tie"}"# },
            ExampleUtterance { text: "find me a good pizza place", args: r#"{"query": "good pizza place"}"# },
            ExampleUtterance { text: "google who invented the telephone", args: r#"{"query": "who invented the telephone"}"# },
            ExampleUtterance { text: "search for free online courses", args: r#"{"query": "free online courses"}"# },
            ExampleUtterance { text: "look up currency exchange rates", args: r#"{"query": "currency exchange rates"}"# },
            // Explicit "search the web" intent phrased without a verb — the
            // user still names the web/internet, so it's a search, not a
            // question for the assistant.
            ExampleUtterance { text: "what does the internet say about async runtimes in rust", args: r#"{"query": "async runtimes in rust"}"# },
            ExampleUtterance { text: "what's online about climate change", args: r#"{"query": "climate change"}"# },
            ExampleUtterance { text: "what does the web say about kubernetes", args: r#"{"query": "kubernetes"}"# },
            // Italian
            ExampleUtterance { text: "cerca ristoranti vicini", args: r#"{"query": "ristoranti vicini"}"# },
            ExampleUtterance { text: "trova una ricetta per la pizza", args: r#"{"query": "ricetta per la pizza"}"# },
            ExampleUtterance { text: "cerca su internet i voli per tokyo", args: r#"{"query": "voli per tokyo"}"# },
        ]
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

        // `speak` omitted deliberately — the frontend produces the
        // platform-appropriate "Searching for X." phrase itself.
        Response::Action(serde_json::json!({
            "v": 1,
            "search": query,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> SkillContext {
        SkillContext::default()
    }

    // Scoring: trigger + query = 0.90, trigger alone = 0.4, nothing = 0.0.
    // Search fires ONLY on explicit triggers — bare questions go to the
    // assistant, not here (search is also kept out of the router catalogue).

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
    fn score_bare_questions_do_not_match() {
        // Questions without an explicit search verb belong to the assistant,
        // not search. The literal bug report is the UAE one.
        let skill = SearchSkill::new();
        assert_eq!(
            skill.score("what is the capital city of the united arab emirates", &ctx()),
            0.0
        );
        assert_eq!(skill.score("where can i get pizza in malta", &ctx()), 0.0);
        assert_eq!(skill.score("how do i cook pasta", &ctx()), 0.0);
        assert_eq!(skill.score("who is the president of france", &ctx()), 0.0);
        assert_eq!(skill.score("why is the sky blue", &ctx()), 0.0);
    }

    #[test]
    fn score_zero_on_unrelated() {
        let skill = SearchSkill::new();
        assert_eq!(skill.score("hello there", &ctx()), 0.0);
        assert_eq!(skill.score("open spotify", &ctx()), 0.0);
    }

    #[test]
    fn not_router_eligible() {
        // Search must never be offered to the semantic routers, or the LLM
        // would claim general questions that belong to the assistant.
        assert!(!SearchSkill::new().router_eligible());
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
                assert_eq!(v["v"], 1);
                assert_eq!(v["search"], "best rust crates");
                assert!(v.get("speak").is_none());
            }
            other => panic!("expected Action, got {other:?}"),
        }
    }

    #[test]
    fn execute_question_preserves_full_input() {
        let skill = SearchSkill::new();
        match skill.execute("where can i get pizza in malta", &ctx()) {
            Response::Action(v) => {
                assert_eq!(v["v"], 1);
                assert_eq!(v["search"], "where can i get pizza in malta");
            }
            other => panic!("expected Action, got {other:?}"),
        }
    }

    #[test]
    fn execute_google_strips_trigger() {
        let skill = SearchSkill::new();
        match skill.execute("google cats", &ctx()) {
            Response::Action(v) => assert_eq!(v["search"], "cats"),
            other => panic!("expected Action, got {other:?}"),
        }
    }

    #[test]
    fn specificity_is_low() {
        assert_eq!(SearchSkill::new().specificity(), Specificity::Low);
    }

    // --- Italian ---

    #[test]
    fn score_italian_explicit_trigger_with_query() {
        let skill = SearchSkill::new();
        // "cerca ristoranti vicini" = search nearby restaurants
        assert_eq!(skill.score("cerca ristoranti vicini", &ctx()), 0.90);
        // "trova una pizzeria" = find a pizzeria
        assert_eq!(skill.score("trova una pizzeria", &ctx()), 0.90);
    }

    #[test]
    fn score_italian_bare_questions_do_not_match() {
        let skill = SearchSkill::new();
        // "come si cuoce la pasta" = how do you cook pasta — a question, so
        // it goes to the assistant, not search.
        assert_eq!(skill.score("come si cuoce la pasta", &ctx()), 0.0);
        // "dove posso trovare una pizza" = where can I find a pizza — no exact
        // trigger word ("trovare" != "trova"), so it's a question too.
        assert_eq!(skill.score("dove posso trovare una pizza", &ctx()), 0.0);
        // The explicit "trova" verb still triggers search.
        assert_eq!(skill.score("trova una pizzeria", &ctx()), 0.90);
    }

    #[test]
    fn extract_strips_italian_trigger_and_filler_keeps_articles() {
        // "puoi cercare le pizzerie" = "can you search the pizzerias".
        // Strips the filler "puoi" and trigger "cercare"; KEEPS the
        // article "le" — articles are part of proper nouns/titles.
        assert_eq!(
            extract_query_explicit("puoi cercare le pizzerie"),
            Some("le pizzerie".to_string())
        );
    }

    #[test]
    fn execute_italian_trigger_strips_skip_words() {
        let skill = SearchSkill::new();
        match skill.execute("trova ristoranti vicini", &ctx()) {
            Response::Action(v) => {
                assert_eq!(v["v"], 1);
                assert_eq!(v["search"], "ristoranti vicini");
            }
            other => panic!("expected Action, got {other:?}"),
        }
    }
}
