use ari_core::{ExampleUtterance, Response, Skill, SkillContext, Specificity};

// A search is a command at the start of what was said: "search for X",
// "look up X", "can you google X". The word "find" or "search" in the
// middle of a question ("where do you find them in Europe?") is a question
// for the assistant, which also has the conversation to say what "them"
// is (ari-android#43). Phrases are matched on whole words, in order.

/// Politeness that may come before the command.
const LEAD_INS: &[&[&str]] = &[
    &["please"],
    &["can", "you"],
    &["could", "you"],
    &["would", "you"],
    &["will", "you"],
    // Italian: puoi/potresti (can/could you), per favore (please)
    &["puoi"],
    &["potresti"],
    &["per", "favore"],
];

/// The commands, longer before shorter so "search for" beats "search".
const COMMANDS: &[&[&str]] = &[
    &["search", "for"],
    &["search"],
    &["look", "up"],
    &["google"],
    &["find", "out", "about"],
    &["find", "out"],
    &["find", "me"],
    &["find"],
    // Italian: cerca/cercare (search), trova (find)
    &["cerca"],
    &["cercare"],
    &["trova"],
];

/// Between the command and the query: "search the web for X".
const BRIDGES: &[&[&str]] = &[
    &["on", "the", "web"],
    &["on", "the", "internet"],
    &["the", "web"],
    &["the", "internet"],
    &["online"],
    &["for"],
    // Italian: su internet (on the internet), in rete (online)
    &["su", "internet"],
    &["in", "rete"],
];

/// After the query: "google that for me", "search for cats please".
const TAILS: &[&[&str]] = &[&["please"], &["for", "me"], &["per", "favore"]];

fn strip_leading<'a>(mut words: &'a [&'a str], phrases: &[&[&str]]) -> &'a [&'a str] {
    'again: loop {
        for phrase in phrases {
            if words.starts_with(phrase) {
                words = &words[phrase.len()..];
                continue 'again;
            }
        }
        return words;
    }
}

fn strip_trailing<'a>(mut words: &'a [&'a str], phrases: &[&[&str]]) -> &'a [&'a str] {
    'again: loop {
        for phrase in phrases {
            if words.ends_with(phrase) {
                words = &words[..words.len() - phrase.len()];
                continue 'again;
            }
        }
        return words;
    }
}

/// The query of an explicit search command: `None` when `input` isn't one,
/// `Some("")` for a bare command such as "search for". Only the command and
/// its politeness are removed; the query keeps every word, articles and
/// prepositions included, since they're part of titles and names.
fn command_query(input: &str) -> Option<String> {
    let words: Vec<&str> = input.split_whitespace().collect();
    let rest = strip_leading(&words, LEAD_INS);
    let command = COMMANDS.iter().find(|c| rest.starts_with(c))?;
    let rest = strip_leading(&rest[command.len()..], BRIDGES);
    Some(strip_trailing(rest, TAILS).join(" "))
}

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
            ExampleUtterance { text: "search for python tutorials", args: r#"{"query": "python tutorials"}"#, weight: 1.0 },
            ExampleUtterance { text: "look up the weather in London", args: r#"{"query": "weather in London"}"#, weight: 1.0 },
            ExampleUtterance { text: "google how to make pasta", args: r#"{"query": "how to make pasta"}"#, weight: 1.0 },
            ExampleUtterance { text: "find information about black holes", args: r#"{"query": "black holes"}"#, weight: 1.0 },
            ExampleUtterance { text: "search for nearby restaurants", args: r#"{"query": "nearby restaurants"}"#, weight: 1.0 },
            ExampleUtterance { text: "look up who won the world cup", args: r#"{"query": "who won the world cup"}"#, weight: 1.0 },
            ExampleUtterance { text: "find me a recipe for brownies", args: r#"{"query": "recipe for brownies"}"#, weight: 1.0 },
            ExampleUtterance { text: "search how tall is mount everest", args: r#"{"query": "how tall is mount everest"}"#, weight: 1.0 },
            ExampleUtterance { text: "google the latest news", args: r#"{"query": "latest news"}"#, weight: 1.0 },
            ExampleUtterance { text: "look up train times to London", args: r#"{"query": "train times to London"}"#, weight: 1.0 },
            ExampleUtterance { text: "find out about the Mars rover", args: r#"{"query": "Mars rover"}"#, weight: 1.0 },
            ExampleUtterance { text: "search for cheap flights to Tokyo", args: r#"{"query": "cheap flights to Tokyo"}"#, weight: 1.0 },
            ExampleUtterance { text: "I need to look something up about batteries", args: r#"{"query": "batteries"}"#, weight: 1.0 },
            ExampleUtterance { text: "can you google that for me", args: r#"{"query": "that"}"#, weight: 1.0 },
            ExampleUtterance { text: "search the web for Ari digital assistant", args: r#"{"query": "Ari digital assistant"}"#, weight: 1.0 },
            ExampleUtterance { text: "find directions to the airport", args: r#"{"query": "directions to the airport"}"#, weight: 1.0 },
            ExampleUtterance { text: "look up symptoms of a cold", args: r#"{"query": "symptoms of a cold"}"#, weight: 1.0 },
            ExampleUtterance { text: "google how to change a tyre", args: r#"{"query": "how to change a tyre"}"#, weight: 1.0 },
            ExampleUtterance { text: "search for best programming languages 2026", args: r#"{"query": "best programming languages 2026"}"#, weight: 1.0 },
            ExampleUtterance { text: "find reviews for the pixel phone", args: r#"{"query": "reviews for the pixel phone"}"#, weight: 1.0 },
            ExampleUtterance { text: "look up the population of Malta", args: r#"{"query": "population of Malta"}"#, weight: 1.0 },
            ExampleUtterance { text: "search for hiking trails near me", args: r#"{"query": "hiking trails near me"}"#, weight: 1.0 },
            ExampleUtterance { text: "google what time the shops close", args: r#"{"query": "what time the shops close"}"#, weight: 1.0 },
            ExampleUtterance { text: "find out when the next bus is", args: r#"{"query": "when the next bus is"}"#, weight: 1.0 },
            ExampleUtterance { text: "search for the meaning of serendipity", args: r#"{"query": "meaning of serendipity"}"#, weight: 1.0 },
            ExampleUtterance { text: "look up how to tie a tie", args: r#"{"query": "how to tie a tie"}"#, weight: 1.0 },
            ExampleUtterance { text: "find me a good pizza place", args: r#"{"query": "good pizza place"}"#, weight: 1.0 },
            ExampleUtterance { text: "google who invented the telephone", args: r#"{"query": "who invented the telephone"}"#, weight: 1.0 },
            ExampleUtterance { text: "search for free online courses", args: r#"{"query": "free online courses"}"#, weight: 1.0 },
            ExampleUtterance { text: "look up currency exchange rates", args: r#"{"query": "currency exchange rates"}"#, weight: 1.0 },
            // Explicit "search the web" intent phrased without a verb — the
            // user still names the web/internet, so it's a search, not a
            // question for the assistant.
            ExampleUtterance { text: "what does the internet say about async runtimes in rust", args: r#"{"query": "async runtimes in rust"}"#, weight: 1.0 },
            ExampleUtterance { text: "what's online about climate change", args: r#"{"query": "climate change"}"#, weight: 1.0 },
            ExampleUtterance { text: "what does the web say about kubernetes", args: r#"{"query": "kubernetes"}"#, weight: 1.0 },
            // Italian
            ExampleUtterance { text: "cerca ristoranti vicini", args: r#"{"query": "ristoranti vicini"}"#, weight: 1.0 },
            ExampleUtterance { text: "trova una ricetta per la pizza", args: r#"{"query": "ricetta per la pizza"}"#, weight: 1.0 },
            ExampleUtterance { text: "cerca su internet i voli per tokyo", args: r#"{"query": "voli per tokyo"}"#, weight: 1.0 },
        ]
    }

    fn score(&self, input: &str, _ctx: &SkillContext) -> f32 {
        match command_query(input) {
            Some(query) if !query.is_empty() => 0.90,
            Some(_) => 0.4,
            None => 0.0,
        }
    }

    fn execute(&self, input: &str, _ctx: &SkillContext) -> Response {
        // Reached without a command only when something else routed here;
        // then the whole input is the query.
        let query = match command_query(input) {
            Some(query) if !query.is_empty() => query,
            _ => input.to_string(),
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

    // Scoring: command + query = 0.90, command alone = 0.4, nothing = 0.0.
    // Search fires ONLY on an explicit command at the start — bare questions go to the
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
    fn score_question_containing_find_goes_to_the_assistant() {
        // ari-android#43: a follow-up the assistant could answer, sent to
        // the web with half its words missing.
        let skill = SearchSkill::new();
        assert_eq!(skill.score("where do you find them in europe", &ctx()), 0.0);
        assert_eq!(skill.score("where do you find these in europe", &ctx()), 0.0);
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

    // --- command_query ---

    #[test]
    fn query_drops_only_the_command() {
        assert_eq!(command_query("search for best rust crates").as_deref(), Some("best rust crates"));
        // An inner "for" is part of the query.
        assert_eq!(
            command_query("search for things to do in paris for kids").as_deref(),
            Some("things to do in paris for kids")
        );
        assert_eq!(command_query("look up how to tie a tie").as_deref(), Some("how to tie a tie"));
    }

    #[test]
    fn query_drops_politeness_and_bridges() {
        assert_eq!(command_query("can you please search for cats").as_deref(), Some("cats"));
        assert_eq!(
            command_query("search the web for ari digital assistant").as_deref(),
            Some("ari digital assistant")
        );
        assert_eq!(command_query("find me a recipe for brownies").as_deref(), Some("a recipe for brownies"));
        assert_eq!(command_query("find out about the mars rover").as_deref(), Some("the mars rover"));
        assert_eq!(command_query("google that for me").as_deref(), Some("that"));
    }

    #[test]
    fn bare_command_has_an_empty_query() {
        assert_eq!(command_query("search for").as_deref(), Some(""));
    }

    #[test]
    fn a_command_word_mid_sentence_is_not_a_command() {
        // ari-android#43: searched for "where do them in europe".
        assert_eq!(command_query("where do you find them in europe"), None);
        assert_eq!(command_query("where can i find a good pizza"), None);
        assert_eq!(command_query("i lost my phone can you help me find it"), None);
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
        assert_eq!(command_query("puoi cercare le pizzerie").as_deref(), Some("le pizzerie"));
        assert_eq!(command_query("cerca su internet i voli per tokyo").as_deref(), Some("i voli per tokyo"));
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
