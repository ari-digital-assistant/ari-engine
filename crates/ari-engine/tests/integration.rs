use ari_core::Response;
use ari_engine::Engine;
use ari_skills::{
    CalculatorSkill, CurrentTimeSkill, DateSkill, GreetingSkill, OpenSkill, SearchSkill,
};

fn full_engine() -> Engine {
    let mut engine = Engine::new();
    engine.register_skill(Box::new(CurrentTimeSkill::new()));
    engine.register_skill(Box::new(DateSkill::new()));
    engine.register_skill(Box::new(CalculatorSkill::new()));
    engine.register_skill(Box::new(GreetingSkill::new()));
    engine.register_skill(Box::new(OpenSkill::new()));
    engine.register_skill(Box::new(SearchSkill::new()));
    engine
}

fn text(resp: &Response) -> &str {
    match resp {
        Response::Text(s) => s,
        other => panic!("expected Text, got {other:?}"),
    }
}

fn action(resp: &Response) -> &serde_json::Value {
    match resp {
        Response::Action(v) => v,
        other => panic!("expected Action, got {other:?}"),
    }
}

// --- Time skill wins over others ---

#[test]
fn time_query_routes_to_time_skill() {
    let engine = full_engine();
    let (_, trace) = engine.process_input_traced("what time is it");
    let trace = trace.unwrap();
    assert_eq!(trace.winner.as_deref(), Some("current_time"));
}

#[test]
fn time_response_format() {
    let engine = full_engine();
    let resp = engine.process_input("tell me the time");
    let s = text(&resp);
    assert!(s.starts_with("It's "));
    assert!(s.ends_with('.'));
}

// --- Date skill wins, doesn't collide with time ---

#[test]
fn date_query_routes_to_date_skill() {
    let engine = full_engine();
    let (_, trace) = engine.process_input_traced("what day is it");
    let trace = trace.unwrap();
    assert_eq!(trace.winner.as_deref(), Some("current_date"));
}

#[test]
fn time_query_does_not_trigger_date() {
    let engine = full_engine();
    let (_, trace) = engine.process_input_traced("what time is it");
    let trace = trace.unwrap();
    let date_score = trace.scores.iter().find(|s| s.skill_id == "current_date").unwrap();
    assert_eq!(date_score.score, 0.0);
}

// --- Calculator wins for math, not for other queries ---

#[test]
fn calculator_handles_bare_expression() {
    let engine = full_engine();
    let resp = engine.process_input("2 + 2");
    assert_eq!(text(&resp), "4");
}

#[test]
fn calculator_handles_natural_language_math() {
    let engine = full_engine();
    let resp = engine.process_input("what is 10 times 5");
    assert_eq!(text(&resp), "50");
}

#[test]
fn calculator_handles_word_numbers() {
    let engine = full_engine();
    // Input gets normalised: "two plus two" → "2 plus 2"
    let resp = engine.process_input("two plus two");
    assert_eq!(text(&resp), "4");
}

#[test]
fn calculator_does_not_steal_greeting() {
    let engine = full_engine();
    let (_, trace) = engine.process_input_traced("hello");
    let trace = trace.unwrap();
    let calc_score = trace.scores.iter().find(|s| s.skill_id == "calculator").unwrap();
    assert_eq!(calc_score.score, 0.0);
}

// --- Greeting routes correctly ---

#[test]
fn greeting_routes_hello() {
    let engine = full_engine();
    let (_, trace) = engine.process_input_traced("hello");
    let trace = trace.unwrap();
    assert_eq!(trace.winner.as_deref(), Some("greeting"));
}

#[test]
fn greeting_responds_to_how_are_you() {
    let engine = full_engine();
    let resp = engine.process_input("how are you");
    assert_eq!(text(&resp), "I'm doing great, thanks for asking! How can I help you?");
}

// --- Open returns Action ---

#[test]
fn open_returns_action_json() {
    let engine = full_engine();
    let resp = engine.process_input("open spotify");
    let v = action(&resp);
    assert_eq!(v["v"], 1);
    assert_eq!(v["launch_app"], "spotify");
}

#[test]
fn open_does_not_steal_unrelated() {
    let engine = full_engine();
    let (_, trace) = engine.process_input_traced("what time is it");
    let trace = trace.unwrap();
    let open_score = trace.scores.iter().find(|s| s.skill_id == "open").unwrap();
    assert_eq!(open_score.score, 0.0);
}

// --- Search handles questions ---

#[test]
fn search_handles_natural_question() {
    let engine = full_engine();
    let resp = engine.process_input("where can i get pizza in malta");
    let v = action(&resp);
    assert_eq!(v["v"], 1);
    assert_eq!(v["search"], "where can i get pizza in malta");
}

#[test]
fn search_explicit_trigger() {
    let engine = full_engine();
    let resp = engine.process_input("search for rust tutorials");
    let v = action(&resp);
    assert_eq!(v["v"], 1);
    assert_eq!(v["search"], "rust tutorials");
}

// --- Fallback ---

#[test]
fn gibberish_returns_fallback() {
    let engine = full_engine();
    let resp = engine.process_input("asdfghjkl qwerty");
    assert_eq!(text(&resp), "Sorry, I didn't understand that.");
}

#[test]
fn empty_returns_fallback() {
    let engine = full_engine();
    let resp = engine.process_input("");
    assert_eq!(text(&resp), "Sorry, I didn't understand that.");
}

// --- Skills don't steal from each other ---

#[test]
fn calculator_beats_search_for_math() {
    let engine = full_engine();
    let (_, trace) = engine.process_input_traced("what is 5 + 3");
    let trace = trace.unwrap();
    // Calculator is High specificity and should win over Search (Low)
    assert_eq!(trace.winner.as_deref(), Some("calculator"));
}

#[test]
fn contraction_normalisation_works_end_to_end() {
    let engine = full_engine();
    // "what's the time" gets normalised to "what is the time"
    let (_, trace) = engine.process_input_traced("what's the time");
    let trace = trace.unwrap();
    assert_eq!(trace.normalized_input, "what is the time");
    assert_eq!(trace.winner.as_deref(), Some("current_time"));
}
