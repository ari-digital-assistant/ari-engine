use ari_core::{AppEntry, Response, Skill, SkillContext, Specificity};
use ari_engine::Engine;
use ari_skills::OpenSkill;

// NOTE: the task brief for this test named the entry point
// `Engine::route_decision`, describing it as returning "the KEYWORD-tier
// winner's skill id". That description matches `Engine::keyword_decision`
// (see its doc comment: "the skill the ranking rounds would execute" /
// "shares the exact ranking logic process_input_traced uses"), not
// `route_decision`, which is documented as "Router-only... runs neither the
// keyword scorer nor the assistant" and short-circuits to `None` whenever no
// FunctionGemma router is loaded (always true for a bare `Engine::new()` in
// a unit test). Verified: with the brief's literal `route_decision` call,
// both tests below fail with `None` for that reason alone, regardless of
// Tasks 3-5's behaviour. Swapping to `keyword_decision` — the method whose
// documented behaviour actually matches the brief's description — makes both
// assertions pass, confirming Tasks 3-5's keyword-tier disambiguation is
// correct. No production code was touched to reach this result.

/// Stand-in for Home Assistant's keyword tier: medium specificity, scores 0.8
/// when the utterance contains an open/close/lock/unlock verb — its real
/// manifest pattern weight.
struct SmartHomeStub;
impl Skill for SmartHomeStub {
    fn id(&self) -> &str { "smarthome" }
    fn specificity(&self) -> Specificity { Specificity::Medium }
    fn score(&self, input: &str, _: &SkillContext) -> f32 {
        let hit = input
            .split_whitespace()
            .any(|w| matches!(w, "open" | "close" | "lock" | "unlock"));
        if hit { 0.8 } else { 0.0 }
    }
    fn execute(&self, _: &str, _: &SkillContext) -> Response {
        Response::Action(serde_json::json!({ "v": 1, "speak": "smart-home" }))
    }
}

fn engine() -> Engine {
    let mut e = Engine::new();
    e.register_skill(Box::new(OpenSkill::new()));
    e.register_skill(Box::new(SmartHomeStub));
    e.set_installed_apps(vec![AppEntry {
        label: "Spotify".to_string(),
        package: "com.spotify.music".to_string(),
    }]);
    e
}

#[test]
fn blinds_route_to_smarthome_not_open() {
    // "the main bedroom blinds" matches no installed app → open scores 0.0 →
    // SmartHomeStub (0.8, medium) wins the keyword tier.
    assert_eq!(
        engine().keyword_decision("open the main bedroom blinds").as_deref(),
        Some("smarthome"),
    );
}

#[test]
fn open_installed_app_routes_to_open() {
    // Spotify is installed → open scores 0.9 and beats SmartHomeStub's 0.8.
    assert_eq!(
        engine().keyword_decision("open spotify").as_deref(),
        Some("open"),
    );
}
