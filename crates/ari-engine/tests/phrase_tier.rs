use ari_core::{normalize_input, normalize_phrase, Skill};
use ari_engine::Engine;
use ari_skills::{
    CalculatorSkill, CurrentTimeSkill, DateSkill, GreetingSkill, OpenSkill, SearchSkill,
};

fn builtins() -> Vec<Box<dyn Skill>> {
    vec![
        Box::new(CurrentTimeSkill::new()),
        Box::new(DateSkill::new()),
        Box::new(GreetingSkill::new()),
        Box::new(CalculatorSkill::new()),
        Box::new(OpenSkill::new()),
        Box::new(SearchSkill::new()),
    ]
}

/// The built-in phrase banks are static, so they are stored already
/// normalised — nothing normalises them at load time the way
/// `PatternScorer` does for a manifest's phrases. An un-normalised entry
/// is not an error anywhere, it just silently never matches, so this is
/// the only thing standing between a typo'd contraction and dead corpus.
#[test]
fn builtin_phrases_are_stored_normalised() {
    let mut dead = Vec::new();
    for skill in builtins() {
        if !skill.router_eligible() {
            continue;
        }
        for locale in ["en", "it"] {
            for e in skill.example_utterances_for(locale) {
                let normalised = normalize_phrase(e.text, locale);
                if normalised != e.text {
                    dead.push(format!(
                        "  [{locale}] {}: {:?} would never match (normalises to {:?})",
                        skill.id(),
                        e.text,
                        normalised
                    ));
                }
            }
        }
    }
    assert!(dead.is_empty(), "unnormalised phrases:\n{}", dead.join("\n"));
}

#[test]
fn every_builtin_phrase_matches_its_own_text() {
    for skill in builtins() {
        if !skill.router_eligible() {
            continue;
        }
        for locale in ["en", "it"] {
            for e in skill.example_utterances_for(locale) {
                if e.text.contains('{') {
                    continue;
                }
                assert!(
                    skill.phrase_score(&normalize_input(e.text, locale), locale) >= e.weight,
                    "[{locale}] {}: {:?} does not match itself",
                    skill.id(),
                    e.text
                );
            }
        }
    }
}

#[test]
fn phrase_tier_catches_what_the_keyword_tier_misses() {
    let mut e = Engine::new();
    e.register_skill(Box::new(CurrentTimeSkill::new()));
    e.register_skill(Box::new(GreetingSkill::new()));

    let oblique = "the clock on the wall stopped can you tell me the time";
    assert_eq!(e.keyword_decision(oblique), None, "no keyword pattern claims it");

    let winner = e.process_input_traced(oblique).1.and_then(|t| t.winner);
    assert_eq!(winner.as_deref(), Some("phrase:current_time"));
}

#[test]
fn keyword_tier_still_outranks_a_phrase_match() {
    let mut e = Engine::new();
    e.register_skill(Box::new(CurrentTimeSkill::new()));
    e.register_skill(Box::new(GreetingSkill::new()));

    // Claimed by current_time's own keyword scorer, so it must win there and
    // never reach the phrase tier — a `{slot}` phrase is the looser signal.
    let winner = e.process_input_traced("what time is it").1.and_then(|t| t.winner);
    assert_eq!(winner.as_deref(), Some("current_time"));
}
