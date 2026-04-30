//! Adapter that turns a parsed declarative `Skillfile` into something that
//! `impl`s the engine's `Skill` trait.
//!
//! Scoring delegates to the shared [`crate::scoring::PatternScorer`], which is
//! also used by the WASM adapter for skills that haven't opted in to a custom
//! `score()` export. Both adapters therefore behave identically for inputs
//! that don't trip the WASM custom-score path.
//!
//! Execution renders the response spec — `Fixed` returns verbatim, `Pick` picks
//! one entry pseudo-randomly, `Template` is currently rendered as-is (capture
//! filling is deferred to a later step). When the manifest also carries an
//! `action`, the response is wrapped in `Response::Action` with both `text` and
//! `action` keys; otherwise it's a plain `Response::Text`.

use crate::localized_manifest::{LocalizedManifestSet, CANONICAL_LOCALE};
use crate::manifest::{AriExtension, Behaviour, ResponseSpec, Skillfile};
use crate::scoring::{PatternScorer, ScorerError};
use ari_core::{Response, Skill, SkillContext, Specificity};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AdapterError {
    #[error("skillfile has no `metadata.ari` extension and cannot be loaded as an Ari skill")]
    NotAnAriSkill,

    #[error("skillfile is a WASM skill; use the WASM adapter instead")]
    NotDeclarative,

    #[error("scorer compile failed: {0}")]
    Scorer(#[from] ScorerError),
}

/// Per-locale slice of a declarative skill: the bits that legitimately
/// differ across `SKILL.{locale}.md` files. Structural fields (id,
/// type, capabilities, behaviour-shape) are validated identical at
/// load time and live on the parent [`DeclarativeSkill`] directly.
#[derive(Debug)]
struct LocalizedDeclarative {
    /// Description used for skill-browser display. (FunctionGemma
    /// routing only ever sees the canonical-English description, so
    /// the per-locale variant matters mainly for UI surfaces.)
    description: String,
    scorer: PatternScorer,
    response: ResponseSpec,
    action: Option<serde_json::Value>,
}

/// A declarative skill, ready to be plugged into the engine. Holds
/// per-locale [`LocalizedDeclarative`] entries; `score()` and
/// `execute()` dispatch on `ctx.locale`, falling back to the
/// canonical English entry when the requested locale isn't shipped.
#[derive(Debug)]
pub struct DeclarativeSkill {
    id: String,
    specificity: Specificity,
    /// Per-locale data, keyed by ISO 639-1 lowercase code. The
    /// canonical ([`CANONICAL_LOCALE`]) entry is guaranteed present;
    /// the loader rejects skills whose `SKILL.en.md` (or legacy
    /// `SKILL.md`) doesn't parse cleanly.
    by_locale: BTreeMap<String, LocalizedDeclarative>,
    /// Monotonic counter used for `response_pick` selection. Avoids pulling in
    /// a random-number crate just for picking one of N strings, while still
    /// giving distinct callers different answers within a session. Shared
    /// across locale variants — they all draw from the same sequence so a
    /// user who flips locales mid-session doesn't see the picker reset.
    pick_counter: AtomicU64,
}

impl DeclarativeSkill {
    /// Build a single-locale skill from a single [`Skillfile`].
    /// Treats the input as the canonical English manifest. Use
    /// [`from_localized`](Self::from_localized) when you have the
    /// full per-locale set.
    pub fn from_skillfile(sf: &Skillfile) -> Result<Self, AdapterError> {
        let ari = sf.ari_extension.as_ref().ok_or(AdapterError::NotAnAriSkill)?;
        let id = ari.id.clone();
        let specificity = ari.specificity.as_core();
        let entry = compile_locale_entry(&sf.description, ari)?;
        let mut by_locale = BTreeMap::new();
        by_locale.insert(CANONICAL_LOCALE.to_string(), entry);
        Ok(DeclarativeSkill {
            id,
            specificity,
            by_locale,
            pick_counter: AtomicU64::new(seed_pick_counter()),
        })
    }

    /// Build a declarative skill from a parsed [`LocalizedManifestSet`].
    /// Each locale variant contributes its own pattern scorer + response
    /// spec; structural identity (id, type, capabilities, behaviour-shape)
    /// is taken from the canonical entry. The set is guaranteed to have a
    /// canonical entry by the localized-manifest parser.
    pub fn from_localized(set: &LocalizedManifestSet) -> Result<Self, AdapterError> {
        let canonical = set.canonical();
        let canonical_ari = canonical
            .ari_extension
            .as_ref()
            .ok_or(AdapterError::NotAnAriSkill)?;
        let id = canonical_ari.id.clone();
        let specificity = canonical_ari.specificity.as_core();

        let mut by_locale: BTreeMap<String, LocalizedDeclarative> = BTreeMap::new();
        for (locale, sf) in &set.manifests {
            // Skip locale variants that don't carry a `metadata.ari`
            // block — they're plain AgentSkills docs that the
            // canonical-set validator already accepted (silent-skip
            // semantics for non-Ari docs).
            let Some(ari) = sf.ari_extension.as_ref() else {
                continue;
            };
            let entry = compile_locale_entry(&sf.description, ari)?;
            by_locale.insert(locale.clone(), entry);
        }

        if !by_locale.contains_key(CANONICAL_LOCALE) {
            // Canonical didn't compile — usually means it wasn't a
            // declarative skill at all, but rather a WASM one. Loader
            // wouldn't have called us in that case, so this is an
            // internal-inconsistency guard.
            return Err(AdapterError::NotDeclarative);
        }

        Ok(DeclarativeSkill {
            id,
            specificity,
            by_locale,
            pick_counter: AtomicU64::new(seed_pick_counter()),
        })
    }

    /// The canonical-English entry. Always present (constructor invariant).
    fn canonical(&self) -> &LocalizedDeclarative {
        self.by_locale
            .get(CANONICAL_LOCALE)
            .expect("canonical entry guaranteed by constructor")
    }

    /// The entry for the given locale, falling back to the canonical
    /// English entry when no locale-specific manifest was shipped.
    /// Best-effort fallback per the multi-language plan — Italian
    /// users invoking a skill that only ships `SKILL.en.md` get
    /// English patterns + English responses, not silent skip.
    fn entry_for_locale(&self, locale: &str) -> &LocalizedDeclarative {
        self.by_locale.get(locale).unwrap_or_else(|| self.canonical())
    }

    /// Test seam: render the response with a caller-supplied pick index. Used
    /// by unit tests to assert specific outputs without depending on internal
    /// counter state.
    #[cfg(test)]
    fn render_with_pick(&self, pick_idx: usize) -> Response {
        build_response(&self.canonical().response, &self.canonical().action, pick_idx)
    }
}

/// Compile the matching/response/action fields of one locale variant
/// into a [`LocalizedDeclarative`]. Shared between the single-Skillfile
/// constructor and the per-locale builder.
fn compile_locale_entry(
    description: &str,
    ari: &AriExtension,
) -> Result<LocalizedDeclarative, AdapterError> {
    let decl = match &ari.behaviour {
        Some(Behaviour::Declarative(d)) => d,
        Some(Behaviour::Wasm(_)) => return Err(AdapterError::NotDeclarative),
        None => return Err(AdapterError::NotDeclarative),
    };
    let matching = ari.matching.as_ref().ok_or(AdapterError::NotDeclarative)?;
    let scorer = PatternScorer::compile(matching)?;
    Ok(LocalizedDeclarative {
        description: description.to_string(),
        scorer,
        response: decl.response.clone(),
        action: decl.action.clone(),
    })
}

/// Seed the pick counter from system time so successive process
/// launches don't all start at index 0. Within a single process the
/// counter still advances monotonically.
fn seed_pick_counter() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

fn build_response(
    response: &ResponseSpec,
    action: &Option<serde_json::Value>,
    pick_idx: usize,
) -> Response {
    let text = match response {
        ResponseSpec::Fixed(s) => s.clone(),
        ResponseSpec::Pick(options) => {
            // Caller is responsible for keeping pick_idx in range; the
            // public path always uses `pick_counter`, which we modulo
            // here. We never index out of bounds because options is
            // guaranteed non-empty by the manifest validator.
            options[pick_idx % options.len()].clone()
        }
        ResponseSpec::Template(t) => t.clone(),
    };

    match action {
        None => Response::Text(text),
        Some(action) => {
            // The manifest's `action:` block is a presentation-envelope
            // fragment (see docs/action-responses.md). Overlay `v` and
            // `speak` onto it — those are protocol-level concerns the
            // manifest author doesn't need to repeat. Any `speak` the
            // author did set wins over the `response`-derived text.
            let mut env = action.clone();
            if let Some(obj) = env.as_object_mut() {
                obj.insert("v".into(), serde_json::json!(1));
                obj.entry("speak".to_string())
                    .or_insert_with(|| serde_json::json!(text));
            }
            Response::Action(env)
        }
    }
}

impl Skill for DeclarativeSkill {
    fn id(&self) -> &str {
        &self.id
    }

    fn description(&self) -> &str {
        // FunctionGemma routes on English description; the skill
        // browser uses ctx-aware paths instead. Default to canonical
        // for callers that don't have a context (router training data
        // export, etc.).
        &self.canonical().description
    }

    fn specificity(&self) -> Specificity {
        self.specificity
    }

    fn score(&self, input: &str, ctx: &SkillContext) -> f32 {
        self.entry_for_locale(&ctx.locale).scorer.score(input)
    }

    fn execute(&self, _input: &str, ctx: &SkillContext) -> Response {
        let entry = self.entry_for_locale(&ctx.locale);
        let idx = self.pick_counter.fetch_add(1, Ordering::Relaxed) as usize;
        build_response(&entry.response, &entry.action, idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(src: &str) -> Skillfile {
        Skillfile::parse(src, None).unwrap()
    }

    fn coin_flip() -> DeclarativeSkill {
        let src = r#"---
name: coin-flip
description: Flips a virtual coin and returns heads or tails. Use when the user asks to flip a coin.
metadata:
  ari:
    id: ai.example.coinflip
    version: "0.1.0"
    engine: ">=0.3"
    capabilities: []
    languages: [en]
    specificity: high
    matching:
      patterns:
        - keywords: [flip, coin]
          weight: 0.95
        - keywords: [toss, coin]
          weight: 0.95
    declarative:
      response_pick: ["Heads.", "Tails."]
---
"#;
        DeclarativeSkill::from_skillfile(&parse(src)).unwrap()
    }

    #[test]
    fn id_and_specificity_round_trip() {
        let s = coin_flip();
        assert_eq!(s.id(), "ai.example.coinflip");
        assert_eq!(s.specificity(), Specificity::High);
    }

    #[test]
    fn scores_matching_input_at_pattern_weight() {
        let s = coin_flip();
        let ctx = SkillContext::default();
        assert_eq!(s.score("flip a coin", &ctx), 0.95);
        assert_eq!(s.score("Toss a Coin!", &ctx), 0.95);
        assert_eq!(s.score("please flip the coin for me", &ctx), 0.95);
    }

    #[test]
    fn scores_non_matching_input_at_zero() {
        let s = coin_flip();
        let ctx = SkillContext::default();
        assert_eq!(s.score("what time is it", &ctx), 0.0);
        assert_eq!(s.score("flip the pancakes", &ctx), 0.0); // no "coin"
        assert_eq!(s.score("a coin", &ctx), 0.0); // no "flip"/"toss"
    }

    #[test]
    fn keyword_match_is_whole_word_only() {
        let s = coin_flip();
        let ctx = SkillContext::default();
        // "flipping" should not match "flip"
        assert_eq!(s.score("flipping a coin", &ctx), 0.0);
    }

    #[test]
    fn execute_returns_one_of_the_picks() {
        let s = coin_flip();
        let ctx = SkillContext::default();
        let r = s.execute("flip a coin", &ctx);
        match r {
            Response::Text(t) => assert!(t == "Heads." || t == "Tails."),
            _ => panic!("expected text response"),
        }
    }

    #[test]
    fn execute_picks_deterministically_with_test_seam() {
        let s = coin_flip();
        match s.render_with_pick(0) {
            Response::Text(t) => assert_eq!(t, "Heads."),
            _ => panic!(),
        }
        match s.render_with_pick(1) {
            Response::Text(t) => assert_eq!(t, "Tails."),
            _ => panic!(),
        }
        match s.render_with_pick(2) {
            Response::Text(t) => assert_eq!(t, "Heads."),
            _ => panic!(),
        }
    }

    #[test]
    fn fixed_response_returns_verbatim() {
        let src = r#"---
name: greet
description: Greets the user. Use when the user says hello.
metadata:
  ari:
    id: ai.example.greet
    version: "1"
    engine: ">=0.3"
    matching:
      patterns:
        - keywords: [hello]
    declarative:
      response: "Oh hey."
---
"#;
        let s = DeclarativeSkill::from_skillfile(&parse(src)).unwrap();
        let ctx = SkillContext::default();
        match s.execute("hello", &ctx) {
            Response::Text(t) => assert_eq!(t, "Oh hey."),
            _ => panic!(),
        }
    }

    #[test]
    fn declarative_with_action_returns_action_response() {
        // Declarative skill authors write the presentation-envelope fragment
        // directly under `action:`. The adapter overlays `v` and `speak`
        // onto it from the `response` field.
        let src = r#"---
name: open-spotify
description: Opens Spotify. Use when the user wants to open Spotify.
metadata:
  ari:
    id: ai.example.open.spotify
    version: "1"
    engine: ">=0.3"
    capabilities: [launch_app]
    matching:
      patterns:
        - keywords: [open, spotify]
    declarative:
      response: "Opening Spotify."
      action:
        launch_app: spotify
---
"#;
        let s = DeclarativeSkill::from_skillfile(&parse(src)).unwrap();
        let ctx = SkillContext::default();
        match s.execute("open spotify", &ctx) {
            Response::Action(v) => {
                assert_eq!(v["v"], 1);
                assert_eq!(v["speak"], "Opening Spotify.");
                assert_eq!(v["launch_app"], "spotify");
            }
            _ => panic!("expected action response"),
        }
    }

    #[test]
    fn regex_pattern_compiles_and_matches() {
        let src = r#"---
name: weather-q
description: Catches weather questions. Use when the user asks about the weather.
metadata:
  ari:
    id: ai.example.weatherq
    version: "1"
    engine: ">=0.3"
    matching:
      patterns:
        - regex: "what.*(weather|temperature)"
          weight: 0.85
    declarative:
      response: "I have no idea, ask a window."
---
"#;
        let s = DeclarativeSkill::from_skillfile(&parse(src)).unwrap();
        let ctx = SkillContext::default();
        assert_eq!(s.score("what is the weather", &ctx), 0.85);
        assert_eq!(s.score("what is the temperature today", &ctx), 0.85);
        assert_eq!(s.score("the weather is nice", &ctx), 0.0);
    }

    #[test]
    fn invalid_regex_is_caught_at_load_time() {
        let src = r#"---
name: bad-regex
description: Has a busted regex. Use never.
metadata:
  ari:
    id: ai.example.badregex
    version: "1"
    engine: ">=0.3"
    matching:
      patterns:
        - regex: "[unclosed"
    declarative:
      response: "x"
---
"#;
        let err = DeclarativeSkill::from_skillfile(&parse(src)).unwrap_err();
        match err {
            AdapterError::Scorer(ScorerError::BadRegex { pattern, .. }) => {
                assert_eq!(pattern, "[unclosed");
            }
            _ => panic!("expected BadRegex"),
        }
    }

    #[test]
    fn highest_weight_pattern_wins_when_multiple_match() {
        let src = r#"---
name: multi
description: Multi-pattern weights. Use to test the scorer.
metadata:
  ari:
    id: ai.example.multi
    version: "1"
    engine: ">=0.3"
    matching:
      patterns:
        - keywords: [foo]
          weight: 0.5
        - keywords: [foo, bar]
          weight: 0.9
        - keywords: [foo, baz]
          weight: 0.7
    declarative:
      response: "ok"
---
"#;
        let s = DeclarativeSkill::from_skillfile(&parse(src)).unwrap();
        let ctx = SkillContext::default();
        // Only the first pattern matches
        assert_eq!(s.score("foo alone", &ctx), 0.5);
        // First and third match → 0.7 wins
        assert_eq!(s.score("foo and baz", &ctx), 0.7);
        // All three match → 0.9 wins
        assert_eq!(s.score("foo bar and baz", &ctx), 0.9);
    }

    #[test]
    fn wasm_skillfile_rejected_by_declarative_adapter() {
        let src = r#"---
name: weather
description: Weather lookup. Use when the user asks about the weather.
metadata:
  ari:
    id: ai.example.weather
    version: "1"
    engine: ">=0.3"
    capabilities: [http]
    matching:
      patterns:
        - keywords: [weather]
    wasm:
      module: skill.wasm
---
"#;
        let err = DeclarativeSkill::from_skillfile(&parse(src)).unwrap_err();
        assert!(matches!(err, AdapterError::NotDeclarative));
    }

    #[test]
    fn skillfile_without_ari_extension_rejected() {
        let src = "---\nname: x\ndescription: Plain AgentSkills doc, no Ari extension.\n---\n";
        let err = DeclarativeSkill::from_skillfile(&parse(src)).unwrap_err();
        assert!(matches!(err, AdapterError::NotAnAriSkill));
    }

    #[test]
    fn pick_counter_advances_across_executes() {
        let s = coin_flip();
        let ctx = SkillContext::default();
        let mut seen_heads = false;
        let mut seen_tails = false;
        for _ in 0..10 {
            match s.execute("flip a coin", &ctx) {
                Response::Text(t) if t == "Heads." => seen_heads = true,
                Response::Text(t) if t == "Tails." => seen_tails = true,
                _ => panic!("unexpected response"),
            }
        }
        // The counter is monotonic, so over 10 calls against a 2-element list
        // we must have seen both outcomes.
        assert!(seen_heads && seen_tails);
    }

    #[test]
    fn input_normalisation_is_applied_before_scoring() {
        let s = coin_flip();
        let ctx = SkillContext::default();
        // Punctuation, capitalisation, contractions all handled by normalize_input
        assert_eq!(s.score("FLIP, a Coin?!", &ctx), 0.95);
    }

    #[test]
    fn locale_aware_scoring_picks_per_locale_patterns() {
        use crate::localized_manifest::LocalizedManifestSet;
        use std::collections::BTreeMap;

        // Same id/type/capabilities/behaviour-shape across locales;
        // patterns and response intentionally differ.
        let en_src = r#"---
name: greet
description: Friendly greetings.
metadata:
  ari:
    id: ai.example.greet
    version: "0.1.0"
    engine: ">=0.3"
    capabilities: []
    languages: [en]
    specificity: medium
    matching:
      patterns:
        - keywords: [hello]
          weight: 1.0
    declarative:
      response: "Hi!"
---
"#;
        let it_src = r#"---
name: greet
description: Saluti amichevoli.
metadata:
  ari:
    id: ai.example.greet
    version: "0.1.0"
    engine: ">=0.3"
    capabilities: []
    languages: [it]
    specificity: medium
    matching:
      patterns:
        - keywords: [ciao]
          weight: 1.0
    declarative:
      response: "Ciao!"
---
"#;

        let mut manifests = BTreeMap::new();
        manifests.insert("en".to_string(), parse(en_src));
        manifests.insert("it".to_string(), parse(it_src));
        let set = LocalizedManifestSet { manifests };
        let skill = DeclarativeSkill::from_localized(&set).unwrap();

        // English context: English patterns match, Italian don't.
        let en_ctx = SkillContext { locale: "en".to_string() };
        assert_eq!(skill.score("hello", &en_ctx), 1.0);
        assert_eq!(skill.score("ciao", &en_ctx), 0.0);

        // Italian context: Italian patterns match, English don't.
        let it_ctx = SkillContext { locale: "it".to_string() };
        assert_eq!(skill.score("ciao", &it_ctx), 1.0);
        assert_eq!(skill.score("hello", &it_ctx), 0.0);

        // Spanish context (no es manifest shipped): falls back to
        // canonical English patterns.
        let es_ctx = SkillContext { locale: "es".to_string() };
        assert_eq!(skill.score("hello", &es_ctx), 1.0);
        assert_eq!(skill.score("ciao", &es_ctx), 0.0);

        // Execute also dispatches per-locale: response text differs.
        match skill.execute("hello", &en_ctx) {
            Response::Text(t) => assert_eq!(t, "Hi!"),
            other => panic!("expected text response, got {other:?}"),
        }
        match skill.execute("ciao", &it_ctx) {
            Response::Text(t) => assert_eq!(t, "Ciao!"),
            other => panic!("expected text response, got {other:?}"),
        }
        // Spanish falls back to canonical English response.
        match skill.execute("hello", &es_ctx) {
            Response::Text(t) => assert_eq!(t, "Hi!"),
            other => panic!("expected text response, got {other:?}"),
        }
    }
}
