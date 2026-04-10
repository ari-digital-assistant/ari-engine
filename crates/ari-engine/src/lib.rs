use ari_core::{normalize_input, Response, Skill, SkillContext, Specificity};
use ari_skill_loader::assistant::ConfigStore;
use ari_skill_loader::manifest::ApiConfig;
use std::sync::Arc;

// ── Skill router (FunctionGemma) ──────────────────────────────────────

/// What the skill router decided.
pub enum RouteResult {
    /// Route to a registered skill by id.
    Skill(String),
    /// Route to a system action (Android intent). The JSON value carries
    /// the action type and parameters for the frontend to dispatch.
    Action(serde_json::Value),
    /// No match — fall through to the assistant.
    NoMatch,
}

/// Trait for an LLM-based skill router that runs after the keyword
/// matcher fails. The router sees the user input and the list of
/// available skills, and either picks one, suggests a system action,
/// or declines.
///
/// Optional — if no router is set on the engine, the flow skips
/// straight from keyword scoring to the assistant.
pub trait SkillRouter: Send + Sync {
    fn route(
        &self,
        input: &str,
        skills: &[(String, String)], // (id, description) pairs
    ) -> RouteResult;
}

/// The text the engine returns when no skill matches the input. Exposed
/// publicly so the FFI layer can detect this exact response and convert it
/// into the dedicated `FfiResponse::NotUnderstood` variant — the Android
/// host uses that signal to trigger an STT retry path.
pub const FALLBACK_RESPONSE: &str = "Sorry, I didn't understand that.";

struct RankingRound {
    high_threshold: f32,
    medium_threshold: f32,
    low_threshold: f32,
}

const RANKING_ROUNDS: &[RankingRound] = &[
    RankingRound { high_threshold: 0.85, medium_threshold: f32::MAX, low_threshold: f32::MAX },
    RankingRound { high_threshold: 0.75, medium_threshold: 0.85, low_threshold: f32::MAX },
    RankingRound { high_threshold: 0.60, medium_threshold: 0.70, low_threshold: 0.80 },
];

#[derive(Debug, Clone)]
pub struct SkillScore {
    pub skill_id: String,
    pub specificity: Specificity,
    pub score: f32,
}

#[derive(Debug, Clone)]
pub struct DebugTrace {
    pub normalized_input: String,
    pub scores: Vec<SkillScore>,
    pub winner: Option<String>,
    pub round: Option<usize>,
}

/// Which assistant is currently active and how to call it.
pub enum ActiveAssistant {
    /// Use the built-in on-device LLM (routes to `self.llm`).
    Builtin,
    /// Use a cloud API via the generic adapter.
    Api {
        skill_id: String,
        config: ApiConfig,
        config_store: Arc<dyn ConfigStore>,
    },
}

pub struct Engine {
    skills: Vec<Box<dyn Skill>>,
    ctx: SkillContext,
    debug: bool,
    #[cfg(feature = "llm")]
    llm: Option<Box<dyn ari_llm::Fallback>>,
    active_assistant: Option<ActiveAssistant>,
    router: Option<Box<dyn SkillRouter>>,
}

impl Engine {
    pub fn new() -> Self {
        Self {
            skills: Vec::new(),
            ctx: SkillContext::default(),
            debug: false,
            #[cfg(feature = "llm")]
            llm: None,
            active_assistant: None,
            router: None,
        }
    }

    pub fn set_debug(&mut self, enabled: bool) {
        self.debug = enabled;
    }

    pub fn register_skill(&mut self, skill: Box<dyn Skill>) {
        self.skills.push(skill);
    }

    /// Set the LLM fallback. When set, the engine will consult the LLM
    /// before returning the fallback response, attempting skill rerouting
    /// or direct answers for unmatched input.
    #[cfg(feature = "llm")]
    pub fn set_llm(&mut self, llm: Box<dyn ari_llm::Fallback>) {
        self.llm = Some(llm);
    }

    /// Remove the LLM fallback, freeing its memory.
    #[cfg(feature = "llm")]
    pub fn set_llm_none(&mut self) {
        self.llm = None;
    }

    /// Set the active assistant provider.
    pub fn set_active_assistant(&mut self, assistant: Option<ActiveAssistant>) {
        self.active_assistant = assistant;
    }

    /// Set the skill router (e.g. FunctionGemma). When set, the engine
    /// consults the router after keyword scoring fails, before falling
    /// through to the assistant. Pass `None` to disable.
    pub fn set_router(&mut self, router: Option<Box<dyn SkillRouter>>) {
        self.router = router;
    }

    pub fn process_input(&self, input: &str) -> Response {
        let (response, trace) = self.process_input_traced(input);
        if self.debug
            && let Some(trace) = trace
        {
            eprintln!("[ari] input: {:?}", trace.normalized_input);
            for s in &trace.scores {
                eprintln!("[ari]   {} ({:?}): {:.3}", s.skill_id, s.specificity, s.score);
            }
            match (&trace.winner, trace.round) {
                (Some(w), Some(r)) => eprintln!("[ari] winner: {} (round {})", w, r + 1),
                _ => eprintln!("[ari] no match"),
            }
        }
        response
    }

    pub fn process_input_traced(&self, input: &str) -> (Response, Option<DebugTrace>) {
        let normalized = normalize_input(input.trim());
        if normalized.is_empty() {
            return (Response::Text(FALLBACK_RESPONSE.to_string()), None);
        }

        let scores: Vec<SkillScore> = self
            .skills
            .iter()
            .map(|s| SkillScore {
                skill_id: s.id().to_string(),
                specificity: s.specificity(),
                score: s.score(&normalized, &self.ctx),
            })
            .collect();

        let mut trace = DebugTrace {
            normalized_input: normalized.clone(),
            scores: scores.clone(),
            winner: None,
            round: None,
        };

        for (round_idx, round) in RANKING_ROUNDS.iter().enumerate() {
            let threshold_for = |spec: Specificity| -> f32 {
                match spec {
                    Specificity::High => round.high_threshold,
                    Specificity::Medium => round.medium_threshold,
                    Specificity::Low => round.low_threshold,
                }
            };

            let best = scores
                .iter()
                .filter(|s| s.score >= threshold_for(s.specificity))
                .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));

            if let Some(winner) = best {
                trace.winner = Some(winner.skill_id.clone());
                trace.round = Some(round_idx);

                let skill = self
                    .skills
                    .iter()
                    .find(|s| s.id() == winner.skill_id)
                    .unwrap();

                let response = skill.execute(&normalized, &self.ctx);
                return (response, Some(trace));
            }
        }

        // No keyword match. Try the skill router (FunctionGemma) if available.
        if let Some(ref router) = self.router {
            let skill_catalog: Vec<(String, String)> = self
                .skills
                .iter()
                .map(|s| (s.id().to_string(), s.description().to_string()))
                .collect();

            match router.route(&normalized, &skill_catalog) {
                RouteResult::Skill(ref id) => {
                    if let Some(skill) = self.skills.iter().find(|s| s.id() == id) {
                        trace.winner = Some(format!("router:{id}"));
                        let response = skill.execute(&normalized, &self.ctx);
                        return (response, Some(trace));
                    }
                }
                RouteResult::Action(action) => {
                    trace.winner = Some("router:action".to_string());
                    return (Response::Action(action), Some(trace));
                }
                RouteResult::NoMatch => {}
            }
        }

        // No skill matched. Delegate to the active assistant, if any.
        match &self.active_assistant {
            Some(ActiveAssistant::Builtin) => {
                #[cfg(feature = "llm")]
                if let Some(ref llm) = self.llm {
                    let catalog: Vec<ari_llm::SkillInfo> = self
                        .skills
                        .iter()
                        .map(|s| ari_llm::SkillInfo {
                            id: s.id().to_string(),
                            description: s.description().to_string(),
                        })
                        .collect();

                    if let Some(ari_llm::FallbackResult::DirectAnswer { text }) =
                        llm.try_answer(&normalized, &catalog)
                    {
                        trace.winner = Some("assistant:builtin".to_string());
                        return (Response::Text(text), Some(trace));
                    }
                }
            }
            Some(ActiveAssistant::Api {
                skill_id,
                config,
                config_store,
            }) => {
                match ari_skill_loader::call_assistant_api(
                    config,
                    skill_id,
                    config_store.as_ref(),
                    &normalized,
                ) {
                    Ok(text) if !text.is_empty() => {
                        trace.winner = Some(format!("assistant:{skill_id}"));
                        return (Response::Text(text), Some(trace));
                    }
                    _ => {}
                }
            }
            None => {}
        }

        (Response::Text(FALLBACK_RESPONSE.to_string()), Some(trace))
    }
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockSkill {
        id: &'static str,
        specificity: Specificity,
        fixed_score: f32,
        response: &'static str,
    }

    impl Skill for MockSkill {
        fn id(&self) -> &str { self.id }
        fn specificity(&self) -> Specificity { self.specificity }
        fn score(&self, _input: &str, _ctx: &SkillContext) -> f32 { self.fixed_score }
        fn execute(&self, _input: &str, _ctx: &SkillContext) -> Response {
            Response::Text(self.response.to_string())
        }
    }

    // --- Fallback behaviour ---

    #[test]
    fn no_skills_returns_fallback() {
        let engine = Engine::new();
        let resp = engine.process_input("hello");
        assert!(matches!(resp, Response::Text(ref s) if s == FALLBACK_RESPONSE));
    }

    #[test]
    fn empty_input_returns_fallback_with_no_trace() {
        let engine = Engine::new();
        let (resp, trace) = engine.process_input_traced("   ");
        assert!(matches!(resp, Response::Text(ref s) if s == FALLBACK_RESPONSE));
        assert!(trace.is_none());
    }

    #[test]
    fn punctuation_only_returns_fallback() {
        let engine = Engine::new();
        // "!!??" normalises to "" (all stripped), no trace
        let (resp, trace) = engine.process_input_traced("!!??");
        assert!(matches!(resp, Response::Text(ref s) if s == FALLBACK_RESPONSE));
        assert!(trace.is_none());
        // "..." normalises to "..." (dots preserved for decimal math), gets trace but no winner
        let (resp2, trace2) = engine.process_input_traced("...");
        assert!(matches!(resp2, Response::Text(ref s) if s == FALLBACK_RESPONSE));
        assert!(trace2.is_some());
    }

    #[test]
    fn below_all_thresholds_returns_fallback_with_trace() {
        let mut engine = Engine::new();
        engine.register_skill(Box::new(MockSkill {
            id: "weak", specificity: Specificity::High, fixed_score: 0.3, response: "nope",
        }));
        let (resp, trace) = engine.process_input_traced("test");
        assert!(matches!(resp, Response::Text(ref s) if s == FALLBACK_RESPONSE));
        let trace = trace.unwrap();
        assert!(trace.winner.is_none());
        assert!(trace.round.is_none());
        assert_eq!(trace.scores.len(), 1);
        assert_eq!(trace.scores[0].score, 0.3);
    }

    // --- Ranking rounds ---

    #[test]
    fn high_specificity_at_085_wins_round_one() {
        let mut engine = Engine::new();
        engine.register_skill(Box::new(MockSkill {
            id: "high", specificity: Specificity::High, fixed_score: 0.85, response: "high",
        }));
        let (_, trace) = engine.process_input_traced("test");
        let trace = trace.unwrap();
        assert_eq!(trace.winner.as_deref(), Some("high"));
        assert_eq!(trace.round, Some(0));
    }

    #[test]
    fn high_specificity_at_084_misses_round_one_hits_round_two() {
        let mut engine = Engine::new();
        engine.register_skill(Box::new(MockSkill {
            id: "high", specificity: Specificity::High, fixed_score: 0.84, response: "high",
        }));
        let (_, trace) = engine.process_input_traced("test");
        let trace = trace.unwrap();
        assert_eq!(trace.winner.as_deref(), Some("high"));
        assert_eq!(trace.round, Some(1));
    }

    #[test]
    fn medium_excluded_from_round_one() {
        let mut engine = Engine::new();
        engine.register_skill(Box::new(MockSkill {
            id: "med", specificity: Specificity::Medium, fixed_score: 0.99, response: "med",
        }));
        let (_, trace) = engine.process_input_traced("test");
        let trace = trace.unwrap();
        // Medium can't win round 1 (threshold is f32::MAX), enters round 2
        assert_eq!(trace.round, Some(1));
    }

    #[test]
    fn low_excluded_from_rounds_one_and_two() {
        let mut engine = Engine::new();
        engine.register_skill(Box::new(MockSkill {
            id: "low", specificity: Specificity::Low, fixed_score: 0.99, response: "low",
        }));
        let (_, trace) = engine.process_input_traced("test");
        let trace = trace.unwrap();
        // Low can't win rounds 1 or 2, enters round 3
        assert_eq!(trace.round, Some(2));
    }

    #[test]
    fn low_at_079_misses_all_rounds() {
        let mut engine = Engine::new();
        engine.register_skill(Box::new(MockSkill {
            id: "low", specificity: Specificity::Low, fixed_score: 0.79, response: "low",
        }));
        let (resp, trace) = engine.process_input_traced("test");
        assert!(matches!(resp, Response::Text(ref s) if s == FALLBACK_RESPONSE));
        assert!(trace.unwrap().winner.is_none());
    }

    #[test]
    fn high_beats_low_even_when_low_scores_higher() {
        let mut engine = Engine::new();
        engine.register_skill(Box::new(MockSkill {
            id: "high", specificity: Specificity::High, fixed_score: 0.86, response: "high wins",
        }));
        engine.register_skill(Box::new(MockSkill {
            id: "low", specificity: Specificity::Low, fixed_score: 0.95, response: "low wins",
        }));
        // High at 0.86 wins round 1. Low at 0.95 can't enter until round 3.
        let (resp, trace) = engine.process_input_traced("test");
        assert!(matches!(resp, Response::Text(ref s) if s == "high wins"));
        assert_eq!(trace.unwrap().round, Some(0));
    }

    #[test]
    fn higher_score_wins_within_same_round() {
        let mut engine = Engine::new();
        engine.register_skill(Box::new(MockSkill {
            id: "a", specificity: Specificity::High, fixed_score: 0.86, response: "a",
        }));
        engine.register_skill(Box::new(MockSkill {
            id: "b", specificity: Specificity::High, fixed_score: 0.92, response: "b",
        }));
        let resp = engine.process_input("test");
        assert!(matches!(resp, Response::Text(ref s) if s == "b"));
    }

    // --- Trace ---

    #[test]
    fn trace_contains_all_scores_and_correct_winner() {
        let mut engine = Engine::new();
        engine.register_skill(Box::new(MockSkill {
            id: "a", specificity: Specificity::High, fixed_score: 0.9, response: "a",
        }));
        engine.register_skill(Box::new(MockSkill {
            id: "b", specificity: Specificity::Medium, fixed_score: 0.5, response: "b",
        }));
        engine.register_skill(Box::new(MockSkill {
            id: "c", specificity: Specificity::Low, fixed_score: 0.1, response: "c",
        }));
        let (_, trace) = engine.process_input_traced("test");
        let trace = trace.unwrap();
        assert_eq!(trace.scores.len(), 3);
        assert_eq!(trace.winner.as_deref(), Some("a"));

        let score_a = trace.scores.iter().find(|s| s.skill_id == "a").unwrap();
        assert_eq!(score_a.score, 0.9);
        assert_eq!(score_a.specificity, Specificity::High);

        let score_c = trace.scores.iter().find(|s| s.skill_id == "c").unwrap();
        assert_eq!(score_c.score, 0.1);
    }

    #[test]
    fn input_is_normalized_before_scoring() {
        let mut engine = Engine::new();
        engine.register_skill(Box::new(MockSkill {
            id: "any", specificity: Specificity::High, fixed_score: 0.95, response: "ok",
        }));
        let (_, trace) = engine.process_input_traced("What's the TIME?!");
        assert_eq!(trace.unwrap().normalized_input, "what is the time");
    }

    #[test]
    fn trace_reports_no_winner_when_no_match() {
        let mut engine = Engine::new();
        engine.register_skill(Box::new(MockSkill {
            id: "x", specificity: Specificity::High, fixed_score: 0.1, response: "x",
        }));
        let (_, trace) = engine.process_input_traced("test");
        let trace = trace.unwrap();
        assert!(trace.winner.is_none());
        assert!(trace.round.is_none());
        assert_eq!(trace.scores.len(), 1);
    }
}
