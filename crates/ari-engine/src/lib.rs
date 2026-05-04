use ari_core::{
    normalize_input, Response, RouteResult, Skill, SkillContext, SkillRouter, Specificity,
};
use ari_skill_loader::assistant::{AssistantApiError, ConfigStore};
use ari_skill_loader::manifest::ApiConfig;
use ari_skill_loader::wasm::{LogLevel, LogSink};
use std::sync::Arc;

pub mod named_assistant;
pub use named_assistant::NamedAssistantBinding;

/// Pseudo skill-id used for engine-emitted log lines so they surface in
/// `adb logcat -s AriSkill` alongside real skill traces without being
/// mistaken for a registered skill.
const ENGINE_LOG_TAG: &str = "ari-engine";

/// Host-implemented sink that receives envelopes produced outside the
/// synchronous `process_input` flow — currently only the phase-2
/// envelope from a Layer C assistant round-trip. Implementations must
/// be safe to call from any thread and are responsible for marshalling
/// to the UI thread themselves.
///
/// Mirrors [`LogSink`] shape and is installed via
/// [`Engine::set_envelope_sink`]. Pass `None` to keep the engine
/// strictly synchronous (all `consult_assistant` directives become
/// inert, skill's first envelope is returned unchanged).
pub trait EnvelopeSink: Send + Sync {
    /// Push a JSON-serialised envelope plus the emitting skill id (so
    /// the frontend can resolve `asset:` references in it). Skill id
    /// matches the value [`Engine::process_input_with_skill`] returns.
    fn push(&self, envelope_json: &str, skill_id: Option<&str>);
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
#[derive(Clone)]
pub enum ActiveAssistant {
    /// Use the built-in on-device LLM (routes to `self.llm`). Carries
    /// the size tier of the loaded model so Layer C can gate
    /// consultation: small is too dim for structured JSON, medium and
    /// large are eligible.
    Builtin { tier: ari_llm::BuiltinTier },
    /// Use a cloud API via the generic adapter.
    Api {
        skill_id: String,
        config: ApiConfig,
        config_store: Arc<dyn ConfigStore>,
    },
}

pub struct Engine {
    /// Stored as `Arc<dyn Skill>` so Layer C's background thread can
    /// clone a reference to the winning skill and invoke
    /// [`Skill::execute_continuation`] on it after the assistant
    /// round-trip. Skill trait is `Send + Sync`, so the clone is
    /// safe to move across threads.
    skills: Vec<Arc<dyn Skill>>,
    ctx: SkillContext,
    debug: bool,
    #[cfg(feature = "llm")]
    llm: Option<Arc<dyn ari_llm::Fallback>>,
    active_assistant: Option<ActiveAssistant>,
    router: Option<Box<dyn SkillRouter>>,
    /// Optional sink so engine-internal paths (currently Layer C) can
    /// surface diagnostics in the same channel skills use. `None` means
    /// those log calls are no-ops — no formatting cost either.
    log_sink: Option<Arc<dyn LogSink>>,
    /// Optional sink for asynchronously-produced envelopes — currently
    /// only phase-2 of a Layer C round-trip. When `None`, the
    /// `consult_assistant` directive is inert (skill's first envelope
    /// is returned unchanged).
    envelope_sink: Option<Arc<dyn EnvelopeSink>>,
    /// Named cloud assistants addressable as "ask <alias> ...". Pushed
    /// by [`AriEngine::AssistantRegistry::apply_to_engine`] from the
    /// installed skill set. Empty when no community assistants are
    /// installed (or none declare aliases).
    named_assistants: Vec<NamedAssistantBinding>,
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
            log_sink: None,
            envelope_sink: None,
            named_assistants: Vec::new(),
        }
    }

    /// Install a log sink for engine-internal diagnostics. Currently only
    /// Layer C (assistant consultation on low-confidence envelopes) uses
    /// it. Pass `None` to silence. Separate from skill logging — the
    /// skill-loader has its own sink threaded through `reload_*` helpers.
    pub fn set_log_sink(&mut self, sink: Option<Arc<dyn LogSink>>) {
        self.log_sink = sink;
    }

    /// Update the locale that the engine threads into [`SkillContext`]
    /// on every subsequent `process_input` call. Skills read it via
    /// `ctx.locale` to dispatch their per-locale pattern scorers and
    /// response specs. Callers refresh this from the host's locale
    /// provider (frontend DataStore on Android) before each utterance
    /// — Phase 1 of the multi-language plan put `LocaleProvider` on
    /// the FFI engine; this is where it lands inside the inner engine's
    /// SkillContext.
    pub fn set_locale(&mut self, locale: String) {
        self.ctx.locale = locale;
    }

    /// Install an envelope sink so the engine can push phase-2 Layer C
    /// envelopes (produced asynchronously after the assistant replies)
    /// back to the host. When `None`, the `consult_assistant` directive
    /// is inert: the skill's first envelope is returned unchanged and no
    /// assistant round-trip runs. Set at startup before the first
    /// `process_input` call.
    pub fn set_envelope_sink(&mut self, sink: Option<Arc<dyn EnvelopeSink>>) {
        self.envelope_sink = sink;
    }

    fn log(&self, level: LogLevel, message: &str) {
        if let Some(ref sink) = self.log_sink {
            sink.log(ENGINE_LOG_TAG, level, message);
        }
    }

    pub fn set_debug(&mut self, enabled: bool) {
        self.debug = enabled;
    }

    pub fn register_skill(&mut self, skill: Box<dyn Skill>) {
        // Box<dyn Skill> → Arc<dyn Skill> via the std From impl. Arc is
        // needed so Layer C's background thread can hold a reference to
        // the winning skill and drive its continuation.
        self.skills.push(Arc::from(skill));
    }

    /// Set the LLM fallback. When set, the engine will consult the LLM
    /// before returning the fallback response, attempting skill rerouting
    /// or direct answers for unmatched input. Stored as `Arc` so the
    /// Layer C worker thread can clone a handle for on-device assistant
    /// consultation.
    #[cfg(feature = "llm")]
    pub fn set_llm(&mut self, llm: Arc<dyn ari_llm::Fallback>) {
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

    /// Replace the list of name-addressable assistants. Pushed by the
    /// FFI registry on every install/uninstall and on every active-
    /// assistant change. An empty list disables "ask <alias> ..."
    /// routing without affecting the active-assistant fallback.
    pub fn set_named_assistants(&mut self, list: Vec<NamedAssistantBinding>) {
        self.named_assistants = list;
    }

    /// Set the skill router (e.g. FunctionGemma). When set, the engine
    /// consults the router after keyword scoring fails, before falling
    /// through to the assistant. Pass `None` to disable.
    pub fn set_router(&mut self, router: Option<Box<dyn SkillRouter>>) {
        self.router = router;
    }

    pub fn process_input(&self, input: &str) -> Response {
        self.process_input_with_skill(input).0
    }

    /// Like [`process_input`] but also returns the id of the skill whose
    /// `execute` produced the response, or `None` if the response came from
    /// a non-skill path (empty input, generic fallback, router-direct
    /// action, or assistant API). The Android frontend uses this to resolve
    /// `asset:<path>` references in action envelopes back to the emitting
    /// skill's bundle directory.
    pub fn process_input_with_skill(&self, input: &str) -> (Response, Option<String>) {
        let (response, trace) = self.process_input_traced(input);
        if self.debug
            && let Some(ref trace) = trace
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
        let skill_id = trace.and_then(|t| {
            // Strip routing-path prefixes so the returned id is always the
            // raw emitting-skill id (e.g. "dev.heyari.timer"), never a
            // synthetic marker like "router:action" or "assistant:...".
            t.winner.and_then(|w| {
                if w == "router:action" {
                    None
                } else if let Some(rest) = w.strip_prefix("router:") {
                    Some(rest.to_string())
                } else if let Some(rest) = w.strip_prefix("named_assistant:") {
                    Some(rest.to_string())
                } else if let Some(rest) = w.strip_prefix("assistant:") {
                    Some(rest.to_string())
                } else {
                    Some(w)
                }
            })
        });
        (response, skill_id)
    }

    pub fn process_input_traced(&self, input: &str) -> (Response, Option<DebugTrace>) {
        let normalized = normalize_input(input.trim(), &self.ctx.locale);
        if normalized.is_empty() {
            return (Response::Text(FALLBACK_RESPONSE.to_string()), None);
        }

        // "Ask <assistant> X" short-circuit. Runs before keyword
        // scoring so a high-specificity skill (e.g. time) can't snatch
        // utterances like "ask chatgpt what time is it" from the named
        // assistant. If no alias matches, the normal pipeline below
        // runs untouched.
        if let Some(m) = named_assistant::match_named(&normalized, &self.named_assistants) {
            let trace = DebugTrace {
                normalized_input: normalized.clone(),
                scores: Vec::new(),
                winner: Some(format!("named_assistant:{}", m.binding.skill_id)),
                round: None,
            };
            self.log(
                LogLevel::Info,
                &format!(
                    "named_assistant: dispatching skill={} (prompt_len={})",
                    m.binding.skill_id,
                    m.remainder.len()
                ),
            );
            let response = dispatch_named_assistant(m.binding, &m.remainder, |level, msg| {
                self.log(level, msg)
            });
            return (response, Some(trace));
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
                    .unwrap()
                    .clone();

                let response = skill.execute(&normalized, &self.ctx);
                let response = self.maybe_intercept_consult(skill, response);
                return (response, Some(trace));
            }
        }

        // No keyword match. Try the skill router (FunctionGemma) if available.
        if let Some(ref router) = self.router {
            let skill_catalog: Vec<(String, String, String)> = self
                .skills
                .iter()
                .map(|s| (
                    s.id().to_string(),
                    s.description().to_string(),
                    s.parameters_schema().to_string(),
                ))
                .collect();

            let route_result = router.route(&normalized, &skill_catalog);

            // Diagnostic: log the model's raw output so we can see what
            // FunctionGemma actually emits — function name + args block +
            // stop tokens. Useful for verifying whether the model is
            // producing usable typed-args we can consume, or whether the
            // training/inference prompt needs work before we plumb args
            // through to skills.
            if let Some(raw) = router.last_raw_output() {
                self.log(
                    LogLevel::Info,
                    &format!("router: raw output ({} bytes): {raw:?}", raw.len()),
                );
            }

            match route_result {
                RouteResult::Skill { ref id, confidence } => {
                    if confidence < ari_core::MIN_ROUTER_CONFIDENCE {
                        self.log(
                            LogLevel::Info,
                            &format!(
                                "router: skipping skill={id} — confidence {confidence:.3} \
                                 below threshold {threshold:.3}; falling through to assistant",
                                threshold = ari_core::MIN_ROUTER_CONFIDENCE,
                            ),
                        );
                    } else if let Some(skill) = self.skills.iter().find(|s| s.id() == id).cloned() {
                        trace.winner = Some(format!("router:{id}"));
                        self.log(
                            LogLevel::Info,
                            &format!("router: dispatching skill={id} (confidence {confidence:.3})"),
                        );
                        let response = skill.execute(&normalized, &self.ctx);
                        let response = self.maybe_intercept_consult(skill, response);
                        return (response, Some(trace));
                    }
                }
                RouteResult::SkillWithArgs {
                    ref id,
                    ref args_json,
                    confidence,
                } => {
                    if confidence < ari_core::MIN_ROUTER_CONFIDENCE {
                        self.log(
                            LogLevel::Info,
                            &format!(
                                "router: skipping skill={id} — confidence {confidence:.3} \
                                 below threshold {threshold:.3}; falling through to assistant",
                                threshold = ari_core::MIN_ROUTER_CONFIDENCE,
                            ),
                        );
                    } else if let Some(skill) = self.skills.iter().find(|s| s.id() == id).cloned() {
                        trace.winner = Some(format!("router:{id}+args"));
                        self.log(
                            LogLevel::Info,
                            &format!(
                                "router: dispatching skill={id} with typed args ({} bytes, confidence {confidence:.3})",
                                args_json.len()
                            ),
                        );
                        let response = skill.execute_with_args(&normalized, args_json, &self.ctx);
                        let response = self.maybe_intercept_consult(skill, response);
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
            Some(ActiveAssistant::Builtin { .. }) => {
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

                    self.log(
                        LogLevel::Info,
                        &format!(
                            "assistant:builtin: invoking llm.try_answer (input_len={})",
                            normalized.len()
                        ),
                    );
                    let result = llm.try_answer(&normalized, &catalog);
                    match result {
                        Some(ari_llm::FallbackResult::DirectAnswer { text }) => {
                            let preview: String = text.chars().take(160).collect();
                            self.log(
                                LogLevel::Info,
                                &format!(
                                    "assistant:builtin: try_answer returned answer ({} bytes): {preview:?}",
                                    text.len()
                                ),
                            );
                            trace.winner = Some("assistant:builtin".to_string());
                            return (Response::Text(text), Some(trace));
                        }
                        None => {
                            let detail = llm
                                .last_error()
                                .unwrap_or_else(|| "(no error reason recorded)".to_string());
                            self.log(
                                LogLevel::Warn,
                                &format!(
                                    "assistant:builtin: try_answer returned None — {detail}. \
                                     Falling through to FALLBACK_RESPONSE."
                                ),
                            );
                        }
                    }
                } else {
                    self.log(
                        LogLevel::Warn,
                        "assistant:builtin: no LLM loaded — falling through to FALLBACK_RESPONSE",
                    );
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

    /// If the skill's response envelope carries a `consult_assistant`
    /// directive (Layer C v2), split it out: strip the directive from
    /// the phase-1 envelope we return synchronously, and spawn a
    /// background thread that runs the assistant round-trip and pushes
    /// the phase-2 envelope via [`EnvelopeSink`]. When anything is
    /// missing (no sink, malformed directive, etc.) the skill's first
    /// envelope is returned unchanged — no assistant call happens.
    fn maybe_intercept_consult(
        &self,
        skill: Arc<dyn Skill>,
        response: Response,
    ) -> Response {
        let mut action = match response {
            Response::Action(v) => v,
            other => return other,
        };

        let directive_value = match action
            .as_object_mut()
            .and_then(|obj| obj.remove("consult_assistant"))
        {
            Some(v) => v,
            None => return Response::Action(action),
        };

        let directive = match parse_consult_directive(&directive_value) {
            Some(d) => d,
            None => {
                self.log(
                    LogLevel::Warn,
                    "layer-c: consult_assistant directive malformed — ignoring, returning phase-1 envelope unchanged",
                );
                return Response::Action(action);
            }
        };

        let sink = match self.envelope_sink.clone() {
            Some(s) => s,
            None => {
                self.log(
                    LogLevel::Warn,
                    "layer-c: consult_assistant requested but no envelope sink installed — phase-2 suppressed",
                );
                return Response::Action(action);
            }
        };

        let assistant = self.active_assistant.clone();
        #[cfg(feature = "llm")]
        let llm = self.llm.clone();
        let log_sink = self.log_sink.clone();
        let ctx = self.ctx.clone();
        let skill_id = skill.id().to_string();

        self.log(
            LogLevel::Info,
            &format!(
                "layer-c: phase-1 returned, spawning phase-2 for skill={} prompt_len={}",
                skill_id,
                directive.prompt.len()
            ),
        );

        std::thread::spawn(move || {
            run_consult_phase_two(
                skill,
                skill_id,
                directive,
                assistant,
                #[cfg(feature = "llm")]
                llm,
                ctx,
                sink,
                log_sink,
            );
        });

        Response::Action(action)
    }
}

/// Parsed form of the `consult_assistant` envelope directive. Shape is
/// stable — skills compose these JSON blobs and the engine extracts
/// them at phase-1 interception time.
#[derive(Debug, Clone)]
struct ConsultDirective {
    /// Final prompt the engine sends to the assistant verbatim. Skills
    /// perform their own `{utterance}` / `{unparsed}` substitution
    /// before assembling this string.
    prompt: String,
    /// Opaque string the skill uses to carry state into its
    /// continuation invocation. Engine treats it as a black box.
    continuation_context: String,
}

/// Pretty user-facing label for an assistant skill id. Strips the
/// `dev.heyari.assistant.` prefix and capitalises — best-effort, the
/// frontend can do better but the engine is the only thing that
/// surfaces error text for named-assistant dispatch.
fn assistant_display_name(skill_id: &str) -> String {
    let stem = skill_id
        .rsplit('.')
        .next()
        .unwrap_or(skill_id);
    let mut chars = stem.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().chain(chars).collect(),
        None => skill_id.to_string(),
    }
}

/// Dispatch a "ask <alias> X" match to the cloud API and translate any
/// failure into a user-facing text response. The closure logs detailed
/// diagnostics via the engine's log_sink — never leaks raw API error
/// bodies to the user.
fn dispatch_named_assistant<F: Fn(LogLevel, &str)>(
    binding: &NamedAssistantBinding,
    prompt: &str,
    log: F,
) -> Response {
    let display = assistant_display_name(&binding.skill_id);
    match ari_skill_loader::call_assistant_api(
        &binding.config,
        &binding.skill_id,
        binding.config_store.as_ref(),
        prompt,
    ) {
        Ok(text) if !text.is_empty() => Response::Text(text),
        Ok(_) => {
            log(
                LogLevel::Warn,
                &format!(
                    "named_assistant: skill={} returned empty body",
                    binding.skill_id
                ),
            );
            Response::Text(format!("{display} couldn't reply right now."))
        }
        Err(AssistantApiError::MissingConfig { ref key }) => {
            log(
                LogLevel::Warn,
                &format!(
                    "named_assistant: skill={} missing config key={}",
                    binding.skill_id, key
                ),
            );
            Response::Text(format!(
                "{display} isn't set up yet. Add your API key in Settings → Assistants."
            ))
        }
        Err(AssistantApiError::Timeout) => {
            log(
                LogLevel::Warn,
                &format!("named_assistant: skill={} timed out", binding.skill_id),
            );
            Response::Text(format!("{display} took too long to reply — try again."))
        }
        Err(AssistantApiError::ApiError { status, ref body }) => {
            log(
                LogLevel::Warn,
                &format!(
                    "named_assistant: skill={} api error {status}: {body}",
                    binding.skill_id
                ),
            );
            // Anthropic, OpenAI, and Gemini all nest the user-facing
            // reason at error.message in the JSON response. Surface it
            // when present so problems like "out of credits" or "model
            // not found" are actionable instead of generic. Cap at
            // ~200 chars to keep an accidental verbose body from
            // dumping into the conversation UI.
            match extract_api_error_message(body) {
                Some(msg) => Response::Text(format!("{display}: {msg}")),
                None => Response::Text(format!(
                    "{display} returned an error (HTTP {status})."
                )),
            }
        }
        Err(e) => {
            log(
                LogLevel::Warn,
                &format!("named_assistant: skill={} failed: {e}", binding.skill_id),
            );
            Response::Text(format!("{display} couldn't reply right now."))
        }
    }
}

/// Pull a user-facing reason out of an API error body. All three of
/// our cloud providers (Anthropic, OpenAI, Gemini-OpenAI-compat) nest
/// the message at `error.message`. Returns `None` if the body isn't
/// JSON or the field is missing.
fn extract_api_error_message(body: &str) -> Option<String> {
    const MAX_LEN: usize = 200;
    let v: serde_json::Value = serde_json::from_str(body).ok()?;
    let msg = v.get("error")?.get("message")?.as_str()?.trim();
    if msg.is_empty() {
        return None;
    }
    if msg.chars().count() > MAX_LEN {
        let truncated: String = msg.chars().take(MAX_LEN).collect();
        Some(format!("{truncated}…"))
    } else {
        Some(msg.to_string())
    }
}

fn parse_consult_directive(v: &serde_json::Value) -> Option<ConsultDirective> {
    let obj = v.as_object()?;
    let prompt = obj.get("prompt").and_then(|p| p.as_str())?.to_string();
    if prompt.is_empty() {
        return None;
    }
    let continuation_context = obj
        .get("continuation_context")
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();
    Some(ConsultDirective {
        prompt,
        continuation_context,
    })
}

/// How long Layer C will wait for an assistant reply before pushing
/// a "still working on it" delay phrase to the user. Most cloud
/// round-trips finish well inside this; saying anything before then
/// just gets in the way of the actual answer.
const DELAY_PHRASE_AFTER: std::time::Duration = std::time::Duration::from_secs(4);

/// Hard upper bound on a cloud Layer C round-trip. The cloud path's
/// reqwest client carries its own 30s ceiling already; this is the
/// outer guard. On timeout we abandon the worker and fall through to
/// the skill's warn-and-commit continuation.
const MAX_API_WAIT: std::time::Duration = std::time::Duration::from_secs(30);

/// Hard upper bound on an on-device Layer C round-trip. Generous
/// because thermally-throttled phones and software-emulated AVDs run
/// inference much slower than a flagship — E2B at 12-20 tok/s on a
/// real phone is ~10s for a typical reminder prompt, but on an x86_64
/// emulator without GPU passthrough it can be 30-60s. Hard enough that
/// truly stuck inference still bails, loose enough that the realistic
/// slow path can complete.
const MAX_ONDEVICE_WAIT: std::time::Duration = std::time::Duration::from_secs(60);

/// Conversational filler the engine speaks when the assistant takes
/// longer than [`DELAY_PHRASE_AFTER`]. One is picked per slow
/// round-trip — no need for cryptographic randomness, just enough
/// rotation that consecutive slow calls don't repeat the same line.
const DELAY_PHRASES: &[&str] = &[
    "Hang on...",
    "One moment...",
    "Just a sec...",
    "Working...",
    "Checking...",
    "Be right with you...",
];

fn pick_delay_phrase() -> &'static str {
    let idx = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as usize)
        .unwrap_or(0)
        % DELAY_PHRASES.len();
    DELAY_PHRASES[idx]
}

fn run_consult_phase_two(
    skill: Arc<dyn Skill>,
    skill_id: String,
    directive: ConsultDirective,
    assistant: Option<ActiveAssistant>,
    #[cfg(feature = "llm")] llm: Option<Arc<dyn ari_llm::Fallback>>,
    ctx: SkillContext,
    sink: Arc<dyn EnvelopeSink>,
    log_sink: Option<Arc<dyn LogSink>>,
) {
    let log = |level: LogLevel, msg: &str| {
        if let Some(ref s) = log_sink {
            s.log(ENGINE_LOG_TAG, level, msg);
        }
    };

    // Pick the wall-clock ceiling per assistant variant. On-device
    // gets a more generous budget because emulators and thermally-
    // throttled phones can run E2B/E4B much slower than a flagship.
    let max_wait = match assistant {
        Some(ActiveAssistant::Builtin { .. }) => MAX_ONDEVICE_WAIT,
        _ => MAX_API_WAIT,
    };

    // Run the assistant call on its own thread so we can recv-with-
    // timeout and push a "still working" phrase if the round-trip
    // takes more than DELAY_PHRASE_AFTER. Most calls finish well
    // before that threshold and the user sees a single bubble
    // (the answer); slow calls produce two — the delay phrase, then
    // the answer.
    let (tx, rx) = std::sync::mpsc::channel();
    let prompt_for_thread = directive.prompt.clone();
    let assistant_for_thread = assistant.clone();
    #[cfg(feature = "llm")]
    let llm_for_thread = llm.clone();
    std::thread::spawn(move || {
        #[cfg(feature = "llm")]
        let result = call_assistant_for_consult(
            &assistant_for_thread,
            &llm_for_thread,
            &prompt_for_thread,
        );
        #[cfg(not(feature = "llm"))]
        let result = call_assistant_for_consult(&assistant_for_thread, &prompt_for_thread);
        let _ = tx.send(result);
    });

    let assistant_outcome = match rx.recv_timeout(DELAY_PHRASE_AFTER) {
        Ok(result) => result,
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            // Slow round-trip: tell the user we're still on it, then
            // wait up to max_wait total before giving up.
            let phrase = pick_delay_phrase();
            log(
                LogLevel::Info,
                &format!("layer-c: assistant slow (>{}s) — pushing delay phrase {phrase:?}",
                    DELAY_PHRASE_AFTER.as_secs()),
            );
            let delay_envelope = serde_json::json!({ "v": 1, "speak": phrase });
            if let Ok(delay_json) = serde_json::to_string(&delay_envelope) {
                sink.push(&delay_json, Some(&skill_id));
            }
            let remaining = max_wait.saturating_sub(DELAY_PHRASE_AFTER);
            match rx.recv_timeout(remaining) {
                Ok(result) => result,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => Err(format!(
                    "assistant exceeded {}s wall-clock — abandoning",
                    max_wait.as_secs()
                )),
                Err(_) => Err("assistant worker thread vanished before delivering a result".into()),
            }
        }
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            Err("assistant worker thread vanished before delivering a result".into())
        }
    };

    // Fetch the assistant's response (or an empty string on failure —
    // the skill's continuation handler owns the fallback logic, since
    // it's the only layer with enough context to produce a sensible
    // recovery envelope).
    let response_for_skill = match assistant_outcome {
        Ok(text) => {
            log(
                LogLevel::Info,
                &format!("layer-c: assistant response ok ({} bytes)", text.len()),
            );
            text
        }
        Err(reason) => {
            log(
                LogLevel::Warn,
                &format!("layer-c: assistant unavailable ({reason}) — invoking continuation with empty response so the skill can run its own fallback"),
            );
            String::new()
        }
    };

    let continuation = skill.execute_continuation(
        &directive.continuation_context,
        &response_for_skill,
        &ctx,
    );

    let envelope = match continuation {
        Response::Action(v) => strip_nested_consult(v, &log),
        Response::Text(s) => serde_json::json!({ "v": 1, "speak": s }),
        Response::Binary { .. } => {
            log(
                LogLevel::Warn,
                "layer-c: continuation returned Binary response — unsupported, emitting generic error",
            );
            serde_json::json!({ "v": 1, "speak": "Something went wrong with that request." })
        }
    };

    let json = match serde_json::to_string(&envelope) {
        Ok(s) => s,
        Err(e) => {
            log(LogLevel::Error, &format!("layer-c: envelope serialisation failed: {e}"));
            return;
        }
    };
    log(
        LogLevel::Info,
        &format!("layer-c: pushing phase-2 envelope ({} bytes)", json.len()),
    );
    sink.push(&json, Some(&skill_id));
}

#[cfg(feature = "llm")]
fn call_assistant_for_consult(
    assistant: &Option<ActiveAssistant>,
    llm: &Option<Arc<dyn ari_llm::Fallback>>,
    prompt: &str,
) -> Result<String, String> {
    match assistant {
        Some(ActiveAssistant::Api {
            skill_id,
            config,
            config_store,
        }) => {
            let text = ari_skill_loader::call_assistant_api(
                config,
                skill_id,
                config_store.as_ref(),
                prompt,
            )
            .map_err(|e| e.to_string())?;
            if text.trim().is_empty() {
                Err("assistant returned empty response".into())
            } else {
                Ok(text)
            }
        }
        Some(ActiveAssistant::Builtin {
            tier: ari_llm::BuiltinTier::Small,
        }) => Err(
            "Layer C round-trip is gated to medium/large on-device tiers; \
             small is too small for reliable structured JSON"
                .into(),
        ),
        Some(ActiveAssistant::Builtin { tier: _ }) => {
            let llm = llm
                .as_ref()
                .ok_or_else(|| "on-device LLM not loaded".to_string())?;
            let raw = llm.run_prompt(prompt).map_err(|e| e.to_string())?;
            let stripped = ari_llm::strip_thinking(&raw);
            // Diagnostic: emit raw and stripped lengths + a preview so we
            // can tell from logcat whether Gemma produced content that
            // strip_thinking devoured (orphan <think> with no close,
            // typical for runs that hit MAX_GENERATION_TOKENS mid-think)
            // versus the model genuinely producing nothing.
            // Diagnostic preview lets us see from logcat whether
            // strip_thinking ate the answer (orphan <think> with no
            // close → everything stripped) vs Gemma producing nothing.
            // Returned via the Err string on the empty path.
            if stripped.trim().is_empty() {
                let raw_preview: String = raw.chars().take(200).collect();
                Err(format!(
                    "on-device LLM returned empty after strip_thinking (raw_len={}, raw_preview={raw_preview:?})",
                    raw.len()
                ))
            } else {
                Ok(stripped)
            }
        }
        None => Err("no active assistant configured".into()),
    }
}

#[cfg(not(feature = "llm"))]
fn call_assistant_for_consult(
    assistant: &Option<ActiveAssistant>,
    prompt: &str,
) -> Result<String, String> {
    match assistant {
        Some(ActiveAssistant::Api {
            skill_id,
            config,
            config_store,
        }) => {
            let text = ari_skill_loader::call_assistant_api(
                config,
                skill_id,
                config_store.as_ref(),
                prompt,
            )
            .map_err(|e| e.to_string())?;
            if text.trim().is_empty() {
                Err("assistant returned empty response".into())
            } else {
                Ok(text)
            }
        }
        Some(ActiveAssistant::Builtin { .. }) => Err(
            "on-device LLM not compiled in (llm feature disabled)".into(),
        ),
        None => Err("no active assistant configured".into()),
    }
}

/// Loop protection: strip any nested `consult_assistant` directive
/// from a phase-2 envelope. Prevents a skill from initiating an
/// unbounded chain of assistant round-trips per user utterance.
fn strip_nested_consult(
    mut action: serde_json::Value,
    log: &dyn Fn(LogLevel, &str),
) -> serde_json::Value {
    if let Some(obj) = action.as_object_mut() {
        if obj.remove("consult_assistant").is_some() {
            log(
                LogLevel::Warn,
                "layer-c: continuation envelope carried a nested consult_assistant directive — stripped (loop protection caps round-trips at 1)",
            );
        }
    }
    action
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

    // --- Named-assistant API error extraction ---

    #[test]
    fn extracts_anthropic_error_message() {
        let body = r#"{"type":"error","error":{"type":"invalid_request_error","message":"Your credit balance is too low."}}"#;
        assert_eq!(
            extract_api_error_message(body).as_deref(),
            Some("Your credit balance is too low.")
        );
    }

    #[test]
    fn extracts_openai_error_message() {
        let body = r#"{"error":{"message":"Incorrect API key provided.","type":"invalid_request_error","code":"invalid_api_key"}}"#;
        assert_eq!(
            extract_api_error_message(body).as_deref(),
            Some("Incorrect API key provided.")
        );
    }

    #[test]
    fn extract_returns_none_on_unstructured_body() {
        assert!(extract_api_error_message("not json at all").is_none());
        assert!(extract_api_error_message(r#"{"foo": "bar"}"#).is_none());
        assert!(extract_api_error_message(r#"{"error": "string not object"}"#).is_none());
    }

    #[test]
    fn extract_truncates_runaway_message() {
        let long = "a".repeat(500);
        let body = format!(r#"{{"error":{{"message":"{long}"}}}}"#);
        let extracted = extract_api_error_message(&body).unwrap();
        assert!(extracted.chars().count() <= 201, "got {} chars", extracted.chars().count());
        assert!(extracted.ends_with('…'));
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

    // --- Layer C v2: consult_assistant directive ---

    struct ActionSkill {
        id: &'static str,
        action: serde_json::Value,
    }

    impl Skill for ActionSkill {
        fn id(&self) -> &str { self.id }
        fn specificity(&self) -> Specificity { Specificity::High }
        fn score(&self, _input: &str, _ctx: &SkillContext) -> f32 { 0.95 }
        fn execute(&self, _input: &str, _ctx: &SkillContext) -> Response {
            Response::Action(self.action.clone())
        }
    }

    #[test]
    fn directive_parses_minimal_shape() {
        let v = serde_json::json!({
            "prompt": "what did they mean?",
            "continuation_context": "ctx"
        });
        let d = parse_consult_directive(&v).unwrap();
        assert_eq!(d.prompt, "what did they mean?");
        assert_eq!(d.continuation_context, "ctx");
    }

    #[test]
    fn directive_rejects_missing_prompt() {
        // Prompt is the one mandatory field — no prompt, no round-trip.
        let v = serde_json::json!({ "continuation_context": "x" });
        assert!(parse_consult_directive(&v).is_none());
    }

    #[test]
    fn directive_rejects_empty_prompt() {
        let v = serde_json::json!({ "prompt": "", "continuation_context": "x" });
        assert!(parse_consult_directive(&v).is_none());
    }

    #[test]
    fn directive_defaults_empty_context_when_absent() {
        let v = serde_json::json!({ "prompt": "anything" });
        let d = parse_consult_directive(&v).unwrap();
        assert_eq!(d.continuation_context, "");
    }

    #[test]
    fn strip_nested_consult_removes_nested_directive() {
        let silent = |_: LogLevel, _: &str| {};
        let with_nested = serde_json::json!({
            "v": 1,
            "speak": "done",
            "consult_assistant": { "prompt": "re-run", "continuation_context": "" }
        });
        let stripped = strip_nested_consult(with_nested, &silent);
        assert!(stripped.get("consult_assistant").is_none());
        assert_eq!(stripped["speak"], "done");
    }

    #[test]
    fn strip_nested_consult_leaves_clean_envelope_alone() {
        let silent = |_: LogLevel, _: &str| {};
        let clean = serde_json::json!({ "v": 1, "speak": "ok" });
        let out = strip_nested_consult(clean.clone(), &silent);
        assert_eq!(out, clean);
    }

    #[test]
    fn consult_directive_inert_without_envelope_sink() {
        // When the skill emits a consult_assistant but no sink is
        // installed, the engine returns the phase-1 envelope with the
        // directive stripped — no thread spawned, no hang.
        let mut engine = Engine::new();
        let payload = serde_json::json!({
            "v": 1,
            "speak": "ack",
            "consult_assistant": {
                "prompt": "anything",
                "continuation_context": "ctx"
            }
        });
        engine.register_skill(Box::new(ActionSkill {
            id: "test.consult",
            action: payload,
        }));
        let (resp, _) = engine.process_input_traced("trigger");
        match resp {
            Response::Action(v) => {
                assert_eq!(v["speak"], "ack");
                assert!(
                    v.get("consult_assistant").is_none(),
                    "consult_assistant must be stripped even when sink is absent — frontend shouldn't see the engine-internal directive"
                );
            }
            _ => panic!("expected Action response"),
        }
    }

    #[test]
    fn malformed_directive_returns_envelope_unchanged_without_field() {
        // Malformed consult_assistant → engine logs a warning, strips
        // the field, returns the remaining envelope. Skill's speak /
        // cards still render.
        let mut engine = Engine::new();
        let payload = serde_json::json!({
            "v": 1,
            "speak": "ack",
            "consult_assistant": {
                // missing required "prompt" field
                "continuation_context": "x"
            }
        });
        engine.register_skill(Box::new(ActionSkill {
            id: "test.malformed",
            action: payload,
        }));
        let (resp, _) = engine.process_input_traced("trigger");
        match resp {
            Response::Action(v) => {
                assert_eq!(v["speak"], "ack");
                assert!(v.get("consult_assistant").is_none());
            }
            _ => panic!("expected Action response"),
        }
    }

    #[test]
    fn non_action_response_passes_through() {
        // Text responses from skills bypass Layer C entirely — there's
        // no envelope to check.
        let mut engine = Engine::new();
        engine.register_skill(Box::new(MockSkill {
            id: "text.skill", specificity: Specificity::High, fixed_score: 0.95, response: "plain",
        }));
        let (resp, _) = engine.process_input_traced("anything");
        assert!(matches!(resp, Response::Text(ref s) if s == "plain"));
    }

    /// Test EnvelopeSink implementation that records every push into
    /// a shared `Vec`. Used by the integration-style tests below that
    /// want to verify the phase-2 envelope contents after the round-
    /// trip completes.
    struct RecordingSink(Arc<Mutex<Vec<(String, Option<String>)>>>);

    impl EnvelopeSink for RecordingSink {
        fn push(&self, envelope_json: &str, skill_id: Option<&str>) {
            self.0.lock().unwrap().push((
                envelope_json.to_string(),
                skill_id.map(|s| s.to_string()),
            ));
        }
    }

    /// Skill that emits a consult_assistant on first call and whose
    /// continuation returns a canned final envelope. Lets tests cover
    /// the full phase-1 → phase-2 round-trip without a real assistant.
    struct ConsultingSkill {
        id: &'static str,
        first_envelope: serde_json::Value,
        continuation_envelope: serde_json::Value,
    }

    impl Skill for ConsultingSkill {
        fn id(&self) -> &str { self.id }
        fn specificity(&self) -> Specificity { Specificity::High }
        fn score(&self, _input: &str, _ctx: &SkillContext) -> f32 { 0.95 }
        fn execute(&self, _input: &str, _ctx: &SkillContext) -> Response {
            Response::Action(self.first_envelope.clone())
        }
        fn execute_continuation(
            &self,
            _context: &str,
            _response: &str,
            _ctx: &SkillContext,
        ) -> Response {
            Response::Action(self.continuation_envelope.clone())
        }
    }

    use std::sync::Mutex;

    #[test]
    fn consult_without_assistant_pushes_fallback_via_skill_continuation() {
        // No active_assistant → call_assistant_for_consult errors →
        // skill.execute_continuation is still called (with empty
        // response string) → skill emits its fallback envelope →
        // engine pushes it. The thread we spawn is joined implicitly
        // via the recording sink; poll the sink briefly for the push.
        let recorded: Arc<Mutex<Vec<_>>> = Arc::new(Mutex::new(Vec::new()));
        let sink: Arc<dyn EnvelopeSink> = Arc::new(RecordingSink(recorded.clone()));

        let mut engine = Engine::new();
        engine.set_envelope_sink(Some(sink));
        engine.register_skill(Box::new(ConsultingSkill {
            id: "test.consulting",
            first_envelope: serde_json::json!({
                "v": 1,
                "speak": "let me check",
                "consult_assistant": {
                    "prompt": "interpret",
                    "continuation_context": "the utterance"
                }
            }),
            continuation_envelope: serde_json::json!({
                "v": 1,
                "speak": "fallback written"
            }),
        }));

        // Phase-1 return should have consult_assistant stripped.
        let (resp, _) = engine.process_input_traced("go");
        match resp {
            Response::Action(v) => {
                assert_eq!(v["speak"], "let me check");
                assert!(v.get("consult_assistant").is_none());
            }
            _ => panic!("expected phase-1 Action"),
        }

        // Background thread should push the phase-2 envelope quickly
        // — no real assistant call happens (no active_assistant), so
        // the continuation fires immediately. Poll for up to 2s.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while std::time::Instant::now() < deadline {
            if !recorded.lock().unwrap().is_empty() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        let pushed = recorded.lock().unwrap().clone();
        assert_eq!(pushed.len(), 1, "expected exactly one phase-2 push");
        let (json, skill_id) = &pushed[0];
        assert_eq!(skill_id.as_deref(), Some("test.consulting"));
        let v: serde_json::Value = serde_json::from_str(json).unwrap();
        assert_eq!(v["speak"], "fallback written");
    }

    #[cfg(feature = "llm")]
    #[test]
    fn consult_with_builtin_small_tier_falls_through_to_warn_and_commit() {
        // ActiveAssistant::Builtin { tier: Small } is rejected by
        // call_assistant_for_consult — the engine falls through to the
        // empty-string continuation path, same as no-assistant. Verifies
        // the size gate is wired correctly and Small never reaches
        // run_prompt (which would also fail because no LLM is loaded,
        // but the gate fires first with a clearer error).
        let recorded: Arc<Mutex<Vec<_>>> = Arc::new(Mutex::new(Vec::new()));
        let sink: Arc<dyn EnvelopeSink> = Arc::new(RecordingSink(recorded.clone()));

        let mut engine = Engine::new();
        engine.set_envelope_sink(Some(sink));
        engine.set_active_assistant(Some(ActiveAssistant::Builtin {
            tier: ari_llm::BuiltinTier::Small,
        }));
        engine.register_skill(Box::new(ConsultingSkill {
            id: "test.tier_gated",
            first_envelope: serde_json::json!({
                "v": 1,
                "speak": "let me check",
                "consult_assistant": {
                    "prompt": "interpret",
                    "continuation_context": "ctx"
                }
            }),
            continuation_envelope: serde_json::json!({
                "v": 1,
                "speak": "warn-and-commit fallback"
            }),
        }));

        let _ = engine.process_input_traced("go");

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while std::time::Instant::now() < deadline {
            if !recorded.lock().unwrap().is_empty() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        let pushed = recorded.lock().unwrap().clone();
        assert_eq!(
            pushed.len(),
            1,
            "Small-tier Builtin should be rejected and skill continuation should still fire"
        );
        let v: serde_json::Value = serde_json::from_str(&pushed[0].0).unwrap();
        assert_eq!(v["speak"], "warn-and-commit fallback");
    }

    #[test]
    fn nested_consult_in_continuation_is_stripped() {
        // Continuation envelope carrying its own consult_assistant
        // directive must have that field stripped before being pushed,
        // preventing an unbounded chain of assistant calls.
        let recorded: Arc<Mutex<Vec<_>>> = Arc::new(Mutex::new(Vec::new()));
        let sink: Arc<dyn EnvelopeSink> = Arc::new(RecordingSink(recorded.clone()));

        let mut engine = Engine::new();
        engine.set_envelope_sink(Some(sink));
        engine.register_skill(Box::new(ConsultingSkill {
            id: "test.nested",
            first_envelope: serde_json::json!({
                "v": 1,
                "speak": "ack",
                "consult_assistant": {
                    "prompt": "anything",
                    "continuation_context": ""
                }
            }),
            continuation_envelope: serde_json::json!({
                "v": 1,
                "speak": "phase-2",
                "consult_assistant": {
                    "prompt": "sneaky second round",
                    "continuation_context": ""
                }
            }),
        }));

        let _ = engine.process_input_traced("go");

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while std::time::Instant::now() < deadline {
            if !recorded.lock().unwrap().is_empty() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        let pushed = recorded.lock().unwrap().clone();
        assert_eq!(pushed.len(), 1);
        let v: serde_json::Value = serde_json::from_str(&pushed[0].0).unwrap();
        assert_eq!(v["speak"], "phase-2");
        assert!(
            v.get("consult_assistant").is_none(),
            "loop protection should strip nested consult_assistant"
        );
    }

}
