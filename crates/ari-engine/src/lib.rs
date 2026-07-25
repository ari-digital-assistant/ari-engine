use ari_core::{
    normalize_input, AppEntry, Response, RouteResult, Skill, SkillContext, SkillRouter,
    Specificity,
};
use ari_skill_loader::assistant::{AssistantApiError, ConfigStore};
use ari_skill_loader::manifest::ApiConfig;
use ari_skill_loader::wasm::{LogLevel, LogSink};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

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

/// Engine-internal continuation signal the assistant emits inline at the
/// end of an answer; parsed and stripped before the answer reaches the
/// user. See docs/superpowers/specs/2026-06-26-conversation-continuation-design.md.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ContinuationFlag {
    Continuation,
    New,
}

/// Split a trailing `[continuation]` / `[new]` marker off an assistant
/// answer. Matched case-insensitively, tolerant of surrounding quotes,
/// and only when it is the final bracketed token. Returns the cleaned
/// answer (trailing whitespace trimmed) and the flag; a missing or
/// unrecognised marker defaults to `Continuation` so context is kept.
pub(crate) fn parse_continuation_flag(raw: &str) -> (String, ContinuationFlag) {
    let trimmed = raw.trim_end();
    if let Some(open) = trimmed.rfind('[') {
        let inner = trimmed[open..]
            .trim_start_matches('[')
            .trim_end_matches(']')
            .trim()
            .trim_matches(|c| c == '\'' || c == '"')
            .trim()
            .to_ascii_lowercase();
        let flag = match inner.as_str() {
            "new" => Some(ContinuationFlag::New),
            "continuation" => Some(ContinuationFlag::Continuation),
            _ => None,
        };
        if let Some(flag) = flag {
            return (trimmed[..open].trim_end().to_string(), flag);
        }
    }
    (trimmed.to_string(), ContinuationFlag::Continuation)
}

/// The locale-appropriate version of [`FALLBACK_RESPONSE`]. Used when
/// every routing layer (skill regex → router → assistant) returned
/// nothing — engine surfaces this so the user gets a graceful "I'm
/// not sure" rather than silence. Falls back to the English
/// constant for unknown locales.
pub fn fallback_response_for(locale: &str) -> &'static str {
    match locale {
        "it" => "Scusa, non ho capito.",
        "es" => "Lo siento, no entendí.",
        "fr" => "Désolé, je n'ai pas compris.",
        "de" => "Entschuldigung, das habe ich nicht verstanden.",
        _ => FALLBACK_RESPONSE,
    }
}

/// How long a pending question stays answerable before a crash/missed cancel
/// is treated as abandoned. Safety net only — the frontend cancels eagerly.
const PENDING_TURN_TTL: Duration = Duration::from_secs(60);

/// Inactivity window after which a conversation's context is dropped and
/// the next assistant turn starts fresh.
const CONVERSATION_TTL: Duration = Duration::from_secs(90);
/// Most recent exchanges retained (10 role/content entries).
const MAX_CONVERSATION_TURNS: usize = 5;

/// Upper bound on stored personal facts. Bounds the recall block injected
/// into the assistant prompt; a capture at the cap evicts the oldest.
const MAX_REMEMBERED_FACTS: usize = 50;

/// Reserved `pending_turn` id for the engine-internal name-capture round-trip.
const NAME_CAPTURE_SENTINEL: &str = "__ari_name_capture";

/// A skill is awaiting the user's reply to a question it just asked.
#[derive(Clone, Debug)]
pub struct PendingTurn {
    pub skill_id: String,
    pub context: String,
    pub created_at: Instant,
}

/// One recorded assistant exchange: the user's query and the assistant's
/// answer (continuation marker already stripped).
#[derive(Clone, Debug)]
pub(crate) struct ConversationTurn {
    pub(crate) user: String,
    pub(crate) assistant: String,
}

/// Recent assistant conversation, in chronological order, plus the last
/// time any turn touched it (drives the inactivity TTL).
#[derive(Debug)]
struct ConversationBuffer {
    turns: Vec<ConversationTurn>,
    last_activity: Instant,
}

/// Localized acknowledgement spoken when the user cancels a pending question.
pub const CANCEL_ACK: &str = "Okay.";
pub fn cancel_ack_for(locale: &str) -> &'static str {
    match locale {
        "it" => "Va bene.",
        // es/fr/de: fall back to English until natively reviewed (see plan
        // Global Constraints — do not machine-translate).
        _ => CANCEL_ACK,
    }
}

/// Phrases that cancel a pending question. Compared against the
/// post-`normalize_input` utterance (lowercased, contractions expanded).
fn cancel_phrases(locale: &str) -> &'static [&'static str] {
    match locale {
        // it: needs native review (see plan Global Constraints).
        "it" => &["annulla", "lascia stare", "lascia perdere", "ferma", "ferma tutto", "stop"],
        _ => &["cancel", "never mind", "nevermind", "stop", "forget it"],
    }
}

fn is_cancel_phrase(normalized: &str, locale: &str) -> bool {
    cancel_phrases(locale).contains(&normalized)
}

// "Let's talk" mode entry/exit phrases. Matched against POST-normalised
// input (see normalize_input): English expands contractions, so "let's"
// arrives as "let us" and "that's" as "that is". Matched as WHOLE
// utterances (exact eq) so "stop the timer" never triggers an exit.
// it: DRAFT — needs native review (see plan Global Constraints).
fn enter_conversation_phrases(locale: &str) -> &'static [&'static str] {
    match locale {
        "it" => &[
            "parliamo",
            "chiacchieriamo",
            "conversiamo",
            "continua ad ascoltare",
            "inizia una conversazione",
        ],
        _ => &[
            "let us talk",
            "let us chat",
            "let us have a conversation",
            "keep listening",
            "start a conversation",
            // Clipped trigger: a mid-phrase pause in "let's [pause] talk" makes
            // the streaming recogniser endpoint on the stable partial "let us"
            // before "talk" arrives. Accept the bare truncation so entry still
            // fires. Safe because matching is whole-utterance exact: "let us go
            // to the shop" (≠ "let us") never enters.
            "let us",
        ],
    }
}

fn exit_conversation_phrases(locale: &str) -> &'static [&'static str] {
    match locale {
        "it" => &["basta", "arrivederci", "è tutto", "abbiamo finito", "fine conversazione"],
        // "stopped"/"stops": the STT model often transcribes a spoken "stop"
        // as one of these (verified on-device), so accept them as exits too.
        _ => &["stop", "stopped", "stops", "goodbye", "that is all", "we are done", "end conversation"],
    }
}

fn is_enter_conversation_phrase(normalized: &str, locale: &str) -> bool {
    enter_conversation_phrases(locale).contains(&normalized)
}

fn is_exit_conversation_phrase(normalized: &str, locale: &str) -> bool {
    exit_conversation_phrases(locale).contains(&normalized)
}

pub fn enter_conversation_ack_for(locale: &str) -> &'static str {
    match locale {
        "it" => "Va bene, ti ascolto.",
        _ => "Okay, I'm listening.",
    }
}

pub fn exit_conversation_ack_for(locale: &str) -> &'static str {
    match locale {
        "it" => "Va bene.",
        _ => "Okay.",
    }
}

/// Spoken when the user tries to enter "Let's Talk" mode while conversation
/// memory is switched off — the mode is meaningless without it.
pub fn conversation_memory_required_msg_for(locale: &str) -> &'static str {
    match locale {
        "it" => "Dovrai attivare la memoria delle conversazioni per la modalità “Parliamo”.",
        _ => "You'll need to turn on conversation memory for Let's Talk mode.",
    }
}

// ---- Remembered Facts (personal memory) command recognition ----
// All matching is against POST-normalize_input text (lowercase, contractions
// expanded, punctuation collapsed to spaces). Capture/forget are prefix
// commands; forget-all and the recall query are whole-utterance phrases.

/// Localized "remember …" capture prefixes. MUST be ordered longest-first
/// within a shared stem so the stripper doesn't clip a longer prefix (e.g.
/// "ricordati che " must precede "ricordati "). Each carries a trailing space
/// so a bare command word can't match.
fn capture_prefixes(locale: &str) -> &'static [&'static str] {
    match locale {
        // it: accepts both "ricorda che" and "ricordati che" (and the bare
        // "ricorda"/"ricordati") — the settings copy tells users "ricorda che…".
        "it" => &["ricordati che ", "ricorda che ", "ricordati ", "ricorda "],
        _ => &["remember that ", "remember "],
    }
}

/// Localized "forget …" (forget-one) prefixes, same ordering rule as
/// [`capture_prefixes`].
fn forget_prefixes(locale: &str) -> &'static [&'static str] {
    match locale {
        "it" => &["dimenticati che ", "dimentica che ", "dimenticati ", "dimentica "],
        _ => &["forget that ", "forget "],
    }
}

/// Strip the first matching command prefix and return the trimmed remainder,
/// or None when the utterance isn't such a command. A bare command (the input
/// is exactly a prefix with no remainder, e.g. "remember that" / "ricorda
/// che") returns None so it isn't captured as the fact "that" / "che".
fn strip_command_prefix<'a>(normalized: &'a str, prefixes: &[&str]) -> Option<&'a str> {
    if prefixes.iter().any(|p| normalized == p.trim_end()) {
        return None;
    }
    for p in prefixes {
        if let Some(rest) = normalized.strip_prefix(p) {
            let rest = rest.trim();
            if !rest.is_empty() {
                return Some(rest);
            }
        }
    }
    None
}

/// Strip a leading remember-command prefix and return the trimmed remainder,
/// or None when the utterance isn't a capture (or the remainder is empty).
fn remembered_fact_capture<'a>(normalized: &'a str, locale: &str) -> Option<&'a str> {
    strip_command_prefix(normalized, capture_prefixes(locale))
}

/// Strip a leading forget-command prefix and return the trimmed remainder,
/// or None when the utterance isn't a forget-one (or the remainder is empty).
fn remembered_fact_forget<'a>(normalized: &'a str, locale: &str) -> Option<&'a str> {
    strip_command_prefix(normalized, forget_prefixes(locale))
}

fn forget_all_phrases(locale: &str) -> &'static [&'static str] {
    match locale {
        "it" => &["dimentica tutto su di me", "dimentica tutto quello che sai su di me"],
        _ => &["forget everything about me", "forget everything you know about me"],
    }
}

fn is_forget_all_phrase(normalized: &str, locale: &str) -> bool {
    forget_all_phrases(locale).contains(&normalized)
}

fn recall_query_phrases(locale: &str) -> &'static [&'static str] {
    match locale {
        "it" => &["cosa ti ricordi di me", "cosa sai di me"],
        _ => &["what do you remember about me", "what do you know about me"],
    }
}

fn is_recall_query_phrase(normalized: &str, locale: &str) -> bool {
    recall_query_phrases(locale).contains(&normalized)
}

/// Spoken when a fact is stored.
pub fn fact_remembered_ack_for(locale: &str) -> &'static str {
    match locale {
        "it" => "Fatto, me ne ricorderò.",
        _ => "Got it — I'll remember that.",
    }
}

/// True when a stripped capture payload is a bare "my name" request (no name
/// given) — the trigger to elicit the name rather than store a fact.
fn is_bare_name_request(payload: &str, locale: &str) -> bool {
    match locale {
        "it" => payload == "il mio nome",
        _ => payload == "my name",
    }
}

/// Lead-in filler words skipped before the name token.
fn name_fillers(locale: &str) -> &'static [&'static str] {
    match locale {
        "it" => &["mi", "chiamo", "sono", "il", "mio", "nome", "è", "chiamami",
                  "puoi", "chiamarmi", "ecco", "beh"],
        _ => &["my", "name", "name's", "is", "the", "i", "i'm", "im", "am", "it",
               "it's", "its", "call", "me", "you", "can", "just", "um", "uh", "er", "well"],
    }
}

/// Obvious refusals — a reply of one of these yields no name.
fn name_refusals(locale: &str) -> &'static [&'static str] {
    match locale {
        "it" => &["no", "niente", "nulla", "boh"],
        _ => &["no", "nope", "nah", "nothing", "none", "nevermind"],
    }
}

/// Capitalise the first letter iff the token is all-lowercase (leave names
/// that already carry uppercase, e.g. "McDonald", intact).
fn capitalize_first(s: &str) -> String {
    if s.chars().any(|c| c.is_uppercase()) {
        return s.to_string();
    }
    let mut chars = s.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

/// Extract a display-ready first name from a raw reply, or None. Skips leading
/// filler words, rejects obvious refusals, takes the first name-like token.
fn extract_name(raw_input: &str, locale: &str) -> Option<String> {
    let fillers = name_fillers(locale);
    let refusals = name_refusals(locale);
    for word in raw_input.split_whitespace() {
        let token: String = word
            .chars()
            .filter(|c| c.is_alphabetic() || *c == '-' || *c == '\'')
            .collect();
        let token = token.trim_matches(|c| c == '-' || c == '\'').to_string();
        if token.is_empty() {
            continue;
        }
        let lower = token.to_lowercase();
        if fillers.contains(&lower.as_str()) {
            continue;
        }
        if refusals.contains(&lower.as_str()) {
            return None;
        }
        if token.chars().count() > 30 {
            return None;
        }
        return Some(capitalize_first(&token));
    }
    None
}

/// The locale-natural remembered fact for a captured name. Must stay readable
/// by the frontend's `detectUserName`.
fn name_fact_for(locale: &str, name: &str) -> String {
    match locale {
        "it" => format!("mi chiamo {name}"),
        _ => format!("my name is {name}"),
    }
}

/// Prompt asking the user for their name.
fn name_prompt_for(locale: &str) -> &'static str {
    match locale {
        "it" => "Come ti chiami?",
        _ => "What's your name?",
    }
}

/// Warm ack once the name is captured.
fn name_captured_ack_for(locale: &str, name: &str) -> String {
    match locale {
        "it" => format!("Piacere di conoscerti, {name}!"),
        _ => format!("Nice to meet you, {name}!"),
    }
}

/// Spoken when the reply held no usable name (single-shot; no retry).
fn name_not_caught_for(locale: &str) -> &'static str {
    match locale {
        "it" => "Scusa, non ho capito il nome — puoi dirmelo quando vuoi con \"mi chiamo …\".",
        _ => "Sorry, I didn't catch a name — you can tell me any time by saying \"my name is …\".",
    }
}

/// Spoken when a specific fact is forgotten.
pub fn fact_forgotten_ack_for(locale: &str) -> &'static str {
    match locale {
        "it" => "Va bene, l'ho dimenticato.",
        _ => "Okay, I've forgotten that.",
    }
}

/// Spoken when a forget command matched nothing.
pub fn fact_not_found_ack_for(locale: &str) -> &'static str {
    match locale {
        "it" => "Non me lo ricordavo comunque.",
        _ => "I didn't have that one.",
    }
}

/// Spoken when all facts are cleared.
pub fn facts_cleared_ack_for(locale: &str) -> &'static str {
    match locale {
        "it" => "Va bene, ho dimenticato tutto quello che sapevo su di te.",
        _ => "Okay, I've forgotten everything I knew about you.",
    }
}

/// Lead-in spoken before the recalled facts list.
pub fn recall_query_intro_for(locale: &str) -> &'static str {
    match locale {
        "it" => "Ecco cosa ricordo di te:",
        _ => "Here's what I remember about you:",
    }
}

/// Spoken for the recall query when nothing is stored.
pub fn no_facts_remembered_for(locale: &str) -> &'static str {
    match locale {
        "it" => "Non ricordo ancora nulla su di te.",
        _ => "I don't remember anything about you yet.",
    }
}

/// Strip an `await_reply` field from an action envelope, returning its
/// `context` string if present. Mirrors the `consult_assistant` strip pattern.
fn extract_await_reply(action: &mut serde_json::Value) -> Option<String> {
    let obj = action.as_object_mut()?;
    let v = obj.remove("await_reply")?;
    v.get("context").and_then(|c| c.as_str()).map(|s| s.to_string())
}

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
    #[cfg(feature = "llm")]
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
    /// The locale the loaded router's model was trained for. Always `Some`
    /// while `router` is `Some` — see [`Engine::set_router`].
    router_locale: Option<String>,
    /// Per-model confidence floor, from the model's own manifest
    /// (`min_confidence`, derived per-model by CI's floor sweep). `None`
    /// falls back to the compiled [`ari_core::MIN_ROUTER_CONFIDENCE`] —
    /// the pre-T5 behaviour, and the behaviour for manifests without the
    /// field. Calibration is per-MODEL: two same-corpus retrains have
    /// measured viable floors of -0.0233 and none-at-all, so a constant
    /// tuned on one roll silently mis-gates the next.
    router_confidence_floor: Option<f32>,
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
    /// Config store for reading skill settings from engine-internal paths
    /// (used by the fallback tier(s)' required-setting gate). `None` in
    /// bare/test engines that never wired one.
    config_store: Option<Arc<dyn ConfigStore>>,
    /// Per-instance multi-turn slot. When a skill's response carries an
    /// `await_reply { context }` field, the engine records `(skill_id,
    /// context)` here; the NEXT utterance bypasses routing and is handed to
    /// that skill's `execute_reply`. Guarded by a TTL so a missed cancel
    /// can't strand the slot. `&self` methods mutate it via the `Mutex`.
    pending_turn: Mutex<Option<PendingTurn>>,
    /// Passive multi-turn context for assistant-routed queries (behaviour
    /// B). Unlike `pending_turn`, this never hijacks routing — it only
    /// enriches assistant answers and is read non-destructively.
    conversation: Mutex<Option<ConversationBuffer>>,
    // "Let's talk" mode. `conversation_active` is owned by the frontend
    // (set via set_conversation_active) and mirrored here so the engine
    // can (a) interpret exit phrases only while in the mode and (b) record
    // skill turns into the buffer. enter/exit_signal are transient: set
    // during process_input_traced, read-and-cleared by the FFI layer.
    conversation_active: AtomicBool,
    enter_signal: AtomicBool,
    exit_signal: AtomicBool,
    // Behaviour B/C master switch, owned by the frontend (set via
    // set_conversation_memory_enabled). When false the engine keeps NO
    // conversation buffer and refuses "let's talk" entry.
    conversation_memory_enabled: AtomicBool,
    // Durable personal facts (explicit "remember that ..." captures). Canonical
    // runtime copy; the frontend owns disk persistence and hydrates this at
    // build via set_remembered_facts. facts_changed is a transient per-turn
    // signal (set on capture/forget, read-and-cleared by the FFI layer) telling
    // the frontend to re-read and persist.
    remembered_facts: Mutex<Vec<String>>,
    facts_changed: AtomicBool,
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
            router_locale: None,
            router_confidence_floor: None,
            log_sink: None,
            envelope_sink: None,
            named_assistants: Vec::new(),
            config_store: None,
            pending_turn: Mutex::new(None),
            conversation: Mutex::new(None),
            conversation_active: AtomicBool::new(false),
            enter_signal: AtomicBool::new(false),
            exit_signal: AtomicBool::new(false),
            conversation_memory_enabled: AtomicBool::new(true),
            remembered_facts: Mutex::new(Vec::new()),
            facts_changed: AtomicBool::new(false),
        }
    }

    fn set_pending_turn(&self, skill_id: &str, context: String) {
        *self.pending_turn.lock().expect("pending_turn poisoned") = Some(PendingTurn {
            skill_id: skill_id.to_string(),
            context,
            created_at: Instant::now(),
        });
    }

    /// Take the pending turn iff it exists and is within the TTL. A stale
    /// slot is taken (cleared) and `None` returned so input routes normally.
    fn take_pending_turn_if_fresh(&self) -> Option<PendingTurn> {
        let mut guard = self.pending_turn.lock().expect("pending_turn poisoned");
        let p = guard.take()?;
        if p.created_at.elapsed() < PENDING_TURN_TTL {
            Some(p)
        } else {
            None
        }
    }

    /// True when a fresh (within-TTL) pending turn is recorded. The FFI layer
    /// surfaces this as the `rearm` signal after each `process_input`.
    pub fn has_pending_turn(&self) -> bool {
        self.pending_turn
            .lock()
            .expect("pending_turn poisoned")
            .as_ref()
            .map(|p| p.created_at.elapsed() < PENDING_TURN_TTL)
            .unwrap_or(false)
    }

    /// Clear any pending turn. Called by the frontend on dismiss, listen
    /// timeout, and fresh wake word; a no-op when nothing is pending.
    pub fn clear_pending_turn(&self) {
        *self.pending_turn.lock().expect("pending_turn poisoned") = None;
    }

    pub fn set_conversation_active(&self, active: bool) {
        // "Let's talk" mode is meaningless without conversation memory; never
        // activate while memory is off, even if a caller asks (defence in depth
        // against a frontend/engine state mismatch).
        let active = active && self.is_conversation_memory_enabled();
        self.conversation_active.store(active, Ordering::SeqCst);
        // Leaving the mode (exit phrase, silence timeout, or error) must not
        // strand a skill's pending question.
        if !active {
            self.clear_pending_turn();
        }
    }

    fn is_conversation_active(&self) -> bool {
        self.conversation_active.load(Ordering::SeqCst)
    }

    /// Master switch for conversation memory (behaviours B and C). When set
    /// to `false` the current buffer is wiped immediately — flipping the
    /// toggle off must leave nothing retained in RAM.
    pub fn set_conversation_memory_enabled(&self, enabled: bool) {
        self.conversation_memory_enabled.store(enabled, Ordering::SeqCst);
        if !enabled {
            // Nothing must linger in RAM, and any active "let's talk" session
            // ends — talk mode can't run without memory.
            *self.conversation.lock().expect("conversation poisoned") = None;
            self.set_conversation_active(false);
        }
    }

    fn is_conversation_memory_enabled(&self) -> bool {
        self.conversation_memory_enabled.load(Ordering::SeqCst)
    }

    /// Replace the stored facts wholesale (frontend hydration + settings-screen
    /// edits). Applies dedup + cap; does NOT raise the changed signal, so
    /// hydration can't trigger a write-back loop.
    pub fn set_remembered_facts(&self, facts: Vec<String>) {
        let mut deduped: Vec<String> = Vec::with_capacity(facts.len().min(MAX_REMEMBERED_FACTS));
        for f in facts {
            if !deduped.contains(&f) {
                deduped.push(f);
            }
        }
        if deduped.len() > MAX_REMEMBERED_FACTS {
            let overflow = deduped.len() - MAX_REMEMBERED_FACTS;
            deduped.drain(0..overflow);
        }
        *self.remembered_facts.lock().expect("remembered_facts poisoned") = deduped;
    }

    /// Snapshot of the stored facts in insertion order (oldest first).
    pub fn remembered_facts(&self) -> Vec<String> {
        self.remembered_facts.lock().expect("remembered_facts poisoned").clone()
    }

    /// Store a new fact. Exact duplicate is a no-op; at the cap the oldest is
    /// evicted. Returns true iff the list changed (and then raises the signal).
    pub(crate) fn capture_fact(&self, text: &str) -> bool {
        let mut guard = self.remembered_facts.lock().expect("remembered_facts poisoned");
        if guard.iter().any(|f| f == text) {
            return false;
        }
        if guard.len() >= MAX_REMEMBERED_FACTS {
            guard.remove(0);
        }
        guard.push(text.to_string());
        drop(guard);
        self.facts_changed.store(true, Ordering::SeqCst);
        true
    }

    /// Remove the first fact exactly equal to `text`. Returns true iff a fact
    /// was removed (and then raises the signal).
    pub(crate) fn forget_fact(&self, text: &str) -> bool {
        let mut guard = self.remembered_facts.lock().expect("remembered_facts poisoned");
        if let Some(idx) = guard.iter().position(|f| f == text) {
            guard.remove(idx);
            drop(guard);
            self.facts_changed.store(true, Ordering::SeqCst);
            true
        } else {
            false
        }
    }

    /// Clear all facts. Returns true iff the list was non-empty (and then
    /// raises the signal).
    pub(crate) fn forget_all_facts(&self) -> bool {
        let mut guard = self.remembered_facts.lock().expect("remembered_facts poisoned");
        if guard.is_empty() {
            return false;
        }
        guard.clear();
        drop(guard);
        self.facts_changed.store(true, Ordering::SeqCst);
        true
    }

    /// Read-and-clear the per-turn "facts changed" signal. The FFI layer
    /// surfaces this after each process_input so the frontend persists.
    pub fn take_facts_changed_signal(&self) -> bool {
        self.facts_changed.swap(false, Ordering::SeqCst)
    }

    pub fn take_enter_signal(&self) -> bool {
        self.enter_signal.swap(false, Ordering::SeqCst)
    }

    pub fn take_exit_signal(&self) -> bool {
        self.exit_signal.swap(false, Ordering::SeqCst)
    }

    /// Recent assistant turns if the buffer is within its inactivity TTL,
    /// refreshing the timer so any turn (skill or assistant) keeps the
    /// conversation alive. A stale buffer is dropped and an empty list
    /// returned. Non-destructive for a fresh buffer — behaviour B is
    /// passive (contrast `take_pending_turn_if_fresh`).
    pub(crate) fn conversation_context(&self) -> Vec<ConversationTurn> {
        if !self.is_conversation_memory_enabled() {
            return Vec::new();
        }
        let mut guard = self.conversation.lock().expect("conversation poisoned");
        match guard.as_mut() {
            Some(buf) if buf.last_activity.elapsed() < CONVERSATION_TTL => {
                buf.last_activity = Instant::now();
                buf.turns.clone()
            }
            Some(_) => {
                *guard = None;
                Vec::new()
            }
            None => Vec::new(),
        }
    }

    /// Record an assistant answer (marker already stripped) as the latest
    /// turn. `New` starts a fresh conversation seeded with just this turn;
    /// `Continuation` appends and trims to the most recent
    /// `MAX_CONVERSATION_TURNS`.
    pub(crate) fn record_assistant_turn(&self, user: &str, assistant: &str, flag: ContinuationFlag) {
        if !self.is_conversation_memory_enabled() {
            return;
        }
        let turn = ConversationTurn { user: user.to_string(), assistant: assistant.to_string() };
        let mut guard = self.conversation.lock().expect("conversation poisoned");
        match (flag, guard.as_mut()) {
            (ContinuationFlag::Continuation, Some(buf)) => {
                buf.turns.push(turn);
                let len = buf.turns.len();
                if len > MAX_CONVERSATION_TURNS {
                    buf.turns.drain(0..len - MAX_CONVERSATION_TURNS);
                }
                buf.last_activity = Instant::now();
            }
            // `New`, or first turn of a fresh buffer: seed with this turn.
            _ => {
                *guard = Some(ConversationBuffer { turns: vec![turn], last_activity: Instant::now() });
            }
        }
    }

    // Records a skill's outgoing turn into the conversation buffer so it's
    // visible to the assistant on later turns. Only while "let's talk" mode
    // is active — outside it, skill wins stay transparent (behaviour B).
    fn record_skill_turn(&self, user: &str, response: &Response) {
        if !self.is_conversation_active() {
            return;
        }
        let spoken = match response {
            Response::Text(s) => s.clone(),
            Response::Action(v) => v
                .get("speak")
                .and_then(|x| x.as_str())
                .unwrap_or_default()
                .to_string(),
            Response::Binary { .. } => String::new(),
        };
        if spoken.is_empty() {
            return;
        }
        self.record_assistant_turn(user, &spoken, ContinuationFlag::Continuation);
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

    /// Replace the engine's snapshot of installed launchable apps. The frontend
    /// pushes this at build time and refreshes it on app install/uninstall; the
    /// `open` skill consults it so "open <app>" launches while "open <device>"
    /// (no matching app) steps aside for smart-home skills.
    pub fn set_installed_apps(&mut self, apps: Vec<AppEntry>) {
        self.ctx.installed_apps = apps;
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

    /// Swap the entire skill set in place, leaving every other field of the
    /// engine untouched — LLM, router, remembered facts, conversation-memory
    /// toggle, pending turn, let's-talk state, conversation buffer, sinks and
    /// named assistants all survive.
    ///
    /// This is what the host's reload path (community skill install / update /
    /// uninstall, and startup) must use. Rebuilding a fresh `Engine` to pick
    /// up a new skill set — the previous approach — silently discarded all of
    /// that runtime state, which the frontend hydrates once and does not
    /// re-apply after a reload.
    pub fn replace_skills(&mut self, skills: Vec<Box<dyn Skill>>) {
        self.skills = skills.into_iter().map(Arc::from).collect();
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

    /// Set the skill router (e.g. FunctionGemma) together with the locale
    /// its model was trained for. When set, the engine consults the router
    /// after keyword scoring fails, before falling through to the
    /// assistant — but only while that locale matches the active one, so a
    /// model left over from a language switch can never route the wrong
    /// language. Pass `None` to disable.
    ///
    /// Router and locale travel as one value deliberately: a router with no
    /// locale would be unusable, and this makes that state unrepresentable.
    pub fn set_router(&mut self, router: Option<(Box<dyn SkillRouter>, String)>) {
        match router {
            Some((router, locale)) => {
                self.router = Some(router);
                self.router_locale = Some(locale);
            }
            None => {
                self.router = None;
                self.router_locale = None;
            }
        }
    }

    /// Set the confidence floor the router's picks must clear, from the
    /// installed model's manifest (`min_confidence`). `None` reverts to
    /// the compiled [`ari_core::MIN_ROUTER_CONFIDENCE`] — correct for
    /// models whose manifest predates per-model floors. Call alongside
    /// [`Self::set_router`]: the floor belongs to the MODEL, not the
    /// device, so it must change when the loaded model does.
    pub fn set_router_confidence_floor(&mut self, floor: Option<f32>) {
        self.router_confidence_floor = floor;
    }

    /// The on-device router, but only when the active language is English.
    ///
    /// FunctionGemma is English-only: at 270M it routes other languages
    /// confidently but wrongly, so the engine never consults it outside `en`.
    /// Non-English routing is the cloud LLM's job (see `uses_assistant_routing`).
    /// The debug/eval entry points read `self.router` directly and are
    /// deliberately exempt, so the eval harness can still score any model.
    ///
    /// The `router_locale == ctx.locale` check stays as a second guard: the
    /// host swaps models asynchronously on a language switch, and routing one
    /// language through another's model would be confident nonsense.
    fn router_for_active_locale(&self) -> Option<&dyn SkillRouter> {
        if self.ctx.locale != "en" {
            return None;
        }
        let router = self.router.as_ref()?;
        if self.router_locale.as_deref() != Some(self.ctx.locale.as_str()) {
            return None;
        }
        Some(router.as_ref())
    }

    /// The effective floor: the loaded model's own, else the compiled
    /// constant.
    fn router_floor(&self) -> f32 {
        self.router_confidence_floor
            .unwrap_or(ari_core::MIN_ROUTER_CONFIDENCE)
    }

    /// Install the config store used to read skill settings from
    /// engine-internal paths (the fallback tier(s)).
    pub fn set_config_store(&mut self, store: Option<Arc<dyn ConfigStore>>) {
        self.config_store = store;
    }

    /// Settings-time invocation: route to a loaded skill by id and run its
    /// `settings_query`. Returns an error result if the skill isn't loaded.
    pub fn query_skill_setting(
        &self,
        skill_id: &str,
        field: &str,
        values_json: &str,
    ) -> ari_core::SettingsQueryResult {
        match self.skills.iter().find(|s| s.id() == skill_id) {
            Some(skill) => skill.settings_query(field, values_json),
            None => ari_core::SettingsQueryResult {
                ok: false,
                error: Some(format!("skill not loaded: {skill_id}")),
                options: Vec::new(),
                message: None,
                refresh: false,
            },
        }
    }

    /// Settings-time effectful invocation: route to a loaded skill by id and
    /// run its `settings_action` (e.g. a "Sign in" button). Reuses the
    /// `SettingsQueryResult` shape. Returns an error result if the skill isn't
    /// loaded.
    pub fn settings_action(
        &self,
        skill_id: &str,
        action: &str,
        values_json: &str,
    ) -> ari_core::SettingsQueryResult {
        match self.skills.iter().find(|s| s.id() == skill_id) {
            Some(skill) => skill.settings_action(action, values_json),
            None => ari_core::SettingsQueryResult {
                ok: false,
                error: Some(format!("skill not loaded: {skill_id}")),
                options: Vec::new(),
                message: None,
                refresh: false,
            },
        }
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

    /// True when a skill is usable for routing. A skill that declares a
    /// `fallback.requires_setting` is only "ready" while that setting holds
    /// a non-empty value — an unconfigured gated skill (e.g. Home Assistant
    /// before its `base_url` is set) is excluded from BOTH the keyword scorer
    /// and the router catalog so it can't shadow or destabilise routing for
    /// unrelated utterances. Skills without a `requires_setting` are always
    /// ready. Mirrors the gate the fallback-tier loop already applies.
    fn skill_is_ready(&self, skill: &dyn Skill) -> bool {
        let Some(tier) = skill.fallback_tier() else {
            return true;
        };
        let Some(key) = tier.requires_setting else {
            return true;
        };
        self.config_store
            .as_ref()
            .and_then(|cs| cs.get(skill.id(), &key))
            .map(|v| !v.trim().is_empty())
            .unwrap_or(false)
    }

    /// The skill catalogue handed to the semantic routers (FunctionGemma and
    /// the assistant-routing path). Excludes unready gated skills and
    /// keyword-only skills (`router_eligible() == false`) so neither router
    /// can claim a query that belongs to the keyword scorer or the assistant.
    fn router_catalog(&self) -> Vec<(String, String, String)> {
        self.skills
            .iter()
            .filter(|s| self.skill_is_ready(s.as_ref()))
            .filter(|s| s.router_eligible())
            .map(|s| {
                (
                    s.id().to_string(),
                    s.description().to_string(),
                    s.parameters_schema().to_string(),
                )
            })
            .collect()
    }

    /// Debug helper for the `/router` chat command: run ONLY the FunctionGemma
    /// router against `input` and report its raw pick — bypassing the keyword
    /// scorer, the assistant, and the confidence gate. Lets us inspect the
    /// on-device router even when normal routing goes through a cloud
    /// assistant (which skips FunctionGemma entirely).
    pub fn debug_route(&self, input: &str) -> String {
        let normalized = normalize_input(input.trim(), &self.ctx.locale);
        if normalized.is_empty() {
            return "empty input".to_string();
        }
        let Some(ref router) = self.router else {
            return "router not loaded".to_string();
        };
        let catalog = self.router_catalog();
        let result = router.route(&normalized, &catalog);
        let raw = router.last_raw_output().unwrap_or_default();
        let floor = self.router_floor();
        let gate = |c: f32| {
            if c < floor {
                "below threshold → would fall through"
            } else {
                "above threshold → would dispatch"
            }
        };
        let verdict = match &result {
            RouteResult::Skill { id, confidence } => {
                format!("skill={id} (confidence {confidence:.3}, {})", gate(*confidence))
            }
            RouteResult::SkillWithArgs { id, args_json, confidence } => format!(
                "skill={id} args={args_json} (confidence {confidence:.3}, {})",
                gate(*confidence)
            ),
            RouteResult::Action(a) => format!("action={a}"),
            RouteResult::NoMatch => "NoMatch → falls through to assistant".to_string(),
        };
        format!("input: {normalized:?}\n{verdict}\nraw: {raw}")
    }

    /// Router-only routing decision, mirroring the router branch of
    /// `process_input_traced`: run the router against the live catalogue and
    /// apply the confidence gate. Returns the skill id that WOULD be
    /// dispatched, or `None` for NoMatch / below-threshold (i.e. "falls
    /// through to the assistant"). Backs the routing-eval promotion gate;
    /// runs neither the keyword scorer nor the assistant.
    ///
    /// Deliberately NOT locale-gated, unlike production routing: this is an
    /// explicit "run this model" primitive, and route-eval grades a candidate
    /// model against a locale's eval bank by setting both to match. Gating
    /// here would only ever mask a harness misconfiguration as an abstention.
    pub fn route_decision(&self, input: &str) -> Option<String> {
        let normalized = normalize_input(input.trim(), &self.ctx.locale);
        if normalized.is_empty() {
            return None;
        }
        let router = self.router.as_ref()?;
        match router.route(&normalized, &self.router_catalog()) {
            RouteResult::Skill { id, confidence }
            | RouteResult::SkillWithArgs { id, confidence, .. } => {
                (confidence >= self.router_floor()).then_some(id)
            }
            RouteResult::Action(_) | RouteResult::NoMatch => None,
        }
    }

    /// The keyword scorer's verdict for `input`: the skill the ranking rounds
    /// would execute, or `None` when no skill clears its threshold (in which
    /// case production consults the router). Shares the exact ranking logic
    /// `process_input_traced` uses — deliberately not a replica, because a
    /// second copy would drift from the path it is supposed to describe.
    pub fn keyword_decision(&self, input: &str) -> Option<String> {
        let normalized = normalize_input(input.trim(), &self.ctx.locale);
        if normalized.is_empty() {
            return None;
        }
        self.keyword_winner(&normalized).map(|(s, _round)| s.skill_id)
    }

    /// Score every ready skill and run the ranking rounds. Returns the winning
    /// score and the round index it won in, or `None` if no round produced a
    /// winner. Single source of truth for keyword ranking.
    fn keyword_winner(&self, normalized: &str) -> Option<(SkillScore, usize)> {
        let scores = self.keyword_scores(normalized);
        Self::rank(&scores).map(|(s, r)| (s.clone(), r))
    }

    /// Per-skill keyword scores for an already-normalised input.
    fn keyword_scores(&self, normalized: &str) -> Vec<SkillScore> {
        self.skills
            .iter()
            .filter(|s| self.skill_is_ready(s.as_ref()))
            .map(|s| SkillScore {
                skill_id: s.id().to_string(),
                specificity: s.specificity(),
                score: s.score(normalized, &self.ctx),
            })
            .collect()
    }

    /// Run the ranking rounds over pre-computed scores. Returns the winner and
    /// its round index.
    fn rank(scores: &[SkillScore]) -> Option<(&SkillScore, usize)> {
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
                return Some((winner, round_idx));
            }
        }
        None
    }

    /// Raw router emission for analysis: the function the router picked and its
    /// confidence (mean per-token log-prob), BEFORE the confidence gate.
    /// `None` when the router abstained (emitted no function call). Used by
    /// route-eval's verbose mode to study whether misroutes sit at lower
    /// confidence than correct routes.
    pub fn route_raw(&self, input: &str) -> Option<(String, f32)> {
        let normalized = normalize_input(input.trim(), &self.ctx.locale);
        if normalized.is_empty() {
            return None;
        }
        let router = self.router.as_ref()?;
        match router.route(&normalized, &self.router_catalog()) {
            RouteResult::Skill { id, confidence }
            | RouteResult::SkillWithArgs { id, confidence, .. } => Some((id, confidence)),
            RouteResult::Action(_) | RouteResult::NoMatch => None,
        }
    }

    pub fn process_input_traced(&self, input: &str) -> (Response, Option<DebugTrace>) {
        let normalized = normalize_input(input.trim(), &self.ctx.locale);
        if normalized.is_empty() {
            return (
                Response::Text(fallback_response_for(&self.ctx.locale).to_string()),
                None,
            );
        }

        // "Let's talk" mode intercepts. These run BEFORE the pending-turn
        // reply path and all routing so entry/exit phrases are never
        // answered or mistaken for commands. Exit is gated on the active
        // flag so a bare "stop" stays a normal command outside the mode.
        // Reset the transient signals first so a stale one can't leak.
        self.enter_signal.store(false, Ordering::SeqCst);
        self.exit_signal.store(false, Ordering::SeqCst);
        if is_enter_conversation_phrase(&normalized, &self.ctx.locale) {
            // "Let's talk" is pointless without memory — guide the user to the
            // toggle instead of entering the mode. Note: still intercepted here
            // (before routing) so the phrase is never answered as a query.
            if !self.is_conversation_memory_enabled() {
                return (
                    Response::Text(
                        conversation_memory_required_msg_for(&self.ctx.locale).to_string(),
                    ),
                    None,
                );
            }
            self.enter_signal.store(true, Ordering::SeqCst);
            return (
                Response::Text(enter_conversation_ack_for(&self.ctx.locale).to_string()),
                None,
            );
        }
        if self.is_conversation_active()
            && is_exit_conversation_phrase(&normalized, &self.ctx.locale)
        {
            self.exit_signal.store(true, Ordering::SeqCst);
            // Bailing out of the mode abandons any skill sub-question.
            self.clear_pending_turn();
            return (
                Response::Text(exit_conversation_ack_for(&self.ctx.locale).to_string()),
                None,
            );
        }

        // Personal memory ("remember that ...") intercepts. Like the let's-talk
        // phrases, these run BEFORE routing so a command is never answered by
        // the assistant or mistaken for a skill query. Order: forget-all and
        // the recall query (whole-utterance) before the forget/remember
        // prefixes, so "forget everything about me" isn't parsed as
        // forget-fact("everything about me").
        // Guard: a whole-utterance cancel phrase ("forget it", "cancel",
        // "stop", "never mind"…) must fall through to the pending-turn cancel
        // path below — never be captured here. Only "forget it" actually
        // collides (it matches the `forget ` prefix → forget-fact("it")); the
        // others don't match any personal-memory prefix, so guarding the whole
        // block is behaviour-preserving for them and keeps the intent clear.
        if !is_cancel_phrase(&normalized, &self.ctx.locale) {
            if is_recall_query_phrase(&normalized, &self.ctx.locale) {
                let facts = self.remembered_facts();
                let spoken = if facts.is_empty() {
                    no_facts_remembered_for(&self.ctx.locale).to_string()
                } else {
                    let mut s = recall_query_intro_for(&self.ctx.locale).to_string();
                    for f in &facts {
                        s.push_str("\n- ");
                        s.push_str(f);
                    }
                    s
                };
                return (Response::Text(spoken), None);
            }
            if is_forget_all_phrase(&normalized, &self.ctx.locale) {
                let cleared = self.forget_all_facts();
                let spoken = if cleared {
                    facts_cleared_ack_for(&self.ctx.locale)
                } else {
                    no_facts_remembered_for(&self.ctx.locale)
                };
                return (Response::Text(spoken.to_string()), None);
            }
            if let Some(fact) = remembered_fact_forget(&normalized, &self.ctx.locale) {
                let removed = self.forget_fact(fact);
                let spoken = if removed {
                    fact_forgotten_ack_for(&self.ctx.locale)
                } else {
                    fact_not_found_ack_for(&self.ctx.locale)
                };
                return (Response::Text(spoken.to_string()), None);
            }
            if let Some(fact) = remembered_fact_capture(&normalized, &self.ctx.locale) {
                if is_bare_name_request(fact, &self.ctx.locale) {
                    self.set_pending_turn(NAME_CAPTURE_SENTINEL, String::new());
                    return (
                        Response::Text(name_prompt_for(&self.ctx.locale).to_string()),
                        None,
                    );
                }
                self.capture_fact(fact);
                return (
                    Response::Text(fact_remembered_ack_for(&self.ctx.locale).to_string()),
                    None,
                );
            }
        }

        // Behaviour B: read (and refresh) the conversation buffer for this
        // turn. Any turn refreshes the inactivity timer; only assistant
        // answers below extend the buffer. Pending-turn replies and skill
        // wins are transparent (refresh, never record).
        let conversation = self.conversation_context();
        let history = history_messages(&conversation);
        // Personal memory: the durable facts snapshot handed to the assistant
        // for THIS turn's free-text answer. Skill-serving assistant calls
        // (routing, named-assistant, layer-C) pass &[] — facts are user-answer
        // context only.
        let facts = self.remembered_facts();

        // Multi-turn: if a skill is awaiting a reply, this utterance belongs
        // to it — bypass all routing (scoring, router, assistant).
        if let Some(pending) = self.take_pending_turn_if_fresh() {
            // Verbal cancel escapes the pending turn.
            if is_cancel_phrase(&normalized, &self.ctx.locale) {
                return (
                    Response::Text(cancel_ack_for(&self.ctx.locale).to_string()),
                    None,
                );
            }
            // Engine-internal name capture: the reply is the user's name. Use the
            // raw `input` (not `normalized`) so the name's casing survives.
            if pending.skill_id == NAME_CAPTURE_SENTINEL {
                return match extract_name(input, &self.ctx.locale) {
                    Some(name) => {
                        self.capture_fact(&name_fact_for(&self.ctx.locale, &name));
                        (
                            Response::Text(name_captured_ack_for(&self.ctx.locale, &name)),
                            None,
                        )
                    }
                    None => (
                        Response::Text(name_not_caught_for(&self.ctx.locale).to_string()),
                        None,
                    ),
                };
            }
            // The reply goes ONLY to the asking skill. If the skill vanished
            // (e.g. a community-skill reload), fail cleanly.
            if let Some(skill) = self
                .skills
                .iter()
                .find(|s| s.id() == pending.skill_id)
                .cloned()
            {
                let resp = skill.execute_reply(&pending.context, &normalized, &self.ctx);
                // Route through the same chokepoint so a chained await_reply
                // (the skill asks again) re-arms.
                let resp = self.maybe_intercept_consult(skill, &normalized, resp);
                return (resp, None);
            }
            return (
                Response::Text(fallback_response_for(&self.ctx.locale).to_string()),
                None,
            );
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
            let response = dispatch_named_assistant(m.binding, &m.remainder, &self.ctx.locale, |level, msg| {
                self.log(level, msg)
            });
            return (response, Some(trace));
        }

        let scores: Vec<SkillScore> = self.keyword_scores(&normalized);

        let mut trace = DebugTrace {
            normalized_input: normalized.clone(),
            scores: scores.clone(),
            winner: None,
            round: None,
        };

        if let Some((winner, round_idx)) = Self::rank(&scores) {
            trace.winner = Some(winner.skill_id.clone());
            trace.round = Some(round_idx);

            let skill = self
                .skills
                .iter()
                .find(|s| s.id() == winner.skill_id)
                .unwrap()
                .clone();

            let response = skill.execute(&normalized, &self.ctx);
            let response = self.maybe_intercept_consult(skill, &normalized, response);
            return (response, Some(trace));
        }

        // No keyword match. Two routing tiers, in this order:
        //
        // 1. The on-device router (FunctionGemma), but only for English — at
        //    270M it routes other languages confidently but wrongly, so
        //    `router_for_active_locale` gates it to `en`. Sub-second, offline,
        //    free. It only speaks up when its confidence clears the floor its
        //    own manifest shipped, so the phrasings it isn't sure about cost
        //    nothing and fall to tier 2.
        // 2. The assistant — cloud (one call that routes OR answers) or the
        //    on-device LLM, which only answers. Slower and, for cloud, a
        //    network round-trip, but it handles what tier 1 declined and
        //    answers general questions.

        let skill_catalog = self.router_catalog();

        if let Some(router) = self.router_for_active_locale() {
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
                    if confidence < self.router_floor() {
                        self.log(
                            LogLevel::Info,
                            &format!(
                                "router: skipping skill={id} — confidence {confidence:.3} \
                                 below threshold {threshold:.3}; falling through to assistant",
                                threshold = self.router_floor(),
                            ),
                        );
                    } else if let Some(skill) = self.skills.iter().find(|s| s.id() == id).cloned() {
                        trace.winner = Some(format!("router:{id}"));
                        self.log(
                            LogLevel::Info,
                            &format!("router: dispatching skill={id} (confidence {confidence:.3})"),
                        );
                        let response = skill.execute(&normalized, &self.ctx);
                        let response = self.maybe_intercept_consult(skill, &normalized, response);
                        return (response, Some(trace));
                    }
                }
                RouteResult::SkillWithArgs {
                    ref id,
                    ref args_json,
                    confidence,
                } => {
                    if confidence < self.router_floor() {
                        self.log(
                            LogLevel::Info,
                            &format!(
                                "router: skipping skill={id} — confidence {confidence:.3} \
                                 below threshold {threshold:.3}; falling through to assistant",
                                threshold = self.router_floor(),
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
                        let response = self.maybe_intercept_consult(skill, &normalized, response);
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

        // Tier 2. Whether the assistant is asked to ROUTE what tier 1 left
        // behind (as opposed to only answering it) turns on ONE thing: is a
        // cloud assistant wired? Routing is cloud-only.
        //
        // - A cloud assistant (ChatGPT et al.) arbitrates reliably — it picks
        //   a skill or says NONE, so a general "what is X" falls through to
        //   the answer path instead of being force-mapped onto the nearest
        //   skill. For English that's the one-shot below (route AND answer in
        //   a single call); other languages get the two-step routing prompt.
        // - No cloud assistant, any locale: ask nothing. The on-device LLM is
        //   far too slow at routing (the catalogue prefill dominates) and, at
        //   its size, unreliable — so tier 1's verdict stands (English only)
        //   and anything left goes to the answer path, answered directly.
        let has_cloud_assistant =
            matches!(&self.active_assistant, Some(ActiveAssistant::Api { .. }));
        let use_assistant_routing = uses_assistant_routing(has_cloud_assistant);

        if use_assistant_routing && self.ctx.locale == "en" {
            // English + cloud assistant: ONE-SHOT route-or-answer — a single
            // call that either routes to a skill or answers directly, instead
            // of route-then-separate-answer. The combined prompt is English
            // only, so non-English keeps the translated two-step below.
            match self.route_or_answer(&normalized, &history, &facts) {
                Ok(RouteOrAnswer::Skill(id)) => {
                    if let Some(skill) = self.skills.iter().find(|s| s.id() == id).cloned() {
                        trace.winner = Some(format!("router:assistant:{id}"));
                        self.log(
                            LogLevel::Info,
                            &format!("router:assistant: one-shot routed skill={id}"),
                        );
                        let response = skill.execute(&normalized, &self.ctx);
                        let response = self.maybe_intercept_consult(skill, &normalized, response);
                        return (response, Some(trace));
                    }
                }
                Ok(RouteOrAnswer::Answer(text)) => {
                    let (clean, flag) = parse_continuation_flag(&text);
                    self.record_assistant_turn(&normalized, &clean, flag);
                    let label = match &self.active_assistant {
                        Some(ActiveAssistant::Api { skill_id, .. }) => format!("assistant:{skill_id}"),
                        _ => "assistant:one-shot".to_string(),
                    };
                    trace.winner = Some(label);
                    return (Response::Text(clean), Some(trace));
                }
                Err(reason) => {
                    self.log(
                        LogLevel::Warn,
                        &format!("router:assistant: one-shot failed: {reason}; falling through"),
                    );
                }
            }
            // Unknown id / call failed — fall through to the fallback /
            // assistant-answer path below.
        } else if use_assistant_routing {
            // Non-English: translated two-step routing prompt (FunctionGemma is
            // English-only). A general question falls through to the
            // assistant-answer path below.
            if let Some(picked_id) = self.try_assistant_route(&normalized, &skill_catalog) {
                if let Some(skill) = self
                    .skills
                    .iter()
                    .find(|s| s.id() == picked_id)
                    .cloned()
                {
                    trace.winner = Some(format!("router:assistant:{picked_id}"));
                    self.log(
                        LogLevel::Info,
                        &format!(
                            "router:assistant: dispatching skill={picked_id} (locale={})",
                            self.ctx.locale
                        ),
                    );
                    let response = skill.execute(&normalized, &self.ctx);
                    let response = self.maybe_intercept_consult(skill, &normalized, response);
                    return (response, Some(trace));
                }
            }
        }

        // Fallback tier(s). When the router + scorers all miss, forward the raw
        // utterance to any skill that declared `metadata.ari.fallback`. Gated
        // integrations (a `requires_setting` engaged only while configured, e.g.
        // Home Assistant) take precedence over unconditional catch-alls (the
        // `open` launcher's last-resort "couldn't find an app" reply), so a
        // configured smart-home forward beats the generic app miss. Stable sort
        // preserves registration order within each group. The first fallback
        // whose response is not `_ari_no_match` wins; otherwise we fall through
        // to the assistant below.
        let mut fallbacks: Vec<Arc<dyn Skill>> = self
            .skills
            .iter()
            .filter(|s| s.fallback_tier().is_some())
            .cloned()
            .collect();
        fallbacks.sort_by_key(|s| {
            // false (gated) sorts before true (unconditional).
            s.fallback_tier()
                .map(|t| t.requires_setting.is_none())
                .unwrap_or(true)
        });
        for skill in fallbacks {
            let tier = skill.fallback_tier().expect("filtered to Some above");
            if let Some(key) = &tier.requires_setting {
                let ready = self
                    .config_store
                    .as_ref()
                    .and_then(|cs| cs.get(skill.id(), key))
                    .map(|v| !v.trim().is_empty())
                    .unwrap_or(false);
                if !ready {
                    continue;
                }
            }
            let response = skill.execute(&normalized, &self.ctx);
            let fell_through = matches!(
                &response,
                Response::Action(v)
                    if v.get("_ari_no_match").and_then(|b| b.as_bool()).unwrap_or(false)
            );
            if !fell_through {
                trace.winner = Some(format!("fallback:{}", skill.id()));
                let response = self.maybe_intercept_consult(skill, &normalized, response);
                return (response, Some(trace));
            }
        }

        // No skill matched. Delegate to the active assistant, if any.
        match &self.active_assistant {
            #[cfg(feature = "llm")]
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
                    let result = llm.try_answer(&normalized, &catalog, &self.ctx.locale, &history, &facts);
                    match result {
                        Some(ari_llm::FallbackResult::DirectAnswer { text }) => {
                            let (clean, flag) = parse_continuation_flag(&text);
                            self.record_assistant_turn(&normalized, &clean, flag);
                            let preview: String = clean.chars().take(160).collect();
                            self.log(
                                LogLevel::Info,
                                &format!(
                                    "assistant:builtin: try_answer returned answer ({} bytes): {preview:?}",
                                    clean.len()
                                ),
                            );
                            trace.winner = Some("assistant:builtin".to_string());
                            return (Response::Text(clean), Some(trace));
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
                    &self.ctx.locale,
                    &history,
                    &facts,
                ) {
                    Ok(text) if !text.is_empty() => {
                        let (clean, flag) = parse_continuation_flag(&text);
                        self.record_assistant_turn(&normalized, &clean, flag);
                        trace.winner = Some(format!("assistant:{skill_id}"));
                        return (Response::Text(clean), Some(trace));
                    }
                    _ => {}
                }
            }
            None => {}
        }

        (
            Response::Text(fallback_response_for(&self.ctx.locale).to_string()),
            Some(trace),
        )
    }

    /// If the skill's response envelope carries a `consult_assistant`
    /// directive (Layer C v2), split it out: strip the directive from
    /// the phase-1 envelope we return synchronously, and spawn a
    /// background thread that runs the assistant round-trip and pushes
    /// the phase-2 envelope via [`EnvelopeSink`]. When anything is
    /// missing (no sink, malformed directive, etc.) the skill's first
    /// envelope is returned unchanged — no assistant call happens.
    /// Phase-5 non-English routing path. Build a routing prompt
    /// asking the active assistant LLM to pick a skill_id from the
    /// catalogue, dispatch to whichever backend the user has wired
    /// (cloud assistant API or on-device Gemma E2B/E4B), and parse
    /// the response back to a skill id.
    ///
    /// Returns `None` when:
    /// - No active assistant is configured (regex-only world; engine
    ///   surfaces FALLBACK_RESPONSE downstream).
    /// - The active assistant is Builtin but the LLM isn't loaded
    ///   (or the `llm` feature isn't compiled in).
    /// - The model call failed.
    /// - The model picked "NONE" or returned an unparseable response.
    /// - The model picked an id that isn't in the catalogue.
    fn try_assistant_route(
        &self,
        input: &str,
        skill_catalog: &[(String, String, String)],
    ) -> Option<String> {
        let prompt = build_assistant_routing_prompt(input, skill_catalog, &self.ctx.locale);
        let response = self.call_active_assistant(&prompt, &[], &[]).ok()?;
        let picked = parse_assistant_routing_response(&response, skill_catalog);
        if picked.is_none() {
            let preview: String = response.chars().take(120).collect();
            self.log(
                LogLevel::Info,
                &format!(
                    "router:assistant: no skill picked (response preview={preview:?})"
                ),
            );
        }
        picked
    }

    /// Send `prompt` to whichever assistant backend is active (cloud API or
    /// on-device LLM) and return its raw text reply. `None` when no assistant
    /// is configured, the on-device LLM isn't loaded, or the call failed.
    /// Shared by the routing prompt, the one-shot route-or-answer, and the
    /// debug commands.
    fn call_active_assistant(&self, prompt: &str, history: &[(String, String)], facts: &[String]) -> Result<String, String> {
        let fail = |reason: String| -> Result<String, String> {
            self.log(LogLevel::Warn, &format!("assistant: {reason}"));
            Err(reason)
        };
        match &self.active_assistant {
            Some(ActiveAssistant::Api {
                skill_id,
                config,
                config_store,
            }) => match ari_skill_loader::call_assistant_api(
                config,
                skill_id,
                config_store.as_ref(),
                prompt,
                &self.ctx.locale,
                history,
                facts,
            ) {
                Ok(text) => Ok(text),
                Err(e) => fail(format!("cloud call failed: {e}")),
            },
            #[cfg(feature = "llm")]
            Some(ActiveAssistant::Builtin { .. }) => {
                let Some(llm) = self.llm.as_ref() else {
                    return fail(
                        "on-device LLM not loaded — wrapper is None (freed under memory \
                         pressure, or never loaded). Nothing reloads it on demand."
                            .to_string(),
                    );
                };
                match llm.run_prompt(prompt) {
                    Ok(text) => Ok(ari_llm::strip_thinking(&text)),
                    Err(e) => fail(format!("on-device LLM error: {e}")),
                }
            }
            None => fail("no assistant configured".to_string()),
        }
    }

    /// One-shot routing: a single assistant call that either hands the request
    /// to a skill or answers it directly — folding the old two-step (route,
    /// then a separate answer call) into one round-trip. `input` must already
    /// be normalised. `None` when no assistant is configured or the call
    /// failed. English-instruction prompt for now; non-English callers should
    /// keep the translated two-step until a localised combined prompt exists.
    fn route_or_answer(&self, input: &str, history: &[(String, String)], facts: &[String]) -> Result<RouteOrAnswer, String> {
        let catalog = self.router_catalog();
        let prompt = build_combined_route_or_answer_prompt(input, &catalog);
        let response = self.call_active_assistant(&prompt, history, facts)?;
        Ok(parse_combined_response(&response, &catalog))
    }

    fn maybe_intercept_consult(
        &self,
        skill: Arc<dyn Skill>,
        user: &str,
        response: Response,
    ) -> Response {
        // Every return path funnels through this labelled block so the final
        // (possibly await_reply-stripped) response is recorded exactly once
        // below, just before we hand it back to the caller.
        let response = 'intercept: {
        let mut action = match response {
            Response::Action(v) => v,
            other => break 'intercept other,
        };

        // Enforce declared capabilities before anything downstream sees the
        // envelope: a skill that didn't declare `critical_alert` can't emit a
        // lock-screen takeover alert, no matter what it put in the JSON.
        let clamped =
            clamp_undeclared_critical_alerts(&mut action, skill.has_capability(CRITICAL_ALERT_CAP));
        if clamped > 0 {
            self.log(
                LogLevel::Warn,
                &format!(
                    "skill '{}' emitted {clamped} critical full-takeover alert(s) without declaring `{CRITICAL_ALERT_CAP}` — downgraded to high-priority",
                    skill.id(),
                ),
            );
        }

        // Multi-turn: a skill asking a question records a pending turn keyed
        // by its id, and the field is stripped so the frontend never sees it.
        if let Some(context) = extract_await_reply(&mut action) {
            self.set_pending_turn(skill.id(), context);
        }

        let directive_value = match action
            .as_object_mut()
            .and_then(|obj| obj.remove("consult_assistant"))
        {
            Some(v) => v,
            None => break 'intercept Response::Action(action),
        };

        let directive = match parse_consult_directive(&directive_value) {
            Some(d) => d,
            None => {
                self.log(
                    LogLevel::Warn,
                    "layer-c: consult_assistant directive malformed — ignoring, returning phase-1 envelope unchanged",
                );
                break 'intercept Response::Action(action);
            }
        };

        let sink = match self.envelope_sink.clone() {
            Some(s) => s,
            None => {
                self.log(
                    LogLevel::Warn,
                    "layer-c: consult_assistant requested but no envelope sink installed — phase-2 suppressed",
                );
                break 'intercept Response::Action(action);
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
        };

        // Visible skill turns: record the final response while in the mode.
        self.record_skill_turn(user, &response);
        response
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
/// Phase-5 routing prompt for the active assistant LLM. Asks the
/// model to pick a single skill_id from the catalogue (or "NONE" if
/// none fit). Locale is used to phrase the instructions in the
/// user's language so monolingual cloud assistants don't get
/// confused — the skill_ids themselves stay in their canonical
/// form (reverse-DNS, ASCII).
///
/// Cloud-LLM-friendly format: instructions first, catalogue as a
/// bulleted list, then the user's input fenced in quotes. Output
/// constraint kept minimal so we get a parseable response on the
/// first line even when the model insists on adding prose.
/// Whether the assistant is asked to ROUTE what the router left behind, rather
/// than only to answer it. Consulted AFTER the router has had its turn (see
/// `process_input_traced`), so this decides the second tier, not the first.
///
/// Routing is cloud-only. A cloud assistant arbitrates well — it tells a skill
/// request from a general question reliably, in any language. The on-device
/// Gemma cannot (it's ~22s to route, catalogue prefill dominates, and at its
/// size the picks aren't reliable), so when there's no cloud assistant nothing
/// is asked to route: the request falls to the answer path and is answered
/// directly. English still gets FunctionGemma as its offline routing tier;
/// other languages get keyword matching and direct answers offline.
fn uses_assistant_routing(has_cloud_assistant: bool) -> bool {
    has_cloud_assistant
}

/// Outcome of a one-shot [`Engine::route_or_answer`]: either the assistant
/// handed the request to a skill, or it answered directly.
enum RouteOrAnswer {
    Skill(String),
    Answer(String),
}

/// One-shot prompt: ask the assistant to EITHER hand the request to a skill
/// (by emitting a `SKILL: <id>` line) OR answer it directly — in a single
/// call. English-only for now (don't fabricate translations); the catalogue
/// ids are language-neutral. The personality/voice comes from the backend's
/// own system prompt (cloud) so we keep the instructions terse here.
fn build_combined_route_or_answer_prompt(input: &str, skills: &[(String, String, String)]) -> String {
    let mut catalog = String::new();
    for (id, description, _schema) in skills {
        catalog.push_str(&format!("- {id}: {description}\n"));
    }
    format!(
        "The user said:\n\"{input}\"\n\n\
         You can either hand this request to one of your skills, or answer it \
         yourself. Available skills:\n\
         {catalog}\n\
         If exactly one skill clearly handles the request, reply with ONLY this \
         line and nothing else:\n\
         SKILL: <skill id>\n\
         using the id exactly as written above. Otherwise, answer the user's \
         question directly and naturally — do not mention skills or this choice."
    )
}

/// Parse a one-shot response: a leading `SKILL: <known id>` line means route;
/// anything else is treated as a direct answer. Only the first non-empty line
/// is inspected for the sentinel, so a general answer that merely contains the
/// word "skill" isn't mistaken for a route.
fn parse_combined_response(response: &str, skills: &[(String, String, String)]) -> RouteOrAnswer {
    let known: std::collections::HashSet<&str> =
        skills.iter().map(|(id, _, _)| id.as_str()).collect();
    for line in response.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let sentinel = trimmed
            .strip_prefix("SKILL:")
            .or_else(|| trimmed.strip_prefix("skill:"));
        if let Some(rest) = sentinel {
            let id = rest.trim().trim_matches(|c: char| {
                c == '`' || c == '"' || c == '\'' || c == '*' || c == '_'
            });
            if known.contains(id) {
                return RouteOrAnswer::Skill(id.to_string());
            }
        }
        break;
    }
    RouteOrAnswer::Answer(response.trim().to_string())
}

fn build_assistant_routing_prompt(
    input: &str,
    skills: &[(String, String, String)],
    locale: &str,
) -> String {
    let header = match locale {
        "it" => "Sei un router di skill. L'utente ha detto:",
        "es" => "Eres un enrutador de habilidades. El usuario dijo:",
        "fr" => "Tu es un routeur de compétences. L'utilisateur a dit:",
        "de" => "Du bist ein Skill-Router. Der Benutzer sagte:",
        // Unknown locale falls back to English instructions — the
        // skill list itself is what the model needs to match against,
        // and skill ids are language-neutral.
        _ => "You are a skill router. The user said:",
    };
    let pick_line = match locale {
        "it" => "Scegli la skill che meglio corrisponde. Rispondi solo con l'id della skill sulla prima riga, oppure \"NONE\" se nessuna è appropriata. Non aggiungere spiegazioni.",
        "es" => "Elige la habilidad que mejor coincida. Responde solo con el id de la habilidad en la primera línea, o \"NONE\" si ninguna es apropiada. No expliques.",
        "fr" => "Choisis la compétence qui correspond le mieux. Réponds uniquement avec l'id de la compétence sur la première ligne, ou \"NONE\" si aucune ne convient. N'explique pas.",
        "de" => "Wähle den passenden Skill. Antworte nur mit der Skill-ID in der ersten Zeile, oder \"NONE\", wenn keiner passt. Keine Erklärungen.",
        _ => "Pick the skill that best matches. Respond with just the skill id on the first line, or \"NONE\" if none fit. Do not explain.",
    };

    let mut catalogue = String::new();
    for (id, description, _schema) in skills {
        catalogue.push_str(&format!("- {id}: {description}\n"));
    }
    format!("{header} \"{input}\"\n\n{catalogue}\n{pick_line}")
}

/// Parse the active assistant LLM's routing response. The contract
/// is "skill_id on the first line, or NONE". Real models routinely
/// violate it (markdown fences, leading explanation, trailing
/// punctuation), so the parser is forgiving:
///
/// 1. Look at every non-empty line in turn.
/// 2. Strip surrounding markdown (`*`, `_`, backticks, quotes).
/// 3. Match (case-sensitive) against any skill id in the catalogue.
/// 4. The first match wins.
///
/// Returns `None` for "NONE" responses, empty responses, or when no
/// line matches a known skill id. The engine treats `None` as
/// "fall through to the assistant-answer path" — the same behaviour
/// FunctionGemma's `RouteResult::NoMatch` produces for English.
fn parse_assistant_routing_response(
    response: &str,
    skills: &[(String, String, String)],
) -> Option<String> {
    let known_ids: std::collections::HashSet<&str> =
        skills.iter().map(|(id, _, _)| id.as_str()).collect();
    for raw_line in response.lines() {
        let cleaned = raw_line
            .trim()
            .trim_matches(|c: char| {
                c == '`'
                    || c == '*'
                    || c == '_'
                    || c == '"'
                    || c == '\''
                    || c == '.'
                    || c == ','
                    || c == ':'
                    || c == ';'
                    || c.is_whitespace()
            });
        if cleaned.is_empty() {
            continue;
        }
        if cleaned.eq_ignore_ascii_case("NONE") {
            return None;
        }
        if known_ids.contains(cleaned) {
            return Some(cleaned.to_string());
        }
    }
    None
}

fn dispatch_named_assistant<F: Fn(LogLevel, &str)>(
    binding: &NamedAssistantBinding,
    prompt: &str,
    locale: &str,
    log: F,
) -> Response {
    let display = assistant_display_name(&binding.skill_id);
    match ari_skill_loader::call_assistant_api(
        &binding.config,
        &binding.skill_id,
        binding.config_store.as_ref(),
        prompt,
        locale,
        &[],
        &[],
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
#[cfg(feature = "llm")]
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

/// The capability a skill must declare to emit a critical, full-takeover
/// alert — one that breaks through Do Not Disturb and takes over the
/// locked screen. Snake_case, matching the manifest spelling.
const CRITICAL_ALERT_CAP: &str = "critical_alert";

/// Demote any critical / full-takeover alert in an action envelope that the
/// emitting skill never declared. A skill without the `critical_alert`
/// capability may still raise alerts — they're just clamped to an ordinary
/// high-priority one: the screen-takeover flag is stripped and `critical`
/// urgency drops to `high`. Returns how many alerts were clamped (0 when the
/// skill declared the capability, or there was nothing to clamp) so the
/// caller can log it. The frontend trusts the envelope, so this is the
/// engine's job — a skill must not be able to do what it never declared.
fn clamp_undeclared_critical_alerts(action: &mut serde_json::Value, declared: bool) -> usize {
    if declared {
        return 0;
    }
    let alerts = match action.get_mut("alerts").and_then(|a| a.as_array_mut()) {
        Some(a) => a,
        None => return 0,
    };
    let mut clamped = 0;
    for alert in alerts.iter_mut() {
        let obj = match alert.as_object_mut() {
            Some(o) => o,
            None => continue,
        };
        let is_takeover = obj
            .get("full_takeover")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        let is_critical =
            obj.get("urgency").and_then(serde_json::Value::as_str) == Some("critical");
        if !is_takeover && !is_critical {
            continue;
        }
        obj.remove("full_takeover");
        if is_critical {
            obj.insert("urgency".into(), serde_json::Value::String("high".into()));
        }
        clamped += 1;
    }
    clamped
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
        #[cfg(feature = "llm")]
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
    let locale_for_thread = ctx.locale.clone();
    #[cfg(feature = "llm")]
    let llm_for_thread = llm.clone();
    std::thread::spawn(move || {
        #[cfg(feature = "llm")]
        let result = call_assistant_for_consult(
            &assistant_for_thread,
            &llm_for_thread,
            &prompt_for_thread,
            &locale_for_thread,
        );
        #[cfg(not(feature = "llm"))]
        let result = call_assistant_for_consult(
            &assistant_for_thread,
            &prompt_for_thread,
            &locale_for_thread,
        );
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

    // Async await_reply (a skill asking a follow-up AFTER a background
    // consult) is recognised and stripped to keep the pushed envelope
    // clean, but the Android mic re-arm for the async path is not yet
    // wired (see spec "Out of scope"). We deliberately do NOT set a
    // pending turn here: nothing can answer it while the overlay is gone.
    let mut continuation = continuation;
    if let Response::Action(ref mut v) = continuation {
        if extract_await_reply(v).is_some() {
            log(
                LogLevel::Info,
                "multi-turn: async await_reply stripped (async mic re-arm not yet wired)",
            );
        }
    }

    let envelope = match continuation {
        Response::Action(v) => {
            // Same capability enforcement as the phase-1 chokepoint — the
            // phase-2 envelope reaches the frontend via the sink, bypassing
            // maybe_intercept_consult, so it has to be clamped here too.
            let mut v = strip_nested_consult(v, &log);
            let clamped =
                clamp_undeclared_critical_alerts(&mut v, skill.has_capability(CRITICAL_ALERT_CAP));
            if clamped > 0 {
                log(
                    LogLevel::Warn,
                    &format!(
                        "skill '{skill_id}' emitted {clamped} critical full-takeover alert(s) without declaring `{CRITICAL_ALERT_CAP}` — downgraded to high-priority"
                    ),
                );
            }
            v
        }
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
    locale: &str,
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
                locale,
                &[],
                &[],
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
    locale: &str,
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
                locale,
                &[],
                &[],
            )
            .map_err(|e| e.to_string())?;
            if text.trim().is_empty() {
                Err("assistant returned empty response".into())
            } else {
                Ok(text)
            }
        }
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

/// Prior turns only — the current user turn is appended by the caller, last.
fn history_messages(turns: &[ConversationTurn]) -> Vec<(String, String)> {
    let mut msgs = Vec::with_capacity(turns.len() * 2);
    for t in turns {
        msgs.push(("user".to_string(), t.user.clone()));
        msgs.push(("assistant".to_string(), t.assistant.clone()));
    }
    msgs
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ari_core::FallbackTier;

    #[test]
    fn set_installed_apps_reaches_scoring() {
        struct Probe;
        impl Skill for Probe {
            fn id(&self) -> &str { "probe" }
            fn specificity(&self) -> Specificity { Specificity::Low }
            fn score(&self, _: &str, ctx: &SkillContext) -> f32 { ctx.installed_apps.len() as f32 }
            fn execute(&self, _: &str, _: &SkillContext) -> Response { Response::Text("x".to_string()) }
        }
        let mut engine = Engine::new();
        engine.register_skill(Box::new(Probe));
        engine.set_installed_apps(vec![
            ari_core::AppEntry { label: "Spotify".to_string(), package: "com.spotify.music".to_string() },
            ari_core::AppEntry { label: "Camera".to_string(), package: "com.android.camera".to_string() },
        ]);
        let scores = engine.keyword_scores("anything");
        let probe = scores.iter().find(|s| s.skill_id == "probe").expect("probe scored");
        assert_eq!(probe.score, 2.0, "the two pushed apps reach ctx.installed_apps");
    }

    #[test]
    fn bare_name_request_detected_per_locale() {
        assert!(is_bare_name_request("my name", "en"));
        assert!(!is_bare_name_request("my name is john", "en"));
        assert!(!is_bare_name_request("i like pizza", "en"));
        assert!(is_bare_name_request("il mio nome", "it"));
        assert!(!is_bare_name_request("my name", "it"));
    }

    #[test]
    fn extract_name_handles_phrasings_and_casing() {
        assert_eq!(extract_name("John", "en").as_deref(), Some("John"));
        assert_eq!(extract_name("i'm sarah", "en").as_deref(), Some("Sarah"));
        assert_eq!(extract_name("my name is bob", "en").as_deref(), Some("Bob"));
        assert_eq!(extract_name("John Smith", "en").as_deref(), Some("John"));
        assert_eq!(extract_name("sono Giovanni", "it").as_deref(), Some("Giovanni"));
        assert_eq!(extract_name("il mio nome è Anna", "it").as_deref(), Some("Anna"));
        assert_eq!(extract_name("nope", "en"), None);
        assert_eq!(extract_name("", "en"), None);
    }

    #[test]
    fn name_fact_is_locale_natural() {
        assert_eq!(name_fact_for("en", "Keith"), "my name is Keith");
        assert_eq!(name_fact_for("it", "Giovanni"), "mi chiamo Giovanni");
        assert_eq!(name_fact_for("fr", "Marie"), "my name is Marie");
    }

    #[test]
    fn name_strings_are_localised() {
        assert_eq!(name_prompt_for("en"), "What's your name?");
        assert_eq!(name_prompt_for("it"), "Come ti chiami?");
        assert_eq!(name_prompt_for("fr"), "What's your name?");
        assert_eq!(name_captured_ack_for("en", "Keith"), "Nice to meet you, Keith!");
        assert_eq!(name_captured_ack_for("it", "Giovanni"), "Piacere di conoscerti, Giovanni!");
    }

    #[test]
    fn remember_my_name_elicits_then_captures_en() {
        let mut engine = Engine::new();
        let (resp, _) = engine.process_input_traced("remember my name");
        assert!(matches!(resp, Response::Text(ref s) if s == name_prompt_for("en")),
            "bare request must ask for the name; got {resp:?}");
        assert!(engine.has_pending_turn(), "name capture must arm a pending turn");
        assert_eq!(engine.pending_turn.lock().unwrap().as_ref().unwrap().skill_id,
            NAME_CAPTURE_SENTINEL);

        let (resp, _) = engine.process_input_traced("i'm Keith");
        assert!(matches!(resp, Response::Text(ref s) if s == "Nice to meet you, Keith!"),
            "got {resp:?}");
        assert_eq!(engine.remembered_facts(), vec!["my name is Keith".to_string()]);
        assert!(!engine.has_pending_turn(), "slot must clear after capture");
    }

    #[test]
    fn remember_my_name_elicits_then_captures_it() {
        let mut engine = Engine::new();
        engine.set_locale("it".to_string());
        let (resp, _) = engine.process_input_traced("ricorda il mio nome");
        assert!(matches!(resp, Response::Text(ref s) if s == name_prompt_for("it")));
        let (resp, _) = engine.process_input_traced("sono Giovanni");
        assert!(matches!(resp, Response::Text(ref s) if s == "Piacere di conoscerti, Giovanni!"),
            "got {resp:?}");
        assert_eq!(engine.remembered_facts(), vec!["mi chiamo Giovanni".to_string()]);
    }

    #[test]
    fn name_reply_with_no_name_apologises_once() {
        let mut engine = Engine::new();
        let _ = engine.process_input_traced("remember my name");
        let (resp, _) = engine.process_input_traced("nope");
        assert!(matches!(resp, Response::Text(ref s) if s == name_not_caught_for("en")));
        assert!(engine.remembered_facts().is_empty(), "no name fact on unusable reply");
        assert!(!engine.has_pending_turn(), "single-shot: slot cleared, no retry");
    }

    #[test]
    fn cancel_during_name_capture_escapes() {
        let mut engine = Engine::new();
        let _ = engine.process_input_traced("remember my name");
        let (resp, _) = engine.process_input_traced("cancel");
        assert!(matches!(resp, Response::Text(ref s) if s == cancel_ack_for("en")));
        assert!(engine.remembered_facts().is_empty());
        assert!(!engine.has_pending_turn());
    }

    #[test]
    fn remember_my_name_with_name_stores_directly() {
        // Regression: the non-bare form must NOT elicit — it stores as-is.
        let mut engine = Engine::new();
        let (resp, _) = engine.process_input_traced("remember my name is Dave");
        assert!(matches!(resp, Response::Text(ref s) if s == fact_remembered_ack_for("en")));
        assert!(!engine.has_pending_turn());
        assert_eq!(engine.remembered_facts(), vec!["my name is dave".to_string()]);
    }

    struct MockSkill {
        id: &'static str,
        specificity: Specificity,
        fixed_score: f32,
        response: &'static str,
        requires_setting: Option<&'static str>, // NEW: gates readiness
    }

    impl Skill for MockSkill {
        fn id(&self) -> &str { self.id }
        fn specificity(&self) -> Specificity { self.specificity }
        fn score(&self, _input: &str, _ctx: &SkillContext) -> f32 { self.fixed_score }
        fn execute(&self, _input: &str, _ctx: &SkillContext) -> Response {
            Response::Text(self.response.to_string())
        }
        fn fallback_tier(&self) -> Option<FallbackTier> {
            self.requires_setting.map(|k| FallbackTier {
                requires_setting: Some(k.to_string()),
            })
        }
    }

    // --- Reload state preservation (P0) ---
    //
    // `replace_skills` is the in-place seam the FFI `reload_community_skills`
    // uses instead of building a throwaway Engine. Only the skill set may
    // change; every other field must survive. The old rebuild-and-swap silently
    // wiped remembered facts (and the first later "remember" clobbered the
    // on-disk list), re-enabled conversation memory, and dropped the pending
    // turn / let's-talk state / router / LLM. These lock that door shut.

    fn mock(id: &'static str) -> Box<dyn Skill> {
        Box::new(MockSkill {
            id,
            specificity: Specificity::High,
            fixed_score: 1.0,
            response: "ok",
            requires_setting: None,
        })
    }

    #[test]
    fn replace_skills_installs_the_new_set_and_drops_the_old() {
        let mut engine = Engine::new();
        engine.register_skill(mock("old"));
        engine.replace_skills(vec![mock("new_a"), mock("new_b")]);
        let ids: Vec<&str> = engine.skills.iter().map(|s| s.id()).collect();
        assert_eq!(ids, vec!["new_a", "new_b"], "old set replaced wholesale");
    }

    #[test]
    fn replace_skills_preserves_remembered_facts() {
        let mut engine = Engine::new();
        engine.set_remembered_facts(vec!["my name is Keith".to_string()]);
        engine.replace_skills(vec![mock("x")]);
        assert_eq!(
            engine.remembered_facts(),
            vec!["my name is Keith".to_string()],
            "reload must not wipe remembered facts",
        );
    }

    #[test]
    fn replace_skills_preserves_conversation_memory_toggle() {
        let mut engine = Engine::new();
        engine.set_conversation_memory_enabled(false);
        engine.replace_skills(vec![mock("x")]);
        assert!(
            !engine.is_conversation_memory_enabled(),
            "reload must not silently re-enable conversation memory",
        );
    }

    #[test]
    fn replace_skills_preserves_pending_turn() {
        let mut engine = Engine::new();
        engine.set_pending_turn("music", "pick a service".to_string());
        assert!(engine.has_pending_turn(), "sanity: pending turn armed");
        engine.replace_skills(vec![mock("x")]);
        assert!(
            engine.has_pending_turn(),
            "reload must not strand an in-flight pending turn",
        );
    }

    #[test]
    fn replace_skills_preserves_lets_talk_active_state() {
        let mut engine = Engine::new();
        engine.set_conversation_active(true); // memory defaults on, so this sticks
        assert!(
            engine.conversation_active.load(Ordering::SeqCst),
            "sanity: let's-talk active",
        );
        engine.replace_skills(vec![mock("x")]);
        assert!(
            engine.conversation_active.load(Ordering::SeqCst),
            "reload must not drop an active let's-talk session",
        );
    }

    // --- Multi-turn pending-turn ---

    /// Asks a question on first input; on a reply (reserved `_ari_reply`
    /// envelope) it plays the chosen service. Mirrors the music skill shape.
    struct AskingSkill;
    impl Skill for AskingSkill {
        fn id(&self) -> &str { "asker" }
        fn specificity(&self) -> Specificity { Specificity::High }
        fn score(&self, _: &str, _: &SkillContext) -> f32 { 1.0 }
        fn execute(&self, input: &str, _: &SkillContext) -> Response {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(input) {
                if let Some(reply) = v.get("_ari_reply") {
                    let ctx = reply["context"].as_str().unwrap_or("");
                    let text = reply["text"].as_str().unwrap_or("");
                    // Echo what we received as a plain action; no await_reply.
                    return Response::Action(serde_json::json!({
                        "v": 1, "speak": format!("ctx={ctx} text={text}")
                    }));
                }
            }
            // First turn: ask, and request a reply.
            Response::Action(serde_json::json!({
                "v": 1,
                "speak": "which service?",
                "await_reply": { "context": "Q1" }
            }))
        }
    }

    #[test]
    fn question_sets_pending_turn_and_strips_await_reply() {
        let mut engine = Engine::new();
        engine.register_skill(Box::new(AskingSkill));
        let (resp, _) = engine.process_input_traced("play music");
        // await_reply must be stripped from the envelope the frontend sees.
        match resp {
            Response::Action(v) => {
                assert!(v.get("await_reply").is_none(), "await_reply must be stripped");
                assert_eq!(v["speak"], "which service?");
            }
            other => panic!("expected Action, got {other:?}"),
        }
        assert!(engine.has_pending_turn(), "a pending turn must be recorded");
        let pending = engine.pending_turn.lock().unwrap().clone().unwrap();
        assert_eq!(pending.skill_id, "asker");
        assert_eq!(pending.context, "Q1");
    }

    #[test]
    fn reply_bypasses_routing_and_reaches_asking_skill() {
        let mut engine = Engine::new();
        engine.register_skill(Box::new(AskingSkill));
        let _ = engine.process_input_traced("play music"); // arm
        let (resp, skill_id_trace) = engine.process_input_with_skill("spotify");
        match resp {
            Response::Action(v) => assert_eq!(v["speak"], "ctx=Q1 text=spotify"),
            other => panic!("expected Action, got {other:?}"),
        }
        let _ = skill_id_trace; // skill_id attribution covered separately
        assert!(!engine.has_pending_turn(), "slot must clear after the reply");
    }

    #[test]
    fn cancel_word_clears_pending_and_acks() {
        let mut engine = Engine::new();
        engine.register_skill(Box::new(AskingSkill));
        let _ = engine.process_input_traced("play music"); // arm
        let (resp, _) = engine.process_input_traced("never mind");
        assert!(matches!(resp, Response::Text(ref s) if s == cancel_ack_for("en")));
        assert!(!engine.has_pending_turn(), "cancel must clear the slot");
    }

    #[test]
    fn forget_it_cancels_pending_turn_not_personal_memory() {
        // Regression: "forget it" is a registered cancel phrase. It must reach
        // the pending-turn cancel path, NOT be captured by the personal-memory
        // forget intercept (which would strip "forget " → "it", find no match,
        // ack "I didn't have that one." and leave the pending turn armed).
        let mut engine = Engine::new();
        engine.register_skill(Box::new(AskingSkill));
        let _ = engine.process_input_traced("play music"); // arm
        assert!(engine.has_pending_turn(), "pending turn must be armed");
        let (resp, _) = engine.process_input_traced("forget it");
        assert!(
            matches!(resp, Response::Text(ref s) if s == cancel_ack_for("en")),
            "\"forget it\" must speak the cancel ack, not the personal-memory not-found ack; got {resp:?}"
        );
        assert!(!engine.has_pending_turn(), "cancel must consume the pending turn");
    }

    #[test]
    fn expired_pending_turn_is_ignored_and_input_routes_normally() {
        use std::time::{Duration, Instant};
        let mut engine = Engine::new();
        engine.register_skill(Box::new(AskingSkill));
        // Manually plant a stale pending turn (older than the TTL).
        *engine.pending_turn.lock().unwrap() = Some(PendingTurn {
            skill_id: "asker".to_string(),
            context: "Q1".to_string(),
            created_at: Instant::now()
                .checked_sub(Duration::from_secs(120))
                .expect("clock supports subtraction"),
        });
        assert!(!engine.has_pending_turn(), "stale slot must read as absent");
        // "play music" routes normally → asks again (fresh pending).
        let (resp, _) = engine.process_input_traced("play music");
        assert!(matches!(resp, Response::Action(_)));
        assert!(engine.has_pending_turn());
    }

    // --- Visible skill turns during let's-talk mode ---

    /// Minimal keyword skill returning a spoken `Response::Text` on "hello".
    /// Mirrors the `AskingSkill` double: fixed win, plain text out — so the
    /// assertion can be on the conversation buffer, not the skill.
    struct GreetSkill;
    impl Skill for GreetSkill {
        fn id(&self) -> &str { "greet" }
        fn specificity(&self) -> Specificity { Specificity::High }
        fn score(&self, input: &str, _: &SkillContext) -> f32 {
            if input == "hello" { 1.0 } else { 0.0 }
        }
        fn execute(&self, _: &str, _: &SkillContext) -> Response {
            Response::Text("Hi there.".to_string())
        }
    }

    #[test]
    fn skill_turn_recorded_only_when_conversation_active() {
        let mut engine = Engine::new();
        engine.register_skill(Box::new(GreetSkill)); // existing test skill, keyword "hello"

        // Inactive: skill win is transparent (not recorded).
        let _ = engine.process_input_traced("hello");
        assert_eq!(engine.conversation_context().len(), 0, "inactive: no record");

        // Active: skill win is recorded as (user, spoken).
        engine.set_conversation_active(true);
        let _ = engine.process_input_traced("hello");
        let turns = engine.conversation_context();
        assert_eq!(turns.len(), 1, "active: skill turn recorded");
        assert_eq!(turns[0].user, "hello");
        assert_eq!(turns[0].assistant, "Hi there.", "records the spoken response");
    }

    // --- Router catalogue eligibility ---

    /// A router that records the skill catalogue it was handed and then
    /// declines to route, so the test can inspect exactly which skills the
    /// engine offered the model.
    struct CatalogCapturingRouter {
        seen: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    }

    impl SkillRouter for CatalogCapturingRouter {
        fn route(&self, _input: &str, skills: &[(String, String, String)]) -> RouteResult {
            *self.seen.lock().unwrap() = skills.iter().map(|(id, _, _)| id.clone()).collect();
            RouteResult::NoMatch
        }
    }

    #[test]
    fn router_ineligible_skill_is_excluded_from_catalogue() {
        use std::sync::{Arc, Mutex};

        /// A keyword-only skill (like search): scores nothing here and opts
        /// out of the router catalogue.
        struct KeywordOnlySkill;
        impl Skill for KeywordOnlySkill {
            fn id(&self) -> &str { "keyword_only" }
            fn description(&self) -> &str { "keyword only" }
            fn specificity(&self) -> Specificity { Specificity::Low }
            fn score(&self, _: &str, _: &SkillContext) -> f32 { 0.0 }
            fn router_eligible(&self) -> bool { false }
            fn execute(&self, _: &str, _: &SkillContext) -> Response {
                Response::Text("keyword".into())
            }
        }

        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut engine = Engine::new();
        engine.register_skill(Box::new(KeywordOnlySkill));
        engine.register_skill(Box::new(MockSkill {
            id: "eligible", specificity: Specificity::Low, fixed_score: 0.0,
            response: "eligible", requires_setting: None,
        }));
        engine.set_router(Some((Box::new(CatalogCapturingRouter { seen: seen.clone() }), "en".to_string())));

        // Neither skill scores, so the router runs and we can inspect the
        // catalogue it received.
        let _ = engine.process_input_traced(
            "what is the capital city of the united arab emirates",
        );

        let seen = seen.lock().unwrap();
        assert!(
            seen.contains(&"eligible".to_string()),
            "router-eligible skill must be offered to the router, got {seen:?}"
        );
        assert!(
            !seen.contains(&"keyword_only".to_string()),
            "router-ineligible skill must be filtered out, got {seen:?}"
        );
    }

    // --- Router locale gating and precedence ---

    /// A router that always claims `target` at maximum confidence, so a test
    /// can prove the engine consulted it by observing the skill it dispatched.
    struct AlwaysRoutesTo {
        target: &'static str,
    }

    impl SkillRouter for AlwaysRoutesTo {
        fn route(&self, _input: &str, _skills: &[(String, String, String)]) -> RouteResult {
            RouteResult::Skill {
                id: self.target.to_string(),
                confidence: 0.0,
            }
        }
    }

    /// A skill that never scores, so only a router can dispatch it.
    fn unreachable_by_keyword(id: &'static str, response: &'static str) -> MockSkill {
        MockSkill {
            id,
            specificity: Specificity::Low,
            fixed_score: 0.0,
            response,
            requires_setting: None,
        }
    }

    /// A cloud assistant pointed at a closed port. Present so
    /// `has_cloud_assistant` is true; any actual call fails immediately
    /// rather than reaching the network.
    fn unreachable_cloud_assistant() -> ActiveAssistant {
        use ari_skill_loader::assistant::MemoryConfigStore;
        use ari_skill_loader::manifest::{AuthScheme, LocalizedPrompt, RequestFormat};

        ActiveAssistant::Api {
            skill_id: "test.assistant".to_string(),
            config: ApiConfig {
                endpoint: Some("http://127.0.0.1:1/v1/chat".to_string()),
                endpoint_config_key: None,
                default_endpoint: None,
                auth: AuthScheme::None,
                auth_header: None,
                auth_config_key: None,
                model_config_key: None,
                default_model: "test-model".to_string(),
                system_prompt: LocalizedPrompt::from_english("test".to_string()),
                request_format: RequestFormat::Openai,
                response_path: "choices.0.message.content".to_string(),
                api_version: None,
                api_version_header: None,
                max_tokens: 64,
                temperature: 0.0,
            },
            config_store: Arc::new(MemoryConfigStore::new()),
        }
    }

    #[test]
    fn router_is_english_only_even_with_a_matching_locale_model() {
        use std::sync::{Arc, Mutex};

        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut engine = Engine::new();
        engine.set_locale("it".to_string());
        engine.register_skill(Box::new(unreachable_by_keyword("meteo", "Sole.")));
        // An Italian-locale model is loaded and matches the active locale...
        engine.set_router(Some((
            Box::new(CatalogCapturingRouter { seen: seen.clone() }),
            "it".to_string(),
        )));

        let (response, _) = engine.process_input_traced("che tempo fa");

        // ...but FunctionGemma is English-only, so it is never consulted.
        assert!(
            seen.lock().unwrap().is_empty(),
            "FunctionGemma must not be consulted for a non-English locale, got {:?}",
            seen.lock().unwrap()
        );
        match response {
            Response::Text(t) => assert_eq!(t, fallback_response_for("it")),
            other => panic!("expected the Italian fallback, got {other:?}"),
        }
    }

    #[test]
    fn router_is_skipped_when_its_model_is_for_another_locale() {
        use std::sync::{Arc, Mutex};

        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut engine = Engine::new();
        engine.set_locale("it".to_string());
        engine.register_skill(Box::new(unreachable_by_keyword("meteo", "Sole.")));
        engine.set_router(Some((
            Box::new(CatalogCapturingRouter { seen: seen.clone() }),
            "en".to_string(),
        )));

        let (response, _) = engine.process_input_traced("che tempo fa");

        let seen = seen.lock().unwrap();
        assert!(
            seen.is_empty(),
            "an English model must never be asked to route Italian, got {seen:?}"
        );
        match response {
            Response::Text(t) => assert_eq!(t, fallback_response_for("it")),
            other => panic!("expected the Italian fallback, got {other:?}"),
        }
    }

    #[test]
    fn router_outranks_a_cloud_assistant_when_a_matching_model_is_loaded() {
        let mut engine = Engine::new();
        engine.register_skill(Box::new(unreachable_by_keyword("weather", "Sunny.")));
        engine.set_active_assistant(Some(unreachable_cloud_assistant()));
        engine.set_router(Some((
            Box::new(AlwaysRoutesTo { target: "weather" }),
            "en".to_string(),
        )));

        let (response, trace) = engine.process_input_traced("is it raining out there");

        match response {
            Response::Text(t) => assert_eq!(t, "Sunny."),
            other => panic!("expected the routed skill's text, got {other:?}"),
        }
        assert_eq!(
            trace.unwrap().winner,
            Some("router:weather".to_string()),
            "the on-device router must win before the cloud one-shot is called"
        );
    }

    #[test]
    fn unloading_the_router_clears_its_locale() {
        let mut engine = Engine::new();
        engine.register_skill(Box::new(unreachable_by_keyword("weather", "Sunny.")));
        engine.set_router(Some((
            Box::new(AlwaysRoutesTo { target: "weather" }),
            "en".to_string(),
        )));
        engine.set_router(None);

        let (response, _) = engine.process_input_traced("is it raining out there");

        match response {
            Response::Text(t) => assert_eq!(
                t,
                fallback_response_for("en"),
                "no router loaded means nothing routes"
            ),
            other => panic!("expected the fallback, got {other:?}"),
        }
    }

    // --- Readiness gate (skill_is_ready) ---

    #[test]
    fn unconfigured_gated_skill_is_excluded_from_scorer() {
        use ari_skill_loader::assistant::MemoryConfigStore;
        use std::sync::Arc;

        let mut engine = Engine::new();
        // A high-specificity skill that would win at its score, but is gated
        // on `base_url` which is NOT set in the config store.
        engine.register_skill(Box::new(MockSkill {
            id: "gated", specificity: Specificity::High, fixed_score: 1.0,
            response: "gated-ran", requires_setting: Some("base_url"),
        }));
        // Empty config store → "base_url" unset → skill not ready.
        engine.set_config_store(Some(Arc::new(MemoryConfigStore::new())));

        let (resp, trace) = engine.process_input_traced("test");
        // The gated skill must NOT win the scorer round even at score 1.0.
        let trace = trace.unwrap();
        assert_eq!(trace.winner, None, "unready gated skill must not win scoring");
        assert!(matches!(resp, Response::Text(ref s) if s == FALLBACK_RESPONSE));
    }

    #[test]
    fn configured_gated_skill_participates_in_scorer() {
        use ari_skill_loader::assistant::MemoryConfigStore;
        use std::sync::Arc;

        let mut engine = Engine::new();
        engine.register_skill(Box::new(MockSkill {
            id: "gated", specificity: Specificity::High, fixed_score: 0.85,
            response: "gated-ran", requires_setting: Some("base_url"),
        }));
        let mut store = MemoryConfigStore::new();
        store.set("gated", "base_url", "http://homeassistant.local:8123");
        engine.set_config_store(Some(Arc::new(store)));

        let (resp, trace) = engine.process_input_traced("test");
        assert_eq!(trace.unwrap().winner.as_deref(), Some("gated"));
        assert!(matches!(resp, Response::Text(ref s) if s == "gated-ran"));
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
            requires_setting: None,
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
            requires_setting: None,
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
            requires_setting: None,
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
            requires_setting: None,
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
            requires_setting: None,
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
            requires_setting: None,
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
            requires_setting: None,
        }));
        engine.register_skill(Box::new(MockSkill {
            id: "low", specificity: Specificity::Low, fixed_score: 0.95, response: "low wins",
            requires_setting: None,
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
            requires_setting: None,
        }));
        engine.register_skill(Box::new(MockSkill {
            id: "b", specificity: Specificity::High, fixed_score: 0.92, response: "b",
            requires_setting: None,
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
            requires_setting: None,
        }));
        engine.register_skill(Box::new(MockSkill {
            id: "b", specificity: Specificity::Medium, fixed_score: 0.5, response: "b",
            requires_setting: None,
        }));
        engine.register_skill(Box::new(MockSkill {
            id: "c", specificity: Specificity::Low, fixed_score: 0.1, response: "c",
            requires_setting: None,
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
            requires_setting: None,
        }));
        let (_, trace) = engine.process_input_traced("What's the TIME?!");
        assert_eq!(trace.unwrap().normalized_input, "what is the time");
    }

    #[test]
    fn trace_reports_no_winner_when_no_match() {
        let mut engine = Engine::new();
        engine.register_skill(Box::new(MockSkill {
            id: "x", specificity: Specificity::High, fixed_score: 0.1, response: "x",
            requires_setting: None,
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

    // --- critical_alert capability enforcement ---

    /// Like ActionSkill, but declares the `critical_alert` capability — the
    /// stand-in for the timer skill.
    struct CapableAlertSkill {
        id: &'static str,
        action: serde_json::Value,
    }

    impl Skill for CapableAlertSkill {
        fn id(&self) -> &str { self.id }
        fn has_capability(&self, name: &str) -> bool { name == CRITICAL_ALERT_CAP }
        fn specificity(&self) -> Specificity { Specificity::High }
        fn score(&self, _input: &str, _ctx: &SkillContext) -> f32 { 0.95 }
        fn execute(&self, _input: &str, _ctx: &SkillContext) -> Response {
            Response::Action(self.action.clone())
        }
    }

    fn critical_takeover_envelope() -> serde_json::Value {
        serde_json::json!({
            "v": 1,
            "alerts": [{
                "id": "t", "title": "Timer", "urgency": "critical", "full_takeover": true
            }]
        })
    }

    #[test]
    fn clamp_is_noop_when_skill_declared_capability() {
        let mut env = critical_takeover_envelope();
        let before = env.clone();
        assert_eq!(clamp_undeclared_critical_alerts(&mut env, true), 0);
        assert_eq!(env, before);
    }

    #[test]
    fn clamp_strips_takeover_and_lowers_critical_when_undeclared() {
        let mut env = critical_takeover_envelope();
        assert_eq!(clamp_undeclared_critical_alerts(&mut env, false), 1);
        let alert = &env["alerts"][0];
        assert!(alert.get("full_takeover").is_none(), "takeover flag must be stripped");
        assert_eq!(alert["urgency"], "high");
        assert_eq!(alert["title"], "Timer", "non-privileged fields survive untouched");
    }

    #[test]
    fn clamp_lowers_critical_without_takeover() {
        let mut env = serde_json::json!({ "alerts": [{"id":"a","urgency":"critical"}] });
        assert_eq!(clamp_undeclared_critical_alerts(&mut env, false), 1);
        assert_eq!(env["alerts"][0]["urgency"], "high");
    }

    #[test]
    fn clamp_strips_takeover_on_non_critical_alert() {
        // full_takeover is a privilege in its own right, even at high urgency.
        let mut env = serde_json::json!({ "alerts": [{"id":"a","urgency":"high","full_takeover":true}] });
        assert_eq!(clamp_undeclared_critical_alerts(&mut env, false), 1);
        let alert = &env["alerts"][0];
        assert!(alert.get("full_takeover").is_none());
        assert_eq!(alert["urgency"], "high");
    }

    #[test]
    fn clamp_leaves_ordinary_alerts_untouched() {
        let mut env = serde_json::json!({ "alerts": [{"id":"a","urgency":"normal"}] });
        let before = env.clone();
        assert_eq!(clamp_undeclared_critical_alerts(&mut env, false), 0);
        assert_eq!(env, before);
    }

    #[test]
    fn clamp_handles_envelope_without_alerts() {
        let mut env = serde_json::json!({ "v": 1, "speak": "hi" });
        let before = env.clone();
        assert_eq!(clamp_undeclared_critical_alerts(&mut env, false), 0);
        assert_eq!(env, before);
    }

    #[test]
    fn clamp_only_touches_offending_alerts_in_a_batch() {
        let mut env = serde_json::json!({
            "alerts": [
                {"id":"ok","urgency":"normal"},
                {"id":"bad","urgency":"critical","full_takeover":true}
            ]
        });
        assert_eq!(clamp_undeclared_critical_alerts(&mut env, false), 1);
        assert_eq!(env["alerts"][0]["urgency"], "normal");
        assert_eq!(env["alerts"][1]["urgency"], "high");
        assert!(env["alerts"][1].get("full_takeover").is_none());
    }

    #[test]
    fn engine_clamps_critical_alert_from_undeclared_skill() {
        let mut engine = Engine::new();
        engine.register_skill(Box::new(ActionSkill {
            id: "rogue",
            action: critical_takeover_envelope(),
        }));
        let resp = engine.process_input("trigger");
        let v = match resp {
            Response::Action(v) => v,
            other => panic!("expected Action, got {other:?}"),
        };
        let alert = &v["alerts"][0];
        assert!(alert.get("full_takeover").is_none(), "undeclared takeover must be stripped");
        assert_eq!(alert["urgency"], "high", "undeclared critical must drop to high");
    }

    #[test]
    fn engine_preserves_critical_alert_from_declared_skill() {
        let mut engine = Engine::new();
        engine.register_skill(Box::new(CapableAlertSkill {
            id: "timer",
            action: critical_takeover_envelope(),
        }));
        let resp = engine.process_input("trigger");
        let v = match resp {
            Response::Action(v) => v,
            other => panic!("expected Action, got {other:?}"),
        };
        let alert = &v["alerts"][0];
        assert_eq!(alert["full_takeover"], true, "declared skill keeps its takeover");
        assert_eq!(alert["urgency"], "critical", "declared skill keeps critical urgency");
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
            requires_setting: None,
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

    // --- Phase 5: assistant-routing prompt + parser ---

    fn router_test_skills() -> Vec<(String, String, String)> {
        vec![
            (
                "current_time".to_string(),
                "Tells the current time.".to_string(),
                "{}".to_string(),
            ),
            (
                "dev.heyari.timer".to_string(),
                "Sets a countdown timer.".to_string(),
                "{}".to_string(),
            ),
        ]
    }

    #[test]
    fn parse_combined_routes_on_leading_sentinel() {
        let skills = router_test_skills();
        match parse_combined_response("SKILL: current_time", &skills) {
            RouteOrAnswer::Skill(id) => assert_eq!(id, "current_time"),
            RouteOrAnswer::Answer(_) => panic!("expected a route"),
        }
        // Markdown around the id is tolerated.
        match parse_combined_response("SKILL: `current_time`", &skills) {
            RouteOrAnswer::Skill(id) => assert_eq!(id, "current_time"),
            RouteOrAnswer::Answer(_) => panic!("expected a route"),
        }
    }

    #[test]
    fn parse_combined_freeform_is_answer() {
        let skills = router_test_skills();
        let text = "The capital of the UAE is Abu Dhabi.";
        match parse_combined_response(text, &skills) {
            RouteOrAnswer::Answer(a) => assert_eq!(a, text),
            RouteOrAnswer::Skill(_) => panic!("expected an answer"),
        }
    }

    #[test]
    fn parse_combined_unknown_id_is_answer() {
        let skills = router_test_skills();
        // Sentinel present but id not in catalogue → not a valid route.
        match parse_combined_response("SKILL: not_a_real_skill", &skills) {
            RouteOrAnswer::Answer(_) => {}
            RouteOrAnswer::Skill(_) => panic!("must not route to an unknown id"),
        }
    }

    #[test]
    fn parse_combined_skill_word_mid_answer_is_answer() {
        let skills = router_test_skills();
        // "skill" mid-sentence must not be mistaken for the leading sentinel.
        match parse_combined_response("I have no skill for that; the answer is 42.", &skills) {
            RouteOrAnswer::Answer(_) => {}
            RouteOrAnswer::Skill(_) => panic!("expected an answer"),
        }
    }

    #[test]
    fn routing_backend_choice() {
        // Routing via the assistant happens only when a cloud assistant is
        // present — for every locale. Offline (on-device LLM only), a
        // non-keyword request gets a direct answer, never a routing round-trip
        // through the slow on-device LLM.
        assert!(uses_assistant_routing(true));
        assert!(!uses_assistant_routing(false));
    }

    #[test]
    fn assistant_routing_prompt_uses_italian_for_italian_locale() {
        let skills = router_test_skills();
        let prompt = build_assistant_routing_prompt("che ore sono", &skills, "it");
        assert!(prompt.contains("L'utente ha detto"));
        assert!(prompt.contains("\"che ore sono\""));
        assert!(prompt.contains("- current_time:"));
        assert!(prompt.contains("Scegli la skill"));
        // Catalogue includes the dotted skill id verbatim.
        assert!(prompt.contains("- dev.heyari.timer:"));
    }

    #[test]
    fn assistant_routing_prompt_falls_back_to_english_for_unknown_locale() {
        let skills = router_test_skills();
        let prompt = build_assistant_routing_prompt("what time is it", &skills, "ja");
        assert!(prompt.contains("You are a skill router"));
        assert!(prompt.contains("Pick the skill"));
    }

    #[test]
    fn parses_routing_response_first_line_skill_id() {
        let skills = router_test_skills();
        let picked = parse_assistant_routing_response("current_time", &skills);
        assert_eq!(picked.as_deref(), Some("current_time"));
    }

    #[test]
    fn parses_routing_response_strips_markdown_fences() {
        let skills = router_test_skills();
        // Common cloud-LLM behaviour: backtick-fences the answer.
        let picked = parse_assistant_routing_response("`dev.heyari.timer`", &skills);
        assert_eq!(picked.as_deref(), Some("dev.heyari.timer"));
    }

    #[test]
    fn parses_routing_response_skips_explanation_lines() {
        let skills = router_test_skills();
        // Model added prose despite "do not explain". Find the
        // skill id on a later line.
        let response = "Sure, here's my pick:\n\ncurrent_time";
        let picked = parse_assistant_routing_response(response, &skills);
        assert_eq!(picked.as_deref(), Some("current_time"));
    }

    #[test]
    fn parses_routing_response_returns_none_for_none_sentinel() {
        let skills = router_test_skills();
        assert!(parse_assistant_routing_response("NONE", &skills).is_none());
        assert!(parse_assistant_routing_response("none", &skills).is_none());
        assert!(parse_assistant_routing_response("**NONE**", &skills).is_none());
    }

    #[test]
    fn parses_routing_response_returns_none_for_unknown_id() {
        let skills = router_test_skills();
        // Model hallucinated a skill that doesn't exist — must not
        // dispatch to a non-existent skill.
        assert!(parse_assistant_routing_response("ai.example.bogus", &skills).is_none());
    }

    #[test]
    fn parses_routing_response_returns_none_for_empty() {
        let skills = router_test_skills();
        assert!(parse_assistant_routing_response("", &skills).is_none());
        assert!(parse_assistant_routing_response("   \n\n   ", &skills).is_none());
    }

    #[test]
    fn fallback_response_localised_per_supported_locale() {
        assert_eq!(
            fallback_response_for("en"),
            "Sorry, I didn't understand that."
        );
        assert_eq!(fallback_response_for("it"), "Scusa, non ho capito.");
        assert_eq!(fallback_response_for("es"), "Lo siento, no entendí.");
        assert_eq!(fallback_response_for("fr"), "Désolé, je n'ai pas compris.");
        // Unknown locale falls back to English.
        assert_eq!(
            fallback_response_for("ja"),
            "Sorry, I didn't understand that."
        );
    }

    // --- Generic manifest-declared fallback tier ---

    /// A neutral fake fallback skill. Drives every branch of the generic
    /// dispatch loop: which tier it declares, and whether `execute` reports a
    /// no-match envelope.
    struct FakeFallbackSkill {
        tier: Option<FallbackTier>,
        no_match: bool,
    }

    impl Skill for FakeFallbackSkill {
        fn id(&self) -> &str { "test.fallback" }
        fn description(&self) -> &str { "fake fallback" }
        fn specificity(&self) -> Specificity { Specificity::Medium }
        fn score(&self, _input: &str, _ctx: &SkillContext) -> f32 { 0.0 }
        fn execute(&self, _input: &str, _ctx: &SkillContext) -> Response {
            if self.no_match {
                Response::Action(serde_json::json!({
                    "v": 1, "speak": "no", "_ari_no_match": true
                }))
            } else {
                Response::Text("from fallback".into())
            }
        }
        fn fallback_tier(&self) -> Option<FallbackTier> {
            self.tier.clone()
        }
    }

    /// Build an engine with a single fake fallback skill, optionally seeding a
    /// `("test.fallback", "base_url")` config entry.
    fn engine_with_fallback(
        tier: Option<FallbackTier>,
        no_match: bool,
        base_url: Option<&str>,
    ) -> Engine {
        use ari_skill_loader::assistant::MemoryConfigStore;
        use std::sync::Arc;
        let mut e = Engine::new();
        e.register_skill(Box::new(FakeFallbackSkill { tier, no_match }));
        if let Some(url) = base_url {
            let mut store = MemoryConfigStore::new();
            store.set("test.fallback", "base_url", url);
            e.set_config_store(Some(Arc::new(store)));
        }
        e
    }

    #[test]
    fn fallback_engaged_when_required_setting_present() {
        let e = engine_with_fallback(
            Some(FallbackTier { requires_setting: Some("base_url".into()) }),
            false,
            Some("http://hass.local:8123"),
        );
        let (resp, trace) = e.process_input_traced("asdf qwer");
        assert_eq!(
            trace.unwrap().winner,
            Some("fallback:test.fallback".to_string())
        );
        match resp {
            Response::Text(t) => assert_eq!(t, "from fallback"),
            other => panic!("expected fallback text, got {other:?}"),
        }
    }

    #[test]
    fn fallback_skipped_when_required_setting_absent() {
        // Tier requires base_url, but the config store has none → skip.
        let e = engine_with_fallback(
            Some(FallbackTier { requires_setting: Some("base_url".into()) }),
            false,
            None,
        );
        let (resp, trace) = e.process_input_traced("asdf qwer");
        assert_ne!(
            trace.unwrap().winner,
            Some("fallback:test.fallback".to_string())
        );
        match resp {
            Response::Text(t) => assert_eq!(t, fallback_response_for("en")),
            other => panic!("expected fallback text, got {other:?}"),
        }
    }

    #[test]
    fn fallback_falls_through_on_no_match() {
        // base_url IS set, but the skill reports `_ari_no_match` → fall through.
        let e = engine_with_fallback(
            Some(FallbackTier { requires_setting: Some("base_url".into()) }),
            true,
            Some("http://hass.local:8123"),
        );
        let (resp, trace) = e.process_input_traced("asdf qwer");
        assert_ne!(
            trace.unwrap().winner,
            Some("fallback:test.fallback".to_string())
        );
        match resp {
            Response::Text(t) => assert_eq!(t, fallback_response_for("en")),
            other => panic!("expected fallback text, got {other:?}"),
        }
    }

    #[test]
    fn fallback_without_required_setting_always_engages() {
        // `requires_setting: None` → engage with no config entry at all.
        let e = engine_with_fallback(
            Some(FallbackTier { requires_setting: None }),
            false,
            None,
        );
        let (resp, trace) = e.process_input_traced("asdf qwer");
        assert_eq!(
            trace.unwrap().winner,
            Some("fallback:test.fallback".to_string())
        );
        match resp {
            Response::Text(t) => assert_eq!(t, "from fallback"),
            other => panic!("expected fallback text, got {other:?}"),
        }
    }

    #[test]
    fn non_fallback_skill_is_never_a_fallback() {
        // A skill whose `fallback_tier()` is the default `None` must never be
        // engaged as a fallback for an unmatched utterance.
        let e = engine_with_fallback(None, false, Some("http://hass.local:8123"));
        let (resp, trace) = e.process_input_traced("asdf qwer");
        assert_ne!(
            trace.unwrap().winner,
            Some("fallback:test.fallback".to_string())
        );
        match resp {
            Response::Text(t) => assert_eq!(t, fallback_response_for("en")),
            other => panic!("expected fallback text, got {other:?}"),
        }
    }

    #[test]
    fn fallback_prefers_gated_integration_over_unconditional_catch_all() {
        use ari_skill_loader::assistant::MemoryConfigStore;
        use std::sync::Arc;

        // Unconditional catch-all, registered FIRST (mimics a built-in before WASM).
        struct Uncond;
        impl Skill for Uncond {
            fn id(&self) -> &str { "uncond" }
            fn specificity(&self) -> Specificity { Specificity::Low }
            fn score(&self, _: &str, _: &SkillContext) -> f32 { 0.0 }
            fn execute(&self, _: &str, _: &SkillContext) -> Response {
                Response::Text("uncond".to_string())
            }
            fn fallback_tier(&self) -> Option<FallbackTier> {
                Some(FallbackTier { requires_setting: None })
            }
        }

        // Gated integration, registered SECOND, requires setting "base_url".
        struct Gated;
        impl Skill for Gated {
            fn id(&self) -> &str { "gated" }
            fn specificity(&self) -> Specificity { Specificity::Low }
            fn score(&self, _: &str, _: &SkillContext) -> f32 { 0.0 }
            fn execute(&self, _: &str, _: &SkillContext) -> Response {
                Response::Text("gated".to_string())
            }
            fn fallback_tier(&self) -> Option<FallbackTier> {
                Some(FallbackTier { requires_setting: Some("base_url".to_string()) })
            }
        }

        let mut engine = Engine::new();
        engine.register_skill(Box::new(Uncond)); // first
        engine.register_skill(Box::new(Gated));  // second
        let mut store = MemoryConfigStore::new();
        store.set("gated", "base_url", "http://ha.local");
        engine.set_config_store(Some(Arc::new(store)));

        // Nothing keyword/router/assistant matches "asdf qwer" → both fallbacks
        // eligible. Gated must win despite being registered AFTER Uncond.
        let (resp, trace) = engine.process_input_traced("asdf qwer");
        assert_eq!(trace.unwrap().winner, Some("fallback:gated".to_string()));
        match resp {
            Response::Text(t) => assert_eq!(t, "gated"),
            other => panic!("expected gated fallback text, got {other:?}"),
        }
    }

    #[test]
    fn query_skill_setting_routes_to_skill_by_id() {
        use ari_core::SettingsQueryResult;
        struct FakeSettingsSkill;
        impl Skill for FakeSettingsSkill {
            fn id(&self) -> &str { "dev.example.s" }
            fn score(&self, _: &str, _: &SkillContext) -> f32 { 0.0 }
            fn specificity(&self) -> Specificity { Specificity::Low }
            fn execute(&self, _: &str, _: &SkillContext) -> Response { Response::Text(String::new()) }
            fn settings_query(&self, field: &str, values_json: &str) -> SettingsQueryResult {
                SettingsQueryResult {
                    ok: true,
                    error: None,
                    message: Some(format!("{field}|{values_json}")),
                    options: vec![],
                    refresh: false,
                }
            }
        }
        let mut e = Engine::new();
        e.register_skill(Box::new(FakeSettingsSkill));
        let r = e.query_skill_setting("dev.example.s", "agent_id", r#"{"base_url":"h"}"#);
        assert_eq!(r.ok, true);
        assert_eq!(r.message.as_deref(), Some("agent_id|{\"base_url\":\"h\"}"));
        // unknown skill → ok:false, no panic
        let miss = e.query_skill_setting("nope", "x", "{}");
        assert_eq!(miss.ok, false);
    }

    #[test]
    fn settings_action_unknown_skill_returns_error() {
        let engine = Engine::new();
        let r = engine.settings_action("does.not.exist", "sign_in", "{}");
        assert_eq!(r.ok, false);
        assert!(r.error.unwrap().contains("does.not.exist"));
    }

    #[test]
    fn parse_flag_continuation_inline() {
        let (text, flag) = parse_continuation_flag("Abu Dhabi is the capital. [continuation]");
        assert_eq!(text, "Abu Dhabi is the capital.");
        assert_eq!(flag, ContinuationFlag::Continuation);
    }

    #[test]
    fn parse_flag_new_inline() {
        let (text, flag) = parse_continuation_flag("27 times 3 is 81. [new]");
        assert_eq!(text, "27 times 3 is 81.");
        assert_eq!(flag, ContinuationFlag::New);
    }

    #[test]
    fn parse_flag_case_and_quotes_tolerant() {
        let (text, flag) = parse_continuation_flag("Sure. ['NEW']");
        assert_eq!(text, "Sure.");
        assert_eq!(flag, ContinuationFlag::New);
    }

    #[test]
    fn parse_flag_missing_defaults_to_continuation() {
        let (text, flag) = parse_continuation_flag("Just an answer with no marker.");
        assert_eq!(text, "Just an answer with no marker.");
        assert_eq!(flag, ContinuationFlag::Continuation);
    }

    #[test]
    fn parse_flag_leaves_non_marker_brackets_intact() {
        let (text, flag) = parse_continuation_flag("See item [3] on the list.");
        assert_eq!(text, "See item [3] on the list.");
        assert_eq!(flag, ContinuationFlag::Continuation);
    }

    #[test]
    fn parse_flag_strips_trailing_newline_before_marker() {
        let (text, flag) = parse_continuation_flag("Paris.\n[new]");
        assert_eq!(text, "Paris.");
        assert_eq!(flag, ContinuationFlag::New);
    }

    #[test]
    fn record_continuation_appends_turn() {
        let engine = Engine::new();
        engine.record_assistant_turn("q1", "a1", ContinuationFlag::Continuation);
        engine.record_assistant_turn("q2", "a2", ContinuationFlag::Continuation);
        let turns = engine.conversation_context();
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].user, "q1");
        assert_eq!(turns[1].assistant, "a2");
    }

    #[test]
    fn record_new_reseeds_buffer_with_single_turn() {
        let engine = Engine::new();
        engine.record_assistant_turn("q1", "a1", ContinuationFlag::Continuation);
        engine.record_assistant_turn("q2", "a2", ContinuationFlag::Continuation);
        engine.record_assistant_turn("27 times 3?", "81.", ContinuationFlag::New);
        let turns = engine.conversation_context();
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].user, "27 times 3?");
        assert_eq!(turns[0].assistant, "81.");
    }

    #[test]
    fn record_caps_to_five_most_recent_exchanges() {
        let engine = Engine::new();
        for i in 0..7 {
            engine.record_assistant_turn(&format!("q{i}"), &format!("a{i}"), ContinuationFlag::Continuation);
        }
        let turns = engine.conversation_context();
        assert_eq!(turns.len(), 5);
        assert_eq!(turns[0].user, "q2"); // q0,q1 dropped
        assert_eq!(turns[4].user, "q6");
    }

    #[test]
    fn conversation_context_empty_when_no_buffer() {
        let engine = Engine::new();
        assert!(engine.conversation_context().is_empty());
    }

    #[test]
    fn conversation_context_expires_stale_buffer() {
        use std::time::{Duration, Instant};
        let engine = Engine::new();
        engine.record_assistant_turn("q1", "a1", ContinuationFlag::Continuation);
        // Age the buffer past the TTL.
        {
            let mut g = engine.conversation.lock().unwrap();
            g.as_mut().unwrap().last_activity =
                Instant::now().checked_sub(Duration::from_secs(120)).unwrap();
        }
        assert!(engine.conversation_context().is_empty(), "stale buffer must read empty");
        assert!(engine.conversation.lock().unwrap().is_none(), "stale buffer must be dropped");
    }

    #[test]
    fn conversation_context_refreshes_activity_when_fresh() {
        use std::time::{Duration, Instant};
        let engine = Engine::new();
        engine.record_assistant_turn("q1", "a1", ContinuationFlag::Continuation);
        // Age it to 80s — still inside the 90s TTL.
        {
            let mut g = engine.conversation.lock().unwrap();
            g.as_mut().unwrap().last_activity =
                Instant::now().checked_sub(Duration::from_secs(80)).unwrap();
        }
        let turns = engine.conversation_context();
        assert_eq!(turns.len(), 1, "fresh buffer returns its turns");
        let elapsed = engine.conversation.lock().unwrap().as_ref().unwrap().last_activity.elapsed();
        assert!(elapsed < Duration::from_secs(5), "activity timer must be refreshed");
    }

    #[test]
    fn memory_enabled_by_default() {
        let engine = Engine::new();
        engine.record_assistant_turn("hi", "hello", ContinuationFlag::New);
        assert_eq!(engine.conversation_context().len(), 1, "memory on by default");
    }

    #[test]
    fn record_is_noop_while_memory_disabled() {
        let engine = Engine::new();
        engine.set_conversation_memory_enabled(false);
        engine.record_assistant_turn("hi", "hello", ContinuationFlag::New);
        // Re-enable and read: nothing should have been recorded.
        engine.set_conversation_memory_enabled(true);
        assert!(
            engine.conversation_context().is_empty(),
            "nothing recorded while memory disabled"
        );
    }

    #[test]
    fn disabling_memory_returns_empty_context_and_wipes_buffer() {
        let engine = Engine::new();
        engine.record_assistant_turn("hi", "hello", ContinuationFlag::New);
        assert_eq!(engine.conversation_context().len(), 1);

        engine.set_conversation_memory_enabled(false);
        assert!(engine.conversation_context().is_empty(), "no history while disabled");

        // Re-enabling reveals nothing — disabling wiped the buffer, not just hid it.
        engine.set_conversation_memory_enabled(true);
        assert!(
            engine.conversation_context().is_empty(),
            "disabling memory wiped the buffer"
        );
    }

    #[cfg(feature = "llm")]
    struct RecordingFallback {
        last_history: std::sync::Mutex<Vec<(String, String)>>,
        last_facts: std::sync::Mutex<Vec<String>>,
        reply: &'static str,
    }

    #[cfg(feature = "llm")]
    impl ari_llm::Fallback for RecordingFallback {
        fn try_answer(
            &self,
            _input: &str,
            _skills: &[ari_llm::SkillInfo],
            _locale: &str,
            history: &[(String, String)],
            facts: &[String],
        ) -> Option<ari_llm::FallbackResult> {
            *self.last_history.lock().unwrap() = history.to_vec();
            *self.last_facts.lock().unwrap() = facts.to_vec();
            Some(ari_llm::FallbackResult::DirectAnswer { text: self.reply.to_string() })
        }
    }

    #[cfg(feature = "llm")]
    #[test]
    fn assistant_turn_records_and_second_turn_carries_history() {
        let mut engine = Engine::new();
        let fb = std::sync::Arc::new(RecordingFallback {
            last_history: std::sync::Mutex::new(Vec::new()),
            last_facts: std::sync::Mutex::new(Vec::new()),
            reply: "Paris. [continuation]",
        });
        engine.set_llm(fb.clone());
        engine.set_active_assistant(Some(ActiveAssistant::Builtin { tier: ari_llm::BuiltinTier::Small }));

        // Turn 1: no history yet; answer recorded; flag stripped from response.
        let (resp, _) = engine.process_input_traced("what is the capital of france");
        assert!(fb.last_history.lock().unwrap().is_empty(), "first turn sends no history");
        match resp { Response::Text(t) => assert_eq!(t, "Paris."), o => panic!("{o:?}") }

        // Turn 2: prior turn supplied as chronological role/content history.
        let _ = engine.process_input_traced("what is the population");
        let hist = fb.last_history.lock().unwrap().clone();
        assert_eq!(hist.len(), 2);
        assert_eq!(hist[0], ("user".to_string(), "what is the capital of france".to_string()));
        assert_eq!(hist[1], ("assistant".to_string(), "Paris.".to_string()));
    }

    #[cfg(feature = "llm")]
    #[test]
    fn assistant_answer_receives_facts() {
        // The remembered-facts snapshot must reach the builtin assistant's
        // free-text answer site (try_answer) verbatim.
        let mut engine = Engine::new();
        let fb = std::sync::Arc::new(RecordingFallback {
            last_history: std::sync::Mutex::new(Vec::new()),
            last_facts: std::sync::Mutex::new(Vec::new()),
            reply: "ok. [continuation]",
        });
        engine.set_llm(fb.clone());
        engine.set_active_assistant(Some(ActiveAssistant::Builtin { tier: ari_llm::BuiltinTier::Small }));
        engine.set_remembered_facts(vec!["i am vegetarian".to_string()]);

        let _ = engine.process_input_traced("tell me a joke");
        assert_eq!(
            *fb.last_facts.lock().unwrap(),
            vec!["i am vegetarian".to_string()],
            "try_answer must receive the stored facts snapshot"
        );
    }

    #[cfg(feature = "llm")]
    #[test]
    fn new_flag_reseeds_conversation() {
        let mut engine = Engine::new();
        let fb = std::sync::Arc::new(RecordingFallback {
            last_history: std::sync::Mutex::new(Vec::new()),
            last_facts: std::sync::Mutex::new(Vec::new()),
            reply: "81. [new]",
        });
        engine.set_llm(fb.clone());
        engine.set_active_assistant(Some(ActiveAssistant::Builtin { tier: ari_llm::BuiltinTier::Small }));
        engine.record_assistant_turn("capital of france", "Paris.", ContinuationFlag::Continuation);

        let (resp, _) = engine.process_input_traced("what is 27 times 3");
        match resp { Response::Text(t) => assert_eq!(t, "81."), o => panic!("{o:?}") }
        let turns = engine.conversation_context();
        assert_eq!(turns.len(), 1, "[new] reseeds to a single turn");
        assert_eq!(turns[0].user, "what is 27 times 3");
        assert_eq!(turns[0].assistant, "81.");
    }

    #[cfg(feature = "llm")]
    #[test]
    fn skill_win_refreshes_but_does_not_record() {
        use std::time::{Duration, Instant};
        let mut engine = Engine::new();
        engine.register_skill(Box::new(MockSkill {
            id: "weather", specificity: Specificity::High, fixed_score: 1.0,
            response: "It is sunny.", requires_setting: None,
        }));
        // A live conversation from a prior assistant turn, aged to 80s.
        engine.record_assistant_turn("capital of france", "Paris.", ContinuationFlag::Continuation);
        {
            let mut g = engine.conversation.lock().unwrap();
            g.as_mut().unwrap().last_activity =
                Instant::now().checked_sub(Duration::from_secs(80)).unwrap();
        }
        let _ = engine.process_input_traced("weather please"); // MockSkill wins
        let turns = engine.conversation_context();
        assert_eq!(turns.len(), 1, "skill turn must NOT record into the buffer");
        let elapsed = engine.conversation.lock().unwrap().as_ref().unwrap().last_activity.elapsed();
        assert!(elapsed < Duration::from_secs(5), "skill turn must refresh the timer");
    }

    #[test]
    fn behaviour_a_reply_does_not_record() {
        let mut engine = Engine::new();
        engine.register_skill(Box::new(AskingSkill));
        engine.record_assistant_turn("capital of france", "Paris.", ContinuationFlag::Continuation);

        let _ = engine.process_input_traced("play music");   // AskingSkill asks → pending turn
        let _ = engine.process_input_traced("spotify");       // reply via execute_reply
        let turns = engine.conversation_context();
        assert_eq!(turns.len(), 1, "behaviour-A reply must not enter the conversation buffer");
        assert_eq!(turns[0].user, "capital of france");
    }

    #[cfg(feature = "llm")]
    #[test]
    fn expired_buffer_reseeds_on_next_assistant_turn() {
        use std::time::{Duration, Instant};
        let mut engine = Engine::new();
        let fb = std::sync::Arc::new(RecordingFallback {
            last_history: std::sync::Mutex::new(Vec::new()),
            last_facts: std::sync::Mutex::new(Vec::new()),
            reply: "Fresh answer. [continuation]",
        });
        engine.set_llm(fb.clone());
        engine.set_active_assistant(Some(ActiveAssistant::Builtin { tier: ari_llm::BuiltinTier::Small }));
        engine.record_assistant_turn("old question", "old answer", ContinuationFlag::Continuation);
        {
            let mut g = engine.conversation.lock().unwrap();
            g.as_mut().unwrap().last_activity =
                Instant::now().checked_sub(Duration::from_secs(120)).unwrap();
        }
        let _ = engine.process_input_traced("a brand new question");
        assert!(fb.last_history.lock().unwrap().is_empty(), "expired buffer sends no history");
        let turns = engine.conversation_context();
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].user, "a brand new question", "buffer reseeds with the fresh turn only");
    }

    #[test]
    fn enter_phrases_match_normalised_forms() {
        // Post-normalise: "let's talk" -> "let us talk".
        for p in ["let us talk", "let us chat", "let us have a conversation",
                  "keep listening", "start a conversation"] {
            assert!(is_enter_conversation_phrase(p, "en"), "should match: {p}");
        }
        assert!(is_enter_conversation_phrase("parliamo", "it"));
        assert!(!is_enter_conversation_phrase("what time is it", "en"));
        assert!(!is_enter_conversation_phrase("let us talk about the weather", "en"));

        // Clipped "let's [pause] talk" finalises as bare "let us" -> still enters.
        assert!(is_enter_conversation_phrase("let us", "en"));
        // ...but only as a whole utterance: "let us" + more words is NOT a
        // misheard trigger and must not enter.
        assert!(!is_enter_conversation_phrase("let us go to the shop", "en"));
        assert!(!is_enter_conversation_phrase("let us pray", "en"));
    }

    #[test]
    fn exit_phrases_match_normalised_forms() {
        for p in ["stop", "goodbye", "that is all", "we are done", "end conversation"] {
            assert!(is_exit_conversation_phrase(p, "en"), "should match: {p}");
        }
        // The STT model transcribes a spoken "stop" as "stopped"/"stops"
        // (verified on-device), so those must exit too.
        for p in ["stopped", "stops"] {
            assert!(is_exit_conversation_phrase(p, "en"), "should match: {p}");
        }
        assert!(is_exit_conversation_phrase("basta", "it"));
        assert!(!is_exit_conversation_phrase("stop the timer", "en"));
    }

    #[test]
    fn conversation_acks_are_localised() {
        assert_eq!(enter_conversation_ack_for("en"), "Okay, I'm listening.");
        assert_eq!(exit_conversation_ack_for("en"), "Okay.");
        assert_eq!(enter_conversation_ack_for("it"), "Va bene, ti ascolto.");
        assert_eq!(exit_conversation_ack_for("it"), "Va bene.");
        // Unknown locale falls back to English (not machine-translated).
        assert_eq!(enter_conversation_ack_for("fr"), "Okay, I'm listening.");
    }

    #[test]
    fn capture_prefix_extracts_remainder() {
        assert_eq!(remembered_fact_capture("remember that i am vegetarian", "en"), Some("i am vegetarian"));
        assert_eq!(remembered_fact_capture("remember i live in valletta", "en"), Some("i live in valletta"));
        // Bare command → not a capture.
        assert_eq!(remembered_fact_capture("remember", "en"), None);
        assert_eq!(remembered_fact_capture("remember that", "en"), None);
        // Unrelated utterance → not a capture.
        assert_eq!(remembered_fact_capture("what is the weather", "en"), None);
    }

    #[test]
    fn forget_prefix_extracts_remainder() {
        assert_eq!(remembered_fact_forget("forget that i am vegetarian", "en"), Some("i am vegetarian"));
        assert_eq!(remembered_fact_forget("forget i live in valletta", "en"), Some("i live in valletta"));
        assert_eq!(remembered_fact_forget("forget", "en"), None);
    }

    #[test]
    fn capture_prefix_italian() {
        // Both "ricorda che" and "ricordati che" (and bare "ricorda"/"ricordati")
        // are accepted; the longest matching prefix is stripped.
        assert_eq!(remembered_fact_capture("ricordati che sono vegetariano", "it"), Some("sono vegetariano"));
        assert_eq!(remembered_fact_capture("ricorda che vivo a carlisle", "it"), Some("vivo a carlisle"));
        assert_eq!(remembered_fact_capture("ricorda vivo a carlisle", "it"), Some("vivo a carlisle"));
        // Bare commands → not a capture (must not yield "che").
        assert_eq!(remembered_fact_capture("ricorda", "it"), None);
        assert_eq!(remembered_fact_capture("ricordati", "it"), None);
        assert_eq!(remembered_fact_capture("ricorda che", "it"), None);
        assert_eq!(remembered_fact_capture("ricordati che", "it"), None);
        // English prefix must NOT match under the Italian locale.
        assert_eq!(remembered_fact_capture("remember that i am vegetarian", "it"), None);
    }

    #[test]
    fn forget_prefix_italian() {
        assert_eq!(remembered_fact_forget("dimentica che sono vegetariano", "it"), Some("sono vegetariano"));
        assert_eq!(remembered_fact_forget("dimentica sono vegetariano", "it"), Some("sono vegetariano"));
        assert_eq!(remembered_fact_forget("dimentica", "it"), None);
        assert_eq!(remembered_fact_forget("dimentica che", "it"), None);
    }

    #[test]
    fn italian_remember_command_captures_and_acks_end_to_end() {
        let mut engine = Engine::new();
        engine.set_locale("it".to_string());
        let (resp, _) = engine.process_input_traced("Ricordati che vivo a Carlisle");
        match resp {
            Response::Text(t) => assert_eq!(t, "Fatto, me ne ricorderò."),
            other => panic!("expected Italian capture ack, got {other:?}"),
        }
        assert_eq!(engine.remembered_facts(), vec!["vivo a carlisle".to_string()]);
        // Italian recall query lists it back with the Italian intro.
        let (resp, _) = engine.process_input_traced("cosa ti ricordi di me");
        match resp {
            Response::Text(t) => assert_eq!(t, "Ecco cosa ricordo di te:\n- vivo a carlisle"),
            other => panic!("expected Italian recall list, got {other:?}"),
        }
    }

    #[test]
    fn forget_all_and_query_phrases_match() {
        assert!(is_forget_all_phrase("forget everything about me", "en"));
        assert!(is_forget_all_phrase("forget everything you know about me", "en"));
        assert!(!is_forget_all_phrase("forget that i am vegetarian", "en"));
        assert!(is_recall_query_phrase("what do you remember about me", "en"));
        assert!(is_recall_query_phrase("what do you know about me", "en"));
        assert!(!is_recall_query_phrase("what is the weather", "en"));
    }

    #[test]
    fn conversation_active_flag_toggles_and_clears_pending() {
        let mut engine = Engine::new();
        engine.register_skill(Box::new(AskingSkill)); // arms a pending turn
        let _ = engine.process_input_traced("play music");
        assert!(engine.has_pending_turn());

        engine.set_conversation_active(true);
        assert!(engine.is_conversation_active());

        // Turning the mode off must also drop any half-finished pending turn.
        engine.set_conversation_active(false);
        assert!(!engine.is_conversation_active());
        assert!(!engine.has_pending_turn(), "deactivation clears pending turn");
    }

    #[test]
    fn set_conversation_active_refused_when_memory_disabled() {
        let engine = Engine::new();
        engine.set_conversation_memory_enabled(false);
        engine.set_conversation_active(true);
        assert!(
            !engine.is_conversation_active(),
            "talk mode must not activate while memory is off"
        );
    }

    #[test]
    fn disabling_memory_ends_active_conversation() {
        let engine = Engine::new();
        engine.set_conversation_active(true);
        assert!(engine.is_conversation_active(), "sanity: active with memory on");
        engine.set_conversation_memory_enabled(false);
        assert!(
            !engine.is_conversation_active(),
            "disabling memory must end any active talk-mode session"
        );
    }

    #[test]
    fn enter_phrase_signals_and_acks_without_routing() {
        let engine = Engine::new();
        let (resp, _) = engine.process_input_traced("let's talk");
        assert!(matches!(resp, Response::Text(ref s) if s == enter_conversation_ack_for("en")));
        assert!(engine.take_enter_signal(), "enter signal must be set");
        assert!(!engine.take_enter_signal(), "signal is one-shot (already consumed)");
    }

    #[test]
    fn enter_phrase_blocked_when_memory_disabled() {
        let engine = Engine::new();
        engine.set_conversation_memory_enabled(false);
        let (resp, _) = engine.process_input_traced("let's talk");
        assert!(
            matches!(resp, Response::Text(ref s) if s == conversation_memory_required_msg_for("en")),
            "must speak the enable-memory guidance"
        );
        assert!(
            !engine.take_enter_signal(),
            "enter signal must NOT fire when memory is disabled"
        );
    }

    #[test]
    fn exit_phrase_only_fires_while_active() {
        let engine = Engine::new();

        // Not in the mode: "stop" is NOT hijacked — no exit signal, routes on.
        let _ = engine.process_input_traced("stop");
        assert!(!engine.take_exit_signal(), "bare stop must pass through when inactive");

        // In the mode: "stop" exits.
        engine.set_conversation_active(true);
        let (resp, _) = engine.process_input_traced("stop");
        assert!(matches!(resp, Response::Text(ref s) if s == exit_conversation_ack_for("en")));
        assert!(engine.take_exit_signal(), "exit signal must be set while active");
    }

    #[test]
    fn capture_fact_stores_and_signals() {
        let engine = Engine::new();
        assert!(engine.capture_fact("i am vegetarian"));
        assert_eq!(engine.remembered_facts(), vec!["i am vegetarian".to_string()]);
        assert!(engine.take_facts_changed_signal());
        // Signal is one-shot.
        assert!(!engine.take_facts_changed_signal());
    }

    #[test]
    fn capture_fact_dedupes_exact_duplicate() {
        let engine = Engine::new();
        assert!(engine.capture_fact("i am vegetarian"));
        let _ = engine.take_facts_changed_signal();
        assert!(!engine.capture_fact("i am vegetarian"));
        assert_eq!(engine.remembered_facts().len(), 1);
        assert!(!engine.take_facts_changed_signal());
    }

    #[test]
    fn capture_fact_caps_and_evicts_oldest() {
        let engine = Engine::new();
        for i in 0..MAX_REMEMBERED_FACTS {
            assert!(engine.capture_fact(&format!("fact {i}")));
        }
        assert_eq!(engine.remembered_facts().len(), MAX_REMEMBERED_FACTS);
        // The 51st capture evicts "fact 0".
        assert!(engine.capture_fact("fact new"));
        let facts = engine.remembered_facts();
        assert_eq!(facts.len(), MAX_REMEMBERED_FACTS);
        assert!(!facts.contains(&"fact 0".to_string()));
        assert_eq!(facts.last(), Some(&"fact new".to_string()));
    }

    #[test]
    fn forget_fact_removes_match_only() {
        let engine = Engine::new();
        engine.capture_fact("i am vegetarian");
        engine.capture_fact("i live in valletta");
        let _ = engine.take_facts_changed_signal();
        assert!(engine.forget_fact("i am vegetarian"));
        assert_eq!(engine.remembered_facts(), vec!["i live in valletta".to_string()]);
        assert!(engine.take_facts_changed_signal());
        // Non-match leaves the list intact and does not signal.
        assert!(!engine.forget_fact("i am nothing"));
        assert_eq!(engine.remembered_facts().len(), 1);
        assert!(!engine.take_facts_changed_signal());
    }

    #[test]
    fn forget_all_clears_and_signals_only_when_nonempty() {
        let engine = Engine::new();
        engine.capture_fact("i am vegetarian");
        let _ = engine.take_facts_changed_signal();
        assert!(engine.forget_all_facts());
        assert!(engine.remembered_facts().is_empty());
        assert!(engine.take_facts_changed_signal());
        // Clearing an already-empty list is a no-op.
        assert!(!engine.forget_all_facts());
        assert!(!engine.take_facts_changed_signal());
    }

    #[test]
    fn set_remembered_facts_replaces_dedups_caps_without_signal() {
        let engine = Engine::new();
        let mut input = vec!["a".to_string(), "a".to_string(), "b".to_string()];
        for i in 0..MAX_REMEMBERED_FACTS {
            input.push(format!("x{i}"));
        }
        engine.set_remembered_facts(input);
        let facts = engine.remembered_facts();
        assert_eq!(facts.len(), MAX_REMEMBERED_FACTS);
        // Dedup collapsed the two "a" entries into one, but "a" and "b" are
        // the oldest entries in insertion order; the cap evicts the oldest
        // (same policy as capture_fact), so neither survives.
        assert_eq!(facts.iter().filter(|f| f.as_str() == "a").count(), 0);
        assert!(!facts.contains(&"b".to_string()));
        assert_eq!(facts.first(), Some(&"x0".to_string()));
        // Hydration never raises the write-back signal.
        assert!(!engine.take_facts_changed_signal());
    }

    #[test]
    fn remember_command_captures_and_acks() {
        let engine = Engine::new();
        let (resp, _) = engine.process_input_traced("Remember that I'm vegetarian");
        match resp {
            Response::Text(t) => assert_eq!(t, "Got it — I'll remember that."),
            other => panic!("expected Text ack, got {other:?}"),
        }
        assert_eq!(engine.remembered_facts(), vec!["i am vegetarian".to_string()]);
        assert!(engine.take_facts_changed_signal());
    }

    #[test]
    fn forget_command_removes_and_acks() {
        let engine = Engine::new();
        engine.process_input_traced("remember that i am vegetarian");
        let _ = engine.take_facts_changed_signal();
        let (resp, _) = engine.process_input_traced("forget that I'm vegetarian");
        match resp {
            Response::Text(t) => assert_eq!(t, "Okay, I've forgotten that."),
            other => panic!("expected Text ack, got {other:?}"),
        }
        assert!(engine.remembered_facts().is_empty());
        assert!(engine.take_facts_changed_signal());
    }

    #[test]
    fn forget_command_non_match_acks_not_found() {
        let engine = Engine::new();
        let (resp, _) = engine.process_input_traced("forget that i like tea");
        match resp {
            Response::Text(t) => assert_eq!(t, "I didn't have that one."),
            other => panic!("expected Text ack, got {other:?}"),
        }
        assert!(!engine.take_facts_changed_signal());
    }

    #[test]
    fn forget_everything_clears_all() {
        let engine = Engine::new();
        engine.process_input_traced("remember that i am vegetarian");
        engine.process_input_traced("remember that i live in valletta");
        let _ = engine.take_facts_changed_signal();
        let (resp, _) = engine.process_input_traced("forget everything about me");
        match resp {
            Response::Text(t) => assert_eq!(t, "Okay, I've forgotten everything I knew about you."),
            other => panic!("expected Text ack, got {other:?}"),
        }
        assert!(engine.remembered_facts().is_empty());
    }

    #[test]
    fn recall_query_lists_facts() {
        let engine = Engine::new();
        engine.process_input_traced("remember that i am vegetarian");
        engine.process_input_traced("remember that i live in valletta");
        let (resp, _) = engine.process_input_traced("what do you remember about me");
        match resp {
            Response::Text(t) => assert_eq!(
                t,
                "Here's what I remember about you:\n- i am vegetarian\n- i live in valletta"
            ),
            other => panic!("expected Text list, got {other:?}"),
        }
    }

    #[test]
    fn recall_query_empty_says_nothing_yet() {
        let engine = Engine::new();
        let (resp, _) = engine.process_input_traced("what do you know about me");
        match resp {
            Response::Text(t) => assert_eq!(t, "I don't remember anything about you yet."),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn bare_remember_is_not_captured() {
        let engine = Engine::new();
        // "remember" alone has no remainder → falls through to normal routing,
        // which on a bare engine yields the fallback response (not an ack).
        let (resp, _) = engine.process_input_traced("remember");
        match resp {
            Response::Text(t) => assert_eq!(t, fallback_response_for("en")),
            other => panic!("expected fallback Text, got {other:?}"),
        }
        assert!(engine.remembered_facts().is_empty());
    }

    #[test]
    fn keyword_decision_reports_hits_and_misses() {
        let mut engine = crate::Engine::new();
        engine.register_skill(Box::new(ari_skills::OpenSkill::new()));
        engine.register_skill(Box::new(ari_skills::CurrentTimeSkill::new()));
        engine.set_locale("it".to_string());
        // `apri spotify` is a plain open trigger — the keyword scorer wins it,
        // so production never consults the router.
        assert_eq!(engine.keyword_decision("apri spotify").as_deref(), Some("open"));
        // Task 3 added the imperative-clitic triggers, so the keyword tier now
        // owns this and the router is no longer responsible for it.
        assert_eq!(engine.keyword_decision("aprimi duolingo").as_deref(), Some("open"));
        // CurrentTimeSkill IS registered, so these exercise its real scorer.
        // Polite conditionals with no `che ora`/`che ore` pair are genuine
        // keyword-misses — the shape the router exists to catch.
        assert_eq!(engine.keyword_decision("sapresti dirmi l'ora"), None);
        assert_eq!(engine.keyword_decision("potresti dirmi l'ora"), None);
        // ...but anything completing a trigger pair IS won by the keyword tier.
        assert_eq!(engine.keyword_decision("che ore sono").as_deref(), Some("current_time"));
        // General knowledge: no skill should claim it.
        assert_eq!(engine.keyword_decision("chi ha scritto la Divina Commedia"), None);
    }

    #[test]
    fn keyword_decision_is_locale_aware() {
        let mut engine = crate::Engine::new();
        engine.register_skill(Box::new(ari_skills::OpenSkill::new()));
        engine.set_locale("en".to_string());
        assert_eq!(engine.keyword_decision("open spotify").as_deref(), Some("open"));
        assert_eq!(engine.keyword_decision("who wrote Romeo and Juliet"), None);
    }
}
