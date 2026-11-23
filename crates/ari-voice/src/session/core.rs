//! The session's state machine. Everything `VoiceSession.kt` decided, in the
//! order it decided it, with its waits turned into messages: a worker's
//! result, the host saying speech finished, a deadline passing.
//!
//! The Kotlin ran as one coroutine whose suspension points were where it
//! waited. A dismiss cancelled it, and the next suspension point threw. Here
//! every timer and worker result carries the epoch it was started in, a
//! dismiss moves the epoch on, and anything from an older one is dropped.
//! While the coroutine was busy with a turn it read no speech-to-text state;
//! here those events wait in a queue until the turn has handed back.

use std::collections::{BTreeMap, VecDeque};
use std::path::PathBuf;

use super::captures::{UtteranceCapture, UtteranceOutcome, UtteranceTurn, WakeCaptureHook};
use super::rules::{
    barge_in_effective, is_no_speech_wake, is_stale_turn, owns_speaking_flag, partial_reveals_wake,
    response_label, should_accept_wake, should_clear_conversation, should_cut_tts, should_enter_conversation,
    should_exit_conversation, should_persist_facts, should_rearm, utterance_outcome, utterance_turn,
    wake_name_verdict,
};
use super::{
    ActionOutcome, CaptureMode, EngineReply, Intercepted, LoggedReply, LoggedTurn, Phrase, SessionConfig,
    SessionError, SessionHost, SessionState, TypedFinish,
};
use crate::pipeline::LogLevel;
use crate::stt::{DEFAULT_REWIND_SECONDS, ListenEnded, ListenParams, SttEvent};
use crate::wake_log::WakeOutcome;
use crate::wake_phrase::EarlyNameCheck;

// 30 s: accommodates the buffered paths (Whisper, Parakeet), which produce
// no partials, so `last_activity_at` can't be refreshed during the utterance
// and decode. The streaming path refreshes it on every non-blank partial, so
// there the timeout trips after 30 s of silence; buffered listens get about
// 30 s for capture, endpointing and decode together. Tighter than this and
// slow emulator decodes (Whisper Turbo int8 takes 5-10 s on x86_64) trip the
// watcher before the transcript lands.
pub(crate) const SILENCE_TIMEOUT_MS: i64 = 30_000;
// The opening turn of a wake session gets a much tighter window: the user
// just said the wake phrase, so 8 s of dead air means it wasn't addressed to
// Ari, and every extra second the mic stays armed is a second in which
// unrelated ambient speech can be taken as a command. Re-armed reply turns
// and dictation keep the full 30 s above: there Ari asked a question or the
// user deliberately opened the mic, so waiting is legitimate.
pub(crate) const WAKE_TURN_SILENCE_TIMEOUT_MS: i64 = 8_000;
// How long a quiet wake waits for anything that counts as speech before
// giving up without transcribing. A real wake has the phrase itself in the
// pre-roll, so it counts at once; only noise waits this long.
pub(crate) const QUIET_GIVE_UP_MS: i64 = 3_000;
// A quiet wake shows nothing while it listens, and the wake word is off
// meanwhile, so it stops here and transcribes what it has, even while the
// television talks on. The longest real command on the test phone took 2.9 s
// after the wake (2026-10-17).
pub(crate) const QUIET_LISTEN_LIMIT_MS: i64 = 6_000;
// How long a quiet wake on the streaming recogniser waits for a partial
// holding the name. The whole pre-roll, "Hey Ari" in it, is decoded the
// moment the listen arms, so a real call names itself in the first partials;
// the wake log's early-name lines show how soon. The wait is also how long
// the wake word stays off for a false wake nobody sees.
pub(crate) const QUIET_NAME_WAIT_MS: i64 = 3_000;
// How long a spoken answer stays on the overlay at minimum, for replies short
// enough that speech finishes before the user has read them.
pub(crate) const MIN_RESPONSE_DWELL_MS: i64 = 3_000;
// 2 s pre-roll snapshot at session start. Taken at start rather than at
// detection: opening the overlay costs a few hundred ms, but the ring holds
// 2 s and "Hey Ari" is about 0.7 s, so the phrase is still inside the window.
pub(crate) const PREROLL_CAPTURE_SECONDS: f32 = 2.0;
// How long to flash the corrected transcript before the answer. Long enough
// for the user to notice the text changed, short enough not to feel like a
// stall.
pub(crate) const CORRECTION_FLASH_MS: i64 = 600;
// Re-arm cue volume: clearly audible after Ari's speech, a touch below the
// fresh cue's full volume, so it reads as "still here" rather than "fresh
// start". The old 0.3 was inaudible on the phone.
pub(crate) const RE_ARM_CUE_VOLUME: f32 = 0.8;
const READY_CUE_VOLUME: f32 = 1.0;
// How long a spoken typed turn (dictated into the chat) waits on the engine
// before Ari says "one moment": not over a fast answer, but the only cue there
// is when the screen isn't being watched.
const STILL_WORKING_DELAY_MS: i64 = 4_000;
// How long an error stays on the overlay before it closes.
const ERROR_SHOWN_MS: i64 = 2_500;
const SILENCE_POLL_MS: i64 = 1_000;
const DICTATION_POLL_MS: i64 = 500;

/// What the speech-to-text side can do right now.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct SttStatus {
    /// A local model is warm, or the cloud or the host transcribes, which
    /// need nothing loaded.
    pub ready: bool,
    /// The loaded recogniser streams: its partials are a usable liveness
    /// signal, and a retry decode means something.
    pub streaming: bool,
    /// The early name check can run (cloud transcription).
    pub early_check: bool,
    pub model_id: Option<String>,
}

/// Blocking work, done off the session thread.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Job {
    LoadSpeechModel,
    EarlyNameCheck(Vec<i16>),
    Redecode(Vec<i16>),
    Intercept { text: String, cards: bool },
    HandleAction { json: String, skill_id: String },
}

/// Engine calls, made in order on one worker. `stream` hands an on-device
/// answer back while it is written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum EngineCall {
    Respond { input: String, alternative: Option<String>, stream: bool },
    CancelPendingReply,
    SetConversationActive(bool),
}

/// A job's or the engine's result.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Finished {
    ModelLoaded(bool),
    NameChecked(Option<EarlyNameCheck>),
    Redecoded(Option<String>),
    Intercepted(Intercepted),
    ActionHandled(ActionOutcome),
    Responded(EngineReply, bool),
    AnswerText(String, Option<String>),
    AnswerSentence(String),
    /// The work failed: a host callback or the engine threw.
    Failed(String),
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Capture {
    Wake { pcm: Vec<i16>, raw: String, hook: WakeCaptureHook },
    Utterance { pcm: Vec<i16>, capture: UtteranceCapture },
}

/// Whose work a job or engine call is: the voice session in a given epoch,
/// or a typed turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Tag {
    Voice(u64),
    Typed(u64),
}

/// The session's side effects.
pub(crate) trait SessionIo {
    fn now(&self) -> i64;
    fn host(&self) -> &dyn SessionHost;
    fn log(&self, level: LogLevel, message: &str);
    fn locale(&self) -> String;
    fn stt(&self, host_transcribes: bool) -> SttStatus;
    /// Start a listen; its events come back tagged with the id returned.
    fn start_listen(&self, params: ListenParams, host_transcribes: bool) -> u64;
    fn stop_listen(&self);
    fn peek_recent(&self, seconds: f32) -> Vec<i16>;
    fn wake_outcome(&self, outcome: WakeOutcome, hidden: bool);
    fn listen_ended(&self, ended: ListenEnded);
    fn early_name(&self, result: EarlyNameCheck, after_ms: i64, in_time: bool);
    fn run(&self, job: Job, tag: Tag);
    fn engine(&self, call: EngineCall, tag: Tag);
    fn capture(&self, dir: PathBuf, capture: Capture);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Flow {
    /// Reading speech-to-text events.
    Collecting,
    /// A cold start waits for the speech model.
    ColdLoading,
    /// A cold start says "say that again".
    ColdRepeat,
    /// An error is on the overlay until it closes.
    ErrorShown,
    /// A final transcript is being dispatched.
    Turn,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stage {
    Intercept,
    Respond,
    RetryParallel,
    Redecode,
    RetryOffline,
    Flash,
    Action,
    Speaking,
    Dwell,
}

/// What to do when the speech being waited on finishes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AfterSpeech {
    ColdRepeat,
    ExitTalk,
    Rearm,
    Dwell { spoken_from: i64, cards: bool },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TimerKind {
    SilencePoll { watcher: u64 },
    QuietNameWait { generation: u64 },
    ErrorDismiss,
    CorrectionFlash,
    Dwell { cards: bool },
    /// A typed turn's "one moment". Outlives a dismiss: typed turns aren't
    /// the voice session's.
    StillWorking { turn: u64 },
}

#[derive(Debug, Clone, Copy)]
struct Timer {
    at: i64,
    seq: u64,
    epoch: u64,
    kind: TimerKind,
}

#[derive(Debug, Clone, Copy)]
struct Watcher {
    id: u64,
    timeout_ms: i64,
    poll_ms: i64,
}

/// One final transcript on its way through dispatch, and the capture record
/// it fills in as it goes.
struct Turn {
    kind: UtteranceTurn,
    text: String,
    parallel: Option<String>,
    audio: Option<Vec<i16>>,
    raw: Option<String>,
    outcome: UtteranceOutcome,
    response: Option<String>,
    offline: Option<String>,
    used: String,
    stage: Stage,
    reply: Option<EngineReply>,
    used_alternative: bool,
    /// The last sentence spoken while the answer was written, and whether it
    /// has finished.
    sentence: Option<u64>,
    sentence_done: bool,
    response_text: String,
    cards: bool,
    attachments: bool,
    bubble: Option<String>,
    skill_id: Option<String>,
}

impl Turn {
    fn new(kind: UtteranceTurn, text: String, parallel: Option<String>, audio: Option<Vec<i16>>, raw: Option<String>) -> Self {
        Self {
            kind,
            used: text.clone(),
            text,
            parallel,
            audio,
            raw,
            outcome: UtteranceOutcome::Blank,
            response: None,
            offline: None,
            stage: Stage::Intercept,
            reply: None,
            used_alternative: false,
            sentence: None,
            sentence_done: false,
            response_text: String::new(),
            cards: false,
            attachments: false,
            bubble: None,
            skill_id: None,
        }
    }
}

/// A typed turn from the chat, on its way through the engine.
struct Typed {
    text: String,
    /// Dictated rather than typed: Ari says the answer.
    spoken: bool,
    /// The answer was shown while it was written.
    streamed: bool,
    /// A sentence of it was spoken while it was written.
    sentence_spoken: bool,
    reply: Option<EngineReply>,
    skill_id: Option<String>,
}

pub(crate) struct Core {
    config: SessionConfig,
    state: SessionState,
    capture_mode: CaptureMode,
    epoch: u64,
    active: bool,
    dictation: bool,
    flow: Flow,

    // Last moment user or recogniser activity was seen. Read by the silence
    // watcher; reset by a re-armed reply so it gets a full fresh window.
    last_activity_at: i64,
    // True between a skill asking a question (rearm) and the next final
    // transcript. While true the card intercept is skipped, so the engine's
    // pending-reply path stays authoritative.
    awaiting_reply: bool,
    // True from a wake-initiated start until that session's first final
    // transcript is accepted. Only the opening turn of a wake session carries
    // the wake phrase in its pre-roll; re-armed reply turns arm with zero
    // rewind and have nothing to verify against.
    //
    // Cleared in exactly three places, all load-bearing: the cold-start
    // branch, the accept path, and dismiss. A re-armed reply does NOT clear
    // it: it doesn't need to, because the accept path always runs first, but
    // that also means deleting any one of the three re-verifies a turn that
    // has no wake phrase to find.
    //
    // Kept apart from `opened_by_wake` on purpose: the two diverge (the
    // cold-start branch clears this while the turn was still opened by a wake).
    verify_wake_pending: bool,
    // `start`'s own verify_wake: did a wake word open this mic? Never changes
    // for the session.
    opened_by_wake: bool,
    // Pre-detection audio held for the duration of an unverified wake turn.
    // Dropped the moment the turn is accepted; kept if it dies silently.
    wake_preroll: Option<Vec<i16>>,
    // True from a wake-initiated start until that wake's ending is in the
    // wake log. Every wake gets exactly one ending: the first path to reach
    // one writes it, and dismiss supplies Dismissed for any wake nothing else
    // ended. Apart from verify_wake_pending because the cold-start branch ends
    // the wake (ColdStart) while verification is still pending.
    wake_outcome_pending: bool,
    // True while a quiet wake runs unseen: no overlay, no chime, and nothing
    // counted towards the false-wake warning, since nobody saw it. Cleared by
    // reveal the moment a name check passes, early, on a streaming partial or
    // on the whole transcript, or by dismiss.
    hidden: bool,
    // True once a quiet wake's early name check heard the name and showed
    // Ari, before the whole transcript was in. See wake_name_verdict.
    name_heard_early: bool,
    // "Let's talk" continuous mode: while true, every turn re-arms the mic
    // (no wake word) until an exit phrase, 30 s of silence, or an error.
    talk_mode: bool,
    // Whether barge-in is active for the current talk session: the setting on
    // and a hardware echo canceller present. Decided once, on entering talk mode.
    barge_in_active: bool,
    // True while Ari's speech plays in a barge-in turn: the mic is armed
    // meanwhile, so a final while this is set is a barge.
    speaking: bool,
    // The barge-in turn's speech, which a barge cuts off.
    tts_job: Option<u64>,
    // The listen now open started while Ari was speaking in a barge-in turn,
    // so Ari's own voice may be in it. Stays set when the speech ends: the
    // listen's transcript can still be the tail of Ari's answer.
    echo_risk: bool,
    // Bumped whenever a turn is superseded (fresh arm or dismiss);
    // snapshotted at arm time into armed_generation. A final whose
    // armed_generation no longer matches turn_generation is a straggler from
    // a turn the session left behind. A barge does NOT bump the generation,
    // so the barge final is never dropped.
    turn_generation: u64,
    armed_generation: u64,
    armed_at: i64,
    last_partial: String,
    early_check: Option<(u64, i64)>,

    listen: Option<u64>,
    watcher: Option<Watcher>,
    next_watcher: u64,
    timers: Vec<Timer>,
    next_timer: u64,
    next_speech: u64,
    awaiting_speech: Option<(u64, AfterSpeech)>,
    pending_stt: VecDeque<(u64, SttEvent)>,
    turn: Option<Turn>,
    typed: BTreeMap<u64, Typed>,
    next_typed: u64,
}

impl Core {
    pub(crate) fn new() -> Self {
        Self {
            config: SessionConfig::default(),
            state: SessionState::Idle { keep_cards: false },
            capture_mode: CaptureMode::Normal,
            epoch: 0,
            active: false,
            dictation: false,
            flow: Flow::Collecting,
            last_activity_at: 0,
            awaiting_reply: false,
            verify_wake_pending: false,
            opened_by_wake: false,
            wake_preroll: None,
            wake_outcome_pending: false,
            hidden: false,
            name_heard_early: false,
            talk_mode: false,
            barge_in_active: false,
            speaking: false,
            tts_job: None,
            echo_risk: false,
            turn_generation: 0,
            armed_generation: 0,
            armed_at: 0,
            last_partial: String::new(),
            early_check: None,
            listen: None,
            watcher: None,
            next_watcher: 0,
            timers: Vec::new(),
            next_timer: 0,
            next_speech: 0,
            awaiting_speech: None,
            pending_stt: VecDeque::new(),
            turn: None,
            typed: BTreeMap::new(),
            next_typed: 0,
        }
    }

    pub(crate) fn configure(&mut self, config: SessionConfig) {
        self.config = config;
    }

    pub(crate) fn is_active(&self) -> bool {
        self.active
    }

    fn tag(&self) -> Tag {
        Tag::Voice(self.epoch)
    }

    fn set_state(&mut self, io: &dyn SessionIo, state: SessionState) {
        if self.state != state {
            self.state = state.clone();
            io.host().state(state);
        }
    }

    fn set_capture_mode(&mut self, io: &dyn SessionIo, mode: CaptureMode) {
        if self.capture_mode != mode {
            self.capture_mode = mode;
            io.host().capture_mode(mode);
        }
    }

    // ── Timers ──────────────────────────────────────────────────────────

    fn schedule(&mut self, io: &dyn SessionIo, delay_ms: i64, kind: TimerKind) {
        self.next_timer += 1;
        self.timers.push(Timer { at: io.now() + delay_ms, seq: self.next_timer, epoch: self.epoch, kind });
    }

    pub(crate) fn next_deadline(&self) -> Option<i64> {
        self.timers.iter().map(|t| t.at).min()
    }

    /// Run every timer that is due, in the order they fall due.
    pub(crate) fn fire_due(&mut self, io: &dyn SessionIo) {
        loop {
            let now = io.now();
            let due = self
                .timers
                .iter()
                .enumerate()
                .filter(|(_, t)| t.at <= now)
                .min_by_key(|(_, t)| (t.at, t.seq))
                .map(|(i, _)| i);
            let Some(index) = due else { return };
            let timer = self.timers.remove(index);
            if let TimerKind::StillWorking { turn } = timer.kind {
                self.still_working(io, turn);
            } else if timer.epoch == self.epoch {
                self.on_timer(io, timer.kind);
            }
        }
    }

    fn on_timer(&mut self, io: &dyn SessionIo, kind: TimerKind) {
        match kind {
            TimerKind::SilencePoll { watcher } => self.on_silence_poll(io, watcher),
            TimerKind::QuietNameWait { generation } => self.on_quiet_name_wait(io, generation),
            TimerKind::ErrorDismiss => self.dismiss(io, false),
            TimerKind::CorrectionFlash => self.present(io),
            TimerKind::Dwell { cards } => self.dismiss(io, cards),
            TimerKind::StillWorking { .. } => {}
        }
    }

    // ── Starting ────────────────────────────────────────────────────────

    /// Begin a voice session; false if one is already running, which is left
    /// alone rather than stomped on.
    ///
    /// `verify_wake` must be true ONLY for a wake-word detection: it subjects
    /// the first final transcript to [`should_accept_wake`], which relies on
    /// the wake phrase being in the pre-roll.
    ///
    /// `quiet` asks for a quiet wake, started by the wake service rather than
    /// by the overlay: no overlay and no chime until a name check has passed,
    /// so a false wake shows nothing at all. On cloud transcription the 2 s
    /// ending at the wake are checked at once, on their own, so a real call
    /// shows while it is still being spoken; otherwise the whole transcript is
    /// checked. The streaming recogniser shows the overlay, with the chime, on
    /// the first partial that holds the name, and turns the wake away unseen
    /// if none has by [`QUIET_NAME_WAIT_MS`]. A cold start opens the overlay
    /// at once, as an ordinary wake.
    pub(crate) fn start(&mut self, io: &dyn SessionIo, verify_wake: bool, quiet: bool) -> bool {
        if self.active {
            io.log(LogLevel::Warn, "VoiceSession.start() called while already active — ignoring");
            return false;
        }
        self.active = true;
        self.dictation = false;
        self.opened_by_wake = verify_wake;
        self.verify_wake_pending = verify_wake;
        self.wake_outcome_pending = verify_wake;
        self.wake_preroll = verify_wake.then(|| io.peek_recent(PREROLL_CAPTURE_SECONDS));
        // A fresh wake cancels any reply the engine was still waiting on from
        // a previous turn: the user has clearly moved on.
        io.engine(EngineCall::CancelPendingReply, self.tag());
        self.awaiting_reply = false;
        self.talk_mode = false;
        // The first state goes out before this returns, so the overlay never
        // sees Idle and finishes itself (the lock-screen race): Listening when
        // warm, Preparing when cold.
        let stt = io.stt(self.config.host_transcribes);
        let model_loaded = stt.ready;
        self.hidden = quiet && verify_wake && model_loaded;
        self.name_heard_early = false;
        if quiet && !self.hidden {
            io.host().reveal();
        }
        if model_loaded {
            self.set_state(io, SessionState::Listening { partial: String::new() });
            // The mic is already open and the ring has held the user's first
            // words since before the wake fired. Arming is instant.
            let params = ListenParams {
                rewind_seconds: DEFAULT_REWIND_SECONDS,
                give_up_without_speech_ms: self.hidden.then_some(QUIET_GIVE_UP_MS),
                limit_ms: self.hidden.then_some(QUIET_LISTEN_LIMIT_MS),
            };
            self.start_listening(io, params);
            self.after_arm(io);
        } else {
            let wait = io.host().phrase(Phrase::PleaseWait);
            self.set_state(io, SessionState::Preparing { message: wait.clone() });
            self.log_wake_outcome(io, WakeOutcome::ColdStart);
            // Cold start: acknowledge, wait for the warm-up, then ask the
            // user to repeat: their words have aged out of the 2 s ring, so
            // there's nothing to transcribe.
            let id = self.speech_id();
            io.host().speak(id, &wait, false);
            self.flow = Flow::ColdLoading;
            io.run(Job::LoadSpeechModel, self.tag());
        }
        true
    }

    fn start_listening(&mut self, io: &dyn SessionIo, params: ListenParams) {
        self.listen = Some(io.start_listen(params, self.config.host_transcribes));
        self.echo_risk = false;
    }

    /// What follows arming the opening listen, warm or after a cold start.
    fn after_arm(&mut self, io: &dyn SessionIo) {
        self.flow = Flow::Collecting;
        if !self.hidden {
            self.ready_cue(io);
        }
        // A fresh arm supersedes any prior turn: bump the generation and
        // snapshot it, so a straggling final from before this listen reads
        // as stale.
        self.turn_generation += 1;
        self.armed_generation = self.turn_generation;
        let stt = io.stt(self.config.host_transcribes);

        // A quiet wake asks about the name at once, on the 2 s that end at the
        // wake, so a real call shows itself while the user is still talking
        // rather than after the whole transcript: about 1 s after the wake
        // instead of 3 on the test phone. A miss changes nothing; the whole
        // transcript decides as before.
        if self.hidden && stt.early_check
            && let Some(clip) = &self.wake_preroll
        {
            self.early_check = Some((self.armed_generation, io.now()));
            io.run(Job::EarlyNameCheck(clip.clone()), self.tag());
        }

        // The streaming recogniser asks on every partial instead, and a wake
        // that has not named itself in time is turned away unseen. A name
        // heard later could not pass the check anyway: it only looks at the
        // first words.
        self.armed_at = io.now();
        self.last_partial.clear();
        if self.hidden && stt.streaming {
            self.schedule(io, QUIET_NAME_WAIT_MS, TimerKind::QuietNameWait { generation: self.armed_generation });
        }

        self.last_activity_at = io.now();
        // A wake turn gets a much shorter window: you just said the wake
        // phrase, so silence means it wasn't you, and every second the mic
        // stays armed after a false accept is a second ambient speech could be
        // mistaken for a command.
        //
        // Streaming path only. The short window measures silence, and only
        // the streaming recogniser gives silence to measure: it refreshes
        // last_activity_at on every non-blank partial. The buffered paths emit
        // no partials between arm and endpoint, so there 8 s would be a hard
        // cap on the whole utterance and would cut Italian speakers off
        // mid-sentence. They keep the 30 s window.
        //
        // Reads opened_by_wake, not verify_wake_pending, and that is
        // deliberate: the timeout asks "did a wake word open this mic?",
        // which never changes for the turn. The pending flag asks "is
        // verification still pending?", which the cold-start branch clears,
        // and cold start is the one path where should_accept_wake never gates
        // the transcript at all, so reading it would hand the least-defended
        // path the longest open mic.
        let timeout = if self.opened_by_wake && stt.streaming { WAKE_TURN_SILENCE_TIMEOUT_MS } else { SILENCE_TIMEOUT_MS };
        self.launch_watcher(io, timeout);
    }

    fn ready_cue(&self, io: &dyn SessionIo) {
        io.host().play_cue(READY_CUE_VOLUME);
    }

    // ── Silence ─────────────────────────────────────────────────────────

    fn launch_watcher(&mut self, io: &dyn SessionIo, timeout_ms: i64) {
        self.next_watcher += 1;
        let poll_ms = if self.dictation { DICTATION_POLL_MS } else { SILENCE_POLL_MS };
        let watcher = Watcher { id: self.next_watcher, timeout_ms, poll_ms };
        self.watcher = Some(watcher);
        self.schedule(io, poll_ms, TimerKind::SilencePoll { watcher: watcher.id });
    }

    fn cancel_watcher(&mut self) {
        self.watcher = None;
    }

    fn on_silence_poll(&mut self, io: &dyn SessionIo, id: u64) {
        let Some(watcher) = self.watcher.filter(|w| w.id == id) else { return };
        if io.now() - self.last_activity_at <= watcher.timeout_ms {
            self.schedule(io, watcher.poll_ms, TimerKind::SilencePoll { watcher: id });
            return;
        }
        if self.dictation {
            io.log(LogLevel::Info, &format!("Dictation: no speech within {}ms — dismissing", watcher.timeout_ms));
            self.dismiss(io, false);
            return;
        }
        io.log(LogLevel::Info, &format!("No speech detected within {} ms — dismissing", watcher.timeout_ms));
        // Guarded on the pre-roll, not on verify_wake_pending: it is only
        // ever held for a turn whose audio genuinely contains a wake phrase
        // nothing acted on, so this can never file a real "Hey Ari" as a hard
        // negative. The cold-start branch clears both for exactly that reason.
        if let Some(preroll) = self.wake_preroll.clone() {
            self.capture_false_trigger(io, Some(preroll), String::new(), WakeCaptureHook::Silent);
            // Counted on the same guard that decides this is a hard negative,
            // and for the same reason: a pre-roll is only held for a turn a
            // wake phrase opened, so nothing here can count a tap-to-talk the
            // user simply walked away from. The count is kept even when the
            // audio is not: it is the event that prices the wake word, and
            // the recording is only for retraining it.
            self.end_wake(io, WakeOutcome::Silent);
        }
        self.dismiss(io, false);
    }

    fn on_quiet_name_wait(&mut self, io: &dyn SessionIo, generation: u64) {
        if !self.hidden || !self.wake_outcome_pending || self.turn_generation != generation {
            return;
        }
        io.log(
            LogLevel::Warn,
            &format!("Wake rejected: no name within {QUIET_NAME_WAIT_MS} ms, partial='{}'", self.last_partial),
        );
        let heard_speech = !self.last_partial.trim().is_empty();
        let (hook, outcome) = if heard_speech {
            (WakeCaptureHook::Rejected, WakeOutcome::Rejected)
        } else {
            (WakeCaptureHook::Silent, WakeOutcome::Silent)
        };
        self.capture_false_trigger(io, self.wake_preroll.clone(), self.last_partial.clone(), hook);
        self.end_wake(io, outcome);
        self.dismiss(io, false);
    }

    // ── Speech-to-text ──────────────────────────────────────────────────

    /// An event from listen `listen`. Dropped unless it is the session's
    /// current listen; held while a turn is being dispatched.
    pub(crate) fn stt_event(&mut self, io: &dyn SessionIo, listen: u64, event: SttEvent) {
        if !self.active || self.listen != Some(listen) {
            return;
        }
        if self.flow != Flow::Collecting {
            self.pending_stt.push_back((listen, event));
            return;
        }
        if matches!(event, SttEvent::Done { .. } | SttEvent::NoSpeech { .. } | SttEvent::Error(_)) {
            self.listen = None;
        }
        if self.dictation {
            self.dictation_event(io, event);
        } else {
            self.session_event(io, event);
        }
    }

    fn drain_pending(&mut self, io: &dyn SessionIo) {
        while self.flow == Flow::Collecting {
            let Some((listen, event)) = self.pending_stt.pop_front() else { return };
            self.stt_event(io, listen, event);
        }
    }

    fn session_event(&mut self, io: &dyn SessionIo, event: SttEvent) {
        match event {
            SttEvent::Listening { partial, name_matched } => {
                if !partial.trim().is_empty() {
                    self.last_activity_at = io.now();
                }
                self.last_partial = partial.clone();
                if self.hidden && partial_reveals_wake(name_matched) {
                    // Not name_heard_early: the whole transcript is this
                    // partial with more on the end, so it is the better
                    // judge. A partial can end mid-word, "Hey Ari" on the way
                    // to "Hey Arianna".
                    let result = if name_matched == Some(true) { EarlyNameCheck::Pass } else { EarlyNameCheck::None };
                    io.early_name(result, io.now() - self.armed_at, true);
                    self.reveal(io);
                    self.ready_cue(io);
                }
                self.set_state(io, SessionState::Listening { partial });
            }
            SttEvent::Transcribing => {
                // A buffered decode could take seconds: show Ari is working,
                // not still listening.
                self.cancel_watcher();
                self.set_state(io, SessionState::Thinking);
            }
            SttEvent::Done { text, parallel, audio, raw, name_matched, listen_end, speech_heard } => {
                self.on_final(io, text, parallel, audio, raw, name_matched, listen_end, speech_heard)
            }
            SttEvent::Error(error) => {
                // A quiet wake shows itself for an error: the failed request
                // may have been a real one, and silence would read as Ari
                // ignoring them.
                self.reveal(io);
                self.log_wake_outcome(io, WakeOutcome::Error);
                self.set_state(io, SessionState::Error(SessionError::Stt(error)));
                self.cancel_watcher();
                self.show_error_then_dismiss(io);
            }
            SttEvent::NoSpeech { audio, listen_end } => {
                // Shown like any other error, as it always was; only a wake
                // that heard nothing is kept and counted instead of vanishing.
                self.cancel_watcher();
                self.log_wake_listen_end(io, listen_end);
                if self.name_heard_early && self.verify_wake_pending {
                    // The early check heard the name in this same audio and
                    // Ari is showing: the user said "Hey Ari" and waits for
                    // the chime.
                    self.log_wake_outcome(io, WakeOutcome::Accepted);
                    let preroll = self.wake_preroll.take();
                    self.capture_accepted_wake(io, preroll, String::new());
                    self.verify_wake_pending = false;
                    self.listen_for_command(io);
                    self.launch_watcher(io, SILENCE_TIMEOUT_MS);
                    return;
                }
                if self.verify_wake_pending {
                    let pcm = audio.or_else(|| self.wake_preroll.clone());
                    self.on_no_speech_wake(io, pcm);
                }
                if self.hidden {
                    // Nothing was ever shown, so there is nothing to explain.
                    self.dismiss(io, false);
                    return;
                }
                self.set_state(io, SessionState::Error(SessionError::NoSpeech));
                self.show_error_then_dismiss(io);
            }
        }
    }

    fn show_error_then_dismiss(&mut self, io: &dyn SessionIo) {
        self.flow = Flow::ErrorShown;
        self.schedule(io, ERROR_SHOWN_MS, TimerKind::ErrorDismiss);
    }

    #[allow(clippy::too_many_arguments)]
    fn on_final(
        &mut self,
        io: &dyn SessionIo,
        text: String,
        parallel: Option<String>,
        audio: Option<Vec<i16>>,
        raw: Option<String>,
        name_matched: Option<bool>,
        listen_end: Option<ListenEnded>,
        speech_heard: Option<bool>,
    ) {
        let final_generation = self.armed_generation;
        self.cancel_watcher();
        if self.echo_risk && speech_heard == Some(false) {
            // Ari's own answer, through the echo canceller: the recogniser
            // made words of it, the speech detector heard nobody, and taken as
            // the user's reply it had Ari answer itself ("A spirit of speak
            // of", "Nice", 2026-11-22). Keep listening; Ari isn't cut off.
            io.log(
                LogLevel::Info,
                &format!("Ignored '{}' while Ari spoke: no speech in it", raw.as_deref().unwrap_or(&text)),
            );
            self.set_state(io, SessionState::Listening { partial: String::new() });
            self.start_listening(io, ListenParams { rewind_seconds: 0.0, ..ListenParams::default() });
            self.echo_risk = self.speaking;
            self.launch_watcher(io, SILENCE_TIMEOUT_MS);
            return;
        }
        if should_cut_tts(self.speaking, self.barge_in_active) {
            // The user barged in: stop Ari at once. This utterance becomes the
            // next turn: a barge does NOT bump the generation, so it isn't stale.
            io.host().stop_speaking();
            self.speaking = false;
            self.tts_job = None;
        }
        if is_stale_turn(final_generation, self.turn_generation) {
            // A straggler from a superseded turn: ignore it so it can't fire
            // a phantom turn, and keep listening if a reply is still awaited.
            if self.awaiting_reply {
                self.launch_watcher(io, SILENCE_TIMEOUT_MS);
            }
            return;
        }
        self.log_wake_listen_end(io, listen_end);
        if !self.name_heard_early && is_no_speech_wake(self.verify_wake_pending, raw.as_deref()) {
            // Whisper's way of saying it heard nothing. The overlay closes as
            // quietly as it did when this went down the accept path.
            io.log(LogLevel::Info, "Wake heard no speech");
            let pcm = audio.or_else(|| self.wake_preroll.clone());
            self.on_no_speech_wake(io, pcm);
            self.dismiss(io, false);
            return;
        }
        if !should_accept_wake(
            self.verify_wake_pending,
            raw.as_deref(),
            wake_name_verdict(self.name_heard_early, name_matched),
        ) {
            io.log(LogLevel::Warn, &format!("Wake rejected: raw='{}'", raw.as_deref().unwrap_or("null")));
            // Falls back to the pre-roll: the buffered path reports no audio,
            // and English on Whisper is a configuration that forms a verdict,
            // so without this the rejection that most wants capturing would
            // capture nothing.
            let pcm = audio.or_else(|| self.wake_preroll.clone());
            self.capture_false_trigger(io, pcm, raw.unwrap_or_default(), WakeCaptureHook::Rejected);
            self.end_wake(io, WakeOutcome::Rejected);
            self.dismiss(io, false);
            return;
        }
        // A quiet wake that heard only "Hey Ari": the user paused for the
        // chime they never got.
        let bare_wake = (self.hidden || self.name_heard_early) && text.trim().is_empty();
        self.reveal(io);
        // An accepted wake is a confirmed true positive: keep the wake moment
        // (the pre-roll) before it is dropped. The whole utterance is the
        // utterance capture's job.
        self.log_wake_outcome(io, WakeOutcome::Accepted);
        let preroll = self.wake_preroll.take();
        self.capture_accepted_wake(io, preroll, raw.clone().unwrap_or_default());
        self.verify_wake_pending = false;
        if bare_wake {
            self.listen_for_command(io);
            self.launch_watcher(io, SILENCE_TIMEOUT_MS);
            return;
        }
        self.handle_final_text(io, text, parallel, audio, raw);
    }

    fn log_wake_listen_end(&self, io: &dyn SessionIo, ended: Option<ListenEnded>) {
        if self.wake_outcome_pending
            && let Some(ended) = ended
        {
            io.listen_ended(ended);
        }
    }

    // ── A turn ──────────────────────────────────────────────────────────

    /// Dispatch one final transcript. The turn kind is sampled before
    /// dispatch (it reads state the dispatch changes), and the capture is
    /// written when the turn ends however it ends: a blank transcript, an
    /// intercept, a dismiss halfway.
    fn handle_final_text(
        &mut self,
        io: &dyn SessionIo,
        text: String,
        parallel: Option<String>,
        audio: Option<Vec<i16>>,
        raw: Option<String>,
    ) {
        let kind = utterance_turn(self.talk_mode, self.awaiting_reply);
        let blank = text.trim().is_empty();
        self.turn = Some(Turn::new(kind, text.clone(), parallel, audio, raw));
        self.flow = Flow::Turn;
        if blank {
            self.dismiss(io, false);
            return;
        }
        self.set_state(io, SessionState::Thinking);
        // "Stop" over a ringing alarm silences it, ahead of everything else,
        // a pending reply included: the dismiss drops that. While re-armed for
        // a skill's question, the engine holds the pending reply and is
        // authoritative, so the card shortcut is skipped and the answer flows
        // back through the engine.
        io.run(Job::Intercept { text, cards: !self.awaiting_reply }, self.tag());
    }

    fn turn_mut(&mut self) -> &mut Turn {
        self.turn.as_mut().expect("a turn in dispatch")
    }

    /// A worker's or the engine's result, from epoch `epoch`.
    pub(crate) fn finished(&mut self, io: &dyn SessionIo, tag: Tag, result: Finished) {
        let epoch = match tag {
            Tag::Voice(epoch) => epoch,
            Tag::Typed(turn) => return self.typed_result(io, turn, result),
        };
        if epoch != self.epoch || !self.active {
            return;
        }
        match result {
            Finished::ModelLoaded(ok) => self.on_model_loaded(io, ok),
            Finished::NameChecked(result) => self.on_name_checked(io, result),
            Finished::Failed(message) => self.failed(io, &message),
            other => self.turn_result(io, other),
        }
    }

    fn turn_result(&mut self, io: &dyn SessionIo, result: Finished) {
        let Some(stage) = self.turn.as_ref().map(|t| t.stage) else { return };
        match (stage, result) {
            (Stage::Intercept, Finished::Intercepted(intercepted)) => self.on_intercepted(io, intercepted),
            (Stage::Respond, Finished::AnswerText(so_far, _)) => {
                self.set_state(io, SessionState::Responding { text: so_far, cards: false });
            }
            (Stage::Respond, Finished::AnswerSentence(sentence)) => {
                // An on-device answer is spoken a sentence at a time while
                // the model writes it. The first sentence cuts off whatever
                // is playing; the rest queue behind it.
                let id = self.speech_id();
                let turn = self.turn_mut();
                let flush = turn.sentence.is_none();
                turn.sentence = Some(id);
                turn.sentence_done = false;
                io.host().speak(id, &sentence, flush);
            }
            (Stage::Respond, Finished::Responded(reply, used_alternative)) => self.on_responded(io, reply, used_alternative),
            (Stage::RetryParallel, Finished::Responded(retry, _)) => {
                if !matches!(retry, EngineReply::NotUnderstood { .. }) {
                    io.log(LogLevel::Info, "Retry succeeded with parallel transcript");
                    let turn = self.turn_mut();
                    turn.reply = Some(retry);
                    turn.used = turn.parallel.clone().unwrap_or_default();
                }
                self.offline_retry(io);
            }
            (Stage::Redecode, Finished::Redecoded(offline)) => self.on_redecoded(io, offline),
            (Stage::RetryOffline, Finished::Responded(retry, _)) => {
                if !matches!(retry, EngineReply::NotUnderstood { .. }) {
                    io.log(LogLevel::Info, "Offline retry succeeded");
                    let turn = self.turn_mut();
                    turn.reply = Some(retry);
                    turn.used = turn.offline.clone().unwrap_or_default();
                }
                self.conclude(io);
            }
            (Stage::Action, Finished::ActionHandled(outcome)) => {
                let turn = self.turn_mut();
                turn.attachments = outcome.attachments;
                turn.cards = outcome.cards;
                turn.bubble = outcome.display_text;
                turn.response_text = outcome.text;
                self.respond(io);
            }
            _ => {}
        }
    }

    fn on_intercepted(&mut self, io: &dyn SessionIo, intercepted: Intercepted) {
        let heard = self.turn_mut().text.clone();
        match intercepted {
            Intercepted::Alert => {
                self.turn_mut().outcome = UtteranceOutcome::Alert;
                io.host().turn_logged(LoggedTurn { clear: false, heard, reply: None });
                self.dismiss(io, false);
            }
            Intercepted::CardSilent => {
                self.turn_mut().outcome = UtteranceOutcome::Card;
                io.host().turn_logged(LoggedTurn { clear: false, heard, reply: None });
                self.dismiss(io, false);
            }
            Intercepted::CardSpoken { text, cards, attachments } => {
                let turn = self.turn_mut();
                turn.outcome = UtteranceOutcome::Card;
                // A fresh reply: nothing was spoken a sentence at a time.
                turn.sentence = None;
                io.host().turn_logged(LoggedTurn { clear: false, heard, reply: card_reply(&text, attachments) });
                self.set_state(io, SessionState::Responding { text: text.clone(), cards });
                self.speak_then_dismiss(io, &text, cards);
            }
            Intercepted::No => {
                // From here on this is a normal engine dispatch. If the engine
                // asks another question, the rearm turns this back on.
                self.awaiting_reply = false;
                let turn = self.turn_mut();
                turn.stage = Stage::Respond;
                // Layer 1b, the parallel transcript before the assistant: the
                // engine acts on it instead when this one finds no skill and
                // that one does. An on-device assistant answers anything, so
                // the not-understood retries below would never get a turn.
                // "reminded me to buy milk" found nothing; the parallel
                // stream's "remind me to buy milk" found the reminders skill.
                let alternative = turn.parallel.clone().filter(|p| !p.trim().is_empty() && *p != turn.text);
                let call = EngineCall::Respond { input: turn.text.clone(), alternative, stream: true };
                io.engine(call, self.tag());
            }
        }
    }

    fn on_responded(&mut self, io: &dyn SessionIo, reply: EngineReply, used_alternative: bool) {
        let turn = self.turn_mut();
        turn.used_alternative = used_alternative;
        if used_alternative {
            turn.used = turn.parallel.clone().unwrap_or_default();
        }
        let not_understood = matches!(reply, EngineReply::NotUnderstood { .. });
        turn.reply = Some(reply);
        let (text, parallel) = (turn.text.clone(), turn.parallel.clone());
        if used_alternative {
            io.log(
                LogLevel::Info,
                &format!("Engine acted on the parallel transcript '{}' over '{text}'", parallel.as_deref().unwrap_or("")),
            );
        }

        // Layers 2 and 3 apply to the streaming path only. The buffered path
        // sees the whole utterance before committing any token, so there is
        // nothing for a parallel decoder or a re-decode to improve. Layer 2
        // skips itself there (no parallel transcript); layer 3 is gated on
        // streaming, because the buffered path does hand over its audio, for
        // the capture, not as an invitation to decode it a second time.

        // Layer 2: the parallel stream's transcript.
        if let Some(parallel) = parallel.filter(|p| not_understood && !used_alternative && !p.trim().is_empty() && *p != text) {
            io.log(LogLevel::Info, &format!("NotUnderstood for '{text}' — retrying with parallel '{parallel}'"));
            self.turn_mut().stage = Stage::RetryParallel;
            io.engine(EngineCall::Respond { input: parallel, alternative: None, stream: false }, self.tag());
            return;
        }
        self.offline_retry(io);
    }

    /// Layer 3: decode the whole utterance again, finalised.
    fn offline_retry(&mut self, io: &dyn SessionIo) {
        let streaming = io.stt(self.config.host_transcribes).streaming;
        let turn = self.turn_mut();
        let not_understood = matches!(turn.reply, Some(EngineReply::NotUnderstood { .. }));
        if not_understood
            && streaming
            && let Some(audio) = turn.audio.clone()
        {
            io.log(LogLevel::Info, &format!("Parallel also failed — running offline retry ({} samples)", audio.len()));
            self.turn_mut().stage = Stage::Redecode;
            io.run(Job::Redecode(audio), self.tag());
            return;
        }
        self.conclude(io);
    }

    fn on_redecoded(&mut self, io: &dyn SessionIo, offline: Option<String>) {
        let turn = self.turn_mut();
        turn.offline = offline.clone();
        if let Some(offline) = offline
            && !offline.trim().is_empty()
            && offline != turn.text
            && Some(&offline) != turn.parallel.as_ref()
        {
            io.log(LogLevel::Info, &format!("Offline produced '{offline}' — retrying engine"));
            turn.stage = Stage::RetryOffline;
            io.engine(EngineCall::Respond { input: offline, alternative: None, stream: false }, self.tag());
            return;
        }
        self.conclude(io);
    }

    /// The transcript is settled: record it, and flash it first if a retry
    /// corrected it, so the user sees Ari corrected itself before the answer.
    fn conclude(&mut self, io: &dyn SessionIo) {
        let turn = self.turn_mut();
        let Some(reply) = &turn.reply else { return };
        let corrected = turn.used != turn.text;
        turn.outcome = utterance_outcome(reply, corrected);
        turn.response = Some(response_label(reply));
        if corrected {
            turn.stage = Stage::Flash;
            let used = turn.used.clone();
            self.set_state(io, SessionState::Listening { partial: used });
            self.schedule(io, CORRECTION_FLASH_MS, TimerKind::CorrectionFlash);
            return;
        }
        self.present(io);
    }

    /// The words to say: the reply's own, or the action envelope's.
    fn present(&mut self, io: &dyn SessionIo) {
        let Some(turn) = self.turn.as_mut() else { return };
        let Some(reply) = turn.reply.clone() else { return };
        match reply {
            EngineReply::Text { body, .. } => turn.response_text = body,
            EngineReply::Action { json, skill_id, .. } => {
                // Only action replies carry an id; a plain text answer is
                // unattributable, so a report on one names no skill.
                turn.skill_id = Some(skill_id.clone()).filter(|s| !s.trim().is_empty());
                turn.stage = Stage::Action;
                io.run(Job::HandleAction { json, skill_id }, self.tag());
                return;
            }
            EngineReply::Binary { mime, len } => turn.response_text = format!("[Binary: {mime}, {len} bytes]"),
            EngineReply::NotUnderstood { body } => turn.response_text = body,
        }
        self.respond(io);
    }

    /// The rest of the turn: facts, the log, the overlay, then speak and
    /// either re-arm for a reply or close.
    fn respond(&mut self, io: &dyn SessionIo) {
        let Some(turn) = self.turn.as_ref() else { return };
        let Some(reply) = turn.reply.clone() else { return };
        let response_text = turn.response_text.clone();
        let cards = turn.cards;
        // Computed off the final reply, after any retry replaced it.
        let enter = should_enter_conversation(&reply);
        let exit = should_exit_conversation(&reply);
        // Re-arm if the engine wants a reply, or in or entering talk mode.
        let rearm = should_rearm(&reply) || self.talk_mode || enter;

        if should_persist_facts(&reply) {
            // Before the exit branch below, which returns early, so the write
            // back still happens on a turn that both changes facts and exits
            // talk mode.
            io.host().facts_changed();
        }

        // The log gets the transcript Ari acted on, after the retries, not
        // the raw one. Not-understood replies are logged too. A silent
        // envelope (blank text, no attachments) logs no reply. A cleared
        // conversation leaves only Ari's acknowledgement.
        let markdown = match &reply {
            EngineReply::Text { markdown, .. } => markdown.clone(),
            _ => None,
        };
        let logged_reply = (!response_text.trim().is_empty() || turn.attachments).then(|| LoggedReply {
            text: turn.bubble.clone().unwrap_or_else(|| response_text.clone()),
            markdown,
            skill_id: turn.skill_id.clone(),
        });
        io.host().turn_logged(LoggedTurn {
            clear: should_clear_conversation(&reply),
            heard: turn.used.clone(),
            reply: logged_reply,
        });

        self.set_state(io, SessionState::Responding { text: response_text.clone(), cards });

        if exit {
            // Say the acknowledgement, then leave the mode and close.
            let speech = self.reply_speech(io, &response_text);
            self.await_speech(io, speech, AfterSpeech::ExitTalk);
            return;
        }

        if enter {
            self.talk_mode = true;
            io.engine(EngineCall::SetConversationActive(true), self.tag());
            self.barge_in_active = barge_in_effective(self.config.barge_in_enabled, self.config.echo_canceller);
            if self.barge_in_active {
                self.set_capture_mode(io, CaptureMode::Conversation);
            }
        }

        if rearm && self.talk_mode && self.barge_in_active {
            // Barge-in: arm the mic now (the echo canceller removes Ari's
            // voice from the capture) and speak at the same time. A final
            // while speaking cuts Ari off. The re-arm opens the mic with zero
            // rewind and bumps the turn generation.
            self.rearm_for_reply(io);
            self.speaking = true;
            let speech = self.reply_speech(io, &response_text);
            self.tts_job = speech;
            if speech.is_none() {
                self.speaking = false;
            }
            self.echo_risk = self.speaking;
            self.finish_turn(io);
        } else if rearm {
            // Turn-based (a skill's question, or talk mode without an echo
            // canceller): speak first, then open the mic. It waits for speech
            // to finish rather than guessing, or the rewind would take in the
            // tail of Ari's own question and it would answer itself ("…to use,
            // Apple Music"). The re-arm uses zero rewind for the same reason.
            let speech = self.reply_speech(io, &response_text);
            self.await_speech(io, speech, AfterSpeech::Rearm);
        } else {
            self.speak_then_dismiss(io, &response_text, cards);
        }
    }

    /// Say an answer, wait for it, hold the overlay a moment, then close.
    ///
    /// Waits on the speech itself rather than guessing its length from the
    /// character count: the dismiss stops speech, so a reply that outran the
    /// guess (a full forecast easily does) lost its last few words. An answer
    /// already spoken a sentence at a time is waited out.
    ///
    /// An answer with cards leaves them on the overlay after the session
    /// ends; the frontend times them out.
    fn speak_then_dismiss(&mut self, io: &dyn SessionIo, text: &str, cards: bool) {
        let spoken_from = io.now();
        let speech = self.reply_speech(io, text);
        self.await_speech(io, speech, AfterSpeech::Dwell { spoken_from, cards });
    }

    fn speech_id(&mut self) -> u64 {
        self.next_speech += 1;
        self.next_speech
    }

    /// Speak the turn's answer, or, when it was already spoken a sentence at
    /// a time, wait for the last sentence. `None` when there is nothing to
    /// wait for: blank text (speaking blank would cut off whatever else is
    /// playing), or a last sentence already done.
    fn reply_speech(&mut self, io: &dyn SessionIo, text: &str) -> Option<u64> {
        if let Some(turn) = &self.turn
            && let Some(id) = turn.sentence
        {
            return (!turn.sentence_done).then_some(id);
        }
        if text.trim().is_empty() {
            return None;
        }
        let id = self.speech_id();
        io.host().speak(id, text, true);
        Some(id)
    }

    fn await_speech(&mut self, io: &dyn SessionIo, speech: Option<u64>, then: AfterSpeech) {
        if let Some(turn) = self.turn.as_mut() {
            turn.stage = Stage::Speaking;
        }
        match speech {
            Some(id) => self.awaiting_speech = Some((id, then)),
            None => self.after_speech(io, then),
        }
    }

    /// The host reports speech `id` has finished.
    pub(crate) fn speech_finished(&mut self, io: &dyn SessionIo, id: u64) {
        if owns_speaking_flag(self.tts_job, id) {
            self.speaking = false;
        }
        if let Some(turn) = self.turn.as_mut()
            && turn.sentence == Some(id)
        {
            turn.sentence_done = true;
        }
        if self.awaiting_speech.is_some_and(|(awaited, _)| awaited == id) {
            let (_, then) = self.awaiting_speech.take().expect("checked");
            self.after_speech(io, then);
        }
    }

    fn after_speech(&mut self, io: &dyn SessionIo, then: AfterSpeech) {
        match then {
            AfterSpeech::ColdRepeat => {
                self.set_state(io, SessionState::Listening { partial: String::new() });
                // Nothing left to verify against: the wake phrase aged out of
                // the ring with the rest of the original utterance, and the
                // user is repeating a bare command. Same reasoning as the
                // re-armed reply: verify only what carries a wake phrase in
                // its pre-roll.
                //
                // The snapshot goes with it. It holds the user's genuine "Hey
                // Ari", and if this turn is then abandoned the silence watcher
                // would file that as a hard negative, training the next model
                // not to fire on its own wake word.
                self.verify_wake_pending = false;
                self.wake_preroll = None;
                // Zero rewind: the repeat prompt and cue must not be taken in
                // as the user's answer.
                self.start_listening(io, ListenParams { rewind_seconds: 0.0, ..ListenParams::default() });
                self.after_arm(io);
            }
            AfterSpeech::ExitTalk => {
                self.talk_mode = false;
                self.barge_in_active = false;
                self.set_capture_mode(io, CaptureMode::Normal);
                io.engine(EngineCall::SetConversationActive(false), self.tag());
                self.dismiss(io, false);
            }
            AfterSpeech::Rearm => {
                self.rearm_for_reply(io);
                self.finish_turn(io);
            }
            AfterSpeech::Dwell { spoken_from, cards } => {
                // A floor, not a ceiling: "Yes" is spoken and done inside a
                // second, and the overlay would blink out before anyone read it.
                let wait = MIN_RESPONSE_DWELL_MS - (io.now() - spoken_from);
                if let Some(turn) = self.turn.as_mut() {
                    turn.stage = Stage::Dwell;
                }
                if wait > 0 {
                    self.schedule(io, wait, TimerKind::Dwell { cards });
                } else {
                    self.dismiss(io, cards);
                }
            }
        }
    }

    /// The turn handed back with the session still running: a reply is
    /// awaited. Its capture is written and the queued events read.
    fn finish_turn(&mut self, io: &dyn SessionIo) {
        if let Some(turn) = self.turn.take() {
            self.capture_turn(io, turn);
        }
        self.flow = Flow::Collecting;
        // If the skill asked a follow-up, the turn re-armed the mic instead
        // of dismissing: watch the reply turn's silence.
        if self.active && self.awaiting_reply {
            self.launch_watcher(io, SILENCE_TIMEOUT_MS);
        }
        self.drain_pending(io);
    }

    /// Re-arm the mic for a skill's follow-up question without a second
    /// wake word.
    fn rearm_for_reply(&mut self, io: &dyn SessionIo) {
        // Re-arming supersedes the turn just finished: bump the generation and
        // snapshot it so any late final from the previous turn reads as stale.
        self.turn_generation += 1;
        self.armed_generation = self.turn_generation;
        self.awaiting_reply = true;
        // A full fresh silence window for the reply.
        self.last_activity_at = io.now();
        self.set_state(io, SessionState::Listening { partial: String::new() });
        io.host().play_cue(RE_ARM_CUE_VOLUME);
        // Zero rewind: a follow-up answer comes after the prompt, so there's
        // no pre-roll worth keeping, and a rewind would replay the tail of
        // Ari's question and cue into the recogniser.
        self.start_listening(io, ListenParams { rewind_seconds: 0.0, ..ListenParams::default() });
    }

    /// The chime the user was waiting for, and a fresh listen for the
    /// command itself. Zero rewind, so the "Hey Ari" already heard is not
    /// heard again, and an opening turn rather than a reply, since nothing was
    /// asked.
    fn listen_for_command(&mut self, io: &dyn SessionIo) {
        self.turn_generation += 1;
        self.armed_generation = self.turn_generation;
        self.last_activity_at = io.now();
        self.set_state(io, SessionState::Listening { partial: String::new() });
        self.ready_cue(io);
        self.start_listening(io, ListenParams { rewind_seconds: 0.0, ..ListenParams::default() });
    }

    // ── Cold start and the early check ──────────────────────────────────

    fn on_model_loaded(&mut self, io: &dyn SessionIo, ok: bool) {
        if self.flow != Flow::ColdLoading {
            return;
        }
        if !ok {
            self.set_state(io, SessionState::Error(SessionError::NoSpeechModel));
            self.show_error_then_dismiss(io);
            return;
        }
        self.flow = Flow::ColdRepeat;
        let repeat = io.host().phrase(Phrase::PleaseRepeat);
        let speech = (!repeat.trim().is_empty()).then(|| {
            let id = self.speech_id();
            io.host().speak(id, &repeat, true);
            id
        });
        self.await_speech(io, speech, AfterSpeech::ColdRepeat);
    }

    fn on_name_checked(&mut self, io: &dyn SessionIo, result: Option<EarlyNameCheck>) {
        let Some(result) = result else { return };
        let Some((generation, asked_at)) = self.early_check.take() else { return };
        let in_time = self.hidden && self.wake_outcome_pending && self.turn_generation == generation;
        io.early_name(result, io.now() - asked_at, in_time);
        if in_time && result == EarlyNameCheck::Pass {
            self.name_heard_early = true;
            self.reveal(io);
        }
    }

    /// Work the session waited on failed.
    fn failed(&mut self, io: &dyn SessionIo, message: &str) {
        if self.dictation {
            io.log(LogLevel::Warn, &format!("Dictation failed: {message}"));
            self.dismiss(io, false);
            return;
        }
        io.log(LogLevel::Warn, &format!("Voice session failed: {message}"));
        if let Some(turn) = self.turn.take() {
            self.capture_turn(io, turn);
        }
        self.cancel_watcher();
        self.reveal(io);
        self.log_wake_outcome(io, WakeOutcome::Error);
        self.set_state(io, SessionState::Error(SessionError::Failed(message.to_string())));
        self.show_error_then_dismiss(io);
    }

    // ── Wakes ───────────────────────────────────────────────────────────

    /// Write how this wake ended to the wake log, unless something already
    /// has. Returns whether this call was the one that ended it.
    fn log_wake_outcome(&mut self, io: &dyn SessionIo, outcome: WakeOutcome) -> bool {
        if !self.wake_outcome_pending {
            return false;
        }
        self.wake_outcome_pending = false;
        io.wake_outcome(outcome, self.hidden);
        true
    }

    /// Show a quiet wake.
    fn reveal(&mut self, io: &dyn SessionIo) {
        if !self.hidden {
            return;
        }
        self.hidden = false;
        io.host().reveal();
    }

    /// End the wake and, if it was a false one the user saw, count it
    /// towards the warning that offers to make Ari less sensitive. Every
    /// false wake the user saw is counted, whichever path noticed it: fed
    /// from the silence timeout alone, the warning never fired on cloud
    /// transcription, which cannot reach it.
    fn end_wake(&mut self, io: &dyn SessionIo, outcome: WakeOutcome) {
        let seen = !self.hidden;
        if self.log_wake_outcome(io, outcome) && outcome.false_wake() && seen {
            io.host().false_wake();
        }
    }

    /// A wake with no speech in it: keep the audio if the user asked to, and count it.
    fn on_no_speech_wake(&mut self, io: &dyn SessionIo, pcm: Option<Vec<i16>>) {
        self.capture_false_trigger(io, pcm, String::new(), WakeCaptureHook::NoSpeech);
        self.end_wake(io, WakeOutcome::NoSpeech);
    }

    // ── Captures ────────────────────────────────────────────────────────

    /// Audio that falsely triggered the wake word, if the user opted in.
    fn capture_false_trigger(&self, io: &dyn SessionIo, pcm: Option<Vec<i16>>, raw: String, hook: WakeCaptureHook) {
        // Keep-everything implies this capture: either setting keeps the
        // containment clips flowing.
        if self.config.keep_false_trigger_audio || self.config.keep_everything_audio {
            self.capture_wake(io, pcm, raw, hook);
        }
    }

    /// The wake moment of an accepted turn, a confirmed true positive. `pcm`
    /// is the wake pre-roll; none on tap-to-talk and reply turns, which have
    /// no wake moment.
    fn capture_accepted_wake(&self, io: &dyn SessionIo, pcm: Option<Vec<i16>>, raw: String) {
        if self.config.keep_wake_audio || self.config.keep_everything_audio {
            self.capture_wake(io, pcm, raw, WakeCaptureHook::Accepted);
        }
    }

    fn capture_wake(&self, io: &dyn SessionIo, pcm: Option<Vec<i16>>, raw: String, hook: WakeCaptureHook) {
        let (Some(pcm), Some(dir)) = (pcm.filter(|p| !p.is_empty()), self.config.captures_dir.clone()) else { return };
        io.capture(dir, Capture::Wake { pcm, raw, hook });
    }

    /// What the user said, and every transcript it produced, if the user opted in.
    fn capture_utterance(&self, io: &dyn SessionIo, pcm: Option<Vec<i16>>, capture: UtteranceCapture) {
        if !(self.config.keep_utterance_audio || self.config.keep_everything_audio) {
            return;
        }
        let (Some(pcm), Some(dir)) = (pcm.filter(|p| !p.is_empty()), self.config.captures_dir.clone()) else { return };
        io.capture(dir, Capture::Utterance { pcm, capture });
    }

    fn capture_turn(&self, io: &dyn SessionIo, turn: Turn) {
        let capture = UtteranceCapture {
            turn: turn.kind,
            outcome: turn.outcome,
            response: turn.response,
            raw: turn.raw,
            transcript: turn.text,
            parallel: turn.parallel,
            offline: turn.offline,
            used: turn.used,
            locale: io.locale(),
            model: io.stt(self.config.host_transcribes).model_id,
        };
        self.capture_utterance(io, turn.audio, capture);
    }

    // ── Typed turns ─────────────────────────────────────────────────────

    /// A turn typed into the chat, or dictated into it (`spoken`: Ari says
    /// the answer). The chat has already shown the words. Runs beside the
    /// voice session, never in it.
    pub(crate) fn submit_text(&mut self, io: &dyn SessionIo, text: String, spoken: bool) {
        if text.trim().is_empty() {
            return;
        }
        self.next_typed += 1;
        let turn = self.next_typed;
        self.typed.insert(
            turn,
            Typed { text: text.clone(), spoken, streamed: false, sentence_spoken: false, reply: None, skill_id: None },
        );
        // A word that stops a sounding alert, or names a button on the card
        // on screen, answers that rather than going to the engine.
        io.run(Job::Intercept { text, cards: true }, Tag::Typed(turn));
    }

    /// The chat was reset: typed turns in flight say and log nothing more.
    pub(crate) fn cancel_typed(&mut self) {
        self.typed.clear();
        self.timers.retain(|t| !matches!(t.kind, TimerKind::StillWorking { .. }));
    }

    fn typed_result(&mut self, io: &dyn SessionIo, turn: u64, result: Finished) {
        let Some(typed) = self.typed.get_mut(&turn) else { return };
        match result {
            Finished::Intercepted(Intercepted::No) => {
                io.host().typed_thinking(turn);
                if typed.spoken {
                    self.next_timer += 1;
                    let at = io.now() + STILL_WORKING_DELAY_MS;
                    self.timers.push(Timer { at, seq: self.next_timer, epoch: self.epoch, kind: TimerKind::StillWorking { turn } });
                }
                let call = EngineCall::Respond { input: typed.text.clone(), alternative: None, stream: true };
                io.engine(call, Tag::Typed(turn));
            }
            Finished::Intercepted(Intercepted::Alert | Intercepted::CardSilent) => {
                self.typed.remove(&turn);
                io.host().typed_finished(turn, TypedFinish { clear: false, reply: None });
            }
            Finished::Intercepted(Intercepted::CardSpoken { text, attachments, .. }) => {
                let spoken = typed.spoken;
                self.typed.remove(&turn);
                io.host().typed_finished(turn, TypedFinish { clear: false, reply: card_reply(&text, attachments) });
                if spoken && !text.trim().is_empty() {
                    self.speak_after(io, &text);
                }
            }
            Finished::AnswerText(so_far, markdown) => {
                typed.streamed = true;
                io.host().answer_text(turn, &so_far, markdown.as_deref());
            }
            Finished::AnswerSentence(sentence) => {
                if typed.spoken {
                    typed.sentence_spoken = true;
                    self.end_still_working(turn);
                    self.speak_after(io, &sentence);
                }
            }
            Finished::Responded(reply, _) => {
                self.end_still_working(turn);
                // Before anything else, so a turn that both remembers and
                // fails further on still keeps the fact.
                if should_persist_facts(&reply) {
                    io.host().facts_changed();
                }
                let typed = self.typed.get_mut(&turn).expect("checked");
                typed.reply = Some(reply.clone());
                match reply {
                    EngineReply::Action { json, skill_id, .. } => {
                        // Only action replies carry an id; a plain text
                        // answer is unattributable, and a report on one says
                        // so rather than guessing.
                        typed.skill_id = Some(skill_id.clone()).filter(|s| !s.trim().is_empty());
                        io.run(Job::HandleAction { json, skill_id }, Tag::Typed(turn));
                    }
                    EngineReply::Text { body, .. } | EngineReply::NotUnderstood { body } => {
                        self.finish_typed(io, turn, body, false, None);
                    }
                    EngineReply::Binary { mime, len } => {
                        self.finish_typed(io, turn, format!("[Binary: {mime}, {len} bytes]"), false, None);
                    }
                }
            }
            Finished::ActionHandled(outcome) => {
                self.finish_typed(io, turn, outcome.text, outcome.attachments, outcome.display_text);
            }
            Finished::Failed(message) => {
                io.log(LogLevel::Warn, &format!("Typed turn failed: {message}"));
                self.end_still_working(turn);
                self.typed.remove(&turn);
                io.host().typed_finished(turn, TypedFinish { clear: false, reply: None });
            }
            _ => {}
        }
    }

    fn finish_typed(&mut self, io: &dyn SessionIo, turn: u64, text: String, attachments: bool, bubble: Option<String>) {
        let Some(typed) = self.typed.remove(&turn) else { return };
        let Some(reply) = typed.reply else { return };
        let markdown = match &reply {
            EngineReply::Text { markdown, .. } => markdown.clone(),
            _ => None,
        };
        // An answer shown while it was written always takes its final form.
        // Otherwise an empty one logs nothing: a silent envelope (the first
        // half of an assistant round trip) would flash a ghost bubble.
        let logged = (typed.streamed || !text.trim().is_empty() || attachments).then(|| LoggedReply {
            text: bubble.unwrap_or_else(|| text.clone()),
            markdown,
            skill_id: typed.skill_id,
        });
        io.host().typed_finished(turn, TypedFinish { clear: should_clear_conversation(&reply), reply: logged });
        // Read out unless its sentences already were.
        if typed.spoken && !typed.sentence_spoken && !text.trim().is_empty() && !matches!(reply, EngineReply::Binary { .. }) {
            self.speak_after(io, &text);
        }
    }

    fn end_still_working(&mut self, turn: u64) {
        self.timers.retain(|t| !matches!(t.kind, TimerKind::StillWorking { turn: t2 } if t2 == turn));
    }

    fn still_working(&mut self, io: &dyn SessionIo, turn: u64) {
        if self.typed.contains_key(&turn) {
            let phrase = io.host().phrase(Phrase::PleaseWait);
            self.speak_after(io, &phrase);
        }
    }

    /// Say `text` after whatever is playing. Nothing waits for it.
    fn speak_after(&mut self, io: &dyn SessionIo, text: &str) {
        let id = self.speech_id();
        io.host().speak(id, text, false);
    }

    // ── Dictation ───────────────────────────────────────────────────────

    /// In-place dictation: speech-to-text only. No engine, no speech, no
    /// re-arm, no barge-in. Partials go out as `Listening`, the transcript to
    /// [`SessionHost::dictated`].
    pub(crate) fn start_dictation(&mut self, io: &dyn SessionIo) -> bool {
        if self.active {
            io.log(LogLevel::Warn, "startDictation() called while a session is active — ignoring");
            return false;
        }
        self.active = true;
        self.dictation = true;
        self.flow = Flow::Collecting;
        self.set_state(io, SessionState::Listening { partial: String::new() });
        self.last_activity_at = io.now();
        // Bound the mic: the streaming recogniser ends a listen only on a
        // stable non-empty partial, so pure silence never endpoints. Without
        // this a dictation the user never spoke into would keep the mic hot.
        self.launch_watcher(io, SILENCE_TIMEOUT_MS);
        self.start_listening(io, ListenParams::default());
        true
    }

    /// Cancel a dictation. No transcript, so nothing is submitted; the
    /// frontend keeps the last partial.
    pub(crate) fn stop_dictation(&mut self, io: &dyn SessionIo) {
        if self.active {
            self.dismiss(io, false);
        }
    }

    fn dictation_event(&mut self, io: &dyn SessionIo, event: SttEvent) {
        match event {
            SttEvent::Listening { partial, .. } => {
                if !partial.trim().is_empty() {
                    self.last_activity_at = io.now();
                }
                self.set_state(io, SessionState::Listening { partial });
            }
            SttEvent::Transcribing => self.set_state(io, SessionState::Thinking),
            SttEvent::Done { text, parallel, audio, raw, .. } => {
                let capture = UtteranceCapture {
                    turn: UtteranceTurn::Dictation,
                    outcome: UtteranceOutcome::Dictated,
                    response: None,
                    raw,
                    transcript: text.clone(),
                    parallel,
                    offline: None,
                    used: text.clone(),
                    locale: io.locale(),
                    model: io.stt(self.config.host_transcribes).model_id,
                };
                self.capture_utterance(io, audio, capture);
                io.host().dictated(&text);
                self.dismiss(io, false);
            }
            SttEvent::Error(error) => {
                io.log(LogLevel::Warn, &format!("Dictation STT error: {error:?}"));
                self.dismiss(io, false);
            }
            SttEvent::NoSpeech { .. } => self.dismiss(io, false),
        }
    }

    // ── Ending ──────────────────────────────────────────────────────────

    /// End the session. `keep_cards` leaves an answer's cards on the overlay.
    pub(crate) fn dismiss(&mut self, io: &dyn SessionIo, keep_cards: bool) {
        io.log(LogLevel::Info, "Dismissing voice session");
        self.log_wake_outcome(io, WakeOutcome::Dismissed);
        self.hidden = false;
        self.name_heard_early = false;
        // Supersede any turn in flight, so a final that decodes after this
        // can't fire a phantom turn.
        self.turn_generation += 1;
        self.speaking = false;
        self.tts_job = None;
        self.talk_mode = false;
        self.barge_in_active = false;
        self.set_capture_mode(io, CaptureMode::Normal);
        // Drop any reply the engine was still waiting on. A no-op on the
        // normal path; covers a tap to dismiss, the silence timeout and the
        // frontend going away.
        io.engine(EngineCall::SetConversationActive(false), self.tag());
        io.engine(EngineCall::CancelPendingReply, self.tag());
        self.awaiting_reply = false;
        self.verify_wake_pending = false;
        self.wake_preroll = None;
        io.stop_listen();
        self.listen = None;
        io.host().stop_speaking();
        if let Some(turn) = self.turn.take() {
            self.capture_turn(io, turn);
        }
        self.epoch += 1;
        self.timers.retain(|t| matches!(t.kind, TimerKind::StillWorking { .. }));
        self.watcher = None;
        self.awaiting_speech = None;
        self.pending_stt.clear();
        self.early_check = None;
        self.flow = Flow::Collecting;
        self.active = false;
        self.dictation = false;
        self.set_state(io, SessionState::Idle { keep_cards });
    }
}

/// A card's answer to log: its words, unless it had neither words nor
/// attachments.
fn card_reply(text: &str, attachments: bool) -> Option<LoggedReply> {
    (!text.trim().is_empty() || attachments).then(|| LoggedReply { text: text.to_string(), markdown: None, skill_id: None })
}
