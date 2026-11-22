//! The voice session, moved from ari-android's `voice/VoiceSession.kt`: what
//! a wake, a tap on the mic or a dictation does from the first word to the
//! last, with the frontend only showing, playing and running what it is told.
//!
//! One session at a time. It runs on a thread of its own ([`runner`]) as a
//! plain state machine ([`core`]), so the scenario tests can drive it with a
//! fake host and a clock that only moves when they say.

mod core;
pub mod captures;
pub mod rules;
pub mod runner;

#[cfg(test)]
mod scenarios;

use std::path::PathBuf;
use std::sync::Arc;

use crate::stt::{ListenParams, SttError};

pub use captures::{UtteranceCapture, UtteranceOutcome, UtteranceTurn, WakeCaptureHook};
pub use runner::HostHeard;

/// What the overlay shows.
#[derive(Debug, Clone, PartialEq)]
pub enum SessionState {
    /// No session. `keep_cards`: the answer's cards stay on the overlay
    /// after it, which the frontend times out itself.
    Idle { keep_cards: bool },
    /// The speech model is still loading; `message` is being said.
    Preparing { message: String },
    Listening { partial: String },
    Thinking,
    /// `cards`: the answer has cards, which the frontend holds from the
    /// [`SessionHost::handle_action`] or [`SessionHost::intercept`] call that
    /// made them.
    Responding { text: String, cards: bool },
    Error(SessionError),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionError {
    /// A cold start found no speech model it could load.
    NoSpeechModel,
    /// Speech-to-text failed this way.
    Stt(SttError),
    /// The listen heard no speech.
    NoSpeech,
    /// Something else went wrong; the message is for the log more than the user.
    Failed(String),
}

/// Words the session needs said, in the frontend's own translations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phrase {
    /// While the speech model warms up.
    PleaseWait,
    /// Once it has: the words that woke Ari aged out of the ring meanwhile.
    PleaseRepeat,
}

/// How the shared mic is being captured. `Normal` is wake-word listening;
/// `Conversation` is echo-cancelled capture during a "let's talk" session
/// with barge-in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureMode {
    Normal,
    Conversation,
}

/// The frontend's settings the session acts on. Pushed again whenever one
/// changes; the session reads them at the moment it needs them, as the
/// Kotlin read the settings store.
#[derive(Debug, Clone, Default)]
pub struct SessionConfig {
    pub barge_in_enabled: bool,
    /// The device has a hardware echo canceller. Barge-in needs one.
    pub echo_canceller: bool,
    pub keep_false_trigger_audio: bool,
    pub keep_wake_audio: bool,
    pub keep_utterance_audio: bool,
    /// Implies all three above.
    pub keep_everything_audio: bool,
    /// Where the captures go, the directory each store sits under. No
    /// captures are written without one.
    pub captures_dir: Option<PathBuf>,
    /// The frontend transcribes listens itself (Android's system speech
    /// service): the session asks it to through [`SessionHost::host_listen`].
    pub host_transcribes: bool,
}

/// What the frontend did with words that went to an alert or a card instead
/// of the engine. It has already acted and written the conversation log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Intercepted {
    /// Nothing claimed them: on to the engine.
    No,
    /// They stopped a sounding alert.
    Alert,
    /// A card's button took them and had nothing to say.
    CardSilent,
    /// A card's button took them and answered.
    CardSpoken { text: String, cards: bool },
}

/// An action envelope, run by the frontend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionOutcome {
    /// What Ari says, and shows on the overlay.
    pub text: String,
    /// What the conversation log shows instead, when the skill words it differently.
    pub display_text: Option<String>,
    pub cards: bool,
    /// Any attachment at all, cards included.
    pub attachments: bool,
}

/// One spoken turn for the conversation log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoggedTurn {
    /// The user cleared the conversation: empty the log first, and log
    /// `heard` no longer.
    pub clear: bool,
    /// The transcript Ari acted on.
    pub heard: String,
    /// Ari's answer, when there is one to log. The frontend adds the
    /// attachments it holds from this turn's action.
    pub reply: Option<LoggedReply>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoggedReply {
    pub text: String,
    pub markdown: Option<String>,
    pub skill_id: Option<String>,
}

/// Flags the engine sets on a reply.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReplyFlags {
    /// A skill asked a question and waits for a spoken answer.
    pub rearm: bool,
    pub enter_conversation: bool,
    pub exit_conversation: bool,
    pub facts_changed: bool,
    pub clear_conversation: bool,
}

/// The engine's reply to one input.
#[derive(Debug, Clone, PartialEq)]
pub enum EngineReply {
    Text { body: String, markdown: Option<String>, flags: ReplyFlags },
    Action { json: String, skill_id: String, flags: ReplyFlags },
    Binary { mime: String, len: usize },
    NotUnderstood { body: String },
}

/// An on-device answer while the model writes it.
pub trait AnswerSink: Send + Sync {
    /// The words so far, each call replacing the last.
    fn text(&self, so_far: &str, markdown: Option<&str>);
    /// A finished sentence, to start speaking.
    fn sentence(&self, text: &str);
}

/// The engine a session's turns go to. `ari-ffi` implements it over its
/// `AriEngine`.
pub trait TurnEngine: Send + Sync {
    /// One turn. `alternative` is a second transcript the engine may act on
    /// instead (see `Engine::set_alternative_transcript`); the bool says
    /// whether it did. Blocking.
    fn respond(
        &self,
        input: &str,
        alternative: Option<&str>,
        answer: Option<Arc<dyn AnswerSink>>,
    ) -> (EngineReply, bool);

    /// Drop a question the engine waits on an answer to. No-op when none.
    fn cancel_pending_reply(&self);

    /// "Let's talk" mode began or ended.
    fn set_conversation_active(&self, active: bool);
}

/// What the session needs from the frontend it runs in. Called on the
/// session's thread, or on a worker for the ones that block; none may call
/// back into the pipeline before returning.
pub trait SessionHost: Send + Sync {
    fn state(&self, state: SessionState);

    /// Show a quiet wake: it has passed a name check.
    fn reveal(&self);

    /// The "your turn" cue, at `volume` (0 to 1).
    fn play_cue(&self, volume: f32);

    /// Say `text`, cutting off anything playing when `flush`, else after it.
    /// Report [`speech_finished`](crate::pipeline::VoicePipeline::speech_finished)
    /// with `id` when it has finished, been stopped, failed, or could not
    /// start.
    fn speak(&self, id: u64, text: &str, flush: bool);

    fn stop_speaking(&self);

    fn phrase(&self, phrase: Phrase) -> String;

    fn capture_mode(&self, mode: CaptureMode);

    /// Load the user's speech model, for a wake that found none warm.
    /// Blocking; false when there is none to load or it would not.
    fn load_speech_model(&self) -> bool;

    /// Transcribe a listen ([`SessionConfig::host_transcribes`]), reporting
    /// through [`host_heard`](crate::pipeline::VoicePipeline::host_heard).
    fn host_listen(&self, params: ListenParams) {
        let _ = params;
    }

    fn host_stop_listen(&self) {}

    /// Give the words to a sounding alert or the card on screen, if one
    /// claims them (`cards`: the card may). Blocking.
    fn intercept(&self, text: &str, cards: bool) -> Intercepted;

    /// Run an action envelope. Blocking.
    fn handle_action(&self, json: &str, skill_id: &str) -> ActionOutcome;

    fn turn_logged(&self, turn: LoggedTurn);

    /// A remember or forget command changed the engine's facts.
    fn facts_changed(&self);

    /// The user saw a wake nobody had called Ari for.
    fn false_wake(&self);

    /// A dictation's transcript.
    fn dictated(&self, text: &str);

    /// The engine finished a call, whichever turn it was for.
    fn engine_turn_finished(&self) {}
}
