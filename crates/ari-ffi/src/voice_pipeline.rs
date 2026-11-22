//! UniFFI wrapper around `ari_voice`'s audio side: the voice pipeline, its
//! wake log, and a wake detector for the sensitivity tuner and parity tests.
//!
//! The microphone's audio does not come through UniFFI. It goes through
//! [`ari_voice_push_audio`], a plain C function, and on Android through the
//! JNI entry point below it: the capture thread calls it every 30 ms all day,
//! and UniFFI's route (JNA, a copy into a buffer, boxing) costs more than the
//! JNI call it replaces.

use std::path::PathBuf;
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ari_voice::cloud_stt::{CloudSttFailure, CloudSttService};
use ari_voice::detector::{OWW_EMBEDDING_FILE, OWW_MELSPECTROGRAM_FILE, WakeDetector};
use ari_voice::pipeline::{self, LogLevel, VoiceHost, VoicePipeline, WakeConfig};
use ari_voice::session::{
    ActionOutcome, AnswerSink, CaptureMode, EngineReply, Intercepted, LoggedReply, LoggedTurn, Phrase, ReplyFlags,
    HostHeard, SessionConfig, SessionError, SessionHost, SessionState, TurnEngine,
};
use ari_voice::stt::{ListenParams, SttError, SttModelSpec};
use ari_voice::wake_log::{WakeLog, WakeOutcome};
use ari_voice::wake_model::WakeEngine;

use crate::voice::{FfiCloudSttService, FfiOperatingPoint, FfiWakeSensitivity, FfiWakeWordModel};
use crate::{AriEngine, FfiLogLevel, FfiResponse};

/// How a wake the user could see ended (`ari_voice::wake_log::WakeOutcome`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum FfiWakeOutcome {
    Accepted,
    Rejected,
    NoSpeech,
    Silent,
    Error,
    Dismissed,
    ColdStart,
    NotShown,
}

impl From<FfiWakeOutcome> for WakeOutcome {
    fn from(o: FfiWakeOutcome) -> Self {
        match o {
            FfiWakeOutcome::Accepted => Self::Accepted,
            FfiWakeOutcome::Rejected => Self::Rejected,
            FfiWakeOutcome::NoSpeech => Self::NoSpeech,
            FfiWakeOutcome::Silent => Self::Silent,
            FfiWakeOutcome::Error => Self::Error,
            FfiWakeOutcome::Dismissed => Self::Dismissed,
            FfiWakeOutcome::ColdStart => Self::ColdStart,
            FfiWakeOutcome::NotShown => Self::NotShown,
        }
    }
}

#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum FfiVoiceError {
    #[error("{reason}")]
    Failed { reason: String },
}

impl From<String> for FfiVoiceError {
    fn from(reason: String) -> Self {
        Self::Failed { reason }
    }
}

/// The wake log file (`ari_voice::wake_log::WakeLog`). One per process,
/// shared by the pipeline and the frontend.
#[derive(uniffi::Object)]
pub struct FfiWakeLog {
    inner: Arc<WakeLog>,
}

#[uniffi::export]
impl FfiWakeLog {
    #[uniffi::constructor]
    pub fn new(path: String) -> Arc<Self> {
        Arc::new(Self { inner: Arc::new(WakeLog::new(path)) })
    }

    /// How a wake ended that never reached a session: the overlay could not
    /// be opened. The session writes every other ending itself.
    pub fn outcome(&self, outcome: FfiWakeOutcome, hidden: bool) {
        self.inner.outcome(outcome.into(), hidden);
    }

    /// The last day and week summarised, then the week's lines, for a bug report.
    pub fn report(&self, now_ms: i64, listening_now: bool) -> String {
        self.inner.report(now_ms, listening_now)
    }
}

/// The cloud transcription service the user set up, with its key.
#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiCloudSttSetup {
    pub service: FfiCloudSttService,
    pub api_key: Option<String>,
}

/// Why a listen could not produce a transcript.
#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum FfiSttFailure {
    /// No model is loaded and the cloud is not selected.
    NoModel,
    /// Something else holds the listen feed.
    Busy,
    CloudNotConfigured,
    CloudNetwork,
    CloudAuth,
    CloudServer,
    /// The host's own transcriber refused the request.
    ServiceRefused,
    /// The host's own transcriber failed.
    ServiceFailed,
}

impl From<SttError> for FfiSttFailure {
    fn from(error: SttError) -> Self {
        match error {
            SttError::NoModel => Self::NoModel,
            SttError::Busy => Self::Busy,
            SttError::Cloud(CloudSttFailure::NotConfigured) => Self::CloudNotConfigured,
            SttError::Cloud(CloudSttFailure::Network) => Self::CloudNetwork,
            SttError::Cloud(CloudSttFailure::Auth) => Self::CloudAuth,
            // An empty transcript arrives as NoSpeech; this is for completeness.
            SttError::Cloud(CloudSttFailure::Server | CloudSttFailure::Empty) => Self::CloudServer,
            SttError::HostRefused => Self::ServiceRefused,
            SttError::HostFailed => Self::ServiceFailed,
        }
    }
}

impl From<FfiSttFailure> for SttError {
    fn from(failure: FfiSttFailure) -> Self {
        match failure {
            FfiSttFailure::NoModel => Self::NoModel,
            FfiSttFailure::Busy => Self::Busy,
            FfiSttFailure::CloudNotConfigured => Self::Cloud(CloudSttFailure::NotConfigured),
            FfiSttFailure::CloudNetwork => Self::Cloud(CloudSttFailure::Network),
            FfiSttFailure::CloudAuth => Self::Cloud(CloudSttFailure::Auth),
            FfiSttFailure::CloudServer => Self::Cloud(CloudSttFailure::Server),
            FfiSttFailure::ServiceRefused => Self::HostRefused,
            FfiSttFailure::ServiceFailed => Self::HostFailed,
        }
    }
}

/// What the app's own transcriber (the system speech service) heard, for
/// the listen the session asked it for. Audio is 16-bit little-endian PCM.
#[derive(Debug, Clone, uniffi::Enum)]
pub enum FfiHostHeard {
    /// The words so far.
    Partial { text: String },
    /// The transcript, and all the audio the transcriber was handed.
    Final { text: String, audio: Vec<u8> },
    NoSpeech { audio: Vec<u8> },
    Failed { failure: FfiSttFailure },
}

impl From<FfiHostHeard> for HostHeard {
    fn from(heard: FfiHostHeard) -> Self {
        match heard {
            FfiHostHeard::Partial { text } => Self::Partial(text),
            FfiHostHeard::Final { text, audio } => Self::Final { text, audio: samples(&audio) },
            FfiHostHeard::NoSpeech { audio } => Self::NoSpeech { audio: samples(&audio) },
            FfiHostHeard::Failed { failure } => Self::Failed(failure.into()),
        }
    }
}

fn pcm_bytes(samples: &[i16]) -> Vec<u8> {
    samples.iter().flat_map(|s| s.to_le_bytes()).collect()
}

/// What the pipeline needs from the app: model files, wakes, settings and
/// log lines. Called on engine threads and the capture thread;
/// hand work on rather than doing it in the call.
#[uniffi::export(with_foreign)]
pub trait FfiVoiceHost: Send + Sync {
    /// A model file by name, or `None` when it cannot be read.
    fn load_model(&self, name: String) -> Option<Vec<u8>>;
    /// The wake word was heard and passed the debounce and the speech check.
    fn wake_detected(&self);
    fn log(&self, level: FfiLogLevel, message: String);
    /// The user's language code (`en`, `it`).
    fn locale(&self) -> String;
    /// The cloud transcription service and key, asked for at each request.
    fn cloud_stt(&self) -> Option<FfiCloudSttSetup>;
}

struct ForeignVoiceHost(Arc<dyn FfiVoiceHost>);

impl VoiceHost for ForeignVoiceHost {
    fn load_model(&self, name: &str) -> Option<Vec<u8>> {
        self.0.load_model(name.to_string())
    }
    fn wake_detected(&self) {
        self.0.wake_detected();
    }
    fn log(&self, level: LogLevel, message: &str) {
        let level = match level {
            LogLevel::Debug => FfiLogLevel::Debug,
            LogLevel::Info => FfiLogLevel::Info,
            LogLevel::Warn => FfiLogLevel::Warn,
        };
        self.0.log(level, message.to_string());
    }
    fn locale(&self) -> String {
        self.0.locale()
    }
    fn cloud_stt(&self) -> Option<(CloudSttService, Option<String>)> {
        self.0.cloud_stt().map(|setup| (setup.service.into(), setup.api_key))
    }
}

/// A model in a directory, as the app's `SttModel` describes it.
#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiSttModelSpec {
    pub id: String,
    pub model_type: String,
    pub dir: String,
    pub encoder_file: String,
    pub decoder_file: String,
    pub joiner_file: Option<String>,
    pub tokens_file: String,
}

impl From<FfiSttModelSpec> for SttModelSpec {
    fn from(s: FfiSttModelSpec) -> Self {
        Self {
            id: s.id,
            model_type: s.model_type,
            dir: PathBuf::from(s.dir),
            encoder_file: s.encoder_file,
            decoder_file: s.decoder_file,
            joiner_file: s.joiner_file,
            tokens_file: s.tokens_file,
        }
    }
}

/// One read from a [`FfiListenFeed`].
#[derive(Debug, Clone, uniffi::Enum)]
pub enum FfiFeedRead {
    /// 16-bit little-endian PCM.
    Chunk { pcm: Vec<u8> },
    /// Nothing arrived in time; the feed is still open.
    Timeout,
    /// The feed was disarmed.
    Closed,
}

/// The listen feed for readers that live in the app: the pre-roll as 100 ms
/// slices, then every chunk, until it is closed. No wake fires meanwhile.
#[derive(uniffi::Object)]
pub struct FfiListenFeed {
    pipeline: Arc<VoicePipeline>,
    receiver: Mutex<Receiver<Vec<i16>>>,
}

#[uniffi::export]
impl FfiListenFeed {
    /// The next chunk, waiting up to `timeout_ms` for it.
    pub fn next(&self, timeout_ms: u64) -> FfiFeedRead {
        let receiver = self.receiver.lock().unwrap_or_else(|e| e.into_inner());
        match receiver.recv_timeout(Duration::from_millis(timeout_ms)) {
            Ok(chunk) => FfiFeedRead::Chunk { pcm: pcm_bytes(&chunk) },
            Err(RecvTimeoutError::Timeout) => FfiFeedRead::Timeout,
            Err(RecvTimeoutError::Disconnected) => FfiFeedRead::Closed,
        }
    }

    /// The end of this feed, and of the wake word's pause for it.
    pub fn disarm(&self) {
        self.pipeline.disarm();
    }
}

/// How to listen for the wake word, from the app's settings.
#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiWakeConfig {
    /// With the device's tuned ladder already applied (`with_ladder`).
    pub model: FfiWakeWordModel,
    pub sensitivity: FfiWakeSensitivity,
    pub speech_check: bool,
    pub quiet_wake: bool,
}

/// The process's voice pipeline. Building one makes it the one
/// [`ari_voice_push_audio`] feeds.
#[derive(uniffi::Object)]
pub struct FfiVoicePipeline {
    inner: Arc<VoicePipeline>,
}

#[uniffi::export]
impl FfiVoicePipeline {
    #[uniffi::constructor]
    pub fn new(host: Arc<dyn FfiVoiceHost>, wake_log: Arc<FfiWakeLog>) -> Arc<Self> {
        let inner = VoicePipeline::new(Arc::new(ForeignVoiceHost(host)), Arc::clone(&wake_log.inner));
        Arc::new(Self { inner })
    }

    /// Build the wake engine, before the microphone opens.
    pub fn configure_wake(&self, config: FfiWakeConfig) -> Result<(), FfiVoiceError> {
        Ok(self.inner.configure_wake(WakeConfig {
            model: config.model.into(),
            sensitivity: config.sensitivity.into(),
            speech_check: config.speech_check,
            quiet_wake: config.quiet_wake,
        })?)
    }

    /// The microphone is open and its audio is arriving.
    pub fn start_wake(&self) {
        self.inner.start_wake();
    }

    /// The microphone is closing.
    pub fn stop_wake(&self) {
        self.inner.stop_wake();
    }

    /// Mid-conversation: no wake detection, audio still kept.
    pub fn set_wake_paused(&self, paused: bool) {
        self.inner.set_wake_paused(paused);
    }

    /// Load a speech-to-text model and warm it up, releasing any other.
    /// Blocking: call it off the UI thread.
    pub fn load_stt(&self, spec: FfiSttModelSpec) -> Result<(), FfiVoiceError> {
        Ok(self.inner.load_stt(&spec.into())?)
    }

    pub fn unload_stt(&self) {
        self.inner.unload_stt();
    }

    /// Transcribe through the cloud (`cloud_stt` on the host) instead of a model.
    pub fn set_cloud_mode(&self, enabled: bool) {
        self.inner.set_cloud_mode(enabled);
    }

    pub fn stt_model_loaded(&self) -> bool {
        self.inner.stt_model_loaded()
    }

    pub fn stt_model_id(&self) -> Option<String> {
        self.inner.stt_model_id()
    }

    /// Start the voice session, with `host` to show, play and run what it
    /// decides. Once, before any turn.
    pub fn attach_session_host(&self, host: Arc<dyn FfiSessionHost>) {
        self.inner.attach_session_host(Arc::new(ForeignSessionHost(host)));
    }

    /// The engine voice turns go to, once it is built. A turn that reaches it
    /// sooner waits.
    pub fn attach_engine(&self, engine: Arc<AriEngine>) {
        self.inner.attach_engine(engine);
    }

    /// The settings the session acts on. Again whenever one changes.
    pub fn configure_session(&self, config: FfiSessionConfig) {
        self.inner.configure_session(config.into());
    }

    /// Begin a voice turn: `verify_wake` for a wake word, `quiet` for one the
    /// wake service started, unseen until a name check passes. False when a
    /// session is already running. The first state has reached the host by
    /// the time this returns; never call it from inside a host callback.
    pub fn start_turn(&self, verify_wake: bool, quiet: bool) -> bool {
        self.inner.start_turn(verify_wake, quiet)
    }

    /// Begin a dictation; false when a session is already running.
    pub fn start_dictation(&self) -> bool {
        self.inner.start_dictation()
    }

    pub fn stop_dictation(&self) {
        self.inner.stop_dictation();
    }

    /// End the session: the overlay was closed.
    pub fn dismiss(&self) {
        self.inner.dismiss();
    }

    /// The speech the session asked for under `id` has finished, been
    /// stopped, or could not start.
    pub fn speech_finished(&self, id: u64) {
        self.inner.speech_finished(id);
    }

    /// What the app's own transcriber heard, for the listen it was asked to do.
    pub fn host_heard(&self, heard: FfiHostHeard) {
        self.inner.host_heard(heard.into());
    }

    pub fn is_session_active(&self) -> bool {
        self.inner.is_session_active()
    }

    /// The listen feed for a reader in the app, or `None` while another holds it.
    pub fn open_feed(&self, rewind_seconds: f32) -> Option<Arc<FfiListenFeed>> {
        let receiver = self.inner.arm(rewind_seconds)?;
        Some(Arc::new(FfiListenFeed { pipeline: Arc::clone(&self.inner), receiver: Mutex::new(receiver) }))
    }

    pub fn is_feed_armed(&self) -> bool {
        self.inner.is_armed()
    }
}

/// What the overlay shows (`ari_voice::session::SessionState`).
#[derive(Debug, Clone, PartialEq, uniffi::Enum)]
pub enum FfiSessionState {
    /// No session. `keep_cards`: leave the answer's cards on the overlay.
    Idle { keep_cards: bool },
    /// The speech model is loading; `message` is being said.
    Preparing { message: String },
    Listening { partial: String },
    Thinking,
    /// `cards`: show the cards the last `handle_action` or `intercept` made.
    Responding { text: String, cards: bool },
    Error { error: FfiSessionError },
}

#[derive(Debug, Clone, PartialEq, uniffi::Enum)]
pub enum FfiSessionError {
    NoSpeechModel,
    Stt { failure: FfiSttFailure },
    NoSpeech,
    Failed { message: String },
}

impl From<SessionState> for FfiSessionState {
    fn from(state: SessionState) -> Self {
        match state {
            SessionState::Idle { keep_cards } => Self::Idle { keep_cards },
            SessionState::Preparing { message } => Self::Preparing { message },
            SessionState::Listening { partial } => Self::Listening { partial },
            SessionState::Thinking => Self::Thinking,
            SessionState::Responding { text, cards } => Self::Responding { text, cards },
            SessionState::Error(error) => Self::Error {
                error: match error {
                    SessionError::NoSpeechModel => FfiSessionError::NoSpeechModel,
                    SessionError::Stt(e) => FfiSessionError::Stt { failure: e.into() },
                    SessionError::NoSpeech => FfiSessionError::NoSpeech,
                    SessionError::Failed(message) => FfiSessionError::Failed { message },
                },
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum FfiPhrase {
    PleaseWait,
    PleaseRepeat,
}

/// How the mic is captured: wake-word listening, or echo-cancelled for a
/// "let's talk" session with barge-in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum FfiCaptureMode {
    Normal,
    Conversation,
}

/// The settings the session acts on.
#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiSessionConfig {
    pub barge_in_enabled: bool,
    pub echo_canceller: bool,
    pub keep_false_trigger_audio: bool,
    pub keep_wake_audio: bool,
    pub keep_utterance_audio: bool,
    pub keep_everything_audio: bool,
    /// The directory the capture directories sit under.
    pub captures_dir: Option<String>,
    /// The app transcribes listens itself (the system speech service).
    pub host_transcribes: bool,
}

impl From<FfiSessionConfig> for SessionConfig {
    fn from(c: FfiSessionConfig) -> Self {
        Self {
            barge_in_enabled: c.barge_in_enabled,
            echo_canceller: c.echo_canceller,
            keep_false_trigger_audio: c.keep_false_trigger_audio,
            keep_wake_audio: c.keep_wake_audio,
            keep_utterance_audio: c.keep_utterance_audio,
            keep_everything_audio: c.keep_everything_audio,
            captures_dir: c.captures_dir.map(PathBuf::from),
            host_transcribes: c.host_transcribes,
        }
    }
}

/// What the app did with words an alert or a card claimed. It has already
/// acted and written the conversation log.
#[derive(Debug, Clone, PartialEq, Eq, uniffi::Enum)]
pub enum FfiIntercepted {
    No,
    Alert,
    CardSilent,
    CardSpoken { text: String, cards: bool },
}

/// An action envelope the app ran.
#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiActionOutcome {
    pub text: String,
    pub display_text: Option<String>,
    pub cards: bool,
    pub attachments: bool,
}

/// A spoken turn for the conversation log.
#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiLoggedTurn {
    /// Clear the log first, and log `heard` no longer.
    pub clear: bool,
    pub heard: String,
    /// With the attachments the app holds from this turn's action.
    pub reply: Option<FfiLoggedReply>,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiLoggedReply {
    pub text: String,
    pub markdown: Option<String>,
    pub skill_id: Option<String>,
}

impl From<LoggedTurn> for FfiLoggedTurn {
    fn from(t: LoggedTurn) -> Self {
        Self {
            clear: t.clear,
            heard: t.heard,
            reply: t.reply.map(|r: LoggedReply| FfiLoggedReply { text: r.text, markdown: r.markdown, skill_id: r.skill_id }),
        }
    }
}

/// A listen for the app's own transcriber.
#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiListenParams {
    pub rewind_seconds: f32,
    /// A quiet wake's limit: stop the audio here, so the transcriber has to end.
    pub limit_ms: Option<i64>,
}

/// What the voice session needs from the app. Called on the session's
/// threads; the ones that block run on workers. None may call back into the
/// pipeline's start commands before returning.
#[uniffi::export(with_foreign)]
pub trait FfiSessionHost: Send + Sync {
    fn state(&self, state: FfiSessionState);
    /// Show a quiet wake: it passed a name check.
    fn reveal(&self);
    fn play_cue(&self, volume: f32);
    /// Say `text`, cutting off whatever plays when `flush`. Report
    /// `speech_finished(id)` once it has finished, been stopped, or could not start.
    fn speak(&self, id: u64, text: String, flush: bool);
    fn stop_speaking(&self);
    fn phrase(&self, phrase: FfiPhrase) -> String;
    fn capture_mode(&self, mode: FfiCaptureMode);
    /// Blocking: load the user's speech model for a cold start.
    fn load_speech_model(&self) -> bool;
    /// Transcribe a listen, reporting through `host_heard`.
    fn host_listen(&self, params: FfiListenParams);
    fn host_stop_listen(&self);
    /// Blocking: give the words to a sounding alert or, when `cards`, the card on screen.
    fn intercept(&self, text: String, cards: bool) -> FfiIntercepted;
    /// Blocking: run an action envelope.
    fn handle_action(&self, json: String, skill_id: String) -> FfiActionOutcome;
    fn turn_logged(&self, turn: FfiLoggedTurn);
    fn facts_changed(&self);
    fn false_wake(&self);
    fn dictated(&self, text: String);
    /// The engine finished a call.
    fn engine_turn_finished(&self);
}

struct ForeignSessionHost(Arc<dyn FfiSessionHost>);

impl SessionHost for ForeignSessionHost {
    fn state(&self, state: SessionState) {
        self.0.state(state.into());
    }
    fn reveal(&self) {
        self.0.reveal();
    }
    fn play_cue(&self, volume: f32) {
        self.0.play_cue(volume);
    }
    fn speak(&self, id: u64, text: &str, flush: bool) {
        self.0.speak(id, text.to_string(), flush);
    }
    fn stop_speaking(&self) {
        self.0.stop_speaking();
    }
    fn phrase(&self, phrase: Phrase) -> String {
        self.0.phrase(match phrase {
            Phrase::PleaseWait => FfiPhrase::PleaseWait,
            Phrase::PleaseRepeat => FfiPhrase::PleaseRepeat,
        })
    }
    fn capture_mode(&self, mode: CaptureMode) {
        self.0.capture_mode(match mode {
            CaptureMode::Normal => FfiCaptureMode::Normal,
            CaptureMode::Conversation => FfiCaptureMode::Conversation,
        });
    }
    fn load_speech_model(&self) -> bool {
        self.0.load_speech_model()
    }
    fn host_listen(&self, params: ListenParams) {
        self.0.host_listen(FfiListenParams { rewind_seconds: params.rewind_seconds, limit_ms: params.limit_ms });
    }
    fn host_stop_listen(&self) {
        self.0.host_stop_listen();
    }
    fn intercept(&self, text: &str, cards: bool) -> Intercepted {
        match self.0.intercept(text.to_string(), cards) {
            FfiIntercepted::No => Intercepted::No,
            FfiIntercepted::Alert => Intercepted::Alert,
            FfiIntercepted::CardSilent => Intercepted::CardSilent,
            FfiIntercepted::CardSpoken { text, cards } => Intercepted::CardSpoken { text, cards },
        }
    }
    fn handle_action(&self, json: &str, skill_id: &str) -> ActionOutcome {
        let o = self.0.handle_action(json.to_string(), skill_id.to_string());
        ActionOutcome { text: o.text, display_text: o.display_text, cards: o.cards, attachments: o.attachments }
    }
    fn turn_logged(&self, turn: LoggedTurn) {
        self.0.turn_logged(turn.into());
    }
    fn facts_changed(&self) {
        self.0.facts_changed();
    }
    fn false_wake(&self) {
        self.0.false_wake();
    }
    fn dictated(&self, text: &str) {
        self.0.dictated(text.to_string());
    }
    fn engine_turn_finished(&self) {
        self.0.engine_turn_finished();
    }
}

/// An answer the engine writes, handed on to the session.
struct VoiceAnswerSink(Arc<dyn AnswerSink>);

impl ari_engine::AnswerSink for VoiceAnswerSink {
    fn text(&self, so_far: &str, markdown: Option<&str>) {
        self.0.text(so_far, markdown);
    }
    fn sentence(&self, text: &str) {
        self.0.sentence(text);
    }
}

fn engine_reply(response: FfiResponse) -> EngineReply {
    match response {
        FfiResponse::Text { body, markdown, rearm, enter_conversation, exit_conversation, facts_changed, clear_conversation } => {
            EngineReply::Text {
                body,
                markdown,
                flags: ReplyFlags { rearm, enter_conversation, exit_conversation, facts_changed, clear_conversation },
            }
        }
        FfiResponse::Action { json, skill_id, rearm, enter_conversation, exit_conversation, facts_changed, clear_conversation } => {
            EngineReply::Action {
                json,
                skill_id,
                flags: ReplyFlags { rearm, enter_conversation, exit_conversation, facts_changed, clear_conversation },
            }
        }
        FfiResponse::Binary { mime, data } => EngineReply::Binary { mime, len: data.len() },
        FfiResponse::NotUnderstood { body } => EngineReply::NotUnderstood { body },
    }
}

impl TurnEngine for AriEngine {
    fn respond(&self, input: &str, alternative: Option<&str>, answer: Option<Arc<dyn AnswerSink>>) -> (EngineReply, bool) {
        let answer = answer.map(|a| Arc::new(VoiceAnswerSink(a)) as Arc<dyn ari_engine::AnswerSink>);
        let (response, used_alternative) = self.respond_with(input.to_string(), alternative.map(str::to_string), answer);
        (engine_reply(response), used_alternative)
    }

    fn cancel_pending_reply(&self) {
        AriEngine::cancel_pending_reply(self);
    }

    fn set_conversation_active(&self, active: bool) {
        AriEngine::set_conversation_active(self, active);
    }
}

/// A model's files, read once and shared by many detectors: the tuner builds a
/// fresh engine for every step of every take's search.
#[derive(uniffi::Object)]
pub struct FfiWakeModelFiles {
    model: Vec<u8>,
    melspectrogram: Option<Vec<u8>>,
    embedding: Option<Vec<u8>>,
}

#[uniffi::export]
impl FfiWakeModelFiles {
    /// `model` is the model's own file; the feature models are needed for
    /// openWakeWord only.
    #[uniffi::constructor]
    pub fn new(model: Vec<u8>, melspectrogram: Option<Vec<u8>>, embedding: Option<Vec<u8>>) -> Arc<Self> {
        Arc::new(Self { model, melspectrogram, embedding })
    }
}

/// The names of the files `model`'s detector needs, its own first.
#[uniffi::export]
pub fn wake_model_files(model: FfiWakeWordModel) -> Vec<String> {
    let mut files = vec![model.asset_filename.clone()];
    if WakeEngine::from(model.engine) == WakeEngine::OpenWakeWord {
        files.push(OWW_MELSPECTROGRAM_FILE.into());
        files.push(OWW_EMBEDDING_FILE.into());
    }
    files
}

/// 16-bit little-endian PCM as samples. Audio crosses UniFFI as bytes: a
/// `Vec<i16>` arrives in Kotlin as a boxed `List<Short>`, and the tuner sends
/// a whole session's audio through many times over.
fn samples(pcm: &[u8]) -> Vec<i16> {
    pcm.as_chunks::<2>().0.iter().map(|b| i16::from_le_bytes(*b)).collect()
}

/// Write `pcm` as `<stem>.wav` and `sidecar` as `<stem>.txt` in `dir`, then
/// evict the oldest pairs past the caps: the wake sample recorder's store,
/// written as the session writes its captures.
#[uniffi::export]
pub fn save_clip(dir: String, stem: String, pcm: Vec<u8>, sidecar: String, max_files: u32, max_bytes: u64) -> Result<(), FfiVoiceError> {
    ari_voice::session::captures::save_clip(&PathBuf::from(dir), &stem, &samples(&pcm), &sidecar, max_files as usize, max_bytes)
        .map_err(|e| e.to_string().into())
}

/// One wake engine outside the pipeline, for the tuner and the parity test.
/// Audio is 16 kHz mono as 16-bit little-endian bytes.
#[derive(uniffi::Object)]
pub struct FfiWakeDetector {
    inner: Mutex<WakeDetector>,
}

#[uniffi::export]
impl FfiWakeDetector {
    #[uniffi::constructor]
    pub fn new(
        model: FfiWakeWordModel,
        point: FfiOperatingPoint,
        reset_on_detection: bool,
        files: Arc<FfiWakeModelFiles>,
    ) -> Result<Arc<Self>, FfiVoiceError> {
        let load = |name: &str| -> Result<Vec<u8>, String> {
            match name {
                OWW_MELSPECTROGRAM_FILE => files.melspectrogram.clone(),
                OWW_EMBEDDING_FILE => files.embedding.clone(),
                _ => None,
            }
            .ok_or_else(|| format!("{name} was not given"))
        };
        let detector = WakeDetector::from_bytes(&model.into(), point.into(), reset_on_detection, &files.model, &load)?;
        Ok(Arc::new(Self { inner: Mutex::new(detector) }))
    }

    /// Whether the wake word was detected in `pcm`.
    pub fn process_audio(&self, pcm: Vec<u8>) -> bool {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).process_audio(&samples(&pcm))
    }

    /// `pcm` fed in reads of `chunk` samples, as the capture loop would:
    /// whether any read detected. One crossing for a whole take.
    pub fn process_reads(&self, pcm: Vec<u8>, chunk: u32) -> bool {
        let mut detector = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let mut detected = false;
        for read in samples(&pcm).chunks(chunk.max(1) as usize) {
            detected |= detector.process_audio(read);
        }
        detected
    }

    /// openWakeWord only: the score of each 80 ms frame `pcm` completes.
    /// Empty for microWakeWord, which has no score to give.
    pub fn scores(&self, pcm: Vec<u8>) -> Vec<f32> {
        match &mut *self.inner.lock().unwrap_or_else(|e| e.into_inner()) {
            WakeDetector::Open(engine) => engine.scores(&samples(&pcm)),
            WakeDetector::Micro(_) => Vec::new(),
        }
    }

    pub fn reset(&self) {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).reset();
    }
}

/// The microphone's audio into the live pipeline: `count` samples of 16 kHz
/// mono. Does nothing until a pipeline has been built.
///
/// # Safety
///
/// `samples` must be valid for reading `count` samples, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ari_voice_push_audio(samples: *const i16, count: usize) {
    if samples.is_null() || count == 0 {
        return;
    }
    // SAFETY: the caller vouches for the buffer.
    pipeline::push_to_live(unsafe { std::slice::from_raw_parts(samples, count) });
}

/// `dev.heyari.ari.audio.VoiceAudio.nativePush(ShortArray, Int)`: the JNI shim
/// in front of [`ari_voice_push_audio`]. Copies the read out of the Java array
/// rather than pinning it, as the old wake engine's JNI did.
#[cfg(target_os = "android")]
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_dev_heyari_ari_audio_VoiceAudio_nativePush(
    env: *mut jni_sys::JNIEnv,
    _class: jni_sys::jclass,
    samples: jni_sys::jshortArray,
    count: jni_sys::jint,
) {
    if env.is_null() || samples.is_null() || count <= 0 {
        return;
    }
    // A read is 480 samples; anything up to 4096 needs no allocation.
    let mut stack = [0i16; 4096];
    let mut heap = Vec::new();
    let buffer: &mut [i16] = if count as usize <= stack.len() {
        &mut stack[..count as usize]
    } else {
        heap.resize(count as usize, 0);
        &mut heap
    };
    // SAFETY: a live JNIEnv for this thread, a short array of at least
    // `count` elements (the caller passes what AudioRecord.read filled), and
    // a buffer of exactly `count`.
    unsafe {
        let Some(get_region) = (**env).GetShortArrayRegion else { return };
        get_region(env, samples, 0, count, buffer.as_mut_ptr());
        let Some(exception_check) = (**env).ExceptionCheck else { return };
        if exception_check(env) != 0 {
            return;
        }
    }
    pipeline::push_to_live(buffer);
}

/// Silero's speech probability for each whole 512-sample window of `pcm`
/// (16-bit little-endian),
/// from the start, carrying state window to window. For the phone test that
/// holds this VAD to sherpa-onnx's `Vad.compute` while both are in the app.
#[uniffi::export]
pub fn silero_vad_probabilities(model: Vec<u8>, pcm: Vec<u8>) -> Result<Vec<f32>, FfiVoiceError> {
    use ari_voice::native::{SileroVad, VAD_WINDOW_SAMPLES};
    let mut vad = SileroVad::new(&model)?;
    samples(&pcm)
        .as_chunks::<VAD_WINDOW_SAMPLES>()
        .0
        .iter()
        .map(|w| vad.compute(&w.map(|s| s as f32 / 32768.0)).ok_or_else(|| "Silero failed to run".to_string().into()))
        .collect()
}
