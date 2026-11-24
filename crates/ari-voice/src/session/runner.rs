//! The session's thread, and the pipeline's commands that reach it.
//!
//! One thread owns the [`Core`] and reads one queue, waking for the earliest
//! deadline. Blocking work goes elsewhere: every engine call to one worker,
//! in order, and each host call that blocks to a thread of its own. Their
//! results come back through the same queue.

use std::any::Any;
use std::cell::Cell;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, SyncSender};
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::time::Duration;

use chrono::Local;

use super::captures::{save_utterance_capture, save_wake_capture};
use super::core::{Capture, Core, EngineCall, Finished, Job, SessionIo, SttStatus, Tag};
use super::intercept::intercept;
use super::{AnswerSink, SessionConfig, SessionHost, TurnEngine};
use crate::pipeline::{LogLevel, VoicePipeline};
use crate::stt::{ListenEnded, ListenParams, SttEvent};
use crate::wake_log::WakeOutcome;
use crate::stt::SttError;
use crate::wake_phrase::{EarlyNameCheck, match_wake_phrase, streaming_partial, wake_verdict};

/// What the host's own transcriber reports for a listen.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostHeard {
    /// The words so far, as the transcriber heard them.
    Partial(String),
    /// The transcript, and everything the transcriber was handed.
    Final { text: String, audio: Vec<i16> },
    NoSpeech { audio: Vec<i16> },
    Failed(SttError),
}

enum Msg {
    Start { verify_wake: bool, quiet: bool, reply: SyncSender<bool> },
    StartDictation { reply: SyncSender<bool> },
    StopDictation,
    Dismiss,
    SpeechFinished(u64),
    Stt { listen: u64, event: SttEvent },
    HostStt(SttEvent),
    Finished { tag: Tag, result: Finished },
    SubmitText { text: String, spoken: bool },
    CancelTyped,
    Configure(SessionConfig),
    Shutdown,
}

/// The engine, once the frontend has built it. A turn waits for it; the
/// bookkeeping calls skip it until then, as the Kotlin's `peek()` did.
#[derive(Default)]
pub(crate) struct EngineSlot {
    engine: Mutex<Option<Arc<dyn TurnEngine>>>,
    attached: Condvar,
}

impl EngineSlot {
    fn peek(&self) -> Option<Arc<dyn TurnEngine>> {
        self.engine.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn wait(&self) -> Arc<dyn TurnEngine> {
        let mut engine = self.engine.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(engine) = engine.as_ref() {
                return Arc::clone(engine);
            }
            engine = self.attached.wait(engine).unwrap_or_else(|e| e.into_inner());
        }
    }
}

/// A running session: its queue and what the pipeline reads without asking.
/// The pipeline copies what it needs out of its lock before using any of it:
/// the session thread calls into the pipeline while it handles a command, so
/// a caller waiting on the session with the lock held would never be answered.
pub(crate) struct SessionRunner {
    tx: Sender<Msg>,
    active: Arc<AtomicBool>,
}

impl Drop for SessionRunner {
    fn drop(&mut self) {
        let _ = self.tx.send(Msg::Shutdown);
    }
}

fn panic_message(panic: Box<dyn Any + Send>) -> String {
    panic
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_else(|| "a host callback failed".into())
}

struct PostingSink {
    tx: Sender<Msg>,
    tag: Tag,
}

impl AnswerSink for PostingSink {
    fn text(&self, so_far: &str, markdown: Option<&str>) {
        let result = Finished::AnswerText(so_far.into(), markdown.map(str::to_string));
        let _ = self.tx.send(Msg::Finished { tag: self.tag, result });
    }

    fn sentence(&self, text: &str) {
        let _ = self.tx.send(Msg::Finished { tag: self.tag, result: Finished::AnswerSentence(text.into()) });
    }
}

struct RunnerIo {
    pipeline: Weak<VoicePipeline>,
    host: Arc<dyn SessionHost>,
    tx: Sender<Msg>,
    engine_tx: Sender<(EngineCall, Tag)>,
    /// The listen the host is transcribing, which its events belong to.
    host_listen: Cell<Option<u64>>,
}

impl RunnerIo {
    fn pipeline(&self) -> Option<Arc<VoicePipeline>> {
        self.pipeline.upgrade()
    }
}

impl SessionIo for RunnerIo {
    fn now(&self) -> i64 {
        self.pipeline().map(|p| p.now_ms()).unwrap_or_default()
    }

    fn host(&self) -> &dyn SessionHost {
        self.host.as_ref()
    }

    fn log(&self, level: LogLevel, message: &str) {
        if let Some(p) = self.pipeline() {
            p.host().log(level, message);
        }
    }

    fn locale(&self) -> String {
        self.pipeline().map(|p| p.host().locale()).unwrap_or_else(|| "en".into())
    }

    fn stt(&self, host_transcribes: bool) -> SttStatus {
        let Some(p) = self.pipeline() else { return SttStatus::default() };
        SttStatus {
            ready: host_transcribes || p.cloud_mode() || p.stt_model_loaded(),
            streaming: !host_transcribes && p.stt_streaming(),
            early_check: !host_transcribes && p.can_check_name_early(),
            model_id: p.stt_model_id(),
        }
    }

    fn start_listen(&self, params: ListenParams, host_transcribes: bool) -> u64 {
        let Some(p) = self.pipeline() else { return 0 };
        if host_transcribes {
            let id = p.next_listen_id();
            self.host_listen.set(Some(id));
            self.host.host_listen(params);
            id
        } else {
            p.start_listen(params)
        }
    }

    fn stop_listen(&self) {
        self.host_listen.set(None);
        self.host.host_stop_listen();
        if let Some(p) = self.pipeline() {
            p.stop_listen();
        }
    }

    fn peek_recent(&self, seconds: f32) -> Vec<i16> {
        self.pipeline().map(|p| p.peek_recent(seconds)).unwrap_or_default()
    }

    fn wake_outcome(&self, outcome: WakeOutcome, hidden: bool) {
        if let Some(p) = self.pipeline() {
            p.wake_log().outcome(outcome, hidden);
        }
    }

    fn listen_ended(&self, ended: ListenEnded) {
        if let Some(p) = self.pipeline() {
            p.wake_log().listen_ended(ended.end.slug(), ended.after_ms);
        }
    }

    fn early_name(&self, result: EarlyNameCheck, after_ms: i64, in_time: bool) {
        if let Some(p) = self.pipeline() {
            p.wake_log().early_name_check(result.slug(), after_ms, in_time);
        }
    }

    fn run(&self, job: Job, tag: Tag) {
        let tx = self.tx.clone();
        let host = Arc::clone(&self.host);
        let pipeline = self.pipeline.clone();
        let failed = tx.clone();
        let spawned = std::thread::Builder::new().name("AriSessionJob".into()).spawn(move || {
            thread_started(&pipeline);
            let result = catch_unwind(AssertUnwindSafe(|| match job {
                Job::LoadSpeechModel => Finished::ModelLoaded(host.load_speech_model()),
                Job::EarlyNameCheck(pcm) => {
                    Finished::NameChecked(pipeline.upgrade().and_then(|p| p.check_name_early(&pcm)))
                }
                Job::Redecode(pcm) => Finished::Redecoded(pipeline.upgrade().and_then(|p| p.transcribe_again(&pcm))),
                Job::Intercept { text, cards } => {
                    let locale = pipeline.upgrade().map_or_else(|| "en".into(), |p| p.host().locale());
                    Finished::Intercepted(intercept(&*host, &text, cards, &locale))
                }
                Job::HandleAction { json, skill_id } => Finished::ActionHandled(host.handle_action(&json, &skill_id)),
            }))
            .unwrap_or_else(|panic| Finished::Failed(panic_message(panic)));
            let _ = tx.send(Msg::Finished { tag, result });
        });
        if let Err(e) = spawned {
            let _ = failed.send(Msg::Finished { tag, result: Finished::Failed(format!("could not start a job: {e}")) });
        }
    }

    fn engine(&self, call: EngineCall, tag: Tag) {
        let _ = self.engine_tx.send((call, tag));
    }

    fn capture(&self, dir: PathBuf, capture: Capture) {
        let pipeline = self.pipeline.clone();
        let now = self.now_wall();
        let _ = std::thread::Builder::new().name("AriCapture".into()).spawn(move || {
            thread_started(&pipeline);
            let (what, samples, saved) = match &capture {
                Capture::Wake { pcm, raw, hook } => (hook.slug(), pcm.len(), save_wake_capture(&dir, pcm, raw, *hook, now)),
                Capture::Utterance { pcm, capture } => {
                    ("utterance", pcm.len(), save_utterance_capture(&dir, pcm, capture, now, &Local))
                }
            };
            let Some(p) = pipeline.upgrade() else { return };
            match saved {
                Ok(wav) => {
                    let name = |p: Option<&std::path::Path>| {
                        p.and_then(|p| p.file_name()).map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
                    };
                    let shown = format!("{}/{}", name(wav.parent()), name(Some(&wav)));
                    p.host().log(LogLevel::Info, &format!("Captured {shown} ({samples} samples)"));
                }
                Err(e) => p.host().log(LogLevel::Warn, &format!("Failed to capture {what} audio: {e}")),
            }
        });
    }
}

impl RunnerIo {
    /// Captures are named for the wall clock, as the Kotlin named them.
    fn now_wall(&self) -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or_default()
    }
}

/// The pipeline host's [`thread_started`](crate::pipeline::VoiceHost::thread_started),
/// first thing on a thread the session starts.
fn thread_started(pipeline: &Weak<VoicePipeline>) {
    if let Some(p) = pipeline.upgrade() {
        p.host().thread_started();
    }
}

fn engine_worker(
    rx: Receiver<(EngineCall, Tag)>,
    slot: Arc<EngineSlot>,
    tx: Sender<Msg>,
    host: Arc<dyn SessionHost>,
    pipeline: Weak<VoicePipeline>,
) {
    // The engine calls the frontend's providers from here, as the voice
    // session's turns run on this thread rather than one of the frontend's.
    thread_started(&pipeline);
    for (call, tag) in rx {
        match call {
            EngineCall::Respond { input, alternative, stream } => {
                let engine = slot.wait();
                let sink = stream.then(|| Arc::new(PostingSink { tx: tx.clone(), tag }) as Arc<dyn AnswerSink>);
                let result = match catch_unwind(AssertUnwindSafe(|| engine.respond(&input, alternative.as_deref(), sink))) {
                    Ok((reply, used_alternative)) => {
                        host.engine_turn_finished();
                        Finished::Responded(reply, used_alternative)
                    }
                    Err(panic) => Finished::Failed(panic_message(panic)),
                };
                let _ = tx.send(Msg::Finished { tag, result });
            }
            EngineCall::CancelPendingReply => {
                if let Some(engine) = slot.peek() {
                    let _ = catch_unwind(AssertUnwindSafe(|| engine.cancel_pending_reply()));
                }
            }
            EngineCall::SetConversationActive(active) => {
                if let Some(engine) = slot.peek() {
                    let _ = catch_unwind(AssertUnwindSafe(|| engine.set_conversation_active(active)));
                }
            }
        }
    }
}

fn handle(core: &mut Core, io: &RunnerIo, active: &AtomicBool, msg: Msg) -> bool {
    match msg {
        Msg::Start { verify_wake, quiet, reply } => {
            let started = core.start(io, verify_wake, quiet);
            active.store(core.is_active(), Ordering::SeqCst);
            let _ = reply.send(started);
        }
        Msg::StartDictation { reply } => {
            let started = core.start_dictation(io);
            active.store(core.is_active(), Ordering::SeqCst);
            let _ = reply.send(started);
        }
        Msg::StopDictation => core.stop_dictation(io),
        Msg::Dismiss => core.dismiss(io, false),
        Msg::SpeechFinished(id) => core.speech_finished(io, id),
        Msg::Stt { listen, event } => core.stt_event(io, listen, event),
        Msg::HostStt(event) => {
            if let Some(listen) = io.host_listen.get() {
                core.stt_event(io, listen, event);
            }
        }
        Msg::Finished { tag, result } => core.finished(io, tag, result),
        Msg::SubmitText { text, spoken } => core.submit_text(io, text, spoken),
        Msg::CancelTyped => core.cancel_typed(),
        Msg::Configure(config) => core.configure(config),
        Msg::Shutdown => return false,
    }
    true
}

impl SessionRunner {
    fn spawn(pipeline: &Arc<VoicePipeline>, host: Arc<dyn SessionHost>) -> Self {
        let (tx, rx) = mpsc::channel();
        let (engine_tx, engine_rx) = mpsc::channel();
        let active = Arc::new(AtomicBool::new(false));
        {
            let slot = Arc::clone(&pipeline.engine);
            let tx = tx.clone();
            let host = Arc::clone(&host);
            let weak = Arc::downgrade(pipeline);
            std::thread::Builder::new()
                .name("AriEngineCalls".into())
                .spawn(move || engine_worker(engine_rx, slot, tx, host, weak))
                .expect("the session's engine thread");
        }
        let io = RunnerIo {
            pipeline: Arc::downgrade(pipeline),
            host,
            tx: tx.clone(),
            engine_tx,
            host_listen: Cell::new(None),
        };
        let flag = Arc::clone(&active);
        std::thread::Builder::new()
            .name("AriVoiceSession".into())
            .spawn(move || {
                thread_started(&io.pipeline);
                let mut core = Core::new();
                loop {
                    let msg = match core.next_deadline() {
                        Some(at) => {
                            let wait = (at - io.now()).max(0) as u64;
                            match rx.recv_timeout(Duration::from_millis(wait)) {
                                Ok(msg) => Some(msg),
                                Err(RecvTimeoutError::Timeout) => None,
                                Err(RecvTimeoutError::Disconnected) => break,
                            }
                        }
                        None => match rx.recv() {
                            Ok(msg) => Some(msg),
                            Err(_) => break,
                        },
                    };
                    let handled = catch_unwind(AssertUnwindSafe(|| {
                        let keep = msg.is_none_or(|msg| handle(&mut core, &io, &flag, msg));
                        core.fire_due(&io);
                        keep
                    }));
                    match handled {
                        Ok(true) => {}
                        Ok(false) => break,
                        Err(panic) => {
                            // A host callback threw on this thread. Close the
                            // session rather than leave it half way through a
                            // turn, or lose the thread and every session after.
                            io.log(LogLevel::Warn, &format!("Voice session failed: {}", panic_message(panic)));
                            let _ = catch_unwind(AssertUnwindSafe(|| core.dismiss(&io, false)));
                        }
                    }
                    flag.store(core.is_active(), Ordering::SeqCst);
                }
            })
            .expect("the voice session's thread");
        Self { tx, active }
    }
}

/// The voice session's commands. Each needs [`attach_session_host`]
/// first, and does nothing before it.
///
/// [`attach_session_host`]: VoicePipeline::attach_session_host
impl VoicePipeline {
    /// Start the voice session with `host` to show, play and run what it
    /// decides. Once per pipeline; a second call replaces the first.
    pub fn attach_session_host(self: &Arc<Self>, host: Arc<dyn SessionHost>) {
        let runner = SessionRunner::spawn(self, host);
        *self.session() = Some(runner);
    }

    fn session_parts(&self) -> Option<(Sender<Msg>, Arc<AtomicBool>)> {
        self.session().as_ref().map(|s| (s.tx.clone(), Arc::clone(&s.active)))
    }

    fn send(&self, msg: Msg) -> bool {
        self.session_parts().is_some_and(|(tx, _)| tx.send(msg).is_ok())
    }

    fn ask(&self, msg: impl FnOnce(SyncSender<bool>) -> Msg) -> bool {
        let (reply, answer) = mpsc::sync_channel(1);
        self.send(msg(reply)) && answer.recv().unwrap_or(false)
    }

    /// The engine turns go to, before or after the session host. A turn
    /// that reaches the engine before this waits for it.
    pub fn attach_engine(&self, engine: Arc<dyn TurnEngine>) {
        *self.engine.engine.lock().unwrap_or_else(|e| e.into_inner()) = Some(engine);
        self.engine.attached.notify_all();
    }

    pub fn configure_session(&self, config: SessionConfig) {
        self.send(Msg::Configure(config));
    }

    /// Begin a voice turn; false when one is already running. Its first
    /// state has gone to the host by the time this returns. Never call it
    /// from inside a host callback.
    pub fn start_turn(&self, verify_wake: bool, quiet: bool) -> bool {
        self.ask(|reply| Msg::Start { verify_wake, quiet, reply })
    }

    /// Begin a dictation; false when a session is already running.
    pub fn start_dictation(&self) -> bool {
        self.ask(|reply| Msg::StartDictation { reply })
    }

    pub fn stop_dictation(&self) {
        self.send(Msg::StopDictation);
    }

    /// End the session: the overlay was closed.
    pub fn dismiss(&self) {
        self.send(Msg::Dismiss);
    }

    /// Speech the session asked for with id `id` has finished, been stopped,
    /// or could not start.
    pub fn speech_finished(&self, id: u64) {
        self.send(Msg::SpeechFinished(id));
    }

    /// What the host's own transcriber heard, for the listen it was asked to
    /// do. Its words are judged for the wake phrase here, as the pipeline's
    /// own listens judge theirs.
    pub fn host_heard(&self, heard: HostHeard) {
        let locale = self.host().locale();
        let event = match heard {
            HostHeard::Partial(text) => {
                let Some(partial) = streaming_partial(&match_wake_phrase(&text, &locale), &locale) else { return };
                SttEvent::Listening { partial: partial.text, name_matched: partial.name_matched }
            }
            HostHeard::Final { text, audio } => {
                let heard = match_wake_phrase(&text, &locale);
                self.host().log(LogLevel::Info, &format!("Service transcript: raw='{text}' cleaned='{}'", heard.text));
                SttEvent::Done {
                    name_matched: wake_verdict(&heard, &locale),
                    text: heard.text,
                    parallel: None,
                    audio: Some(audio),
                    raw: Some(text),
                    listen_end: None,
                    speech_heard: None,
                }
            }
            HostHeard::NoSpeech { audio } => SttEvent::NoSpeech { audio: Some(audio), listen_end: None },
            HostHeard::Failed(error) => SttEvent::Error(error),
        };
        self.send(Msg::HostStt(event));
    }

    /// A turn typed into the chat, or dictated into it (`spoken`: Ari says
    /// the answer), through the same steps as a spoken one. The chat shows
    /// the words itself; what follows comes through the session host's
    /// typed-turn callbacks.
    pub fn submit_text(&self, text: String, spoken: bool) {
        self.send(Msg::SubmitText { text, spoken });
    }

    /// The chat was reset: typed turns in flight say and log nothing more.
    pub fn cancel_typed(&self) {
        self.send(Msg::CancelTyped);
    }

    pub fn is_session_active(&self) -> bool {
        self.session_parts().is_some_and(|(_, active)| active.load(Ordering::SeqCst))
    }

    /// A listen's event: to the session when there is one, else to the host.
    pub(crate) fn stt_event(&self, listen: u64, event: SttEvent) {
        match self.session_parts() {
            Some((tx, _)) => {
                let _ = tx.send(Msg::Stt { listen, event });
            }
            None => self.host().stt_event(event),
        }
    }
}
