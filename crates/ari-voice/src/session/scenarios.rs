//! Voice turns end to end against a fake host and a clock that moves only
//! when a scenario says so. Each scenario is what `VoiceSession.kt` did for
//! that situation on the phone.

use std::sync::Mutex;
use std::collections::VecDeque;
use std::path::PathBuf;

use super::core::{Capture, Core, EngineCall, Finished, Job, SessionIo, SttStatus, Tag};
use super::{
    ActionOutcome, CaptureMode, EngineReply, Intercepted, LoggedReply, LoggedTurn, Phrase, ReplyFlags,
    SessionConfig, SessionError, SessionHost, SessionState, TypedFinish, UtteranceOutcome, UtteranceTurn,
    WakeCaptureHook,
};
use crate::cloud_stt::CloudSttFailure;
use crate::endpoint::ListenEnd;
use crate::pipeline::LogLevel;
use crate::stt::{ListenEnded, ListenParams, SttError, SttEvent};
use crate::wake_log::WakeOutcome;
use crate::wake_phrase::EarlyNameCheck;

#[derive(Debug, Clone, PartialEq)]
enum Ev {
    State(SessionState),
    Reveal,
    Cue(f32),
    Speak { id: u64, text: String, flush: bool },
    StopSpeaking,
    CaptureMode(CaptureMode),
    Listen { id: u64, rewind: f32, give_up: Option<i64>, limit: Option<i64> },
    StopListen,
    Outcome(WakeOutcome, bool),
    ListenEnded(ListenEnd, i64),
    EarlyName(EarlyNameCheck, i64, bool),
    Capture(Capture),
    Logged(LoggedTurn),
    FactsChanged,
    FalseWake,
    Dictated(String),
    TypedThinking(u64),
    AnswerText(u64, String, Option<String>),
    TypedFinished(u64, TypedFinish),
}

struct Fake {
    now: Mutex<i64>,
    events: Mutex<Vec<Ev>>,
    logs: Mutex<Vec<String>>,
    stt: Mutex<SttStatus>,
    next_listen: Mutex<u64>,
    listen: Mutex<u64>,
    jobs: Mutex<VecDeque<(Job, Tag)>>,
    engine: Mutex<VecDeque<(EngineCall, Tag)>>,
}

const PREROLL: [i16; 4] = [1, 2, 3, 4];
const UTTERANCE: [i16; 3] = [7, 8, 9];

impl Fake {
    fn push(&self, ev: Ev) {
        self.events.lock().unwrap().push(ev);
    }
}

impl SessionHost for Fake {
    fn state(&self, state: SessionState) {
        self.push(Ev::State(state));
    }
    fn reveal(&self) {
        self.push(Ev::Reveal);
    }
    fn play_cue(&self, volume: f32) {
        self.push(Ev::Cue(volume));
    }
    fn speak(&self, id: u64, text: &str, flush: bool) {
        self.push(Ev::Speak { id, text: text.into(), flush });
    }
    fn stop_speaking(&self) {
        self.push(Ev::StopSpeaking);
    }
    fn phrase(&self, phrase: Phrase) -> String {
        match phrase {
            Phrase::PleaseWait => "One moment.".into(),
            Phrase::PleaseRepeat => "Say that again?".into(),
        }
    }
    fn capture_mode(&self, mode: CaptureMode) {
        self.push(Ev::CaptureMode(mode));
    }
    fn load_speech_model(&self) -> bool {
        unreachable!("a job")
    }
    fn intercept(&self, _text: &str, _cards: bool) -> Intercepted {
        unreachable!("a job")
    }
    fn handle_action(&self, _json: &str, _skill_id: &str) -> ActionOutcome {
        unreachable!("a job")
    }
    fn turn_logged(&self, turn: LoggedTurn) {
        self.push(Ev::Logged(turn));
    }
    fn facts_changed(&self) {
        self.push(Ev::FactsChanged);
    }
    fn false_wake(&self) {
        self.push(Ev::FalseWake);
    }
    fn dictated(&self, text: &str) {
        self.push(Ev::Dictated(text.into()));
    }
    fn typed_thinking(&self, turn: u64) {
        self.push(Ev::TypedThinking(turn));
    }
    fn answer_text(&self, turn: u64, so_far: &str, markdown: Option<&str>) {
        self.push(Ev::AnswerText(turn, so_far.into(), markdown.map(str::to_string)));
    }
    fn typed_finished(&self, turn: u64, finish: TypedFinish) {
        self.push(Ev::TypedFinished(turn, finish));
    }
}

impl SessionIo for Fake {
    fn now(&self) -> i64 {
        *self.now.lock().unwrap()
    }
    fn host(&self) -> &dyn SessionHost {
        self
    }
    fn log(&self, _level: LogLevel, message: &str) {
        self.logs.lock().unwrap().push(message.into());
    }
    fn locale(&self) -> String {
        "en".into()
    }
    fn stt(&self, _host_transcribes: bool) -> SttStatus {
        self.stt.lock().unwrap().clone()
    }
    fn start_listen(&self, params: ListenParams, _host_transcribes: bool) -> u64 {
        let id = *self.next_listen.lock().unwrap() + 1;
        *self.next_listen.lock().unwrap() = id;
        *self.listen.lock().unwrap() = id;
        self.push(Ev::Listen {
            id,
            rewind: params.rewind_seconds,
            give_up: params.give_up_without_speech_ms,
            limit: params.limit_ms,
        });
        id
    }
    fn stop_listen(&self) {
        self.push(Ev::StopListen);
    }
    fn peek_recent(&self, seconds: f32) -> Vec<i16> {
        assert_eq!(seconds, 2.0);
        PREROLL.to_vec()
    }
    fn wake_outcome(&self, outcome: WakeOutcome, hidden: bool) {
        self.push(Ev::Outcome(outcome, hidden));
    }
    fn listen_ended(&self, ended: ListenEnded) {
        self.push(Ev::ListenEnded(ended.end, ended.after_ms));
    }
    fn early_name(&self, result: EarlyNameCheck, after_ms: i64, in_time: bool) {
        self.push(Ev::EarlyName(result, after_ms, in_time));
    }
    fn run(&self, job: Job, tag: Tag) {
        self.jobs.lock().unwrap().push_back((job, tag));
    }
    fn engine(&self, call: EngineCall, tag: Tag) {
        self.engine.lock().unwrap().push_back((call, tag));
    }
    fn capture(&self, dir: PathBuf, capture: Capture) {
        assert_eq!(dir, PathBuf::from("/captures"));
        self.push(Ev::Capture(capture));
    }
}

/// A scenario: the core, its fake world, and the moves a test makes.
struct S {
    core: Core,
    io: Fake,
}

fn streaming() -> SttStatus {
    SttStatus { ready: true, streaming: true, early_check: false, model_id: Some("kroko".into()) }
}

fn cloud() -> SttStatus {
    SttStatus { ready: true, streaming: false, early_check: true, model_id: None }
}

fn keep_everything() -> SessionConfig {
    SessionConfig { keep_everything_audio: true, captures_dir: Some("/captures".into()), ..SessionConfig::default() }
}

fn text(body: &str) -> EngineReply {
    EngineReply::Text { body: body.into(), markdown: None, flags: ReplyFlags::default() }
}

fn flagged(body: &str, flags: ReplyFlags) -> EngineReply {
    EngineReply::Text { body: body.into(), markdown: None, flags }
}

fn not_understood() -> EngineReply {
    EngineReply::NotUnderstood { body: "Sorry, I didn't catch that.".into() }
}

fn listening(partial: &str) -> Ev {
    Ev::State(SessionState::Listening { partial: partial.into() })
}

fn responding(text: &str, cards: bool) -> Ev {
    Ev::State(SessionState::Responding { text: text.into(), cards })
}

fn idle(keep_cards: bool) -> Ev {
    Ev::State(SessionState::Idle { keep_cards })
}

fn done(text: &str, raw: &str, name_matched: Option<bool>) -> SttEvent {
    SttEvent::Done {
        text: text.into(),
        parallel: None,
        audio: Some(UTTERANCE.to_vec()),
        raw: Some(raw.into()),
        name_matched,
        listen_end: None,
    }
}

fn partial(text: &str, name_matched: Option<bool>) -> SttEvent {
    SttEvent::Listening { partial: text.into(), name_matched }
}

impl S {
    fn new(stt: SttStatus, config: SessionConfig) -> Self {
        let io = Fake {
            now: Mutex::new(1_000),
            events: Mutex::default(),
            logs: Mutex::default(),
            stt: Mutex::new(stt),
            next_listen: Mutex::new(0),
            listen: Mutex::new(0),
            jobs: Mutex::default(),
            engine: Mutex::default(),
        };
        let mut core = Core::new();
        core.configure(config);
        Self { core, io }
    }

    /// Everything the host saw since the last call.
    fn take(&self) -> Vec<Ev> {
        std::mem::take(&mut *self.io.events.lock().unwrap())
    }

    fn start(&mut self, verify_wake: bool, quiet: bool) -> bool {
        self.core.start(&self.io, verify_wake, quiet)
    }

    fn stt(&mut self, event: SttEvent) {
        let listen = *self.io.listen.lock().unwrap();
        self.core.stt_event(&self.io, listen, event);
    }

    fn advance(&mut self, ms: i64) {
        let target = *self.io.now.lock().unwrap() + ms;
        while let Some(at) = self.core.next_deadline().filter(|at| *at <= target) {
            { let v = at.max(*self.io.now.lock().unwrap()); *self.io.now.lock().unwrap() = v; }
            self.core.fire_due(&self.io);
        }
        { let v = target; *self.io.now.lock().unwrap() = v; }
    }

    fn job(&self) -> (Job, Tag) {
        self.io.jobs.lock().unwrap().pop_front().expect("a job")
    }

    fn no_job(&self) {
        assert_eq!(self.io.jobs.lock().unwrap().front(), None);
    }

    fn finish(&mut self, epoch: Tag, result: Finished) {
        self.core.finished(&self.io, epoch, result);
    }

    /// The next engine turn, skipping the bookkeeping calls before it.
    fn respond_call(&self) -> (String, Option<String>, bool, Tag) {
        loop {
            let (call, epoch) = self.io.engine.lock().unwrap().pop_front().expect("an engine call");
            if let EngineCall::Respond { input, alternative, stream } = call {
                return (input, alternative, stream, epoch);
            }
        }
    }

    fn engine_calls(&self) -> Vec<EngineCall> {
        self.io.engine.lock().unwrap().drain(..).map(|(c, _)| c).collect()
    }

    /// Answer the intercept with nothing and the engine's turn with `reply`.
    fn answer(&mut self, reply: EngineReply) {
        let (job, epoch) = self.job();
        assert!(matches!(job, Job::Intercept { .. }), "{job:?}");
        self.finish(epoch, Finished::Intercepted(Intercepted::No));
        let (_, _, _, epoch) = self.respond_call();
        self.finish(epoch, Finished::Responded(reply, false));
    }

    fn speech_done(&mut self, id: u64) {
        self.core.speech_finished(&self.io, id);
    }

    fn logs(&self) -> Vec<String> {
        self.io.logs.lock().unwrap().clone()
    }
}

fn spoken(events: &[Ev]) -> Vec<(u64, String, bool)> {
    events
        .iter()
        .filter_map(|e| match e {
            Ev::Speak { id, text, flush } => Some((*id, text.clone(), *flush)),
            _ => None,
        })
        .collect()
}

fn utterance(events: &[Ev]) -> Option<super::UtteranceCapture> {
    events.iter().find_map(|e| match e {
        Ev::Capture(Capture::Utterance { capture, .. }) => Some(capture.clone()),
        _ => None,
    })
}

fn wake_captures(events: &[Ev]) -> Vec<(WakeCaptureHook, String, Vec<i16>)> {
    events
        .iter()
        .filter_map(|e| match e {
            Ev::Capture(Capture::Wake { pcm, raw, hook }) => Some((*hook, raw.clone(), pcm.clone())),
            _ => None,
        })
        .collect()
}

#[test]
fn a_quiet_wake_shows_on_the_partial_that_names_ari_and_answers() {
    let mut s = S::new(streaming(), keep_everything());
    assert!(s.start(true, true));
    assert_eq!(
        s.take(),
        [listening(""), Ev::Listen { id: 1, rewind: 2.0, give_up: Some(3_000), limit: Some(6_000) }]
    );

    s.advance(400);
    s.stt(partial("", Some(false)));
    s.stt(partial("what time", Some(true)));
    assert_eq!(s.take(), [Ev::EarlyName(EarlyNameCheck::Pass, 400, true), Ev::Reveal, Ev::Cue(1.0), listening("what time")]);

    // The name wait finds the wake already shown and does nothing.
    s.advance(3_000);
    s.stt(done("what time is it", "hey ari what time is it", Some(true)));
    let events = s.take();
    assert_eq!(events[0], Ev::Outcome(WakeOutcome::Accepted, false));
    assert_eq!(wake_captures(&events), [(WakeCaptureHook::Accepted, "hey ari what time is it".into(), PREROLL.to_vec())]);
    assert_eq!(events.last(), Some(&Ev::State(SessionState::Thinking)));

    let (job, epoch) = s.job();
    assert_eq!(job, Job::Intercept { text: "what time is it".into(), cards: true });
    s.finish(epoch, Finished::Intercepted(Intercepted::No));
    assert_eq!(s.respond_call(), ("what time is it".into(), None, true, epoch));
    s.finish(epoch, Finished::Responded(text("It's 9:41."), false));
    let events = s.take();
    assert_eq!(
        events,
        [
            Ev::Logged(LoggedTurn {
                clear: false,
                heard: "what time is it".into(),
                reply: Some(LoggedReply { text: "It's 9:41.".into(), markdown: None, skill_id: None }),
            }),
            responding("It's 9:41.", false),
            Ev::Speak { id: 1, text: "It's 9:41.".into(), flush: true },
        ]
    );

    // Spoken in a second: the overlay holds to three seconds from the start
    // of speech, then closes.
    s.advance(1_000);
    s.speech_done(1);
    s.advance(1_999);
    assert_eq!(s.take(), []);
    s.advance(1);
    let events = s.take();
    assert_eq!(events.last(), Some(&idle(false)));
    // The wake already has its ending: the dismiss adds none.
    assert!(!events.iter().any(|e| matches!(e, Ev::Outcome(..))));
    let capture = utterance(&events).expect("the utterance");
    assert_eq!(capture.turn, UtteranceTurn::Opening);
    assert_eq!(capture.outcome, UtteranceOutcome::Answered);
    assert_eq!(capture.response.as_deref(), Some("text"));
    assert_eq!(capture.used, "what time is it");
    assert_eq!(capture.model.as_deref(), Some("kroko"));
}

#[test]
fn a_wake_heard_without_the_name_is_turned_away() {
    // An ordinary wake, shown from the start, so the user saw it and it counts.
    let mut s = S::new(streaming(), keep_everything());
    s.start(true, false);
    assert_eq!(s.take()[..2], [listening(""), Ev::Listen { id: 1, rewind: 2.0, give_up: None, limit: None }]);
    s.stt(done("going out", "we're going out", Some(false)));
    let events = s.take();
    assert_eq!(wake_captures(&events), [(WakeCaptureHook::Rejected, "we're going out".into(), UTTERANCE.to_vec())]);
    assert!(events.contains(&Ev::Outcome(WakeOutcome::Rejected, false)));
    assert!(events.contains(&Ev::FalseWake));
    assert_eq!(events.last(), Some(&idle(false)));
    assert!(s.logs().contains(&"Wake rejected: raw='we're going out'".to_string()));
}

#[test]
fn a_quiet_wake_turned_away_unseen_is_not_counted() {
    let mut s = S::new(cloud(), keep_everything());
    s.start(true, true);
    s.take();
    let (job, _) = s.job();
    assert_eq!(job, Job::EarlyNameCheck(PREROLL.to_vec()));
    s.stt(SttEvent::Transcribing);
    s.stt(done("going out", "we're going out", Some(false)));
    let events = s.take();
    assert!(events.contains(&Ev::Outcome(WakeOutcome::Rejected, true)));
    assert!(!events.contains(&Ev::FalseWake));
    assert!(!events.contains(&Ev::Reveal));
}

#[test]
fn a_quiet_wake_that_names_nobody_in_time_is_turned_away() {
    let mut s = S::new(streaming(), keep_everything());
    s.start(true, true);
    s.stt(partial("we're going", Some(false)));
    s.take();
    s.advance(2_999);
    assert_eq!(s.take(), []);
    s.advance(1);
    let events = s.take();
    assert_eq!(wake_captures(&events), [(WakeCaptureHook::Rejected, "we're going".into(), PREROLL.to_vec())]);
    assert!(events.contains(&Ev::Outcome(WakeOutcome::Rejected, true)));
    assert!(!events.contains(&Ev::FalseWake));
    assert_eq!(events.last(), Some(&idle(false)));

    // With nothing said at all, the clip is a silent one.
    let mut s = S::new(streaming(), keep_everything());
    s.start(true, true);
    s.advance(3_000);
    let events = s.take();
    assert_eq!(wake_captures(&events), [(WakeCaptureHook::Silent, String::new(), PREROLL.to_vec())]);
    assert!(events.contains(&Ev::Outcome(WakeOutcome::Silent, true)));
}

#[test]
fn the_early_check_shows_a_quiet_wake_and_its_verdict_stands() {
    let mut s = S::new(cloud(), keep_everything());
    s.start(true, true);
    s.take();
    s.advance(900);
    let (_, epoch) = s.job();
    s.finish(epoch, Finished::NameChecked(Some(EarlyNameCheck::Pass)));
    assert_eq!(s.take(), [Ev::EarlyName(EarlyNameCheck::Pass, 900, true), Ev::Reveal]);
    // The whole transcript misses the name ("Hayori navigate to McDonald's"),
    // but the user has seen Ari answer to it.
    s.stt(SttEvent::Transcribing);
    s.stt(SttEvent::Done {
        text: "navigate to McDonald's".into(),
        parallel: None,
        audio: None,
        raw: Some("Hayori navigate to McDonald's".into()),
        name_matched: Some(false),
        listen_end: Some(ListenEnded { end: ListenEnd::Endpoint, after_ms: 2_400 }),
    });
    let events = s.take();
    assert_eq!(events[..3], [
        Ev::State(SessionState::Thinking),
        Ev::ListenEnded(ListenEnd::Endpoint, 2_400),
        Ev::Outcome(WakeOutcome::Accepted, false),
    ]);
    let (job, _) = s.job();
    assert_eq!(job, Job::Intercept { text: "navigate to McDonald's".into(), cards: true });
}

#[test]
fn a_late_early_check_changes_nothing() {
    let mut s = S::new(cloud(), keep_everything());
    s.start(true, true);
    s.take();
    let (_, epoch) = s.job();
    s.stt(SttEvent::Transcribing);
    s.stt(done("what time is it", "hey ari what time is it", Some(true)));
    s.take();
    s.advance(2_000);
    s.finish(epoch, Finished::NameChecked(Some(EarlyNameCheck::Pass)));
    assert_eq!(s.take(), [Ev::EarlyName(EarlyNameCheck::Pass, 2_000, false)]);
}

#[test]
fn a_quiet_wake_that_hears_nothing_closes_unseen_and_keeps_the_audio() {
    let mut s = S::new(cloud(), keep_everything());
    s.start(true, true);
    s.take();
    s.stt(SttEvent::NoSpeech {
        audio: Some(UTTERANCE.to_vec()),
        listen_end: Some(ListenEnded { end: ListenEnd::GiveUp, after_ms: 3_000 }),
    });
    let events = s.take();
    assert_eq!(events[0], Ev::ListenEnded(ListenEnd::GiveUp, 3_000));
    assert_eq!(wake_captures(&events), [(WakeCaptureHook::NoSpeech, String::new(), UTTERANCE.to_vec())]);
    assert!(events.contains(&Ev::Outcome(WakeOutcome::NoSpeech, true)));
    assert!(!events.iter().any(|e| matches!(e, Ev::State(SessionState::Error(_)))));
    assert_eq!(events.last(), Some(&idle(false)));
}

#[test]
fn a_blank_whisper_transcript_on_a_wake_is_no_speech() {
    let mut s = S::new(streaming(), keep_everything());
    s.start(true, false);
    s.take();
    s.stt(done("", "", None));
    let events = s.take();
    assert!(events.contains(&Ev::Outcome(WakeOutcome::NoSpeech, false)));
    assert!(events.contains(&Ev::FalseWake));
    assert!(s.logs().contains(&"Wake heard no speech".to_string()));
}

#[test]
fn a_tap_to_talk_turn_that_hears_nothing_shows_why() {
    let mut s = S::new(cloud(), keep_everything());
    s.start(false, false);
    s.take();
    s.stt(SttEvent::NoSpeech { audio: Some(UTTERANCE.to_vec()), listen_end: None });
    assert_eq!(s.take(), [Ev::State(SessionState::Error(SessionError::NoSpeech))]);
    s.advance(2_500);
    assert_eq!(s.take().last(), Some(&idle(false)));
}

#[test]
fn a_bare_hey_ari_gets_its_chime_and_a_fresh_listen() {
    // The buffered paths: the streaming one never ends a listen on the wake
    // phrase alone, so its silence watcher handles a bare "Hey Ari".
    let mut s = S::new(cloud(), keep_everything());
    s.start(true, true);
    s.job();
    s.stt(SttEvent::Transcribing);
    s.take();
    s.stt(done("", "hey ari", Some(true)));
    let events = s.take();
    assert_eq!(events[..2], [Ev::Reveal, Ev::Outcome(WakeOutcome::Accepted, false)]);
    assert_eq!(events[events.len() - 3..], [listening(""), Ev::Cue(1.0), Ev::Listen { id: 2, rewind: 0.0, give_up: None, limit: None }]);
    // The command itself is not verified again.
    s.stt(done("what time is it", "what time is it", Some(false)));
    let (job, _) = s.job();
    assert_eq!(job, Job::Intercept { text: "what time is it".into(), cards: true });
}

#[test]
fn a_cold_start_says_one_moment_then_asks_again() {
    let mut s = S::new(SttStatus::default(), keep_everything());
    assert!(s.start(true, true));
    assert_eq!(
        s.take(),
        [
            Ev::Reveal,
            Ev::State(SessionState::Preparing { message: "One moment.".into() }),
            Ev::Outcome(WakeOutcome::ColdStart, false),
            Ev::Speak { id: 1, text: "One moment.".into(), flush: false },
        ]
    );
    let (job, epoch) = s.job();
    assert_eq!(job, Job::LoadSpeechModel);
    *s.io.stt.lock().unwrap() = streaming();
    s.finish(epoch, Finished::ModelLoaded(true));
    assert_eq!(s.take(), [Ev::Speak { id: 2, text: "Say that again?".into(), flush: true }]);
    s.speech_done(1);
    assert_eq!(s.take(), []);
    s.speech_done(2);
    assert_eq!(s.take(), [listening(""), Ev::Listen { id: 1, rewind: 0.0, give_up: None, limit: None }, Ev::Cue(1.0)]);
    // A wake opened this mic, so the window is the short one; with the
    // wake phrase long gone, silence keeps no audio and counts nothing.
    s.advance(9_000);
    let events = s.take();
    assert!(wake_captures(&events).is_empty());
    assert!(!events.contains(&Ev::FalseWake));
    assert_eq!(events.last(), Some(&idle(false)));
    assert!(s.logs().contains(&"No speech detected within 8000 ms — dismissing".to_string()));
}

#[test]
fn a_cold_start_with_no_model_says_so() {
    let mut s = S::new(SttStatus::default(), keep_everything());
    s.start(true, true);
    s.take();
    let (_, epoch) = s.job();
    s.finish(epoch, Finished::ModelLoaded(false));
    assert_eq!(s.take(), [Ev::State(SessionState::Error(SessionError::NoSpeechModel))]);
    s.advance(2_499);
    assert_eq!(s.take(), []);
    s.advance(1);
    assert_eq!(s.take().last(), Some(&idle(false)));
}

#[test]
fn a_skills_question_rearms_the_mic_once_it_has_been_asked() {
    let mut s = S::new(streaming(), keep_everything());
    s.start(false, false);
    s.stt(done("play some music", "play some music", None));
    s.take();
    s.answer(flagged("Which app?", ReplyFlags { rearm: true, ..ReplyFlags::default() }));
    let events = s.take();
    assert_eq!(spoken(&events), [(1, "Which app?".into(), true)]);
    // The mic stays shut until the question has been asked.
    assert!(!events.iter().any(|e| matches!(e, Ev::Listen { .. })));
    s.speech_done(1);
    let events = s.take();
    assert_eq!(events[..3], [listening(""), Ev::Cue(0.8), Ev::Listen { id: 2, rewind: 0.0, give_up: None, limit: None }]);
    assert_eq!(utterance(&events).map(|c| c.turn), Some(UtteranceTurn::Opening));

    // The answer goes to the engine, not to a card.
    s.stt(done("spotify", "spotify", None));
    let (job, epoch) = s.job();
    assert_eq!(job, Job::Intercept { text: "spotify".into(), cards: false });
    s.finish(epoch, Finished::Intercepted(Intercepted::No));
    let (_, _, _, epoch) = s.respond_call();
    s.finish(epoch, Finished::Responded(text("Playing."), false));
    s.speech_done(2);
    s.advance(3_000);
    assert_eq!(utterance(&s.take()).map(|c| c.turn), Some(UtteranceTurn::Reply));
}

#[test]
fn a_reply_turn_that_hears_nothing_times_out_after_thirty_seconds() {
    let mut s = S::new(streaming(), keep_everything());
    s.start(false, false);
    s.stt(done("play some music", "play some music", None));
    s.answer(flagged("Which app?", ReplyFlags { rearm: true, ..ReplyFlags::default() }));
    s.speech_done(1);
    s.take();
    s.advance(30_000);
    assert_eq!(s.take(), []);
    s.advance(1_000);
    assert_eq!(s.take().last(), Some(&idle(false)));
    assert!(s.engine_calls().contains(&EngineCall::CancelPendingReply));
}

#[test]
fn lets_talk_turn_based_enters_rearms_and_leaves() {
    let mut s = S::new(streaming(), keep_everything());
    s.start(false, false);
    s.stt(done("let's talk", "let's talk", None));
    s.answer(flagged("Okay, I'm listening.", ReplyFlags { enter_conversation: true, ..ReplyFlags::default() }));
    assert!(s.engine_calls().contains(&EngineCall::SetConversationActive(true)));
    // No barge-in configured: speech first, then the mic.
    s.speech_done(1);
    let events = s.take();
    assert!(!events.contains(&Ev::CaptureMode(CaptureMode::Conversation)));
    assert!(events.contains(&Ev::Listen { id: 2, rewind: 0.0, give_up: None, limit: None }));

    s.stt(done("stop", "stop", None));
    s.answer(flagged("Okay.", ReplyFlags { exit_conversation: true, ..ReplyFlags::default() }));
    assert_eq!(utterance(&s.take()), None);
    s.speech_done(2);
    let events = s.take();
    assert_eq!(utterance(&events).map(|c| c.turn), Some(UtteranceTurn::Talk));
    assert_eq!(events.last(), Some(&idle(false)));
    assert!(s.engine_calls().contains(&EngineCall::SetConversationActive(false)));
}

#[test]
fn a_barge_in_cuts_ari_off_and_becomes_the_next_turn() {
    let config = SessionConfig { barge_in_enabled: true, echo_canceller: true, ..keep_everything() };
    let mut s = S::new(streaming(), config);
    s.start(false, false);
    s.stt(done("let's talk", "let's talk", None));
    s.answer(flagged("Okay, I'm listening.", ReplyFlags { enter_conversation: true, ..ReplyFlags::default() }));
    let events = s.take();
    // The mic opens while Ari speaks.
    let listen = events.iter().position(|e| matches!(e, Ev::Listen { .. })).expect("a listen");
    let speak = events.iter().position(|e| matches!(e, Ev::Speak { .. })).expect("speech");
    assert!(events.contains(&Ev::CaptureMode(CaptureMode::Conversation)));
    assert!(listen < speak);

    s.stt(done("what's the weather", "what's the weather", None));
    let events = s.take();
    assert_eq!(events[0], Ev::StopSpeaking);
    let (job, _) = s.job();
    assert_eq!(job, Job::Intercept { text: "what's the weather".into(), cards: false });
    // The cut speech reporting in late changes nothing.
    s.speech_done(1);
    assert_eq!(s.take(), []);
}

#[test]
fn barge_in_without_an_echo_canceller_stays_turn_based() {
    let config = SessionConfig { barge_in_enabled: true, echo_canceller: false, ..keep_everything() };
    let mut s = S::new(streaming(), config);
    s.start(false, false);
    s.stt(done("let's talk", "let's talk", None));
    s.take();
    s.answer(flagged("Okay, I'm listening.", ReplyFlags { enter_conversation: true, ..ReplyFlags::default() }));
    let events = s.take();
    assert!(!events.iter().any(|e| matches!(e, Ev::Listen { .. } | Ev::CaptureMode(_))));
}

#[test]
fn events_from_a_listen_that_has_gone_are_dropped() {
    let mut s = S::new(streaming(), keep_everything());
    s.start(false, false);
    let old = *s.io.listen.lock().unwrap();
    s.core.dismiss(&s.io, false);
    s.take();
    s.core.stt_event(&s.io, old, done("what time is it", "what time is it", None));
    assert_eq!(s.take(), []);
    s.no_job();

    // A result from the dismissed session is dropped too.
    s.start(false, false);
    s.stt(done("what time is it", "what time is it", None));
    let (_, epoch) = s.job();
    s.core.dismiss(&s.io, false);
    s.take();
    s.finish(epoch, Finished::Intercepted(Intercepted::No));
    assert!(s.io.engine.lock().unwrap().iter().all(|(c, _)| !matches!(c, EngineCall::Respond { .. })));
}

#[test]
fn a_dismiss_mid_turn_still_keeps_what_was_said() {
    let mut s = S::new(streaming(), keep_everything());
    s.start(false, false);
    s.stt(done("what time is it", "what time is it", None));
    let (_, epoch) = s.job();
    s.finish(epoch, Finished::Intercepted(Intercepted::No));
    s.take();
    s.core.dismiss(&s.io, false);
    let capture = utterance(&s.take()).expect("the utterance");
    assert_eq!(capture.outcome, UtteranceOutcome::Blank);
    assert_eq!(capture.transcript, "what time is it");
    let (_, _, _, epoch) = s.respond_call();
    s.finish(epoch, Finished::Responded(text("It's 9:41."), false));
    assert_eq!(s.take(), []);
}

#[test]
fn a_wake_nobody_speaks_to_times_out_at_eight_seconds() {
    let mut s = S::new(streaming(), keep_everything());
    s.start(true, false);
    s.take();
    s.advance(8_999);
    assert_eq!(s.take(), []);
    s.advance(1);
    let events = s.take();
    assert_eq!(wake_captures(&events), [(WakeCaptureHook::Silent, String::new(), PREROLL.to_vec())]);
    assert!(events.contains(&Ev::Outcome(WakeOutcome::Silent, false)));
    assert!(events.contains(&Ev::FalseWake));
}

#[test]
fn a_tap_on_the_mic_gets_thirty_seconds_and_counts_nothing() {
    let mut s = S::new(streaming(), keep_everything());
    s.start(false, false);
    s.take();
    s.advance(30_000);
    assert_eq!(s.take(), []);
    s.advance(1_000);
    let events = s.take();
    assert!(wake_captures(&events).is_empty());
    assert!(!events.iter().any(|e| matches!(e, Ev::Outcome(..) | Ev::FalseWake)));
    assert_eq!(events.last(), Some(&idle(false)));
}

#[test]
fn speech_keeps_the_silence_window_open() {
    let mut s = S::new(streaming(), keep_everything());
    s.start(true, false);
    s.advance(7_000);
    s.stt(partial("what's the", Some(true)));
    s.advance(7_500);
    assert!(!s.take().contains(&idle(false)));
}

#[test]
fn the_parallel_transcript_rescues_a_turn_and_is_flashed() {
    let mut s = S::new(streaming(), keep_everything());
    s.start(false, false);
    s.stt(SttEvent::Done {
        text: "whats the leather".into(),
        parallel: Some("whats the weather".into()),
        audio: Some(UTTERANCE.to_vec()),
        raw: Some("whats the leather".into()),
        name_matched: None,
        listen_end: None,
    });
    let (_, epoch) = s.job();
    s.finish(epoch, Finished::Intercepted(Intercepted::No));
    assert_eq!(s.respond_call(), ("whats the leather".into(), Some("whats the weather".into()), true, epoch));
    s.finish(epoch, Finished::Responded(not_understood(), false));
    assert_eq!(s.respond_call(), ("whats the weather".into(), None, false, epoch));
    s.take();
    s.finish(epoch, Finished::Responded(text("Sunny."), false));
    assert_eq!(s.take(), [listening("whats the weather")]);
    s.advance(600);
    let events = s.take();
    assert!(matches!(&events[0], Ev::Logged(t) if t.heard == "whats the weather"));
    s.speech_done(1);
    s.advance(3_000);
    let capture = utterance(&s.take()).expect("the utterance");
    assert_eq!(capture.outcome, UtteranceOutcome::Rescued);
    assert_eq!(capture.used, "whats the weather");
}

#[test]
fn the_engine_acting_on_the_parallel_transcript_needs_no_retry() {
    let mut s = S::new(streaming(), keep_everything());
    s.start(false, false);
    s.stt(SttEvent::Done {
        text: "reminded me to buy milk".into(),
        parallel: Some("remind me to buy milk".into()),
        audio: Some(UTTERANCE.to_vec()),
        raw: None,
        name_matched: None,
        listen_end: None,
    });
    let (_, epoch) = s.job();
    s.finish(epoch, Finished::Intercepted(Intercepted::No));
    s.respond_call();
    s.take();
    s.finish(epoch, Finished::Responded(text("I'll remind you."), true));
    assert!(s.io.engine.lock().unwrap().is_empty());
    assert_eq!(s.take(), [listening("remind me to buy milk")]);
}

#[test]
fn the_offline_decode_is_the_last_retry() {
    let mut s = S::new(streaming(), keep_everything());
    s.start(false, false);
    s.stt(done("whats the leather", "whats the leather", None));
    s.answer(not_understood());
    let (job, epoch) = s.job();
    assert_eq!(job, Job::Redecode(UTTERANCE.to_vec()));
    s.finish(epoch, Finished::Redecoded(Some("whats the weather".into())));
    assert_eq!(s.respond_call(), ("whats the weather".into(), None, false, epoch));
    s.finish(epoch, Finished::Responded(text("Sunny."), false));
    s.advance(600);
    s.speech_done(1);
    s.advance(3_000);
    let capture = utterance(&s.take()).expect("the utterance");
    assert_eq!(capture.offline.as_deref(), Some("whats the weather"));
    assert_eq!(capture.outcome, UtteranceOutcome::Rescued);
}

#[test]
fn nothing_understood_is_said_and_recorded() {
    let mut s = S::new(cloud(), keep_everything());
    s.start(false, false);
    s.stt(SttEvent::Transcribing);
    s.stt(done("blah", "blah", None));
    s.answer(not_understood());
    // Not streaming: no re-decode.
    s.no_job();
    let events = s.take();
    assert!(spoken(&events).iter().any(|(_, t, _)| t == "Sorry, I didn't catch that."));
    s.speech_done(1);
    s.advance(3_000);
    assert_eq!(utterance(&s.take()).map(|c| c.outcome), Some(UtteranceOutcome::NotUnderstood));
}

#[test]
fn an_on_device_answer_is_shown_and_spoken_as_it_is_written() {
    let mut s = S::new(streaming(), keep_everything());
    s.start(false, false);
    s.stt(done("tell me about otters", "tell me about otters", None));
    let (_, epoch) = s.job();
    s.finish(epoch, Finished::Intercepted(Intercepted::No));
    s.take();
    s.finish(epoch, Finished::AnswerText("Otters are".into(), None));
    s.finish(epoch, Finished::AnswerSentence("Otters are mustelids.".into()));
    s.finish(epoch, Finished::AnswerText("Otters are mustelids. They".into(), None));
    s.finish(epoch, Finished::AnswerSentence("They hold hands.".into()));
    assert_eq!(
        s.take(),
        [
            responding("Otters are", false),
            Ev::Speak { id: 1, text: "Otters are mustelids.".into(), flush: true },
            responding("Otters are mustelids. They", false),
            Ev::Speak { id: 2, text: "They hold hands.".into(), flush: false },
        ]
    );
    s.speech_done(1);
    s.finish(epoch, Finished::Responded(text("Otters are mustelids. They hold hands."), false));
    // The whole answer is not said again: the last sentence is waited out.
    assert!(spoken(&s.take()).is_empty());
    s.speech_done(2);
    s.advance(3_000);
    assert_eq!(s.take().last(), Some(&idle(false)));
}

#[test]
fn an_action_is_run_by_the_frontend_and_its_cards_linger() {
    let mut s = S::new(streaming(), keep_everything());
    s.start(false, false);
    s.stt(done("set a timer for ten minutes", "set a timer for ten minutes", None));
    let (_, epoch) = s.job();
    s.finish(epoch, Finished::Intercepted(Intercepted::No));
    let (_, _, _, epoch) = s.respond_call();
    s.finish(
        epoch,
        Finished::Responded(
            EngineReply::Action { json: "{}".into(), skill_id: "dev.heyari.timer".into(), flags: ReplyFlags::default() },
            false,
        ),
    );
    let (job, epoch) = s.job();
    assert_eq!(job, Job::HandleAction { json: "{}".into(), skill_id: "dev.heyari.timer".into() });
    s.take();
    s.finish(
        epoch,
        Finished::ActionHandled(ActionOutcome {
            text: "Timer set for ten minutes.".into(),
            display_text: Some("⏱ 10:00".into()),
            cards: true,
            attachments: true,
        }),
    );
    let events = s.take();
    assert_eq!(
        events[..2],
        [
            Ev::Logged(LoggedTurn {
                clear: false,
                heard: "set a timer for ten minutes".into(),
                reply: Some(LoggedReply {
                    text: "⏱ 10:00".into(),
                    markdown: None,
                    skill_id: Some("dev.heyari.timer".into()),
                }),
            }),
            responding("Timer set for ten minutes.", true),
        ]
    );
    s.speech_done(1);
    s.advance(3_000);
    let events = s.take();
    assert_eq!(events.last(), Some(&idle(true)));
    assert_eq!(utterance(&events).and_then(|c| c.response), Some("action(dev.heyari.timer)".into()));
}

#[test]
fn facts_and_clearing_reach_the_frontend() {
    let mut s = S::new(streaming(), keep_everything());
    s.start(false, false);
    s.stt(done("clear history", "clear history", None));
    s.take();
    s.answer(flagged(
        "Done, fresh start.",
        ReplyFlags { facts_changed: true, clear_conversation: true, ..ReplyFlags::default() },
    ));
    let events = s.take();
    assert_eq!(events[0], Ev::FactsChanged);
    assert!(matches!(&events[1], Ev::Logged(t) if t.clear));
}

#[test]
fn a_card_button_said_aloud_answers_the_card() {
    let mut s = S::new(streaming(), keep_everything());
    s.start(false, false);
    s.stt(done("yes", "yes", None));
    let (_, epoch) = s.job();
    s.take();
    s.finish(epoch, Finished::Intercepted(Intercepted::CardSpoken { text: "Deleted.".into(), cards: false, attachments: false }));
    assert_eq!(
        s.take(),
        [
            Ev::Logged(LoggedTurn {
                clear: false,
                heard: "yes".into(),
                reply: Some(LoggedReply { text: "Deleted.".into(), markdown: None, skill_id: None }),
            }),
            responding("Deleted.", false),
            Ev::Speak { id: 1, text: "Deleted.".into(), flush: true },
        ]
    );
    assert!(s.io.engine.lock().unwrap().iter().all(|(c, _)| !matches!(c, EngineCall::Respond { .. })));
    s.speech_done(1);
    s.advance(3_000);
    assert_eq!(utterance(&s.take()).map(|c| c.outcome), Some(UtteranceOutcome::Card));
}

#[test]
fn stop_over_a_ringing_alarm_stops_it() {
    let mut s = S::new(streaming(), keep_everything());
    s.start(false, false);
    s.stt(done("stop", "stop", None));
    let (_, epoch) = s.job();
    s.take();
    s.finish(epoch, Finished::Intercepted(Intercepted::Alert));
    let events = s.take();
    assert_eq!(events[0], Ev::Logged(LoggedTurn { clear: false, heard: "stop".into(), reply: None }));
    assert_eq!(events.last(), Some(&idle(false)));
    assert_eq!(utterance(&events).map(|c| c.outcome), Some(UtteranceOutcome::Alert));
}

#[test]
fn a_transcription_error_shows_itself_then_closes() {
    let mut s = S::new(cloud(), keep_everything());
    s.start(true, true);
    s.take();
    s.stt(SttEvent::Error(SttError::Cloud(CloudSttFailure::Network)));
    assert_eq!(
        s.take(),
        [
            Ev::Reveal,
            Ev::Outcome(WakeOutcome::Error, false),
            Ev::State(SessionState::Error(SessionError::Stt(SttError::Cloud(CloudSttFailure::Network)))),
        ]
    );
    s.advance(2_500);
    assert_eq!(s.take().last(), Some(&idle(false)));
}

#[test]
fn a_failure_shows_itself_then_closes() {
    let mut s = S::new(streaming(), keep_everything());
    s.start(false, false);
    s.stt(done("what time is it", "what time is it", None));
    let (_, epoch) = s.job();
    s.take();
    s.finish(epoch, Finished::Failed("the card store threw".into()));
    let events = s.take();
    assert_eq!(utterance(&events).map(|c| c.outcome), Some(UtteranceOutcome::Blank));
    assert_eq!(events.last(), Some(&Ev::State(SessionState::Error(SessionError::Failed("the card store threw".into())))));
    s.advance(2_500);
    assert_eq!(s.take().last(), Some(&idle(false)));
}

#[test]
fn a_second_start_leaves_the_running_session_alone() {
    let mut s = S::new(streaming(), keep_everything());
    assert!(s.start(true, true));
    s.take();
    assert!(!s.start(true, false));
    assert_eq!(s.take(), []);
}

#[test]
fn dictation_types_what_was_said() {
    let mut s = S::new(streaming(), keep_everything());
    assert!(s.core.start_dictation(&s.io));
    assert_eq!(s.take(), [listening(""), Ev::Listen { id: 1, rewind: 2.0, give_up: None, limit: None }]);
    s.stt(partial("buy", None));
    s.stt(done("buy milk", "buy milk", None));
    let events = s.take();
    assert_eq!(events[0], listening("buy"));
    assert!(events.contains(&Ev::Dictated("buy milk".into())));
    let capture = utterance(&events).expect("the utterance");
    assert_eq!((capture.turn, capture.outcome), (UtteranceTurn::Dictation, UtteranceOutcome::Dictated));
    assert_eq!(events.last(), Some(&idle(false)));
    s.no_job();
}

#[test]
fn a_dictation_nobody_speaks_into_closes_after_thirty_seconds() {
    let mut s = S::new(streaming(), keep_everything());
    s.core.start_dictation(&s.io);
    s.take();
    s.advance(30_000);
    assert_eq!(s.take(), []);
    s.advance(500);
    assert_eq!(s.take().last(), Some(&idle(false)));
}

#[test]
fn nothing_is_kept_unless_the_user_asked() {
    let mut s = S::new(streaming(), SessionConfig::default());
    s.start(true, false);
    s.stt(done("going out", "we're going out", Some(false)));
    let events = s.take();
    assert!(!events.iter().any(|e| matches!(e, Ev::Capture(_))));
    assert!(events.contains(&Ev::FalseWake));

    // Keep-wake keeps accepted wakes only.
    let config = SessionConfig { keep_wake_audio: true, captures_dir: Some("/captures".into()), ..SessionConfig::default() };
    let mut s = S::new(streaming(), config.clone());
    s.start(true, false);
    s.stt(done("what time is it", "hey ari what time is it", Some(true)));
    assert_eq!(wake_captures(&s.take()).len(), 1);
    let mut s = S::new(streaming(), config);
    s.start(true, false);
    s.stt(done("going out", "we're going out", Some(false)));
    assert!(wake_captures(&s.take()).is_empty());
}

// ── Typed turns ─────────────────────────────────────────────────────────

fn typed(turn: u64) -> Tag {
    Tag::Typed(turn)
}

fn finished_with(turn: u64, clear: bool, reply: Option<(&str, Option<&str>)>) -> Ev {
    Ev::TypedFinished(
        turn,
        TypedFinish {
            clear,
            reply: reply.map(|(text, skill)| LoggedReply {
                text: text.into(),
                markdown: None,
                skill_id: skill.map(str::to_string),
            }),
        },
    )
}

#[test]
fn a_typed_turn_goes_through_the_intercepts_and_the_engine_silently() {
    let mut s = S::new(streaming(), keep_everything());
    s.core.submit_text(&s.io, "what time is it".into(), false);
    let (job, tag) = s.job();
    assert_eq!((job, tag), (Job::Intercept { text: "what time is it".into(), cards: true }, typed(1)));
    s.finish(tag, Finished::Intercepted(Intercepted::No));
    assert_eq!(s.take(), [Ev::TypedThinking(1)]);
    assert_eq!(s.respond_call(), ("what time is it".into(), None, true, typed(1)));
    s.finish(tag, Finished::Responded(text("It's 9:41."), false));
    // Typed, so nothing is said; the chat already shows the question.
    assert_eq!(s.take(), [finished_with(1, false, Some(("It's 9:41.", None)))]);
    // No voice session was touched.
    assert!(!s.core.is_active());
    s.advance(10_000);
    assert_eq!(s.take(), []);
}

#[test]
fn a_dictated_turn_says_one_moment_when_the_engine_is_slow_then_speaks_as_it_writes() {
    let mut s = S::new(streaming(), keep_everything());
    s.core.submit_text(&s.io, "tell me about otters".into(), true);
    let (_, tag) = s.job();
    s.finish(tag, Finished::Intercepted(Intercepted::No));
    s.take();
    s.advance(3_999);
    assert_eq!(s.take(), []);
    s.advance(1);
    assert_eq!(s.take(), [Ev::Speak { id: 1, text: "One moment.".into(), flush: false }]);
    s.finish(tag, Finished::AnswerText("Otters".into(), Some("**Otters**".into())));
    s.finish(tag, Finished::AnswerSentence("Otters are mustelids.".into()));
    s.finish(tag, Finished::Responded(text("Otters are mustelids."), false));
    assert_eq!(
        s.take(),
        [
            Ev::AnswerText(1, "Otters".into(), Some("**Otters**".into())),
            Ev::Speak { id: 2, text: "Otters are mustelids.".into(), flush: false },
            finished_with(1, false, Some(("Otters are mustelids.", None))),
        ]
    );
}

#[test]
fn a_first_sentence_in_time_means_no_one_moment() {
    let mut s = S::new(streaming(), keep_everything());
    s.core.submit_text(&s.io, "tell me about otters".into(), true);
    let (_, tag) = s.job();
    s.finish(tag, Finished::Intercepted(Intercepted::No));
    s.advance(2_000);
    s.finish(tag, Finished::AnswerSentence("Otters are mustelids.".into()));
    s.advance(5_000);
    assert!(!spoken(&s.take()).iter().any(|(_, t, _)| t == "One moment."));
}

#[test]
fn a_dictated_answer_written_all_at_once_is_read_out_after_it_is_logged() {
    let mut s = S::new(streaming(), keep_everything());
    s.core.submit_text(&s.io, "what time is it".into(), true);
    let (_, tag) = s.job();
    s.finish(tag, Finished::Intercepted(Intercepted::No));
    s.take();
    s.finish(tag, Finished::Responded(text("It's 9:41."), false));
    assert_eq!(
        s.take(),
        [finished_with(1, false, Some(("It's 9:41.", None))), Ev::Speak { id: 1, text: "It's 9:41.".into(), flush: false }]
    );
}

#[test]
fn a_typed_action_is_run_and_logged_as_the_skill_words_it() {
    let mut s = S::new(streaming(), keep_everything());
    s.core.submit_text(&s.io, "clear history".into(), false);
    let (_, tag) = s.job();
    s.finish(tag, Finished::Intercepted(Intercepted::No));
    s.take();
    s.finish(
        tag,
        Finished::Responded(
            EngineReply::Action {
                json: "{}".into(),
                skill_id: "dev.heyari.memory".into(),
                flags: ReplyFlags { facts_changed: true, clear_conversation: true, ..ReplyFlags::default() },
            },
            false,
        ),
    );
    assert_eq!(s.take(), [Ev::FactsChanged]);
    let (job, tag) = s.job();
    assert_eq!(job, Job::HandleAction { json: "{}".into(), skill_id: "dev.heyari.memory".into() });
    s.finish(
        tag,
        Finished::ActionHandled(ActionOutcome {
            text: "Done, fresh start.".into(),
            display_text: Some("Fresh start".into()),
            cards: false,
            attachments: false,
        }),
    );
    assert_eq!(s.take(), [finished_with(1, true, Some(("Fresh start", Some("dev.heyari.memory"))))]);
}

#[test]
fn a_silent_envelope_logs_no_reply_unless_its_answer_was_shown() {
    let silent = || {
        Finished::Responded(
            EngineReply::Action { json: "{}".into(), skill_id: "s".into(), flags: ReplyFlags::default() },
            false,
        )
    };
    let quiet = || {
        Finished::ActionHandled(ActionOutcome { text: String::new(), display_text: None, cards: false, attachments: false })
    };
    let mut s = S::new(streaming(), keep_everything());
    s.core.submit_text(&s.io, "ask the assistant".into(), false);
    let (_, tag) = s.job();
    s.finish(tag, Finished::Intercepted(Intercepted::No));
    s.finish(tag, silent());
    let (_, tag) = s.job();
    s.take();
    s.finish(tag, quiet());
    assert_eq!(s.take(), [finished_with(1, false, None)]);
}

#[test]
fn a_typed_word_can_answer_a_card_or_stop_an_alert() {
    let mut s = S::new(streaming(), keep_everything());
    s.core.submit_text(&s.io, "cancel".into(), true);
    let (_, tag) = s.job();
    s.finish(tag, Finished::Intercepted(Intercepted::CardSpoken { text: "Cancelled.".into(), cards: false, attachments: false }));
    assert_eq!(
        s.take(),
        [finished_with(1, false, Some(("Cancelled.", None))), Ev::Speak { id: 1, text: "Cancelled.".into(), flush: false }]
    );
    assert!(s.io.engine.lock().unwrap().is_empty());

    s.core.submit_text(&s.io, "stop".into(), false);
    let (_, tag) = s.job();
    s.finish(tag, Finished::Intercepted(Intercepted::Alert));
    assert_eq!(s.take(), [finished_with(2, false, None)]);
}

#[test]
fn a_reset_drops_the_typed_turn_in_flight() {
    let mut s = S::new(streaming(), keep_everything());
    s.core.submit_text(&s.io, "tell me about otters".into(), true);
    let (_, tag) = s.job();
    s.finish(tag, Finished::Intercepted(Intercepted::No));
    s.take();
    s.core.cancel_typed();
    s.finish(tag, Finished::AnswerText("Otters".into(), None));
    s.finish(tag, Finished::Responded(text("Otters are mustelids."), false));
    s.advance(10_000);
    assert_eq!(s.take(), []);
}

#[test]
fn a_voice_dismiss_leaves_a_typed_turn_alone() {
    let mut s = S::new(streaming(), keep_everything());
    s.core.submit_text(&s.io, "what time is it".into(), true);
    let (_, tag) = s.job();
    s.finish(tag, Finished::Intercepted(Intercepted::No));
    s.start(false, false);
    s.core.dismiss(&s.io, false);
    s.take();
    s.advance(4_000);
    s.finish(tag, Finished::Responded(text("It's 9:41."), false));
    let events = s.take();
    assert_eq!(events[0], Ev::Speak { id: 1, text: "One moment.".into(), flush: false });
    assert_eq!(events[1], finished_with(1, false, Some(("It's 9:41.", None))));
}

#[test]
fn a_blank_typed_turn_does_nothing() {
    let mut s = S::new(streaming(), keep_everything());
    s.core.submit_text(&s.io, "  ".into(), false);
    s.no_job();
    assert_eq!(s.take(), []);
}
