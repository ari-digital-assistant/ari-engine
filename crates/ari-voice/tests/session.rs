//! A voice turn through the session's own threads: the start that answers
//! only once its first state is out, a cloud listen against a local stand-in
//! for a Whisper server, the intercept and the engine on their workers, and
//! the speech the host reports finished. The scenarios in
//! `src/session/scenarios.rs` cover the decisions; this covers the plumbing.

mod common;

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ari_voice::cloud_stt::CloudSttService;
use ari_voice::pipeline::{LogLevel, VoiceHost, VoicePipeline};
use ari_voice::session::{
    ActionOutcome, AnswerSink, CaptureMode, CardButton, CardPressed, CardTarget, EngineReply, InterceptTargets,
    LoggedTurn, Phrase, ReplyFlags, SessionConfig, SessionHost, SessionState, TurnEngine,
};
use ari_voice::wake_log::{WakeLog, WakeOutcome};
use common::{asset, use_host_library, wav};

const READ: usize = 480;
const READ_MS: i64 = 30;

#[derive(Default)]
struct Host {
    cloud: Mutex<Option<CloudSttService>>,
    lines: Mutex<Vec<String>>,
}

impl VoiceHost for Host {
    fn load_model(&self, name: &str) -> Option<Vec<u8>> {
        asset(name).ok()
    }
    fn wake_detected(&self) {}
    fn log(&self, _level: LogLevel, message: &str) {
        self.lines.lock().unwrap().push(message.to_string());
    }
    fn cloud_stt(&self) -> Option<(CloudSttService, Option<String>)> {
        self.cloud.lock().unwrap().clone().map(|s| (s, None))
    }
}

#[derive(Default)]
struct Overlay {
    states: Mutex<Vec<SessionState>>,
    spoken: Mutex<Vec<(u64, String)>>,
    targets_asked: Mutex<usize>,
    logged: Mutex<Vec<LoggedTurn>>,
}

impl SessionHost for Overlay {
    fn state(&self, state: SessionState) {
        self.states.lock().unwrap().push(state);
    }
    fn reveal(&self) {}
    fn play_cue(&self, _volume: f32) {}
    fn speak(&self, id: u64, text: &str, _flush: bool) {
        self.spoken.lock().unwrap().push((id, text.into()));
    }
    fn stop_speaking(&self) {}
    fn phrase(&self, _phrase: Phrase) -> String {
        "One moment.".into()
    }
    fn capture_mode(&self, _mode: CaptureMode) {}
    fn load_speech_model(&self) -> bool {
        false
    }
    fn intercept_targets(&self) -> InterceptTargets {
        *self.targets_asked.lock().unwrap() += 1;
        let cancel = CardButton { id: "cancel".into(), label: "Cancel".into() };
        InterceptTargets { alerts: vec![], cards: vec![CardTarget { id: "timer".into(), buttons: vec![cancel] }] }
    }
    fn stop_alert(&self, _alert_id: &str) {
        unreachable!("no alert rings")
    }
    fn press_card_button(&self, _card_id: &str, _button_id: &str) -> CardPressed {
        unreachable!("the words name no button")
    }
    fn handle_action(&self, _json: &str, _skill_id: &str) -> ActionOutcome {
        unreachable!("a text reply")
    }
    fn turn_logged(&self, turn: LoggedTurn) {
        self.logged.lock().unwrap().push(turn);
    }
    fn facts_changed(&self) {}
    fn false_wake(&self) {}
    fn dictated(&self, _text: &str) {}
}

#[derive(Default)]
struct Engine {
    inputs: Mutex<Vec<(String, Option<String>)>>,
}

impl TurnEngine for Engine {
    fn respond(&self, input: &str, alternative: Option<&str>, _answer: Option<Arc<dyn AnswerSink>>) -> (EngineReply, bool) {
        self.inputs.lock().unwrap().push((input.into(), alternative.map(str::to_string)));
        (EngineReply::Text { body: "It's 9:41.".into(), markdown: None, flags: ReplyFlags::default() }, false)
    }
    fn cancel_pending_reply(&self) {}
    fn set_conversation_active(&self, _active: bool) {}
}

fn whisper_server(text: &'static str) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}/v1", listener.local_addr().unwrap());
    std::thread::spawn(move || {
        let (mut socket, _) = listener.accept().unwrap();
        let mut request = Vec::new();
        let mut buf = [0u8; 65536];
        loop {
            let n = socket.read(&mut buf).unwrap();
            request.extend_from_slice(&buf[..n]);
            let Some(head_end) = request.windows(4).position(|w| w == b"\r\n\r\n") else { continue };
            let head = String::from_utf8_lossy(&request[..head_end]).to_lowercase();
            let length: usize =
                head.lines().find_map(|l| l.strip_prefix("content-length:")).map_or(0, |v| v.trim().parse().unwrap());
            if request.len() >= head_end + 4 + length {
                break;
            }
        }
        let body = format!("{{\"text\": \"{text}\"}}");
        let reply =
            format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len());
        socket.write_all(reply.as_bytes()).unwrap();
    });
    endpoint
}

fn wait_for(what: &str, mut done: impl FnMut() -> bool) {
    let started = Instant::now();
    while !done() {
        assert!(started.elapsed() < Duration::from_secs(20), "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn a_cloud_wake_runs_through_the_session_threads() {
    use_host_library();
    let dir = tempfile::tempdir().unwrap();
    let host = Arc::new(Host::default());
    *host.cloud.lock().unwrap() = Some(CloudSttService::SelfHosted {
        endpoint: whisper_server("Hey Ari, what time is it?"),
        model: "whisper-1".into(),
    });
    let clock = Arc::new(AtomicI64::new(1_000_000));
    let ticks = Arc::clone(&clock);
    let wake_log = Arc::new(WakeLog::new(dir.path().join("wake-log.txt")));
    let pipeline = VoicePipeline::with_clock(host.clone(), Arc::clone(&wake_log), Box::new(move || ticks.load(Ordering::Relaxed)));
    let overlay = Arc::new(Overlay::default());
    let engine = Arc::new(Engine::default());
    pipeline.set_cloud_mode(true);
    pipeline.attach_session_host(overlay.clone());
    pipeline.configure_session(SessionConfig::default());

    let audio = wav("oww_parity.wav");
    let (preroll, live) = audio.split_at(32_000);
    for read in preroll.chunks(READ) {
        pipeline.push_audio(read);
    }

    // The first state is out by the time the start returns.
    assert!(pipeline.start_turn(true, false));
    assert_eq!(overlay.states.lock().unwrap()[0], SessionState::Listening { partial: String::new() });
    assert!(pipeline.is_session_active());
    assert!(!pipeline.start_turn(true, false));

    // The engine arrives after the turn has begun, as it may on a cold
    // process; the turn waits for it.
    let silence = vec![0i16; READ];
    let mut reads = live.chunks(READ).chain(std::iter::repeat(&silence[..]));
    let mut attached = false;
    wait_for("the answer to be spoken", || {
        if !attached && *overlay.targets_asked.lock().unwrap() > 0 {
            pipeline.attach_engine(engine.clone());
            attached = true;
        }
        clock.fetch_add(READ_MS, Ordering::Relaxed);
        pipeline.push_audio(reads.next().unwrap());
        std::thread::sleep(Duration::from_millis(READ_MS as u64));
        !overlay.spoken.lock().unwrap().is_empty()
    });

    assert_eq!(*overlay.targets_asked.lock().unwrap(), 1);
    assert_eq!(*engine.inputs.lock().unwrap(), [("what time is it?".to_string(), None)]);
    assert_eq!(overlay.logged.lock().unwrap()[0].heard, "what time is it?");
    let states = overlay.states.lock().unwrap().clone();
    assert!(states.contains(&SessionState::Thinking), "{states:?}");
    assert_eq!(states.last(), Some(&SessionState::Responding { text: "It's 9:41.".into(), cards: false }));

    let (id, text) = overlay.spoken.lock().unwrap()[0].clone();
    assert_eq!(text, "It's 9:41.");
    pipeline.speech_finished(id);
    clock.fetch_add(3_000, Ordering::Relaxed);
    wait_for("the overlay to close", || {
        overlay.states.lock().unwrap().last() == Some(&SessionState::Idle { keep_cards: false })
    });
    assert!(!pipeline.is_session_active());

    wait_for("the wake's ending in the log", || {
        wake_log.entries().iter().any(|e| e.kind == "outcome" && e.args.first().map(String::as_str) == Some(WakeOutcome::Accepted.slug()))
    });
}
