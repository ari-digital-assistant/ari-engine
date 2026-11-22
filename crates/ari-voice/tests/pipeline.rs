//! The capture loop's rules through the real engines: the wake word, the
//! debounce, the pause, the armed listen and the speech check, on the parity
//! fixture's three synthetic "Hey Ari"s. Needs `native/voice/build-host.sh`.

mod common;

use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ari_voice::pipeline::{LogLevel, VoiceHost, VoicePipeline, WakeConfig};
use ari_voice::wake_log::{WakeLog, WakeLogEntry, model_fingerprint};
use ari_voice::wake_model::{WakeSensitivity, by_id};
use common::{asset, use_host_library, wav};

/// The capture loop's read: 30 ms.
const READ: usize = 480;
const READ_MS: i64 = 30;

#[derive(Default)]
struct Host {
    wakes: AtomicUsize,
    missing: Vec<&'static str>,
    lines: Mutex<Vec<String>>,
}

impl VoiceHost for Host {
    fn load_model(&self, name: &str) -> Option<Vec<u8>> {
        if self.missing.contains(&name) { None } else { asset(name).ok() }
    }
    fn wake_detected(&self) {
        self.wakes.fetch_add(1, Ordering::Relaxed);
    }
    fn log(&self, _level: LogLevel, message: &str) {
        self.lines.lock().unwrap().push(message.to_string());
    }
}

struct Run {
    host: Arc<Host>,
    pipeline: Arc<VoicePipeline>,
    clock: Arc<AtomicI64>,
    log: Arc<WakeLog>,
    _dir: tempfile::TempDir,
}

fn run(host: Host, speech_check: bool) -> Run {
    use_host_library();
    let dir = tempfile::tempdir().unwrap();
    let log = Arc::new(WakeLog::new(dir.path().join("wake-log.txt")));
    let host = Arc::new(host);
    let clock = Arc::new(AtomicI64::new(1_000_000));
    let ticks = Arc::clone(&clock);
    let pipeline = VoicePipeline::with_clock(
        host.clone(),
        Arc::clone(&log),
        Box::new(move || ticks.load(Ordering::Relaxed)),
    );
    pipeline
        .configure_wake(WakeConfig {
            model: by_id(Some("hey_ari_oww")),
            sensitivity: WakeSensitivity::Medium,
            speech_check,
            quiet_wake: true,
        })
        .unwrap();
    pipeline.start_wake();
    Run { host, pipeline, clock, log, _dir: dir }
}

impl Run {
    /// The fixture in real time: the clock moves 30 ms with each read.
    fn push_fixture(&self) {
        for read in wav("oww_parity.wav").chunks(READ) {
            self.clock.fetch_add(READ_MS, Ordering::Relaxed);
            self.pipeline.push_audio(read);
        }
    }

    fn wakes(&self) -> usize {
        self.host.wakes.load(Ordering::Relaxed)
    }

    /// The log once its writer has caught up with `expected` lines.
    fn log_lines(&self, expected: usize) -> Vec<WakeLogEntry> {
        for _ in 0..400 {
            let entries = self.log.entries();
            if entries.len() >= expected {
                return entries;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        self.log.entries()
    }
}

#[test]
fn each_synthetic_hey_ari_is_one_wake_and_one_log_line() {
    let run = run(Host::default(), false);
    run.push_fixture();
    let wakes = run.wakes();
    assert!(wakes >= 2, "{wakes} wakes");

    run.pipeline.stop_wake();
    let lines = run.log_lines(wakes + 2);
    let kinds: Vec<&str> = lines.iter().map(|e| e.kind.as_str()).collect();
    let mut expected = vec!["listen-on"];
    expected.extend(std::iter::repeat_n("wake", wakes));
    expected.push("listen-off");
    assert_eq!(expected, kinds);
    let fingerprint = model_fingerprint(&asset("hey_ari_oww.onnx").unwrap());
    assert_eq!(
        vec!["hey_ari_oww", "MEDIUM", "0.55", "1", "speech-check=off", "quiet-wake=on", &format!("file={fingerprint}")],
        lines[0].args
    );
}

#[test]
fn the_speech_check_lets_spoken_wakes_through() {
    let run = run(Host::default(), true);
    run.push_fixture();
    assert!(run.wakes() >= 2, "{} wakes", run.wakes());
    std::thread::sleep(Duration::from_millis(100));
    assert!(!run.log.entries().iter().any(|e| e.kind == "speech-check-failed"));
}

#[test]
fn a_speech_check_that_cannot_load_lets_wakes_through_and_says_so() {
    let run = run(Host { missing: vec!["silero_vad.onnx"], ..Host::default() }, true);
    run.push_fixture();
    let wakes = run.wakes();
    assert!(wakes >= 2, "{wakes} wakes");
    let lines = run.log_lines(1 + 2 * wakes);
    let failed: Vec<&Vec<String>> =
        lines.iter().filter(|e| e.kind == "speech-check-failed").map(|e| &e.args).collect();
    assert_eq!(wakes, failed.len());
    assert!(failed.iter().all(|args| args == &&vec!["hey_ari_oww".to_string(), "unavailable".into()]));
    assert!(run.host.lines.lock().unwrap().iter().any(|l| l.starts_with("Speech check unavailable")));
}

#[test]
fn nothing_wakes_while_paused_or_while_a_listen_is_armed() {
    let paused = run(Host::default(), false);
    paused.pipeline.set_wake_paused(true);
    paused.push_fixture();
    assert_eq!(0, paused.wakes());

    let armed = run(Host::default(), false);
    armed.pipeline.set_listen_armed(true);
    armed.push_fixture();
    assert_eq!(0, armed.wakes());
}

#[test]
fn wakes_inside_the_debounce_are_one_wake() {
    let run = run(Host::default(), false);
    let audio = wav("oww_parity.wav");
    // The whole fixture with the clock standing still: one wake, however many
    // detections.
    for read in audio.chunks(READ) {
        run.pipeline.push_audio(read);
    }
    assert_eq!(1, run.wakes());
}

#[test]
fn a_stopped_pipeline_hears_nothing() {
    let run = run(Host::default(), false);
    run.pipeline.stop_wake();
    run.push_fixture();
    assert_eq!(0, run.wakes());
}
