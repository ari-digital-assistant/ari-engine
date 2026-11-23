//! Listening through the pipeline: the errors, the buffered path end to end
//! against a local stand-in for a Whisper server, and (ignored, needing the
//! model) Kroko streaming and the replay of the phone's utterance captures.

mod common;

use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ari_voice::cloud_stt::CloudSttService;
use ari_voice::pipeline::{LogLevel, VoiceHost, VoicePipeline};
use ari_voice::stt::{ListenParams, SttError, SttEvent, SttModelSpec};
use ari_voice::wake_log::WakeLog;
use common::{asset, use_host_library, wav, workspace};

const READ: usize = 480;
const READ_MS: i64 = 30;

#[derive(Default)]
struct Host {
    events: Mutex<Vec<SttEvent>>,
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
    fn stt_event(&self, event: SttEvent) {
        self.events.lock().unwrap().push(event);
    }
    fn cloud_stt(&self) -> Option<(CloudSttService, Option<String>)> {
        self.cloud.lock().unwrap().clone().map(|s| (s, None))
    }
}

struct Run {
    host: Arc<Host>,
    pipeline: Arc<VoicePipeline>,
    clock: Arc<AtomicI64>,
    _dir: tempfile::TempDir,
}

fn pipeline() -> Run {
    use_host_library();
    ari_voice::sherpa::set_library_path(workspace().join("target/voice-native/libsherpa-onnx-c-api.so"));
    let dir = tempfile::tempdir().unwrap();
    let host = Arc::new(Host::default());
    let clock = Arc::new(AtomicI64::new(1_000_000));
    let ticks = Arc::clone(&clock);
    let pipeline = VoicePipeline::with_clock(
        host.clone(),
        Arc::new(WakeLog::new(dir.path().join("wake-log.txt"))),
        Box::new(move || ticks.load(Ordering::Relaxed)),
    );
    Run { host, pipeline, clock, _dir: dir }
}

impl Run {
    fn events(&self) -> Vec<SttEvent> {
        self.host.events.lock().unwrap().clone()
    }

    fn finished(&self) -> Option<SttEvent> {
        self.events().into_iter().find(|e| {
            matches!(e, SttEvent::Done { .. } | SttEvent::NoSpeech { .. } | SttEvent::Error(_))
        })
    }

    /// `audio` as the phone delivers it: its first 2 s in the ring before the
    /// listen arms, the rest in 30 ms reads with the clock moving, then
    /// silence until the listen ends. In real time: the listen reads the clock
    /// as it processes each chunk, so a feed that ran ahead of it would make
    /// time jump and end listens early, which the phone never sees.
    fn listen_to(&self, audio: &[i16], params: ListenParams) -> SttEvent {
        let (preroll, live) = audio.split_at(audio.len().min(32_000));
        for read in preroll.chunks(READ) {
            self.pipeline.push_audio(read);
        }
        self.pipeline.start_listen(params);
        let silence = vec![0i16; READ];
        let mut reads = live.chunks(READ).chain(std::iter::repeat(&silence[..]));
        for _ in 0..4_000 {
            if let Some(end) = self.finished() {
                return end;
            }
            self.clock.fetch_add(READ_MS, Ordering::Relaxed);
            self.pipeline.push_audio(reads.next().unwrap());
            std::thread::sleep(Duration::from_millis(READ_MS as u64));
        }
        for _ in 0..2_000 {
            if let Some(end) = self.finished() {
                return end;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        panic!("the listen never ended: {:?}", self.host.lines.lock().unwrap());
    }
}

#[test]
fn a_listen_with_no_model_and_no_cloud_says_so() {
    let run = pipeline();
    run.pipeline.start_listen(ListenParams::default());
    assert_eq!(vec![SttEvent::Error(SttError::NoModel)], run.events());
}

#[test]
fn a_listen_while_the_feed_is_taken_says_it_is_busy() {
    let run = pipeline();
    run.pipeline.set_cloud_mode(true);
    let _held = run.pipeline.arm(0.0).unwrap();
    run.pipeline.start_listen(ListenParams::default());
    assert_eq!(vec![SttEvent::Error(SttError::Busy)], run.events());
}

#[test]
fn a_new_listen_replaces_the_pipeline_s_own_unfinished_one() {
    // The session starts the next listen as soon as it hears the last one's
    // final event, which can be before that listen has let go of the feed.
    let run = pipeline();
    run.pipeline.set_cloud_mode(true);
    let first = run.pipeline.start_listen(ListenParams::default());
    assert!(run.pipeline.is_armed());
    let second = run.pipeline.start_listen(ListenParams { rewind_seconds: 0.0, ..ListenParams::default() });
    assert_ne!(first, second);
    assert!(!run.events().contains(&SttEvent::Error(SttError::Busy)), "{:?}", run.events());
    assert!(run.pipeline.is_armed());
    run.pipeline.stop_listen();
}

/// A one-request stand-in for a Whisper server: answers `text`, and hands
/// back the body it was sent.
fn whisper_server(text: &'static str) -> (String, std::thread::JoinHandle<Vec<u8>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}/v1", listener.local_addr().unwrap());
    let handle = std::thread::spawn(move || {
        let (mut socket, _) = listener.accept().unwrap();
        let mut request = Vec::new();
        let mut buf = [0u8; 65536];
        loop {
            let n = socket.read(&mut buf).unwrap();
            request.extend_from_slice(&buf[..n]);
            let Some(head_end) = request.windows(4).position(|w| w == b"\r\n\r\n") else { continue };
            let head = String::from_utf8_lossy(&request[..head_end]).to_lowercase();
            let length: usize = head
                .lines()
                .find_map(|l| l.strip_prefix("content-length:"))
                .map_or(0, |v| v.trim().parse().unwrap());
            if request.len() >= head_end + 4 + length {
                break;
            }
        }
        let body = format!("{{\"text\": \"{text}\"}}");
        let reply = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        socket.write_all(reply.as_bytes()).unwrap();
        request
    });
    (endpoint, handle)
}

#[test]
fn a_cloud_listen_ends_on_silence_and_transcribes_the_whole_utterance() {
    let run = pipeline();
    let (endpoint, server) = whisper_server("Hey Ari, what time is it?");
    *run.host.cloud.lock().unwrap() = Some(CloudSttService::SelfHosted { endpoint, model: "whisper-1".into() });
    run.pipeline.set_cloud_mode(true);

    let audio = wav("oww_parity.wav");
    let end = run.listen_to(&audio, ListenParams::default());

    let SttEvent::Done { text, raw, name_matched, audio: heard, listen_end, parallel, speech_heard } = end else {
        panic!("{end:?}");
    };
    assert_eq!("what time is it?", text);
    assert_eq!(Some(true), speech_heard);
    assert_eq!(Some("Hey Ari, what time is it?".into()), raw);
    assert_eq!(Some(true), name_matched);
    assert_eq!(None, parallel);
    let listen_end = listen_end.unwrap();
    assert_eq!("endpoint", listen_end.end.slug());
    // What was heard is the fixture from its start, pre-roll included, up to
    // the pause after its first sentence, where the endpoint ended it.
    let heard = heard.unwrap();
    assert!(heard.len() > 48_000 && heard.len() < audio.len(), "{}", heard.len());
    assert_eq!(audio[..heard.len()], heard[..]);
    // And sent: the request's body holds the WAV.
    let request = server.join().unwrap();
    assert!(request.windows(4).any(|w| w == b"RIFF"));
    let events = run.events();
    assert_eq!(SttEvent::Listening { partial: String::new(), name_matched: Some(false) }, events[0]);
    assert!(events.contains(&SttEvent::Transcribing));
    assert!(!run.pipeline.is_armed(), "a listen that ends disarms the feed");
}

#[test]
fn a_quiet_wake_that_hears_nothing_gives_up_untranscribed() {
    let run = pipeline();
    run.pipeline.set_cloud_mode(true);
    let params = ListenParams { rewind_seconds: 2.0, give_up_without_speech_ms: Some(3_000), limit_ms: Some(6_000) };
    let end = run.listen_to(&vec![0i16; 32_000], params);
    let SttEvent::NoSpeech { listen_end, .. } = end else { panic!("{end:?}") };
    assert_eq!("give-up", listen_end.unwrap().end.slug());
}

fn kroko() -> SttModelSpec {
    let models = std::env::var_os("ARI_STT_MODELS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/ari/stt"));
    SttModelSpec {
        id: "kroko-2025-08-06".into(),
        model_type: "zipformer2".into(),
        dir: models.join("kroko-2025-08-06"),
        encoder_file: "encoder.onnx".into(),
        decoder_file: "decoder.onnx".into(),
        joiner_file: Some("joiner.onnx".into()),
        tokens_file: "tokens.txt".into(),
    }
}

#[test]
#[ignore = "needs the Kroko model"]
fn kroko_streams_a_listen_with_partials_and_a_parallel_opinion() {
    let run = pipeline();
    run.pipeline.load_stt(&kroko()).unwrap();
    assert!(run.pipeline.stt_streaming());
    let end = run.listen_to(&wav("oww_parity.wav"), ListenParams::default());
    let SttEvent::Done { raw, audio, speech_heard, .. } = &end else { panic!("{end:?}") };
    println!("Kroko listen: {end:?}");
    assert!(raw.as_deref().unwrap().contains("Bad news travels fast"));
    assert_eq!(Some(true), *speech_heard);
    let partials = run.events().iter().filter(|e| matches!(e, SttEvent::Listening { .. })).count();
    assert!(partials > 5, "{partials} partials");
    // The retry decodes the same audio again, whole.
    println!("Retry: {:?}", run.pipeline.transcribe_again(audio.as_ref().unwrap()));
}

/// Every English utterance capture on this laptop, replayed through Kroko as
/// the phone chunked it, against what its sidecar says the phone heard. Not
/// pass/fail: many captures were transcribed by OpenAI, not Kroko, so every
/// difference is printed to be read.
#[test]
#[ignore = "needs the Kroko model and ari-tools' captures"]
fn replay_the_utterance_captures() {
    let captures = std::env::var_os("ARI_CAPTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace().join("../ari-tools/wakeword/captures"));
    let mut clips: Vec<PathBuf> = Vec::new();
    collect(&captures, &mut clips);
    clips.sort_by_key(|p| p.file_name().unwrap().to_owned());
    clips.dedup_by_key(|p| p.file_name().unwrap().to_owned());

    let run = pipeline();
    run.pipeline.load_stt(&kroko()).unwrap();
    let (mut same, mut different) = (0, 0);
    for clip in &clips {
        let sidecar = std::fs::read_to_string(clip.with_extension("txt")).unwrap_or_default();
        let field = |name: &str| {
            sidecar.lines().find_map(|l| l.strip_prefix(&format!("{name}: "))).unwrap_or("").to_string()
        };
        if field("locale") != "en" || field("raw").is_empty() || field("raw") == "(none)" {
            continue;
        }
        run.host.events.lock().unwrap().clear();
        let audio = read_wav(clip);
        let end = run.listen_to(&audio, ListenParams::default());
        let replay = match &end {
            SttEvent::Done { raw, .. } => raw.clone().unwrap_or_default(),
            other => format!("{other:?}"),
        };
        if replay == field("raw") {
            same += 1;
        } else {
            different += 1;
            println!(
                "{} (model {}):\n  phone:  {}\n  replay: {}",
                clip.file_name().unwrap().to_string_lossy(),
                field("model"),
                field("raw"),
                replay
            );
        }
    }
    println!("{same} identical, {different} different, of {} clips", clips.len());
}

fn collect(dir: &Path, into: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect(&path, into);
        } else if path.extension().is_some_and(|e| e == "wav")
            && path.to_string_lossy().contains("utterance-captures")
        {
            into.push(path);
        }
    }
}

fn read_wav(path: &Path) -> Vec<i16> {
    let bytes = std::fs::read(path).unwrap();
    let mut offset = 12;
    while offset + 8 <= bytes.len() {
        let size = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap()) as usize;
        if &bytes[offset..offset + 4] == b"data" {
            let end = (offset + 8 + size).min(bytes.len());
            return bytes[offset + 8..end].as_chunks::<2>().0.iter().map(|b| i16::from_le_bytes(*b)).collect();
        }
        offset += 8 + size + (size & 1);
    }
    Vec::new()
}
