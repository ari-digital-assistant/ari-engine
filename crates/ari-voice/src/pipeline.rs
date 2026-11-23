//! The voice pipeline's audio side: every chunk the microphone produces comes
//! in through [`VoicePipeline::push_audio`], on the frontend's capture thread,
//! and the wake word is detected, debounced and speech-checked there, as
//! `WakeWordService.captureLoop` did.
//!
//! The frontend opens the microphone, keeps the settings and acts on a wake;
//! this decides what counts as one.

use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};
use std::time::Instant;

use crate::detector::{SILERO_VAD_FILE, WakeDetector};
use crate::cloud_stt::CloudSttService;
use crate::ring::{AudioRing, RING_CAPACITY_SAMPLES};
use crate::speech_check::{SPEECH_CHECK_BUFFER_SAMPLES, SPEECH_CHECK_TAIL_SAMPLES, SpeechCheckPacing, SpeechGate};
use crate::session::runner::{EngineSlot, SessionRunner};
use crate::stt::{Stt, SttEvent};
use crate::wake_log::{ALIVE_INTERVAL_MS, WakeLog, model_fingerprint};
use crate::wake_model::{WakeSensitivity, WakeWordModel};

/// Detections closer together than this are one wake.
const DETECTION_DEBOUNCE_MS: i64 = 4_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Debug,
    Info,
    Warn,
}

/// What the pipeline needs from the frontend it runs in.
pub trait VoiceHost: Send + Sync {
    /// A model file by name (`hey_ari_oww.onnx`, `silero_vad.onnx`), from
    /// wherever the frontend keeps them. `None` when it cannot be read.
    fn load_model(&self, name: &str) -> Option<Vec<u8>>;

    /// The wake word was heard and passed the debounce and the speech check.
    /// Called on the capture thread, inside `push_audio`: hand it on rather
    /// than doing the work here, and never call back into the pipeline's
    /// audio path from it.
    fn wake_detected(&self);

    /// A line for the frontend's own log. Nothing is stored.
    fn log(&self, level: LogLevel, message: &str) {
        let _ = (level, message);
    }

    /// The user's language, as the engine's locale codes go (`en`, `it`):
    /// for the wake phrase rules and Whisper.
    fn locale(&self) -> String {
        "en".into()
    }

    /// What a listen has to say. Called on the listen's thread; hand it on.
    fn stt_event(&self, event: SttEvent) {
        let _ = event;
    }

    /// The cloud transcription service the user set up, and its key, asked
    /// for at each transcription so a settings change needs no push. `None`
    /// when there is none.
    fn cloud_stt(&self) -> Option<(CloudSttService, Option<String>)> {
        None
    }

    /// Called first on every thread the pipeline starts that calls the host:
    /// listens, the voice session and its workers. A JVM host keeps the
    /// thread attached for its life here. Otherwise JNA attaches it for each
    /// callback and detaches it after, and a callback that calls back into
    /// the pipeline, which then calls the host again, has the thread
    /// detached under it when the inner call returns: ART aborts the process.
    fn thread_started(&self) {}
}

/// How to listen for the wake word, as the frontend's settings say.
#[derive(Debug, Clone)]
pub struct WakeConfig {
    /// With the device's tuned ladder already applied.
    pub model: WakeWordModel,
    pub sensitivity: WakeSensitivity,
    /// Drop a detection with no speech in the second before it.
    pub speech_check: bool,
    /// Written to the wake log's listen-on line; the session acts on it.
    pub quiet_wake: bool,
}

/// One listening spell's wake state: built by `configure_wake`, live from
/// `start_wake` to `stop_wake`.
struct Wake {
    config: WakeConfig,
    detector: WakeDetector,
    fingerprint: String,
    started: bool,
    alive_logged_at: i64,
    // Per spell like the detector: the speech model is not thread-safe, and
    // it is built at the first detection that needs it, since loading it per
    // wake would cost more than the check.
    pacing: SpeechCheckPacing,
    gate: Option<SpeechGate>,
    gate_unavailable: bool,
}

struct Capture {
    ring: AudioRing,
    wake: Option<Wake>,
    /// The armed listen feed. Unbounded: the capture thread never blocks on a
    /// slow decode, and no audio is ever dropped for one. Dropping it under
    /// load was causing intermittent transcription corruption ("what time is
    /// it" became "wheat" or "okay").
    feed: Option<Sender<Vec<i16>>>,
}

/// Pre-roll goes to a feed as 100 ms slices, not one blob. sherpa is a
/// streaming zipformer and gets confused by 2 s in a single acceptWaveform
/// call: the encoder rhythm breaks and transcription truncates ("tell me some
/// wisdom" became "tell"). Slices make the pre-roll look like live audio.
pub const PREROLL_SLICE_SAMPLES: usize = 1_600;

/// The one voice pipeline in a process: one microphone, one pipeline.
pub struct VoicePipeline {
    host: Arc<dyn VoiceHost>,
    wake_log: Arc<WakeLog>,
    capture: Mutex<Capture>,
    paused: AtomicBool,
    last_detection_at: AtomicI64,
    clock: Box<dyn Fn() -> i64 + Send + Sync>,
    pub(crate) stt: Stt,
    session: Mutex<Option<SessionRunner>>,
    pub(crate) engine: Arc<EngineSlot>,
}

static LIVE: OnceLock<Mutex<Weak<VoicePipeline>>> = OnceLock::new();

fn live() -> MutexGuard<'static, Weak<VoicePipeline>> {
    LIVE.get_or_init(|| Mutex::new(Weak::new())).lock().unwrap_or_else(|e| e.into_inner())
}

/// Hand `samples` to the live pipeline, if there is one. What
/// `ari_voice_push_audio` calls, so no pointer to the pipeline crosses the C
/// boundary.
pub fn push_to_live(samples: &[i16]) {
    let pipeline = live().upgrade();
    if let Some(pipeline) = pipeline {
        pipeline.push_audio(samples);
    }
}

impl VoicePipeline {
    /// Builds the pipeline and makes it the live one, replacing any other.
    pub fn new(host: Arc<dyn VoiceHost>, wake_log: Arc<WakeLog>) -> Arc<Self> {
        let epoch = Instant::now();
        Self::with_clock(host, wake_log, Box::new(move || epoch.elapsed().as_millis() as i64))
    }

    /// As [`new`](Self::new), reading the time in milliseconds from `clock`,
    /// which only ever has to move forward. For tests that drive time with
    /// the audio they push.
    pub fn with_clock(
        host: Arc<dyn VoiceHost>,
        wake_log: Arc<WakeLog>,
        clock: Box<dyn Fn() -> i64 + Send + Sync>,
    ) -> Arc<Self> {
        let pipeline = Arc::new(Self {
            host,
            wake_log,
            capture: Mutex::new(Capture { ring: AudioRing::new(RING_CAPACITY_SAMPLES), wake: None, feed: None }),
            paused: AtomicBool::new(false),
            last_detection_at: AtomicI64::new(i64::MIN / 2),
            clock,
            stt: Stt::default(),
            session: Mutex::new(None),
            engine: Arc::default(),
        });
        *live() = Arc::downgrade(&pipeline);
        pipeline
    }

    pub fn wake_log(&self) -> &Arc<WakeLog> {
        &self.wake_log
    }

    fn now(&self) -> i64 {
        (self.clock)()
    }

    pub(crate) fn now_ms(&self) -> i64 {
        self.now()
    }

    pub(crate) fn host(&self) -> &Arc<dyn VoiceHost> {
        &self.host
    }

    pub(crate) fn session(&self) -> MutexGuard<'_, Option<SessionRunner>> {
        self.session.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn capture(&self) -> MutexGuard<'_, Capture> {
        self.capture.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn load(&self, name: &str) -> Result<Vec<u8>, String> {
        self.host.load_model(name).ok_or_else(|| format!("could not read {name}"))
    }

    /// Build the wake engine for `config`, before the microphone opens. It
    /// does nothing until [`start_wake`](Self::start_wake). Replaces any
    /// spell already configured, stopping it first if it was running.
    pub fn configure_wake(&self, config: WakeConfig) -> Result<(), String> {
        let point = config.model.operating_point(config.sensitivity);
        self.host.log(
            LogLevel::Info,
            &format!(
                "Loading wake word model: {} @ sensitivity={} (cutoff={}, window={})",
                config.model.id,
                config.sensitivity.name(),
                point.probability_cutoff,
                point.sliding_window_size
            ),
        );
        let model_bytes = self.load(&config.model.asset_filename)?;
        // The speech check needs to decline a detection without the engine's
        // cool-off swallowing the phrase that follows it. Every path below
        // that acts on a detection resets the engine itself. Off, the engine
        // behaves exactly as it always has.
        let detector =
            WakeDetector::from_bytes(&config.model, point, !config.speech_check, &model_bytes, &|n| self.load(n))?;
        let wake = Wake {
            fingerprint: model_fingerprint(&model_bytes),
            config,
            detector,
            started: false,
            alive_logged_at: 0,
            pacing: SpeechCheckPacing::default(),
            gate: None,
            gate_unavailable: false,
        };
        let mut capture = self.capture();
        if capture.wake.as_ref().is_some_and(|w| w.started) {
            self.wake_log.listening_stopped();
        }
        capture.wake = Some(wake);
        Ok(())
    }

    /// The microphone is open: start detecting with the configured engine.
    pub fn start_wake(&self) {
        let now = self.now();
        let mut capture = self.capture();
        let Some(wake) = capture.wake.as_mut() else { return };
        if wake.started {
            return;
        }
        wake.started = true;
        wake.alive_logged_at = now;
        let point = wake.config.model.operating_point(wake.config.sensitivity);
        self.wake_log.listening_started(
            &wake.config.model,
            wake.config.sensitivity,
            point,
            wake.config.speech_check,
            wake.config.quiet_wake,
            &wake.fingerprint,
        );
    }

    /// The microphone is closing: drop the engine. Safe to call when nothing
    /// was configured or started.
    pub fn stop_wake(&self) {
        let mut capture = self.capture();
        if capture.wake.take().is_some_and(|w| w.started) {
            self.wake_log.listening_stopped();
        }
    }

    /// Mid-conversation: audio still fills the ring, but nothing is detected.
    pub fn set_wake_paused(&self, paused: bool) {
        self.paused.store(paused, Ordering::Relaxed);
    }

    /// Open a feed: the last `rewind_seconds` of the ring as 100 ms slices,
    /// then every chunk the microphone delivers, until [`disarm`](Self::disarm).
    /// `None` while another feed is armed: one consumer at a time. While a
    /// feed is armed the wake word never wakes.
    pub fn arm(&self, rewind_seconds: f32) -> Option<Receiver<Vec<i16>>> {
        let mut capture = self.capture();
        if capture.feed.is_some() {
            self.host.log(LogLevel::Warn, "arm() called while already armed — ignoring");
            return None;
        }
        let head = capture.ring.samples_written();
        let rewind = (16_000.0 * rewind_seconds) as u64;
        let from = head.saturating_sub(rewind);
        let preroll = capture.ring.snapshot(from, head);
        let (sender, receiver) = mpsc::channel();
        let mut slices = 0;
        for slice in preroll.chunks(PREROLL_SLICE_SAMPLES) {
            // The receiver is still in hand, so this cannot fail.
            let _ = sender.send(slice.to_vec());
            slices += 1;
        }
        capture.feed = Some(sender);
        self.host.log(
            LogLevel::Info,
            &format!("armed: head={head} from={from} prerollSamples={} slices={slices}", preroll.len()),
        );
        Some(receiver)
    }

    /// Close the feed, which ends its consumer's receive. Idempotent.
    pub fn disarm(&self) {
        if self.capture().feed.take().is_some() {
            self.host.log(LogLevel::Info, "disarmed");
        }
    }

    pub fn is_armed(&self) -> bool {
        self.capture().feed.is_some()
    }

    /// The last `seconds` of audio, without arming: kept while a wake turn
    /// plays out, so a wake nobody answers can be kept as a hard negative.
    pub fn peek_recent(&self, seconds: f32) -> Vec<i16> {
        self.capture().ring.recent((16_000.0 * seconds) as usize)
    }

    /// One chunk of 16 kHz mono audio from the microphone.
    pub fn push_audio(&self, samples: &[i16]) {
        // The host hears about a wake once the capture state is let go, so it
        // can call back in without waiting on itself.
        if !samples.is_empty() && self.detect(samples) {
            self.host.wake_detected();
        }
    }

    /// Fills the ring and runs the wake word over `samples`: true for a wake.
    fn detect(&self, samples: &[i16]) -> bool {
        let now = self.now();
        let mut capture = self.capture();
        let capture = &mut *capture;
        capture.ring.write(samples);
        if let Some(feed) = &capture.feed
            && feed.send(samples.to_vec()).is_err()
        {
            // The consumer went away without disarming.
            capture.feed = None;
        }
        let armed = capture.feed.is_some();
        let Some(wake) = capture.wake.as_mut().filter(|w| w.started) else { return false };

        if now - wake.alive_logged_at >= ALIVE_INTERVAL_MS {
            wake.alive_logged_at = now;
            self.wake_log.still_listening();
        }
        if self.paused.load(Ordering::Relaxed) {
            return false;
        }
        if !wake.detector.process_audio(samples) {
            return false;
        }
        // Belt-and-braces: don't fire wake while a listen is armed. The
        // debounce below covers the common case, but the mic is permanently
        // open, so an in-utterance fire is possible.
        if armed {
            wake.detector.reset();
            return false;
        }
        if now - self.last_detection_at.load(Ordering::Relaxed) < DETECTION_DEBOUNCE_MS {
            self.host.log(LogLevel::Debug, "Wake word detected within debounce window — ignoring");
            wake.detector.reset();
            return false;
        }
        // No reset when this declines: the engine reports again while it still
        // hears the wake word, and the check runs again once speech arrives.
        if wake.config.speech_check && !self.speech_before_wake(wake, &capture.ring, now) {
            return false;
        }
        self.last_detection_at.store(now, Ordering::Relaxed);
        self.host.log(LogLevel::Info, "Wake word detected!");
        self.wake_log.wake(&wake.config.model.id);
        wake.detector.reset();
        true
    }

    /// Whether this detection may open Ari: speech in the second before it,
    /// on the rule measured offline. A model that cannot load fails open,
    /// because a check that cannot run must never cost somebody their wake.
    fn speech_before_wake(&self, wake: &mut Wake, ring: &AudioRing, now: i64) -> bool {
        if !wake.pacing.should_check(now) {
            return false;
        }
        let model_id = wake.config.model.id.clone();
        let Some(speech) = self.speech_now(wake, ring) else {
            let reason = if wake.gate_unavailable { "unavailable" } else { "error" };
            self.wake_log.speech_check_failed(&model_id, reason);
            return true;
        };
        if speech {
            if wake.pacing.passed() {
                self.wake_log.passed_after_drop(&model_id);
            }
            return true;
        }
        if wake.pacing.dropped(now) {
            self.host.log(LogLevel::Info, "Wake dropped: no speech in the second before it");
            self.wake_log.dropped(&model_id);
        }
        false
    }

    /// Whether there was speech in the second before now, or `None` when the
    /// check cannot run.
    fn speech_now(&self, wake: &mut Wake, ring: &AudioRing) -> Option<bool> {
        if wake.gate.is_none() && !wake.gate_unavailable {
            match self.load(SILERO_VAD_FILE).and_then(|model| SpeechGate::new(&model)) {
                Ok(gate) => wake.gate = Some(gate),
                Err(e) => {
                    self.host.log(LogLevel::Warn, &format!("Speech check unavailable, letting wakes through: {e}"));
                    wake.gate_unavailable = true;
                }
            }
        }
        let gate = wake.gate.as_mut()?;
        match gate.speech_in_tail(&ring.recent(SPEECH_CHECK_BUFFER_SAMPLES), SPEECH_CHECK_TAIL_SAMPLES) {
            Ok(speech) => Some(speech),
            Err(e) => {
                self.host.log(LogLevel::Warn, &format!("Speech check failed, letting the wake through: {e}"));
                None
            }
        }
    }
}
