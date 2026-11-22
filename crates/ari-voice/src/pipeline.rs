//! The voice pipeline's audio side: every chunk the microphone produces comes
//! in through [`VoicePipeline::push_audio`], on the frontend's capture thread,
//! and the wake word is detected, debounced and speech-checked there, as
//! `WakeWordService.captureLoop` did.
//!
//! The frontend opens the microphone, keeps the settings and acts on a wake;
//! this decides what counts as one.

use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};
use std::time::Instant;

use crate::detector::{SILERO_VAD_FILE, WakeDetector};
use crate::ring::{AudioRing, RING_CAPACITY_SAMPLES};
use crate::speech_check::{SPEECH_CHECK_BUFFER_SAMPLES, SPEECH_CHECK_TAIL_SAMPLES, SpeechCheckPacing, SpeechGate};
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
}

/// The one voice pipeline in a process: one microphone, one pipeline.
pub struct VoicePipeline {
    host: Arc<dyn VoiceHost>,
    wake_log: Arc<WakeLog>,
    capture: Mutex<Capture>,
    paused: AtomicBool,
    listen_armed: AtomicBool,
    last_detection_at: AtomicI64,
    clock: Box<dyn Fn() -> i64 + Send + Sync>,
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
            capture: Mutex::new(Capture { ring: AudioRing::new(RING_CAPACITY_SAMPLES), wake: None }),
            paused: AtomicBool::new(false),
            listen_armed: AtomicBool::new(false),
            last_detection_at: AtomicI64::new(i64::MIN / 2),
            clock,
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

    /// Interim, until speech-to-text moves here: whether a listen is consuming
    /// the microphone, during which a detection is never a wake.
    pub fn set_listen_armed(&self, armed: bool) {
        self.listen_armed.store(armed, Ordering::Relaxed);
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
        if self.listen_armed.load(Ordering::Relaxed) {
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
