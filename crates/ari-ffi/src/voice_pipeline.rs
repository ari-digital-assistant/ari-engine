//! UniFFI wrapper around `ari_voice`'s audio side: the voice pipeline, its
//! wake log, and a wake detector for the sensitivity tuner and parity tests.
//!
//! The microphone's audio does not come through UniFFI. It goes through
//! [`ari_voice_push_audio`], a plain C function, and on Android through the
//! JNI entry point below it: the capture thread calls it every 30 ms all day,
//! and UniFFI's route (JNA, a copy into a buffer, boxing) costs more than the
//! JNI call it replaces.

use std::sync::{Arc, Mutex};

use ari_voice::detector::{OWW_EMBEDDING_FILE, OWW_MELSPECTROGRAM_FILE, WakeDetector};
use ari_voice::pipeline::{self, LogLevel, VoiceHost, VoicePipeline, WakeConfig};
use ari_voice::wake_log::{WakeLog, WakeOutcome};
use ari_voice::wake_model::WakeEngine;

use crate::FfiLogLevel;
use crate::voice::{FfiOperatingPoint, FfiWakeSensitivity, FfiWakeWordModel};

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

#[uniffi::export]
impl FfiWakeOutcome {
    /// The overlay opened and nobody had called Ari.
    pub fn false_wake(&self) -> bool {
        WakeOutcome::from(*self).false_wake()
    }

    pub fn slug(&self) -> String {
        WakeOutcome::from(*self).slug().to_string()
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

/// The wake log file (`ari_voice::wake_log::WakeLog`). One per process, shared
/// by the pipeline and the frontend's voice session.
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

    /// `hidden`: a quiet wake that ended without ever showing anything.
    pub fn outcome(&self, outcome: FfiWakeOutcome, hidden: bool) {
        self.inner.outcome(outcome.into(), hidden);
    }

    /// Interim until the session moves (step 4): how a wake's listen ended.
    pub fn listen_ended(&self, end: String, after_ms: i64) {
        self.inner.listen_ended(&end, after_ms);
    }

    /// Interim until the session moves (step 4): a quiet wake's early name check.
    pub fn early_name_check(&self, result: String, after_ms: i64, in_time: bool) {
        self.inner.early_name_check(&result, after_ms, in_time);
    }

    /// The last day and week summarised, then the week's lines, for a bug report.
    pub fn report(&self, now_ms: i64, listening_now: bool) -> String {
        self.inner.report(now_ms, listening_now)
    }
}

/// What the pipeline needs from the app: model files, wakes, and log lines.
/// Called on engine threads and the capture thread; hand work on rather than
/// doing it in the call.
#[uniffi::export(with_foreign)]
pub trait FfiVoiceHost: Send + Sync {
    /// A model file by name, or `None` when it cannot be read.
    fn load_model(&self, name: String) -> Option<Vec<u8>>;
    /// The wake word was heard and passed the debounce and the speech check.
    fn wake_detected(&self);
    fn log(&self, level: FfiLogLevel, message: String);
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

    /// Interim until speech-to-text moves (step 3): a listen is consuming the mic.
    pub fn set_listen_armed(&self, armed: bool) {
        self.inner.set_listen_armed(armed);
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
