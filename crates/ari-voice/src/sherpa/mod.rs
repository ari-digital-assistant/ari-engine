//! sherpa-onnx's C API, `libsherpa-onnx-c-api.so`, opened at run time like
//! the other native libraries: on Android the app ships it inside sherpa's AAR,
//! and a host points at a copy with [`set_library_path`].
//!
//! Only version [`SHERPA_VERSION`] is used: the structs in [`sys`] are that
//! version's layout, and another version's library is refused when it opens.
//!
//! The configs reproduce what Ari's Kotlin set through sherpa's Kotlin API,
//! including the Kotlin API's own defaults where they differ from the C API's
//! (Whisper's tail padding is 1000 there, -1 here). A recogniser is shared by
//! threads; a stream belongs to the one thread that made it and is freed there.

// The C names are CamelCase, and c_api! binds by name.
#![allow(non_snake_case)]

mod sys;

use std::ffi::{CStr, CString, c_char};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::{Arc, OnceLock};

/// The sherpa-onnx release whose C API [`sys`] was generated from, and which
/// the app ships (`sherpa-onnx-2.0.1.aar`).
pub const SHERPA_VERSION: &str = "2.0.1";

/// Every recogniser here runs at 16 kHz.
pub const SAMPLE_RATE: i32 = 16_000;

type OnlineRec = sys::SherpaOnnxOnlineRecognizer;
type OnlineStr = sys::SherpaOnnxOnlineStream;
type OfflineRec = sys::SherpaOnnxOfflineRecognizer;
type OfflineStr = sys::SherpaOnnxOfflineStream;

ari_native::c_api! {
    /// The parts of `c-api.h` Ari uses, each field its name without `SherpaOnnx`.
    struct Api in "libsherpa-onnx-c-api.so", prefix "SherpaOnnx" {
        GetVersionStr: fn() -> *const c_char;
        CreateOnlineRecognizer: fn(*const sys::SherpaOnnxOnlineRecognizerConfig) -> *const OnlineRec;
        DestroyOnlineRecognizer: fn(*const OnlineRec);
        CreateOnlineStream: fn(*const OnlineRec) -> *const OnlineStr;
        DestroyOnlineStream: fn(*const OnlineStr);
        OnlineStreamAcceptWaveform: fn(*const OnlineStr, i32, *const f32, i32);
        IsOnlineStreamReady: fn(*const OnlineRec, *const OnlineStr) -> i32;
        DecodeOnlineStream: fn(*const OnlineRec, *const OnlineStr);
        GetOnlineStreamResult: fn(*const OnlineRec, *const OnlineStr) -> *const sys::SherpaOnnxOnlineRecognizerResult;
        DestroyOnlineRecognizerResult: fn(*const sys::SherpaOnnxOnlineRecognizerResult);
        OnlineStreamInputFinished: fn(*const OnlineStr);
        CreateOfflineRecognizer: fn(*const sys::SherpaOnnxOfflineRecognizerConfig) -> *const OfflineRec;
        DestroyOfflineRecognizer: fn(*const OfflineRec);
        CreateOfflineStream: fn(*const OfflineRec) -> *const OfflineStr;
        DestroyOfflineStream: fn(*const OfflineStr);
        AcceptWaveformOffline: fn(*const OfflineStr, i32, *const f32, i32);
        DecodeOfflineStream: fn(*const OfflineRec, *const OfflineStr);
        GetOfflineStreamResult: fn(*const OfflineStr) -> *const sys::SherpaOnnxOfflineRecognizerResult;
        DestroyOfflineRecognizerResult: fn(*const sys::SherpaOnnxOfflineRecognizerResult);
    }
}

static LIBRARY_PATH: OnceLock<PathBuf> = OnceLock::new();

/// Open the library from `path` instead of the loader's search path. Only the
/// first call counts, and only before the first recogniser; returns whether it took.
pub fn set_library_path(path: PathBuf) -> bool {
    LIBRARY_PATH.set(path).is_ok()
}

fn api() -> Result<&'static Api, String> {
    static API: OnceLock<Result<Api, String>> = OnceLock::new();
    API.get_or_init(|| {
        let name = LIBRARY_PATH
            .get()
            .map_or_else(|| "libsherpa-onnx-c-api.so".into(), |p| p.as_os_str().to_owned());
        // SAFETY: loading runs the library's initialisers, which have no
        // preconditions; the version check comes before any other call.
        let library = unsafe { ari_native::libloading::Library::new(&name) }
            .map_err(|e| format!("can't open {}: {e}", name.to_string_lossy()))?;
        // SAFETY: every sherpa-onnx C API exports these names; the version
        // below decides whether their structs match ours.
        let api = unsafe { Api::bind(library) }?;
        // SAFETY: returns a static string.
        let version = unsafe { CStr::from_ptr((api.GetVersionStr)()) }.to_string_lossy().into_owned();
        if version != SHERPA_VERSION {
            return Err(format!(
                "libsherpa-onnx-c-api.so is sherpa-onnx {version}; Ari's bindings are for {SHERPA_VERSION}"
            ));
        }
        Ok(api)
    })
    .as_ref()
    .map_err(Clone::clone)
}

/// Strings a config points into, kept alive until the create call returns.
#[derive(Default)]
struct Strings(Vec<CString>);

impl Strings {
    fn of(&mut self, s: &str) -> Result<*const c_char, String> {
        let s = CString::new(s).map_err(|_| format!("{s:?} contains a NUL"))?;
        let ptr = s.as_ptr();
        self.0.push(s);
        Ok(ptr)
    }

    fn path(&mut self, p: &Path) -> Result<*const c_char, String> {
        self.of(p.to_str().ok_or_else(|| format!("{} is not UTF-8", p.display()))?)
    }
}

/// A transducer's files, as `SttModel` names them in a model directory.
#[derive(Debug, Clone)]
pub struct TransducerFiles {
    pub encoder: PathBuf,
    pub decoder: PathBuf,
    pub joiner: PathBuf,
    pub tokens: PathBuf,
}

/// What an offline recogniser runs.
#[derive(Debug, Clone)]
pub enum OfflineModel {
    /// Whisper bakes its language into the recogniser; there is no per-decode override.
    Whisper { encoder: PathBuf, decoder: PathBuf, tokens: PathBuf, language: String },
    /// A NeMo transducer (Parakeet), which finds the language itself.
    Transducer { files: TransducerFiles, model_type: String },
}

fn text_of(ptr: *const c_char) -> String {
    if ptr.is_null() {
        String::new()
    } else {
        // SAFETY: sherpa returns NUL-terminated text that lives with its result.
        unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned()
    }
}

/// A streaming recogniser (Kroko, Nemotron: zipformer transducers).
pub struct OnlineRecognizer {
    api: &'static Api,
    ptr: NonNull<OnlineRec>,
}

// SAFETY: sherpa's recogniser decodes any number of streams, each from its
// own thread, which is how the Kotlin used it; its state is the streams'.
unsafe impl Send for OnlineRecognizer {}
unsafe impl Sync for OnlineRecognizer {}

impl OnlineRecognizer {
    /// `model_type` is `zipformer` or `zipformer2`.
    pub fn new(files: &TransducerFiles, model_type: &str) -> Result<Arc<Self>, String> {
        let api = api()?;
        let mut s = Strings::default();
        // SAFETY: an all-zero config is sherpa's own "everything default";
        // every field Ari sets is set below.
        let mut config: sys::SherpaOnnxOnlineRecognizerConfig = unsafe { std::mem::zeroed() };
        config.feat_config.sample_rate = SAMPLE_RATE;
        config.feat_config.feature_dim = 80;
        config.model_config.transducer.encoder = s.path(&files.encoder)?;
        config.model_config.transducer.decoder = s.path(&files.decoder)?;
        config.model_config.transducer.joiner = s.path(&files.joiner)?;
        config.model_config.tokens = s.path(&files.tokens)?;
        config.model_config.num_threads = 2;
        config.model_config.provider = s.of("cpu")?;
        config.model_config.model_type = s.of(model_type)?;
        // Sherpa's energy-based endpoint detection is disabled because it
        // freezes the stream the moment it fires, even if we don't call
        // reset(). And reset() destroys the encoder context, which clips the
        // next ~500ms of speech. Ari does its own endpointing; see
        // `endpoint::should_endpoint`. The rules are kept as they were set.
        config.enable_endpoint = 0;
        config.rule1_min_trailing_silence = 2.0;
        config.rule2_min_trailing_silence = 1.0;
        config.rule3_min_utterance_length = 20.0;
        config.decoding_method = s.of("greedy_search")?;
        config.max_active_paths = 4;
        config.hotwords_score = 1.5;
        // SAFETY: a complete config whose strings outlive the call.
        let ptr = unsafe { (api.CreateOnlineRecognizer)(&config) };
        let ptr = NonNull::new(ptr.cast_mut()).ok_or("sherpa-onnx would not load the streaming model")?;
        Ok(Arc::new(Self { api, ptr }))
    }

    pub fn create_stream(self: &Arc<Self>) -> Result<OnlineStream, String> {
        // SAFETY: a live recogniser.
        let ptr = unsafe { (self.api.CreateOnlineStream)(self.ptr.as_ptr()) };
        let ptr = NonNull::new(ptr.cast_mut()).ok_or("sherpa-onnx would not create a stream")?;
        Ok(OnlineStream { recognizer: Arc::clone(self), ptr })
    }
}

impl Drop for OnlineRecognizer {
    fn drop(&mut self) {
        // SAFETY: the last reference: no stream is left, since each holds one.
        unsafe { (self.api.DestroyOnlineRecognizer)(self.ptr.as_ptr()) }
    }
}

/// One stream through an [`OnlineRecognizer`], which it keeps alive. Owned by
/// one thread and freed there: two threads freeing the same stream is the
/// double free GrapheneOS's allocator aborts on.
pub struct OnlineStream {
    recognizer: Arc<OnlineRecognizer>,
    ptr: NonNull<OnlineStr>,
}

// SAFETY: moved between threads at most, never shared (not Sync).
unsafe impl Send for OnlineStream {}

impl OnlineStream {
    fn api(&self) -> &'static Api {
        self.recognizer.api
    }

    /// 16 kHz samples scaled to [-1, 1).
    pub fn accept_waveform(&mut self, samples: &[f32]) {
        // SAFETY: a live stream and a buffer of the length given.
        unsafe {
            (self.api().OnlineStreamAcceptWaveform)(self.ptr.as_ptr(), SAMPLE_RATE, samples.as_ptr(), samples.len() as i32)
        }
    }

    /// Decode everything the stream has ready.
    pub fn decode_ready(&mut self) {
        let (rec, s) = (self.recognizer.ptr.as_ptr(), self.ptr.as_ptr());
        // SAFETY: a live recogniser and stream, decoded by this thread alone.
        unsafe {
            while (self.api().IsOnlineStreamReady)(rec, s) != 0 {
                (self.api().DecodeOnlineStream)(rec, s);
            }
        }
    }

    /// No more audio: lets the decoder commit what it holds back for context.
    pub fn input_finished(&mut self) {
        // SAFETY: a live stream.
        unsafe { (self.api().OnlineStreamInputFinished)(self.ptr.as_ptr()) }
    }

    /// The transcript so far, untrimmed.
    pub fn text(&self) -> String {
        // SAFETY: a live recogniser and stream; the result is freed below.
        unsafe {
            let result = (self.api().GetOnlineStreamResult)(self.recognizer.ptr.as_ptr(), self.ptr.as_ptr());
            if result.is_null() {
                return String::new();
            }
            let text = text_of((*result).text);
            (self.api().DestroyOnlineRecognizerResult)(result);
            text
        }
    }
}

impl Drop for OnlineStream {
    fn drop(&mut self) {
        // SAFETY: a live stream, never used again, on the thread that owns it.
        unsafe { (self.api().DestroyOnlineStream)(self.ptr.as_ptr()) }
    }
}

/// A recogniser that decodes a whole utterance at once (Whisper, Parakeet).
pub struct OfflineRecognizer {
    api: &'static Api,
    ptr: NonNull<OfflineRec>,
}

// SAFETY: as for OnlineRecognizer.
unsafe impl Send for OfflineRecognizer {}
unsafe impl Sync for OfflineRecognizer {}

impl OfflineRecognizer {
    pub fn new(model: &OfflineModel) -> Result<Arc<Self>, String> {
        let api = api()?;
        let mut s = Strings::default();
        // SAFETY: as for the online config.
        let mut config: sys::SherpaOnnxOfflineRecognizerConfig = unsafe { std::mem::zeroed() };
        config.feat_config.sample_rate = SAMPLE_RATE;
        config.feat_config.feature_dim = 80;
        match model {
            OfflineModel::Whisper { encoder, decoder, tokens, language } => {
                let w = &mut config.model_config.whisper;
                w.encoder = s.path(encoder)?;
                w.decoder = s.path(decoder)?;
                w.language = s.of(language)?;
                w.task = s.of("transcribe")?;
                // The Kotlin API's default; the C API's would be -1.
                w.tail_paddings = 1000;
                config.model_config.tokens = s.path(tokens)?;
                config.model_config.model_type = s.of("whisper")?;
            }
            OfflineModel::Transducer { files, model_type } => {
                let t = &mut config.model_config.transducer;
                t.encoder = s.path(&files.encoder)?;
                t.decoder = s.path(&files.decoder)?;
                t.joiner = s.path(&files.joiner)?;
                config.model_config.tokens = s.path(&files.tokens)?;
                config.model_config.model_type = s.of(model_type)?;
            }
        }
        config.model_config.num_threads = 2;
        config.model_config.provider = s.of("cpu")?;
        config.decoding_method = s.of("greedy_search")?;
        config.max_active_paths = 4;
        config.hotwords_score = 1.5;
        // SAFETY: a complete config whose strings outlive the call.
        let ptr = unsafe { (api.CreateOfflineRecognizer)(&config) };
        let ptr = NonNull::new(ptr.cast_mut()).ok_or("sherpa-onnx would not load the model")?;
        Ok(Arc::new(Self { api, ptr }))
    }

    /// Decode `samples` (16 kHz, [-1, 1)) in one go and return the text, untrimmed.
    pub fn transcribe(self: &Arc<Self>, samples: &[f32]) -> Result<String, String> {
        // SAFETY: a live recogniser; the stream is made, used and freed here,
        // on this thread, and the result freed before it.
        unsafe {
            let stream = (self.api.CreateOfflineStream)(self.ptr.as_ptr());
            if stream.is_null() {
                return Err("sherpa-onnx would not create a stream".into());
            }
            (self.api.AcceptWaveformOffline)(stream, SAMPLE_RATE, samples.as_ptr(), samples.len() as i32);
            (self.api.DecodeOfflineStream)(self.ptr.as_ptr(), stream);
            let result = (self.api.GetOfflineStreamResult)(stream);
            let text = if result.is_null() { String::new() } else { text_of((*result).text) };
            if !result.is_null() {
                (self.api.DestroyOfflineRecognizerResult)(result);
            }
            (self.api.DestroyOfflineStream)(stream);
            Ok(text)
        }
    }
}

impl Drop for OfflineRecognizer {
    fn drop(&mut self) {
        // SAFETY: the last reference.
        unsafe { (self.api.DestroyOfflineRecognizer)(self.ptr.as_ptr()) }
    }
}

/// 16-bit samples as sherpa takes them.
pub fn to_float(samples: &[i16]) -> Vec<f32> {
    samples.iter().map(|s| *s as f32 / 32768.0).collect()
}
