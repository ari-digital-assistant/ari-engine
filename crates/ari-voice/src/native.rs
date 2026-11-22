//! `libari_voice_native`: the wake engines and the speech check's VAD, in C++
//! under `native/voice/`, opened at run time by its bare name as
//! `liblitert-lm.so` is. On Android the app ships it next to `libari_ffi.so`;
//! on a host, [`set_library_path`] points at a build from
//! `native/voice/build-host.sh`.
//!
//! Each handle owns one engine and is used from one thread at a time, which is
//! what the C++ requires: `Send`, not `Sync`.

use std::ffi::c_void;
use std::path::PathBuf;
use std::ptr::NonNull;
use std::sync::OnceLock;

/// The C interface major version these bindings follow (`ari_voice_native_api_major`).
const API_MAJOR: i32 = 1;

/// Samples per VAD window. Fixed by the bundled model's input shape.
pub const VAD_WINDOW_SAMPLES: usize = 512;

/// Samples per openWakeWord frame (80 ms).
pub const OWW_FRAME_SAMPLES: usize = 1280;

type Handle = *mut c_void;

ari_native::c_api! {
    /// The C interface in `ari_voice_native.h`, each field its name without `ari_`.
    struct Api in "libari_voice_native.so", prefix "ari_" {
        voice_native_api_major: fn() -> i32;
        mww_create: fn(*const u8, usize, i32, i32, f32, i32, bool) -> Handle;
        mww_process: fn(Handle, *const i16, usize) -> bool;
        mww_reset: fn(Handle);
        mww_destroy: fn(Handle);
        oww_create: fn(*const u8, usize, *const u8, usize, *const u8, usize, f32, bool) -> Handle;
        oww_process: fn(Handle, *const i16, usize, *mut f32, usize, *mut usize) -> bool;
        oww_reset: fn(Handle);
        oww_destroy: fn(Handle);
        vad_create: fn(*const u8, usize) -> Handle;
        vad_window_samples: fn() -> usize;
        vad_compute: fn(Handle, *const f32) -> f32;
        vad_reset: fn(Handle);
        vad_destroy: fn(Handle);
    }
}

static LIBRARY_PATH: OnceLock<PathBuf> = OnceLock::new();

/// Open the library from `path` instead of the loader's search path. For hosts
/// that keep it elsewhere, and tests. Only the first call counts, and only if
/// it comes before the first engine is made; returns whether it took.
pub fn set_library_path(path: PathBuf) -> bool {
    LIBRARY_PATH.set(path).is_ok()
}

/// The library, opened once per process. The error says why it couldn't be.
fn api() -> Result<&'static Api, String> {
    static API: OnceLock<Result<Api, String>> = OnceLock::new();
    API.get_or_init(|| {
        let name = LIBRARY_PATH
            .get()
            .map_or_else(|| "libari_voice_native.so".into(), |p| p.as_os_str().to_owned());
        // SAFETY: loading runs the library's initialisers, which have no
        // preconditions; the version check below comes before any other call.
        let library = unsafe { ari_native::libloading::Library::new(&name) }
            .map_err(|e| format!("can't open {}: {e}", name.to_string_lossy()))?;
        // SAFETY: every major-1 library exports these signatures.
        let api = unsafe { Api::bind(library) }?;
        // SAFETY: no arguments, no preconditions.
        let major = unsafe { (api.voice_native_api_major)() };
        if major != API_MAJOR {
            return Err(format!("libari_voice_native.so has C API {major}; Ari needs {API_MAJOR}"));
        }
        // SAFETY: as above.
        let window = unsafe { (api.vad_window_samples)() };
        if window != VAD_WINDOW_SAMPLES {
            return Err(format!("libari_voice_native.so's VAD takes {window} samples, not {VAD_WINDOW_SAMPLES}"));
        }
        Ok(api)
    })
    .as_ref()
    .map_err(Clone::clone)
}

fn created(handle: Handle, what: &str) -> Result<NonNull<c_void>, String> {
    NonNull::new(handle).ok_or_else(|| format!("{what} would not start (see the native log)"))
}

/// microWakeWord (`MicroWakeWordEngine.cpp`): a TFLite Micro streaming model.
pub struct MicroWakeWord {
    api: &'static Api,
    handle: NonNull<c_void>,
}

// SAFETY: the engine is owned by this value alone and the C++ has no thread
// affinity; it only must not be used from two threads at once, which `&mut`
// already rules out.
unsafe impl Send for MicroWakeWord {}

impl MicroWakeWord {
    /// `model` is copied, so it can be dropped once this returns.
    pub fn new(
        model: &[u8],
        feature_step_ms: i32,
        probability_cutoff: f32,
        sliding_window_size: i32,
        reset_on_detection: bool,
    ) -> Result<Self, String> {
        let api = api()?;
        // SAFETY: `model` is valid for its length for the duration of the call.
        let handle = unsafe {
            (api.mww_create)(
                model.as_ptr(),
                model.len(),
                16_000,
                feature_step_ms,
                probability_cutoff,
                sliding_window_size,
                reset_on_detection,
            )
        };
        Ok(Self { api, handle: created(handle, "microWakeWord")? })
    }

    /// Whether the wake word was detected in `samples`.
    pub fn process(&mut self, samples: &[i16]) -> bool {
        // SAFETY: a live handle, and `samples` valid for its length.
        unsafe { (self.api.mww_process)(self.handle.as_ptr(), samples.as_ptr(), samples.len()) }
    }

    pub fn reset(&mut self) {
        // SAFETY: a live handle.
        unsafe { (self.api.mww_reset)(self.handle.as_ptr()) }
    }
}

impl Drop for MicroWakeWord {
    fn drop(&mut self) {
        // SAFETY: a live handle, never used again.
        unsafe { (self.api.mww_destroy)(self.handle.as_ptr()) }
    }
}

/// openWakeWord (`OpenWakeWordEngine.cpp`): a classifier over a speech
/// embedding, on ONNX Runtime.
pub struct OpenWakeWord {
    api: &'static Api,
    handle: NonNull<c_void>,
}

// SAFETY: as for MicroWakeWord.
unsafe impl Send for OpenWakeWord {}

impl OpenWakeWord {
    /// The three models are copied, so they can be dropped once this returns.
    pub fn new(
        melspectrogram: &[u8],
        embedding: &[u8],
        classifier: &[u8],
        probability_cutoff: f32,
        reset_on_detection: bool,
    ) -> Result<Self, String> {
        let api = api()?;
        // SAFETY: each slice is valid for its length for the duration of the call.
        let handle = unsafe {
            (api.oww_create)(
                melspectrogram.as_ptr(),
                melspectrogram.len(),
                embedding.as_ptr(),
                embedding.len(),
                classifier.as_ptr(),
                classifier.len(),
                probability_cutoff,
                reset_on_detection,
            )
        };
        Ok(Self { api, handle: created(handle, "openWakeWord")? })
    }

    /// Whether the wake word was detected in `samples`.
    pub fn process(&mut self, samples: &[i16]) -> bool {
        // SAFETY: a live handle, `samples` valid for its length, no score buffer.
        unsafe {
            (self.api.oww_process)(
                self.handle.as_ptr(),
                samples.as_ptr(),
                samples.len(),
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut(),
            )
        }
    }

    /// Processes `samples` as [`process`](Self::process) does, and returns the
    /// score of each 80 ms frame they complete instead of whether one crossed
    /// the cutoff. For checking the port against openwakeword's own scores.
    pub fn scores(&mut self, samples: &[i16]) -> Vec<f32> {
        // Each frame needs 1280 new samples, and up to one more completes from
        // samples left over by the previous call.
        let mut scores = vec![0.0; samples.len() / OWW_FRAME_SAMPLES + 1];
        let mut count = 0;
        // SAFETY: a live handle, and every buffer valid for the length given.
        unsafe {
            (self.api.oww_process)(
                self.handle.as_ptr(),
                samples.as_ptr(),
                samples.len(),
                scores.as_mut_ptr(),
                scores.len(),
                &mut count,
            );
        }
        scores.truncate(count);
        scores
    }

    pub fn reset(&mut self) {
        // SAFETY: a live handle.
        unsafe { (self.api.oww_reset)(self.handle.as_ptr()) }
    }
}

impl Drop for OpenWakeWord {
    fn drop(&mut self) {
        // SAFETY: a live handle, never used again.
        unsafe { (self.api.oww_destroy)(self.handle.as_ptr()) }
    }
}

/// Silero VAD (`SileroVad.cpp`), one [`VAD_WINDOW_SAMPLES`] window at a time.
pub struct SileroVad {
    api: &'static Api,
    handle: NonNull<c_void>,
}

// SAFETY: as for MicroWakeWord.
unsafe impl Send for SileroVad {}

impl SileroVad {
    /// `model` is copied, so it can be dropped once this returns.
    pub fn new(model: &[u8]) -> Result<Self, String> {
        let api = api()?;
        // SAFETY: `model` is valid for its length for the duration of the call.
        let handle = unsafe { (api.vad_create)(model.as_ptr(), model.len()) };
        Ok(Self { api, handle: created(handle, "Silero VAD")? })
    }

    /// The speech probability of `window`, samples scaled to [-1, 1). Carries
    /// the model's state on to the next window. `None` if the model failed.
    pub fn compute(&mut self, window: &[f32; VAD_WINDOW_SAMPLES]) -> Option<f32> {
        // SAFETY: a live handle, and a window of the length the library checked at load.
        let p = unsafe { (self.api.vad_compute)(self.handle.as_ptr(), window.as_ptr()) };
        (p >= 0.0).then_some(p)
    }

    /// Zero the state, as for a new stream.
    pub fn reset(&mut self) {
        // SAFETY: a live handle.
        unsafe { (self.api.vad_reset)(self.handle.as_ptr()) }
    }
}

impl Drop for SileroVad {
    fn drop(&mut self) {
        // SAFETY: a live handle, never used again.
        unsafe { (self.api.vad_destroy)(self.handle.as_ptr()) }
    }
}
