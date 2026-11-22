//! A wake word engine as the capture loop and the tuner use it, whichever kind
//! the model needs.

use crate::native::{MicroWakeWord, OpenWakeWord};
use crate::wake_model::{OperatingPoint, WakeEngine, WakeWordModel};

/// openwakeword 0.6.0's feature models, from its release v0.5.1, shared by
/// every openWakeWord classifier.
pub const OWW_MELSPECTROGRAM_FILE: &str = "openwakeword_melspectrogram.onnx";
pub const OWW_EMBEDDING_FILE: &str = "openwakeword_embedding.onnx";

/// The speech check's VAD model.
pub const SILERO_VAD_FILE: &str = "silero_vad.onnx";

/// A wake word engine. Detection state lives in the engine, so one detector
/// follows one audio stream.
pub enum WakeDetector {
    Micro(MicroWakeWord),
    Open(OpenWakeWord),
}

impl WakeDetector {
    /// `model`'s engine at `point`. `load` reads a model file by name: the
    /// model's own [`asset_filename`](WakeWordModel::asset_filename), and for
    /// openWakeWord the two feature models every classifier shares.
    ///
    /// `reset_on_detection` false leaves the engine alone when it reports a
    /// detection, so it reports again while its window stays above the cutoff;
    /// whoever acts on a detection must then reset it.
    pub fn new(
        model: &WakeWordModel,
        point: OperatingPoint,
        reset_on_detection: bool,
        load: &dyn Fn(&str) -> Result<Vec<u8>, String>,
    ) -> Result<Self, String> {
        let bytes = load(&model.asset_filename)?;
        Self::from_bytes(model, point, reset_on_detection, &bytes, load)
    }

    /// As [`new`](Self::new), with the model's own file already read.
    pub fn from_bytes(
        model: &WakeWordModel,
        point: OperatingPoint,
        reset_on_detection: bool,
        model_bytes: &[u8],
        load: &dyn Fn(&str) -> Result<Vec<u8>, String>,
    ) -> Result<Self, String> {
        Ok(match model.engine {
            WakeEngine::MicroWakeWord => Self::Micro(MicroWakeWord::new(
                model_bytes,
                model.feature_step_size_ms,
                point.probability_cutoff,
                point.sliding_window_size,
                reset_on_detection,
            )?),
            WakeEngine::OpenWakeWord => Self::Open(OpenWakeWord::new(
                &load(OWW_MELSPECTROGRAM_FILE)?,
                &load(OWW_EMBEDDING_FILE)?,
                model_bytes,
                point.probability_cutoff,
                reset_on_detection,
            )?),
        })
    }

    /// Whether the wake word was detected in `samples`.
    pub fn process_audio(&mut self, samples: &[i16]) -> bool {
        match self {
            Self::Micro(engine) => engine.process(samples),
            Self::Open(engine) => engine.process(samples),
        }
    }

    pub fn reset(&mut self) {
        match self {
            Self::Micro(engine) => engine.reset(),
            Self::Open(engine) => engine.reset(),
        }
    }
}
