//! The on-device speech-to-text models Ari offers, and which can serve which
//! language. Moved from ari-android's `SttModel.kt`. Each frontend downloads
//! them (following each one's manifest, `ari_core::model_manifest`) and names
//! them in its own words; the user only chooses on-device or cloud, and the
//! language decides the model.

use std::path::PathBuf;

use crate::stt::SttModelSpec;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SttModel {
    pub id: &'static str,
    /// The whole download.
    pub total_bytes: u64,
    pub encoder_file: &'static str,
    pub decoder_file: &'static str,
    /// Transducers (Kroko, Parakeet) have one; Whisper doesn't.
    pub joiner_file: Option<&'static str>,
    pub tokens_file: &'static str,
    /// Each file is at `<base_url>/<file>`, for when the manifest can't be had.
    pub base_url: &'static str,
    /// As [`SttModelSpec::model_type`].
    pub model_type: &'static str,
    /// The bundle manifest covering every file, with their checksums.
    pub manifest_url: &'static str,
}

impl SttModel {
    /// The files that make up the model.
    pub fn files(&self) -> Vec<&'static str> {
        [Some(self.encoder_file), Some(self.decoder_file), self.joiner_file, Some(self.tokens_file)]
            .into_iter()
            .flatten()
            .collect()
    }

    /// The model, downloaded into `dir`, for [`crate::pipeline::VoicePipeline::load_stt`].
    pub fn spec(&self, dir: PathBuf) -> SttModelSpec {
        SttModelSpec {
            id: self.id.to_string(),
            model_type: self.model_type.to_string(),
            dir,
            encoder_file: self.encoder_file.to_string(),
            decoder_file: self.decoder_file.to_string(),
            joiner_file: self.joiner_file.map(str::to_string),
            tokens_file: self.tokens_file.to_string(),
        }
    }
}

/// Kroko's streaming English Zipformer: small, and it writes as you speak.
pub const KROKO: SttModel = SttModel {
    id: "kroko-2025-08-06",
    total_bytes: 71_500_000,
    encoder_file: "encoder.onnx",
    decoder_file: "decoder.onnx",
    joiner_file: Some("joiner.onnx"),
    tokens_file: "tokens.txt",
    base_url: "https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06/resolve/main",
    model_type: "zipformer2",
    manifest_url: "https://github.com/ari-digital-assistant/ari-tools/releases/download/stt-kroko-latest/manifest.json",
};

/// Moondream's Parakeet Ultra, a further-trained NVIDIA Parakeet TDT 0.6B v3,
/// int8 by csukuangfj, CC-BY-4.0. Buffered: no partials, one decode at the
/// end of speech. On the November 2026 bench (ari-android#26) it matched
/// Whisper Turbo on Italian at a ninth of the decode time, and beat Kroko on
/// English accuracy, without Kroko's partials, at nine times the download.
pub const PARAKEET_ULTRA: SttModel = SttModel {
    id: "parakeet-ultra-int8-2026-11",
    total_bytes: 670_600_000,
    encoder_file: "encoder.int8.onnx",
    decoder_file: "decoder.int8.onnx",
    joiner_file: Some("joiner.int8.onnx"),
    tokens_file: "tokens.txt",
    base_url: "https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-ultra-int8/resolve/main",
    model_type: "nemo_transducer",
    manifest_url: "https://github.com/ari-digital-assistant/ari-tools/releases/download/stt-parakeet-ultra-latest/manifest.json",
};

/// OpenAI's whisper-large-v3-turbo, int8 by csukuangfj: 99 languages, for
/// those Parakeet doesn't cover. Buffered, as Parakeet is.
pub const WHISPER_TURBO: SttModel = SttModel {
    id: "whisper-turbo-int8-2024-09",
    total_bytes: 1_037_000_000,
    encoder_file: "turbo-encoder.int8.onnx",
    decoder_file: "turbo-decoder.int8.onnx",
    joiner_file: None,
    tokens_file: "turbo-tokens.txt",
    base_url: "https://huggingface.co/csukuangfj/sherpa-onnx-whisper-turbo/resolve/main",
    model_type: "whisper",
    manifest_url: "https://github.com/ari-digital-assistant/ari-tools/releases/download/stt-whisper-turbo-latest/manifest.json",
};

/// Every model offered.
pub const ALL: [SttModel; 3] = [KROKO, PARAKEET_ULTRA, WHISPER_TURBO];

/// Models no longer offered. An install using one has to move to another:
/// [`by_id`] doesn't know it, which would leave no recogniser at all.
///
/// `nemotron-0.6b-int8-2026-01-14` was dropped after replaying the capture
/// set through it: on the same audio it heard "how's the weat" and "ton"
/// where the 71 MB Kroko heard "how's the weather" and "turn on the kitchen
/// table". Ten times the size and an order of magnitude worse, most likely
/// from int8 quantisation of a 0.6B streaming transducer.
pub const RETIRED_IDS: [&str; 1] = ["nemotron-0.6b-int8-2026-01-14"];

/// The 25 languages of Parakeet TDT v3, which Ultra is trained from.
const PARAKEET_LANGUAGES: [&str; 25] = [
    "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "lt", "lv", "mt", "nl", "pl", "pt",
    "ro", "ru", "sk", "sl", "sv", "uk",
];

pub fn by_id(id: &str) -> Option<SttModel> {
    ALL.into_iter().find(|m| m.id == id)
}

/// The models that can transcribe `locale`, recommended first. Kroko is
/// English-only and stays first there, being small and streaming; Parakeet
/// leads for the other languages it covers; every other language has only
/// Whisper.
pub fn on_device_choices(locale: &str) -> Vec<SttModel> {
    let language = locale.split(['-', '_']).next().unwrap_or_default().to_lowercase();
    if language == "en" {
        vec![KROKO, PARAKEET_ULTRA]
    } else if PARAKEET_LANGUAGES.contains(&language.as_str()) {
        vec![PARAKEET_ULTRA, WHISPER_TURBO]
    } else {
        vec![WHISPER_TURBO]
    }
}

/// The model to use for `locale`: `current` while it can transcribe it, so
/// a choice made in settings outlives a trip to the cloud and back, else the
/// recommended one.
pub fn on_device_for(locale: &str, current: Option<&str>) -> SttModel {
    let choices = on_device_choices(locale);
    current.and_then(|id| choices.iter().copied().find(|m| m.id == id)).unwrap_or(choices[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn english_gets_kroko_whatever_the_region() {
        for locale in ["en", "en-GB", "en_US", "EN"] {
            assert_eq!(on_device_for(locale, None), KROKO, "{locale}");
        }
    }

    #[test]
    fn parakeet_languages_get_parakeet_ultra() {
        for locale in ["it", "mt", "de-CH"] {
            assert_eq!(on_device_for(locale, None), PARAKEET_ULTRA, "{locale}");
        }
    }

    #[test]
    fn every_other_language_gets_whisper_turbo() {
        // Kroko is English-only and Parakeet European-only, so every other
        // locale must land on Whisper or there's no on-device option.
        for locale in ["ja", "zh-Hans", ""] {
            assert_eq!(on_device_for(locale, None), WHISPER_TURBO, "{locale}");
        }
    }

    #[test]
    fn a_chosen_model_is_kept_while_it_serves_the_language() {
        assert_eq!(on_device_for("en", Some(PARAKEET_ULTRA.id)), PARAKEET_ULTRA);
        // Kroko can't transcribe Italian, so it gives way to the recommendation.
        assert_eq!(on_device_for("it", Some(KROKO.id)), PARAKEET_ULTRA);
        assert_eq!(on_device_for("en", Some("no-such-model")), KROKO);
    }

    #[test]
    fn the_choices_lead_with_the_recommended_model() {
        assert_eq!(on_device_choices("en-GB"), [KROKO, PARAKEET_ULTRA]);
        assert_eq!(on_device_choices("it"), [PARAKEET_ULTRA, WHISPER_TURBO]);
        assert_eq!(on_device_choices("ja"), [WHISPER_TURBO]);
    }

    #[test]
    fn a_retired_model_is_never_offered_or_found() {
        // by_id finding nothing is what makes a frontend move the install
        // on; if a retired id ever resolved again, that would silently stop.
        for id in RETIRED_IDS {
            assert_eq!(by_id(id), None);
            assert!(ALL.iter().all(|m| m.id != id));
        }
        assert_eq!(by_id(KROKO.id), Some(KROKO));
    }

    #[test]
    fn a_model_s_files_and_spec_name_what_it_downloads() {
        assert_eq!(KROKO.files(), ["encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"]);
        assert_eq!(WHISPER_TURBO.files(), ["turbo-encoder.int8.onnx", "turbo-decoder.int8.onnx", "turbo-tokens.txt"]);
        let spec = WHISPER_TURBO.spec(PathBuf::from("/models/whisper"));
        assert_eq!(spec.id, WHISPER_TURBO.id);
        assert_eq!(spec.model_type, "whisper");
        assert_eq!(spec.dir, PathBuf::from("/models/whisper"));
        assert_eq!(spec.joiner_file, None);
        assert_eq!(spec.tokens_file, "turbo-tokens.txt");
    }
}
