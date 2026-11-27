//! [`ari_voice::stt_models`] and [`ari_core::model_manifest`] across the
//! UniFFI boundary: the speech-to-text models on offer, which serves which
//! language, and the manifests models are downloaded by.

use ari_core::model_manifest::{ManifestFile, ModelManifest};
use ari_voice::stt_models::{self, SttModel};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, uniffi::Record)]
pub struct FfiSttModel {
    pub id: String,
    pub total_bytes: i64,
    pub encoder_file: String,
    pub decoder_file: String,
    pub joiner_file: Option<String>,
    pub tokens_file: String,
    pub base_url: String,
    pub model_type: String,
    pub manifest_url: String,
}

impl From<SttModel> for FfiSttModel {
    fn from(m: SttModel) -> Self {
        Self {
            id: m.id.to_string(),
            total_bytes: m.total_bytes as i64,
            encoder_file: m.encoder_file.to_string(),
            decoder_file: m.decoder_file.to_string(),
            joiner_file: m.joiner_file.map(str::to_string),
            tokens_file: m.tokens_file.to_string(),
            base_url: m.base_url.to_string(),
            model_type: m.model_type.to_string(),
            manifest_url: m.manifest_url.to_string(),
        }
    }
}

/// Every speech-to-text model offered.
#[uniffi::export]
pub fn stt_models() -> Vec<FfiSttModel> {
    stt_models::ALL.into_iter().map(Into::into).collect()
}

/// The models that can transcribe `locale`, recommended first.
#[uniffi::export]
pub fn stt_on_device_choices(locale: String) -> Vec<FfiSttModel> {
    stt_models::on_device_choices(&locale).into_iter().map(Into::into).collect()
}

/// The model for `locale`: `current` while it serves it, else the
/// recommended one.
#[uniffi::export]
pub fn stt_on_device_for(locale: String, current: Option<String>) -> FfiSttModel {
    stt_models::on_device_for(&locale, current.as_deref()).into()
}

/// Model ids no longer offered, which an install has to move off.
#[uniffi::export]
pub fn stt_retired_ids() -> Vec<String> {
    stt_models::RETIRED_IDS.iter().map(|id| id.to_string()).collect()
}

#[derive(Debug, Clone, PartialEq, Eq, uniffi::Record)]
pub struct FfiModelManifest {
    pub version: String,
    pub released_at: Option<String>,
    pub files: Vec<FfiManifestFile>,
}

#[derive(Debug, Clone, PartialEq, Eq, uniffi::Record)]
pub struct FfiManifestFile {
    pub name: String,
    pub url: String,
    pub sha256: String,
    pub size_bytes: i64,
}

#[derive(Debug, Error, uniffi::Error)]
#[uniffi(flat_error)]
pub enum FfiManifestError {
    #[error("model manifest: {message}")]
    Invalid { message: String },
}

impl From<ManifestFile> for FfiManifestFile {
    fn from(f: ManifestFile) -> Self {
        Self { name: f.name, url: f.url, sha256: f.sha256, size_bytes: f.size_bytes as i64 }
    }
}

/// The model manifest in `json` ([`ari_core::model_manifest`]).
#[uniffi::export]
pub fn parse_model_manifest(json: String) -> Result<FfiModelManifest, FfiManifestError> {
    let manifest = ModelManifest::parse(&json).map_err(|e| FfiManifestError::Invalid { message: e.0 })?;
    Ok(FfiModelManifest {
        version: manifest.version,
        released_at: manifest.released_at,
        files: manifest.files.into_iter().map(Into::into).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_models_cross_with_their_files_and_sizes() {
        let models = stt_models();
        assert_eq!(
            models.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
            ["kroko-2025-08-06", "parakeet-ultra-int8-2026-11", "whisper-turbo-int8-2024-09"]
        );
        assert_eq!(models[2].joiner_file, None);
        assert_eq!(models[2].total_bytes, 1_037_000_000);
        assert_eq!(stt_on_device_for("it".into(), Some("kroko-2025-08-06".into())).id, "parakeet-ultra-int8-2026-11");
        assert_eq!(stt_on_device_choices("en-GB".into()).len(), 2);
        assert_eq!(stt_retired_ids(), ["nemotron-0.6b-int8-2026-01-14"]);
    }

    #[test]
    fn a_manifest_crosses_or_says_why_not() {
        let manifest =
            parse_model_manifest(r#"{"version":"2","url":"https://h/a.onnx","sha256":"s","size_bytes":5}"#.into())
                .unwrap();
        assert_eq!(manifest.files[0].name, "a.onnx");
        assert_eq!(manifest.files[0].size_bytes, 5);
        assert!(matches!(parse_model_manifest("{}".into()), Err(FfiManifestError::Invalid { .. })));
    }
}
