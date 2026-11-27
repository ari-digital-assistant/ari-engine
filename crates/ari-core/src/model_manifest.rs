//! The manifest published beside each downloadable model: its version and
//! the files to fetch, each with its checksum. Moved from ari-android's
//! `ModelManifest.kt`.
//!
//! Two shapes arrive. A single file (the router, an on-device assistant):
//!
//! ```json
//! { "version": "…", "url": "…", "sha256": "…", "size_bytes": 1, "released_at": "…" }
//! ```
//!
//! and a bundle (a speech-to-text model):
//!
//! ```json
//! { "version": "…", "released_at": "…",
//!   "files": [{ "name": "…", "url": "…", "sha256": "…", "size_bytes": 1 }] }
//! ```
//!
//! Both read as a list of files under one version. A single file is named
//! after the last part of its URL.

use serde_json::{Map, Value};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelManifest {
    pub version: String,
    pub released_at: Option<String>,
    pub files: Vec<ManifestFile>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestFile {
    pub name: String,
    pub url: String,
    pub sha256: String,
    pub size_bytes: u64,
}

/// Why a manifest couldn't be read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestError(pub String);

impl std::fmt::Display for ManifestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ManifestError {}

impl ModelManifest {
    pub fn parse(json: &str) -> Result<Self, ManifestError> {
        let root: Value = serde_json::from_str(json).map_err(|e| ManifestError(e.to_string()))?;
        let root = root.as_object().ok_or_else(|| ManifestError("not an object".into()))?;
        let files = match root.get("files") {
            Some(files) => files
                .as_array()
                .ok_or_else(|| ManifestError("`files` isn't a list".into()))?
                .iter()
                .map(|file| {
                    let file = file.as_object().ok_or_else(|| ManifestError("a file isn't an object".into()))?;
                    Ok(ManifestFile {
                        name: string(file, "name")?,
                        url: string(file, "url")?,
                        sha256: string(file, "sha256")?,
                        size_bytes: size(file)?,
                    })
                })
                .collect::<Result<Vec<_>, ManifestError>>()?,
            None => {
                let url = string(root, "url")?;
                vec![ManifestFile {
                    name: url.rsplit('/').next().unwrap_or_default().to_string(),
                    sha256: string(root, "sha256")?,
                    size_bytes: size(root)?,
                    url,
                }]
            }
        };
        if files.is_empty() {
            return Err(ManifestError("no files".into()));
        }
        Ok(Self {
            version: string(root, "version")?,
            released_at: root.get("released_at").and_then(Value::as_str).filter(|s| !s.is_empty()).map(str::to_string),
            files,
        })
    }

    pub fn total_size_bytes(&self) -> u64 {
        self.files.iter().map(|f| f.size_bytes).sum()
    }
}

fn string(o: &Map<String, Value>, key: &str) -> Result<String, ManifestError> {
    o.get(key).and_then(Value::as_str).map(str::to_string).ok_or_else(|| ManifestError(format!("no `{key}`")))
}

/// `size_bytes`, as a number or, as Android's JSON reader took it, a number
/// written as a string.
fn size(o: &Map<String, Value>) -> Result<u64, ManifestError> {
    match o.get("size_bytes") {
        Some(Value::Number(n)) => n.as_u64(),
        Some(Value::String(s)) => s.trim().parse().ok(),
        _ => None,
    }
    .ok_or_else(|| ManifestError("no `size_bytes`".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bundle_lists_its_files_under_one_version() {
        let manifest = ModelManifest::parse(
            r#"{"version":"3","released_at":"2026-11-20T10:00:00Z","files":[
                {"name":"encoder.onnx","url":"https://x/encoder.onnx","sha256":"aa","size_bytes":70000000},
                {"name":"tokens.txt","url":"https://x/tokens.txt","sha256":"bb","size_bytes":"1500"}]}"#,
        )
        .unwrap();
        assert_eq!(manifest.version, "3");
        assert_eq!(manifest.released_at.as_deref(), Some("2026-11-20T10:00:00Z"));
        assert_eq!(
            manifest.files,
            [
                ManifestFile {
                    name: "encoder.onnx".into(),
                    url: "https://x/encoder.onnx".into(),
                    sha256: "aa".into(),
                    size_bytes: 70_000_000,
                },
                ManifestFile {
                    name: "tokens.txt".into(),
                    url: "https://x/tokens.txt".into(),
                    sha256: "bb".into(),
                    size_bytes: 1500,
                },
            ]
        );
        assert_eq!(manifest.total_size_bytes(), 70_001_500);
    }

    #[test]
    fn a_single_file_is_named_after_its_url() {
        let manifest = ModelManifest::parse(
            r#"{"version":"v7","url":"https://h/models/router-q4.gguf","sha256":"cc","size_bytes":42,"released_at":""}"#,
        )
        .unwrap();
        assert_eq!(manifest.released_at, None);
        assert_eq!(manifest.files.len(), 1);
        assert_eq!(manifest.files[0].name, "router-q4.gguf");
        assert_eq!(manifest.files[0].size_bytes, 42);
    }

    #[test]
    fn a_manifest_missing_what_a_download_needs_is_refused() {
        for json in [
            "not json",
            "[]",
            r#"{"url":"https://h/a","sha256":"cc","size_bytes":1}"#,
            r#"{"version":"1","sha256":"cc","size_bytes":1}"#,
            r#"{"version":"1","url":"https://h/a","size_bytes":1}"#,
            r#"{"version":"1","url":"https://h/a","sha256":"cc"}"#,
            r#"{"version":"1","files":[]}"#,
            r#"{"version":"1","files":[{"name":"a","url":"u","sha256":"s"}]}"#,
            r#"{"version":"1","files":"none"}"#,
        ] {
            assert!(ModelManifest::parse(json).is_err(), "{json}");
        }
    }
}
