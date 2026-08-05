//! Tier → model resolution for cloud assistant skills.
//!
//! Cloud assistant manifests used to enumerate concrete model IDs in their
//! `settings` block, which meant a skill release every time a provider shipped
//! a new model. Instead they now offer three stable tiers — `fast`, `balanced`,
//! `smartest` — and the concrete ID comes from the registry's `models.json`,
//! regenerated nightly by `ari-skills/tools/build-models.sh` and signed by the
//! same key as every bundle.
//!
//! This module is deliberately just parse + lookup. Where the catalog file
//! lives on disk is a frontend decision (same as the config store), and
//! fetching it is [`crate::registry`]'s job.
//!
//! A missing or stale catalog is not an error: [`crate::assistant`] falls back
//! to the per-tier `default_models` pinned in the skill's own manifest, so a
//! first run, an offline device, or a failed fetch still resolves to a working
//! model.

use serde::Deserialize;
use std::collections::BTreeMap;
use thiserror::Error;

/// Catalog schema this build understands. `build-models.sh` writes 1.
pub const SUPPORTED_CATALOG_VERSION: u32 = 1;

/// Sanity cap on catalog size. The real file is a few kilobytes; anything
/// larger is a misconfiguration or a MITM trying to wedge the parser.
pub const MAX_CATALOG_BYTES: usize = 1024 * 1024;

#[derive(Debug, Error)]
pub enum ModelCatalogError {
    #[error("could not parse models.json: {0}")]
    Parse(String),

    #[error("models.json declares schema_version {found}, this build supports {supported}")]
    UnsupportedVersion { found: u32, supported: u32 },

    #[error("models.json is {size} bytes, over the {max} byte limit")]
    TooLarge { size: usize, max: usize },
}

/// One resolved model: which ID to send, and which request fields to leave out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TierModel {
    pub id: String,
    /// Request fields this model rejects outright. Newer reasoning families
    /// return HTTP 400 for sampling parameters rather than ignoring them, so
    /// this is the difference between a working skill and one that errors on
    /// every call.
    pub omit_params: Vec<String>,
}

impl TierModel {
    pub fn omits(&self, param: &str) -> bool {
        self.omit_params.iter().any(|p| p == param)
    }
}

#[derive(Debug, Deserialize)]
struct RawCatalog {
    schema_version: u32,
    #[serde(default)]
    providers: BTreeMap<String, BTreeMap<String, Option<RawTier>>>,
}

/// Only the two fields the engine acts on. Everything else in the file —
/// pricing, context window, release date, selection_method — exists so a human
/// reviewing the nightly refresh PR can see what changed, and is ignored here.
#[derive(Debug, Deserialize)]
struct RawTier {
    id: String,
    #[serde(default)]
    omit_params: Vec<String>,
}

/// Parsed `models.json`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelCatalog {
    providers: BTreeMap<String, BTreeMap<String, TierModel>>,
}

impl ModelCatalog {
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, ModelCatalogError> {
        if bytes.len() > MAX_CATALOG_BYTES {
            return Err(ModelCatalogError::TooLarge {
                size: bytes.len(),
                max: MAX_CATALOG_BYTES,
            });
        }

        let raw: RawCatalog = serde_json::from_slice(bytes)
            .map_err(|e| ModelCatalogError::Parse(e.to_string()))?;

        if raw.schema_version != SUPPORTED_CATALOG_VERSION {
            return Err(ModelCatalogError::UnsupportedVersion {
                found: raw.schema_version,
                supported: SUPPORTED_CATALOG_VERSION,
            });
        }

        // A null tier means the generator found no model for it. Drop it rather
        // than carrying an empty entry, so lookup misses and the caller falls
        // back to the manifest pin. Same for a blank id.
        let providers = raw
            .providers
            .into_iter()
            .map(|(provider, tiers)| {
                let tiers = tiers
                    .into_iter()
                    .filter_map(|(tier, raw_tier)| {
                        let raw_tier = raw_tier?;
                        if raw_tier.id.is_empty() {
                            return None;
                        }
                        Some((
                            tier,
                            TierModel {
                                id: raw_tier.id,
                                omit_params: raw_tier.omit_params,
                            },
                        ))
                    })
                    .collect();
                (provider, tiers)
            })
            .collect();

        Ok(ModelCatalog { providers })
    }

    pub fn lookup(&self, provider: &str, tier: &str) -> Option<&TierModel> {
        self.providers.get(provider)?.get(tier)
    }

    pub fn is_empty(&self) -> bool {
        self.providers.values().all(|tiers| tiers.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CATALOG: &str = r#"{
      "schema_version": 1,
      "generated_at": "2026-08-05T11:58:48Z",
      "providers": {
        "openai": {
          "fast": {
            "id": "gpt-5.6-luna",
            "name": "GPT-5.6 Luna",
            "omit_params": ["temperature", "top_p", "top_k"],
            "pricing_per_million_tokens": {"input": 0.2, "output": 1.2}
          },
          "balanced": {"id": "gpt-5.6-terra", "omit_params": ["temperature"]},
          "smartest": {"id": "gpt-5.6-sol", "omit_params": []}
        },
        "anthropic": {
          "fast": {"id": "claude-haiku-4-5", "omit_params": []},
          "balanced": null,
          "smartest": {"id": "", "omit_params": []}
        }
      }
    }"#;

    #[test]
    fn parses_and_looks_up_a_tier() {
        let catalog = ModelCatalog::from_json_bytes(CATALOG.as_bytes()).unwrap();
        let model = catalog.lookup("openai", "fast").unwrap();
        assert_eq!(model.id, "gpt-5.6-luna");
        assert_eq!(
            model.omit_params,
            vec!["temperature".to_string(), "top_p".to_string(), "top_k".to_string()]
        );
        assert!(model.omits("temperature"));
        assert!(model.omits("top_k"));
        assert!(!model.omits("max_tokens"));
    }

    #[test]
    fn ignores_informational_fields() {
        let catalog = ModelCatalog::from_json_bytes(CATALOG.as_bytes()).unwrap();
        assert_eq!(catalog.lookup("openai", "balanced").unwrap().id, "gpt-5.6-terra");
    }

    #[test]
    fn empty_omit_params_means_send_everything() {
        let catalog = ModelCatalog::from_json_bytes(CATALOG.as_bytes()).unwrap();
        let model = catalog.lookup("openai", "smartest").unwrap();
        assert!(model.omit_params.is_empty());
        assert!(!model.omits("temperature"));
    }

    #[test]
    fn null_and_blank_tiers_are_dropped_so_the_caller_falls_back() {
        let catalog = ModelCatalog::from_json_bytes(CATALOG.as_bytes()).unwrap();
        assert_eq!(catalog.lookup("anthropic", "fast").unwrap().id, "claude-haiku-4-5");
        assert!(catalog.lookup("anthropic", "balanced").is_none());
        assert!(catalog.lookup("anthropic", "smartest").is_none());
    }

    #[test]
    fn unknown_provider_or_tier_misses() {
        let catalog = ModelCatalog::from_json_bytes(CATALOG.as_bytes()).unwrap();
        assert!(catalog.lookup("google", "fast").is_none());
        assert!(catalog.lookup("openai", "cheapest").is_none());
    }

    #[test]
    fn rejects_a_future_schema_version() {
        let json = CATALOG.replace("\"schema_version\": 1", "\"schema_version\": 2");
        let err = ModelCatalog::from_json_bytes(json.as_bytes()).unwrap_err();
        assert!(matches!(
            err,
            ModelCatalogError::UnsupportedVersion { found: 2, supported: 1 }
        ));
    }

    #[test]
    fn rejects_malformed_json() {
        let err = ModelCatalog::from_json_bytes(b"{not json").unwrap_err();
        assert!(matches!(err, ModelCatalogError::Parse(_)));
    }

    #[test]
    fn rejects_an_oversized_catalog() {
        let padding = " ".repeat(MAX_CATALOG_BYTES + 1);
        let err = ModelCatalog::from_json_bytes(padding.as_bytes()).unwrap_err();
        assert!(matches!(
            err,
            ModelCatalogError::TooLarge { max: MAX_CATALOG_BYTES, .. }
        ));
    }

    #[test]
    fn a_catalog_with_no_usable_tiers_reports_empty() {
        let json = r#"{"schema_version": 1, "providers": {"openai": {"fast": null}}}"#;
        let catalog = ModelCatalog::from_json_bytes(json.as_bytes()).unwrap();
        assert!(catalog.is_empty());
        assert!(catalog.lookup("openai", "fast").is_none());
    }

    #[test]
    fn parses_the_real_registry_catalog() {
        // Byte-for-byte copy of what tools/build-models.sh produced on
        // 2026-08-05, trimmed to one provider. Guards against the generator
        // and the parser drifting apart.
        let json = r#"{
          "schema_version": 1,
          "generated_at": "2026-08-05T11:58:48Z",
          "source": "https://models.dev/api.json",
          "include_preview": false,
          "providers": {
            "anthropic": {
              "fast": {
                "id": "claude-haiku-4-5",
                "name": "Claude Haiku 4.5",
                "family": "claude-haiku",
                "release_date": "2025-10-15",
                "last_updated": "2025-10-15",
                "preview": false,
                "omit_params": [],
                "pricing_per_million_tokens": {"input": 1, "output": 5},
                "context_tokens": 200000,
                "selection_method": "semantic-name",
                "confidence": "high"
              },
              "smartest": {
                "id": "claude-opus-5",
                "name": "Claude Opus 5",
                "family": "claude-opus",
                "release_date": "2026-07-24",
                "last_updated": "2026-07-24",
                "preview": false,
                "omit_params": ["temperature", "top_p", "top_k"],
                "pricing_per_million_tokens": {"input": 5, "output": 25},
                "context_tokens": 1000000,
                "selection_method": "semantic-name",
                "confidence": "high"
              }
            }
          }
        }"#;
        let catalog = ModelCatalog::from_json_bytes(json.as_bytes()).unwrap();
        assert_eq!(catalog.lookup("anthropic", "fast").unwrap().id, "claude-haiku-4-5");
        assert!(!catalog.lookup("anthropic", "fast").unwrap().omits("temperature"));
        assert!(catalog.lookup("anthropic", "smartest").unwrap().omits("temperature"));
    }
}
