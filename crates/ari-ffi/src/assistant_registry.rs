//! UniFFI exports for the assistant skill system.
//!
//! Surfaces the list of available assistant providers (built-in + community),
//! the active selection, and per-assistant config management. The built-in
//! local LLM assistant is always present; community assistants come from
//! installed SKILL.md manifests with `type: assistant`.

use ari_engine::ActiveAssistant;
use ari_skill_loader::manifest::{
    AssistantManifest, AssistantProvider, ConfigFieldType, Privacy, Skillfile,
};
use ari_skill_loader::{AssistantEntry, load_skill_directory_with};

use crate::android_load_options;
use crate::settings_store::SkillSettingsStore;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

// ── Built-in local LLM assistant manifest ──────────────────────────────

const BUILTIN_ASSISTANT_SKILL_MD: &str = r#"---
name: local-llm
description: >
  Ari will use local AI models to understand your commands. Requires no
  internet connection, but may have limited capabilities compared to cloud
  assistants, depending on model chosen.
metadata:
  ari:
    id: dev.heyari.assistant.local
    version: "0.1.0"
    type: assistant
    author: Ari Project
    homepage: https://github.com/ari-digital-assistant/ari
    engine: ">=0.3"
    languages: [en]
    settings:
      - key: model_tier
        label: Model
        type: select
        required: true
        options:
          - value: small
            label: "Gemma 3 1B (~769 MB)"
            download_url: "https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf"
            download_bytes: 806354944
          - value: medium
            label: "Gemma 4 E2B (~3.1 GB)"
            download_url: "https://huggingface.co/unsloth/gemma-4-e2b-it-GGUF/resolve/main/gemma-4-e2b-it-Q4_K_M.gguf"
            download_bytes: 3326083072
          - value: large
            label: "Gemma 4 E4B (~5.0 GB)"
            download_url: "https://huggingface.co/unsloth/gemma-4-e4b-it-GGUF/resolve/main/gemma-4-e4b-it-Q4_K_M.gguf"
            download_bytes: 5368709120
    assistant:
      provider: builtin
      privacy: local
---
Ari will use local AI models to understand your commands. Requires no
internet connection, but may have limited capabilities compared to cloud
assistants, depending on model chosen.
"#;

fn parse_builtin_assistant() -> (String, String, String, AssistantManifest, String) {
    let sf = Skillfile::parse(BUILTIN_ASSISTANT_SKILL_MD, None)
        .expect("built-in assistant SKILL.md must parse");
    let ari = sf.ari_extension.expect("built-in assistant must have ari extension");
    let assistant = ari.assistant.expect("built-in assistant must have assistant block");
    (ari.id, sf.name, sf.description, assistant, sf.body)
}

// ── FFI types ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiAssistantEntry {
    pub id: String,
    pub name: String,
    pub description: String,
    pub provider: String,
    pub privacy: String,
    pub body: String,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiConfigField {
    pub key: String,
    pub label: String,
    pub field_type: String,
    pub required: bool,
    pub default_value: Option<String>,
    pub current_value: Option<String>,
    pub options: Vec<FfiSelectOption>,
    /// Optional visibility gate — when non-null, the field should
    /// only render if the field identified by [`show_when_key`] has a
    /// current-or-default value matching one of [`show_when_equals`].
    /// Null means "always visible". Flat-pair surface because uniffi
    /// nested optional records are more awkward to consume than a
    /// flag + a list.
    pub show_when_key: Option<String>,
    pub show_when_equals: Vec<String>,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiSelectOption {
    pub value: String,
    pub label: String,
    pub download_url: Option<String>,
    pub download_bytes: Option<u64>,
}

// ── AssistantRegistry ─────────────────────────────────────────────────

#[derive(uniffi::Object)]
pub struct AssistantRegistry {
    /// Built-in assistant (always present).
    builtin: (String, String, String, AssistantManifest, String),
    /// Community assistants loaded from the skill store.
    community: Mutex<Vec<AssistantEntry>>,
    /// Process-wide settings store shared with [`crate::SkillRegistry`]
    /// and (via [`apply_to_engine`]) with the engine's runtime API call
    /// path. Constructed once in Android DI and injected into both
    /// registries so writes from any path are immediately visible to
    /// every reader.
    settings_store: Arc<SkillSettingsStore>,
    /// Which assistant is active (skill ID or None).
    active_id: Mutex<Option<String>>,
    /// Paths for rescanning community assistants.
    skill_store_dir: String,
    storage_dir: String,
}

#[uniffi::export]
impl AssistantRegistry {
    #[uniffi::constructor]
    pub fn new(
        skill_store_dir: String,
        storage_dir: String,
        settings_store: Arc<SkillSettingsStore>,
    ) -> Arc<Self> {
        let builtin = parse_builtin_assistant();

        let options = android_load_options(&storage_dir);
        let community = match load_skill_directory_with(
            &PathBuf::from(&skill_store_dir),
            &options,
        ) {
            Ok(report) => report.assistants,
            Err(_) => Vec::new(),
        };

        Arc::new(Self {
            builtin,
            community: Mutex::new(community),
            settings_store,
            active_id: Mutex::new(None),
            skill_store_dir,
            storage_dir,
        })
    }

    /// List all available assistant providers (built-in first, then community).
    pub fn list_assistants(&self) -> Vec<FfiAssistantEntry> {
        let mut out = Vec::new();

        // Built-in is always first.
        let (ref id, ref name, ref desc, ref manifest, ref body) = self.builtin;
        out.push(FfiAssistantEntry {
            id: id.clone(),
            name: name.clone(),
            description: desc.clone(),
            provider: provider_str(manifest.provider),
            privacy: privacy_str(manifest.privacy),
            body: body.clone(),
        });

        // Community assistants.
        let community = self.community.lock().expect("community lock poisoned");
        for entry in community.iter() {
            out.push(FfiAssistantEntry {
                id: entry.id.clone(),
                name: entry.name.clone(),
                description: entry.description.clone(),
                provider: provider_str(entry.manifest.provider),
                privacy: privacy_str(entry.manifest.privacy),
                body: entry.body.clone(),
            });
        }

        out
    }

    /// Get the ID of the currently active assistant, or null if none.
    pub fn get_active_assistant(&self) -> Option<String> {
        self.active_id
            .lock()
            .expect("active_id lock poisoned")
            .clone()
    }

    /// Set the active assistant by ID. Pass null to deactivate.
    /// Returns the `ActiveAssistant` enum value the engine needs,
    /// but since we can't pass that across UniFFI directly, the
    /// caller should use `apply_to_engine` instead.
    pub fn set_active_assistant(&self, id: Option<String>) {
        *self.active_id.lock().expect("active_id lock poisoned") = id;
    }

    /// Apply the current active assistant selection to the engine.
    /// Must be called after `set_active_assistant` and whenever the
    /// engine is rebuilt (e.g. after `reload_community_skills`). For
    /// the built-in assistant, reads `model_tier` from the settings
    /// store and threads it into [`ActiveAssistant::Builtin`] so Layer
    /// C can gate consultation by tier. If `model_tier` is missing or
    /// unparseable (fresh install before the user has picked a model),
    /// the active assistant is set to `None` rather than silent-defaulting,
    /// to avoid masking misconfiguration.
    pub fn apply_to_engine(&self, engine: &crate::AriEngine) {
        let active_id = self.active_id.lock().expect("active_id lock poisoned").clone();

        let assistant = match active_id {
            None => None,
            Some(ref id) if id == &self.builtin.0 => {
                let raw_tier = self.settings_store.inner.get_value(id, "model_tier");
                match raw_tier.as_deref().and_then(ari_llm::BuiltinTier::parse) {
                    Some(tier) => Some(ActiveAssistant::Builtin { tier }),
                    None => None,
                }
            }
            Some(ref id) => {
                let community = self.community.lock().expect("community lock poisoned");
                community
                    .iter()
                    .find(|e| &e.id == id)
                    .and_then(|entry| {
                        entry.manifest.api.as_ref().map(|api_config| {
                            ActiveAssistant::Api {
                                skill_id: entry.id.clone(),
                                config: api_config.clone(),
                                config_store: self.settings_store.as_config_store(),
                            }
                        })
                    })
            }
        };

        let mut engine_inner = engine.inner.lock().expect("engine mutex poisoned");
        engine_inner.set_active_assistant(assistant);
    }

    /// Get the config schema for an assistant, with current values filled in.
    pub fn get_assistant_config(&self, id: String) -> Vec<FfiConfigField> {
        let manifest = self.find_manifest(&id);
        let Some(manifest) = manifest else {
            return Vec::new();
        };

        manifest
            .config
            .iter()
            .map(|field| {
                let current_value = if matches!(field.field_type, ConfigFieldType::Secret) {
                    // Never return secret values across FFI.
                    if self.settings_store.inner.get_value(&id, &field.key).is_some() {
                        Some("••••••••".to_string())
                    } else {
                        None
                    }
                } else {
                    self.settings_store.inner.get_value(&id, &field.key)
                };

                FfiConfigField {
                    key: field.key.clone(),
                    label: field.label.clone(),
                    field_type: field_type_str(&field.field_type),
                    required: field.required,
                    default_value: field.default.clone(),
                    current_value,
                    options: match &field.field_type {
                        ConfigFieldType::Select { options } => options
                            .iter()
                            .map(|o| FfiSelectOption {
                                value: o.value.clone(),
                                label: o.label.clone(),
                                download_url: o.download_url.clone(),
                                download_bytes: o.download_bytes,
                            })
                            .collect(),
                        _ => Vec::new(),
                    },
                    show_when_key: field.show_when.as_ref().map(|s| s.key.clone()),
                    show_when_equals: field
                        .show_when
                        .as_ref()
                        .map(|s| s.equals.clone())
                        .unwrap_or_default(),
                }
            })
            .collect()
    }

    /// Set a config value for an assistant skill. Equivalent to calling
    /// [`SkillSettingsStore::set_value`] directly — kept on this struct
    /// so the existing assistant settings UI doesn't need rewiring just
    /// to read its own writes.
    pub fn set_assistant_config_value(
        &self,
        skill_id: String,
        key: String,
        value: String,
    ) {
        self.settings_store.set_value(skill_id, key, value);
    }

    /// Rescan the skill store for community assistant skills (call after
    /// install/uninstall).
    pub fn reload_community_assistants(&self) {
        let options = android_load_options(&self.storage_dir);
        let new_community = match load_skill_directory_with(
            &PathBuf::from(&self.skill_store_dir),
            &options,
        ) {
            Ok(report) => report.assistants,
            Err(_) => Vec::new(),
        };
        *self.community.lock().expect("community lock poisoned") = new_community;
    }
}

impl AssistantRegistry {
    fn find_manifest(&self, id: &str) -> Option<AssistantManifest> {
        if id == self.builtin.0 {
            return Some(self.builtin.3.clone());
        }
        let community = self.community.lock().expect("community lock poisoned");
        community
            .iter()
            .find(|e| e.id == id)
            .map(|e| e.manifest.clone())
    }
}

fn provider_str(p: AssistantProvider) -> String {
    match p {
        AssistantProvider::Builtin => "builtin".to_string(),
        AssistantProvider::Api => "api".to_string(),
    }
}

fn privacy_str(p: Privacy) -> String {
    match p {
        Privacy::Local => "local".to_string(),
        Privacy::Cloud => "cloud".to_string(),
    }
}

fn field_type_str(ft: &ConfigFieldType) -> String {
    match ft {
        ConfigFieldType::Text => "text".to_string(),
        ConfigFieldType::Secret => "secret".to_string(),
        ConfigFieldType::Select { .. } => "select".to_string(),
        ConfigFieldType::DeviceCalendar => "device_calendar".to_string(),
        ConfigFieldType::DeviceTaskList => "device_task_list".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_assistant_parses_successfully() {
        let (id, name, desc, manifest, body) = parse_builtin_assistant();
        assert_eq!(id, "dev.heyari.assistant.local");
        assert_eq!(name, "local-llm");
        assert!(!desc.is_empty());
        assert_eq!(manifest.provider, AssistantProvider::Builtin);
        assert_eq!(manifest.privacy, Privacy::Local);
        assert!(manifest.api.is_none());
        assert_eq!(manifest.config.len(), 1);
        assert_eq!(manifest.config[0].key, "model_tier");
        match &manifest.config[0].field_type {
            ConfigFieldType::Select { options } => {
                assert_eq!(options.len(), 3);
                assert_eq!(options[0].value, "small");
                assert!(options[0].download_url.is_some());
                assert!(options[0].download_bytes.is_some());
            }
            _ => panic!("expected select config field"),
        }
        assert!(!body.is_empty());
    }

    #[test]
    fn settings_store_round_trip_via_assistant_registry_path() {
        let store = SkillSettingsStore::new();
        assert!(store.inner.get_value("a", "b").is_none());
        store.set_value("a".into(), "b".into(), "val".into());
        assert_eq!(store.inner.get_value("a", "b").as_deref(), Some("val"));
        store.set_value("a".into(), "b".into(), "updated".into());
        assert_eq!(store.inner.get_value("a", "b").as_deref(), Some("updated"));
    }
}
