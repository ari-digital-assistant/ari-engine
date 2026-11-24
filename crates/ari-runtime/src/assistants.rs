//! The assistants a user can choose from, and which one is active. The
//! built-in on-device assistant is always there; community assistants come
//! from installed skills with `type: assistant`. Moved from `ari-ffi`'s
//! `assistant_registry.rs`.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use ari_engine::{ActiveAssistant, NamedAssistantBinding};
use ari_skill_loader::{AssistantEntry, AssistantManifest, AssistantProvider, LoadOptions, Privacy, Skillfile, load_skill_directory_with};

use crate::Runtime;
use crate::settings::{SettingField, SettingsStore, setting_fields};

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
    let sf = Skillfile::parse(BUILTIN_ASSISTANT_SKILL_MD, None).expect("built-in assistant SKILL.md must parse");
    let ari = sf.ari_extension.expect("built-in assistant must have ari extension");
    let assistant = ari.assistant.expect("built-in assistant must have assistant block");
    (ari.id, sf.name, sf.description, assistant, sf.body)
}

/// One assistant on offer.
#[derive(Debug, Clone, PartialEq)]
pub struct Assistant {
    pub id: String,
    pub name: String,
    pub description: String,
    /// `builtin` or `api`.
    pub provider: String,
    /// `local` or `cloud`.
    pub privacy: String,
    pub body: String,
}

pub struct AssistantRegistry {
    builtin: (String, String, String, AssistantManifest, String),
    community: Mutex<Vec<AssistantEntry>>,
    /// Shared with the skill registry and, through [`Self::apply_to`], with
    /// the engine's calls to a cloud assistant, so a setting written through
    /// any of them is what every reader sees.
    settings: Arc<SettingsStore>,
    /// The active assistant's id, if any.
    active_id: Mutex<Option<String>>,
    skill_store_dir: PathBuf,
    options: LoadOptions,
}

impl AssistantRegistry {
    /// The assistants installed under `skill_store_dir`, loaded with
    /// `options` (see [`crate::load_options`]), and the built-in one.
    pub fn new(skill_store_dir: PathBuf, options: LoadOptions, settings: Arc<SettingsStore>) -> Self {
        let registry = Self {
            builtin: parse_builtin_assistant(),
            community: Mutex::new(Vec::new()),
            settings,
            active_id: Mutex::new(None),
            skill_store_dir,
            options,
        };
        registry.reload_community_assistants();
        registry
    }

    /// Every assistant on offer, the built-in one first.
    pub fn list_assistants(&self) -> Vec<Assistant> {
        let (id, name, description, manifest, body) = &self.builtin;
        let mut out = vec![Assistant {
            id: id.clone(),
            name: name.clone(),
            description: description.clone(),
            provider: provider_str(manifest.provider).to_string(),
            privacy: privacy_str(manifest.privacy).to_string(),
            body: body.clone(),
        }];
        let community = self.community.lock().expect("community lock poisoned");
        out.extend(community.iter().map(|entry| Assistant {
            id: entry.id.clone(),
            name: entry.name.clone(),
            description: entry.description.clone(),
            provider: provider_str(entry.manifest.provider).to_string(),
            privacy: privacy_str(entry.manifest.privacy).to_string(),
            body: entry.body.clone(),
        }));
        out
    }

    pub fn active_assistant(&self) -> Option<String> {
        self.active_id.lock().expect("active_id lock poisoned").clone()
    }

    /// Choose the active assistant, or none. [`Self::apply_to`] hands the
    /// choice to the engine.
    pub fn set_active_assistant(&self, id: Option<String>) {
        *self.active_id.lock().expect("active_id lock poisoned") = id;
    }

    /// Hand the active assistant, and the assistants addressable by name, to
    /// `runtime`'s engine. Needed after every choice and after every reload of
    /// the skills. The built-in assistant takes `model_tier` from the settings
    /// so the engine can gate consulting it by tier; with no tier, or one it
    /// can't read (a fresh install before a model was picked), there is no
    /// active assistant rather than a silent default that would hide the
    /// misconfiguration.
    pub fn apply_to(&self, runtime: &Runtime) {
        let active_id = self.active_assistant();

        let assistant = match active_id {
            None => None,
            // Without the `llm` feature there is no on-device LLM to be
            // built-in *to*, and `ActiveAssistant::Builtin` does not exist.
            // Dropping the arm lets the built-in id fall through to the
            // community lookup below, which won't find it and yields
            // `None` — the same outcome as an unparseable tier.
            #[cfg(feature = "llm")]
            Some(ref id) if id == &self.builtin.0 => self
                .settings
                .get_value(id, "model_tier")
                .as_deref()
                .and_then(ari_llm::BuiltinTier::parse)
                .map(|tier| ActiveAssistant::Builtin { tier }),
            Some(ref id) => {
                let community = self.community.lock().expect("community lock poisoned");
                community.iter().find(|e| &e.id == id).and_then(|entry| {
                    entry.manifest.api.as_ref().map(|api_config| ActiveAssistant::Api {
                        skill_id: entry.id.clone(),
                        config: api_config.clone(),
                        config_store: self.settings.clone(),
                    })
                })
            }
        };

        let named = self.build_named_bindings();

        let mut engine = runtime.engine();
        engine.set_active_assistant(assistant);
        engine.set_named_assistants(named);
    }

    /// An assistant's settings, with their current values.
    pub fn get_assistant_config(&self, id: &str) -> Vec<SettingField> {
        match self.find_manifest(id) {
            Some(manifest) => setting_fields(id, &manifest.config, &self.settings),
            None => Vec::new(),
        }
    }

    /// Rescan the skill store for community assistants, after an install,
    /// update or uninstall.
    pub fn reload_community_assistants(&self) {
        let community = match load_skill_directory_with(&self.skill_store_dir, &self.options) {
            Ok(report) => report.assistants,
            Err(_) => Vec::new(),
        };
        *self.community.lock().expect("community lock poisoned") = community;
    }

    fn find_manifest(&self, id: &str) -> Option<AssistantManifest> {
        if id == self.builtin.0 {
            return Some(self.builtin.3.clone());
        }
        let community = self.community.lock().expect("community lock poisoned");
        community.iter().find(|e| e.id == id).map(|e| e.manifest.clone())
    }

    /// Build the list of name-addressable assistants pushed to the
    /// engine. Built-in is intentionally excluded — it has no aliases
    /// (it's already the typical default fallback). An entry only
    /// makes the list when its manifest declares non-empty aliases
    /// AND has a populated `api` config block (without one we couldn't
    /// dispatch even if we matched).
    fn build_named_bindings(&self) -> Vec<NamedAssistantBinding> {
        let community = self.community.lock().expect("community lock poisoned");
        community
            .iter()
            .filter_map(|entry| {
                if entry.manifest.aliases.is_empty() {
                    return None;
                }
                let api = entry.manifest.api.as_ref()?;
                Some(NamedAssistantBinding {
                    skill_id: entry.id.clone(),
                    aliases: entry.manifest.aliases.clone(),
                    config: api.clone(),
                    config_store: self.settings.clone(),
                })
            })
            .collect()
    }
}

fn provider_str(p: AssistantProvider) -> &'static str {
    match p {
        AssistantProvider::Builtin => "builtin",
        AssistantProvider::Api => "api",
    }
}

fn privacy_str(p: Privacy) -> &'static str {
    match p {
        Privacy::Local => "local",
        Privacy::Cloud => "cloud",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ari_skill_loader::ConfigFieldType;

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

    fn empty_registry() -> AssistantRegistry {
        let dir = std::env::temp_dir().join(format!("ari-runtime-assistants-{}", std::process::id()));
        AssistantRegistry::new(dir, LoadOptions::default(), Arc::new(SettingsStore::new()))
    }

    #[test]
    fn with_nothing_installed_only_the_built_in_is_on_offer() {
        let list = empty_registry().list_assistants();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].id, "dev.heyari.assistant.local");
        assert_eq!((list[0].provider.as_str(), list[0].privacy.as_str()), ("builtin", "local"));
    }

    #[test]
    fn the_built_in_assistant_s_settings_show_the_chosen_tier() {
        let registry = empty_registry();
        registry.settings.set("dev.heyari.assistant.local", "model_tier", "medium");
        let config = registry.get_assistant_config("dev.heyari.assistant.local");
        assert_eq!(config.len(), 1);
        assert_eq!(config[0].current_value.as_deref(), Some("medium"));
        assert_eq!(config[0].options.len(), 3);
        assert!(registry.get_assistant_config("nobody.here").is_empty());
    }
}
