//! UniFFI exports for the assistant skill system, wrapping
//! `ari_runtime::assistants`.
//!
//! Surfaces the list of available assistant providers (built-in + community),
//! the active selection, and per-assistant config management. The built-in
//! local LLM assistant is always present; community assistants come from
//! installed SKILL.md manifests with `type: assistant`.

use ari_runtime::assistants::{self, Assistant};
use ari_runtime::settings::{SettingField, SettingOption};

use crate::android_load_options;
use crate::settings_store::SkillSettingsStore;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiAssistantEntry {
    pub id: String,
    pub name: String,
    pub description: String,
    pub provider: String,
    pub privacy: String,
    pub body: String,
}

impl From<Assistant> for FfiAssistantEntry {
    fn from(a: Assistant) -> Self {
        Self { id: a.id, name: a.name, description: a.description, provider: a.provider, privacy: a.privacy, body: a.body }
    }
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
    /// Other field keys whose committed values this field's
    /// `dynamic_select`/`validate` query depends on. A change to any of
    /// them (when all are non-empty) re-runs the query.
    pub depends_on: Vec<String>,
    /// When true, the frontend runs the skill's `settings_query` for this
    /// field on `depends_on` change and shows a validity result.
    pub validate: bool,
    /// Optional guidance shown near the field in Settings. Null = no help.
    pub help_text: Option<String>,
    /// Optional disclosure-group label. Fields sharing a non-null value are
    /// rendered collapsed under one expander carrying this label. Null =
    /// always shown at top level.
    pub collapsed_group: Option<String>,
    /// Optional soft-keyboard hint for text entry — currently only `"url"`.
    /// Null means an ordinary text keyboard. A frontend that doesn't know a
    /// value here must fall back to plain text rather than hide the field.
    pub keyboard: Option<String>,
}

impl From<SettingField> for FfiConfigField {
    fn from(f: SettingField) -> Self {
        Self {
            key: f.key,
            label: f.label,
            field_type: f.field_type,
            required: f.required,
            default_value: f.default_value,
            current_value: f.current_value,
            options: f.options.into_iter().map(Into::into).collect(),
            show_when_key: f.show_when_key,
            show_when_equals: f.show_when_equals,
            depends_on: f.depends_on,
            validate: f.validate,
            help_text: f.help_text,
            collapsed_group: f.collapsed_group,
            keyboard: f.keyboard,
        }
    }
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiSelectOption {
    pub value: String,
    pub label: String,
    pub download_url: Option<String>,
    pub download_bytes: Option<u64>,
}

impl From<SettingOption> for FfiSelectOption {
    fn from(o: SettingOption) -> Self {
        Self { value: o.value, label: o.label, download_url: o.download_url, download_bytes: o.download_bytes }
    }
}

#[derive(uniffi::Object)]
pub struct AssistantRegistry {
    inner: assistants::AssistantRegistry,
    /// For `set_assistant_config_value`: the same store `inner` reads.
    settings_store: Arc<SkillSettingsStore>,
}

#[uniffi::export]
impl AssistantRegistry {
    #[uniffi::constructor]
    pub fn new(
        skill_store_dir: String,
        storage_dir: String,
        settings_store: Arc<SkillSettingsStore>,
    ) -> Arc<Self> {
        let inner = assistants::AssistantRegistry::new(
            PathBuf::from(skill_store_dir),
            android_load_options(&storage_dir),
            settings_store.inner.clone(),
        );
        Arc::new(Self { inner, settings_store })
    }

    /// List all available assistant providers (built-in first, then community).
    pub fn list_assistants(&self) -> Vec<FfiAssistantEntry> {
        self.inner.list_assistants().into_iter().map(Into::into).collect()
    }

    /// Get the ID of the currently active assistant, or null if none.
    pub fn get_active_assistant(&self) -> Option<String> {
        self.inner.active_assistant()
    }

    /// Set the active assistant by ID. Pass null to deactivate.
    /// Returns the `ActiveAssistant` enum value the engine needs,
    /// but since we can't pass that across UniFFI directly, the
    /// caller should use `apply_to_engine` instead.
    pub fn set_active_assistant(&self, id: Option<String>) {
        self.inner.set_active_assistant(id);
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
        self.inner.apply_to(&engine.runtime);
    }

    /// Rescan the skill store for community assistants AND push the
    /// active+named assistant state to the engine in one call. This is
    /// the install/update/uninstall entry point — calling
    /// `reload_community_assistants` alone wouldn't propagate the new
    /// alias list to the engine until the next `apply_to_engine`, so
    /// the frontend should always use this wrapper after a skill
    /// install completes.
    pub fn reload_and_apply(&self, engine: &crate::AriEngine) {
        self.reload_community_assistants();
        self.apply_to_engine(engine);
    }

    /// Get the config schema for an assistant, with current values filled in.
    pub fn get_assistant_config(&self, id: String) -> Vec<FfiConfigField> {
        self.inner.get_assistant_config(&id).into_iter().map(Into::into).collect()
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
        self.inner.reload_community_assistants();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
