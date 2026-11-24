//! Skill settings: the values the engine reads, and the fields a settings
//! screen shows. Moved from `ari-ffi`'s `settings_store.rs` and the field
//! mapping its two registries each had.

use std::collections::HashMap;
use std::sync::RwLock;

use ari_skill_loader::assistant::ConfigStore;
use ari_skill_loader::{ConfigField, ConfigFieldType};

/// Every skill's setting values, in memory. One per process, shared by the
/// registries and handed to the engine as its [`ConfigStore`], so a value
/// written through any of them is what every reader sees.
///
/// Intentionally amnesiac: the frontend persists the values (Android to
/// DataStore, and secrets to EncryptedSharedPreferences) and writes each one
/// back in at process start.
#[derive(Default)]
pub struct SettingsStore {
    values: RwLock<HashMap<(String, String), String>>,
}

impl SettingsStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&self, skill_id: &str, key: &str, value: &str) {
        self.values
            .write()
            .expect("config store lock poisoned")
            .insert((skill_id.to_string(), key.to_string()), value.to_string());
    }

    pub fn get_value(&self, skill_id: &str, key: &str) -> Option<String> {
        self.values
            .read()
            .expect("config store lock poisoned")
            .get(&(skill_id.to_string(), key.to_string()))
            .cloned()
    }
}

impl ConfigStore for SettingsStore {
    fn get(&self, skill_id: &str, key: &str) -> Option<String> {
        self.get_value(skill_id, key)
    }
}

/// One settings field as a settings screen draws it, with its current value.
#[derive(Debug, Clone, PartialEq)]
pub struct SettingField {
    pub key: String,
    pub label: String,
    /// `text`, `secret`, `select`, `device_calendar`, `device_task_list`,
    /// `dynamic_select` or `action`.
    pub field_type: String,
    pub required: bool,
    pub default_value: Option<String>,
    pub current_value: Option<String>,
    pub options: Vec<SettingOption>,
    /// Optional visibility gate: when set, the field only renders if the field
    /// `show_when_key` names has a current-or-default value among
    /// `show_when_equals`.
    pub show_when_key: Option<String>,
    pub show_when_equals: Vec<String>,
    /// Other field keys whose committed values this field's
    /// `dynamic_select`/`validate` query depends on. A change to any of them
    /// (when all are non-empty) re-runs the query.
    pub depends_on: Vec<String>,
    /// The frontend runs the skill's `settings_query` for this field on a
    /// `depends_on` change and shows a validity result.
    pub validate: bool,
    pub help_text: Option<String>,
    /// Fields sharing a label are rendered collapsed under one expander
    /// carrying it.
    pub collapsed_group: Option<String>,
    /// A soft-keyboard hint for text entry, currently only `"url"`. A
    /// frontend that doesn't know a value must fall back to plain text rather
    /// than hide the field.
    pub keyboard: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SettingOption {
    pub value: String,
    pub label: String,
    pub download_url: Option<String>,
    pub download_bytes: Option<u64>,
}

/// `fields` as a settings screen draws them, with `skill_id`'s current values
/// from `store`.
pub fn setting_fields(skill_id: &str, fields: &[ConfigField], store: &SettingsStore) -> Vec<SettingField> {
    fields
        .iter()
        .map(|field| {
            let current_value = if matches!(field.field_type, ConfigFieldType::Secret) {
                // Never hand a secret's value out: the frontend already has it
                // in encrypted storage, so a set one is signalled with the
                // bullet placeholder and the UI masks the input.
                store.get_value(skill_id, &field.key).map(|_| "••••••••".to_string())
            } else {
                store.get_value(skill_id, &field.key)
            };
            SettingField {
                key: field.key.clone(),
                label: field.label.clone(),
                field_type: field_type_str(&field.field_type).to_string(),
                required: field.required,
                default_value: field.default.clone(),
                current_value,
                options: match &field.field_type {
                    ConfigFieldType::Select { options } => options
                        .iter()
                        .map(|o| SettingOption {
                            value: o.value.clone(),
                            label: o.label.clone(),
                            download_url: o.download_url.clone(),
                            download_bytes: o.download_bytes,
                        })
                        .collect(),
                    _ => Vec::new(),
                },
                show_when_key: field.show_when.as_ref().map(|s| s.key.clone()),
                show_when_equals: field.show_when.as_ref().map(|s| s.equals.clone()).unwrap_or_default(),
                depends_on: field.depends_on.clone(),
                validate: field.validate,
                help_text: field.help_text.clone(),
                collapsed_group: field.collapsed_group.clone(),
                keyboard: field.keyboard.map(|k| k.as_str().to_string()),
            }
        })
        .collect()
}

fn field_type_str(ft: &ConfigFieldType) -> &'static str {
    match ft {
        ConfigFieldType::Text => "text",
        ConfigFieldType::Secret => "secret",
        ConfigFieldType::Select { .. } => "select",
        ConfigFieldType::DeviceCalendar => "device_calendar",
        ConfigFieldType::DeviceTaskList => "device_task_list",
        ConfigFieldType::DynamicSelect => "dynamic_select",
        ConfigFieldType::Action => "action",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field(key: &str, field_type: ConfigFieldType) -> ConfigField {
        ConfigField {
            key: key.into(),
            label: key.to_uppercase(),
            field_type,
            required: false,
            default: None,
            show_when: None,
            validate: false,
            depends_on: Vec::new(),
            help_text: None,
            collapsed_group: None,
            keyboard: None,
        }
    }

    #[test]
    fn the_store_round_trips_a_value() {
        let store = SettingsStore::new();
        assert!(store.get_value("a", "b").is_none());
        store.set("a", "b", "val");
        assert_eq!(store.get_value("a", "b").as_deref(), Some("val"));
        store.set("a", "b", "updated");
        assert_eq!(store.get_value("a", "b").as_deref(), Some("updated"));
        assert_eq!(ConfigStore::get(&store, "a", "b").as_deref(), Some("updated"));
    }

    #[test]
    fn a_field_shows_its_current_value_and_a_secret_shows_bullets() {
        let store = SettingsStore::new();
        store.set("dev.heyari.ha", "base_url", "http://ha.local:8123");
        store.set("dev.heyari.ha", "token", "real-token");
        let fields = [
            field("base_url", ConfigFieldType::Text),
            field("token", ConfigFieldType::Secret),
            field("room", ConfigFieldType::Text),
        ];
        let shown = setting_fields("dev.heyari.ha", &fields, &store);
        assert_eq!(shown[0].current_value.as_deref(), Some("http://ha.local:8123"));
        assert_eq!(shown[1].field_type, "secret");
        assert_eq!(shown[1].current_value.as_deref(), Some("••••••••"));
        assert_eq!(shown[2].current_value, None);
    }

    #[test]
    fn field_type_str_covers_dynamic_select() {
        assert_eq!(field_type_str(&ConfigFieldType::DynamicSelect), "dynamic_select");
    }

    #[test]
    fn field_type_str_covers_action() {
        assert_eq!(field_type_str(&ConfigFieldType::Action), "action");
    }
}
