//! Process-wide store for skill settings, exposed across UniFFI as
//! [`SkillSettingsStore`]. Both [`crate::AssistantRegistry`] and
//! [`crate::SkillRegistry`] take an `Arc<SkillSettingsStore>` at
//! construction so that writes from either side land in the same
//! HashMap and the engine's runtime read path (via [`ConfigStore`])
//! sees them all.
//!
//! This is the in-memory mirror — Android persists the same values to
//! DataStore (non-secret) and EncryptedSharedPreferences (secrets),
//! and re-hydrates the store at process start by calling
//! [`SkillSettingsStore::set_value`] for each entry. Restart-safety
//! lives on the Android side; this struct is intentionally amnesiac.

use ari_skill_loader::assistant::ConfigStore;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Inner storage. Separated from the UniFFI Object wrapper so the
/// engine can hold an `Arc<dyn ConfigStore>` pointing at the same
/// HashMap.
pub(crate) struct InnerStore {
    inner: RwLock<HashMap<(String, String), String>>,
}

impl InnerStore {
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(HashMap::new()),
        }
    }

    pub fn set(&self, skill_id: &str, key: &str, value: &str) {
        self.inner
            .write()
            .expect("config store lock poisoned")
            .insert((skill_id.to_string(), key.to_string()), value.to_string());
    }

    pub fn get_value(&self, skill_id: &str, key: &str) -> Option<String> {
        self.inner
            .read()
            .expect("config store lock poisoned")
            .get(&(skill_id.to_string(), key.to_string()))
            .cloned()
    }
}

impl ConfigStore for InnerStore {
    fn get(&self, skill_id: &str, key: &str) -> Option<String> {
        self.get_value(skill_id, key)
    }
}

/// UniFFI handle exposed to Android. Construct once at app startup and
/// inject into both registries; never construct two of these or the
/// in-memory state will drift apart.
#[derive(uniffi::Object)]
pub struct SkillSettingsStore {
    pub(crate) inner: Arc<InnerStore>,
}

#[uniffi::export]
impl SkillSettingsStore {
    #[uniffi::constructor]
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            inner: Arc::new(InnerStore::new()),
        })
    }

    /// Write a single setting value into the shared store. Android calls
    /// this on every UI edit and during the startup hydration loop that
    /// reads from DataStore + EncryptedSharedPreferences.
    pub fn set_value(&self, skill_id: String, key: String, value: String) {
        self.inner.set(&skill_id, &key, &value);
    }
}

impl SkillSettingsStore {
    /// Borrow the inner Arc for use as a [`ConfigStore`] trait object.
    /// Used by `AssistantRegistry::apply_to_engine` to give the engine's
    /// API call path read access to the same map.
    pub(crate) fn as_config_store(&self) -> Arc<dyn ConfigStore> {
        self.inner.clone() as Arc<dyn ConfigStore>
    }
}
