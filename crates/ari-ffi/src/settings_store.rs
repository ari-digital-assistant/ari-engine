//! UniFFI handle on `ari_runtime`'s [`SettingsStore`], the process-wide store
//! for skill settings. Both [`crate::AssistantRegistry`] and
//! [`crate::SkillRegistry`] take an `Arc<SkillSettingsStore>` at
//! construction so that writes from either side land in the same
//! HashMap and the engine's runtime read path (via [`ConfigStore`])
//! sees them all.

use ari_runtime::settings::SettingsStore;
use ari_skill_loader::assistant::ConfigStore;
use std::sync::Arc;

/// UniFFI handle exposed to Android. Construct once at app startup and
/// inject into both registries; never construct two of these or the
/// in-memory state will drift apart.
#[derive(uniffi::Object)]
pub struct SkillSettingsStore {
    pub(crate) inner: Arc<SettingsStore>,
}

#[uniffi::export]
impl SkillSettingsStore {
    #[uniffi::constructor]
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            inner: Arc::new(SettingsStore::new()),
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
    /// The store as the engine's [`ConfigStore`], for [`crate::AriEngineBuilder`].
    pub(crate) fn as_config_store(&self) -> Arc<dyn ConfigStore> {
        self.inner.clone()
    }
}
