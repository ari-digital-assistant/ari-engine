//! UniFFI wrapper around `ari_runtime::skills`, so Android (and any other
//! future frontend) can list installed skills, check for updates, and apply
//! them.
//!
//! This is intentionally a thin shim. All the policy lives in
//! `ari-skill-loader` and `ari-runtime`; this file just exposes stable,
//! uniffi-friendly types.
//!
//! Blocking semantics: both [`SkillRegistry::check_for_updates`] and
//! [`SkillRegistry::install_skill_update`] perform synchronous HTTPS
//! requests. Callers MUST invoke them off the main thread — on Android
//! that means a `CoroutineWorker` or `Dispatchers.IO`. UniFFI generates
//! blocking Kotlin methods so the thread discipline is the caller's
//! problem, not the generator's.

use ari_runtime::skills::{self, BrowseEntry, InstalledSkill, SkillManifest, SkillRegistryError, SkillUpdate};

use crate::assistant_registry::FfiConfigField;
use crate::settings_store::SkillSettingsStore;

use crate::android_load_options;
use std::path::PathBuf;
use std::sync::Arc;
use thiserror::Error;

/// One already-installed skill, flattened into a uniffi-safe record.
#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiInstalledSkill {
    pub id: String,
    pub version: String,
    /// Absolute path to the extracted skill directory on disk.
    pub install_dir: String,
    /// Locale codes the skill actually supports for end users. Auto-
    /// derived from on-disk file presence: a locale is supported iff
    /// `SKILL.{locale}.md` is present, AND (if the skill ships any
    /// `strings/` files at all) `strings/{locale}.json` is also
    /// present. The skill browser uses this for language badges and
    /// the install flow uses it for the locale-mismatch warning when
    /// the user's active locale isn't in the list.
    pub languages: Vec<String>,
}

/// One update the registry has for a skill that's already installed.
/// `name` + `description` come from the registry index so the UI can
/// show a changelog-adjacent blurb without reading the local SKILL.md.
#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiSkillUpdate {
    pub id: String,
    pub installed_version: String,
    pub available_version: String,
    pub name: String,
    pub description: String,
}

/// One row for the "Browse registry" screen — every skill the registry
/// carries, with an `installed` flag so the UI can mark rows for skills
/// the user already has on disk. `version` is the registry's version,
/// which may be ahead of the installed one; the UI can decide whether
/// to render that as "update available" or just "installed".
#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiBrowseEntry {
    pub id: String,
    pub version: String,
    pub name: String,
    pub description: String,
    /// "skill" or "assistant" — from the registry index. Lets the browse
    /// UI filter assistants without substring-matching id/name/description.
    pub skill_type: String,
    pub installed: bool,
    pub license: Option<String>,
    pub author: Option<String>,
    pub homepage: Option<String>,
    pub capabilities: Vec<String>,
    pub languages: Vec<String>,
    /// Per-locale display strings keyed by ISO 639-1 lowercase code
    /// (`"it"`, `"es"`, …). Comes straight from `index.json`'s
    /// `localizations` block, populated by the publish pipeline from
    /// every `SKILL.{locale}.md` variant. English is intentionally
    /// absent — the canonical `name` + `description` above already
    /// carry it. Empty for skills that haven't been migrated to the
    /// per-locale layout, which is also the case for older registry
    /// indexes (pre-Phase 11) where the field doesn't exist on disk.
    ///
    /// Frontends should pick the user's active-locale entry here and
    /// fall back to `name` / `description` when missing — see
    /// [`IndexEntry::display_for_locale`] in ari-skill-loader for the
    /// reference resolution logic.
    pub localizations: std::collections::HashMap<String, FfiLocalizedDisplay>,
}

/// Per-locale display strings for a registry entry. Mirrors
/// [`ari_skill_loader::registry::LocalizedDisplay`] across the FFI
/// boundary so frontends can render localised name + description on
/// browse rows without extra lookups.
#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiLocalizedDisplay {
    pub name: String,
    pub description: String,
}

/// Rich manifest details for an already-installed skill, parsed from the
/// on-disk SKILL.md. Used by the detail screen to show fields that don't
/// fit on a list row: author, homepage, capabilities the skill requires,
/// supported languages, and the full SKILL.md body (which typically
/// contains the human-readable long description / usage examples).
#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiSkillManifest {
    pub id: String,
    pub version: String,
    pub name: String,
    pub description: String,
    pub author: Option<String>,
    pub homepage: Option<String>,
    pub license: Option<String>,
    /// Capability names as they appear in the manifest (e.g. `http`,
    /// `storage_kv`). Stable strings — see
    /// [`ari_skill_loader::capability_name`].
    pub capabilities: Vec<String>,
    pub languages: Vec<String>,
    /// Full SKILL.md body after the frontmatter — markdown, verbatim.
    pub body: String,
    /// Example utterances the skill declares in `metadata.ari.examples`
    /// (the `text` of each entry, declaration order). Skills declare >=5.
    /// Surfaced so the frontend can offer real, skill-agnostic suggestion
    /// chips without inventing per-skill copy.
    pub examples: Vec<String>,
}

impl From<InstalledSkill> for FfiInstalledSkill {
    fn from(s: InstalledSkill) -> Self {
        Self {
            id: s.id,
            version: s.version,
            install_dir: s.install_dir.to_string_lossy().into_owned(),
            languages: s.languages,
        }
    }
}

impl From<SkillUpdate> for FfiSkillUpdate {
    fn from(u: SkillUpdate) -> Self {
        Self {
            id: u.id,
            installed_version: u.installed_version,
            available_version: u.available_version,
            name: u.name,
            description: u.description,
        }
    }
}

impl From<BrowseEntry> for FfiBrowseEntry {
    fn from(e: BrowseEntry) -> Self {
        Self {
            id: e.id,
            version: e.version,
            name: e.name,
            description: e.description,
            skill_type: e.skill_type,
            installed: e.installed,
            license: e.license,
            author: e.author,
            homepage: e.homepage,
            capabilities: e.capabilities,
            languages: e.languages,
            localizations: e
                .localizations
                .into_iter()
                .map(|(locale, d)| (locale, FfiLocalizedDisplay { name: d.name, description: d.description }))
                .collect(),
        }
    }
}

impl From<SkillManifest> for FfiSkillManifest {
    fn from(m: SkillManifest) -> Self {
        Self {
            id: m.id,
            version: m.version,
            name: m.name,
            description: m.description,
            author: m.author,
            homepage: m.homepage,
            license: m.license,
            capabilities: m.capabilities,
            languages: m.languages,
            body: m.body,
            examples: m.examples,
        }
    }
}

#[derive(Debug, Error, uniffi::Error)]
#[uniffi(flat_error)]
pub enum FfiRegistryError {
    #[error("registry: {message}")]
    Registry { message: String },

    #[error("skill store: {message}")]
    Store { message: String },

    #[error("skill not found in registry: {id}")]
    NotFound { id: String },

    #[error("skill not installed: {id}")]
    NotInstalled { id: String },

    #[error("manifest: {message}")]
    Manifest { message: String },

    /// The registry knows about this skill but doesn't carry a preview
    /// manifest sidecar — typically an index row generated before the
    /// sidecar pipeline was introduced. UI should fall back to the
    /// lightweight [`FfiBrowseEntry`] fields rather than showing an error.
    #[error("no preview manifest available for skill: {id}")]
    ManifestUnavailable { id: String },

    #[error("trust key: {message}")]
    TrustKey { message: String },

    /// Couldn't reach the registry / a transport-level failure. This is the
    /// only variant that genuinely warrants a "check your connection" hint.
    #[error("network error: {message}")]
    Network { message: String },

    /// The registry host answered, but with a non-success HTTP status
    /// (404 for a missing bundle, 5xx for an outage, …).
    #[error("registry returned status {status}")]
    BadStatus { status: u32 },

    /// The bundle exceeds the install size cap. Distinct from a network
    /// failure so the UI can tell the user the skill is too large rather
    /// than blaming their connection.
    #[error("skill bundle is too large to install (limit {limit_bytes} bytes)")]
    TooLarge { limit_bytes: u64 },

    /// The downloaded bundle failed verification — sha256 mismatch, bad
    /// signature, unsafe tar entry, or a corrupt archive. The skill was
    /// not installed.
    #[error("skill failed verification: {message}")]
    Integrity { message: String },
}

impl From<SkillRegistryError> for FfiRegistryError {
    fn from(e: SkillRegistryError) -> Self {
        match e {
            SkillRegistryError::Registry { message } => Self::Registry { message },
            SkillRegistryError::Store { message } => Self::Store { message },
            SkillRegistryError::NotFound { id } => Self::NotFound { id },
            SkillRegistryError::NotInstalled { id } => Self::NotInstalled { id },
            SkillRegistryError::Manifest { message } => Self::Manifest { message },
            SkillRegistryError::ManifestUnavailable { id } => Self::ManifestUnavailable { id },
            SkillRegistryError::TrustKey { message } => Self::TrustKey { message },
            SkillRegistryError::Network { message } => Self::Network { message },
            SkillRegistryError::BadStatus { status } => Self::BadStatus { status },
            SkillRegistryError::TooLarge { limit_bytes } => Self::TooLarge { limit_bytes },
            SkillRegistryError::Integrity { message } => Self::Integrity { message },
        }
    }
}

/// Thread-safe handle to a skill store. One instance per process, created
/// at app startup and injected wherever it's needed. `storage_dir` is
/// retained so install calls can rebuild [`LoadOptions`] with the same
/// `storage_kv` root the store was opened with.
#[derive(uniffi::Object)]
pub struct SkillRegistry {
    inner: skills::SkillRegistry,
    /// For `set_skill_setting`: the same store `inner` reads.
    settings_store: Arc<SkillSettingsStore>,
}

#[uniffi::export]
impl SkillRegistry {
    /// Open (or create) a skill store rooted at `skill_store_dir`, with
    /// per-skill `storage_kv` files living under `storage_dir`. Both
    /// paths should be inside the app's private files directory on
    /// Android (`context.filesDir`). `settings_store` is the shared
    /// in-memory mirror of per-skill settings — typically the same
    /// instance also handed to [`crate::AssistantRegistry::new`].
    #[uniffi::constructor]
    pub fn new(
        skill_store_dir: String,
        storage_dir: String,
        settings_store: Arc<SkillSettingsStore>,
    ) -> Result<Arc<Self>, FfiRegistryError> {
        let inner = skills::SkillRegistry::open(
            PathBuf::from(skill_store_dir),
            android_load_options(&storage_dir),
            settings_store.inner.clone(),
        )?;
        Ok(Arc::new(Self { inner, settings_store }))
    }

    /// Where the registry index is cached each time it's fetched. Hand it
    /// to [`crate::AriEngine::set_skill_suggestions_path`] at startup, so
    /// Ari can suggest store skills that aren't installed (ari-android#44).
    pub fn index_cache_path(&self) -> String {
        self.inner.index_cache_path().to_string_lossy().into_owned()
    }

    /// Every skill currently installed in the store, sorted by id so the
    /// Android list has a stable order without needing a second sort pass.
    pub fn list_installed(&self) -> Vec<FfiInstalledSkill> {
        self.inner.list_installed().into_iter().map(Into::into).collect()
    }

    /// The file an envelope's `asset:<path>` reference from skill `skill_id`
    /// names, or None when there's no such asset. See
    /// [`skills::SkillRegistry::asset_path`].
    pub fn asset_path(&self, skill_id: String, reference: String) -> Option<String> {
        let path = self.inner.asset_path(&skill_id, &reference)?;
        Some(path.to_string_lossy().into_owned())
    }

    /// Fetch the registry index and return the updates available for
    /// already-installed skills. An empty list means "up to date".
    ///
    /// Blocks on the network — callers must run this off the main thread.
    pub fn check_for_updates(&self) -> Result<Vec<FfiSkillUpdate>, FfiRegistryError> {
        Ok(self.inner.check_for_updates()?.into_iter().map(Into::into).collect())
    }

    /// Fetch, verify and cache the registry's tier→model catalog, returning the
    /// path it was written to. Hand that path to
    /// [`crate::AriEngine::load_model_catalog`] so cloud assistant skills
    /// resolve their `fast`/`balanced`/`smartest` setting to a current model ID.
    ///
    /// Cached next to the installed skills so a device that later goes offline
    /// keeps using the last known catalog instead of falling all the way back to
    /// the per-tier pins baked into each skill's manifest. `SkillStore::rescan`
    /// only looks at directories, so the file is inert there.
    ///
    /// Blocks on the network — callers must run this off the main thread.
    pub fn refresh_model_catalog(&self) -> Result<String, FfiRegistryError> {
        Ok(self.inner.refresh_model_catalog()?.to_string_lossy().into_owned())
    }

    /// Download and install the registry's current version of `id` over
    /// whatever's installed locally. Returns details about the newly
    /// installed version, or an error if the registry doesn't carry the
    /// skill, the hash/signature don't match, or extraction fails.
    ///
    /// Blocks on the network — callers must run this off the main thread.
    pub fn install_skill_update(
        &self,
        id: String,
    ) -> Result<FfiInstalledSkill, FfiRegistryError> {
        Ok(self.inner.install_skill_update(&id)?.into())
    }

    /// Fetch the registry index and return every entry as a [`FfiBrowseEntry`],
    /// with `installed` set for skills that already exist in the local store.
    /// Powers the "Browse registry" screen where the user picks skills to
    /// install for the first time.
    ///
    /// Blocks on the network — callers must run this off the main thread.
    pub fn browse_registry(&self) -> Result<Vec<FfiBrowseEntry>, FfiRegistryError> {
        Ok(self.inner.browse_registry()?.into_iter().map(Into::into).collect())
    }

    /// Download and install the registry's current version of `id`, even
    /// if the skill isn't already installed locally. This is the "Browse →
    /// tap install" path, complementing [`install_skill_update`] which is
    /// for already-installed skills.
    ///
    /// Blocks on the network — callers must run this off the main thread.
    pub fn install_skill_by_id(
        &self,
        id: String,
    ) -> Result<FfiInstalledSkill, FfiRegistryError> {
        Ok(self.inner.install_skill_by_id(&id)?.into())
    }

    /// Download the registry's preview manifest sidecar for `id` and
    /// return it as a rich [`FfiSkillManifest`] — same shape as
    /// [`Self::read_installed_manifest`], but sourced from the registry
    /// instead of the local store, so the Browse → detail view can
    /// render the full SKILL.md body *before* the user commits to an
    /// install.
    ///
    /// The sidecar is a verbatim copy of the skill's SKILL.md published
    /// alongside the signed bundle. It is **not** covered by the bundle
    /// signature — good enough for a read-only preview, not suitable for
    /// anything load-bearing. Install still goes through the full
    /// signature + sha256 pipeline.
    ///
    /// Errors:
    ///   * [`FfiRegistryError::NotFound`] — the id isn't in the registry.
    ///   * [`FfiRegistryError::ManifestUnavailable`] — the index row has
    ///     no sidecar (older index format); UI should fall back to the
    ///     lightweight browse entry.
    ///   * [`FfiRegistryError::Registry`] — network / HTTP failure.
    ///   * [`FfiRegistryError::Manifest`] — the sidecar doesn't parse as
    ///     a valid Skillfile.
    ///
    /// Blocks on the network — callers must run this off the main thread.
    pub fn fetch_manifest_preview(
        &self,
        id: String,
    ) -> Result<FfiSkillManifest, FfiRegistryError> {
        Ok(self.inner.fetch_manifest_preview(&id)?.into())
    }

    /// URLs of the registry's preview screenshots for `id`, in display
    /// order, for a frontend running on `platform` (`"android"`,
    /// `"linux"`, …).
    ///
    /// A skill with no shots for this platform falls back to whichever
    /// platform it *has* been photographed on — see
    /// [`ari_skill_loader::IndexEntry::screenshots_for_platform`]. An
    /// empty result means the skill ships none at all, and the UI should
    /// simply not render a gallery.
    ///
    /// Screenshots live in the registry rather than the bundle, so this
    /// works the same for installed and not-yet-installed skills. The
    /// frontend fetches the image bytes itself — every platform already
    /// has an image pipeline and a cache policy better than anything we'd
    /// bolt on here.
    ///
    /// Errors:
    ///   * [`FfiRegistryError::NotFound`] — the id isn't in the registry.
    ///   * [`FfiRegistryError::Registry`] — network / HTTP failure.
    ///
    /// Blocks on the network — callers must run this off the main thread.
    pub fn fetch_screenshot_urls(
        &self,
        id: String,
        platform: String,
    ) -> Result<Vec<String>, FfiRegistryError> {
        Ok(self.inner.fetch_screenshot_urls(&id, &platform)?)
    }

    /// Read the on-disk manifest for an already-installed skill and
    /// return the rich record the list/row view doesn't have room for —
    /// author, homepage, capabilities, supported languages, full body.
    ///
    /// `locale` is the user's active ISO 639-1 code (`"en"`, `"it"`,
    /// …). When the skill ships a `SKILL.{locale}.md`, fields that
    /// legitimately differ across locales (`name`, `description`,
    /// `body`) come from that variant — Italian users see Italian
    /// description on the installed-skill detail screen instead of
    /// the canonical English. Locales the skill hasn't translated
    /// fall back to canonical English via `LocalizedManifestSet::for_locale`.
    ///
    /// Returns [`FfiRegistryError::NotInstalled`] if `id` isn't in the
    /// store, or [`FfiRegistryError::Manifest`] if the manifest is
    /// missing or fails to parse.
    pub fn read_installed_manifest(
        &self,
        id: String,
        locale: String,
    ) -> Result<FfiSkillManifest, FfiRegistryError> {
        Ok(self.inner.read_installed_manifest(&id, &locale)?.into())
    }

    /// Read the user-configurable settings schema for an installed
    /// skill, with current values from the shared [`SkillSettingsStore`]
    /// merged in. Empty list if the skill declares no settings.
    ///
    /// Used by the per-skill detail page on Android to render an inline
    /// settings panel above the (collapsible) about/manifest section.
    /// Mirrors the shape of
    /// [`crate::AssistantRegistry::get_assistant_config`] so the same
    /// renderer composable works for both call sites — the only
    /// difference is the source of the schema (top-level
    /// `metadata.ari.settings` here, vs the assistant manifest there;
    /// after the migration both resolve to the same field, but we keep
    /// the dual entry points so neither caller has to know about the
    /// other's existence).
    ///
    /// Returns [`FfiRegistryError::NotInstalled`] if `id` isn't in the
    /// store, or [`FfiRegistryError::Manifest`] if the manifest fails
    /// to parse (shouldn't happen for skills we installed ourselves).
    ///
    /// `locale` follows the same rule as
    /// [`Self::read_installed_manifest`] — pass the user's active ISO
    /// 639-1 code so field labels (`Save reminders to` vs
    /// `Salva i promemoria in`) and select-option labels render in
    /// the user's language. Falls back to canonical English when the
    /// skill hasn't translated for the requested locale.
    pub fn get_skill_settings(
        &self,
        id: String,
        locale: String,
    ) -> Result<Vec<FfiConfigField>, FfiRegistryError> {
        Ok(self.inner.get_skill_settings(&id, &locale)?.into_iter().map(Into::into).collect())
    }

    /// Write a single setting value to the shared store. Equivalent to
    /// calling [`SkillSettingsStore::set_value`] directly — kept on this
    /// struct so the per-skill settings UI can flow through one
    /// dependency.
    pub fn set_skill_setting(&self, skill_id: String, key: String, value: String) {
        self.settings_store.set_value(skill_id, key, value);
    }

    /// Remove an installed skill from disk and wipe its `storage_kv`
    /// state. Returns [`FfiRegistryError::NotInstalled`] if `id` isn't in
    /// the local store. The caller should invoke
    /// [`AriEngine::reload_community_skills`] afterwards so the next
    /// `process_input` sees the updated skill set.
    pub fn uninstall_skill_by_id(&self, id: String) -> Result<(), FfiRegistryError> {
        Ok(self.inner.uninstall_skill_by_id(&id)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_skill_setting_round_trips_through_shared_store() {
        // Verify the writes done via SkillRegistry are visible via the
        // shared SkillSettingsStore — that's the contract that lets
        // assistant config and per-skill settings co-exist on the same
        // backing map.
        let base = std::env::temp_dir().join(format!("ari-ffi-registry-test-setting-{}", std::process::id()));
        let store = SkillSettingsStore::new();
        let reg = SkillRegistry::new(
            base.join("skills").to_string_lossy().into_owned(),
            base.join("storage").to_string_lossy().into_owned(),
            store.clone(),
        )
        .unwrap();
        reg.set_skill_setting(
            "dev.heyari.reminder".into(),
            "default_calendar".into(),
            "Personal".into(),
        );
        assert_eq!(
            store
                .inner
                .get_value("dev.heyari.reminder", "default_calendar")
                .as_deref(),
            Some("Personal"),
        );
        let _ = std::fs::remove_dir_all(&base);
    }
}
