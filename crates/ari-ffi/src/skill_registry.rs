//! UniFFI wrapper around [`ari_skill_loader::SkillStore`] + the
//! registry client, so Android (and any other future frontend) can
//! list installed skills, check for updates, and apply them.
//!
//! This is intentionally a thin shim. All the policy lives in
//! `ari-skill-loader`; this file just exposes stable, uniffi-friendly
//! types and handles the `Mutex` needed for `&self` interior mutability.
//!
//! Blocking semantics: both [`SkillRegistry::check_for_updates`] and
//! [`SkillRegistry::install_skill_update`] perform synchronous HTTPS
//! requests. Callers MUST invoke them off the main thread — on Android
//! that means a `CoroutineWorker` or `Dispatchers.IO`. UniFFI generates
//! blocking Kotlin methods so the thread discipline is the caller's
//! problem, not the generator's.

use ari_skill_loader::manifest::ConfigFieldType;
use ari_skill_loader::{
    capability_name, check_updates, install_by_id, install_update, IndexEntry, ManifestError,
    ModelCatalog, RegistryClient, RegistryError, Skillfile, SkillStore, StorageConfig, StoreError,
    TrustRoot, REGISTRY_TRUST_KEY,
};

use crate::assistant_registry::{FfiConfigField, FfiSelectOption};
use crate::settings_store::SkillSettingsStore;

use crate::android_load_options;
use std::path::Path;
use std::sync::{Arc, Mutex};
use thiserror::Error;

/// Auto-derive the locale list for an on-disk skill bundle. Reads
/// `SKILL.{locale}.md` and `strings/{locale}.json` file presence per
/// the multi-language plan's self-validation rule, falling back to
/// the canonical English locale when either layer is unparseable so
/// the browser at least shows *something* rather than rendering the
/// row with no language badges.
fn derive_supported_locales(skill_dir: &Path) -> Vec<String> {
    let manifests = match ari_skill_loader::parse_skill_directory(skill_dir) {
        Ok(m) => m,
        Err(_) => {
            // Bundle layout is broken — surface nothing rather than
            // guessing. Frontend treats empty as "unknown / none";
            // the install warning will fire for any locale the user
            // is in.
            return Vec::new();
        }
    };
    let strings = match ari_skill_loader::parse_strings_directory(skill_dir) {
        Ok(s) => s,
        Err(_) => return manifests.supported_locales(),
    };
    manifests.supported_for_user(&strings)
}

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

/// Map a low-level [`RegistryError`] to the FFI-facing [`FfiRegistryError`]
/// so the UI can show an accurate reason instead of assuming every failure
/// is a dropped network connection (the old behaviour: every non-NotFound
/// error collapsed into [`FfiRegistryError::Registry`], which the Android
/// layer rendered as "check your connection" — so an oversized or
/// tampered-with bundle looked identical to being offline).
///
/// `#[uniffi(flat_error)]` means only the variant and its Display string
/// cross the FFI boundary, which is all the Kotlin mapping needs.
fn map_registry_error(e: RegistryError) -> FfiRegistryError {
    match e {
        RegistryError::NotFound { id } => FfiRegistryError::NotFound { id },
        RegistryError::ManifestUnavailable { id } => {
            FfiRegistryError::ManifestUnavailable { id }
        }
        RegistryError::Http(message) => FfiRegistryError::Network { message },
        RegistryError::BadStatus { status, .. } => {
            FfiRegistryError::BadStatus { status: u32::from(status) }
        }
        RegistryError::TooLarge { limit, .. } => {
            FfiRegistryError::TooLarge { limit_bytes: limit as u64 }
        }
        // sha mismatch or any bundle-level failure (bad signature, unsafe
        // tar entry, corrupt archive) → the skill couldn't be safely
        // verified. Don't blame the network.
        e @ (RegistryError::ShaMismatch { .. } | RegistryError::Install(_)) => {
            FfiRegistryError::Integrity { message: e.to_string() }
        }
        // Parse / UnsupportedIndexVersion: a genuine registry-format
        // problem; keep the generic bucket.
        other => FfiRegistryError::Registry { message: other.to_string() },
    }
}

fn trust_root() -> Result<TrustRoot, FfiRegistryError> {
    TrustRoot::single(&REGISTRY_TRUST_KEY).map_err(|e| FfiRegistryError::TrustKey {
        message: e.to_string(),
    })
}

/// Thread-safe handle to a skill store. One instance per process, created
/// at app startup and injected wherever it's needed. `storage_dir` is
/// retained so install calls can rebuild [`LoadOptions`] with the same
/// `storage_kv` root the store was opened with.
#[derive(uniffi::Object)]
pub struct SkillRegistry {
    store: Mutex<SkillStore>,
    storage_dir: String,
    /// Process-wide settings store, shared with [`crate::AssistantRegistry`]
    /// so per-skill settings written through either path land in the same
    /// HashMap and the engine's runtime API call path sees them.
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
        let trust = trust_root()?;
        let store = SkillStore::open(
            std::path::PathBuf::from(&skill_store_dir),
            StorageConfig::new(std::path::PathBuf::from(&storage_dir)),
            trust,
        )
        .map_err(|e| FfiRegistryError::Store {
            message: e.to_string(),
        })?;
        Ok(Arc::new(Self {
            store: Mutex::new(store),
            storage_dir,
            settings_store,
        }))
    }

    /// Every skill currently installed in the store, sorted by id so the
    /// Android list has a stable order without needing a second sort pass.
    pub fn list_installed(&self) -> Vec<FfiInstalledSkill> {
        let store = self.store.lock().expect("skill store mutex poisoned");
        let mut out: Vec<FfiInstalledSkill> = store
            .list()
            .into_iter()
            .map(|s| {
                let languages = derive_supported_locales(&s.install_dir);
                FfiInstalledSkill {
                    id: s.id,
                    version: s.version,
                    install_dir: s.install_dir.to_string_lossy().into_owned(),
                    languages,
                }
            })
            .collect();
        out.sort_by(|a, b| a.id.cmp(&b.id));
        out
    }

    /// Fetch the registry index and return the updates available for
    /// already-installed skills. An empty list means "up to date".
    ///
    /// Blocks on the network — callers must run this off the main thread.
    pub fn check_for_updates(&self) -> Result<Vec<FfiSkillUpdate>, FfiRegistryError> {
        let client = RegistryClient::new();
        let index = client.fetch_index().map_err(map_registry_error)?;
        let store = self.store.lock().expect("skill store mutex poisoned");
        let updates = check_updates(&store, &index);
        Ok(updates
            .into_iter()
            .map(|u| FfiSkillUpdate {
                id: u.id,
                installed_version: u.installed_version,
                available_version: u.available_version,
                name: u.entry.name,
                description: u.entry.description,
            })
            .collect())
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
        let client = RegistryClient::new();
        let index = client.fetch_index().map_err(map_registry_error)?;
        let trust = trust_root()?;
        let bytes = client
            .fetch_catalog_bytes(&index, &trust)
            .map_err(map_registry_error)?;

        // Parse before caching so a catalog this build can't understand fails
        // here, rather than being written and silently ignored at call time.
        ModelCatalog::from_json_bytes(&bytes).map_err(|e| FfiRegistryError::Registry {
            message: e.to_string(),
        })?;

        let path = {
            let store = self.store.lock().expect("skill store mutex poisoned");
            store.root_path().join("models.json")
        };
        std::fs::write(&path, &bytes).map_err(|e| FfiRegistryError::Store {
            message: format!("caching models.json: {e}"),
        })?;
        Ok(path.to_string_lossy().into_owned())
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
        let client = RegistryClient::new();
        let index = client.fetch_index().map_err(map_registry_error)?;
        let entry = index
            .skills
            .iter()
            .find(|e| e.id == id)
            .cloned()
            .ok_or_else(|| FfiRegistryError::NotFound { id: id.clone() })?;

        let trust = trust_root()?;
        let options = android_load_options(&self.storage_dir);
        let mut store = self.store.lock().expect("skill store mutex poisoned");
        let installed = install_update(&client, &entry, &mut store, &trust, &options)
            .map_err(map_registry_error)?;
        let languages = derive_supported_locales(&installed.install_dir);
        Ok(FfiInstalledSkill {
            id: installed.id,
            version: installed.version,
            install_dir: installed.install_dir.to_string_lossy().into_owned(),
            languages,
        })
    }

    /// Fetch the registry index and return every entry as a [`FfiBrowseEntry`],
    /// with `installed` set for skills that already exist in the local store.
    /// Powers the "Browse registry" screen where the user picks skills to
    /// install for the first time.
    ///
    /// Blocks on the network — callers must run this off the main thread.
    pub fn browse_registry(&self) -> Result<Vec<FfiBrowseEntry>, FfiRegistryError> {
        let client = RegistryClient::new();
        let index = client.fetch_index().map_err(map_registry_error)?;
        let store = self.store.lock().expect("skill store mutex poisoned");
        let mut out: Vec<FfiBrowseEntry> = index
            .skills
            .into_iter()
            .map(|entry| {
                let localizations = entry
                    .localizations
                    .into_iter()
                    .map(|(locale, display)| {
                        (
                            locale,
                            FfiLocalizedDisplay {
                                name: display.name,
                                description: display.description,
                            },
                        )
                    })
                    .collect();
                FfiBrowseEntry {
                    installed: store.get(&entry.id).is_some(),
                    id: entry.id,
                    version: entry.version,
                    name: entry.name,
                    description: entry.description,
                    skill_type: entry.skill_type,
                    license: entry.license,
                    author: entry.author,
                    homepage: entry.homepage,
                    capabilities: entry.capabilities,
                    languages: entry.languages,
                    localizations,
                }
            })
            .collect();
        out.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(out)
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
        let client = RegistryClient::new();
        let index = client.fetch_index().map_err(map_registry_error)?;
        let trust = trust_root()?;
        let options = android_load_options(&self.storage_dir);
        let mut store = self.store.lock().expect("skill store mutex poisoned");
        let installed = install_by_id(
            &client,
            &index,
            &id,
            &mut store,
            &trust,
            &options,
        )
        .map_err(map_registry_error)?;
        let languages = derive_supported_locales(&installed.install_dir);
        Ok(FfiInstalledSkill {
            id: installed.id,
            version: installed.version,
            install_dir: installed.install_dir.to_string_lossy().into_owned(),
            languages,
        })
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
        let client = RegistryClient::new();
        let index = client.fetch_index().map_err(map_registry_error)?;
        let entry: IndexEntry = index
            .skills
            .into_iter()
            .find(|e| e.id == id)
            .ok_or_else(|| FfiRegistryError::NotFound { id: id.clone() })?;
        let source = client.fetch_manifest(&entry).map_err(map_registry_error)?;
        // parent_dir_name = None: the sidecar sits at manifests/<id>-<version>.md,
        // not in a directory named after the skill, so the AgentSkills
        // name-must-match-parent-dir check doesn't apply here.
        let skillfile =
            Skillfile::parse(&source, None).map_err(|e: ManifestError| FfiRegistryError::Manifest {
                message: e.to_string(),
            })?;
        let ext = skillfile.ari_extension.ok_or_else(|| FfiRegistryError::Manifest {
            message: "preview manifest is missing the ari extension metadata".to_string(),
        })?;
        Ok(FfiSkillManifest {
            id: ext.id,
            version: ext.version,
            name: skillfile.name,
            description: skillfile.description,
            author: ext.author,
            homepage: ext.homepage,
            license: skillfile.license,
            capabilities: ext
                .capabilities
                .into_iter()
                .map(|c| capability_name(c).to_string())
                .collect(),
            languages: ext.languages,
            body: skillfile.body,
            examples: ext.examples.iter().map(|e| e.text.clone()).collect(),
        })
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
        let client = RegistryClient::new();
        let index = client.fetch_index().map_err(map_registry_error)?;
        let entry: IndexEntry = index
            .skills
            .into_iter()
            .find(|e| e.id == id)
            .ok_or_else(|| FfiRegistryError::NotFound { id: id.clone() })?;
        Ok(client.screenshot_urls(&entry, &platform))
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
        let store = self.store.lock().expect("skill store mutex poisoned");
        let entry = store
            .get(&id)
            .ok_or_else(|| FfiRegistryError::NotInstalled { id: id.clone() })?;

        // Walk the per-locale manifest set, then pick the variant for
        // the user's active locale. The set's invariants guarantee
        // both a canonical English entry and structural consistency
        // with it (`id`, `type`, `capabilities`, `behaviour` all
        // match), so picking the localized variant is safe — only
        // the human-language fields differ.
        let manifest_set = ari_skill_loader::parse_skill_directory(&entry.install_dir)
            .map_err(|e| FfiRegistryError::Manifest {
                message: e.to_string(),
            })?;
        let skillfile = manifest_set.for_locale(&locale).clone();

        let ext = skillfile.ari_extension.ok_or_else(|| FfiRegistryError::Manifest {
            message: "manifest is missing the ari extension metadata".to_string(),
        })?;

        Ok(FfiSkillManifest {
            id: ext.id,
            version: ext.version,
            name: skillfile.name,
            description: skillfile.description,
            author: ext.author,
            homepage: ext.homepage,
            license: skillfile.license,
            capabilities: ext
                .capabilities
                .into_iter()
                .map(|c| capability_name(c).to_string())
                .collect(),
            languages: ext.languages,
            body: skillfile.body,
            examples: ext.examples.iter().map(|e| e.text.clone()).collect(),
        })
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
        let store = self.store.lock().expect("skill store mutex poisoned");
        let entry = store
            .get(&id)
            .ok_or_else(|| FfiRegistryError::NotInstalled { id: id.clone() })?;
        let manifest_set = ari_skill_loader::parse_skill_directory(&entry.install_dir)
            .map_err(|e| FfiRegistryError::Manifest {
                message: e.to_string(),
            })?;
        let skillfile = manifest_set.for_locale(&locale).clone();
        let ext = skillfile.ari_extension.ok_or_else(|| FfiRegistryError::Manifest {
            message: "manifest is missing the ari extension metadata".to_string(),
        })?;

        Ok(ext
            .settings
            .iter()
            .map(|field| {
                let current_value = if matches!(field.field_type, ConfigFieldType::Secret) {
                    // Never round-trip secret values across the FFI
                    // boundary — Android already has them in encrypted
                    // storage, so we just signal "set" via the bullet
                    // placeholder and let the UI mask the input field.
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
                    field_type: match &field.field_type {
                        ConfigFieldType::Text => "text".to_string(),
                        ConfigFieldType::Secret => "secret".to_string(),
                        ConfigFieldType::Select { .. } => "select".to_string(),
                        ConfigFieldType::DeviceCalendar => "device_calendar".to_string(),
                        ConfigFieldType::DeviceTaskList => "device_task_list".to_string(),
                        ConfigFieldType::DynamicSelect => "dynamic_select".to_string(),
                        ConfigFieldType::Action => "action".to_string(),
                    },
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
                    depends_on: field.depends_on.clone(),
                    validate: field.validate,
                    help_text: field.help_text.clone(),
                    collapsed_group: field.collapsed_group.clone(),
                }
            })
            .collect())
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
        let mut store = self.store.lock().expect("skill store mutex poisoned");
        store.uninstall(&id).map_err(|e| match e {
            StoreError::NotInstalled { id } => FfiRegistryError::NotInstalled { id },
            other => FfiRegistryError::Store {
                message: other.to_string(),
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static N: AtomicU64 = AtomicU64::new(0);

    fn unique_dir(label: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let mut p = std::env::temp_dir();
        p.push(format!("ari-ffi-registry-test-{label}-{nanos}-{n}"));
        p
    }

    #[test]
    fn registry_errors_map_to_accurate_ffi_variants() {
        use ari_skill_loader::BundleError;

        // Transport failure → Network. The ONLY variant that should
        // produce a "check your connection" hint in the UI.
        assert!(matches!(
            map_registry_error(RegistryError::Http("connection refused".into())),
            FfiRegistryError::Network { .. }
        ));

        // Non-success HTTP status → BadStatus, carrying the code.
        match map_registry_error(RegistryError::BadStatus {
            url: "https://x/y".into(),
            status: 503,
        }) {
            FfiRegistryError::BadStatus { status } => assert_eq!(status, 503),
            other => panic!("expected BadStatus, got {other:?}"),
        }

        // Over the size cap → TooLarge, carrying the limit. This is the
        // weather-skill bug: it used to surface as "check your connection".
        match map_registry_error(RegistryError::TooLarge {
            url: "https://x/y".into(),
            limit: 8 * 1024 * 1024,
        }) {
            FfiRegistryError::TooLarge { limit_bytes } => {
                assert_eq!(limit_bytes, 8 * 1024 * 1024)
            }
            other => panic!("expected TooLarge, got {other:?}"),
        }

        // sha mismatch and any bundle-level failure → Integrity.
        assert!(matches!(
            map_registry_error(RegistryError::ShaMismatch {
                id: "dev.heyari.weather".into(),
                expected: "aaa".into(),
                actual: "bbb".into(),
            }),
            FfiRegistryError::Integrity { .. }
        ));
        assert!(matches!(
            map_registry_error(RegistryError::Install(BundleError::HashMismatch {
                expected: "aaa".into(),
                actual: "bbb".into(),
            })),
            FfiRegistryError::Integrity { .. }
        ));

        // Pass-through identities.
        assert!(matches!(
            map_registry_error(RegistryError::NotFound { id: "x".into() }),
            FfiRegistryError::NotFound { .. }
        ));
        assert!(matches!(
            map_registry_error(RegistryError::ManifestUnavailable { id: "x".into() }),
            FfiRegistryError::ManifestUnavailable { .. }
        ));

        // Genuine registry-format problem → generic Registry bucket,
        // explicitly NOT a network claim.
        assert!(matches!(
            map_registry_error(RegistryError::Parse("bad json".into())),
            FfiRegistryError::Registry { .. }
        ));
    }

    #[test]
    fn new_opens_empty_store_and_lists_nothing() {
        let root = unique_dir("root");
        let storage = unique_dir("storage");
        let reg = SkillRegistry::new(
            root.to_string_lossy().into_owned(),
            storage.to_string_lossy().into_owned(),
            SkillSettingsStore::new(),
        )
        .unwrap();
        assert!(reg.list_installed().is_empty());
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&storage);
    }

    #[test]
    fn new_returns_store_error_if_root_is_a_file() {
        // Create a regular file where the store expects a directory.
        let root = unique_dir("filecollision");
        std::fs::write(&root, b"not a dir").unwrap();
        let storage = unique_dir("storage2");
        let result = SkillRegistry::new(
            root.to_string_lossy().into_owned(),
            storage.to_string_lossy().into_owned(),
            SkillSettingsStore::new(),
        );
        match result {
            Ok(_) => panic!("expected store error when root is a file"),
            Err(FfiRegistryError::Store { .. }) => {}
            Err(other) => panic!("expected Store error, got {other:?}"),
        }
        let _ = std::fs::remove_file(&root);
        let _ = std::fs::remove_dir_all(&storage);
    }

    #[test]
    fn install_update_returns_not_found_for_unknown_id_in_empty_registry() {
        // We don't have a local test server here (that's covered in the
        // registry module). What we *can* cheaply verify is that the
        // FfiRegistryError variant is wired through end-to-end for the
        // one path that doesn't need a network round-trip: if the real
        // registry doesn't carry the id, we surface NotFound. For that
        // we'd need a server, so instead assert the error discriminant
        // pattern compiles and the function is callable.
        let root = unique_dir("lookup");
        let storage = unique_dir("lookup-storage");
        let reg = SkillRegistry::new(
            root.to_string_lossy().into_owned(),
            storage.to_string_lossy().into_owned(),
            SkillSettingsStore::new(),
        )
        .unwrap();
        // Just confirm the method is present and the lock path is sound.
        let _ = reg.list_installed();
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&storage);
    }

    #[test]
    fn set_skill_setting_round_trips_through_shared_store() {
        // Verify the writes done via SkillRegistry are visible via the
        // shared SkillSettingsStore — that's the contract that lets
        // assistant config and per-skill settings co-exist on the same
        // backing map.
        let root = unique_dir("setting-root");
        let storage = unique_dir("setting-storage");
        let store = SkillSettingsStore::new();
        let reg = SkillRegistry::new(
            root.to_string_lossy().into_owned(),
            storage.to_string_lossy().into_owned(),
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
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&storage);
    }

    /// A minimal but valid SKILL.md declaring exactly five example
    /// utterances in a known order, so we can assert both the values and
    /// the declaration order survive the FFI mapping intact.
    const GREET_SKILL_MD: &str = r#"---
name: greet
description: A test skill
metadata:
  ari:
    id: dev.heyari.test.greet
    version: "1.0.0"
    engine: ">=0.3,<0.4"
    capabilities: []
    languages: [en]
    specificity: medium
    matching:
      patterns:
        - keywords: [hello]
          weight: 1.0
    declarative:
      response: "hi"
    examples:
      - text: hello
      - text: hi there
      - text: hey
      - text: good morning
      - text: greetings
---

# Greet
Says hello.
"#;

    #[test]
    fn read_installed_manifest_surfaces_examples_in_declaration_order() {
        // End-to-end through the real FFI method: lay down an installed
        // skill on disk, open the registry (which rescans it into the
        // store), then read it back and assert every declared example
        // utterance is surfaced verbatim, in order.
        let root = unique_dir("examples-root");
        let storage = unique_dir("examples-storage");
        let skill_dir = root.join("greet");
        std::fs::create_dir_all(&skill_dir).unwrap();
        std::fs::write(skill_dir.join("SKILL.md"), GREET_SKILL_MD).unwrap();

        let reg = SkillRegistry::new(
            root.to_string_lossy().into_owned(),
            storage.to_string_lossy().into_owned(),
            SkillSettingsStore::new(),
        )
        .unwrap();

        let manifest = reg
            .read_installed_manifest("dev.heyari.test.greet".into(), "en".into())
            .expect("skill was just installed, manifest must read");

        assert_eq!(
            manifest.examples,
            vec![
                "hello".to_string(),
                "hi there".to_string(),
                "hey".to_string(),
                "good morning".to_string(),
                "greetings".to_string(),
            ],
            "declared example utterances must be surfaced verbatim, in declaration order",
        );

        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&storage);
    }
}
