//! The skill store: what is installed, what the registry carries, installing,
//! updating and uninstalling skills, and each skill's settings. Moved from
//! `ari-ffi`'s `skill_registry.rs`; the policy itself lives in
//! `ari-skill-loader`.
//!
//! The registry calls block on the network, so callers run them off any UI
//! thread.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use ari_skill_loader::registry::LocalizedDisplay;
use ari_skill_loader::{
    Index, IndexEntry, LoadOptions, ManifestError, ModelCatalog, REGISTRY_TRUST_KEY, RegistryClient, RegistryError,
    SkillStore, Skillfile, StoreError, TrustRoot, capability_name, check_updates, install_by_id, install_update,
};

use crate::settings::{SettingField, SettingsStore, setting_fields};

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

/// An installed skill.
#[derive(Debug, Clone, PartialEq)]
pub struct InstalledSkill {
    pub id: String,
    pub version: String,
    pub install_dir: PathBuf,
    /// Locale codes the skill supports for end users: `SKILL.{locale}.md` is
    /// present and, if the skill ships any `strings/` files at all,
    /// `strings/{locale}.json` too.
    pub languages: Vec<String>,
}

impl InstalledSkill {
    fn from_store(s: ari_skill_loader::InstalledSkill) -> Self {
        let languages = derive_supported_locales(&s.install_dir);
        Self { id: s.id, version: s.version, install_dir: s.install_dir, languages }
    }
}

/// An update the registry has for an installed skill, with the registry's
/// name and description for it.
#[derive(Debug, Clone, PartialEq)]
pub struct SkillUpdate {
    pub id: String,
    pub installed_version: String,
    pub available_version: String,
    pub name: String,
    pub description: String,
}

/// One skill the registry carries. `version` is the registry's, which may be
/// ahead of the installed one.
#[derive(Debug, Clone)]
pub struct BrowseEntry {
    pub id: String,
    pub version: String,
    pub name: String,
    pub description: String,
    /// `skill` or `assistant`.
    pub skill_type: String,
    pub installed: bool,
    pub license: Option<String>,
    pub author: Option<String>,
    pub homepage: Option<String>,
    pub capabilities: Vec<String>,
    /// What this frontend can't provide of `capabilities`: the skill wouldn't
    /// load here, so it isn't offered. Empty when it would.
    pub missing_capabilities: Vec<String>,
    pub languages: Vec<String>,
    /// Display strings per ISO 639-1 code, from every `SKILL.{locale}.md`.
    /// English is absent: `name` and `description` carry it. Frontends fall
    /// back to those when the user's locale has no entry.
    pub localizations: HashMap<String, LocalizedDisplay>,
}

/// A skill's manifest in full: what a detail screen shows.
#[derive(Debug, Clone, PartialEq)]
pub struct SkillManifest {
    pub id: String,
    pub version: String,
    pub name: String,
    pub description: String,
    pub author: Option<String>,
    pub homepage: Option<String>,
    pub license: Option<String>,
    /// Capability names as the manifest has them (`http`, `storage_kv`, …).
    pub capabilities: Vec<String>,
    pub languages: Vec<String>,
    /// The SKILL.md body after the frontmatter, verbatim markdown.
    pub body: String,
    /// The `text` of each example the skill declares, in order.
    pub examples: Vec<String>,
}

impl SkillManifest {
    fn from_skillfile(skillfile: Skillfile, missing_ari: &str) -> Result<Self, SkillRegistryError> {
        let ext = skillfile.ari_extension.ok_or_else(|| SkillRegistryError::Manifest { message: missing_ari.to_string() })?;
        Ok(Self {
            id: ext.id,
            version: ext.version,
            name: skillfile.name,
            description: skillfile.description,
            author: ext.author,
            homepage: ext.homepage,
            license: skillfile.license,
            capabilities: ext.capabilities.into_iter().map(|c| capability_name(c).to_string()).collect(),
            languages: ext.languages,
            body: skillfile.body,
            examples: ext.examples.iter().map(|e| e.text.clone()).collect(),
        })
    }
}

/// Why a store or registry call failed, told apart so a frontend can give the
/// user the real reason rather than blaming the connection for everything.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SkillRegistryError {
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
    /// The registry has no preview manifest for this skill (an index row from
    /// before the sidecars): show the browse entry instead.
    #[error("no preview manifest available for skill: {id}")]
    ManifestUnavailable { id: String },
    #[error("trust key: {message}")]
    TrustKey { message: String },
    /// The registry couldn't be reached: the only failure that warrants a
    /// "check your connection" hint.
    #[error("network error: {message}")]
    Network { message: String },
    /// The registry answered with a failure status (404 for a missing bundle,
    /// 5xx for an outage, …).
    #[error("registry returned status {status}")]
    BadStatus { status: u32 },
    /// The bundle is over the install size cap.
    #[error("skill bundle is too large to install (limit {limit_bytes} bytes)")]
    TooLarge { limit_bytes: u64 },
    /// The bundle failed verification (sha256, signature, an unsafe tar entry,
    /// a corrupt archive) and was not installed.
    #[error("skill failed verification: {message}")]
    Integrity { message: String },
}

impl From<RegistryError> for SkillRegistryError {
    /// Every failure used to collapse into the generic registry error, which
    /// the frontend showed as "check your connection", so an oversized or
    /// tampered-with bundle looked identical to being offline.
    fn from(e: RegistryError) -> Self {
        match e {
            RegistryError::NotFound { id } => Self::NotFound { id },
            RegistryError::ManifestUnavailable { id } => Self::ManifestUnavailable { id },
            RegistryError::Http(message) => Self::Network { message },
            RegistryError::BadStatus { status, .. } => Self::BadStatus { status: u32::from(status) },
            RegistryError::TooLarge { limit, .. } => Self::TooLarge { limit_bytes: limit as u64 },
            // sha mismatch or any bundle-level failure (bad signature, unsafe
            // tar entry, corrupt archive) → the skill couldn't be safely
            // verified. Don't blame the network.
            e @ (RegistryError::ShaMismatch { .. } | RegistryError::Install(_)) => {
                Self::Integrity { message: e.to_string() }
            }
            // Parse / UnsupportedIndexVersion: a genuine registry-format
            // problem; keep the generic bucket.
            other => Self::Registry { message: other.to_string() },
        }
    }
}

fn trust_root() -> Result<TrustRoot, SkillRegistryError> {
    TrustRoot::single(&REGISTRY_TRUST_KEY).map_err(|e| SkillRegistryError::TrustKey { message: e.to_string() })
}

/// The skill store and the registry it installs from. One per process.
pub struct SkillRegistry {
    store: Mutex<SkillStore>,
    /// The same grants and `storage_kv` root as the engine's reloads use
    /// (see [`crate::load_options`]).
    options: LoadOptions,
    /// Shared with the assistant registry, so a setting written through
    /// either is the one the engine reads.
    settings: Arc<SettingsStore>,
    /// Where each fetched index is kept, for the engine's skill suggestions.
    index_cache: PathBuf,
}

impl SkillRegistry {
    /// Open (or create) the store at `skill_store_dir`, installing and
    /// loading with `options`, whose storage config says where skills'
    /// `storage_kv` files live.
    pub fn open(skill_store_dir: PathBuf, options: LoadOptions, settings: Arc<SettingsStore>) -> Result<Self, SkillRegistryError> {
        let store = SkillStore::open(skill_store_dir, options.storage_config.clone(), trust_root()?)
            .map_err(|e| SkillRegistryError::Store { message: e.to_string() })?;
        // Next to models.json; `SkillStore::rescan` only looks at
        // directories, so the file is inert there.
        let index_cache = store.root_path().join("index.json");
        Ok(Self { store: Mutex::new(store), options, settings, index_cache })
    }

    /// Fetch the registry index and keep a copy for the engine, which
    /// suggests store skills from it (ari-android#44). Every path that
    /// fetches the index goes through here, so Browse and the daily update
    /// check both keep the copy fresh. Caching is best effort: a failed
    /// write costs fresher suggestions, never the caller's operation.
    fn fetch_index(&self, client: &RegistryClient) -> Result<Index, SkillRegistryError> {
        let index = client.fetch_index()?;
        if let Ok(bytes) = serde_json::to_vec(&index) {
            // Written aside and renamed, so the engine never reads half a file.
            let tmp = self.index_cache.with_extension("json.tmp");
            let _ = std::fs::write(&tmp, &bytes).and_then(|_| std::fs::rename(&tmp, &self.index_cache));
        }
        Ok(index)
    }

    fn store(&self) -> std::sync::MutexGuard<'_, SkillStore> {
        self.store.lock().expect("skill store mutex poisoned")
    }

    /// Where the registry index is cached each time it's fetched, for
    /// `Engine::set_skill_suggestions_path`.
    pub fn index_cache_path(&self) -> &Path {
        &self.index_cache
    }

    /// Every installed skill, sorted by id.
    pub fn list_installed(&self) -> Vec<InstalledSkill> {
        let mut out: Vec<InstalledSkill> = self.store().list().into_iter().map(InstalledSkill::from_store).collect();
        out.sort_by(|a, b| a.id.cmp(&b.id));
        out
    }

    /// The updates the registry has for installed skills. Empty means up to
    /// date.
    pub fn check_for_updates(&self) -> Result<Vec<SkillUpdate>, SkillRegistryError> {
        let index = self.fetch_index(&RegistryClient::new())?;
        let updates = check_updates(&self.store(), &index);
        Ok(updates
            .into_iter()
            .map(|u| SkillUpdate {
                id: u.id,
                installed_version: u.installed_version,
                available_version: u.available_version,
                name: u.entry.name,
                description: u.entry.description,
            })
            .collect())
    }

    /// Fetch, verify and cache the registry's tier→model catalog, returning
    /// where it was written, for [`crate::Runtime::load_model_catalog`].
    ///
    /// Cached next to the installed skills so a device that later goes offline
    /// keeps using the last known catalog instead of falling all the way back to
    /// the per-tier pins baked into each skill's manifest. `SkillStore::rescan`
    /// only looks at directories, so the file is inert there.
    pub fn refresh_model_catalog(&self) -> Result<PathBuf, SkillRegistryError> {
        let client = RegistryClient::new();
        let index = self.fetch_index(&client)?;
        let bytes = client.fetch_catalog_bytes(&index, &trust_root()?)?;

        // Parse before caching so a catalog this build can't understand fails
        // here, rather than being written and silently ignored at call time.
        ModelCatalog::from_json_bytes(&bytes).map_err(|e| SkillRegistryError::Registry { message: e.to_string() })?;

        let path = self.store().root_path().join("models.json");
        std::fs::write(&path, &bytes)
            .map_err(|e| SkillRegistryError::Store { message: format!("caching models.json: {e}") })?;
        Ok(path)
    }

    /// Install the registry's current version of installed skill `id` over
    /// the local one.
    pub fn install_skill_update(&self, id: &str) -> Result<InstalledSkill, SkillRegistryError> {
        let client = RegistryClient::new();
        let index = self.fetch_index(&client)?;
        let entry = index
            .skills
            .iter()
            .find(|e| e.id == id)
            .cloned()
            .ok_or_else(|| SkillRegistryError::NotFound { id: id.to_string() })?;
        let trust = trust_root()?;
        let installed = install_update(&client, &entry, &mut self.store(), &trust, &self.options)?;
        Ok(InstalledSkill::from_store(installed))
    }

    /// Every skill the registry carries, sorted by id, marked when it is
    /// installed and with what this frontend can't provide for it.
    pub fn browse_registry(&self) -> Result<Vec<BrowseEntry>, SkillRegistryError> {
        let index = self.fetch_index(&RegistryClient::new())?;
        let store = self.store();
        let mut out: Vec<BrowseEntry> = index
            .skills
            .into_iter()
            .map(|entry| BrowseEntry {
                missing_capabilities: entry.missing_capabilities(&self.options.host_capabilities),
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
                localizations: entry.localizations,
            })
            .collect();
        out.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(out)
    }

    /// Install the registry's current version of `id`, installed already or
    /// not.
    pub fn install_skill_by_id(&self, id: &str) -> Result<InstalledSkill, SkillRegistryError> {
        let client = RegistryClient::new();
        let index = self.fetch_index(&client)?;
        let trust = trust_root()?;
        let installed = install_by_id(&client, &index, id, &mut self.store(), &trust, &self.options)?;
        Ok(InstalledSkill::from_store(installed))
    }

    /// The registry's preview manifest for `id`, so a detail screen can show
    /// a skill before it is installed.
    ///
    /// The sidecar is a verbatim copy of the skill's SKILL.md published
    /// alongside the signed bundle. It is **not** covered by the bundle
    /// signature — good enough for a read-only preview, not suitable for
    /// anything load-bearing. Install still goes through the full
    /// signature + sha256 pipeline.
    pub fn fetch_manifest_preview(&self, id: &str) -> Result<SkillManifest, SkillRegistryError> {
        let client = RegistryClient::new();
        let index = self.fetch_index(&client)?;
        let entry: IndexEntry = index
            .skills
            .into_iter()
            .find(|e| e.id == id)
            .ok_or_else(|| SkillRegistryError::NotFound { id: id.to_string() })?;
        let source = client.fetch_manifest(&entry)?;
        // parent_dir_name = None: the sidecar sits at manifests/<id>-<version>.md,
        // not in a directory named after the skill, so the AgentSkills
        // name-must-match-parent-dir check doesn't apply here.
        let skillfile = Skillfile::parse(&source, None)
            .map_err(|e: ManifestError| SkillRegistryError::Manifest { message: e.to_string() })?;
        SkillManifest::from_skillfile(skillfile, "preview manifest is missing the ari extension metadata")
    }

    /// The registry's screenshots of `id` for a frontend on `platform`
    /// (`android`, `linux`, …), in display order. A skill with none for this
    /// platform falls back to whichever it has; empty means it ships none.
    pub fn fetch_screenshot_urls(&self, id: &str, platform: &str) -> Result<Vec<String>, SkillRegistryError> {
        let client = RegistryClient::new();
        let index = self.fetch_index(&client)?;
        let entry: IndexEntry = index
            .skills
            .into_iter()
            .find(|e| e.id == id)
            .ok_or_else(|| SkillRegistryError::NotFound { id: id.to_string() })?;
        Ok(client.screenshot_urls(&entry, platform))
    }

    /// Installed skill `id`'s manifest, in `locale` where the skill has a
    /// `SKILL.{locale}.md`, else canonical English.
    pub fn read_installed_manifest(&self, id: &str, locale: &str) -> Result<SkillManifest, SkillRegistryError> {
        SkillManifest::from_skillfile(self.installed_skillfile(id, locale)?, "manifest is missing the ari extension metadata")
    }

    /// Installed skill `id`'s settings with their current values, labelled
    /// in `locale` where the skill has a translation.
    pub fn get_skill_settings(&self, id: &str, locale: &str) -> Result<Vec<SettingField>, SkillRegistryError> {
        let skillfile = self.installed_skillfile(id, locale)?;
        let ext = skillfile.ari_extension.ok_or_else(|| SkillRegistryError::Manifest {
            message: "manifest is missing the ari extension metadata".to_string(),
        })?;
        Ok(setting_fields(id, &ext.settings, &self.settings))
    }

    /// Remove installed skill `id` and its `storage_kv` state. Reload the
    /// runtime's skills afterwards ([`crate::Runtime::reload_community_skills`]).
    pub fn uninstall_skill_by_id(&self, id: &str) -> Result<(), SkillRegistryError> {
        self.store().uninstall(id).map_err(|e| match e {
            StoreError::NotInstalled { id } => SkillRegistryError::NotInstalled { id },
            other => SkillRegistryError::Store { message: other.to_string() },
        })
    }

    /// The file an envelope's `asset:<path>` reference from skill `skill_id`
    /// names: `<path>` under the skill's `assets` directory. None when
    /// `reference` isn't an asset, the skill isn't installed (built-in
    /// skills have no assets), the path would leave the directory, or there's
    /// no such file. Moved from ari-android's `AssetResolver.kt`.
    pub fn asset_path(&self, skill_id: &str, reference: &str) -> Option<PathBuf> {
        let path = Path::new(reference.strip_prefix("asset:")?);
        let inside = path.components().next().is_some()
            && path.components().all(|c| matches!(c, std::path::Component::Normal(_)));
        if !inside {
            return None;
        }
        let file = self.store().get(skill_id)?.install_dir.join("assets").join(path);
        file.is_file().then_some(file)
    }

    /// Walk the skill's per-locale manifest set and pick `locale`'s variant.
    /// The set's invariants guarantee a canonical English entry and that
    /// every variant agrees with it on `id`, `type`, `capabilities` and
    /// `behaviour`, so only the human-language fields differ.
    fn installed_skillfile(&self, id: &str, locale: &str) -> Result<Skillfile, SkillRegistryError> {
        let install_dir = self
            .store()
            .get(id)
            .map(|entry| entry.install_dir.clone())
            .ok_or_else(|| SkillRegistryError::NotInstalled { id: id.to_string() })?;
        let manifest_set = ari_skill_loader::parse_skill_directory(&install_dir)
            .map_err(|e| SkillRegistryError::Manifest { message: e.to_string() })?;
        Ok(manifest_set.for_locale(locale).clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static N: AtomicU64 = AtomicU64::new(0);

    fn unique_dir(label: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("ari-runtime-registry-test-{label}-{nanos}-{n}"))
    }

    fn open(root: &Path, storage: &Path) -> Result<SkillRegistry, SkillRegistryError> {
        SkillRegistry::open(root.to_path_buf(), crate::load_options(storage, ari_skill_loader::HostCapabilities::all()), Arc::new(SettingsStore::new()))
    }

    #[test]
    fn registry_errors_map_to_accurate_variants() {
        use ari_skill_loader::BundleError;

        // Transport failure → Network. The ONLY variant that should
        // produce a "check your connection" hint in the UI.
        assert!(matches!(
            SkillRegistryError::from(RegistryError::Http("connection refused".into())),
            SkillRegistryError::Network { .. }
        ));
        // Non-success HTTP status → BadStatus, carrying the code.
        assert_eq!(
            SkillRegistryError::from(RegistryError::BadStatus { url: "https://x/y".into(), status: 503 }),
            SkillRegistryError::BadStatus { status: 503 }
        );
        // Over the size cap → TooLarge, carrying the limit. This is the
        // weather-skill bug: it used to surface as "check your connection".
        assert_eq!(
            SkillRegistryError::from(RegistryError::TooLarge { url: "https://x/y".into(), limit: 8 * 1024 * 1024 }),
            SkillRegistryError::TooLarge { limit_bytes: 8 * 1024 * 1024 }
        );
        // sha mismatch and any bundle-level failure → Integrity.
        assert!(matches!(
            SkillRegistryError::from(RegistryError::ShaMismatch {
                id: "dev.heyari.weather".into(),
                expected: "aaa".into(),
                actual: "bbb".into(),
            }),
            SkillRegistryError::Integrity { .. }
        ));
        assert!(matches!(
            SkillRegistryError::from(RegistryError::Install(BundleError::HashMismatch {
                expected: "aaa".into(),
                actual: "bbb".into(),
            })),
            SkillRegistryError::Integrity { .. }
        ));
        // Pass-through identities.
        assert_eq!(
            SkillRegistryError::from(RegistryError::NotFound { id: "x".into() }),
            SkillRegistryError::NotFound { id: "x".into() }
        );
        assert_eq!(
            SkillRegistryError::from(RegistryError::ManifestUnavailable { id: "x".into() }),
            SkillRegistryError::ManifestUnavailable { id: "x".into() }
        );
        // Genuine registry-format problem → generic Registry bucket,
        // explicitly NOT a network claim.
        assert!(matches!(
            SkillRegistryError::from(RegistryError::Parse("bad json".into())),
            SkillRegistryError::Registry { .. }
        ));
    }

    #[test]
    fn open_creates_an_empty_store_that_lists_nothing() {
        let (root, storage) = (unique_dir("root"), unique_dir("storage"));
        let registry = open(&root, &storage).unwrap();
        assert!(registry.list_installed().is_empty());
        assert_eq!(registry.index_cache_path(), root.join("index.json"));
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&storage);
    }

    #[test]
    fn open_fails_with_a_store_error_if_the_root_is_a_file() {
        let root = unique_dir("filecollision");
        std::fs::write(&root, b"not a dir").unwrap();
        let storage = unique_dir("storage2");
        match open(&root, &storage) {
            Ok(_) => panic!("expected store error when root is a file"),
            Err(SkillRegistryError::Store { .. }) => {}
            Err(other) => panic!("expected Store error, got {other:?}"),
        }
        let _ = std::fs::remove_file(&root);
        let _ = std::fs::remove_dir_all(&storage);
    }

    /// A minimal but valid SKILL.md declaring exactly five example
    /// utterances in a known order, and one setting.
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
    settings:
      - key: greeting
        label: Greeting
        type: text
---

# Greet
Says hello.
"#;

    fn with_greet_installed() -> (SkillRegistry, PathBuf, PathBuf) {
        let (root, storage) = (unique_dir("greet-root"), unique_dir("greet-storage"));
        let skill_dir = root.join("greet");
        std::fs::create_dir_all(&skill_dir).unwrap();
        std::fs::write(skill_dir.join("SKILL.md"), GREET_SKILL_MD).unwrap();
        (open(&root, &storage).unwrap(), root, storage)
    }

    #[test]
    fn read_installed_manifest_surfaces_examples_in_declaration_order() {
        let (registry, root, storage) = with_greet_installed();
        let manifest = registry
            .read_installed_manifest("dev.heyari.test.greet", "en")
            .expect("skill was just installed, manifest must read");
        assert_eq!(
            manifest.examples,
            ["hello", "hi there", "hey", "good morning", "greetings"],
            "declared example utterances must be surfaced verbatim, in declaration order",
        );
        assert_eq!(registry.list_installed()[0].id, "dev.heyari.test.greet");
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&storage);
    }

    #[test]
    fn an_installed_skill_s_settings_show_their_current_values() {
        let (registry, root, storage) = with_greet_installed();
        registry.settings.set("dev.heyari.test.greet", "greeting", "Ciao");
        let fields = registry.get_skill_settings("dev.heyari.test.greet", "en").unwrap();
        assert_eq!(fields.len(), 1);
        assert_eq!((fields[0].key.as_str(), fields[0].current_value.as_deref()), ("greeting", Some("Ciao")));
        assert_eq!(
            registry.get_skill_settings("nobody.here", "en"),
            Err(SkillRegistryError::NotInstalled { id: "nobody.here".into() })
        );
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&storage);
    }

    #[test]
    fn an_asset_reference_names_a_file_in_the_skill_s_assets() {
        let (registry, root, storage) = with_greet_installed();
        let assets = root.join("greet").join("assets");
        std::fs::create_dir_all(assets.join("icons")).unwrap();
        std::fs::write(assets.join("icons").join("wave.png"), b"png").unwrap();
        std::fs::write(root.join("greet").join("secret.txt"), b"not an asset").unwrap();
        let id = "dev.heyari.test.greet";

        assert_eq!(registry.asset_path(id, "asset:icons/wave.png"), Some(assets.join("icons").join("wave.png")));
        // Not an asset reference, a file that isn't there, a directory.
        assert_eq!(registry.asset_path(id, "icons/wave.png"), None);
        assert_eq!(registry.asset_path(id, "system.alarm"), None);
        assert_eq!(registry.asset_path(id, "asset:icons/missing.png"), None);
        assert_eq!(registry.asset_path(id, "asset:icons"), None);
        assert_eq!(registry.asset_path(id, "asset:"), None);
        // Out of the assets directory, even to a file that's there.
        assert_eq!(registry.asset_path(id, "asset:../secret.txt"), None);
        assert_eq!(registry.asset_path(id, "asset:icons/../../secret.txt"), None);
        assert_eq!(registry.asset_path(id, "asset:/etc/hostname"), None);
        // A built-in skill, or one that isn't installed.
        assert_eq!(registry.asset_path("current_time", "asset:icons/wave.png"), None);
        assert_eq!(registry.asset_path("", "asset:icons/wave.png"), None);
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&storage);
    }

    #[test]
    fn uninstalling_a_skill_that_is_not_installed_says_so() {
        let (root, storage) = (unique_dir("uninstall-root"), unique_dir("uninstall-storage"));
        let registry = open(&root, &storage).unwrap();
        assert_eq!(
            registry.uninstall_skill_by_id("nobody.here"),
            Err(SkillRegistryError::NotInstalled { id: "nobody.here".into() })
        );
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&storage);
    }
}
