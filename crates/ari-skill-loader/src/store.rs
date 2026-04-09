//! Long-running owner of an installed-skills directory.
//!
//! Where [`install_from_bytes`] is a one-shot "extract this bundle into that
//! root", `SkillStore` is the stateful counterpart that frontends actually
//! use: it owns `<filesDir>/skills/`, knows what's installed, refuses
//! downgrades, wipes per-skill state on uninstall, and lets the engine
//! enumerate everything in one go at startup.
//!
//! The store identifies skills by their `metadata.ari.id` (the reverse-DNS
//! string from the manifest), not by directory name. Two bundles with the
//! same id but different AgentSkills slugs are treated as upgrades of each
//! other — the old slug directory is removed after the new one is in place.

use crate::bundle::{install_from_bytes, BundleError, InstalledBundle};
use crate::loader::{load_skill_directory_with, LoadOptions, LoadReport};
use crate::manifest::Skillfile;
use crate::signature::TrustRoot;
use crate::storage_config::StorageConfig;
use std::collections::HashMap;
use std::io::Read;
use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum StoreError {
    #[error(transparent)]
    Bundle(#[from] BundleError),

    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("skill {id} is not installed")]
    NotInstalled { id: String },

    #[error("downgrade refused for {id}: installed {installed}, attempted {attempted}")]
    Downgrade {
        id: String,
        installed: String,
        attempted: String,
    },

    #[error("could not peek bundle manifest: {0}")]
    Peek(String),
}

#[derive(Debug, Clone)]
pub struct InstalledSkill {
    pub id: String,
    pub version: String,
    pub install_dir: PathBuf,
}

/// Long-running owner of `<root>/` containing per-skill subdirectories.
///
/// Wraps the one-shot bundle pipeline with the bookkeeping a real frontend
/// needs: an in-memory id → install_dir index, downgrade defence, and
/// storage_kv state cleanup on uninstall.
pub struct SkillStore {
    root: PathBuf,
    storage_config: StorageConfig,
    trust_root: TrustRoot,
    index: HashMap<String, InstalledSkill>,
}

impl SkillStore {
    /// Open a store rooted at `root`. Creates `root` if it doesn't exist.
    /// Scans the directory immediately so [`Self::list`] returns the current
    /// state without a separate refresh call.
    pub fn open(
        root: impl Into<PathBuf>,
        storage_config: StorageConfig,
        trust_root: TrustRoot,
    ) -> Result<Self, StoreError> {
        let root = root.into();
        std::fs::create_dir_all(&root)?;
        let mut store = Self {
            root,
            storage_config,
            trust_root,
            index: HashMap::new(),
        };
        store.rescan()?;
        Ok(store)
    }

    /// Re-read the root directory and rebuild the in-memory index. Cheap;
    /// only parses each SKILL.md, doesn't instantiate WASM.
    pub fn rescan(&mut self) -> Result<(), StoreError> {
        self.index.clear();
        for entry in std::fs::read_dir(&self.root)? {
            let entry = entry?;
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            // Skip leftover staging or backup directories from a crashed install.
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.starts_with(".staging-") || name.ends_with(".old") {
                    continue;
                }
            }
            let manifest_path = path.join("SKILL.md");
            if !manifest_path.is_file() {
                continue;
            }
            let Ok(sf) = Skillfile::parse_file(&manifest_path) else {
                continue;
            };
            let Some(ari) = sf.ari_extension else {
                continue;
            };
            self.index.insert(
                ari.id.clone(),
                InstalledSkill {
                    id: ari.id,
                    version: ari.version,
                    install_dir: path,
                },
            );
        }
        Ok(())
    }

    /// All currently-installed skills, in arbitrary order.
    pub fn list(&self) -> Vec<InstalledSkill> {
        self.index.values().cloned().collect()
    }

    /// Root directory this store is managing. The registry updater needs
    /// this to drive [`crate::bundle::install_from_bytes`] with a
    /// caller-supplied trust root (the one baked into the registry module)
    /// rather than the one the store was opened with.
    pub fn root_path(&self) -> &std::path::Path {
        &self.root
    }

    /// Look up a single installed skill by id.
    pub fn get(&self, id: &str) -> Option<&InstalledSkill> {
        self.index.get(id)
    }

    /// Load every installed skill via the regular loader pipeline. Use this at
    /// engine startup to populate the skill ranking pool.
    pub fn load_all(&self, options: &LoadOptions) -> std::io::Result<LoadReport> {
        load_skill_directory_with(&self.root, options)
    }

    /// Install a bundle. The bundle's manifest is peeked first to enforce
    /// downgrade defence: if a skill with the same id is already installed at
    /// a strictly newer version, the install is rejected before any disk
    /// state changes. The actual extract + atomic swap is delegated to
    /// [`install_from_bytes`].
    pub fn install(
        &mut self,
        bundle_bytes: &[u8],
        signature_bytes: &[u8],
        expected_sha256: &str,
        load_options: &LoadOptions,
    ) -> Result<InstalledSkill, StoreError> {
        let (incoming_id, incoming_version) = peek_bundle_manifest(bundle_bytes)?;

        let prior = self.index.get(&incoming_id).cloned();
        if let Some(prior) = &prior {
            if compare_versions(&incoming_version, &prior.version) == std::cmp::Ordering::Less {
                return Err(StoreError::Downgrade {
                    id: incoming_id,
                    installed: prior.version.clone(),
                    attempted: incoming_version,
                });
            }
        }

        let InstalledBundle {
            install_dir,
            skill_id,
        } = install_from_bytes(
            bundle_bytes,
            signature_bytes,
            expected_sha256,
            &self.trust_root,
            &self.root,
            load_options,
        )?;

        // If the upgrade landed under a different slug than the prior version
        // (i.e. the bundle author renamed the directory), get rid of the old
        // dir so we don't end up with two copies of the same skill.
        if let Some(prior) = prior {
            if prior.install_dir != install_dir && prior.install_dir.exists() {
                let _ = std::fs::remove_dir_all(&prior.install_dir);
            }
        }

        let installed = InstalledSkill {
            id: skill_id.clone(),
            version: incoming_version,
            install_dir,
        };
        self.index.insert(skill_id, installed.clone());
        Ok(installed)
    }

    /// Remove an installed skill and wipe its `storage_kv` state. Returns
    /// `NotInstalled` if `id` was never installed.
    pub fn uninstall(&mut self, id: &str) -> Result<(), StoreError> {
        let entry = self
            .index
            .remove(id)
            .ok_or_else(|| StoreError::NotInstalled { id: id.to_string() })?;

        if entry.install_dir.exists() {
            std::fs::remove_dir_all(&entry.install_dir)?;
        }

        // Wipe per-skill storage_kv state. The file may not exist (skill
        // never called storage_set, or never had the cap granted) — that's
        // fine, treat NotFound as success.
        let storage_file = self.storage_config.file_for(id);
        match std::fs::remove_file(&storage_file) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(StoreError::Io(e)),
        }

        Ok(())
    }
}

/// Peek at a bundle without unpacking it to disk: scan the tar in memory for
/// `*/SKILL.md`, parse the manifest, and return `(id, version)`. Used by the
/// store to enforce downgrade defence before committing to an install.
fn peek_bundle_manifest(bundle_bytes: &[u8]) -> Result<(String, String), StoreError> {
    let gz = flate2::read::GzDecoder::new(bundle_bytes);
    let mut archive = tar::Archive::new(gz);
    let entries = archive
        .entries()
        .map_err(|e| StoreError::Peek(format!("read tar: {e}")))?;

    for entry_res in entries {
        let mut entry =
            entry_res.map_err(|e| StoreError::Peek(format!("read tar entry: {e}")))?;
        let path = entry
            .path()
            .map_err(|e| StoreError::Peek(format!("read entry path: {e}")))?
            .into_owned();
        // Match `<anything>/SKILL.md` at depth 1 (one parent directory).
        let mut comps = path.components();
        let parent = comps.next();
        let file = comps.next();
        let extra = comps.next();
        let (Some(parent), Some(file), None) = (parent, file, extra) else {
            continue;
        };
        let parent_name = match parent {
            std::path::Component::Normal(s) => s.to_string_lossy().into_owned(),
            _ => continue,
        };
        let file_name = match file {
            std::path::Component::Normal(s) => s.to_string_lossy().into_owned(),
            _ => continue,
        };
        if file_name != "SKILL.md" {
            continue;
        }

        let mut buf = String::new();
        entry
            .read_to_string(&mut buf)
            .map_err(|e| StoreError::Peek(format!("read SKILL.md: {e}")))?;
        let sf = Skillfile::parse(&buf, Some(&parent_name))
            .map_err(|e| StoreError::Peek(format!("parse SKILL.md: {e}")))?;
        let ari = sf
            .ari_extension
            .ok_or_else(|| StoreError::Peek("SKILL.md has no metadata.ari".to_string()))?;
        return Ok((ari.id, ari.version));
    }

    Err(StoreError::Peek(
        "bundle contains no SKILL.md".to_string(),
    ))
}

/// Compare two version strings using a tolerant dotted-numeric scheme. Splits
/// on `.`, parses each segment up to its first non-digit character as a u64,
/// and compares lexicographically over the resulting vectors. Missing
/// trailing segments are treated as zero, so `"1.0" == "1.0.0"`.
///
/// This is deliberately not a full semver implementation: skill manifests
/// already validate that `version` is a non-empty string, and the only
/// downstream use is downgrade defence. Pulling in the `semver` crate just
/// for `<` would be slop.
pub(crate) fn compare_versions(a: &str, b: &str) -> std::cmp::Ordering {
    let pa = parse_version(a);
    let pb = parse_version(b);
    let len = pa.len().max(pb.len());
    for i in 0..len {
        let av = pa.get(i).copied().unwrap_or(0);
        let bv = pb.get(i).copied().unwrap_or(0);
        match av.cmp(&bv) {
            std::cmp::Ordering::Equal => continue,
            other => return other,
        }
    }
    std::cmp::Ordering::Equal
}

fn parse_version(s: &str) -> Vec<u64> {
    s.split('.')
        .map(|seg| {
            let digits: String = seg.chars().take_while(|c| c.is_ascii_digit()).collect();
            digits.parse::<u64>().unwrap_or(0)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bundle::sha256_hex;
    use crate::signature::TrustRoot;
    use ed25519_dalek::{Signer, SigningKey};
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use sha2::{Digest, Sha256};
    use std::sync::atomic::{AtomicU64, Ordering};

    static N: AtomicU64 = AtomicU64::new(0);

    fn unique_dir(prefix: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let mut p = std::env::temp_dir();
        p.push(format!("ari-store-test-{prefix}-{nanos}-{n}"));
        p
    }

    fn fixed_keypair(seed: u8) -> SigningKey {
        SigningKey::from_bytes(&[seed; 32])
    }

    fn coin_md(version: &str) -> String {
        format!(
            r#"---
name: coin-flip
description: Flips a coin. Use when the user asks to flip a coin.
metadata:
  ari:
    id: dev.heyari.coinflip
    version: "{version}"
    engine: ">=0.3"
    matching:
      patterns:
        - keywords: [flip, coin]
          weight: 0.95
    declarative:
      response_pick: ["Heads.", "Tails."]
---
"#
        )
    }

    fn counter_md() -> &'static str {
        r#"---
name: counter
description: Counts. Use when the user asks to count.
metadata:
  ari:
    id: dev.heyari.counter
    version: "0.1.0"
    engine: ">=0.3"
    matching:
      patterns:
        - keywords: [count]
    declarative:
      response: "one"
---
"#
    }

    fn make_bundle(slug: &str, skill_md: &str) -> Vec<u8> {
        let mut gz = GzEncoder::new(Vec::new(), Compression::default());
        {
            let mut tar = tar::Builder::new(&mut gz);
            let mut dir_header = tar::Header::new_gnu();
            dir_header.set_path(format!("{slug}/")).unwrap();
            dir_header.set_size(0);
            dir_header.set_mode(0o755);
            dir_header.set_entry_type(tar::EntryType::Directory);
            dir_header.set_cksum();
            tar.append(&dir_header, std::io::empty()).unwrap();

            let body = skill_md.as_bytes();
            let mut hdr = tar::Header::new_gnu();
            hdr.set_path(format!("{slug}/SKILL.md")).unwrap();
            hdr.set_size(body.len() as u64);
            hdr.set_mode(0o644);
            hdr.set_entry_type(tar::EntryType::Regular);
            hdr.set_cksum();
            tar.append(&hdr, body).unwrap();
            tar.finish().unwrap();
        }
        gz.finish().unwrap()
    }

    fn signed_bundle(
        slug: &str,
        skill_md: &str,
        sk: &SigningKey,
    ) -> (Vec<u8>, Vec<u8>, String) {
        let bundle = make_bundle(slug, skill_md);
        let hash = sha256_hex(&bundle);
        let mut hasher = Sha256::new();
        hasher.update(&bundle);
        let sig = sk.sign(&hasher.finalize()).to_bytes().to_vec();
        (bundle, sig, hash)
    }

    fn fresh_store(seed: u8) -> (SkillStore, SigningKey, PathBuf, PathBuf) {
        let root = unique_dir("root");
        let storage_root = unique_dir("storage");
        let sk = fixed_keypair(seed);
        let trust = TrustRoot::single(sk.verifying_key().as_bytes()).unwrap();
        let store = SkillStore::open(&root, StorageConfig::new(&storage_root), trust).unwrap();
        (store, sk, root, storage_root)
    }

    #[test]
    fn open_creates_root_and_starts_empty() {
        let (store, _sk, root, storage_root) = fresh_store(10);
        assert!(root.is_dir());
        assert!(store.list().is_empty());
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&storage_root);
    }

    #[test]
    fn install_then_list_then_uninstall() {
        let (mut store, sk, root, storage_root) = fresh_store(11);
        let (bundle, sig, hash) = signed_bundle("coin-flip", &coin_md("0.1.0"), &sk);

        let installed = store
            .install(&bundle, &sig, &hash, &LoadOptions::default())
            .unwrap();
        assert_eq!(installed.id, "dev.heyari.coinflip");
        assert_eq!(installed.version, "0.1.0");
        assert!(installed.install_dir.join("SKILL.md").is_file());

        let listed = store.list();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, "dev.heyari.coinflip");

        store.uninstall("dev.heyari.coinflip").unwrap();
        assert!(store.list().is_empty());
        assert!(!installed.install_dir.exists());

        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&storage_root);
    }

    #[test]
    fn upgrade_succeeds_and_replaces_version_in_index() {
        let (mut store, sk, root, storage_root) = fresh_store(12);
        let (b1, s1, h1) = signed_bundle("coin-flip", &coin_md("0.1.0"), &sk);
        store.install(&b1, &s1, &h1, &LoadOptions::default()).unwrap();

        let (b2, s2, h2) = signed_bundle("coin-flip", &coin_md("0.2.0"), &sk);
        let upgraded = store.install(&b2, &s2, &h2, &LoadOptions::default()).unwrap();
        assert_eq!(upgraded.version, "0.2.0");
        assert_eq!(store.get("dev.heyari.coinflip").unwrap().version, "0.2.0");

        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&storage_root);
    }

    #[test]
    fn downgrade_is_refused_and_leaves_existing_install_intact() {
        let (mut store, sk, root, storage_root) = fresh_store(13);
        let (b2, s2, h2) = signed_bundle("coin-flip", &coin_md("0.2.0"), &sk);
        store.install(&b2, &s2, &h2, &LoadOptions::default()).unwrap();

        let (b1, s1, h1) = signed_bundle("coin-flip", &coin_md("0.1.0"), &sk);
        let err = store
            .install(&b1, &s1, &h1, &LoadOptions::default())
            .unwrap_err();
        match err {
            StoreError::Downgrade {
                id,
                installed,
                attempted,
            } => {
                assert_eq!(id, "dev.heyari.coinflip");
                assert_eq!(installed, "0.2.0");
                assert_eq!(attempted, "0.1.0");
            }
            other => panic!("expected Downgrade, got {other:?}"),
        }
        // Existing install untouched
        assert_eq!(store.get("dev.heyari.coinflip").unwrap().version, "0.2.0");

        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&storage_root);
    }

    #[test]
    fn equal_version_reinstall_is_allowed() {
        let (mut store, sk, root, storage_root) = fresh_store(14);
        let (b, s, h) = signed_bundle("coin-flip", &coin_md("0.1.0"), &sk);
        store.install(&b, &s, &h, &LoadOptions::default()).unwrap();
        // Reinstalling the same version should be fine — useful for repair.
        store.install(&b, &s, &h, &LoadOptions::default()).unwrap();
        assert_eq!(store.list().len(), 1);

        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&storage_root);
    }

    #[test]
    fn uninstall_unknown_skill_returns_not_installed() {
        let (mut store, _sk, root, storage_root) = fresh_store(15);
        let err = store.uninstall("dev.heyari.nope").unwrap_err();
        assert!(matches!(err, StoreError::NotInstalled { .. }));
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&storage_root);
    }

    #[test]
    fn uninstall_wipes_storage_kv_file() {
        let (mut store, sk, root, storage_root) = fresh_store(16);
        let (b, s, h) = signed_bundle("counter", counter_md(), &sk);
        store.install(&b, &s, &h, &LoadOptions::default()).unwrap();

        // Drop a fake storage_kv file as if the skill had used it.
        std::fs::create_dir_all(&storage_root).unwrap();
        let storage_file = StorageConfig::new(&storage_root).file_for("dev.heyari.counter");
        std::fs::write(&storage_file, br#"{"k":"v"}"#).unwrap();
        assert!(storage_file.is_file());

        store.uninstall("dev.heyari.counter").unwrap();
        assert!(!storage_file.exists());

        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&storage_root);
    }

    #[test]
    fn uninstall_with_no_storage_file_is_ok() {
        let (mut store, sk, root, storage_root) = fresh_store(17);
        let (b, s, h) = signed_bundle("coin-flip", &coin_md("0.1.0"), &sk);
        store.install(&b, &s, &h, &LoadOptions::default()).unwrap();
        // No storage file ever created — uninstall should still succeed.
        store.uninstall("dev.heyari.coinflip").unwrap();
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&storage_root);
    }

    #[test]
    fn rescan_picks_up_skills_dropped_in_by_hand() {
        let (mut store, sk, root, storage_root) = fresh_store(18);
        // Sneak a second install into the same root via the underlying
        // pipeline, then rescan.
        let (b, s, h) = signed_bundle("coin-flip", &coin_md("0.1.0"), &sk);
        let trust = TrustRoot::single(sk.verifying_key().as_bytes()).unwrap();
        install_from_bytes(&b, &s, &h, &trust, &root, &LoadOptions::default()).unwrap();
        assert!(store.list().is_empty(), "store index is stale by design");
        store.rescan().unwrap();
        assert_eq!(store.list().len(), 1);
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&storage_root);
    }

    #[test]
    fn load_all_returns_real_skills_through_loader() {
        let (mut store, sk, root, storage_root) = fresh_store(19);
        let (b, s, h) = signed_bundle("coin-flip", &coin_md("0.1.0"), &sk);
        store.install(&b, &s, &h, &LoadOptions::default()).unwrap();

        let report = store.load_all(&LoadOptions::default()).unwrap();
        assert_eq!(report.failures.len(), 0);
        assert_eq!(report.skills.len(), 1);
        assert_eq!(report.skills[0].id(), "dev.heyari.coinflip");

        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&storage_root);
    }

    #[test]
    fn peek_extracts_id_and_version_without_writing_to_disk() {
        let bundle = make_bundle("coin-flip", &coin_md("1.2.3"));
        let (id, version) = peek_bundle_manifest(&bundle).unwrap();
        assert_eq!(id, "dev.heyari.coinflip");
        assert_eq!(version, "1.2.3");
    }

    #[test]
    fn version_compare_handles_dotted_numeric_correctly() {
        use std::cmp::Ordering::*;
        assert_eq!(compare_versions("0.1.0", "0.2.0"), Less);
        assert_eq!(compare_versions("0.2.0", "0.1.0"), Greater);
        assert_eq!(compare_versions("0.10.0", "0.2.0"), Greater);
        assert_eq!(compare_versions("1.0", "1.0.0"), Equal);
        assert_eq!(compare_versions("1.0.0", "1.0.0"), Equal);
        assert_eq!(compare_versions("1.0.0-beta", "1.0.0"), Equal);
        assert_eq!(compare_versions("2.0.0", "10.0.0"), Less);
    }
}
