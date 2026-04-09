//! Skill bundle install pipeline: hash → verify → extract → atomic swap.
//!
//! A bundle is a gzip-compressed tar archive of a single skill directory:
//!
//! ```text
//! weather-1.2.0.tar.gz
//!   weather/
//!     SKILL.md
//!     skill.wasm
//!     assets/icon.png
//!     ...
//! ```
//!
//! The install flow:
//!
//! 1. **Hash**: SHA-256 the bundle bytes.
//! 2. **Verify**: check the supplied signature against the hash using the
//!    [`TrustRoot`]. The signature signs the **hash**, not the bytes — that
//!    means a tampered bundle changes the hash, which causes verification
//!    to fail.
//! 3. **Extract** to a *temporary* directory inside the destination root.
//!    Tar entries are validated against path-traversal attacks (no `..`,
//!    no absolute paths, no symlinks) before being written.
//! 4. **Validate** the extracted skill — load `SKILL.md`, run the manifest
//!    + capability checks via the same loader code the engine uses at
//!    startup. We don't actually instantiate the WASM here; that happens
//!    when the engine adds the skill to its ranking pipeline. Validation
//!    here is just "does it parse and pass the cap check".
//! 5. **Atomic swap**: rename the temp directory into its final per-skill
//!    location. If a previous version was installed, it's renamed to
//!    `<id>.old` first, the new one is moved into place, then the old one
//!    is deleted. A power loss at any point leaves either the old version
//!    or the new version in place, never half of each.
//!
//! Anything that goes wrong before the swap leaves the destination directory
//! untouched. After the swap, the only failure mode is failing to delete the
//! `.old` directory, which is a leak but not a corruption.

use crate::loader::{load_single_skill_dir_with, LoadOptions};
use crate::signature::{SignatureError, TrustRoot};
use sha2::{Digest, Sha256};
use std::path::{Component, Path, PathBuf};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum BundleError {
    #[error("could not read bundle file {path:?}: {source}")]
    ReadBundle {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("could not read signature file {path:?}: {source}")]
    ReadSignature {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("expected sha256 {expected}, got {actual}")]
    HashMismatch { expected: String, actual: String },

    #[error("signature verification failed: {0}")]
    Signature(#[from] SignatureError),

    #[error("could not decompress bundle: {0}")]
    Decompress(String),

    #[error("could not read tar entry: {0}")]
    Tar(String),

    #[error(
        "tar entry path {path:?} is unsafe (absolute, contains ..,or otherwise tries to escape the bundle root)"
    )]
    UnsafePath { path: PathBuf },

    #[error("tar entry {path:?} is a {kind} which is not allowed in skill bundles")]
    UnsafeEntryType { path: PathBuf, kind: &'static str },

    #[error("bundle root contains no top-level directory (expected exactly one)")]
    NoBundleRoot,

    #[error("bundle root contains multiple top-level directories: {found:?}")]
    MultipleBundleRoots { found: Vec<String> },

    #[error("bundle is missing SKILL.md at the bundle root")]
    MissingSkillFile,

    #[error("io error during install: {0}")]
    Io(#[from] std::io::Error),

    #[error("extracted skill failed validation: {0}")]
    Validation(String),
}

/// Result of a successful install. The bundle is now extracted at
/// [`Self::install_dir`] and ready to be loaded by the engine.
#[derive(Debug)]
pub struct InstalledBundle {
    /// The directory the bundle was extracted into. Contains SKILL.md and
    /// any other files the bundle shipped.
    pub install_dir: PathBuf,
    /// The skill ID from the manifest, for convenience.
    pub skill_id: String,
}

/// Compute the SHA-256 of `bytes` and return it as a lowercase hex string.
/// We use the hex string in errors and bundle metadata, not the raw bytes.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex_encode(&hasher.finalize())
}

/// Lowercase hex encoder. Avoids pulling in a `hex` crate dependency for
/// what's basically a 5-line loop.
fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    s
}

/// Install a bundle from in-memory bytes. Returns details about the freshly
/// installed skill on success.
///
/// `expected_sha256` is the hex-encoded hash from `index.json`. If the
/// computed hash doesn't match, install aborts before signature verification.
/// (Both checks are required; the hash check catches transit corruption
/// fast, the signature check catches everything else.)
///
/// `dest_root` is the directory under which skills are installed. Each skill
/// gets its own subdirectory named after the slug from the bundle.
pub fn install_from_bytes(
    bundle_bytes: &[u8],
    signature_bytes: &[u8],
    expected_sha256: &str,
    trust_root: &TrustRoot,
    dest_root: &Path,
    load_options: &LoadOptions,
) -> Result<InstalledBundle, BundleError> {
    // Step 1: hash and compare.
    let actual = sha256_hex(bundle_bytes);
    if !constant_time_eq_str(&actual, expected_sha256) {
        return Err(BundleError::HashMismatch {
            expected: expected_sha256.to_string(),
            actual,
        });
    }

    // Step 2: signature verification.
    let mut hasher = Sha256::new();
    hasher.update(bundle_bytes);
    let digest = hasher.finalize();
    trust_root.verify(&digest, signature_bytes)?;

    // Step 3: extract to a temp directory inside dest_root. We use a temp
    // dir alongside the final destination so the rename in step 5 is on the
    // same filesystem (otherwise it'd be a copy, not a rename, and not
    // atomic).
    std::fs::create_dir_all(dest_root)?;
    let staging = make_staging_dir(dest_root)?;
    let extract_result = extract_validated(bundle_bytes, &staging);
    let bundle_root = match extract_result {
        Ok(p) => p,
        Err(e) => {
            // Clean up the half-extracted staging dir on error.
            let _ = std::fs::remove_dir_all(&staging);
            return Err(e);
        }
    };

    // Step 4: parse the manifest. We do this through the loader so the same
    // checks the engine uses at startup also gate install. We don't keep
    // the loaded skill — the engine will rediscover it from disk after the
    // swap.
    let report = load_single_skill_dir_with(&bundle_root, load_options);
    if !report.failures.is_empty() {
        let msg = report
            .failures
            .iter()
            .map(|f| f.to_string())
            .collect::<Vec<_>>()
            .join("; ");
        let _ = std::fs::remove_dir_all(&staging);
        return Err(BundleError::Validation(msg));
    }
    let skill = match report.skills.first() {
        Some(s) => s,
        None => {
            // The bundle had a SKILL.md but it didn't yield an Ari skill
            // (e.g. it was a plain AgentSkills doc with no `metadata.ari`).
            let _ = std::fs::remove_dir_all(&staging);
            return Err(BundleError::Validation(
                "bundle contains a SKILL.md but no metadata.ari extension".to_string(),
            ));
        }
    };
    let skill_id = skill.id().to_string();
    drop(report); // releases the loaded skill instance

    // Step 5: atomic swap. The bundle root inside staging is named after
    // the AgentSkills slug; we promote it to its final location.
    let final_name = bundle_root
        .file_name()
        .ok_or_else(|| BundleError::NoBundleRoot)?
        .to_owned();
    let final_path = dest_root.join(&final_name);
    let backup_path = dest_root.join(format!("{}.old", final_name.to_string_lossy()));

    // If a backup exists from a previous failed install, blow it away first.
    if backup_path.exists() {
        let _ = std::fs::remove_dir_all(&backup_path);
    }

    if final_path.exists() {
        std::fs::rename(&final_path, &backup_path)?;
    }

    // The actual atomic moment: rename staging/<slug> to dest_root/<slug>.
    let move_result = std::fs::rename(&bundle_root, &final_path);

    if let Err(e) = move_result {
        // Move failed. Try to restore the backup so we don't leave the user
        // without their previously installed skill.
        if backup_path.exists() {
            let _ = std::fs::rename(&backup_path, &final_path);
        }
        let _ = std::fs::remove_dir_all(&staging);
        return Err(BundleError::Io(e));
    }

    // Move succeeded. Clean up: drop the backup and the now-empty staging.
    if backup_path.exists() {
        let _ = std::fs::remove_dir_all(&backup_path);
    }
    let _ = std::fs::remove_dir_all(&staging);

    Ok(InstalledBundle {
        install_dir: final_path,
        skill_id,
    })
}

/// Constant-time string comparison. Hex-encoded SHA-256 is small enough that
/// the variable-time `==` would only leak a millisecond or two of side-channel
/// information, but it's free to do this right.
fn constant_time_eq_str(a: &str, b: &str) -> bool {
    let a = a.as_bytes();
    let b = b.as_bytes();
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

fn make_staging_dir(dest_root: &Path) -> std::io::Result<PathBuf> {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let pid = std::process::id();
    let staging = dest_root.join(format!(".staging-{pid}-{nanos}"));
    std::fs::create_dir_all(&staging)?;
    Ok(staging)
}

/// Extract the gzipped tar bytes into `staging`, validating each entry's
/// path before writing it. Returns the path of the single top-level
/// directory inside the extracted bundle (the bundle root).
fn extract_validated(bundle_bytes: &[u8], staging: &Path) -> Result<PathBuf, BundleError> {
    let gz = flate2::read::GzDecoder::new(bundle_bytes);
    let mut archive = tar::Archive::new(gz);

    // Don't let the tar crate touch ownership / permissions / mtime in
    // surprising ways.
    archive.set_preserve_permissions(false);
    archive.set_preserve_mtime(false);
    archive.set_overwrite(false);

    let mut top_levels: std::collections::BTreeSet<String> = Default::default();

    for entry_res in archive.entries().map_err(|e| BundleError::Tar(e.to_string()))? {
        let mut entry = entry_res.map_err(|e| BundleError::Tar(e.to_string()))?;
        let header = entry.header();
        let entry_path = entry
            .path()
            .map_err(|e| BundleError::Tar(e.to_string()))?
            .into_owned();

        // Reject anything that isn't a regular file or a directory. No
        // symlinks, no hardlinks, no character/block devices, no fifos.
        let kind = header.entry_type();
        let kind_name = match kind {
            tar::EntryType::Regular => "regular",
            tar::EntryType::Directory => "directory",
            tar::EntryType::Symlink => {
                return Err(BundleError::UnsafeEntryType {
                    path: entry_path,
                    kind: "symlink",
                })
            }
            tar::EntryType::Link => {
                return Err(BundleError::UnsafeEntryType {
                    path: entry_path,
                    kind: "hardlink",
                })
            }
            tar::EntryType::Char => {
                return Err(BundleError::UnsafeEntryType {
                    path: entry_path,
                    kind: "character device",
                })
            }
            tar::EntryType::Block => {
                return Err(BundleError::UnsafeEntryType {
                    path: entry_path,
                    kind: "block device",
                })
            }
            tar::EntryType::Fifo => {
                return Err(BundleError::UnsafeEntryType {
                    path: entry_path,
                    kind: "fifo",
                })
            }
            _ => {
                return Err(BundleError::UnsafeEntryType {
                    path: entry_path,
                    kind: "unknown",
                })
            }
        };
        let _ = kind_name;

        // Validate the path: no absolute, no `..`, no empty.
        if !is_safe_relative_path(&entry_path) {
            return Err(BundleError::UnsafePath { path: entry_path });
        }

        // Identify the top-level directory.
        if let Some(first) = entry_path.components().next() {
            if let Component::Normal(name) = first {
                top_levels.insert(name.to_string_lossy().into_owned());
            }
        }

        // Unpack into staging.
        entry
            .unpack_in(staging)
            .map_err(|e| BundleError::Tar(e.to_string()))?;
    }

    if top_levels.is_empty() {
        return Err(BundleError::NoBundleRoot);
    }
    if top_levels.len() > 1 {
        return Err(BundleError::MultipleBundleRoots {
            found: top_levels.into_iter().collect(),
        });
    }
    let root_name = top_levels.into_iter().next().unwrap();
    let bundle_root = staging.join(&root_name);

    if !bundle_root.join("SKILL.md").is_file() {
        return Err(BundleError::MissingSkillFile);
    }
    Ok(bundle_root)
}

fn is_safe_relative_path(path: &Path) -> bool {
    if path.as_os_str().is_empty() {
        return false;
    }
    if path.is_absolute() {
        return false;
    }
    for c in path.components() {
        match c {
            Component::Normal(_) => {}
            // RootDir / Prefix mean absolute, ParentDir is `..`, CurDir is
            // `.`. None of these are safe in a tar entry.
            _ => return false,
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host_capabilities::HostCapabilities;
    use crate::manifest::Capability;
    use crate::signature::TrustRoot;
    use ed25519_dalek::{Signer, SigningKey};
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};

    static N: AtomicU64 = AtomicU64::new(0);

    fn unique_dest() -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let mut p = std::env::temp_dir();
        p.push(format!("ari-bundle-test-{nanos}-{n}"));
        p
    }

    fn fixed_keypair(seed: u8) -> SigningKey {
        SigningKey::from_bytes(&[seed; 32])
    }

    /// Build a tar.gz bundle with a single SKILL.md inside `<slug>/`.
    fn make_bundle(slug: &str, skill_md: &str) -> Vec<u8> {
        let mut gz = GzEncoder::new(Vec::new(), Compression::default());
        {
            let mut tar = tar::Builder::new(&mut gz);

            // Add the directory entry first so order is deterministic.
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

    fn coin_flip_md() -> &'static str {
        r#"---
name: coin-flip
description: Flips a coin. Use when the user asks to flip a coin.
license: MIT
metadata:
  ari:
    id: dev.heyari.coinflip
    version: "0.1.0"
    engine: ">=0.3"
    languages: [en]
    specificity: high
    matching:
      patterns:
        - keywords: [flip, coin]
          weight: 0.95
    declarative:
      response_pick: ["Heads.", "Tails."]
---
"#
    }

    fn http_skill_md() -> &'static str {
        r#"---
name: needs-http
description: Needs http. Use never.
metadata:
  ari:
    id: dev.heyari.needshttp
    version: "0.1.0"
    engine: ">=0.3"
    capabilities: [http]
    matching:
      patterns:
        - keywords: [http]
    declarative:
      response: "x"
---
"#
    }

    /// Bundle + signature + trust root for a happy-path install. The
    /// returned tuple is (bundle_bytes, sig_bytes, sha256_hex, trust_root).
    fn happy_path_inputs(
        slug: &str,
        skill_md: &str,
        seed: u8,
    ) -> (Vec<u8>, Vec<u8>, String, TrustRoot) {
        let bundle = make_bundle(slug, skill_md);
        let hash = sha256_hex(&bundle);
        let mut hasher = Sha256::new();
        hasher.update(&bundle);
        let digest = hasher.finalize();
        let sk = fixed_keypair(seed);
        let sig = sk.sign(&digest).to_bytes().to_vec();
        let trust = TrustRoot::single(sk.verifying_key().as_bytes()).unwrap();
        (bundle, sig, hash, trust)
    }

    #[test]
    fn happy_path_install_extracts_and_validates() {
        let dest = unique_dest();
        let (bundle, sig, hash, trust) = happy_path_inputs("coin-flip", coin_flip_md(), 1);

        let result = install_from_bytes(
            &bundle,
            &sig,
            &hash,
            &trust,
            &dest,
            &LoadOptions::default(),
        )
        .unwrap();
        assert_eq!(result.skill_id, "dev.heyari.coinflip");
        assert!(result.install_dir.join("SKILL.md").is_file());
        // Cleanup
        let _ = std::fs::remove_dir_all(&dest);
    }

    #[test]
    fn hash_mismatch_aborts_before_signature_check() {
        let dest = unique_dest();
        let (bundle, sig, _hash, trust) = happy_path_inputs("coin-flip", coin_flip_md(), 2);
        let wrong = "ff".repeat(32);
        let err = install_from_bytes(
            &bundle,
            &sig,
            &wrong,
            &trust,
            &dest,
            &LoadOptions::default(),
        )
        .unwrap_err();
        match err {
            BundleError::HashMismatch { expected, actual: _ } => assert_eq!(expected, wrong),
            other => panic!("expected HashMismatch, got {other:?}"),
        }
        // Nothing should have been extracted
        assert!(!dest.join("coin-flip").exists());
        let _ = std::fs::remove_dir_all(&dest);
    }

    #[test]
    fn bad_signature_rejected() {
        let dest = unique_dest();
        let (bundle, _good_sig, hash, trust) = happy_path_inputs("coin-flip", coin_flip_md(), 3);
        // Sign with a different key
        let impostor = fixed_keypair(99);
        let mut hasher = Sha256::new();
        hasher.update(&bundle);
        let bad_sig = impostor.sign(&hasher.finalize()).to_bytes().to_vec();

        let err = install_from_bytes(
            &bundle,
            &bad_sig,
            &hash,
            &trust,
            &dest,
            &LoadOptions::default(),
        )
        .unwrap_err();
        assert!(matches!(err, BundleError::Signature(_)));
        assert!(!dest.join("coin-flip").exists());
        let _ = std::fs::remove_dir_all(&dest);
    }

    #[test]
    fn tampered_bundle_changes_hash_and_is_rejected() {
        let dest = unique_dest();
        let (mut bundle, sig, hash, trust) = happy_path_inputs("coin-flip", coin_flip_md(), 4);
        // Flip a byte. Even one bit changes the hash and breaks verification.
        let mid = bundle.len() / 2;
        bundle[mid] ^= 1;
        // The expected hash from index.json is the original; the actual
        // computed hash will differ.
        let err = install_from_bytes(
            &bundle,
            &sig,
            &hash,
            &trust,
            &dest,
            &LoadOptions::default(),
        )
        .unwrap_err();
        assert!(matches!(err, BundleError::HashMismatch { .. }));
        let _ = std::fs::remove_dir_all(&dest);
    }

    #[test]
    fn validation_failure_aborts_before_swap_and_leaves_dest_clean() {
        // A bundle whose SKILL.md declares a capability the host doesn't
        // grant — install should reject after extraction but before any
        // permanent state changes.
        let dest = unique_dest();
        let (bundle, sig, hash, trust) = happy_path_inputs("needs-http", http_skill_md(), 5);

        let err = install_from_bytes(
            &bundle,
            &sig,
            &hash,
            &trust,
            &dest,
            &LoadOptions::default(), // pure_frontend, no http
        )
        .unwrap_err();
        match err {
            BundleError::Validation(msg) => assert!(msg.contains("Http"), "msg: {msg}"),
            other => panic!("expected Validation, got {other:?}"),
        }

        // The final install dir must not exist; the staging dir must be cleaned.
        assert!(!dest.join("needs-http").exists());
        let staging_count = std::fs::read_dir(&dest)
            .map(|it| it.filter_map(Result::ok).count())
            .unwrap_or(0);
        assert_eq!(
            staging_count, 0,
            "dest should be empty after validation failure"
        );
        let _ = std::fs::remove_dir_all(&dest);
    }

    #[test]
    fn upgrade_replaces_old_install_atomically() {
        let dest = unique_dest();

        // First install: v0.1.0
        let v1 = make_bundle("coin-flip", coin_flip_md());
        let h1 = sha256_hex(&v1);
        let sk = fixed_keypair(6);
        let mut hasher = Sha256::new();
        hasher.update(&v1);
        let s1 = sk.sign(&hasher.finalize()).to_bytes().to_vec();
        let trust = TrustRoot::single(sk.verifying_key().as_bytes()).unwrap();

        install_from_bytes(&v1, &s1, &h1, &trust, &dest, &LoadOptions::default()).unwrap();
        let original_skill = dest.join("coin-flip").join("SKILL.md");
        assert!(original_skill.is_file());
        let v1_contents = std::fs::read_to_string(&original_skill).unwrap();
        assert!(v1_contents.contains("0.1.0"));

        // Second install: v0.2.0 with a different description so we can
        // detect the swap.
        let v2_md = coin_flip_md().replace("0.1.0", "0.2.0").replace(
            "Flips a coin. Use when the user asks to flip a coin.",
            "Flips a coin. Now twice as flippable. Use when the user asks to flip a coin.",
        );
        let v2 = make_bundle("coin-flip", &v2_md);
        let h2 = sha256_hex(&v2);
        let mut hasher2 = Sha256::new();
        hasher2.update(&v2);
        let s2 = sk.sign(&hasher2.finalize()).to_bytes().to_vec();

        install_from_bytes(&v2, &s2, &h2, &trust, &dest, &LoadOptions::default()).unwrap();

        // The install dir still exists with the new contents.
        let new_contents = std::fs::read_to_string(&original_skill).unwrap();
        assert!(new_contents.contains("0.2.0"));
        assert!(new_contents.contains("twice as flippable"));
        // The .old backup should have been cleaned up.
        assert!(!dest.join("coin-flip.old").exists());
        let _ = std::fs::remove_dir_all(&dest);
    }

    #[test]
    fn unsafe_paths_are_rejected() {
        // We can't easily build a "../etc/passwd" tar with the high-level
        // tar Builder (it normalises paths). So instead we test the path
        // validator directly with a wider range of inputs.
        assert!(!is_safe_relative_path(Path::new("")));
        assert!(!is_safe_relative_path(Path::new("/etc/passwd")));
        assert!(!is_safe_relative_path(Path::new("../escape")));
        assert!(!is_safe_relative_path(Path::new("./relative")));
        assert!(!is_safe_relative_path(Path::new("foo/../bar")));
        assert!(is_safe_relative_path(Path::new("weather/SKILL.md")));
        assert!(is_safe_relative_path(Path::new("weather/assets/icon.png")));
        assert!(is_safe_relative_path(Path::new("weather")));
    }

    #[test]
    fn empty_bundle_root_rejected() {
        // A bundle with no top-level directory is malformed.
        let mut gz = GzEncoder::new(Vec::new(), Compression::default());
        {
            let tar = tar::Builder::new(&mut gz);
            // No entries
            tar.into_inner().unwrap();
        }
        let bundle = gz.finish().unwrap();

        let dest = unique_dest();
        let hash = sha256_hex(&bundle);
        let sk = fixed_keypair(7);
        let mut hasher = Sha256::new();
        hasher.update(&bundle);
        let sig = sk.sign(&hasher.finalize()).to_bytes().to_vec();
        let trust = TrustRoot::single(sk.verifying_key().as_bytes()).unwrap();

        let err = install_from_bytes(
            &bundle,
            &sig,
            &hash,
            &trust,
            &dest,
            &LoadOptions::default(),
        )
        .unwrap_err();
        assert!(matches!(err, BundleError::NoBundleRoot));
        let _ = std::fs::remove_dir_all(&dest);
    }

    #[test]
    fn bundle_without_skill_md_rejected() {
        // A bundle whose top dir doesn't contain SKILL.md.
        let mut gz = GzEncoder::new(Vec::new(), Compression::default());
        {
            let mut tar = tar::Builder::new(&mut gz);
            let mut dir_hdr = tar::Header::new_gnu();
            dir_hdr.set_path("weather/").unwrap();
            dir_hdr.set_size(0);
            dir_hdr.set_mode(0o755);
            dir_hdr.set_entry_type(tar::EntryType::Directory);
            dir_hdr.set_cksum();
            tar.append(&dir_hdr, std::io::empty()).unwrap();
            // Add a junk file but not SKILL.md
            let body = b"not a skill manifest";
            let mut hdr = tar::Header::new_gnu();
            hdr.set_path("weather/README.md").unwrap();
            hdr.set_size(body.len() as u64);
            hdr.set_mode(0o644);
            hdr.set_entry_type(tar::EntryType::Regular);
            hdr.set_cksum();
            tar.append(&hdr, &body[..]).unwrap();
            tar.finish().unwrap();
        }
        let bundle = gz.finish().unwrap();

        let dest = unique_dest();
        let hash = sha256_hex(&bundle);
        let sk = fixed_keypair(8);
        let mut hasher = Sha256::new();
        hasher.update(&bundle);
        let sig = sk.sign(&hasher.finalize()).to_bytes().to_vec();
        let trust = TrustRoot::single(sk.verifying_key().as_bytes()).unwrap();

        let err = install_from_bytes(
            &bundle,
            &sig,
            &hash,
            &trust,
            &dest,
            &LoadOptions::default(),
        )
        .unwrap_err();
        assert!(matches!(err, BundleError::MissingSkillFile));
        let _ = std::fs::remove_dir_all(&dest);
    }

    #[test]
    fn sha256_hex_matches_known_vector() {
        // SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        // SHA-256("abc") = ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn hex_encoder_is_lowercase() {
        assert_eq!(hex_encode(&[0x00, 0xff, 0xab]), "00ffab");
    }

    #[test]
    fn constant_time_eq_str_handles_length_mismatch() {
        assert!(!constant_time_eq_str("abc", "abcd"));
        assert!(!constant_time_eq_str("abcd", "abc"));
        assert!(constant_time_eq_str("abcdef", "abcdef"));
        assert!(!constant_time_eq_str("abcdef", "abcdeg"));
    }

    #[test]
    fn capability_check_uses_passed_load_options() {
        // Same bundle as the validation-failure test, but with http granted.
        let dest = unique_dest();
        let (bundle, sig, hash, trust) = happy_path_inputs("needs-http", http_skill_md(), 9);

        let mut opts = LoadOptions::default();
        opts.host_capabilities = HostCapabilities::pure_frontend().with(Capability::Http);

        let result = install_from_bytes(&bundle, &sig, &hash, &trust, &dest, &opts).unwrap();
        assert_eq!(result.skill_id, "dev.heyari.needshttp");
        let _ = std::fs::remove_dir_all(&dest);
    }
}
