//! `ari install` / `ari uninstall` / `ari list` subcommand handlers.
//!
//! These are dispatched from `main` when the first positional argument is
//! one of those keywords. Each handler parses its own little arg slice
//! (deliberately not sharing parser state with the utterance flow — they
//! take a different shape) and prints results to stdout / errors to stderr.

use ari_skill_loader::{
    check_updates, install_update, parse_capability, HostCapabilities, LoadOptions,
    RegistryClient, SkillStore, StorageConfig, StoreError, TrustRoot, REGISTRY_TRUST_KEY,
};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// Default skill store location when `--skill-store` is not given. We use a
/// stable system temp subdir so a `gen-key`/`install`/`list`/utter dance in
/// a single shell session works without flag-juggling. Production frontends
/// should always pass an explicit per-user data dir.
fn default_store_root() -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push("ari-skill-store");
    p
}

#[derive(Default)]
struct InstallArgs {
    bundle: Option<PathBuf>,
    signature: Option<PathBuf>,
    sha256: Option<String>,
    trust_key_hex: Option<String>,
    store_root: Option<PathBuf>,
    storage_dir: Option<PathBuf>,
    host_capabilities: Option<(HostCapabilities, Vec<String>)>,
}

#[derive(Default)]
struct UninstallArgs {
    id: Option<String>,
    store_root: Option<PathBuf>,
    storage_dir: Option<PathBuf>,
}

#[derive(Default)]
struct ListArgs {
    store_root: Option<PathBuf>,
}

pub fn run_install(args: &[String]) -> ExitCode {
    let parsed = match parse_install(args) {
        Ok(p) => p,
        Err(e) => return fail(&e),
    };
    let bundle_path = match parsed.bundle {
        Some(b) => b,
        None => return fail("install requires a bundle path"),
    };
    let trust_hex = match parsed.trust_key_hex {
        Some(h) => h,
        None => return fail("install requires --trust-key-hex <hex>"),
    };
    let trust_bytes = match hex_decode(&trust_hex) {
        Ok(b) => b,
        Err(e) => return fail(&format!("--trust-key-hex: {e}")),
    };
    let trust_root = match TrustRoot::single(&trust_bytes) {
        Ok(t) => t,
        Err(e) => return fail(&format!("trust key: {e}")),
    };

    let bundle_bytes = match std::fs::read(&bundle_path) {
        Ok(b) => b,
        Err(e) => return fail(&format!("read bundle {}: {e}", bundle_path.display())),
    };
    let sig_path = parsed
        .signature
        .clone()
        .unwrap_or_else(|| append_extension(&bundle_path, "sig"));
    let sig_bytes = match std::fs::read(&sig_path) {
        Ok(b) => b,
        Err(e) => return fail(&format!("read signature {}: {e}", sig_path.display())),
    };

    let expected_sha = match parsed.sha256 {
        Some(s) => s,
        None => {
            // Try <bundle>.sha256, fall back to computing it ourselves.
            let sha_path = append_extension(&bundle_path, "sha256");
            match std::fs::read_to_string(&sha_path) {
                Ok(s) => s.trim().to_string(),
                Err(_) => {
                    let mut h = Sha256::new();
                    h.update(&bundle_bytes);
                    hex_encode(&h.finalize())
                }
            }
        }
    };

    let storage_config = match parsed.storage_dir {
        Some(p) => StorageConfig::new(p),
        None => StorageConfig::ephemeral_default(),
    };
    let store_root = parsed.store_root.unwrap_or_else(default_store_root);

    let mut store = match SkillStore::open(&store_root, storage_config.clone(), trust_root) {
        Ok(s) => s,
        Err(e) => return fail(&format!("open skill store {}: {e}", store_root.display())),
    };

    let host_caps = parsed
        .host_capabilities
        .as_ref()
        .map(|(c, _)| c.clone())
        .unwrap_or_else(HostCapabilities::pure_frontend);
    let load_options = LoadOptions {
        host_capabilities: host_caps,
        storage_config,
        ..LoadOptions::default()
    };

    match store.install(&bundle_bytes, &sig_bytes, &expected_sha, &load_options) {
        Ok(installed) => {
            println!(
                "installed {} {} → {}",
                installed.id,
                installed.version,
                installed.install_dir.display()
            );
            ExitCode::SUCCESS
        }
        Err(StoreError::Downgrade {
            id,
            installed,
            attempted,
        }) => fail(&format!(
            "downgrade refused for {id}: installed {installed}, attempted {attempted}"
        )),
        Err(e) => fail(&format!("install failed: {e}")),
    }
}

pub fn run_uninstall(args: &[String]) -> ExitCode {
    let parsed = match parse_uninstall(args) {
        Ok(p) => p,
        Err(e) => return fail(&e),
    };
    let id = match parsed.id {
        Some(i) => i,
        None => return fail("uninstall requires a skill id"),
    };
    let storage_config = match parsed.storage_dir {
        Some(p) => StorageConfig::new(p),
        None => StorageConfig::ephemeral_default(),
    };
    let store_root = parsed.store_root.unwrap_or_else(default_store_root);

    // Uninstall doesn't need to verify signatures, but SkillStore::open
    // wants a TrustRoot. Empty trust root is fine — we never call install().
    let trust_root = match TrustRoot::multi(&[]) {
        Ok(t) => t,
        Err(e) => return fail(&format!("trust root: {e}")),
    };
    let mut store = match SkillStore::open(&store_root, storage_config, trust_root) {
        Ok(s) => s,
        Err(e) => return fail(&format!("open skill store {}: {e}", store_root.display())),
    };
    match store.uninstall(&id) {
        Ok(()) => {
            println!("uninstalled {id}");
            ExitCode::SUCCESS
        }
        Err(StoreError::NotInstalled { id }) => fail(&format!("not installed: {id}")),
        Err(e) => fail(&format!("uninstall failed: {e}")),
    }
}

pub fn run_check_updates(args: &[String]) -> ExitCode {
    let parsed = match parse_registry_args(args, false) {
        Ok(p) => p,
        Err(e) => return fail(&e),
    };
    let store_root = parsed.store_root.clone().unwrap_or_else(default_store_root);
    if !store_root.exists() {
        // Nothing installed → nothing to update.
        return ExitCode::SUCCESS;
    }
    let trust_root = match TrustRoot::multi(&[]) {
        Ok(t) => t,
        Err(e) => return fail(&format!("trust root: {e}")),
    };
    let store =
        match SkillStore::open(&store_root, StorageConfig::ephemeral_default(), trust_root) {
            Ok(s) => s,
            Err(e) => return fail(&format!("open skill store {}: {e}", store_root.display())),
        };
    let client = parsed.build_client();
    let index = match client.fetch_index() {
        Ok(i) => i,
        Err(e) => return fail(&format!("fetch index: {e}")),
    };
    let updates = check_updates(&store, &index);
    if updates.is_empty() {
        println!("up to date");
        return ExitCode::SUCCESS;
    }
    for u in &updates {
        println!(
            "{}\t{}\t→ {}",
            u.id, u.installed_version, u.available_version
        );
    }
    ExitCode::SUCCESS
}

pub fn run_update(args: &[String]) -> ExitCode {
    let parsed = match parse_registry_args(args, true) {
        Ok(p) => p,
        Err(e) => return fail(&e),
    };
    let id = match parsed.id.clone() {
        Some(i) => i,
        None => return fail("update requires a skill id"),
    };
    let store_root = parsed.store_root.clone().unwrap_or_else(default_store_root);
    let storage_config = match parsed.storage_dir.clone() {
        Some(p) => StorageConfig::new(p),
        None => StorageConfig::ephemeral_default(),
    };
    let trust_bytes: Vec<u8> = match &parsed.trust_key_hex {
        Some(h) => match hex_decode(h) {
            Ok(b) => b,
            Err(e) => return fail(&format!("--registry-trust-key-hex: {e}")),
        },
        None => REGISTRY_TRUST_KEY.to_vec(),
    };
    let trust_root = match TrustRoot::single(&trust_bytes) {
        Ok(t) => t,
        Err(e) => return fail(&format!("trust root: {e}")),
    };
    let mut store = match SkillStore::open(&store_root, storage_config.clone(), trust_root.clone())
    {
        Ok(s) => s,
        Err(e) => return fail(&format!("open skill store {}: {e}", store_root.display())),
    };
    let client = parsed.build_client();
    let index = match client.fetch_index() {
        Ok(i) => i,
        Err(e) => return fail(&format!("fetch index: {e}")),
    };
    let entry = match index.skills.iter().find(|e| e.id == id) {
        Some(e) => e.clone(),
        None => return fail(&format!("registry has no entry for {id}")),
    };
    let load_options = LoadOptions {
        host_capabilities: HostCapabilities::pure_frontend(),
        storage_config,
        ..LoadOptions::default()
    };
    match install_update(&client, &entry, &mut store, &trust_root, &load_options) {
        Ok(installed) => {
            println!(
                "updated {} {} → {}",
                installed.id,
                installed.version,
                installed.install_dir.display()
            );
            ExitCode::SUCCESS
        }
        Err(e) => fail(&format!("update failed: {e}")),
    }
}

#[derive(Default)]
struct RegistryArgs {
    id: Option<String>,
    store_root: Option<PathBuf>,
    storage_dir: Option<PathBuf>,
    index_url: Option<String>,
    base_url: Option<String>,
    trust_key_hex: Option<String>,
}

impl RegistryArgs {
    fn build_client(&self) -> RegistryClient {
        let mut c = RegistryClient::new();
        if let Some(u) = &self.index_url {
            c = c.with_index_url(u.clone());
        }
        if let Some(u) = &self.base_url {
            c = c.with_base_url(u.clone());
        }
        c
    }
}

fn parse_registry_args(args: &[String], wants_positional_id: bool) -> Result<RegistryArgs, String> {
    let mut out = RegistryArgs::default();
    let mut positional: Vec<String> = Vec::new();
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--skill-store" => {
                let v = iter.next().ok_or("--skill-store requires a path")?;
                out.store_root = Some(PathBuf::from(v));
            }
            o if o.starts_with("--skill-store=") => {
                out.store_root = Some(PathBuf::from(&o["--skill-store=".len()..]));
            }
            "--storage-dir" => {
                let v = iter.next().ok_or("--storage-dir requires a path")?;
                out.storage_dir = Some(PathBuf::from(v));
            }
            o if o.starts_with("--storage-dir=") => {
                out.storage_dir = Some(PathBuf::from(&o["--storage-dir=".len()..]));
            }
            "--registry-index-url" => {
                let v = iter.next().ok_or("--registry-index-url requires a URL")?;
                out.index_url = Some(v.clone());
            }
            o if o.starts_with("--registry-index-url=") => {
                out.index_url = Some(o["--registry-index-url=".len()..].to_string());
            }
            "--registry-base-url" => {
                let v = iter.next().ok_or("--registry-base-url requires a URL")?;
                out.base_url = Some(v.clone());
            }
            o if o.starts_with("--registry-base-url=") => {
                out.base_url = Some(o["--registry-base-url=".len()..].to_string());
            }
            "--registry-trust-key-hex" => {
                let v = iter.next().ok_or("--registry-trust-key-hex requires a value")?;
                out.trust_key_hex = Some(v.clone());
            }
            o if o.starts_with("--registry-trust-key-hex=") => {
                out.trust_key_hex = Some(o["--registry-trust-key-hex=".len()..].to_string());
            }
            o if o.starts_with("--") => return Err(format!("unknown option: {o}")),
            _ => positional.push(arg.clone()),
        }
    }
    if wants_positional_id {
        let mut pi = positional.into_iter();
        out.id = pi.next();
        if pi.next().is_some() {
            return Err("expected a single skill id".to_string());
        }
    } else if !positional.is_empty() {
        return Err(format!(
            "unexpected positional argument: {:?}",
            positional[0]
        ));
    }
    Ok(out)
}

pub fn run_list(args: &[String]) -> ExitCode {
    let parsed = match parse_list(args) {
        Ok(p) => p,
        Err(e) => return fail(&e),
    };
    let store_root = parsed.store_root.unwrap_or_else(default_store_root);
    if !store_root.exists() {
        // Empty list, not an error.
        return ExitCode::SUCCESS;
    }
    let trust_root = match TrustRoot::multi(&[]) {
        Ok(t) => t,
        Err(e) => return fail(&format!("trust root: {e}")),
    };
    let store = match SkillStore::open(&store_root, StorageConfig::ephemeral_default(), trust_root)
    {
        Ok(s) => s,
        Err(e) => return fail(&format!("open skill store {}: {e}", store_root.display())),
    };
    let mut listing = store.list();
    listing.sort_by(|a, b| a.id.cmp(&b.id));
    for entry in listing {
        println!(
            "{}\t{}\t{}",
            entry.id,
            entry.version,
            entry.install_dir.display()
        );
    }
    ExitCode::SUCCESS
}

fn parse_install(args: &[String]) -> Result<InstallArgs, String> {
    let mut out = InstallArgs::default();
    let mut positional: Vec<String> = Vec::new();
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--trust-key-hex" => {
                let v = iter.next().ok_or("--trust-key-hex requires a value")?;
                out.trust_key_hex = Some(v.clone());
            }
            o if o.starts_with("--trust-key-hex=") => {
                out.trust_key_hex = Some(o["--trust-key-hex=".len()..].to_string());
            }
            "--sha256" => {
                let v = iter.next().ok_or("--sha256 requires a value")?;
                out.sha256 = Some(v.clone());
            }
            o if o.starts_with("--sha256=") => {
                out.sha256 = Some(o["--sha256=".len()..].to_string());
            }
            "--skill-store" => {
                let v = iter.next().ok_or("--skill-store requires a path")?;
                out.store_root = Some(PathBuf::from(v));
            }
            o if o.starts_with("--skill-store=") => {
                out.store_root = Some(PathBuf::from(&o["--skill-store=".len()..]));
            }
            "--storage-dir" => {
                let v = iter.next().ok_or("--storage-dir requires a path")?;
                out.storage_dir = Some(PathBuf::from(v));
            }
            o if o.starts_with("--storage-dir=") => {
                out.storage_dir = Some(PathBuf::from(&o["--storage-dir=".len()..]));
            }
            "--host-capabilities" => {
                let v = iter.next().ok_or("--host-capabilities requires a list")?;
                out.host_capabilities = Some(parse_caps_csv(v)?);
            }
            o if o.starts_with("--host-capabilities=") => {
                out.host_capabilities = Some(parse_caps_csv(&o["--host-capabilities=".len()..])?);
            }
            o if o.starts_with("--") => return Err(format!("unknown option: {o}")),
            _ => positional.push(arg.clone()),
        }
    }
    let mut pi = positional.into_iter();
    out.bundle = pi.next().map(PathBuf::from);
    out.signature = pi.next().map(PathBuf::from);
    if pi.next().is_some() {
        return Err("install takes at most <bundle> <signature>".to_string());
    }
    Ok(out)
}

fn parse_uninstall(args: &[String]) -> Result<UninstallArgs, String> {
    let mut out = UninstallArgs::default();
    let mut positional: Vec<String> = Vec::new();
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--skill-store" => {
                let v = iter.next().ok_or("--skill-store requires a path")?;
                out.store_root = Some(PathBuf::from(v));
            }
            o if o.starts_with("--skill-store=") => {
                out.store_root = Some(PathBuf::from(&o["--skill-store=".len()..]));
            }
            "--storage-dir" => {
                let v = iter.next().ok_or("--storage-dir requires a path")?;
                out.storage_dir = Some(PathBuf::from(v));
            }
            o if o.starts_with("--storage-dir=") => {
                out.storage_dir = Some(PathBuf::from(&o["--storage-dir=".len()..]));
            }
            o if o.starts_with("--") => return Err(format!("unknown option: {o}")),
            _ => positional.push(arg.clone()),
        }
    }
    let mut pi = positional.into_iter();
    out.id = pi.next();
    if pi.next().is_some() {
        return Err("uninstall takes a single skill id".to_string());
    }
    Ok(out)
}

fn parse_list(args: &[String]) -> Result<ListArgs, String> {
    let mut out = ListArgs::default();
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--skill-store" => {
                let v = iter.next().ok_or("--skill-store requires a path")?;
                out.store_root = Some(PathBuf::from(v));
            }
            o if o.starts_with("--skill-store=") => {
                out.store_root = Some(PathBuf::from(&o["--skill-store=".len()..]));
            }
            o if o.starts_with("--") => return Err(format!("unknown option: {o}")),
            _ => return Err(format!("list takes no positional arguments, got {arg:?}")),
        }
    }
    Ok(out)
}

fn parse_caps_csv(value: &str) -> Result<(HostCapabilities, Vec<String>), String> {
    if value.trim().is_empty() {
        return Ok((HostCapabilities::none(), vec!["(none)".to_string()]));
    }
    let mut caps = HostCapabilities::none();
    let mut names: Vec<String> = Vec::new();
    for raw in value.split(',') {
        let name = raw.trim();
        if name.is_empty() {
            continue;
        }
        let cap =
            parse_capability(name).ok_or_else(|| format!("unknown capability: {name:?}"))?;
        caps = caps.with(cap);
        names.push(name.to_string());
    }
    Ok((caps, names))
}

fn fail(msg: &str) -> ExitCode {
    eprintln!("ari: {msg}");
    ExitCode::from(1)
}

fn append_extension(path: &Path, ext: &str) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(".");
    s.push(ext);
    PathBuf::from(s)
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    s
}

fn hex_decode(s: &str) -> Result<Vec<u8>, String> {
    let s = s.trim();
    if s.len() % 2 != 0 {
        return Err(format!("hex string has odd length {}", s.len()));
    }
    let mut out = Vec::with_capacity(s.len() / 2);
    let bytes = s.as_bytes();
    for chunk in bytes.chunks(2) {
        let hi = hex_nibble(chunk[0])?;
        let lo = hex_nibble(chunk[1])?;
        out.push((hi << 4) | lo);
    }
    Ok(out)
}

fn hex_nibble(b: u8) -> Result<u8, String> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        _ => Err(format!("non-hex character: {:?}", b as char)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::sync::atomic::{AtomicU64, Ordering};

    static N: AtomicU64 = AtomicU64::new(0);

    fn unique_dir(label: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let mut p = std::env::temp_dir();
        p.push(format!("ari-cli-test-{label}-{nanos}-{n}"));
        p
    }

    fn make_bundle(slug: &str, skill_md: &str) -> Vec<u8> {
        let mut gz = GzEncoder::new(Vec::new(), Compression::default());
        {
            let mut tar = tar::Builder::new(&mut gz);
            let mut dh = tar::Header::new_gnu();
            dh.set_path(format!("{slug}/")).unwrap();
            dh.set_size(0);
            dh.set_mode(0o755);
            dh.set_entry_type(tar::EntryType::Directory);
            dh.set_cksum();
            tar.append(&dh, std::io::empty()).unwrap();
            let body = skill_md.as_bytes();
            let mut h = tar::Header::new_gnu();
            h.set_path(format!("{slug}/SKILL.md")).unwrap();
            h.set_size(body.len() as u64);
            h.set_mode(0o644);
            h.set_entry_type(tar::EntryType::Regular);
            h.set_cksum();
            tar.append(&h, body).unwrap();
            tar.finish().unwrap();
        }
        gz.finish().unwrap()
    }

    fn coin_md() -> &'static str {
        r#"---
name: coin-flip
description: Flips a coin. Use when the user asks to flip a coin.
metadata:
  ari:
    id: dev.heyari.coinflip
    version: "0.1.0"
    engine: ">=0.3"
    matching:
      patterns:
        - keywords: [flip, coin]
          weight: 0.95
    declarative:
      response_pick: ["Heads.", "Tails."]
---
"#
    }

    /// End-to-end: write a signed bundle to disk, run the parser the way the
    /// CLI would, install via SkillStore, and confirm the install dir
    /// contains the manifest. Exercises the same path the user-facing
    /// `ari install` flow does, minus the `main` dispatch.
    #[test]
    fn install_then_list_then_uninstall_via_store_directly() {
        let tmp = unique_dir("e2e");
        std::fs::create_dir_all(&tmp).unwrap();
        let bundle_path = tmp.join("coin-flip.tar.gz");
        let bundle = make_bundle("coin-flip", coin_md());
        std::fs::write(&bundle_path, &bundle).unwrap();

        let sk = SigningKey::from_bytes(&[42u8; 32]);
        let mut hasher = Sha256::new();
        hasher.update(&bundle);
        let digest = hasher.finalize();
        let sig = sk.sign(&digest).to_bytes();
        std::fs::write(append_extension(&bundle_path, "sig"), sig).unwrap();
        std::fs::write(
            append_extension(&bundle_path, "sha256"),
            hex_encode(&digest),
        )
        .unwrap();

        let store_root = tmp.join("store");
        let trust_hex = hex_encode(sk.verifying_key().as_bytes());

        let install_args = vec![
            bundle_path.to_string_lossy().into_owned(),
            "--trust-key-hex".to_string(),
            trust_hex.clone(),
            "--skill-store".to_string(),
            store_root.to_string_lossy().into_owned(),
        ];
        let exit = run_install(&install_args);
        assert_eq!(format!("{exit:?}"), format!("{:?}", ExitCode::SUCCESS));

        let list_args = vec![
            "--skill-store".to_string(),
            store_root.to_string_lossy().into_owned(),
        ];
        let exit = run_list(&list_args);
        assert_eq!(format!("{exit:?}"), format!("{:?}", ExitCode::SUCCESS));

        // Uninstall should remove it
        let uninstall_args = vec![
            "dev.heyari.coinflip".to_string(),
            "--skill-store".to_string(),
            store_root.to_string_lossy().into_owned(),
        ];
        let exit = run_uninstall(&uninstall_args);
        assert_eq!(format!("{exit:?}"), format!("{:?}", ExitCode::SUCCESS));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn install_rejects_missing_trust_key() {
        let tmp = unique_dir("notrust");
        std::fs::create_dir_all(&tmp).unwrap();
        let bundle_path = tmp.join("x.tar.gz");
        std::fs::write(&bundle_path, b"junk").unwrap();
        let exit = run_install(&[bundle_path.to_string_lossy().into_owned()]);
        // Missing trust key is exit 1
        assert_eq!(format!("{exit:?}"), format!("{:?}", ExitCode::from(1)));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn install_rejects_bad_signature_path() {
        let tmp = unique_dir("badsig");
        std::fs::create_dir_all(&tmp).unwrap();
        let bundle_path = tmp.join("x.tar.gz");
        std::fs::write(&bundle_path, b"junk").unwrap();
        // No .sig file alongside, no explicit signature path → fail.
        let args = vec![
            bundle_path.to_string_lossy().into_owned(),
            "--trust-key-hex".to_string(),
            hex_encode(&[0u8; 32]),
        ];
        let exit = run_install(&args);
        assert_eq!(format!("{exit:?}"), format!("{:?}", ExitCode::from(1)));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn list_on_missing_store_is_success_with_no_output() {
        let tmp = unique_dir("emptylist");
        let exit = run_list(&[
            "--skill-store".to_string(),
            tmp.to_string_lossy().into_owned(),
        ]);
        assert_eq!(format!("{exit:?}"), format!("{:?}", ExitCode::SUCCESS));
    }

    #[test]
    fn uninstall_unknown_id_fails_loudly() {
        let tmp = unique_dir("nope");
        std::fs::create_dir_all(&tmp).unwrap();
        let exit = run_uninstall(&[
            "dev.heyari.nope".to_string(),
            "--skill-store".to_string(),
            tmp.to_string_lossy().into_owned(),
        ]);
        assert_eq!(format!("{exit:?}"), format!("{:?}", ExitCode::from(1)));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn hex_decode_round_trip() {
        let bytes = [0u8, 0xff, 0xab, 0x10, 0x7f];
        let s = hex_encode(&bytes);
        assert_eq!(s, "00ffab107f");
        assert_eq!(hex_decode(&s).unwrap(), bytes);
        assert_eq!(hex_decode("00FFAB107F").unwrap(), bytes);
        assert!(hex_decode("0").is_err());
        assert!(hex_decode("zz").is_err());
    }

    #[test]
    fn install_parser_handles_equals_form() {
        let parsed = parse_install(&[
            "bundle.tar.gz".to_string(),
            "--trust-key-hex=deadbeef".to_string(),
            "--skill-store=/tmp/store".to_string(),
        ])
        .unwrap();
        assert_eq!(parsed.bundle, Some(PathBuf::from("bundle.tar.gz")));
        assert_eq!(parsed.trust_key_hex.as_deref(), Some("deadbeef"));
        assert_eq!(parsed.store_root, Some(PathBuf::from("/tmp/store")));
    }
}
