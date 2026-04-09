//! Skill registry client: fetch `index.json`, diff against what's installed,
//! download + install updates.
//!
//! The registry lives at
//! `https://raw.githubusercontent.com/ari-digital-assistant/ari-skills/main/`.
//! [`REGISTRY_INDEX_URL`] is the JSON index, and every bundle/signature path
//! inside that index is **relative to the same base URL** — the client just
//! concatenates.
//!
//! The registry signing public key is baked into this crate as
//! [`REGISTRY_TRUST_KEY`]. Every bundle the client downloads is verified
//! against this key via the usual [`crate::bundle::install_from_bytes`]
//! pipeline (which means the bundle pipeline doesn't need to care whether
//! bytes came from a sideload or from the network).
//!
//! This module deliberately only covers the **engine-side machinery**. The
//! background scheduling (Android WorkManager, etc.) and the UI that shows
//! "N updates available" live in the frontend crates. What this module
//! provides is: fetch, diff, install. Nothing more.

use crate::bundle::{install_from_bytes, BundleError};
use crate::loader::LoadOptions;
use crate::signature::TrustRoot;
use crate::store::{compare_versions, InstalledSkill, SkillStore};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use thiserror::Error;

/// URL the client fetches `index.json` from by default.
pub const REGISTRY_INDEX_URL: &str =
    "https://raw.githubusercontent.com/ari-digital-assistant/ari-skills/main/index.json";

/// Base URL every bundle/signature path in `index.json` resolves against.
/// Trailing slash is required so simple string concatenation is safe.
pub const REGISTRY_BASE_URL: &str =
    "https://raw.githubusercontent.com/ari-digital-assistant/ari-skills/main/";

/// Ed25519 public key that signs every bundle published by the registry's
/// `sign-and-publish.yml` workflow. Generated with `ari-sign-bundle gen-key`
/// on 2026-04-08. The private half lives only in the `ari-skills` repo's
/// `ARI_REGISTRY_SIGNING_KEY` GitHub Actions secret.
pub const REGISTRY_TRUST_KEY: [u8; 32] = [
    0xfe, 0x55, 0x95, 0x3e, 0x03, 0x00, 0xd9, 0x1d, 0x06, 0xc1, 0xce, 0x90, 0x34, 0x92, 0x66, 0x1d,
    0x67, 0xd3, 0x43, 0xef, 0x5b, 0x6c, 0x30, 0x4e, 0x0b, 0x49, 0x5c, 0x00, 0xb4, 0x7b, 0xf5, 0x71,
];

/// Default HTTP timeout for registry requests. The index is small; bundles
/// are small. 30 seconds is plenty and keeps a dead network from hanging the
/// background updater forever.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// Maximum size of `index.json` the client will accept. Real registries
/// produce kilobyte-scale indexes; anything over this is either a
/// misconfiguration or a malicious MITM trying to wedge the parser.
const MAX_INDEX_BYTES: usize = 4 * 1024 * 1024;

/// Maximum size of a single bundle the client will download. Matches the
/// install pipeline's sanity threshold — an 8 MiB bundle is already
/// enormous for a voice-assistant skill.
const MAX_BUNDLE_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum RegistryError {
    #[error("http error: {0}")]
    Http(String),

    #[error("registry returned status {status} for {url}")]
    BadStatus { url: String, status: u16 },

    #[error("response from {url} exceeds {limit} bytes")]
    TooLarge { url: String, limit: usize },

    #[error("could not parse index.json: {0}")]
    Parse(String),

    #[error("unsupported index version {found}, this client understands {expected}")]
    UnsupportedIndexVersion { expected: u32, found: u32 },

    #[error("index entry {id} sha256 mismatch: index says {expected}, downloaded bundle is {actual}")]
    ShaMismatch {
        id: String,
        expected: String,
        actual: String,
    },

    #[error("skill id {id} is not present in the registry index")]
    NotFound { id: String },

    #[error(transparent)]
    Install(#[from] BundleError),
}

/// The version of the index format this client understands. Bumped if/when
/// the JSON shape on disk changes incompatibly.
const SUPPORTED_INDEX_VERSION: u32 = 1;

/// Parsed `index.json` contents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Index {
    pub index_version: u32,
    pub generated_at: String,
    pub skills: Vec<IndexEntry>,
}

/// One row of `index.json`.
///
/// All of the descriptive fields after `description` are optional with
/// `#[serde(default)]`. This keeps the schema **additively** extensible
/// without needing to bump [`SUPPORTED_INDEX_VERSION`]: old clients
/// ignore fields they don't understand, and new clients reading an old
/// index just see empty / `None` values. That's why the browse UI has
/// to tolerate missing metadata gracefully.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexEntry {
    pub id: String,
    pub version: String,
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub license: Option<String>,
    /// Free-text author / maintainer string from SKILL.md frontmatter.
    #[serde(default)]
    pub author: Option<String>,
    /// Optional homepage or source URL. UI should treat this as
    /// unverified — the client doesn't validate the scheme.
    #[serde(default)]
    pub homepage: Option<String>,
    /// Stable capability names the skill declares (e.g. `http`,
    /// `storage_kv`). Useful so the browse detail view can surface what
    /// a skill will be allowed to do *before* the user commits to
    /// installing it.
    #[serde(default)]
    pub capabilities: Vec<String>,
    /// BCP-47-ish language tags the skill ships matcher patterns for.
    #[serde(default)]
    pub languages: Vec<String>,
    /// Path to the bundle relative to [`REGISTRY_BASE_URL`].
    pub bundle: String,
    /// Path to the detached signature relative to [`REGISTRY_BASE_URL`].
    pub signature: String,
    /// Hex-encoded SHA-256 of the bundle bytes.
    pub sha256: String,
}

/// One available update for an already-installed skill, as reported by
/// [`check_updates`].
#[derive(Debug, Clone)]
pub struct AvailableUpdate {
    pub id: String,
    pub installed_version: String,
    pub available_version: String,
    pub entry: IndexEntry,
}

/// Blocking HTTP client for the registry. Intentionally small and
/// dependency-light: reuses the `reqwest` blocking client that the WASM
/// http_fetch import already pulls in, so no extra crates.
pub struct RegistryClient {
    index_url: String,
    base_url: String,
    http: reqwest::blocking::Client,
}

impl Default for RegistryClient {
    fn default() -> Self {
        Self::new()
    }
}

impl RegistryClient {
    pub fn new() -> Self {
        Self {
            index_url: REGISTRY_INDEX_URL.to_string(),
            base_url: REGISTRY_BASE_URL.to_string(),
            http: reqwest::blocking::Client::builder()
                .timeout(DEFAULT_TIMEOUT)
                .user_agent("ari-skill-loader/registry")
                .use_preconfigured_tls(crate::tls::webpki_roots_config())
                .build()
                .expect("default reqwest client"),
        }
    }

    /// Override the index URL. Used by the CLI's `--registry-index-url`
    /// flag for running against a local registry or a fork, and by the
    /// unit tests below to point at an in-process HTTP server.
    pub fn with_index_url(mut self, url: impl Into<String>) -> Self {
        self.index_url = url.into();
        self
    }

    /// Override the base URL against which bundle/signature paths from
    /// `index.json` are resolved. Must end with a `/`.
    pub fn with_base_url(mut self, base: impl Into<String>) -> Self {
        let mut base = base.into();
        if !base.ends_with('/') {
            base.push('/');
        }
        self.base_url = base;
        self
    }

    pub fn index_url(&self) -> &str {
        &self.index_url
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// GET the index and parse it. Enforces the supported-version check and
    /// the size cap. Does not validate individual entries — the bundle
    /// install pipeline does that at install time.
    pub fn fetch_index(&self) -> Result<Index, RegistryError> {
        let bytes = self.get_bytes(&self.index_url, MAX_INDEX_BYTES)?;
        let index: Index = serde_json::from_slice(&bytes)
            .map_err(|e| RegistryError::Parse(e.to_string()))?;
        if index.index_version != SUPPORTED_INDEX_VERSION {
            return Err(RegistryError::UnsupportedIndexVersion {
                expected: SUPPORTED_INDEX_VERSION,
                found: index.index_version,
            });
        }
        Ok(index)
    }

    /// Download the bundle and its detached signature for one index entry.
    /// Returns the raw bytes — signature verification happens inside
    /// [`install_from_bytes`].
    pub fn fetch_bundle(&self, entry: &IndexEntry) -> Result<(Vec<u8>, Vec<u8>), RegistryError> {
        let bundle_url = self.resolve(&entry.bundle);
        let sig_url = self.resolve(&entry.signature);
        let bundle = self.get_bytes(&bundle_url, MAX_BUNDLE_BYTES)?;
        let sig = self.get_bytes(&sig_url, 1024)?;
        Ok((bundle, sig))
    }

    fn resolve(&self, relative: &str) -> String {
        // Absolute URLs in the index override the base. Useful for mixed
        // hosting (index on GitHub, bundles on a CDN), though we don't
        // currently do that.
        if relative.starts_with("http://") || relative.starts_with("https://") {
            relative.to_string()
        } else {
            format!("{}{}", self.base_url, relative.trim_start_matches('/'))
        }
    }

    fn get_bytes(&self, url: &str, limit: usize) -> Result<Vec<u8>, RegistryError> {
        let resp = self
            .http
            .get(url)
            .send()
            .map_err(|e| RegistryError::Http(format!("{url}: {e}")))?;
        let status = resp.status();
        if !status.is_success() {
            return Err(RegistryError::BadStatus {
                url: url.to_string(),
                status: status.as_u16(),
            });
        }
        let bytes = resp
            .bytes()
            .map_err(|e| RegistryError::Http(format!("{url}: {e}")))?;
        if bytes.len() > limit {
            return Err(RegistryError::TooLarge {
                url: url.to_string(),
                limit,
            });
        }
        Ok(bytes.to_vec())
    }
}

/// Diff the store's installed skills against `index` and return one
/// [`AvailableUpdate`] per skill whose index version is strictly newer than
/// what's installed. Skills in the index that aren't installed locally are
/// ignored — this function is for updates only, not for discovery.
pub fn check_updates(store: &SkillStore, index: &Index) -> Vec<AvailableUpdate> {
    let mut out: Vec<AvailableUpdate> = Vec::new();
    for entry in &index.skills {
        let Some(installed) = store.get(&entry.id) else {
            continue;
        };
        if compare_versions(&entry.version, &installed.version) == std::cmp::Ordering::Greater {
            out.push(AvailableUpdate {
                id: entry.id.clone(),
                installed_version: installed.version.clone(),
                available_version: entry.version.clone(),
                entry: entry.clone(),
            });
        }
    }
    out
}

/// Download one index entry and hand the bytes to the store for the
/// standard install pipeline. Uses the trust root passed in — callers
/// should usually pass `TrustRoot::single(&REGISTRY_TRUST_KEY)`, but tests
/// and `ari-cli --registry-trust-key-hex` override it.
///
/// This is **user-initiated**: the background updater never calls this.
/// All the background worker does is call [`check_updates`] and post a
/// notification; the user then opens the app and taps update, which
/// routes here.
pub fn install_update(
    client: &RegistryClient,
    entry: &IndexEntry,
    store: &mut SkillStore,
    trust_root: &TrustRoot,
    load_options: &LoadOptions,
) -> Result<InstalledSkill, RegistryError> {
    let (bundle, sig) = client.fetch_bundle(entry)?;

    // Sanity-check the sha256 up front so "the index lied" is distinct
    // from "the signature is wrong". install_from_bytes would catch it
    // anyway, but this gives a clearer error at a distance.
    let actual = crate::bundle::sha256_hex(&bundle);
    if actual != entry.sha256 {
        return Err(RegistryError::ShaMismatch {
            id: entry.id.clone(),
            expected: entry.sha256.clone(),
            actual,
        });
    }

    // The store's install() enforces downgrade defence and updates the
    // index. It takes its own trust root via SkillStore::open, so we need
    // a variant that lets us pass one in — currently SkillStore uses the
    // trust root it was opened with. For the registry path we re-open the
    // store's install pipeline with the caller-supplied trust root by
    // going through install_from_bytes directly, then calling rescan()
    // so the store's in-memory index picks up the new install.
    install_from_bytes(
        &bundle,
        &sig,
        &entry.sha256,
        trust_root,
        store.root_path(),
        load_options,
    )?;
    store
        .rescan()
        .map_err(|e| RegistryError::Http(format!("rescan after install: {e}")))?;
    let installed = store
        .get(&entry.id)
        .cloned()
        .ok_or_else(|| RegistryError::Parse(format!("install succeeded but {} not in index after rescan", entry.id)))?;
    Ok(installed)
}

/// Look up `id` in the provided `index` and install that entry into `store`,
/// regardless of whether the skill is already installed locally. Used by
/// the "Browse registry → tap install" path where the user is installing a
/// skill for the first time, as opposed to [`install_update`] which is
/// called after a diff via [`check_updates`].
///
/// Reinstalls of the currently-installed version are allowed — the store's
/// install pipeline overwrites in place. Installing an *older* version than
/// what's currently on disk is a no-op from the perspective of the user but
/// succeeds silently; downgrade protection is a TODO on the store side.
///
/// Returns [`RegistryError::NotFound`] if `id` isn't in the index at all,
/// which is distinct from every other error mode so the FFI layer can
/// surface a "registry no longer carries that skill" message cleanly.
pub fn install_by_id(
    client: &RegistryClient,
    index: &Index,
    id: &str,
    store: &mut SkillStore,
    trust_root: &TrustRoot,
    load_options: &LoadOptions,
) -> Result<InstalledSkill, RegistryError> {
    let entry = index
        .skills
        .iter()
        .find(|e| e.id == id)
        .ok_or_else(|| RegistryError::NotFound { id: id.to_string() })?;
    install_update(client, entry, store, trust_root, load_options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bundle::sha256_hex;
    use crate::storage_config::StorageConfig;
    use ed25519_dalek::{Signer, SigningKey};
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use sha2::{Digest, Sha256};
    use std::collections::HashMap;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};

    static N: AtomicU64 = AtomicU64::new(0);

    fn unique_dir(prefix: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let mut p = std::env::temp_dir();
        p.push(format!("ari-registry-test-{prefix}-{nanos}-{n}"));
        p
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

    /// Minimal multi-route in-process HTTP server. Keyed by path, values are
    /// raw HTTP response strings (status line + headers + body). Serves each
    /// incoming request once by parsing its request-line and looking up the
    /// route; unknown routes get a 404. Loops until [`TestServer::stop`] is
    /// called.
    struct TestServer {
        addr: std::net::SocketAddr,
        stop: Arc<Mutex<bool>>,
        _thread: std::thread::JoinHandle<()>,
    }

    impl TestServer {
        fn start(routes: HashMap<String, Vec<u8>>) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            listener
                .set_nonblocking(false)
                .expect("set_nonblocking false");
            let addr = listener.local_addr().unwrap();
            let stop = Arc::new(Mutex::new(false));
            let stop_clone = stop.clone();
            let thread = std::thread::spawn(move || {
                listener
                    .set_nonblocking(true)
                    .expect("set_nonblocking true");
                loop {
                    if *stop_clone.lock().unwrap() {
                        break;
                    }
                    match listener.accept() {
                        Ok((mut stream, _)) => {
                            stream.set_nonblocking(false).ok();
                            let mut buf = [0u8; 2048];
                            let mut acc: Vec<u8> = Vec::new();
                            loop {
                                let n = match stream.read(&mut buf) {
                                    Ok(0) => break,
                                    Ok(n) => n,
                                    Err(_) => break,
                                };
                                acc.extend_from_slice(&buf[..n]);
                                if acc.windows(4).any(|w| w == b"\r\n\r\n") {
                                    break;
                                }
                            }
                            let req_line = std::str::from_utf8(&acc)
                                .ok()
                                .and_then(|s| s.lines().next())
                                .unwrap_or("")
                                .to_string();
                            let path = req_line
                                .split_whitespace()
                                .nth(1)
                                .unwrap_or("/")
                                .to_string();
                            let response = match routes.get(&path) {
                                Some(r) => r.clone(),
                                None => b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n"
                                    .to_vec(),
                            };
                            let _ = stream.write_all(&response);
                            let _ = stream.flush();
                        }
                        Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                            std::thread::sleep(Duration::from_millis(5));
                        }
                        Err(_) => break,
                    }
                }
            });
            Self {
                addr,
                stop,
                _thread: thread,
            }
        }

        fn url(&self, path: &str) -> String {
            format!("http://{}{}", self.addr, path)
        }

        fn base(&self) -> String {
            format!("http://{}/", self.addr)
        }

        fn stop(&self) {
            *self.stop.lock().unwrap() = true;
        }
    }

    impl Drop for TestServer {
        fn drop(&mut self) {
            self.stop();
        }
    }

    fn http_response(body: &[u8], content_type: &str) -> Vec<u8> {
        let mut out = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            content_type,
            body.len()
        )
        .into_bytes();
        out.extend_from_slice(body);
        out
    }

    #[test]
    fn trust_key_constant_is_valid_ed25519_point() {
        // Protects against a copy-paste error corrupting the baked-in key.
        TrustRoot::single(&REGISTRY_TRUST_KEY).expect("REGISTRY_TRUST_KEY must be a valid point");
    }

    #[test]
    fn fetch_index_parses_valid_payload() {
        let index = r#"{"index_version":1,"generated_at":"2026-04-08T00:00:00Z","skills":[]}"#;
        let mut routes = HashMap::new();
        routes.insert(
            "/index.json".to_string(),
            http_response(index.as_bytes(), "application/json"),
        );
        let server = TestServer::start(routes);
        let client = RegistryClient::new().with_index_url(server.url("/index.json"));
        let parsed = client.fetch_index().unwrap();
        assert_eq!(parsed.index_version, 1);
        assert!(parsed.skills.is_empty());
    }

    #[test]
    fn fetch_index_rejects_unsupported_version() {
        let index = r#"{"index_version":99,"generated_at":"x","skills":[]}"#;
        let mut routes = HashMap::new();
        routes.insert(
            "/index.json".to_string(),
            http_response(index.as_bytes(), "application/json"),
        );
        let server = TestServer::start(routes);
        let client = RegistryClient::new().with_index_url(server.url("/index.json"));
        let err = client.fetch_index().unwrap_err();
        assert!(matches!(
            err,
            RegistryError::UnsupportedIndexVersion { expected: 1, found: 99 }
        ));
    }

    #[test]
    fn fetch_index_rejects_bad_status() {
        let mut routes = HashMap::new();
        // Empty: every request gets 404
        routes.insert("/exists".to_string(), http_response(b"", "text/plain"));
        let server = TestServer::start(routes);
        let client = RegistryClient::new().with_index_url(server.url("/missing.json"));
        let err = client.fetch_index().unwrap_err();
        assert!(matches!(err, RegistryError::BadStatus { status: 404, .. }));
    }

    #[test]
    fn check_updates_reports_only_newer_versions_for_installed_skills() {
        // One installed coin-flip @ 0.1.0. Index has:
        //   - coin-flip @ 0.2.0  → update
        //   - counter   @ 5.0.0  → not installed, ignored
        let tmp_root = unique_dir("store");
        let tmp_storage = unique_dir("storage");
        let sk = SigningKey::from_bytes(&[42u8; 32]);
        let trust = TrustRoot::single(sk.verifying_key().as_bytes()).unwrap();
        let mut store =
            SkillStore::open(&tmp_root, StorageConfig::new(&tmp_storage), trust.clone()).unwrap();
        // Install 0.1.0 using the underlying pipeline directly.
        let bundle = make_bundle("coin-flip", &coin_md("0.1.0"));
        let hash = sha256_hex(&bundle);
        let mut hasher = Sha256::new();
        hasher.update(&bundle);
        let sig = sk.sign(&hasher.finalize()).to_bytes().to_vec();
        store
            .install(&bundle, &sig, &hash, &LoadOptions::default())
            .unwrap();

        let index = Index {
            index_version: 1,
            generated_at: "t".into(),
            skills: vec![
                IndexEntry {
                    id: "dev.heyari.coinflip".into(),
                    version: "0.2.0".into(),
                    name: "coin-flip".into(),
                    description: "".into(),
                    license: None,
                    author: None,
                    homepage: None,
                    capabilities: Vec::new(),
                    languages: Vec::new(),
                    bundle: "bundles/dev.heyari.coinflip-0.2.0.tar.gz".into(),
                    signature: "bundles/dev.heyari.coinflip-0.2.0.tar.gz.sig".into(),
                    sha256: "deadbeef".into(),
                },
                IndexEntry {
                    id: "dev.heyari.counter".into(),
                    version: "5.0.0".into(),
                    name: "counter".into(),
                    description: "".into(),
                    license: None,
                    author: None,
                    homepage: None,
                    capabilities: Vec::new(),
                    languages: Vec::new(),
                    bundle: "bundles/dev.heyari.counter-5.0.0.tar.gz".into(),
                    signature: "bundles/dev.heyari.counter-5.0.0.tar.gz.sig".into(),
                    sha256: "cafe".into(),
                },
            ],
        };

        let updates = check_updates(&store, &index);
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0].id, "dev.heyari.coinflip");
        assert_eq!(updates[0].installed_version, "0.1.0");
        assert_eq!(updates[0].available_version, "0.2.0");

        let _ = std::fs::remove_dir_all(&tmp_root);
        let _ = std::fs::remove_dir_all(&tmp_storage);
    }

    #[test]
    fn check_updates_ignores_equal_and_older_index_versions() {
        let tmp_root = unique_dir("store2");
        let tmp_storage = unique_dir("storage2");
        let sk = SigningKey::from_bytes(&[43u8; 32]);
        let trust = TrustRoot::single(sk.verifying_key().as_bytes()).unwrap();
        let mut store =
            SkillStore::open(&tmp_root, StorageConfig::new(&tmp_storage), trust.clone()).unwrap();
        let bundle = make_bundle("coin-flip", &coin_md("0.2.0"));
        let hash = sha256_hex(&bundle);
        let mut h = Sha256::new();
        h.update(&bundle);
        let sig = sk.sign(&h.finalize()).to_bytes().to_vec();
        store
            .install(&bundle, &sig, &hash, &LoadOptions::default())
            .unwrap();

        let mut entry = IndexEntry {
            id: "dev.heyari.coinflip".into(),
            version: "0.2.0".into(),
            name: "coin-flip".into(),
            description: "".into(),
            license: None,
            author: None,
            homepage: None,
            capabilities: Vec::new(),
            languages: Vec::new(),
            bundle: "x".into(),
            signature: "x".into(),
            sha256: "x".into(),
        };
        let mut index = Index {
            index_version: 1,
            generated_at: "t".into(),
            skills: vec![entry.clone()],
        };
        assert!(check_updates(&store, &index).is_empty());

        // Older in the index → still no update (protects against a rollback
        // in the registry masquerading as an update).
        entry.version = "0.1.0".into();
        index.skills = vec![entry];
        assert!(check_updates(&store, &index).is_empty());

        let _ = std::fs::remove_dir_all(&tmp_root);
        let _ = std::fs::remove_dir_all(&tmp_storage);
    }

    #[test]
    fn install_update_downloads_and_installs_against_test_server() {
        // Full wire-level e2e: spin up a test server, serve index + bundle +
        // signature, point a client at it, run install_update, confirm the
        // installed version flipped from 0.1.0 → 0.2.0.
        let sk = SigningKey::from_bytes(&[77u8; 32]);
        let trust = TrustRoot::single(sk.verifying_key().as_bytes()).unwrap();

        // Build both versions up front.
        let v1 = make_bundle("coin-flip", &coin_md("0.1.0"));
        let v1_hash = sha256_hex(&v1);
        let mut h1 = Sha256::new();
        h1.update(&v1);
        let v1_sig = sk.sign(&h1.finalize()).to_bytes().to_vec();

        let v2 = make_bundle("coin-flip", &coin_md("0.2.0"));
        let v2_hash = sha256_hex(&v2);
        let mut h2 = Sha256::new();
        h2.update(&v2);
        let v2_sig = sk.sign(&h2.finalize()).to_bytes().to_vec();

        // Install 0.1.0 locally first.
        let store_root = unique_dir("update-store");
        let storage_root = unique_dir("update-storage");
        let mut store = SkillStore::open(
            &store_root,
            StorageConfig::new(&storage_root),
            trust.clone(),
        )
        .unwrap();
        store
            .install(&v1, &v1_sig, &v1_hash, &LoadOptions::default())
            .unwrap();
        assert_eq!(store.get("dev.heyari.coinflip").unwrap().version, "0.1.0");

        // Register routes for 0.2.0.
        let index_json = format!(
            r#"{{"index_version":1,"generated_at":"t","skills":[{{
                "id":"dev.heyari.coinflip","version":"0.2.0","name":"coin-flip",
                "description":"","bundle":"bundles/v2.tar.gz",
                "signature":"bundles/v2.tar.gz.sig","sha256":"{}"
            }}]}}"#,
            v2_hash
        );
        let mut routes = HashMap::new();
        routes.insert(
            "/index.json".to_string(),
            http_response(index_json.as_bytes(), "application/json"),
        );
        routes.insert(
            "/bundles/v2.tar.gz".to_string(),
            http_response(&v2, "application/gzip"),
        );
        routes.insert(
            "/bundles/v2.tar.gz.sig".to_string(),
            http_response(&v2_sig, "application/octet-stream"),
        );
        let server = TestServer::start(routes);

        let client = RegistryClient::new()
            .with_index_url(server.url("/index.json"))
            .with_base_url(server.base());
        let index = client.fetch_index().unwrap();
        let updates = check_updates(&store, &index);
        assert_eq!(updates.len(), 1, "expected 1 update, got {updates:?}");

        let entry = &updates[0].entry;
        let installed = install_update(
            &client,
            entry,
            &mut store,
            &trust,
            &LoadOptions::default(),
        )
        .unwrap();
        assert_eq!(installed.id, "dev.heyari.coinflip");
        assert_eq!(installed.version, "0.2.0");
        assert_eq!(store.get("dev.heyari.coinflip").unwrap().version, "0.2.0");

        let _ = std::fs::remove_dir_all(&store_root);
        let _ = std::fs::remove_dir_all(&storage_root);
    }

    #[test]
    fn install_update_rejects_bundle_whose_hash_does_not_match_index() {
        // Server serves the wrong bundle bytes for the advertised hash.
        let sk = SigningKey::from_bytes(&[88u8; 32]);
        let trust = TrustRoot::single(sk.verifying_key().as_bytes()).unwrap();

        let real = make_bundle("coin-flip", &coin_md("0.2.0"));
        let fake = make_bundle("coin-flip", &coin_md("0.2.0-tampered"));
        let real_hash = sha256_hex(&real);
        let mut h = Sha256::new();
        h.update(&fake); // sign the fake so the sig at least parses
        let fake_sig = sk.sign(&h.finalize()).to_bytes().to_vec();

        let index_json = format!(
            r#"{{"index_version":1,"generated_at":"t","skills":[{{
                "id":"dev.heyari.coinflip","version":"0.2.0","name":"coin-flip",
                "description":"","bundle":"bundles/v2.tar.gz",
                "signature":"bundles/v2.tar.gz.sig","sha256":"{}"
            }}]}}"#,
            real_hash
        );
        let mut routes = HashMap::new();
        routes.insert(
            "/index.json".to_string(),
            http_response(index_json.as_bytes(), "application/json"),
        );
        // Serve the fake bytes under the URL that claims the real hash.
        routes.insert(
            "/bundles/v2.tar.gz".to_string(),
            http_response(&fake, "application/gzip"),
        );
        routes.insert(
            "/bundles/v2.tar.gz.sig".to_string(),
            http_response(&fake_sig, "application/octet-stream"),
        );
        let server = TestServer::start(routes);

        let store_root = unique_dir("tamper-store");
        let storage_root = unique_dir("tamper-storage");
        let mut store = SkillStore::open(
            &store_root,
            StorageConfig::new(&storage_root),
            trust.clone(),
        )
        .unwrap();
        let client = RegistryClient::new()
            .with_index_url(server.url("/index.json"))
            .with_base_url(server.base());
        let index = client.fetch_index().unwrap();
        let entry = &index.skills[0];

        let err = install_update(
            &client,
            entry,
            &mut store,
            &trust,
            &LoadOptions::default(),
        )
        .unwrap_err();
        assert!(matches!(err, RegistryError::ShaMismatch { .. }));

        let _ = std::fs::remove_dir_all(&store_root);
        let _ = std::fs::remove_dir_all(&storage_root);
    }

    #[test]
    fn install_by_id_installs_skill_not_previously_present() {
        // Full wire-level e2e for the discovery path: store starts empty,
        // registry serves one entry, install_by_id fetches/verifies/installs
        // and the store reflects the new skill.
        let sk = SigningKey::from_bytes(&[99u8; 32]);
        let trust = TrustRoot::single(sk.verifying_key().as_bytes()).unwrap();

        let bundle = make_bundle("coin-flip", &coin_md("0.1.0"));
        let bundle_hash = sha256_hex(&bundle);
        let mut h = Sha256::new();
        h.update(&bundle);
        let sig = sk.sign(&h.finalize()).to_bytes().to_vec();

        let index_json = format!(
            r#"{{"index_version":1,"generated_at":"t","skills":[{{
                "id":"dev.heyari.coinflip","version":"0.1.0","name":"coin-flip",
                "description":"","bundle":"bundles/v1.tar.gz",
                "signature":"bundles/v1.tar.gz.sig","sha256":"{}"
            }}]}}"#,
            bundle_hash
        );
        let mut routes = HashMap::new();
        routes.insert(
            "/index.json".to_string(),
            http_response(index_json.as_bytes(), "application/json"),
        );
        routes.insert(
            "/bundles/v1.tar.gz".to_string(),
            http_response(&bundle, "application/gzip"),
        );
        routes.insert(
            "/bundles/v1.tar.gz.sig".to_string(),
            http_response(&sig, "application/octet-stream"),
        );
        let server = TestServer::start(routes);

        let store_root = unique_dir("discover-store");
        let storage_root = unique_dir("discover-storage");
        let mut store = SkillStore::open(
            &store_root,
            StorageConfig::new(&storage_root),
            trust.clone(),
        )
        .unwrap();
        assert!(store.get("dev.heyari.coinflip").is_none(), "store should start empty");

        let client = RegistryClient::new()
            .with_index_url(server.url("/index.json"))
            .with_base_url(server.base());
        let index = client.fetch_index().unwrap();

        let installed = install_by_id(
            &client,
            &index,
            "dev.heyari.coinflip",
            &mut store,
            &trust,
            &LoadOptions::default(),
        )
        .unwrap();
        assert_eq!(installed.id, "dev.heyari.coinflip");
        assert_eq!(installed.version, "0.1.0");
        assert_eq!(store.get("dev.heyari.coinflip").unwrap().version, "0.1.0");

        let _ = std::fs::remove_dir_all(&store_root);
        let _ = std::fs::remove_dir_all(&storage_root);
    }

    #[test]
    fn install_by_id_returns_not_found_for_unknown_id() {
        // No server, no network. Hand a caller-built index with zero
        // entries and confirm we get the precise NotFound variant rather
        // than a generic parse/http error — the FFI layer relies on this
        // discriminant to show a "registry no longer carries that skill"
        // message distinct from connectivity failures.
        let sk = SigningKey::from_bytes(&[44u8; 32]);
        let trust = TrustRoot::single(sk.verifying_key().as_bytes()).unwrap();

        let store_root = unique_dir("notfound-store");
        let storage_root = unique_dir("notfound-storage");
        let mut store = SkillStore::open(
            &store_root,
            StorageConfig::new(&storage_root),
            trust.clone(),
        )
        .unwrap();

        let empty_index = Index {
            index_version: 1,
            generated_at: "t".to_string(),
            skills: vec![],
        };
        let client = RegistryClient::new();
        let err = install_by_id(
            &client,
            &empty_index,
            "dev.heyari.nosuchthing",
            &mut store,
            &trust,
            &LoadOptions::default(),
        )
        .unwrap_err();
        match err {
            RegistryError::NotFound { id } => assert_eq!(id, "dev.heyari.nosuchthing"),
            other => panic!("expected NotFound, got {other:?}"),
        }

        let _ = std::fs::remove_dir_all(&store_root);
        let _ = std::fs::remove_dir_all(&storage_root);
    }

    #[test]
    fn resolve_joins_relative_paths_onto_base_url() {
        let c = RegistryClient::new().with_base_url("http://example.test/prefix/");
        assert_eq!(
            c.resolve("bundles/foo.tar.gz"),
            "http://example.test/prefix/bundles/foo.tar.gz"
        );
        assert_eq!(
            c.resolve("/bundles/foo.tar.gz"),
            "http://example.test/prefix/bundles/foo.tar.gz"
        );
        assert_eq!(
            c.resolve("https://cdn.example.test/foo.tar.gz"),
            "https://cdn.example.test/foo.tar.gz"
        );
    }

    #[test]
    fn with_base_url_forces_trailing_slash() {
        let c = RegistryClient::new().with_base_url("http://example.test/prefix");
        assert!(c.base_url().ends_with('/'));
    }
}
