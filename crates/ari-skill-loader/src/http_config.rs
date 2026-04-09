//! Host-side configuration for the `ari::http_fetch` WASM import.
//!
//! Lives on [`crate::loader::LoadOptions`] and is consulted by the WASM
//! adapter when wiring up the http import. The defaults are deliberately
//! strict — HTTPS only, 1 MiB body cap, 10 second timeout, 5 redirects max,
//! fixed user-agent. Tests that need plain HTTP against a local listener can
//! relax the scheme allowlist explicitly.

use std::time::Duration;

const DEFAULT_USER_AGENT: &str = concat!("ari-skill/", env!("CARGO_PKG_VERSION"));
const DEFAULT_MAX_BODY_BYTES: usize = 1024 * 1024; // 1 MiB
const DEFAULT_TIMEOUT_SECS: u64 = 10;
const DEFAULT_MAX_REDIRECTS: u32 = 5;

#[derive(Debug, Clone)]
pub struct HttpConfig {
    /// Schemes the WASM skill is allowed to request. Anything not in this
    /// list is rejected before the request is dispatched. Defaults to
    /// `["https"]`.
    pub allowed_schemes: Vec<String>,
    /// Maximum response body the host will read from a remote server. Bodies
    /// larger than this are truncated and the call returns a body-too-large
    /// error to the skill.
    pub max_body_bytes: usize,
    /// Total request timeout (connect + send + receive). Hard limit; the
    /// skill cannot extend it.
    pub timeout: Duration,
    /// Maximum number of HTTP redirects to follow.
    pub max_redirects: u32,
    /// User-Agent header sent with every request. Cannot be overridden by
    /// the skill.
    pub user_agent: String,
}

impl Default for HttpConfig {
    fn default() -> Self {
        Self::strict()
    }
}

impl HttpConfig {
    /// Production defaults: HTTPS only, 1 MiB body cap, 10 second timeout.
    pub fn strict() -> Self {
        Self {
            allowed_schemes: vec!["https".to_string()],
            max_body_bytes: DEFAULT_MAX_BODY_BYTES,
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
            max_redirects: DEFAULT_MAX_REDIRECTS,
            user_agent: DEFAULT_USER_AGENT.to_string(),
        }
    }

    /// Test-only defaults: same as strict but also allows plain `http`. Use
    /// this when pointing a fixture skill at a local TcpListener-based test
    /// server. **Do not use in production.**
    pub fn permissive_for_tests() -> Self {
        let mut c = Self::strict();
        c.allowed_schemes.push("http".to_string());
        c
    }

    pub fn with_max_body_bytes(mut self, n: usize) -> Self {
        self.max_body_bytes = n;
        self
    }

    pub fn with_timeout(mut self, d: Duration) -> Self {
        self.timeout = d;
        self
    }

    pub fn allows_scheme(&self, scheme: &str) -> bool {
        self.allowed_schemes.iter().any(|s| s == scheme)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_only_allows_https() {
        let c = HttpConfig::strict();
        assert!(c.allows_scheme("https"));
        assert!(!c.allows_scheme("http"));
        assert!(!c.allows_scheme("file"));
        assert!(!c.allows_scheme("ftp"));
    }

    #[test]
    fn permissive_allows_both_http_schemes() {
        let c = HttpConfig::permissive_for_tests();
        assert!(c.allows_scheme("https"));
        assert!(c.allows_scheme("http"));
        assert!(!c.allows_scheme("file"));
    }

    #[test]
    fn defaults_are_strict() {
        let c = HttpConfig::default();
        assert_eq!(c.allowed_schemes, vec!["https".to_string()]);
        assert_eq!(c.max_body_bytes, 1024 * 1024);
        assert_eq!(c.timeout, Duration::from_secs(10));
        assert_eq!(c.max_redirects, 5);
        assert!(c.user_agent.starts_with("ari-skill/"));
    }

    #[test]
    fn builders_round_trip() {
        let c = HttpConfig::strict()
            .with_max_body_bytes(2048)
            .with_timeout(Duration::from_secs(2));
        assert_eq!(c.max_body_bytes, 2048);
        assert_eq!(c.timeout, Duration::from_secs(2));
    }
}
