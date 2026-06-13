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
    /// When true, plain-`http` requests are permitted **only** to private,
    /// loopback, or `.local`/`.lan` hosts (a LAN Home Assistant is almost
    /// always `http://homeassistant.local:8123` or `http://<rfc1918-ip>:8123`).
    /// Plain http to any public host is still rejected. HTTPS is unaffected.
    pub allow_http_to_private_hosts: bool,
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
            allow_http_to_private_hosts: true,
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

    /// Whether the skill is allowed to fetch this URL. HTTPS is allowed if
    /// `https` is in `allowed_schemes`. Plain `http` is allowed if it's in
    /// `allowed_schemes` (test/permissive mode) OR
    /// `allow_http_to_private_hosts` is set and the host is private/loopback/
    /// `.local`. Any other scheme must be explicitly in `allowed_schemes`.
    pub fn allows_url(&self, url: &url::Url) -> bool {
        match url.scheme() {
            "https" => self.allows_scheme("https"),
            "http" => {
                self.allows_scheme("http")
                    || (self.allow_http_to_private_hosts && Self::is_private_host(url))
            }
            other => self.allows_scheme(other),
        }
    }

    fn is_private_host(url: &url::Url) -> bool {
        match url.host() {
            Some(url::Host::Ipv4(ip)) => {
                ip.is_private() || ip.is_loopback()
            }
            Some(url::Host::Ipv6(ip)) => {
                // loopback (::1) or unique-local fc00::/7
                ip.is_loopback() || (ip.segments()[0] & 0xfe00) == 0xfc00
            }
            Some(url::Host::Domain(d)) => {
                let d = d.to_ascii_lowercase();
                d == "localhost" || d.ends_with(".local") || d.ends_with(".lan")
            }
            None => false,
        }
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

    #[test]
    fn https_allowed_anywhere() {
        let c = HttpConfig::strict();
        let u = url::Url::parse("https://example.com/api").unwrap();
        assert!(c.allows_url(&u));
    }

    #[test]
    fn http_allowed_to_private_and_local_hosts() {
        let c = HttpConfig::strict();
        for raw in [
            "http://192.168.1.10:8123/api/",
            "http://10.0.0.5:8123/api/",
            "http://172.16.3.4:8123/",
            "http://127.0.0.1:8123/",
            "http://[::1]:8123/",
            "http://[fd00::1]:8123/",
            "http://homeassistant.local:8123/api/",
            "http://hass.lan/api/",
            "http://localhost:8123/",
        ] {
            let u = url::Url::parse(raw).unwrap();
            assert!(c.allows_url(&u), "expected allowed: {raw}");
        }
    }

    #[test]
    fn http_blocked_to_public_hosts() {
        let c = HttpConfig::strict();
        for raw in [
            "http://example.com/",
            "http://8.8.8.8/",
            "http://my.duckdns.org/",
            "http://[2001:4860:4860::8888]/",
            "http://evil.local.attacker.com/",
            "http://169.254.169.254/",
        ] {
            let u = url::Url::parse(raw).unwrap();
            assert!(!c.allows_url(&u), "expected blocked: {raw}");
        }
    }

    #[test]
    fn private_http_can_be_disabled() {
        let mut c = HttpConfig::strict();
        c.allow_http_to_private_hosts = false;
        let u = url::Url::parse("http://192.168.1.10:8123/").unwrap();
        assert!(!c.allows_url(&u));
    }
}
