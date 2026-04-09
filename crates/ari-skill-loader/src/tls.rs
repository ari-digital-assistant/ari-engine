//! Shared rustls `ClientConfig` builder for every reqwest client in this
//! crate.
//!
//! **Why this exists**: reqwest 0.13's `rustls` feature uses
//! `rustls-platform-verifier`, which needs JNI initialisation on Android
//! before it can read the system trust store. Without that init the first
//! TLS handshake panics the reqwest worker thread, surfacing as
//! `"event loop thread panicked"` through UniFFI — which is exactly what
//! happened the first time we ran the skill registry check on device.
//!
//! Rather than threading a `JavaVM` handle through UniFFI into the loader,
//! we sidestep the problem entirely by building our own `rustls::ClientConfig`
//! from the bundled `webpki-roots` trust store and handing it to reqwest via
//! `ClientBuilder::use_preconfigured_tls`. Adds ~200KB of Mozilla root CAs
//! to the binary, works identically on every platform, no JNI gymnastics.
//!
//! The tradeoff is we don't honour user-added / MDM-deployed certificates,
//! which isn't a concern for the skill registry (a single hardcoded GitHub
//! URL) but *would* be for general HTTP fetches from within skills. That's
//! fine for now — the WASM `http_fetch` host import uses the same helper
//! and serves the same kind of "random internet URL" traffic.

use rustls::ClientConfig;
use std::sync::Arc;

/// Build a fresh `rustls::ClientConfig` whose trust roots are the bundled
/// webpki-roots CA set. Returns a bare `ClientConfig` (not `Arc<_>`)
/// because `reqwest::ClientBuilder::use_preconfigured_tls` downcasts by
/// exact type — it wants `rustls::ClientConfig`, not a wrapped variant.
///
/// We pin the `ring` crypto provider explicitly rather than relying on
/// rustls's auto-select: reqwest's `rustls` feature pulls in
/// `rustls-platform-verifier` which in turn activates `aws-lc-rs`, so two
/// providers are linked into the binary and rustls bails with a
/// "could not automatically determine provider" panic if we don't pick one.
pub(crate) fn webpki_roots_config() -> ClientConfig {
    let mut roots = rustls::RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .expect("rustls ring provider supports TLS 1.2 + 1.3")
        .with_root_certificates(roots)
        .with_no_client_auth()
}
