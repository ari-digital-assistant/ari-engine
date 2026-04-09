//! What the host (engine + frontend) provides to skills.
//!
//! A [`HostCapabilities`] is the set of [`Capability`] values the host has
//! actually wired up. The loader checks every skill's declared capabilities
//! against this set at install time and rejects skills whose declarations
//! aren't a subset, with [`crate::loader::LoadFailureKind::MissingCapabilities`].
//!
//! Two flavours of capability matter here:
//!
//! - **Pure-frontend caps** (`launch_app`, `notifications`, `clipboard`, `tts`)
//!   don't need a WASM host import. The skill emits an `Action` response and
//!   the frontend acts on it. These can be granted by the host even though no
//!   WASM-side implementation exists in the loader.
//! - **Host-import caps** (`http`, `location`, `storage_kv`) require a WASM
//!   host import. The host should only claim them once those imports are
//!   wired up — until then, skills declaring them are correctly rejected.
//!
//! [`HostCapabilities::pure_frontend`] is the right default for a frontend
//! that hasn't shipped any host imports yet.

use crate::manifest::Capability;
use std::collections::HashSet;

/// The set of capabilities the host can satisfy.
#[derive(Debug, Clone, Default)]
pub struct HostCapabilities {
    granted: HashSet<Capability>,
}

impl HostCapabilities {
    /// An empty host capability set. Skills that declare any capability will
    /// be rejected at install time.
    pub fn none() -> Self {
        Self::default()
    }

    /// All capabilities that exist today, regardless of whether the loader
    /// actually implements them. Useful for tests; **do not use in production**
    /// — a skill claiming `http` against this set would link successfully and
    /// then fail at runtime when wasmtime can't resolve the import.
    pub fn all() -> Self {
        let mut s = Self::default();
        for cap in [
            Capability::Http,
            Capability::Location,
            Capability::Notifications,
            Capability::LaunchApp,
            Capability::Clipboard,
            Capability::Tts,
            Capability::StorageKv,
        ] {
            s.granted.insert(cap);
        }
        s
    }

    /// The pure-frontend capability set. None of these require a WASM host
    /// import; the skill emits an `Action` response and the frontend handles
    /// it. Safe to claim from any frontend that intends to honour Action
    /// responses, even before any WASM host imports exist.
    pub fn pure_frontend() -> Self {
        let mut s = Self::default();
        s.granted.insert(Capability::Notifications);
        s.granted.insert(Capability::LaunchApp);
        s.granted.insert(Capability::Clipboard);
        s.granted.insert(Capability::Tts);
        s
    }

    /// Builder-style: add a capability to the set.
    pub fn with(mut self, cap: Capability) -> Self {
        self.granted.insert(cap);
        self
    }

    /// Builder-style: remove a capability from the set.
    pub fn without(mut self, cap: Capability) -> Self {
        self.granted.remove(&cap);
        self
    }

    /// Is the given capability granted by this host?
    pub fn provides(&self, cap: Capability) -> bool {
        self.granted.contains(&cap)
    }

    /// Returns the capabilities the skill needs that this host does not
    /// provide. If the result is empty, the skill is installable.
    pub fn missing_for(&self, declared: &[Capability]) -> Vec<Capability> {
        declared
            .iter()
            .copied()
            .filter(|c| !self.granted.contains(c))
            .collect()
    }
}

/// Parse a capability identifier (snake_case, matching the manifest spelling).
pub fn parse_capability(s: &str) -> Option<Capability> {
    match s {
        "http" => Some(Capability::Http),
        "location" => Some(Capability::Location),
        "notifications" => Some(Capability::Notifications),
        "launch_app" => Some(Capability::LaunchApp),
        "clipboard" => Some(Capability::Clipboard),
        "tts" => Some(Capability::Tts),
        "storage_kv" => Some(Capability::StorageKv),
        _ => None,
    }
}

/// The canonical snake_case identifier for a capability.
pub fn capability_name(cap: Capability) -> &'static str {
    match cap {
        Capability::Http => "http",
        Capability::Location => "location",
        Capability::Notifications => "notifications",
        Capability::LaunchApp => "launch_app",
        Capability::Clipboard => "clipboard",
        Capability::Tts => "tts",
        Capability::StorageKv => "storage_kv",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_grants_nothing() {
        let h = HostCapabilities::none();
        for cap in [
            Capability::Http,
            Capability::LaunchApp,
            Capability::Notifications,
        ] {
            assert!(!h.provides(cap));
        }
    }

    #[test]
    fn pure_frontend_grants_only_frontend_caps() {
        let h = HostCapabilities::pure_frontend();
        assert!(h.provides(Capability::Notifications));
        assert!(h.provides(Capability::LaunchApp));
        assert!(h.provides(Capability::Clipboard));
        assert!(h.provides(Capability::Tts));
        assert!(!h.provides(Capability::Http));
        assert!(!h.provides(Capability::Location));
        assert!(!h.provides(Capability::StorageKv));
    }

    #[test]
    fn all_grants_everything() {
        let h = HostCapabilities::all();
        assert!(h.provides(Capability::Http));
        assert!(h.provides(Capability::Location));
        assert!(h.provides(Capability::StorageKv));
    }

    #[test]
    fn missing_for_returns_only_unsatisfied_caps() {
        let h = HostCapabilities::pure_frontend();
        let needed = [Capability::Notifications, Capability::Http, Capability::Location];
        let missing = h.missing_for(&needed);
        assert_eq!(missing, vec![Capability::Http, Capability::Location]);
    }

    #[test]
    fn missing_for_empty_when_skill_needs_nothing() {
        let h = HostCapabilities::none();
        assert!(h.missing_for(&[]).is_empty());
    }

    #[test]
    fn builder_with_and_without_round_trip() {
        let h = HostCapabilities::none()
            .with(Capability::Http)
            .with(Capability::Location)
            .without(Capability::Http);
        assert!(!h.provides(Capability::Http));
        assert!(h.provides(Capability::Location));
    }

    #[test]
    fn parse_and_format_round_trip_for_every_capability() {
        for cap in [
            Capability::Http,
            Capability::Location,
            Capability::Notifications,
            Capability::LaunchApp,
            Capability::Clipboard,
            Capability::Tts,
            Capability::StorageKv,
        ] {
            assert_eq!(parse_capability(capability_name(cap)), Some(cap));
        }
    }

    #[test]
    fn parse_unknown_capability_is_none() {
        assert_eq!(parse_capability("ftp"), None);
        assert_eq!(parse_capability(""), None);
        assert_eq!(parse_capability("HTTP"), None); // case-sensitive
    }
}
