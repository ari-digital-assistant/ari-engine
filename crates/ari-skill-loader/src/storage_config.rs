//! Host-side configuration for the `ari::storage_get` / `ari::storage_set`
//! WASM imports.
//!
//! Each skill that has the `storage_kv` capability granted gets its own JSON
//! file on disk under [`StorageConfig::root`], named `<skill-id>.json`. The
//! file holds a single JSON object mapping string keys to string values.
//! Load → mutate → atomic save (tmp + rename) per `storage_set`. For voice-
//! assistant call rates this is plenty.
//!
//! Three hard limits, all enforced in the WASM host import: per-key length,
//! per-value length, and total per-skill bytes after a set. Anything that
//! breaks a limit causes the set to fail (return non-zero) and the on-disk
//! state stays as it was before the call.

use std::path::PathBuf;

const DEFAULT_MAX_KEY_BYTES: usize = 256;
const DEFAULT_MAX_VALUE_BYTES: usize = 64 * 1024;
const DEFAULT_MAX_TOTAL_BYTES: usize = 1024 * 1024;

#[derive(Debug, Clone)]
pub struct StorageConfig {
    /// Directory under which each skill's JSON file lives.
    pub root: PathBuf,
    /// Maximum length of a single key in bytes.
    pub max_key_bytes: usize,
    /// Maximum length of a single value in bytes.
    pub max_value_bytes: usize,
    /// Maximum total bytes (sum of all keys + values) per skill after a set.
    pub max_total_bytes: usize,
}

impl StorageConfig {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            max_key_bytes: DEFAULT_MAX_KEY_BYTES,
            max_value_bytes: DEFAULT_MAX_VALUE_BYTES,
            max_total_bytes: DEFAULT_MAX_TOTAL_BYTES,
        }
    }

    /// Default config rooted in the system temp directory under
    /// `ari-skill-storage`. Useful for the CLI when the user hasn't supplied
    /// `--storage-dir`. Production frontends should use [`Self::new`] with a
    /// stable per-user data directory.
    pub fn ephemeral_default() -> Self {
        let mut root = std::env::temp_dir();
        root.push("ari-skill-storage");
        Self::new(root)
    }

    pub fn with_max_key_bytes(mut self, n: usize) -> Self {
        self.max_key_bytes = n;
        self
    }

    pub fn with_max_value_bytes(mut self, n: usize) -> Self {
        self.max_value_bytes = n;
        self
    }

    pub fn with_max_total_bytes(mut self, n: usize) -> Self {
        self.max_total_bytes = n;
        self
    }

    /// Path to a specific skill's storage file. Skill ID is used verbatim
    /// (it's already a reverse-DNS string with no path-traversal characters).
    /// Defensive sanitisation strips anything that isn't alphanumeric, dot,
    /// dash, or underscore as belt-and-braces.
    pub fn file_for(&self, skill_id: &str) -> PathBuf {
        let safe: String = skill_id
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_') {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        self.root.join(format!("{safe}.json"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_sensible() {
        let c = StorageConfig::new("/tmp/x");
        assert_eq!(c.max_key_bytes, 256);
        assert_eq!(c.max_value_bytes, 64 * 1024);
        assert_eq!(c.max_total_bytes, 1024 * 1024);
    }

    #[test]
    fn ephemeral_default_uses_system_temp() {
        let c = StorageConfig::ephemeral_default();
        assert!(c.root.starts_with(std::env::temp_dir()));
        assert!(c.root.ends_with("ari-skill-storage"));
    }

    #[test]
    fn file_for_uses_skill_id_verbatim_when_safe() {
        let c = StorageConfig::new("/tmp/x");
        assert_eq!(
            c.file_for("dev.heyari.counter"),
            PathBuf::from("/tmp/x/dev.heyari.counter.json")
        );
    }

    #[test]
    fn file_for_sanitises_path_traversal_characters() {
        let c = StorageConfig::new("/tmp/x");
        // The manifest validator should never let this through, but the
        // storage layer is the last line of defence.
        assert_eq!(
            c.file_for("../etc/passwd"),
            PathBuf::from("/tmp/x/.._etc_passwd.json")
        );
    }

    #[test]
    fn builders_round_trip() {
        let c = StorageConfig::new("/x")
            .with_max_key_bytes(10)
            .with_max_value_bytes(100)
            .with_max_total_bytes(1000);
        assert_eq!(c.max_key_bytes, 10);
        assert_eq!(c.max_value_bytes, 100);
        assert_eq!(c.max_total_bytes, 1000);
    }
}
