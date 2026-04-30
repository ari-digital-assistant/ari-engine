//! Per-locale skill string-table loader.
//!
//! Skill bundles ship user-facing translations in a `strings/`
//! subdirectory:
//!
//! ```text
//! reminder/
//! ├── SKILL.en.md
//! ├── SKILL.it.md
//! ├── strings/
//! │   ├── en.json
//! │   ├── it.json
//! │   └── ...
//! └── skill.wasm
//! ```
//!
//! Each `{locale}.json` is a **flat** JSON object mapping
//! skill-author-defined keys to template strings. Templates contain
//! `{placeholder}` slots that the [`t()`](LocalizedStrings::render)
//! lookup substitutes from the caller's argument map.
//!
//! Example `strings/en.json`:
//!
//! ```json
//! {
//!   "reminder.confirmation": "OK, I'll remind you {when}",
//!   "reminder.cancelled": "Reminder cancelled."
//! }
//! ```
//!
//! Example `strings/it.json`:
//!
//! ```json
//! {
//!   "reminder.confirmation": "OK, ti ricorderò {when}",
//!   "reminder.cancelled": "Promemoria annullato."
//! }
//! ```
//!
//! Lookup falls back to `strings/en.json` when the requested locale
//! either doesn't have the key or doesn't ship strings at all.
//!
//! ## Validation
//!
//! - Missing `strings/` directory → empty [`LocalizedStrings`], no error.
//!   Skills without translatable text (e.g. pure-action skills that
//!   only emit JSON) don't need a strings dir.
//! - At least one `*.json` file present → `en.json` is required.
//!   English is the canonical fallback locale; without it, future
//!   non-English-keyed lookups would have no fallback path.
//! - Filename must be `{locale}.json` where `{locale}` is a 2-character
//!   lowercase ASCII string (ISO 639-1). `README.md`, `notes.txt`, etc.
//!   are silently ignored. `notes.json` (3+ char "locale") is a hard
//!   reject — same rationale as the manifest filename validator.
//! - Each file must be valid JSON with string keys and string values
//!   only (no nested objects). The skill SDK's `t()` import expects a
//!   flat key→template lookup; nested structure would be a footgun.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use thiserror::Error;

use crate::localized_manifest::CANONICAL_LOCALE;

/// All locale-specific string tables for a single skill, keyed by
/// ISO 639-1 lowercase locale code → key → template. Empty when the
/// skill ships no `strings/` directory.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LocalizedStrings {
    by_locale: BTreeMap<String, BTreeMap<String, String>>,
}

impl LocalizedStrings {
    /// Look up the raw template for a key in the given locale, with
    /// canonical-English fallback. Returns `None` when neither locale
    /// has the key — caller decides whether to surface the bare key
    /// (debug visibility) or an empty string.
    pub fn get(&self, locale: &str, key: &str) -> Option<&str> {
        self.by_locale
            .get(locale)
            .and_then(|m| m.get(key))
            .map(String::as_str)
            .or_else(|| {
                self.by_locale
                    .get(CANONICAL_LOCALE)
                    .and_then(|m| m.get(key))
                    .map(String::as_str)
            })
    }

    /// Resolve a key for the given locale, substituting `{placeholder}`
    /// slots from `args`. Unmatched placeholders are left intact in
    /// the output (useful in dev — a typoed placeholder stays visible
    /// rather than vanishing silently).
    ///
    /// Falls back to canonical-English when the locale doesn't have
    /// the key. Returns `None` only when the key isn't in either
    /// locale — same semantics as [`get`].
    pub fn render(
        &self,
        locale: &str,
        key: &str,
        args: &BTreeMap<String, String>,
    ) -> Option<String> {
        let template = self.get(locale, key)?;
        Some(substitute_placeholders(template, args))
    }

    /// Locale codes that have at least one string table loaded.
    /// Sorted alphabetically.
    pub fn supported_locales(&self) -> Vec<String> {
        self.by_locale.keys().cloned().collect()
    }

    /// True when the skill ships no localizable strings at all.
    pub fn is_empty(&self) -> bool {
        self.by_locale.is_empty()
    }
}

#[derive(Debug, Error)]
pub enum LocalizedStringsError {
    /// At least one `{locale}.json` file is present, but `en.json` is
    /// not. Without the canonical English table there's no fallback
    /// path for non-canonical lookups.
    #[error(
        "skill ships strings/ but no strings/en.json — English is the canonical \
         fallback and must always be present when any translations are shipped"
    )]
    MissingCanonical,

    /// A file in `strings/` had a filename that wasn't `{locale}.json`
    /// with a 2-character lowercase ASCII locale code.
    #[error(
        "`{filename}` doesn't follow the strings/{{locale}}.json pattern — \
         locale segment must be 2 lowercase ASCII letters (ISO 639-1)"
    )]
    InvalidLocaleFilename { filename: String },

    /// A file in `strings/` failed to parse as JSON or wasn't a flat
    /// `{string: string}` object.
    #[error("parsing {path}: {error}")]
    Parse { path: PathBuf, error: String },

    /// I/O failure reading the strings directory or one of its files.
    #[error("I/O failure on {path}: {message}")]
    Io { path: PathBuf, message: String },
}

/// Scan a skill directory's `strings/` subdirectory for `{locale}.json`
/// files and return the resulting [`LocalizedStrings`]. Missing
/// `strings/` is not an error — the result is just empty.
pub fn parse_strings_directory(skill_dir: &Path) -> Result<LocalizedStrings, LocalizedStringsError> {
    let strings_dir = skill_dir.join("strings");
    if !strings_dir.is_dir() {
        return Ok(LocalizedStrings::default());
    }

    let entries = std::fs::read_dir(&strings_dir).map_err(|e| LocalizedStringsError::Io {
        path: strings_dir.clone(),
        message: format!("could not read directory: {e}"),
    })?;

    let mut by_locale: BTreeMap<String, BTreeMap<String, String>> = BTreeMap::new();

    for entry in entries {
        let entry = entry.map_err(|e| LocalizedStringsError::Io {
            path: strings_dir.clone(),
            message: format!("dir entry: {e}"),
        })?;
        let path = entry.path();
        let Some(filename) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let Some(locale) = parse_strings_filename(filename)? else {
            continue;
        };

        let source = std::fs::read_to_string(&path).map_err(|e| LocalizedStringsError::Io {
            path: path.clone(),
            message: format!("could not read: {e}"),
        })?;

        // Parse as a flat `{ string: string }` map. serde_json yields
        // a clearer error for nested objects (or non-string values)
        // than a hand-rolled walker.
        let map: BTreeMap<String, String> = serde_json::from_str(&source).map_err(|e| {
            LocalizedStringsError::Parse {
                path: path.clone(),
                error: format!("expected flat object of string→string: {e}"),
            }
        })?;

        by_locale.insert(locale, map);
    }

    if !by_locale.is_empty() && !by_locale.contains_key(CANONICAL_LOCALE) {
        return Err(LocalizedStringsError::MissingCanonical);
    }

    Ok(LocalizedStrings { by_locale })
}

/// Pull the locale code out of a `{locale}.json` filename.
///
/// - `en.json` → `Some("en")`
/// - `it.json` → `Some("it")`
/// - `README.md` → `None` (caller skips)
/// - `notes.json` (3+ char "locale") → [`LocalizedStringsError::InvalidLocaleFilename`]
fn parse_strings_filename(filename: &str) -> Result<Option<String>, LocalizedStringsError> {
    let Some(stem) = filename.strip_suffix(".json") else {
        return Ok(None);
    };
    if stem.len() != 2 || !stem.chars().all(|c| c.is_ascii_lowercase()) {
        return Err(LocalizedStringsError::InvalidLocaleFilename {
            filename: filename.to_string(),
        });
    }
    Ok(Some(stem.to_string()))
}

fn substitute_placeholders(template: &str, args: &BTreeMap<String, String>) -> String {
    // Single pass over the template string, building the output.
    // Naive `String::replace` per arg works but is O(N*M) on long
    // templates with many args; this is O(N) regardless. Unmatched
    // `{...}` slots are emitted verbatim — visible to the dev when
    // they typo a placeholder, instead of silently disappearing.
    let mut out = String::with_capacity(template.len());
    let bytes = template.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'{' {
            if let Some(end) = find_matching_brace(bytes, i + 1) {
                let key = &template[i + 1..end];
                if let Some(replacement) = args.get(key) {
                    out.push_str(replacement);
                    i = end + 1;
                    continue;
                }
            }
        }
        out.push(template.as_bytes()[i] as char);
        i += 1;
    }
    out
}

fn find_matching_brace(bytes: &[u8], start: usize) -> Option<usize> {
    bytes[start..]
        .iter()
        .position(|&b| b == b'}')
        .map(|offset| start + offset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::write;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Inline tempdir — same pattern as in `localized_manifest::tests`.
    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new() -> Self {
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!("ari-localized-strings-{nanos}-{n}"));
            std::fs::create_dir_all(&path).expect("create temp dir");
            Self { path }
        }
        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    fn make_strings_dir(skill_dir: &Path) -> PathBuf {
        let dir = skill_dir.join("strings");
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn missing_strings_dir_yields_empty() {
        let td = TempDir::new();
        let strings = parse_strings_directory(td.path()).expect("missing dir is fine");
        assert!(strings.is_empty());
        assert_eq!(strings.supported_locales(), Vec::<String>::new());
        assert_eq!(strings.get("en", "any.key"), None);
    }

    #[test]
    fn parses_canonical_only() {
        let td = TempDir::new();
        let dir = make_strings_dir(td.path());
        write(
            dir.join("en.json"),
            r#"{"greet.hello": "Hello!", "greet.bye": "Goodbye {name}."}"#,
        )
        .unwrap();

        let strings = parse_strings_directory(td.path()).expect("should parse");
        assert_eq!(strings.supported_locales(), vec!["en".to_string()]);
        assert_eq!(strings.get("en", "greet.hello"), Some("Hello!"));
        assert_eq!(strings.get("en", "missing"), None);
    }

    #[test]
    fn parses_canonical_plus_italian() {
        let td = TempDir::new();
        let dir = make_strings_dir(td.path());
        write(dir.join("en.json"), r#"{"greet.hello": "Hello!"}"#).unwrap();
        write(dir.join("it.json"), r#"{"greet.hello": "Ciao!"}"#).unwrap();

        let strings = parse_strings_directory(td.path()).expect("should parse both");
        assert_eq!(
            strings.supported_locales(),
            vec!["en".to_string(), "it".to_string()]
        );
        assert_eq!(strings.get("en", "greet.hello"), Some("Hello!"));
        assert_eq!(strings.get("it", "greet.hello"), Some("Ciao!"));
    }

    #[test]
    fn fallback_to_canonical_when_locale_missing_key() {
        let td = TempDir::new();
        let dir = make_strings_dir(td.path());
        write(
            dir.join("en.json"),
            r#"{"greet.hello": "Hello!", "greet.bye": "Goodbye!"}"#,
        )
        .unwrap();
        // Italian only translates "hello"; "bye" should fall back to en.
        write(dir.join("it.json"), r#"{"greet.hello": "Ciao!"}"#).unwrap();

        let strings = parse_strings_directory(td.path()).expect("should parse");
        assert_eq!(strings.get("it", "greet.hello"), Some("Ciao!"));
        assert_eq!(strings.get("it", "greet.bye"), Some("Goodbye!"));
    }

    #[test]
    fn fallback_to_canonical_when_locale_completely_missing() {
        let td = TempDir::new();
        let dir = make_strings_dir(td.path());
        write(dir.join("en.json"), r#"{"greet.hello": "Hello!"}"#).unwrap();

        let strings = parse_strings_directory(td.path()).expect("should parse");
        // No it.json at all, but en.json has the key.
        assert_eq!(strings.get("it", "greet.hello"), Some("Hello!"));
        assert_eq!(strings.get("es", "greet.hello"), Some("Hello!"));
    }

    #[test]
    fn rejects_strings_without_canonical() {
        let td = TempDir::new();
        let dir = make_strings_dir(td.path());
        // Italian only, no English. Without en there's no fallback
        // for any other locale we might add later — disallow.
        write(dir.join("it.json"), r#"{"greet.hello": "Ciao!"}"#).unwrap();

        let err = parse_strings_directory(td.path()).expect_err("must reject missing en.json");
        assert!(matches!(err, LocalizedStringsError::MissingCanonical));
    }

    #[test]
    fn rejects_invalid_locale_filename() {
        let td = TempDir::new();
        let dir = make_strings_dir(td.path());
        write(dir.join("en.json"), r#"{}"#).unwrap();
        // 5-char "locale" — clearly not ISO 639-1, hard reject.
        write(dir.join("notes.json"), r#"{"k": "v"}"#).unwrap();

        let err = parse_strings_directory(td.path()).expect_err("must reject notes.json");
        match err {
            LocalizedStringsError::InvalidLocaleFilename { filename } => {
                assert_eq!(filename, "notes.json");
            }
            other => panic!("expected InvalidLocaleFilename, got {other:?}"),
        }
    }

    #[test]
    fn rejects_nested_json_object() {
        let td = TempDir::new();
        let dir = make_strings_dir(td.path());
        // Nested object — we want flat key→template only. Nested
        // structures are a footgun the SDK can't render via t().
        write(
            dir.join("en.json"),
            r#"{"greet": {"hello": "Hello!"}}"#,
        )
        .unwrap();

        let err = parse_strings_directory(td.path()).expect_err("must reject nested object");
        assert!(matches!(err, LocalizedStringsError::Parse { .. }));
    }

    #[test]
    fn ignores_non_json_files_in_strings_dir() {
        let td = TempDir::new();
        let dir = make_strings_dir(td.path());
        write(dir.join("en.json"), r#"{"hello": "Hi"}"#).unwrap();
        write(dir.join("README.md"), "# Strings notes for translators").unwrap();
        write(dir.join("notes.txt"), "translator scratch pad").unwrap();

        let strings = parse_strings_directory(td.path()).expect("non-json must be ignored");
        assert_eq!(strings.supported_locales(), vec!["en".to_string()]);
    }

    #[test]
    fn render_substitutes_placeholders() {
        let td = TempDir::new();
        let dir = make_strings_dir(td.path());
        write(
            dir.join("en.json"),
            r#"{"greet": "Hi {name}, you have {count} messages."}"#,
        )
        .unwrap();

        let strings = parse_strings_directory(td.path()).expect("should parse");
        let mut args = BTreeMap::new();
        args.insert("name".to_string(), "Keith".to_string());
        args.insert("count".to_string(), "3".to_string());
        let rendered = strings.render("en", "greet", &args).unwrap();
        assert_eq!(rendered, "Hi Keith, you have 3 messages.");
    }

    #[test]
    fn render_leaves_unmatched_placeholders_intact() {
        // Visible-failure UX: a typoed `{nmae}` stays visible in the
        // rendered output so the dev catches it, instead of silently
        // disappearing.
        let td = TempDir::new();
        let dir = make_strings_dir(td.path());
        write(dir.join("en.json"), r#"{"greet": "Hi {nmae}!"}"#).unwrap();

        let strings = parse_strings_directory(td.path()).expect("should parse");
        let mut args = BTreeMap::new();
        args.insert("name".to_string(), "Keith".to_string());
        let rendered = strings.render("en", "greet", &args).unwrap();
        assert_eq!(rendered, "Hi {nmae}!");
    }

    #[test]
    fn render_returns_none_for_unknown_key() {
        let td = TempDir::new();
        let dir = make_strings_dir(td.path());
        write(dir.join("en.json"), r#"{"greet": "Hi"}"#).unwrap();

        let strings = parse_strings_directory(td.path()).expect("should parse");
        let args = BTreeMap::new();
        assert!(strings.render("en", "missing.key", &args).is_none());
    }

    #[test]
    fn render_falls_back_to_canonical_locale() {
        let td = TempDir::new();
        let dir = make_strings_dir(td.path());
        write(
            dir.join("en.json"),
            r#"{"greet": "Hello {name}!", "bye": "Goodbye"}"#,
        )
        .unwrap();
        write(dir.join("it.json"), r#"{"greet": "Ciao {name}!"}"#).unwrap();

        let strings = parse_strings_directory(td.path()).expect("should parse");
        let mut args = BTreeMap::new();
        args.insert("name".to_string(), "Keith".to_string());
        // Italian has "greet" → uses Italian.
        assert_eq!(
            strings.render("it", "greet", &args).unwrap(),
            "Ciao Keith!"
        );
        // Italian missing "bye" → falls back to English.
        assert_eq!(
            strings.render("it", "bye", &args).unwrap(),
            "Goodbye"
        );
    }
}
