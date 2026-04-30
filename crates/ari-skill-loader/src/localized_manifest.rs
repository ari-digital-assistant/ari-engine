//! Per-locale skill manifest loader.
//!
//! A skill directory can contain multiple `SKILL.{locale}.md` files,
//! each a complete localized manifest. They share structural identity
//! (`id`, `type`, `capabilities`, `behaviour`) but legitimately differ
//! in `name`, `description`, `body`, and matching patterns. This module
//! reads them all, validates cross-file consistency, and surfaces them
//! as a [`LocalizedManifestSet`].
//!
//! ## Layout
//!
//! - `SKILL.en.md` — canonical English manifest. Required (or fall back
//!   to the legacy bare `SKILL.md` for backwards compatibility during
//!   the migration window).
//! - `SKILL.{lang}.md` — additional localized manifests, one per ISO
//!   639-1 lowercase code (e.g. `SKILL.it.md`, `SKILL.es.md`). Optional.
//! - `SKILL.md` — legacy single-file manifest. Treated as if it were
//!   `SKILL.en.md` when no `SKILL.en.md` exists in the same directory.
//!   **Both present at once is a hard reject** — migration is a rename,
//!   not a duplicate.
//!
//! ## Validation rules
//!
//! 1. Every locale variant must declare an `metadata.ari` block — a
//!    plain AgentSkills doc without it isn't an Ari skill.
//! 2. The canonical (`en`) manifest's structural fields define the
//!    skill: `id`, `type`, `capabilities`, `behaviour`. Every other
//!    locale variant must agree on those fields exactly.
//! 3. Localized fields (`name`, `description`, `body`, `matching.patterns`)
//!    may differ across locale variants.
//! 4. Mismatches are hard rejects, never warnings — silent drift between
//!    locales is exactly the bug class translation tooling needs to
//!    surface, not paper over.

use crate::manifest::{Behaviour, Capability, ManifestError, Skillfile};
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Locale code that defines the structural truth for a skill. Every
/// other locale variant must agree with this one on `id`, `type`,
/// `capabilities`, and `behaviour`.
pub const CANONICAL_LOCALE: &str = "en";

/// All locale-specific manifests for a single skill, keyed by ISO 639-1
/// lowercase code. The [`CANONICAL_LOCALE`] entry is guaranteed present;
/// every other entry has been validated as structurally consistent with it.
#[derive(Debug, Clone, PartialEq)]
pub struct LocalizedManifestSet {
    /// Per-locale manifests. Sorted by locale code (BTreeMap) for
    /// stable iteration order — matters for the supported-languages
    /// list that the skill index exposes to the frontend.
    pub manifests: BTreeMap<String, Skillfile>,
}

impl LocalizedManifestSet {
    /// The canonical (English) manifest. Always present — the loader
    /// would have rejected the directory if it weren't.
    pub fn canonical(&self) -> &Skillfile {
        self.manifests
            .get(CANONICAL_LOCALE)
            .expect("canonical manifest must be present (constructor invariant)")
    }

    /// The manifest for the given locale, or the canonical manifest
    /// if the requested locale isn't supported. This is the
    /// best-effort fallback rule used by skill matching, the skill
    /// browser display, and any other consumer that needs to render
    /// a skill in the user's locale without failing on missing
    /// translations.
    pub fn for_locale(&self, locale: &str) -> &Skillfile {
        self.manifests
            .get(locale)
            .unwrap_or_else(|| self.canonical())
    }

    /// Locale codes the skill ships translations for. Always includes
    /// [`CANONICAL_LOCALE`] plus any additional `SKILL.{locale}.md`
    /// files that parsed cleanly. Sorted alphabetically.
    pub fn supported_locales(&self) -> Vec<String> {
        self.manifests.keys().cloned().collect()
    }
}

#[derive(Debug, Error)]
pub enum LocalizedManifestError {
    /// Neither `SKILL.md` nor `SKILL.en.md` was found in the skill dir.
    #[error("skill directory has no SKILL.en.md or SKILL.md")]
    MissingCanonical,

    /// Both `SKILL.md` and `SKILL.en.md` exist — author needs to pick
    /// one. The migration story is a rename, not a duplicate.
    #[error(
        "skill directory contains both SKILL.md and SKILL.en.md; \
         rename SKILL.md to SKILL.en.md (and delete the old one) — \
         a single skill cannot have both"
    )]
    DuplicateCanonical,

    /// A file matching the `SKILL.{X}.md` pattern had a locale segment
    /// that wasn't a 2-character lowercase ASCII string (ISO 639-1).
    #[error(
        "`{filename}` doesn't follow the SKILL.{{locale}}.md pattern — \
         locale segment must be 2 lowercase ASCII letters (ISO 639-1)"
    )]
    InvalidLocaleFilename { filename: String },

    /// One of the manifests failed to parse.
    #[error("parsing {path}: {error}")]
    Parse {
        path: PathBuf,
        #[source]
        error: ManifestError,
    },

    /// I/O failure reading the skill directory or one of its files.
    #[error("I/O failure on {path}: {message}")]
    Io { path: PathBuf, message: String },

    /// A non-canonical manifest's structural fields disagree with the
    /// canonical English manifest. Localized fields (name, description,
    /// body, patterns) can differ; structural fields must not.
    #[error(
        "{locale}: SKILL.{locale}.md disagrees with SKILL.en.md on `{field}`. \
         Each per-locale manifest must agree on id, type, capabilities, and behaviour. \
         Localized name, description, body, and patterns are allowed to differ."
    )]
    StructuralMismatch { locale: String, field: String },

    /// A manifest is missing the `metadata.ari` block entirely — it's
    /// a valid AgentSkills doc but not an Ari skill.
    #[error("{locale}: manifest has no `metadata.ari` block; not an Ari skill")]
    MissingAriExtension { locale: String },
}

/// Scan a skill directory for `SKILL.{locale}.md` files (and the legacy
/// `SKILL.md`), parse each, validate cross-file consistency, and return
/// the resulting [`LocalizedManifestSet`].
pub fn parse_skill_directory(dir: &Path) -> Result<LocalizedManifestSet, LocalizedManifestError> {
    let entries = std::fs::read_dir(dir).map_err(|e| LocalizedManifestError::Io {
        path: dir.to_path_buf(),
        message: format!("could not read directory: {e}"),
    })?;

    let mut legacy_skill_md: Option<PathBuf> = None;
    let mut locale_files: Vec<(String, PathBuf)> = Vec::new();

    for entry in entries {
        let entry = entry.map_err(|e| LocalizedManifestError::Io {
            path: dir.to_path_buf(),
            message: format!("dir entry: {e}"),
        })?;
        let path = entry.path();
        let Some(filename) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };

        if filename == "SKILL.md" {
            legacy_skill_md = Some(path);
        } else if let Some(locale) = parse_locale_filename(filename)? {
            locale_files.push((locale, path));
        }
    }

    let has_canonical_locale_file = locale_files
        .iter()
        .any(|(loc, _)| loc == CANONICAL_LOCALE);

    if let Some(ref legacy) = legacy_skill_md {
        if has_canonical_locale_file {
            return Err(LocalizedManifestError::DuplicateCanonical);
        }
        // Legacy `SKILL.md` acts as `SKILL.en.md` for backward
        // compatibility. Skill authors are encouraged to rename;
        // the loader continues to accept the old name as long as
        // it's the only canonical-locale file.
        locale_files.push((CANONICAL_LOCALE.to_string(), legacy.clone()));
    }

    if !locale_files.iter().any(|(loc, _)| loc == CANONICAL_LOCALE) {
        return Err(LocalizedManifestError::MissingCanonical);
    }

    let parent_dir_name = dir.file_name().and_then(|n| n.to_str());
    let mut manifests: BTreeMap<String, Skillfile> = BTreeMap::new();
    for (locale, path) in &locale_files {
        let source = std::fs::read_to_string(path).map_err(|e| LocalizedManifestError::Io {
            path: path.clone(),
            message: format!("could not read: {e}"),
        })?;
        let sf = Skillfile::parse(&source, parent_dir_name).map_err(|e| {
            LocalizedManifestError::Parse {
                path: path.clone(),
                error: e,
            }
        })?;
        manifests.insert(locale.clone(), sf);
    }

    validate_cross_file_consistency(&manifests)?;

    Ok(LocalizedManifestSet { manifests })
}

/// Pull the locale code out of a `SKILL.{locale}.md` filename.
///
/// - `SKILL.en.md` → `Some("en")`
/// - `SKILL.it.md` → `Some("it")`
/// - `README.md`, `skill.wasm`, etc. → `None` (caller skips)
/// - `SKILL.foo.md` (locale segment isn't 2 lowercase ASCII letters) →
///   [`LocalizedManifestError::InvalidLocaleFilename`]
fn parse_locale_filename(filename: &str) -> Result<Option<String>, LocalizedManifestError> {
    let Some(stem) = filename.strip_suffix(".md") else {
        return Ok(None);
    };
    let Some(locale_part) = stem.strip_prefix("SKILL.") else {
        return Ok(None);
    };
    if locale_part.len() != 2 || !locale_part.chars().all(|c| c.is_ascii_lowercase()) {
        return Err(LocalizedManifestError::InvalidLocaleFilename {
            filename: filename.to_string(),
        });
    }
    Ok(Some(locale_part.to_string()))
}

fn validate_cross_file_consistency(
    manifests: &BTreeMap<String, Skillfile>,
) -> Result<(), LocalizedManifestError> {
    let canonical = manifests
        .get(CANONICAL_LOCALE)
        .expect("canonical guaranteed by caller");
    let canonical_ari = canonical
        .ari_extension
        .as_ref()
        .ok_or_else(|| LocalizedManifestError::MissingAriExtension {
            locale: CANONICAL_LOCALE.to_string(),
        })?;

    for (locale, sf) in manifests {
        if locale == CANONICAL_LOCALE {
            continue;
        }
        let other_ari = sf
            .ari_extension
            .as_ref()
            .ok_or_else(|| LocalizedManifestError::MissingAriExtension {
                locale: locale.clone(),
            })?;

        // id — every locale variant of the same skill must share its id.
        if other_ari.id != canonical_ari.id {
            return Err(LocalizedManifestError::StructuralMismatch {
                locale: locale.clone(),
                field: "metadata.ari.id".to_string(),
            });
        }
        // type — a skill can't be a regular skill in English and an
        // assistant in Italian.
        if other_ari.skill_type != canonical_ari.skill_type {
            return Err(LocalizedManifestError::StructuralMismatch {
                locale: locale.clone(),
                field: "metadata.ari.type".to_string(),
            });
        }
        // capabilities — set equality. Authors might list them in
        // different orders across files; we don't care, but the
        // SET must be identical.
        if !capabilities_equal(&canonical_ari.capabilities, &other_ari.capabilities) {
            return Err(LocalizedManifestError::StructuralMismatch {
                locale: locale.clone(),
                field: "metadata.ari.capabilities".to_string(),
            });
        }
        // behaviour — same WASM module path, same declarative response
        // shape. A locale-specific behaviour change would be hidden
        // routing divergence; not something we want to silently allow.
        if !behaviour_equal(&canonical_ari.behaviour, &other_ari.behaviour) {
            return Err(LocalizedManifestError::StructuralMismatch {
                locale: locale.clone(),
                field: "metadata.ari.behaviour".to_string(),
            });
        }
    }

    Ok(())
}

fn capabilities_equal(a: &[Capability], b: &[Capability]) -> bool {
    let set_a: HashSet<&Capability> = a.iter().collect();
    let set_b: HashSet<&Capability> = b.iter().collect();
    set_a == set_b
}

fn behaviour_equal(a: &Option<Behaviour>, b: &Option<Behaviour>) -> bool {
    // Behaviour already derives PartialEq, but we wrap in a function so
    // future schema changes (e.g. ignoring a comment-only field on the
    // declarative response) have one place to land.
    a == b
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::write;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Minimal self-contained tempdir for the tests in this module.
    /// Creates a uniquely-named directory under [`std::env::temp_dir`]
    /// and removes it on drop. Same pattern as `tempdir_lite` in
    /// [`crate::loader`]; duplicated here to keep test deps zero.
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
            let path = std::env::temp_dir().join(format!("ari-localized-manifest-{nanos}-{n}"));
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

    /// Minimal valid English skill with a keyword pattern. The YAML
    /// shape mirrors the canonical examples in `manifest::tests`:
    /// `declarative` and `wasm` sit directly under `metadata.ari`,
    /// not nested in a `behaviour:` map.
    fn english_skill(id_suffix: &str, dir_name: &str) -> String {
        format!(
            r#"---
name: {dir_name}
description: A test skill
metadata:
  ari:
    id: dev.heyari.test.{id_suffix}
    version: "1.0.0"
    engine: ">=0.3,<0.4"
    capabilities: []
    languages: [en]
    specificity: medium
    matching:
      patterns:
        - keywords: [hello]
          weight: 1.0
    declarative:
      response: "hi"
    examples:
      - text: hello
      - text: hi there
      - text: hey
      - text: good morning
      - text: greetings
---
"#
        )
    }

    /// Italian counterpart of `english_skill` — same id/type/capabilities/
    /// behaviour, different name/description/patterns.
    fn italian_skill(id_suffix: &str, dir_name: &str) -> String {
        format!(
            r#"---
name: {dir_name}
description: Una skill di prova
metadata:
  ari:
    id: dev.heyari.test.{id_suffix}
    version: "1.0.0"
    engine: ">=0.3,<0.4"
    capabilities: []
    languages: [it]
    specificity: medium
    matching:
      patterns:
        - keywords: [ciao]
          weight: 1.0
    declarative:
      response: "hi"
    examples:
      - text: ciao
      - text: salve
      - text: buongiorno
      - text: buonasera
      - text: ehi
---
"#
        )
    }

    #[test]
    fn parses_canonical_only() {
        let td = TempDir::new();
        let dir = td.path().join("greet");
        std::fs::create_dir(&dir).unwrap();
        write(dir.join("SKILL.en.md"), english_skill("greet", "greet")).unwrap();

        let set = parse_skill_directory(&dir).expect("should parse");
        assert_eq!(set.supported_locales(), vec!["en".to_string()]);
        assert_eq!(set.canonical().description, "A test skill");
    }

    #[test]
    fn legacy_skill_md_acts_as_canonical() {
        let td = TempDir::new();
        let dir = td.path().join("greet");
        std::fs::create_dir(&dir).unwrap();
        write(dir.join("SKILL.md"), english_skill("greet", "greet")).unwrap();

        let set = parse_skill_directory(&dir).expect("should parse legacy SKILL.md as en");
        assert_eq!(set.supported_locales(), vec!["en".to_string()]);
    }

    #[test]
    fn rejects_skill_md_and_skill_en_md_together() {
        let td = TempDir::new();
        let dir = td.path().join("greet");
        std::fs::create_dir(&dir).unwrap();
        write(dir.join("SKILL.md"), english_skill("greet", "greet")).unwrap();
        write(dir.join("SKILL.en.md"), english_skill("greet", "greet")).unwrap();

        let err = parse_skill_directory(&dir).expect_err("must reject duplicate canonical");
        assert!(
            matches!(err, LocalizedManifestError::DuplicateCanonical),
            "expected DuplicateCanonical, got {err:?}"
        );
    }

    #[test]
    fn parses_canonical_plus_italian() {
        let td = TempDir::new();
        let dir = td.path().join("greet");
        std::fs::create_dir(&dir).unwrap();
        write(dir.join("SKILL.en.md"), english_skill("greet", "greet")).unwrap();
        write(dir.join("SKILL.it.md"), italian_skill("greet", "greet")).unwrap();

        let set = parse_skill_directory(&dir).expect("should parse both");
        assert_eq!(set.supported_locales(), vec!["en".to_string(), "it".to_string()]);
        // Localized fields differ.
        let en = set.for_locale("en");
        let it = set.for_locale("it");
        assert_eq!(en.description, "A test skill");
        assert_eq!(it.description, "Una skill di prova");
        // Structural fields agree (the validator wouldn't have let us
        // get here otherwise) — sanity-check the id one more time.
        let en_ari = en.ari_extension.as_ref().unwrap();
        let it_ari = it.ari_extension.as_ref().unwrap();
        assert_eq!(en_ari.id, it_ari.id);
    }

    #[test]
    fn for_locale_falls_back_to_canonical_when_missing() {
        let td = TempDir::new();
        let dir = td.path().join("greet");
        std::fs::create_dir(&dir).unwrap();
        write(dir.join("SKILL.en.md"), english_skill("greet", "greet")).unwrap();

        let set = parse_skill_directory(&dir).expect("should parse");
        let canonical = set.canonical();
        let it = set.for_locale("it");
        // No SKILL.it.md, so for_locale("it") returns the canonical.
        assert_eq!(canonical.description, it.description);
    }

    #[test]
    fn rejects_when_no_canonical_present() {
        let td = TempDir::new();
        let dir = td.path().join("greet");
        std::fs::create_dir(&dir).unwrap();
        // Italian only, no English. Doesn't matter how good the
        // Italian translation is — without a canonical we can't
        // validate structural identity.
        write(dir.join("SKILL.it.md"), italian_skill("greet", "greet")).unwrap();

        let err = parse_skill_directory(&dir).expect_err("must reject missing canonical");
        assert!(
            matches!(err, LocalizedManifestError::MissingCanonical),
            "expected MissingCanonical, got {err:?}"
        );
    }

    #[test]
    fn rejects_id_mismatch_across_locales() {
        let td = TempDir::new();
        let dir = td.path().join("greet");
        std::fs::create_dir(&dir).unwrap();
        // Italian variant has a different id — structural disagreement.
        write(dir.join("SKILL.en.md"), english_skill("greet_a", "greet")).unwrap();
        write(dir.join("SKILL.it.md"), italian_skill("greet_b", "greet")).unwrap();

        let err = parse_skill_directory(&dir).expect_err("must reject id mismatch");
        match err {
            LocalizedManifestError::StructuralMismatch { locale, field } => {
                assert_eq!(locale, "it");
                assert_eq!(field, "metadata.ari.id");
            }
            other => panic!("expected StructuralMismatch on id, got {other:?}"),
        }
    }

    #[test]
    fn rejects_capabilities_mismatch_across_locales() {
        let td = TempDir::new();
        let dir = td.path().join("greet");
        std::fs::create_dir(&dir).unwrap();

        // English: empty capabilities.
        write(dir.join("SKILL.en.md"), english_skill("greet", "greet")).unwrap();
        // Italian: declares `notifications` capability — not allowed
        // to differ from canonical.
        let italian_with_extra_cap = r#"---
name: greet
description: Una skill di prova
metadata:
  ari:
    id: dev.heyari.test.greet
    version: "1.0.0"
    engine: ">=0.3,<0.4"
    capabilities: [notifications]
    languages: [it]
    specificity: medium
    matching:
      patterns:
        - keywords: [ciao]
          weight: 1.0
    declarative:
      response: "hi"
    examples:
      - text: ciao
      - text: salve
      - text: buongiorno
      - text: buonasera
      - text: ehi
---
"#;
        write(dir.join("SKILL.it.md"), italian_with_extra_cap).unwrap();

        let err = parse_skill_directory(&dir).expect_err("must reject capabilities mismatch");
        match err {
            LocalizedManifestError::StructuralMismatch { locale, field } => {
                assert_eq!(locale, "it");
                assert_eq!(field, "metadata.ari.capabilities");
            }
            other => panic!("expected StructuralMismatch on capabilities, got {other:?}"),
        }
    }

    #[test]
    fn invalid_locale_filename_is_rejected() {
        let td = TempDir::new();
        let dir = td.path().join("greet");
        std::fs::create_dir(&dir).unwrap();
        write(dir.join("SKILL.en.md"), english_skill("greet", "greet")).unwrap();
        // 3-char "locale" is not ISO 639-1 — should fail loudly so
        // the author doesn't get silent skipping of their non-English
        // file.
        write(dir.join("SKILL.foo.md"), english_skill("greet", "greet")).unwrap();

        let err = parse_skill_directory(&dir).expect_err("must reject SKILL.foo.md");
        match err {
            LocalizedManifestError::InvalidLocaleFilename { filename } => {
                assert_eq!(filename, "SKILL.foo.md");
            }
            other => panic!("expected InvalidLocaleFilename, got {other:?}"),
        }
    }

    #[test]
    fn ignores_unrelated_files() {
        let td = TempDir::new();
        let dir = td.path().join("greet");
        std::fs::create_dir(&dir).unwrap();
        write(dir.join("SKILL.en.md"), english_skill("greet", "greet")).unwrap();
        write(dir.join("README.md"), "# Notes").unwrap();
        write(dir.join("skill.wasm"), b"not actually wasm").unwrap();
        std::fs::create_dir(dir.join("strings")).unwrap();
        write(dir.join("strings/en.json"), "{}").unwrap();

        let set = parse_skill_directory(&dir).expect("non-SKILL files must be ignored");
        assert_eq!(set.supported_locales(), vec!["en".to_string()]);
    }

    #[test]
    fn supported_locales_sorted_alphabetically() {
        let td = TempDir::new();
        let dir = td.path().join("greet");
        std::fs::create_dir(&dir).unwrap();
        write(dir.join("SKILL.en.md"), english_skill("greet", "greet")).unwrap();
        write(dir.join("SKILL.it.md"), italian_skill("greet", "greet")).unwrap();
        // Insert Spanish that deliberately matches structural fields
        // by reusing the english template (description text is the
        // same as the English; that's fine — name/description CAN
        // match and we don't care, only that they're allowed to differ).
        write(dir.join("SKILL.es.md"), english_skill("greet", "greet")).unwrap();

        let set = parse_skill_directory(&dir).expect("should parse three locales");
        assert_eq!(
            set.supported_locales(),
            vec!["en".to_string(), "es".to_string(), "it".to_string()]
        );
    }
}
