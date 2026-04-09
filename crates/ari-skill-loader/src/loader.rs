//! Disk-scanning loader: turns a directory of skill folders into ready-to-use
//! `Skill` trait objects.
//!
//! Layout convention (matches the `ari-digital-assistant/ari-skills` registry
//! and the AgentSkills directory format):
//!
//! ```text
//! <root>/
//! ├── coin-flip/
//! │   └── SKILL.md
//! ├── greet/
//! │   └── SKILL.md
//! └── ...
//! ```
//!
//! Each direct child of `<root>` that contains a `SKILL.md` is parsed and, if
//! it carries a `metadata.ari` extension, instantiated. Documents without the
//! extension are silently skipped (they're valid AgentSkills documents that
//! just aren't Ari skills). WASM skills are reported as `LoadFailure::Wasm`
//! for now and will be picked up by the WASM adapter in step 5.
//!
//! The loader can also be pointed at a single skill directory directly via
//! [`load_single_skill_dir`] — useful for the CLI's `--extra-skill-dir` flag
//! when a developer is sideloading a work-in-progress skill.

use crate::declarative::{AdapterError, DeclarativeSkill};
use crate::host_capabilities::HostCapabilities;
use crate::http_config::HttpConfig;
use crate::manifest::{Behaviour, Capability, ManifestError, Skillfile};
use crate::storage_config::StorageConfig;
use crate::wasm::{LogSink, NullLogSink, WasmError, WasmSkill};
use ari_core::Skill;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// What went wrong loading a single skill. The loader keeps going on
/// per-skill failures so one bad apple doesn't sink the rest.
#[derive(Debug)]
pub struct LoadFailure {
    pub path: PathBuf,
    pub kind: LoadFailureKind,
}

#[derive(Debug)]
pub enum LoadFailureKind {
    /// The directory contained no `SKILL.md`.
    MissingSkillfile,
    /// The `SKILL.md` couldn't be parsed.
    Manifest(ManifestError),
    /// Adapter rejected the parsed skillfile (e.g. bad regex).
    Adapter(AdapterError),
    /// WASM adapter failed to load the module (file missing, bad signature, etc).
    Wasm(WasmError),
    /// Skill declared host capabilities the host doesn't provide. Applies to
    /// both declarative and WASM skills — declarative ones can declare caps
    /// like `notifications` for honesty even though they don't need imports.
    MissingCapabilities { missing: Vec<Capability> },
    /// IO error walking the directory.
    Io(std::io::Error),
}

impl std::fmt::Display for LoadFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            LoadFailureKind::MissingSkillfile => {
                write!(f, "{}: no SKILL.md found", self.path.display())
            }
            LoadFailureKind::Manifest(e) => write!(f, "{}: {e}", self.path.display()),
            LoadFailureKind::Adapter(e) => write!(f, "{}: {e}", self.path.display()),
            LoadFailureKind::Wasm(e) => write!(f, "{}: {e}", self.path.display()),
            LoadFailureKind::MissingCapabilities { missing } => write!(
                f,
                "{}: skill declares host capabilities not provided by this host: {:?}",
                self.path.display(),
                missing
            ),
            LoadFailureKind::Io(e) => write!(f, "{}: {e}", self.path.display()),
        }
    }
}

impl std::error::Error for LoadFailure {}

/// What came back from a load operation.
pub struct LoadReport {
    pub skills: Vec<Box<dyn Skill>>,
    pub failures: Vec<LoadFailure>,
}

impl LoadReport {
    fn new() -> Self {
        Self {
            skills: Vec::new(),
            failures: Vec::new(),
        }
    }

    fn merge(&mut self, other: LoadReport) {
        self.skills.extend(other.skills);
        self.failures.extend(other.failures);
    }
}

/// Options bag passed to the loader.
#[derive(Clone)]
pub struct LoadOptions {
    pub log_sink: Arc<dyn LogSink>,
    pub host_capabilities: HostCapabilities,
    pub http_config: HttpConfig,
    pub storage_config: StorageConfig,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            log_sink: Arc::new(NullLogSink),
            host_capabilities: HostCapabilities::pure_frontend(),
            http_config: HttpConfig::strict(),
            storage_config: StorageConfig::ephemeral_default(),
        }
    }
}

/// Scan `root` for skill subdirectories. Each direct child is treated as one
/// skill. Returns successfully-loaded skills alongside any per-skill failures.
/// IO errors reading `root` itself bubble up as `Err`. Uses default options
/// (no-op log sink, pure-frontend host capabilities).
pub fn load_skill_directory(root: &Path) -> std::io::Result<LoadReport> {
    load_skill_directory_with(root, &LoadOptions::default())
}

/// Same as [`load_skill_directory`] but lets the caller pass options.
pub fn load_skill_directory_with(
    root: &Path,
    options: &LoadOptions,
) -> std::io::Result<LoadReport> {
    let mut report = LoadReport::new();
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        report.merge(load_single_skill_dir_with(&path, options));
    }
    Ok(report)
}

/// Load a single skill from its directory. The directory name must match the
/// AgentSkills `name` field in `SKILL.md`.
pub fn load_single_skill_dir(skill_dir: &Path) -> LoadReport {
    load_single_skill_dir_with(skill_dir, &LoadOptions::default())
}

/// Same as [`load_single_skill_dir`] but lets the caller pass options.
pub fn load_single_skill_dir_with(skill_dir: &Path, options: &LoadOptions) -> LoadReport {
    let mut report = LoadReport::new();
    let manifest_path = skill_dir.join("SKILL.md");
    if !manifest_path.is_file() {
        report.failures.push(LoadFailure {
            path: skill_dir.to_path_buf(),
            kind: LoadFailureKind::MissingSkillfile,
        });
        return report;
    }

    let sf = match Skillfile::parse_file(&manifest_path) {
        Ok(sf) => sf,
        Err(e) => {
            report.failures.push(LoadFailure {
                path: manifest_path,
                kind: LoadFailureKind::Manifest(e),
            });
            return report;
        }
    };

    let Some(ari) = sf.ari_extension.as_ref() else {
        // Valid AgentSkills doc but not an Ari skill — silently skip.
        return report;
    };

    // Capability check applies to BOTH declarative and WASM skills. A
    // declarative skill that promises `notifications` is making the same
    // user-consent claim as a WASM one — we honour or reject identically.
    let missing = options.host_capabilities.missing_for(&ari.capabilities);
    if !missing.is_empty() {
        report.failures.push(LoadFailure {
            path: manifest_path,
            kind: LoadFailureKind::MissingCapabilities { missing },
        });
        return report;
    }

    match &ari.behaviour {
        Behaviour::Declarative(_) => match DeclarativeSkill::from_skillfile(&sf) {
            Ok(skill) => report.skills.push(Box::new(skill)),
            Err(e) => report.failures.push(LoadFailure {
                path: manifest_path,
                kind: LoadFailureKind::Adapter(e),
            }),
        },
        Behaviour::Wasm(_) => match WasmSkill::from_skillfile(
            &sf,
            skill_dir,
            options.log_sink.clone(),
            &options.host_capabilities,
            &options.http_config,
            &options.storage_config,
        ) {
            Ok(skill) => report.skills.push(Box::new(skill)),
            Err(e) => report.failures.push(LoadFailure {
                path: manifest_path,
                kind: LoadFailureKind::Wasm(e),
            }),
        },
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;

    fn write(path: &Path, content: &str) {
        if let Some(p) = path.parent() {
            fs::create_dir_all(p).unwrap();
        }
        let mut f = fs::File::create(path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
    }

    fn coin_flip_md() -> &'static str {
        r#"---
name: coin-flip
description: Flips a virtual coin and returns heads or tails. Use when the user asks to flip a coin.
metadata:
  ari:
    id: ai.example.coinflip
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

    fn greet_md() -> &'static str {
        r#"---
name: greet
description: Greets the user. Use when the user says hello.
metadata:
  ari:
    id: ai.example.greet
    version: "0.1.0"
    engine: ">=0.3"
    matching:
      patterns:
        - keywords: [hello]
    declarative:
      response: "Oh hey."
---
"#
    }

    fn tmpdir() -> tempdir_lite::TempDir {
        tempdir_lite::TempDir::new("ari-loader-test")
    }

    #[test]
    fn loads_two_skills_from_a_directory() {
        let dir = tmpdir();
        write(&dir.path().join("coin-flip/SKILL.md"), coin_flip_md());
        write(&dir.path().join("greet/SKILL.md"), greet_md());

        let report = load_skill_directory(dir.path()).unwrap();
        assert_eq!(report.failures.len(), 0);
        assert_eq!(report.skills.len(), 2);
        let ids: Vec<_> = report.skills.iter().map(|s| s.id().to_string()).collect();
        assert!(ids.contains(&"ai.example.coinflip".to_string()));
        assert!(ids.contains(&"ai.example.greet".to_string()));
    }

    #[test]
    fn skips_directories_with_no_skill_md() {
        let dir = tmpdir();
        fs::create_dir_all(dir.path().join("not-a-skill")).unwrap();
        write(&dir.path().join("greet/SKILL.md"), greet_md());

        let report = load_skill_directory(dir.path()).unwrap();
        assert_eq!(report.skills.len(), 1);
        assert_eq!(report.failures.len(), 1);
        assert!(matches!(
            report.failures[0].kind,
            LoadFailureKind::MissingSkillfile
        ));
    }

    #[test]
    fn skips_plain_agentskills_doc_without_ari_extension() {
        let dir = tmpdir();
        write(
            &dir.path().join("pdf-tools/SKILL.md"),
            "---\nname: pdf-tools\ndescription: Helps with PDFs.\n---\n",
        );
        write(&dir.path().join("greet/SKILL.md"), greet_md());

        let report = load_skill_directory(dir.path()).unwrap();
        // pdf-tools is silently skipped, only greet loads
        assert_eq!(report.skills.len(), 1);
        assert_eq!(report.failures.len(), 0);
        assert_eq!(report.skills[0].id(), "ai.example.greet");
    }

    #[test]
    fn one_bad_skill_does_not_sink_others() {
        let dir = tmpdir();
        write(
            &dir.path().join("broken/SKILL.md"),
            "---\nname: broken\ndescription: x\nmetadata:\n  ari:\n    id: a.b.c\n    version: \"1\"\n    engine: \">=0.3\"\n    matching:\n      patterns:\n        - keywords: []\n    declarative:\n      response: hi\n---\n",
        );
        write(&dir.path().join("greet/SKILL.md"), greet_md());

        let report = load_skill_directory(dir.path()).unwrap();
        assert_eq!(report.skills.len(), 1);
        assert_eq!(report.failures.len(), 1);
        assert!(matches!(
            report.failures[0].kind,
            LoadFailureKind::Manifest(ManifestError::EmptyKeywords)
        ));
    }

    #[test]
    fn wasm_skill_with_missing_module_file_fails_loudly() {
        // SKILL.md references skill.wasm but there's no such file on disk.
        // No capabilities declared, so the cap check passes and we reach the
        // file-read step, which is what we're testing.
        let dir = tmpdir();
        write(
            &dir.path().join("weather/SKILL.md"),
            r#"---
name: weather
description: Weather. Use when asked about weather.
metadata:
  ari:
    id: ai.example.weather
    version: "1"
    engine: ">=0.3"
    matching:
      patterns:
        - keywords: [weather]
    wasm:
      module: skill.wasm
---
"#,
        );

        let report = load_skill_directory(dir.path()).unwrap();
        assert_eq!(report.skills.len(), 0);
        assert_eq!(report.failures.len(), 1);
        assert!(matches!(
            report.failures[0].kind,
            LoadFailureKind::Wasm(WasmError::ReadModule { .. })
        ));
    }

    #[test]
    fn loader_rejects_skill_with_unprovided_capability() {
        let dir = tmpdir();
        write(
            &dir.path().join("weather/SKILL.md"),
            r#"---
name: weather
description: Needs http. Use when the user asks about weather.
metadata:
  ari:
    id: ai.example.weather
    version: "1"
    engine: ">=0.3"
    capabilities: [http, location]
    matching:
      patterns:
        - keywords: [weather]
    declarative:
      response: "I have no idea, ask a window."
---
"#,
        );

        // Default options grant only pure-frontend caps, not http/location.
        let report = load_skill_directory(dir.path()).unwrap();
        assert_eq!(report.skills.len(), 0);
        assert_eq!(report.failures.len(), 1);
        match &report.failures[0].kind {
            LoadFailureKind::MissingCapabilities { missing } => {
                assert_eq!(missing, &vec![Capability::Http, Capability::Location]);
            }
            other => panic!("expected MissingCapabilities, got {other:?}"),
        }
    }

    #[test]
    fn loader_loads_declarative_skill_with_pure_frontend_caps() {
        let dir = tmpdir();
        write(
            &dir.path().join("opener/SKILL.md"),
            r#"---
name: opener
description: Opens an app via Action response. Use when the user asks to open something.
metadata:
  ari:
    id: ai.example.opener
    version: "1"
    engine: ">=0.3"
    capabilities: [launch_app]
    matching:
      patterns:
        - keywords: [open, thing]
          weight: 0.9
    declarative:
      response: "Opening it."
      action:
        type: launch_app
        target: thing
---
"#,
        );

        // Default options include launch_app via pure_frontend()
        let report = load_skill_directory(dir.path()).unwrap();
        assert_eq!(report.failures.len(), 0, "{:?}", report.failures);
        assert_eq!(report.skills.len(), 1);
        assert_eq!(report.skills[0].id(), "ai.example.opener");
    }

    /// A WAT module that satisfies the ABI v1 contract:
    /// exports memory + ari_alloc + score + execute, returns "wasm hello"
    /// from execute() at offset 2048.
    fn echo_wat() -> &'static str {
        r#"(module
  (memory (export "memory") 1)
  (data (i32.const 2048) "wasm hello")
  (global $bump (mut i32) (i32.const 1024))
  (func (export "ari_alloc") (param $size i32) (result i32)
    (local $p i32)
    global.get $bump
    local.set $p
    global.get $bump
    local.get $size
    i32.add
    global.set $bump
    local.get $p)
  (func (export "score") (param i32 i32) (result f32)
    f32.const 0.95)
  (func (export "execute") (param i32 i32) (result i64)
    i64.const 2048
    i64.const 32
    i64.shl
    i64.const 10
    i64.or)
)"#
    }

    #[test]
    fn loads_real_wasm_skill_end_to_end() {
        use ari_core::{Response, SkillContext};

        let dir = tmpdir();
        let skill_dir = dir.path().join("echo");
        write(
            &skill_dir.join("SKILL.md"),
            r#"---
name: echo
description: Echo skill. Use to test the WASM loader.
metadata:
  ari:
    id: ai.example.echo
    version: "0.1.0"
    engine: ">=0.3"
    matching:
      patterns:
        - keywords: [echo]
          weight: 0.95
    wasm:
      module: skill.wasm
---
"#,
        );
        let wasm_bytes = wat::parse_str(echo_wat()).unwrap();
        std::fs::write(skill_dir.join("skill.wasm"), &wasm_bytes).unwrap();

        let report = load_skill_directory(dir.path()).unwrap();
        assert_eq!(report.failures.len(), 0, "{:?}", report.failures);
        assert_eq!(report.skills.len(), 1);

        let skill = &report.skills[0];
        assert_eq!(skill.id(), "ai.example.echo");

        let ctx = SkillContext::default();
        // Declarative score path: weight from manifest
        assert_eq!(skill.score("echo me", &ctx), 0.95);
        // Execute path: text comes from inside the wasm module's memory
        match skill.execute("echo me", &ctx) {
            Response::Text(t) => assert_eq!(t, "wasm hello"),
            _ => panic!("expected text response from wasm skill"),
        }
    }

    #[test]
    fn load_single_skill_dir_works_directly() {
        let dir = tmpdir();
        let skill_dir = dir.path().join("coin-flip");
        write(&skill_dir.join("SKILL.md"), coin_flip_md());

        let report = load_single_skill_dir(&skill_dir);
        assert_eq!(report.failures.len(), 0);
        assert_eq!(report.skills.len(), 1);
        assert_eq!(report.skills[0].id(), "ai.example.coinflip");
    }

    #[test]
    fn missing_skill_md_in_single_dir_is_a_failure() {
        let dir = tmpdir();
        fs::create_dir_all(dir.path().join("nope")).unwrap();
        let report = load_single_skill_dir(&dir.path().join("nope"));
        assert_eq!(report.skills.len(), 0);
        assert_eq!(report.failures.len(), 1);
        assert!(matches!(
            report.failures[0].kind,
            LoadFailureKind::MissingSkillfile
        ));
    }

    #[test]
    fn loaded_skill_actually_runs_through_score_and_execute() {
        use ari_core::{Response, SkillContext};

        let dir = tmpdir();
        write(&dir.path().join("coin-flip/SKILL.md"), coin_flip_md());

        let report = load_skill_directory(dir.path()).unwrap();
        let skill = &report.skills[0];
        let ctx = SkillContext::default();
        assert!(skill.score("flip a coin", &ctx) > 0.9);
        assert!(skill.score("what time is it", &ctx) == 0.0);
        match skill.execute("flip a coin", &ctx) {
            Response::Text(t) => assert!(t == "Heads." || t == "Tails."),
            _ => panic!(),
        }
    }
}

// --- Tiny test-only tempdir helper. ---
//
// We don't need a full tempfile crate dependency for unit tests; this
// minimal helper creates a uniquely-named directory under `std::env::temp_dir`
// and removes it on drop.
#[cfg(test)]
mod tempdir_lite {
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    pub struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        pub fn new(prefix: &str) -> Self {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!("{prefix}-{nanos}-{n}"));
            std::fs::create_dir_all(&path).expect("create temp dir");
            Self { path }
        }

        pub fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}
