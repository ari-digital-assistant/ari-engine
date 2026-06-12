//! Standalone validator for skill directories.
//!
//! Re-uses `ari-skill-loader` to do the work, so the same code path catches
//! the same problems CI and the engine itself would. Three output formats:
//!
//! - `text` (default) — human-friendly lines with a final tally.
//! - `pr-comment` — GitHub-flavoured markdown suitable for piping into
//!   `gh pr comment --body-file -`. Renders a table of (path, id, version,
//!   status) plus a details block per failure. Used by `validate.yml`.
//! - `json` — machine-readable array of `{ path, id, version, name,
//!   description, license, status, failures[] }`. Used by the publish
//!   workflow to drive `tools/build-index.sh`.
//!
//! Two invocation shapes:
//! - `ari-skill-validate <path>` where `<path>/SKILL.md` exists → validates
//!   that one skill.
//! - `ari-skill-validate <path>` where `<path>` contains `<slug>/SKILL.md`
//!   subdirectories → validates every skill in the registry-style root.
//!
//! Exit codes: 0 = all good, 1 = at least one skill failed, 2 = bad CLI usage.

use ari_skill_loader::{
    capability_name, load_single_skill_dir_with, load_skill_directory_with, HostCapabilities,
    LoadFailure, LoadOptions, LoadReport, Skillfile,
};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Format {
    Text,
    PrComment,
    Json,
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let mut paths: Vec<PathBuf> = Vec::new();
    let mut quiet = false;
    let mut format = Format::Text;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--quiet" | "-q" => quiet = true,
            "--help" | "-h" => {
                print_usage();
                return ExitCode::SUCCESS;
            }
            "--format" => match args.next().as_deref() {
                Some("text") => format = Format::Text,
                Some("pr-comment") => format = Format::PrComment,
                Some("json") => format = Format::Json,
                Some(other) => {
                    eprintln!("ari-skill-validate: unknown format: {other}");
                    return ExitCode::from(2);
                }
                None => {
                    eprintln!("ari-skill-validate: --format requires a value");
                    return ExitCode::from(2);
                }
            },
            o if o.starts_with("--format=") => match &o["--format=".len()..] {
                "text" => format = Format::Text,
                "pr-comment" => format = Format::PrComment,
                "json" => format = Format::Json,
                other => {
                    eprintln!("ari-skill-validate: unknown format: {other}");
                    return ExitCode::from(2);
                }
            },
            other if other.starts_with('-') => {
                eprintln!("ari-skill-validate: unknown option: {other}");
                print_usage();
                return ExitCode::from(2);
            }
            _ => paths.push(PathBuf::from(arg)),
        }
    }

    if paths.is_empty() {
        eprintln!("ari-skill-validate: at least one path is required");
        print_usage();
        return ExitCode::from(2);
    }

    // The validator intentionally grants every capability so that a skill
    // declaring `http` or `storage_kv` isn't rejected just because the
    // validator's host process doesn't ship those imports. Manifest
    // correctness is what we're checking here, not capability grants —
    // that's the engine's job at install time.
    let options = LoadOptions {
        host_capabilities: HostCapabilities::all(),
        ..LoadOptions::default()
    };

    // Collect rows across all input paths so pr-comment / json can render
    // a single combined report.
    let mut rows: Vec<Row> = Vec::new();

    for path in &paths {
        if !path.exists() {
            rows.push(Row::path_missing(path));
            continue;
        }

        // A directory is a single-skill dir when it has either the legacy
        // bare `SKILL.md` or the canonical `SKILL.en.md` (the localized-
        // manifest entry point — see ari_skill_loader::localized_manifest).
        // Otherwise treat it as a registry root and walk its children.
        if path.join("SKILL.md").is_file() || path.join("SKILL.en.md").is_file() {
            let report = load_single_skill_dir_with(path, &options);
            push_rows_from_report(&mut rows, path, &report);
        } else {
            // Walk the registry-style root ourselves so we can record the
            // per-skill path (load_skill_directory hides that).
            let entries = match std::fs::read_dir(path) {
                Ok(e) => e,
                Err(e) => {
                    rows.push(Row::dir_error(path, &format!("could not read dir: {e}")));
                    continue;
                }
            };
            let mut any_child = false;
            for entry in entries {
                let Ok(entry) = entry else { continue };
                let child = entry.path();
                if !child.is_dir() {
                    continue;
                }
                // Same dual-check as the single-skill branch — a child
                // qualifies if it has either the legacy bare manifest or
                // the canonical English locale manifest.
                if !child.join("SKILL.md").is_file()
                    && !child.join("SKILL.en.md").is_file()
                {
                    continue;
                }
                any_child = true;
                let report = load_single_skill_dir_with(&child, &options);
                push_rows_from_report(&mut rows, &child, &report);
            }
            if !any_child {
                // Could also be a flat root with its own SKILL.md — handled
                // above — so reaching here means genuinely empty.
                let report = match load_skill_directory_with(path, &options) {
                    Ok(r) => r,
                    Err(e) => {
                        rows.push(Row::dir_error(path, &format!("{e}")));
                        continue;
                    }
                };
                push_rows_from_report(&mut rows, path, &report);
            }
        }
    }

    let failures = rows.iter().filter(|r| !r.ok).count();
    let successes = rows.len() - failures;

    match format {
        Format::Text => render_text(&rows, successes, failures, quiet),
        Format::PrComment => render_pr_comment(&rows, successes, failures),
        Format::Json => render_json(&rows),
    }

    if failures > 0 {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}

#[derive(Debug)]
struct Row {
    path: PathBuf,
    ok: bool,
    id: Option<String>,
    version: Option<String>,
    name: Option<String>,
    description: Option<String>,
    license: Option<String>,
    author: Option<String>,
    homepage: Option<String>,
    capabilities: Vec<String>,
    languages: Vec<String>,
    examples: usize,
    /// Per-locale (name, description) pairs lifted from any
    /// `SKILL.{locale}.md` files alongside the canonical manifest. The
    /// publish pipeline writes these into `index.json`'s
    /// `localizations` object so browse-time consumers can render the
    /// right copy without downloading the bundle. English is omitted
    /// — `name` + `description` above already carry it.
    localizations: BTreeMap<String, LocalizedDisplay>,
    failures: Vec<String>,
    warnings: Vec<String>,
}

#[derive(Debug, Clone)]
struct LocalizedDisplay {
    name: String,
    description: String,
}

impl Row {
    fn path_missing(path: &Path) -> Self {
        Self {
            path: path.to_path_buf(),
            ok: false,
            id: None,
            version: None,
            name: None,
            description: None,
            license: None,
            author: None,
            homepage: None,
            capabilities: Vec::new(),
            languages: Vec::new(),
            examples: 0,
            localizations: BTreeMap::new(),
            failures: vec!["path does not exist".to_string()],
            warnings: Vec::new(),
        }
    }
    fn dir_error(path: &Path, msg: &str) -> Self {
        Self {
            path: path.to_path_buf(),
            ok: false,
            id: None,
            version: None,
            name: None,
            description: None,
            license: None,
            author: None,
            homepage: None,
            capabilities: Vec::new(),
            languages: Vec::new(),
            examples: 0,
            localizations: BTreeMap::new(),
            failures: vec![msg.to_string()],
            warnings: Vec::new(),
        }
    }
}

fn push_rows_from_report(out: &mut Vec<Row>, path: &Path, report: &LoadReport) {
    // A report here is one of:
    //   (a) one success (regular skill loaded)
    //   (a') one success (assistant skill loaded)
    //   (b) one failure (couldn't load)
    //   (c) nothing (valid AgentSkills doc with no metadata.ari — not an
    //       Ari skill, silently skipped by the loader)
    if let Some(skill) = report.skills.first() {
        let fields = read_manifest_fields(path);
        out.push(Row {
            path: path.to_path_buf(),
            ok: true,
            id: Some(skill.id().to_string()),
            version: fields.version,
            name: fields.name,
            description: fields.description,
            license: fields.license,
            author: fields.author,
            homepage: fields.homepage,
            capabilities: fields.capabilities,
            languages: fields.languages,
            examples: fields.examples,
            localizations: fields.localizations,
            failures: Vec::new(),
            warnings: fields.warnings,
        });
        return;
    }
    // Assistant skills don't enter `report.skills` — they go into
    // `report.assistants`. Treat them as valid if they parsed.
    if let Some(entry) = report.assistants.first() {
        let fields = read_manifest_fields(path);
        out.push(Row {
            path: path.to_path_buf(),
            ok: true,
            id: Some(entry.id.clone()),
            version: fields.version,
            name: Some(entry.name.clone()),
            description: Some(entry.description.clone()),
            license: fields.license,
            author: fields.author,
            homepage: fields.homepage,
            capabilities: fields.capabilities,
            languages: fields.languages,
            examples: fields.examples,
            localizations: fields.localizations,
            failures: Vec::new(),
            warnings: fields.warnings,
        });
        return;
    }
    if !report.failures.is_empty() {
        out.push(Row {
            path: path.to_path_buf(),
            ok: false,
            id: None,
            version: None,
            name: None,
            description: None,
            license: None,
            author: None,
            homepage: None,
            capabilities: Vec::new(),
            languages: Vec::new(),
            examples: 0,
            localizations: BTreeMap::new(),
            failures: report.failures.iter().map(LoadFailure::to_string).collect(),
            warnings: Vec::new(),
        });
        return;
    }
    out.push(Row {
        path: path.to_path_buf(),
        ok: false,
        id: None,
        version: None,
        name: None,
        description: None,
        license: None,
        author: None,
        homepage: None,
        capabilities: Vec::new(),
        languages: Vec::new(),
        examples: 0,
        localizations: BTreeMap::new(),
        failures: vec!["SKILL.md has no metadata.ari extension (not an Ari skill)".to_string()],
        warnings: Vec::new(),
    });
}

/// Re-parse the canonical-locale manifest to pull the descriptive
/// frontmatter fields for rows the loader accepted. The loader
/// returns `Box<dyn Skill>` which only exposes id/specificity — the
/// rest of the frontmatter isn't on the trait. Cheap to re-parse;
/// we've already loaded the file once.
#[derive(Default)]
struct ManifestFields {
    version: Option<String>,
    name: Option<String>,
    description: Option<String>,
    license: Option<String>,
    author: Option<String>,
    homepage: Option<String>,
    capabilities: Vec<String>,
    languages: Vec<String>,
    examples: usize,
    /// Per-locale display strings lifted from `SKILL.{locale}.md`
    /// files alongside the canonical English manifest. Keyed by ISO
    /// 639-1 lowercase code; English is excluded because the
    /// canonical `name` + `description` fields above already cover it.
    /// Empty for skills using the legacy single-file (`SKILL.md`)
    /// layout — they have nothing to localise.
    localizations: BTreeMap<String, LocalizedDisplay>,
    warnings: Vec<String>,
}

fn read_manifest_fields(skill_dir: &Path) -> ManifestFields {
    // Prefer `SKILL.en.md` (the per-locale layout) and fall back to
    // the legacy `SKILL.md` so skills that haven't migrated yet still
    // get their fields read correctly. Without this fallback, skills
    // on the new layout returned `ManifestFields::default()` and the
    // publish-index pipeline silently dropped them ("skill at X has
    // no id/version — skipping" with no version to report).
    let candidates = [skill_dir.join("SKILL.en.md"), skill_dir.join("SKILL.md")];
    let path = candidates.iter().find(|p| p.is_file());
    let Some(path) = path else {
        return ManifestFields::default();
    };
    let Ok(sf) = Skillfile::parse_file(path) else {
        return ManifestFields::default();
    };
    let mut out = ManifestFields {
        name: Some(sf.name),
        description: Some(sf.description),
        license: sf.license,
        localizations: read_localizations(skill_dir),
        ..ManifestFields::default()
    };
    if let Some(ext) = sf.ari_extension {
        let resp_strings = ext
            .behaviour
            .as_ref()
            .map(declarative_response_strings)
            .unwrap_or_default();
        let en_keys = en_string_keys(skill_dir);
        for s in resp_strings {
            if looks_like_strings_key(&s) && !en_keys.contains(&s) {
                out.warnings.push(format!(
                    "declarative response \"{s}\" looks like a strings key but is not in strings/en.json — it will render verbatim"
                ));
            }
        }
        out.examples = ext.examples.len();
        if let Err(e) = ext.validate_examples() {
            out.warnings.push(e.to_string());
        }
        out.version = Some(ext.version);
        out.author = ext.author;
        out.homepage = ext.homepage;
        out.capabilities = ext
            .capabilities
            .into_iter()
            .map(|c| capability_name(c).to_string())
            .collect();
        out.languages = ext.languages;
    }
    out
}

/// Walk `skill_dir` looking for non-canonical `SKILL.{locale}.md`
/// variants and pull `(name, description)` out of each. The canonical
/// English manifest is intentionally excluded — its values live on the
/// top-level `name` + `description` fields and would duplicate.
///
/// Filenames must be exactly `SKILL.{locale}.md` where `{locale}` is a
/// 2-character lowercase ASCII code (matching `localized_manifest.rs`'s
/// rule). Anything else (`SKILL.md`, `README.md`, `SKILL.it.draft.md`)
/// is silently ignored — same forgiving stance as the loader.
fn read_localizations(skill_dir: &Path) -> BTreeMap<String, LocalizedDisplay> {
    let mut out = BTreeMap::new();
    let Ok(entries) = std::fs::read_dir(skill_dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(filename) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        // Match SKILL.<locale>.md with a 2-char lowercase locale.
        // Hand-rolled rather than using the loader's `parse_locale_filename`
        // because that helper isn't exported from ari-skill-loader; the
        // shape is small and the constraints stable.
        let Some(rest) = filename.strip_prefix("SKILL.") else {
            continue;
        };
        let Some(locale) = rest.strip_suffix(".md") else {
            continue;
        };
        if locale.len() != 2 || !locale.chars().all(|c| c.is_ascii_lowercase()) {
            continue;
        }
        // English is canonical; its strings already live on the top-
        // level `name` + `description` and shouldn't be duplicated into
        // the per-locale map. Skip it here.
        if locale == "en" {
            continue;
        }
        let Ok(sf) = Skillfile::parse_file(&path) else {
            // A failing per-locale parse is a warning-class issue — the
            // caller's `read_manifest_fields` doesn't surface it because
            // the cross-file consistency check inside the loader already
            // would have refused the skill earlier if the variant was
            // structurally bad. If we're here, the file probably has a
            // bad frontmatter that the loader rejected entry-wide.
            continue;
        };
        out.insert(
            locale.to_string(),
            LocalizedDisplay {
                name: sf.name,
                description: sf.description,
            },
        );
    }
    out
}

fn render_text(rows: &[Row], ok: usize, failed: usize, quiet: bool) {
    for row in rows {
        if row.ok {
            if !quiet {
                let id = row.id.as_deref().unwrap_or("?");
                println!("✓ {}: {} ({} examples)", row.path.display(), id, row.examples);
                for w in &row.warnings {
                    eprintln!("  ⚠ {}: {}", row.path.display(), w);
                }
            }
        } else {
            for f in &row.failures {
                eprintln!("✗ {}: {}", row.path.display(), f);
            }
        }
    }
    if !quiet {
        eprintln!();
        eprintln!("validated {ok} skill(s), {failed} failure(s)");
    }
}

fn render_pr_comment(rows: &[Row], ok: usize, failed: usize) {
    let header_emoji = if failed == 0 { "✅" } else { "❌" };
    println!("## {header_emoji} ari-skill-validate");
    println!();
    println!(
        "**{ok}** skill(s) validated, **{failed}** failure(s).",
    );
    println!();
    println!("| Status | Path | ID | Version | Examples |");
    println!("| --- | --- | --- | --- | --- |");
    for row in rows {
        let status = if row.ok { "✅" } else { "❌" };
        let id = row.id.as_deref().unwrap_or("—");
        let version = row.version.as_deref().unwrap_or("—");
        println!(
            "| {} | `{}` | `{}` | `{}` | {} |",
            status,
            row.path.display(),
            escape_pipe(id),
            escape_pipe(version),
            row.examples,
        );
    }
    let failing: Vec<&Row> = rows.iter().filter(|r| !r.ok).collect();
    if !failing.is_empty() {
        println!();
        println!("### Failures");
        println!();
        for row in failing {
            println!("- **`{}`**", row.path.display());
            for f in &row.failures {
                println!("  - {}", escape_markdown(f));
            }
        }
    }
    let warned: Vec<&Row> = rows.iter().filter(|r| !r.warnings.is_empty()).collect();
    if !warned.is_empty() {
        println!();
        println!("### Warnings");
        println!();
        for row in warned {
            let id = row.id.as_deref().unwrap_or("?");
            for w in &row.warnings {
                println!("- ⚠️ **`{id}`**: {}", escape_markdown(w));
            }
        }
    }
    println!();
    println!("<sub>Generated by `ari-skill-validate --format=pr-comment`.</sub>");
}

fn render_json(rows: &[Row]) {
    // Hand-rolled JSON to keep the validator crate dependency-free (it only
    // pulls in ari-skill-loader, nothing else). The shape is small and
    // fixed, so serde here would be overkill.
    let mut out = String::from("[\n");
    for (i, row) in rows.iter().enumerate() {
        if i > 0 {
            out.push_str(",\n");
        }
        out.push_str("  {\n");
        push_json_kv(&mut out, "path", &row.path.display().to_string(), true);
        push_json_bool(&mut out, "ok", row.ok, true);
        push_json_opt(&mut out, "id", row.id.as_deref(), true);
        push_json_opt(&mut out, "version", row.version.as_deref(), true);
        push_json_opt(&mut out, "name", row.name.as_deref(), true);
        push_json_opt(&mut out, "description", row.description.as_deref(), true);
        push_json_opt(&mut out, "license", row.license.as_deref(), true);
        push_json_opt(&mut out, "author", row.author.as_deref(), true);
        push_json_opt(&mut out, "homepage", row.homepage.as_deref(), true);
        push_json_str_array(&mut out, "capabilities", &row.capabilities, true);
        push_json_str_array(&mut out, "languages", &row.languages, true);
        push_json_kv(&mut out, "examples", &row.examples.to_string(), true);
        push_json_str_array(&mut out, "warnings", &row.warnings, true);
        // Per-locale display strings — `{ "it": { "name": "...", "description": "..." } }`.
        // BTreeMap iter is alphabetical-by-key, which keeps the JSON
        // output stable across runs and easier to diff in CI.
        out.push_str("    \"localizations\": {");
        for (j, (locale, display)) in row.localizations.iter().enumerate() {
            if j > 0 {
                out.push(',');
            }
            out.push(' ');
            out.push_str(&json_string(locale));
            out.push_str(": {\"name\": ");
            out.push_str(&json_string(&display.name));
            out.push_str(", \"description\": ");
            out.push_str(&json_string(&display.description));
            out.push('}');
        }
        if !row.localizations.is_empty() {
            out.push(' ');
        }
        out.push_str("},\n");
        out.push_str("    \"failures\": [");
        for (j, f) in row.failures.iter().enumerate() {
            if j > 0 {
                out.push_str(", ");
            }
            out.push_str(&json_string(f));
        }
        out.push_str("]\n");
        out.push_str("  }");
    }
    out.push_str("\n]\n");
    print!("{out}");
}

fn push_json_kv(out: &mut String, key: &str, value: &str, trailing_comma: bool) {
    out.push_str("    ");
    out.push_str(&json_string(key));
    out.push_str(": ");
    out.push_str(&json_string(value));
    if trailing_comma {
        out.push(',');
    }
    out.push('\n');
}

fn push_json_opt(out: &mut String, key: &str, value: Option<&str>, trailing_comma: bool) {
    out.push_str("    ");
    out.push_str(&json_string(key));
    out.push_str(": ");
    match value {
        Some(v) => out.push_str(&json_string(v)),
        None => out.push_str("null"),
    }
    if trailing_comma {
        out.push(',');
    }
    out.push('\n');
}

fn push_json_str_array(out: &mut String, key: &str, values: &[String], trailing_comma: bool) {
    out.push_str("    \"");
    out.push_str(key);
    out.push_str("\": [");
    for (i, v) in values.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&json_string(v));
    }
    out.push(']');
    if trailing_comma {
        out.push(',');
    }
    out.push('\n');
}

fn push_json_bool(out: &mut String, key: &str, value: bool, trailing_comma: bool) {
    out.push_str("    ");
    out.push_str(&json_string(key));
    out.push_str(": ");
    out.push_str(if value { "true" } else { "false" });
    if trailing_comma {
        out.push(',');
    }
    out.push('\n');
}

fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn escape_pipe(s: &str) -> String {
    s.replace('|', "\\|")
}

fn escape_markdown(s: &str) -> String {
    s.replace('|', "\\|").replace('\n', " ")
}

/// True when a string looks like a strings-table key: dotted, lowercase
/// ASCII / digits / underscores, no whitespace, no empty segments (e.g.
/// `coinflip.heads`). Deliberately excludes real literals like "Heads."
/// (uppercase) or "No timers." (space).
fn looks_like_strings_key(s: &str) -> bool {
    s.contains('.')
        && !s.is_empty()
        && s.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '.')
        && s.split('.').all(|seg| !seg.is_empty())
}

/// The keys defined in `strings/en.json`, or empty if absent/unparseable.
fn en_string_keys(skill_dir: &std::path::Path) -> std::collections::HashSet<String> {
    let path = skill_dir.join("strings").join("en.json");
    let Ok(src) = std::fs::read_to_string(&path) else {
        return std::collections::HashSet::new();
    };
    match serde_json::from_str::<std::collections::BTreeMap<String, String>>(&src) {
        Ok(map) => map.into_keys().collect(),
        Err(_) => std::collections::HashSet::new(),
    }
}

/// Response strings of a declarative behaviour (for key-warning scanning).
fn declarative_response_strings(behaviour: &ari_skill_loader::Behaviour) -> Vec<String> {
    use ari_skill_loader::{Behaviour, ResponseSpec};
    match behaviour {
        Behaviour::Declarative(d) => match &d.response {
            ResponseSpec::Fixed(s) => vec![s.clone()],
            ResponseSpec::Pick(v) => v.clone(),
            ResponseSpec::Template(s) => vec![s.clone()],
        },
        Behaviour::Wasm(_) => Vec::new(),
    }
}

fn print_usage() {
    eprintln!("usage: ari-skill-validate [--quiet] [--format text|pr-comment|json] <path>...");
    eprintln!();
    eprintln!("  <path>              a single skill directory (containing SKILL.md), or a");
    eprintln!("                      registry root containing one subdirectory per skill.");
    eprintln!("                      may be repeated.");
    eprintln!("  --quiet             (text format only) suppress success output");
    eprintln!("  --format text       human-friendly default output");
    eprintln!("  --format pr-comment GitHub-flavoured markdown for `gh pr comment --body-file -`");
    eprintln!("  --format json       machine-readable rows for downstream tooling");
    eprintln!();
    eprintln!("exit codes: 0 ok, 1 validation failure, 2 bad usage");
}

// --- Tiny test-only tempdir helper ---
//
// Mirrors the pattern used in ari-skill-loader tests; avoids adding a
// tempfile crate dependency.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warns_on_dotted_response_string_absent_from_en_json() {
        use std::fs;
        let dir = tempdir_lite::TempDir::new("ari-validate-test");
        let skill = dir.path().join("coin");
        fs::create_dir_all(skill.join("strings")).unwrap();
        let md = r#"---
name: coin
description: Flips a coin.
metadata:
  ari:
    id: ai.example.coin
    version: "0.1.0"
    engine: ">=0.3"
    languages: [en]
    matching:
      patterns:
        - keywords: [flip, coin]
          weight: 0.95
    examples:
      - text: "flip a coin"
      - text: "flip coin"
      - text: "toss a coin"
      - text: "coin flip"
      - text: "heads or tails"
    declarative:
      response_pick: ["coin.heads", "coin.tals"]
---
"#;
        fs::write(skill.join("SKILL.en.md"), md).unwrap();
        fs::write(skill.join("strings/en.json"), r#"{"coin.heads":"Heads."}"#).unwrap();

        let fields = read_manifest_fields(&skill);
        assert!(
            fields.warnings.iter().any(|w| w.contains("coin.tals")),
            "expected a warning naming the unresolved key, got {:?}",
            fields.warnings
        );
        assert!(
            !fields.warnings.iter().any(|w| w.contains("coin.heads")),
            "coin.heads is present in en.json — must not warn"
        );
    }

    #[test]
    fn does_not_warn_on_literal_response() {
        use std::fs;
        let dir = tempdir_lite::TempDir::new("ari-validate-test");
        let skill = dir.path().join("coin");
        fs::create_dir_all(&skill).unwrap();
        let md = r#"---
name: coin
description: Flips a coin.
metadata:
  ari:
    id: ai.example.coin
    version: "0.1.0"
    engine: ">=0.3"
    matching:
      patterns:
        - keywords: [flip, coin]
          weight: 0.95
    examples:
      - text: "flip a coin"
      - text: "flip coin"
      - text: "toss a coin"
      - text: "coin flip"
      - text: "heads or tails"
    declarative:
      response_pick: ["Heads.", "Tails."]
---
"#;
        fs::write(skill.join("SKILL.en.md"), md).unwrap();
        let fields = read_manifest_fields(&skill);
        assert!(
            fields.warnings.is_empty(),
            "literal responses must not warn, got {:?}",
            fields.warnings
        );
    }

    #[test]
    fn looks_like_strings_key_accepts_dotted_lowercase() {
        assert!(looks_like_strings_key("coin.heads"));
        assert!(looks_like_strings_key("coinflip.result.heads"));
        assert!(looks_like_strings_key("a.b"));
    }

    #[test]
    fn looks_like_strings_key_rejects_non_keys() {
        assert!(!looks_like_strings_key("Heads."));     // uppercase
        assert!(!looks_like_strings_key("No timers.")); // space
        assert!(!looks_like_strings_key("simple"));     // no dot
        assert!(!looks_like_strings_key("a..b"));       // empty segment
        assert!(!looks_like_strings_key(".leading"));   // leading dot → empty seg
        assert!(!looks_like_strings_key("trailing."));  // trailing dot → empty seg
    }

    #[test]
    fn json_string_escapes_special_chars() {
        assert_eq!(json_string("hi"), "\"hi\"");
        assert_eq!(json_string("a\"b"), "\"a\\\"b\"");
        assert_eq!(json_string("a\\b"), "\"a\\\\b\"");
        assert_eq!(json_string("line\nbreak"), "\"line\\nbreak\"");
        assert_eq!(json_string("\x01"), "\"\\u0001\"");
    }

    #[test]
    fn escape_pipe_doubles_up_pipes_for_markdown_tables() {
        assert_eq!(escape_pipe("a|b"), "a\\|b");
        assert_eq!(escape_pipe("plain"), "plain");
    }
}
