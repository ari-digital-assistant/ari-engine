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

/// Platform directory names allowed under `screenshots/`. A typo here is
/// worse than useless — the shots would publish and then never be shown
/// to anybody — so an unrecognised directory fails the skill outright.
const SCREENSHOT_PLATFORMS: &[&str] = &["android", "ios", "linux", "macos", "windows"];

/// File extensions allowed for screenshots. Lowercase only, checked
/// literally: `Photo.PNG` straight off a desktop fails rather than
/// silently publishing a path some case-sensitive host will 404 on.
const SCREENSHOT_EXTENSIONS: &[&str] = &[".png", ".webp", ".jpg"];

/// Per-file size ceiling. A phone screenshot saved as WebP lands well
/// under this; a raw PNG straight off a 3x display does not, and every
/// browse-time viewer would pay for it.
const MAX_SCREENSHOT_BYTES: u64 = 1024 * 1024;

/// Per-platform count ceiling. Six is already more than anybody scrolls
/// through on a skill detail page.
const MAX_SCREENSHOTS_PER_PLATFORM: usize = 6;

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
    /// "skill" (default) or "assistant", from metadata.ari.type. Lets the
    /// registry index carry a real type so clients can filter without
    /// substring-matching the id/name/description. Emitted as the JSON key
    /// `type` by `render_json` (this crate hand-rolls JSON — no serde derive).
    skill_type: String,
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
    /// Preview screenshots discovered under `screenshots/<platform>/`,
    /// keyed by platform and ordered by filename. Paths are relative to
    /// the skill directory; `tools/build-index.sh` rewrites them to
    /// registry-relative ones when it copies the files out. Emitted as
    /// the JSON key `screenshots`.
    screenshots: BTreeMap<String, Vec<String>>,
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
            skill_type: "skill".to_string(),
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
            screenshots: BTreeMap::new(),
            failures: vec!["path does not exist".to_string()],
            warnings: Vec::new(),
        }
    }
    fn dir_error(path: &Path, msg: &str) -> Self {
        Self {
            path: path.to_path_buf(),
            ok: false,
            id: None,
            skill_type: "skill".to_string(),
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
            screenshots: BTreeMap::new(),
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
        let shots = read_screenshots(path);
        out.push(Row {
            path: path.to_path_buf(),
            ok: shots.failures.is_empty(),
            id: Some(skill.id().to_string()),
            skill_type: "skill".to_string(),
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
            screenshots: shots.by_platform,
            failures: shots.failures,
            warnings: fields.warnings,
        });
        return;
    }
    // Assistant skills don't enter `report.skills` — they go into
    // `report.assistants`. Treat them as valid if they parsed.
    if let Some(entry) = report.assistants.first() {
        let fields = read_manifest_fields(path);
        let shots = read_screenshots(path);
        out.push(Row {
            path: path.to_path_buf(),
            ok: shots.failures.is_empty(),
            id: Some(entry.id.clone()),
            skill_type: "assistant".to_string(),
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
            screenshots: shots.by_platform,
            failures: shots.failures,
            warnings: fields.warnings,
        });
        return;
    }
    if !report.failures.is_empty() {
        out.push(Row {
            path: path.to_path_buf(),
            ok: false,
            id: None,
            skill_type: "skill".to_string(),
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
            screenshots: BTreeMap::new(),
            failures: report.failures.iter().map(LoadFailure::to_string).collect(),
            warnings: Vec::new(),
        });
        return;
    }
    out.push(Row {
        path: path.to_path_buf(),
        ok: false,
        id: None,
        skill_type: "skill".to_string(),
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
        screenshots: BTreeMap::new(),
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

/// Outcome of walking a skill's `screenshots/` directory.
#[derive(Default)]
struct Screenshots {
    /// Platform → filenames-relative-to-the-skill-dir, filename-ordered.
    by_platform: BTreeMap<String, Vec<String>>,
    /// Anything that must block the publish. Screenshots are cosmetic,
    /// but a broken one is invisible until a user hits the detail page,
    /// which is far too late to find out — so these fail the skill
    /// rather than warn.
    failures: Vec<String>,
}

/// Walk `skill_dir/screenshots/` and collect the preview images.
///
/// Layout is convention, not manifest: one directory per platform, images
/// inside it, shown in filename order. That keeps ordering obvious from
/// `ls` and means adding a screenshot never touches the frontmatter.
/// There are no captions on purpose — captions would need translating for
/// every locale the skill ships, and a screenshot that only makes sense
/// with a caption is a bad screenshot.
fn read_screenshots(skill_dir: &Path) -> Screenshots {
    let mut out = Screenshots::default();
    let root = skill_dir.join("screenshots");
    if !root.exists() {
        return out;
    }
    if !root.is_dir() {
        out.failures
            .push("screenshots must be a directory of per-platform directories".to_string());
        return out;
    }
    let entries = match std::fs::read_dir(&root) {
        Ok(e) => e,
        Err(e) => {
            out.failures.push(format!("could not read screenshots/: {e}"));
            return out;
        }
    };
    // Sort the platform dirs before walking them so failure messages come
    // out in a stable order across runs and filesystems.
    let mut platform_dirs: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
    platform_dirs.sort();
    for dir in platform_dirs {
        let Some(name) = dir.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !dir.is_dir() {
            out.failures.push(format!(
                "screenshots/{name} is not inside a platform directory — put it in screenshots/<platform>/"
            ));
            continue;
        }
        if !SCREENSHOT_PLATFORMS.contains(&name) {
            out.failures.push(format!(
                "screenshots/{name}/ is not a known platform — expected one of {}",
                SCREENSHOT_PLATFORMS.join(", ")
            ));
            continue;
        }
        let files = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            Err(e) => {
                out.failures
                    .push(format!("could not read screenshots/{name}/: {e}"));
                continue;
            }
        };
        let mut paths: Vec<PathBuf> = files.flatten().map(|e| e.path()).collect();
        paths.sort();
        let mut kept: Vec<String> = Vec::new();
        for path in paths {
            let Some(filename) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            if !path.is_file() {
                out.failures.push(format!(
                    "screenshots/{name}/{filename} is not a file — platform directories hold images, nothing else"
                ));
                continue;
            }
            if !SCREENSHOT_EXTENSIONS.iter().any(|ext| filename.ends_with(ext)) {
                out.failures.push(format!(
                    "screenshots/{name}/{filename} is not a supported image — use {}",
                    SCREENSHOT_EXTENSIONS.join(", ")
                ));
                continue;
            }
            match path.metadata() {
                Ok(meta) if meta.len() > MAX_SCREENSHOT_BYTES => {
                    out.failures.push(format!(
                        "screenshots/{name}/{filename} is {} KiB — the limit is {} KiB, so compress it or save it as WebP",
                        meta.len() / 1024,
                        MAX_SCREENSHOT_BYTES / 1024
                    ));
                    continue;
                }
                Ok(_) => {}
                Err(e) => {
                    out.failures
                        .push(format!("could not stat screenshots/{name}/{filename}: {e}"));
                    continue;
                }
            }
            kept.push(format!("screenshots/{name}/{filename}"));
        }
        if kept.len() > MAX_SCREENSHOTS_PER_PLATFORM {
            out.failures.push(format!(
                "screenshots/{name}/ has {} images — the limit is {MAX_SCREENSHOTS_PER_PLATFORM}",
                kept.len()
            ));
            continue;
        }
        if !kept.is_empty() {
            out.by_platform.insert(name.to_string(), kept);
        }
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
    print!("{}", json_for_rows(rows));
}

/// Build the machine-readable JSON array for `rows`. Split out from
/// `render_json` so tests can assert the exact emitted shape without
/// capturing stdout.
fn json_for_rows(rows: &[Row]) -> String {
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
        push_json_kv(&mut out, "type", &row.skill_type, true);
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
        // Preview screenshots — `{ "android": ["screenshots/android/01.webp"] }`,
        // paths relative to the skill directory. `tools/build-index.sh`
        // rewrites them as it copies the files into the registry's
        // screenshots/ sidecar.
        out.push_str("    \"screenshots\": {");
        for (j, (platform, files)) in row.screenshots.iter().enumerate() {
            if j > 0 {
                out.push(',');
            }
            out.push(' ');
            out.push_str(&json_string(platform));
            out.push_str(": [");
            for (k, file) in files.iter().enumerate() {
                if k > 0 {
                    out.push_str(", ");
                }
                out.push_str(&json_string(file));
            }
            out.push(']');
        }
        if !row.screenshots.is_empty() {
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
    out
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

    /// Validate a written-out skill dir end-to-end (loader → rows) and
    /// return the single resulting row. Mirrors `main`'s single-skill path.
    fn row_for_skill_md(slug: &str, md: &str) -> Row {
        use std::fs;
        let dir = tempdir_lite::TempDir::new("ari-validate-type-test");
        let skill = dir.path().join(slug);
        fs::create_dir_all(&skill).unwrap();
        fs::write(skill.join("SKILL.en.md"), md).unwrap();

        let options = LoadOptions {
            host_capabilities: HostCapabilities::all(),
            ..LoadOptions::default()
        };
        let report = load_single_skill_dir_with(&skill, &options);
        let mut rows: Vec<Row> = Vec::new();
        push_rows_from_report(&mut rows, &skill, &report);
        assert_eq!(rows.len(), 1, "expected exactly one row, got {rows:?}");
        rows.into_iter().next().unwrap()
    }

    /// Write a screenshot of `bytes` length at `screenshots/<rel>` under a
    /// fresh skill dir, returning the dir so the caller can add more.
    fn skill_dir_with_screenshots(files: &[(&str, usize)]) -> (tempdir_lite::TempDir, PathBuf) {
        use std::fs;
        let dir = tempdir_lite::TempDir::new("ari-validate-shots");
        let skill = dir.path().join("timer");
        fs::create_dir_all(&skill).unwrap();
        for (rel, len) in files {
            let path = skill.join("screenshots").join(rel);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(&path, vec![0u8; *len]).unwrap();
        }
        (dir, skill)
    }

    #[test]
    fn screenshots_are_grouped_by_platform_in_filename_order() {
        let (_tmp, skill) = skill_dir_with_screenshots(&[
            ("android/02-list.webp", 10),
            ("android/01-set.webp", 10),
            ("linux/01-set.png", 10),
        ]);
        let shots = read_screenshots(&skill);
        assert!(shots.failures.is_empty(), "unexpected: {:?}", shots.failures);
        assert_eq!(
            shots.by_platform.get("android").unwrap(),
            &vec![
                "screenshots/android/01-set.webp".to_string(),
                "screenshots/android/02-list.webp".to_string(),
            ]
        );
        assert_eq!(
            shots.by_platform.get("linux").unwrap(),
            &vec!["screenshots/linux/01-set.png".to_string()]
        );
    }

    #[test]
    fn no_screenshots_directory_is_not_a_problem() {
        let (_tmp, skill) = skill_dir_with_screenshots(&[]);
        let shots = read_screenshots(&skill);
        assert!(shots.failures.is_empty());
        assert!(shots.by_platform.is_empty());
    }

    #[test]
    fn unknown_platform_directory_fails() {
        let (_tmp, skill) = skill_dir_with_screenshots(&[("andriod/01.webp", 10)]);
        let shots = read_screenshots(&skill);
        assert_eq!(shots.failures.len(), 1, "got {:?}", shots.failures);
        assert!(
            shots.failures[0].contains("screenshots/andriod/ is not a known platform"),
            "message must name the offending directory, got {:?}",
            shots.failures[0]
        );
        assert!(shots.by_platform.is_empty());
    }

    #[test]
    fn unsupported_extension_fails() {
        let (_tmp, skill) = skill_dir_with_screenshots(&[
            ("android/01.gif", 10),
            ("android/02.PNG", 10),
            ("android/03.webp", 10),
        ]);
        let shots = read_screenshots(&skill);
        assert_eq!(shots.failures.len(), 2, "got {:?}", shots.failures);
        assert!(shots.failures[0].contains("screenshots/android/01.gif"));
        assert!(shots.failures[1].contains("screenshots/android/02.PNG"));
        // The good one still comes through — one bad file doesn't erase the rest.
        assert_eq!(
            shots.by_platform.get("android").unwrap(),
            &vec!["screenshots/android/03.webp".to_string()]
        );
    }

    #[test]
    fn oversized_screenshot_fails() {
        let over = (MAX_SCREENSHOT_BYTES + 1) as usize;
        let (_tmp, skill) = skill_dir_with_screenshots(&[("android/01.png", over)]);
        let shots = read_screenshots(&skill);
        assert_eq!(shots.failures.len(), 1, "got {:?}", shots.failures);
        assert!(
            shots.failures[0].contains("1024 KiB"),
            "message must state the limit, got {:?}",
            shots.failures[0]
        );
        assert!(shots.by_platform.is_empty());
    }

    #[test]
    fn too_many_screenshots_for_one_platform_fails() {
        let files: Vec<(String, usize)> = (0..=MAX_SCREENSHOTS_PER_PLATFORM)
            .map(|i| (format!("android/{i:02}.webp"), 10))
            .collect();
        let refs: Vec<(&str, usize)> = files.iter().map(|(p, n)| (p.as_str(), *n)).collect();
        let (_tmp, skill) = skill_dir_with_screenshots(&refs);
        let shots = read_screenshots(&skill);
        assert_eq!(shots.failures.len(), 1, "got {:?}", shots.failures);
        assert!(shots.failures[0].contains("has 7 images"));
        assert!(shots.by_platform.is_empty());
    }

    #[test]
    fn loose_file_at_the_screenshots_root_fails() {
        use std::fs;
        let (_tmp, skill) = skill_dir_with_screenshots(&[("android/01.webp", 10)]);
        fs::write(skill.join("screenshots/hero.png"), vec![0u8; 10]).unwrap();
        let shots = read_screenshots(&skill);
        assert_eq!(shots.failures.len(), 1, "got {:?}", shots.failures);
        assert!(shots.failures[0].contains("screenshots/hero.png is not inside a platform directory"));
    }

    #[test]
    fn bad_screenshots_fail_the_whole_skill_and_good_ones_reach_the_json() {
        use std::fs;
        let md = r#"---
name: greet
description: Greets the user.
metadata:
  ari:
    id: ai.example.greet
    version: "0.1.0"
    engine: ">=0.3"
    languages: [en]
    matching:
      patterns:
        - keywords: [hello, hi]
          weight: 0.95
    examples:
      - text: "hello"
      - text: "hi there"
      - text: "good morning"
      - text: "hey"
      - text: "greetings"
    declarative:
      response: "Hello!"
---
Greet skill.
"#;
        let dir = tempdir_lite::TempDir::new("ari-validate-shots-row");
        let skill = dir.path().join("greet");
        fs::create_dir_all(skill.join("screenshots/android")).unwrap();
        fs::write(skill.join("SKILL.en.md"), md).unwrap();
        fs::write(skill.join("screenshots/android/01.webp"), vec![0u8; 10]).unwrap();

        let options = LoadOptions {
            host_capabilities: HostCapabilities::all(),
            ..LoadOptions::default()
        };
        let mut rows: Vec<Row> = Vec::new();
        push_rows_from_report(
            &mut rows,
            &skill,
            &load_single_skill_dir_with(&skill, &options),
        );
        assert!(rows[0].ok, "clean screenshots must not fail the skill");
        let json = json_for_rows(&rows);
        assert!(
            json.contains("\"screenshots\": { \"android\": [\"screenshots/android/01.webp\"] }"),
            "JSON must carry the discovered screenshots, got:\n{json}"
        );

        // Now break it: an unknown platform dir takes the whole skill down,
        // so a typo can never reach the registry unnoticed.
        fs::create_dir_all(skill.join("screenshots/nokia")).unwrap();
        fs::write(skill.join("screenshots/nokia/01.webp"), vec![0u8; 10]).unwrap();
        let mut rows: Vec<Row> = Vec::new();
        push_rows_from_report(
            &mut rows,
            &skill,
            &load_single_skill_dir_with(&skill, &options),
        );
        assert!(!rows[0].ok, "an unknown platform dir must fail the skill");
    }

    #[test]
    fn emits_type_skill_for_regular_manifest() {
        let md = r#"---
name: greet
description: Greets the user.
metadata:
  ari:
    id: ai.example.greet
    version: "0.1.0"
    engine: ">=0.3"
    languages: [en]
    matching:
      patterns:
        - keywords: [hello, hi]
          weight: 0.95
    examples:
      - text: "hello"
      - text: "hi there"
      - text: "good morning"
      - text: "hey"
      - text: "greetings"
    declarative:
      response: "Hello!"
---
Greet skill.
"#;
        let row = row_for_skill_md("greet", md);
        assert!(row.ok, "regular skill must load ok, failures: {:?}", row.failures);
        assert_eq!(row.id.as_deref(), Some("ai.example.greet"));
        assert_eq!(row.skill_type, "skill");

        let json = json_for_rows(std::slice::from_ref(&row));
        assert!(
            json.contains("\"type\": \"skill\""),
            "JSON must carry the skill type, got:\n{json}"
        );
    }

    #[test]
    fn emits_type_assistant_for_assistant_manifest() {
        // Mirrors ari-skill-loader's `chatgpt_assistant_md` fixture: a
        // `metadata.ari.type: assistant` manifest with the required
        // `assistant:` block. Such manifests land in `report.assistants`.
        let md = r#"---
name: chatgpt
description: Use OpenAI's ChatGPT to answer general questions.
metadata:
  ari:
    id: dev.heyari.assistant.chatgpt
    version: "0.1.0"
    type: assistant
    engine: ">=0.3"
    assistant:
      provider: api
      privacy: cloud
      api:
        endpoint: https://api.openai.com/v1/chat/completions
        auth: bearer
        auth_config_key: api_key
        default_model: gpt-4o-mini
        system_prompt: You are Ari.
        response_path: "choices[0].message.content"
      config:
        - key: api_key
          label: API Key
          type: secret
          required: true
---
ChatGPT assistant.
"#;
        let row = row_for_skill_md("chatgpt", md);
        assert!(row.ok, "assistant must load ok, failures: {:?}", row.failures);
        assert_eq!(row.id.as_deref(), Some("dev.heyari.assistant.chatgpt"));
        assert_eq!(row.skill_type, "assistant");

        let json = json_for_rows(std::slice::from_ref(&row));
        assert!(
            json.contains("\"type\": \"assistant\""),
            "JSON must carry the assistant type, got:\n{json}"
        );
    }
}
