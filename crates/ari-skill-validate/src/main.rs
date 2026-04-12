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

        if path.join("SKILL.md").is_file() {
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
                if !child.join("SKILL.md").is_file() {
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
    failures: Vec<String>,
    warnings: Vec<String>,
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
        failures: vec!["SKILL.md has no metadata.ari extension (not an Ari skill)".to_string()],
        warnings: Vec::new(),
    });
}

/// Re-parse SKILL.md to pull the descriptive frontmatter fields for rows
/// the loader accepted. The loader returns `Box<dyn Skill>` which only
/// exposes id/specificity — the rest of the frontmatter isn't on the
/// trait. Cheap to re-parse; we've already loaded the file once.
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
    warnings: Vec<String>,
}

fn read_manifest_fields(skill_dir: &Path) -> ManifestFields {
    let path = skill_dir.join("SKILL.md");
    let Ok(sf) = Skillfile::parse_file(&path) else {
        return ManifestFields::default();
    };
    let mut out = ManifestFields {
        name: Some(sf.name),
        description: Some(sf.description),
        license: sf.license,
        ..ManifestFields::default()
    };
    if let Some(ext) = sf.ari_extension {
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

#[cfg(test)]
mod tests {
    use super::*;

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
