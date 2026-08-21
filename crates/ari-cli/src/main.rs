use ari_engine::Engine;
use ari_skill_loader::{
    capability_name, load_single_skill_dir_with, load_skill_directory_with, parse_capability,
    HostCapabilities, LoadOptions, StorageConfig, ALL_CAPABILITIES,
};
use ari_skills::{CalculatorSkill, CurrentTimeSkill, DateSkill, GreetingSkill, OpenSkill, SearchSkill};
use std::io::{self, BufRead};
use std::path::PathBuf;
use std::process::ExitCode;

pub mod store_cli;

fn main() -> ExitCode {
    let raw_args: Vec<String> = std::env::args().skip(1).collect();

    // Subcommand dispatch happens before utterance parsing: if the first
    // positional matches a known store command, it gets routed there.
    if let Some(first) = raw_args.first() {
        match first.as_str() {
            "install" => return store_cli::run_install(&raw_args[1..]),
            "uninstall" => return store_cli::run_uninstall(&raw_args[1..]),
            "list" => return store_cli::run_list(&raw_args[1..]),
            "check-updates" => return store_cli::run_check_updates(&raw_args[1..]),
            "update" => return store_cli::run_update(&raw_args[1..]),
            _ => {}
        }
    }

    let parsed = match parse_args(raw_args) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("ari: {e}");
            eprintln!();
            print_usage();
            return ExitCode::from(2);
        }
    };

    let mut engine = Engine::new();
    engine.set_debug(parsed.debug);
    engine.register_skill(Box::new(CurrentTimeSkill::new()));
    engine.register_skill(Box::new(DateSkill::new()));
    engine.register_skill(Box::new(CalculatorSkill::new()));
    engine.register_skill(Box::new(GreetingSkill::new()));
    engine.register_skill(Box::new(OpenSkill::new()));
    engine.register_skill(Box::new(SearchSkill::new()));

    let storage_config = match &parsed.storage_dir {
        Some(p) => StorageConfig::new(p.clone()),
        None => StorageConfig::ephemeral_default(),
    };

    let load_options = LoadOptions {
        host_capabilities: parsed.host_capabilities.clone(),
        storage_config: storage_config.clone(),
        ..LoadOptions::default()
    };

    if parsed.debug {
        eprintln!(
            "[ari] host capabilities: {:?}",
            parsed.host_capabilities_summary()
        );
        eprintln!("[ari] storage root: {}", storage_config.root.display());
    }

    let mut all_dirs: Vec<PathBuf> = parsed.extra_skill_dirs.clone();
    if let Some(store_path) = &parsed.skill_store {
        all_dirs.push(store_path.clone());
    }

    for path in &all_dirs {
        if !path.exists() {
            // Skill store may not exist yet on a fresh machine — that's not
            // an error, it just means nothing's installed there yet.
            if parsed.debug {
                eprintln!("[ari] skipping missing skill dir: {}", path.display());
            }
            continue;
        }
        let report = if has_skill_md(path) {
            load_single_skill_dir_with(path, &load_options)
        } else {
            match load_skill_directory_with(path, &load_options) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("ari: could not read {}: {e}", path.display());
                    return ExitCode::from(1);
                }
            }
        };

        for failure in &report.failures {
            eprintln!("ari: skipping skill — {failure}");
        }
        for skill in report.skills {
            if parsed.debug {
                eprintln!("[ari] loaded sideloaded skill: {}", skill.id());
            }
            engine.register_skill(skill);
        }
    }

    #[cfg(feature = "llm")]
    if let Some(ref model_path) = parsed.llm_model {
        if parsed.debug {
            eprintln!("[ari] loading LLM model: {}", model_path.display());
        }
        match ari_llm::LlmFallback::load(model_path) {
            Ok(llm) => {
                engine.set_llm(std::sync::Arc::new(llm));
                if parsed.debug {
                    eprintln!("[ari] LLM model loaded");
                }
            }
            Err(e) => {
                eprintln!("ari: failed to load LLM model: {e}");
                return ExitCode::from(1);
            }
        }
    }

    if !parsed.utterance.is_empty() {
        let response = engine.process_input(&parsed.utterance);
        print_response(&response);
        return ExitCode::SUCCESS;
    }

    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.trim().is_empty() {
            continue;
        }
        let response = engine.process_input(&line);
        print_response(&response);
    }
    ExitCode::SUCCESS
}

#[derive(Debug)]
struct ParsedArgs {
    debug: bool,
    extra_skill_dirs: Vec<PathBuf>,
    utterance: String,
    host_capabilities: HostCapabilities,
    storage_dir: Option<PathBuf>,
    skill_store: Option<PathBuf>,
    #[cfg(feature = "llm")]
    llm_model: Option<PathBuf>,
}

impl Default for ParsedArgs {
    fn default() -> Self {
        Self {
            debug: false,
            extra_skill_dirs: Vec::new(),
            utterance: String::new(),
            host_capabilities: HostCapabilities::pure_frontend(),
            storage_dir: None,
            skill_store: None,
            #[cfg(feature = "llm")]
            llm_model: None,
        }
    }
}

impl ParsedArgs {
    /// The capability names the loader will actually be given.
    ///
    /// Read off the set itself rather than a parallel `Vec<String>` built
    /// during parsing: the two could disagree, and the hand-written default
    /// branch did — it named four of the eleven `pure_frontend()` grants.
    fn host_capabilities_summary(&self) -> Vec<&'static str> {
        self.host_capabilities
            .granted()
            .into_iter()
            .map(capability_name)
            .collect()
    }
}

fn parse_args(args: Vec<String>) -> Result<ParsedArgs, String> {
    let mut parsed = ParsedArgs::default();
    let mut positional: Vec<String> = Vec::new();
    let mut iter = args.into_iter();
    let mut explicit_host_caps: Option<HostCapabilities> = None;
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--debug" => parsed.debug = true,
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            "--extra-skill-dir" => {
                let value = iter
                    .next()
                    .ok_or_else(|| "--extra-skill-dir requires a path argument".to_string())?;
                parsed.extra_skill_dirs.push(PathBuf::from(value));
            }
            other if other.starts_with("--extra-skill-dir=") => {
                let value = &other["--extra-skill-dir=".len()..];
                if value.is_empty() {
                    return Err("--extra-skill-dir requires a path argument".to_string());
                }
                parsed.extra_skill_dirs.push(PathBuf::from(value));
            }
            "--host-capabilities" => {
                let value = iter.next().ok_or_else(|| {
                    "--host-capabilities requires a comma-separated list".to_string()
                })?;
                explicit_host_caps = Some(parse_caps_csv(&value)?);
            }
            other if other.starts_with("--host-capabilities=") => {
                let value = &other["--host-capabilities=".len()..];
                explicit_host_caps = Some(parse_caps_csv(value)?);
            }
            "--no-host-capabilities" => {
                explicit_host_caps = Some(HostCapabilities::none());
            }
            "--storage-dir" => {
                let value = iter
                    .next()
                    .ok_or_else(|| "--storage-dir requires a path".to_string())?;
                parsed.storage_dir = Some(PathBuf::from(value));
            }
            other if other.starts_with("--storage-dir=") => {
                let value = &other["--storage-dir=".len()..];
                if value.is_empty() {
                    return Err("--storage-dir requires a path".to_string());
                }
                parsed.storage_dir = Some(PathBuf::from(value));
            }
            "--skill-store" => {
                let value = iter
                    .next()
                    .ok_or_else(|| "--skill-store requires a path".to_string())?;
                parsed.skill_store = Some(PathBuf::from(value));
            }
            other if other.starts_with("--skill-store=") => {
                let value = &other["--skill-store=".len()..];
                if value.is_empty() {
                    return Err("--skill-store requires a path".to_string());
                }
                parsed.skill_store = Some(PathBuf::from(value));
            }
            #[cfg(feature = "llm")]
            "--llm-model" => {
                let value = iter
                    .next()
                    .ok_or_else(|| "--llm-model requires a path to a GGUF model".to_string())?;
                parsed.llm_model = Some(PathBuf::from(value));
            }
            #[cfg(feature = "llm")]
            other_llm if other_llm.starts_with("--llm-model=") => {
                let value = &other_llm["--llm-model=".len()..];
                if value.is_empty() {
                    return Err("--llm-model requires a path to a GGUF model".to_string());
                }
                parsed.llm_model = Some(PathBuf::from(value));
            }
            other if other.starts_with("--") => {
                return Err(format!("unknown option: {other}"));
            }
            _ => positional.push(arg),
        }
    }
    if let Some(caps) = explicit_host_caps {
        parsed.host_capabilities = caps;
    }
    parsed.utterance = positional.join(" ");
    Ok(parsed)
}

/// Column the wrapped capability lists in `--help` start at, and how much room
/// they have before an 80-column terminal wraps them itself.
const HELP_LIST_INDENT: &str = "                                     ";
const HELP_LIST_WIDTH: usize = 41;

/// Break `items` into comma-separated lines of at most `width` characters.
///
/// The capability lists in `--help` are derived from the enum, so they have no
/// fixed length — without this, adding a capability quietly pushes a help line
/// off the edge of the terminal. An item longer than `width` gets its own line
/// rather than being cut.
fn wrap_list(items: &[&str], width: usize) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    for (i, item) in items.iter().enumerate() {
        let piece = if i + 1 == items.len() {
            format!("{item}.")
        } else {
            format!("{item},")
        };
        match lines.last_mut() {
            Some(line) if line.len() + 1 + piece.len() <= width => {
                line.push(' ');
                line.push_str(&piece);
            }
            _ => lines.push(piece),
        }
    }
    lines
}

fn parse_caps_csv(value: &str) -> Result<HostCapabilities, String> {
    let mut caps = HostCapabilities::none();
    for raw in value.split(',') {
        let name = raw.trim();
        if name.is_empty() {
            continue;
        }
        let cap = parse_capability(name)
            .ok_or_else(|| format!("unknown capability: {name:?}"))?;
        caps = caps.with(cap);
    }
    Ok(caps)
}

/// Does `path` look like a single skill directory rather than a registry root?
///
/// Accepts both manifest names the loader accepts: the canonical per-locale
/// `SKILL.en.md` and the legacy bare `SKILL.md`. Checking only the legacy name
/// silently mis-classified every modern skill as a registry root, which then
/// walked into `target/` and loaded nothing.
fn has_skill_md(path: &std::path::Path) -> bool {
    path.join("SKILL.md").is_file() || path.join("SKILL.en.md").is_file()
}

fn print_usage() {
    eprintln!(
        "usage: ari-cli [--debug] [--llm-model <path>] [--extra-skill-dir <path>]... \
         [--host-capabilities <list>|--no-host-capabilities] [utterance...]"
    );
    eprintln!();
    eprintln!("  --debug                          print scoring trace to stderr");
    eprintln!("  --extra-skill-dir <path>         sideload skills from a directory.");
    eprintln!("                                   if <path>/SKILL.en.md (or the legacy <path>/SKILL.md)");
    eprintln!("                                   exists, loads that one skill; otherwise treats <path>");
    eprintln!("                                   as a registry root and loads every skill under it.");
    eprintln!("                                   may be passed multiple times.");
    eprintln!("  --host-capabilities <list>       override the host capability set with a");
    eprintln!("                                   comma-separated list. Valid names:");
    let all: Vec<&str> = ALL_CAPABILITIES.iter().copied().map(capability_name).collect();
    for line in wrap_list(&all, HELP_LIST_WIDTH) {
        eprintln!("{HELP_LIST_INDENT}{line}");
    }
    eprintln!("                                   Default: pure_frontend, which is");
    let default = HostCapabilities::pure_frontend();
    let default: Vec<&str> = default.granted().into_iter().map(capability_name).collect();
    for line in wrap_list(&default, HELP_LIST_WIDTH) {
        eprintln!("{HELP_LIST_INDENT}{line}");
    }
    eprintln!("  --no-host-capabilities           grant the empty capability set; any skill with");
    eprintln!("                                   declared capabilities will be rejected at load.");
    eprintln!("  --llm-model <path>               load a GGUF model for the LLM fallback.");
    eprintln!("                                   when loaded, unmatched input is sent to the model");
    eprintln!("                                   for skill rerouting or direct QA before giving up.");
    eprintln!("  --storage-dir <path>             directory used for the WASM storage_kv per-skill");
    eprintln!("                                   key-value files. Defaults to a system-temp dir,");
    eprintln!("                                   which is fine for sideloading but not persistent");
    eprintln!("                                   across reboots.");
    eprintln!();
    eprintln!("  --skill-store <path>             auto-load every skill installed under this dir");
    eprintln!("                                   (the directory `ari install` writes into).");
    eprintln!();
    eprintln!("subcommands (must be the first argument):");
    eprintln!("  ari install <bundle> [<sig>] --trust-key-hex <hex> [--skill-store <dir>]");
    eprintln!("                                   verify and install a signed .tar.gz bundle.");
    eprintln!("                                   <sig> defaults to <bundle>.sig.");
    eprintln!("                                   sha256 is read from <bundle>.sha256 if present,");
    eprintln!("                                   otherwise computed from the bundle bytes.");
    eprintln!("  ari uninstall <skill-id> [--skill-store <dir>] [--storage-dir <dir>]");
    eprintln!("                                   remove an installed skill and wipe its storage_kv.");
    eprintln!("  ari list [--skill-store <dir>]   print id, version, and path of every installed skill.");
    eprintln!("  ari check-updates [--skill-store <dir>] [--registry-index-url <url>]");
    eprintln!("                                   fetch the registry index and print any skills whose");
    eprintln!("                                   installed version is older than the published one.");
    eprintln!("  ari update <skill-id> [--skill-store <dir>] [--registry-index-url <url>]");
    eprintln!("                                   [--registry-base-url <url>] [--registry-trust-key-hex <hex>]");
    eprintln!("                                   download the current registry version of <skill-id>");
    eprintln!("                                   and install it over the existing one.");
    eprintln!();
    eprintln!("if no utterance is given on the command line, ari-cli reads one per line from stdin.");
}

fn print_response(response: &ari_core::Response) {
    match response {
        ari_core::Response::Text(s) => println!("{s}"),
        ari_core::Response::Action(v) => println!(
            "{}",
            serde_json::to_string_pretty(v).unwrap_or_else(|_| v.to_string())
        ),
        ari_core::Response::Binary { mime, data } => {
            println!("[binary: {mime}, {} bytes]", data.len())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrap_list_packs_lines_up_to_the_width() {
        assert_eq!(
            wrap_list(&["http", "location", "tts"], 20),
            vec!["http, location, tts.".to_string()],
        );
        assert_eq!(
            wrap_list(&["http", "location", "tts"], 19),
            vec!["http, location,".to_string(), "tts.".to_string()],
        );
    }

    #[test]
    fn wrap_list_gives_an_overlong_item_its_own_line() {
        assert_eq!(
            wrap_list(&["tts", "media_services"], 4),
            vec!["tts,".to_string(), "media_services.".to_string()],
        );
    }

    #[test]
    fn wrap_list_handles_one_item_and_none() {
        assert_eq!(wrap_list(&["tts"], 41), vec!["tts.".to_string()]);
        assert!(wrap_list(&[], 41).is_empty());
    }

    #[test]
    fn every_capability_fits_the_help_column() {
        for cap in ALL_CAPABILITIES {
            let name = capability_name(*cap);
            // Every name renders with one trailing comma or full stop.
            let rendered = name.len() + 1;
            assert!(
                rendered <= HELP_LIST_WIDTH,
                "{name} is too long for the --help capability column",
            );
        }
    }

    #[test]
    fn summary_reports_the_set_that_will_be_granted() {
        let parsed = parse_args(vec!["--host-capabilities".into(), "tts,http".into()]).unwrap();
        // ALL_CAPABILITIES order, not the order they were typed.
        assert_eq!(parsed.host_capabilities_summary(), vec!["http", "tts"]);

        let none = parse_args(vec!["--no-host-capabilities".into()]).unwrap();
        assert!(none.host_capabilities_summary().is_empty());

        let default = parse_args(Vec::new()).unwrap();
        assert_eq!(default.host_capabilities_summary().len(), 11);
        assert!(default.host_capabilities_summary().contains(&"send_message"));
    }

    #[test]
    fn unknown_capability_is_rejected() {
        let err = parse_args(vec!["--host-capabilities".into(), "tts,telepathy".into()]).unwrap_err();
        assert_eq!(err, "unknown capability: \"telepathy\"");
    }
}
