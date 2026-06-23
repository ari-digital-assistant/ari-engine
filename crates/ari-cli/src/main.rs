use ari_engine::Engine;
use ari_skill_loader::{
    load_single_skill_dir_with, load_skill_directory_with, parse_capability, HostCapabilities,
    LoadOptions, StorageConfig,
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
    host_capabilities_explicit: Vec<String>,
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
            host_capabilities_explicit: Vec::new(),
            storage_dir: None,
            skill_store: None,
            #[cfg(feature = "llm")]
            llm_model: None,
        }
    }
}

impl ParsedArgs {
    fn host_capabilities_summary(&self) -> Vec<&str> {
        if !self.host_capabilities_explicit.is_empty() {
            self.host_capabilities_explicit
                .iter()
                .map(String::as_str)
                .collect()
        } else {
            vec!["notifications", "launch_app", "clipboard", "tts (default: pure_frontend)"]
        }
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
                let (caps, names) = parse_caps_csv(&value)?;
                explicit_host_caps = Some(caps);
                parsed.host_capabilities_explicit = names;
            }
            other if other.starts_with("--host-capabilities=") => {
                let value = &other["--host-capabilities=".len()..];
                let (caps, names) = parse_caps_csv(value)?;
                explicit_host_caps = Some(caps);
                parsed.host_capabilities_explicit = names;
            }
            "--no-host-capabilities" => {
                explicit_host_caps = Some(HostCapabilities::none());
                parsed.host_capabilities_explicit = vec!["(none)".to_string()];
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

fn parse_caps_csv(value: &str) -> Result<(HostCapabilities, Vec<String>), String> {
    if value.trim().is_empty() {
        return Ok((HostCapabilities::none(), vec!["(none)".to_string()]));
    }
    let mut caps = HostCapabilities::none();
    let mut names: Vec<String> = Vec::new();
    for raw in value.split(',') {
        let name = raw.trim();
        if name.is_empty() {
            continue;
        }
        let cap = parse_capability(name)
            .ok_or_else(|| format!("unknown capability: {name:?}"))?;
        caps = caps.with(cap);
        names.push(name.to_string());
    }
    Ok((caps, names))
}

fn has_skill_md(path: &std::path::Path) -> bool {
    path.join("SKILL.md").is_file()
}

fn print_usage() {
    eprintln!(
        "usage: ari-cli [--debug] [--llm-model <path>] [--extra-skill-dir <path>]... \
         [--host-capabilities <list>|--no-host-capabilities] [utterance...]"
    );
    eprintln!();
    eprintln!("  --debug                          print scoring trace to stderr");
    eprintln!("  --extra-skill-dir <path>         sideload skills from a directory.");
    eprintln!("                                   if <path>/SKILL.md exists, loads that one skill;");
    eprintln!("                                   otherwise treats <path> as a registry root and");
    eprintln!("                                   loads every <path>/<slug>/SKILL.md it finds.");
    eprintln!("                                   may be passed multiple times.");
    eprintln!("  --host-capabilities <list>       override the host capability set with a");
    eprintln!("                                   comma-separated list. Valid names: http,");
    eprintln!("                                   location, notifications, launch_app, clipboard,");
    eprintln!("                                   tts, storage_kv. Default: pure_frontend");
    eprintln!("                                   (notifications, launch_app, clipboard, tts).");
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
