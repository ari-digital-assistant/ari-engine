//! Batch keyword-scorer oracle for the FunctionGemma training pipeline.
//!
//! Reads utterances on stdin (one per line) and writes `true` or `false` per
//! line: `true` means the keyword scorer already claims that utterance, so the
//! router never sees it in production and training on it is wasted capacity.
//!
//! This exists because the router is the FALLBACK tier. `route-eval` enforces
//! the same rule on the eval sets; this enforces it on the training corpus.
//!
//! Usage: `keyword-hit [--locale <xx>] [--skills-dir <path>] < utterances.txt`
//!
//! `--skills-dir` points at a directory of installed skill folders (the
//! `skills/` root of an `ari-skills` checkout). Without it the oracle knows
//! only the six built-ins, so a community skill's examples can never be
//! recognised as keyword-hits even when that skill's own `matching.patterns`
//! win them outright in production — and they stay in the corpus as waste.
//! Omitting the flag preserves the builtin-only behaviour exactly.

use ari_ffi::register_community_skills;
use std::path::{Path, PathBuf};

/// Parsed command line.
#[derive(Debug, PartialEq)]
struct Args {
    locale: String,
    skills_dir: Option<PathBuf>,
}

/// Parse args: an optional `--locale <xx>` (defaulting to "en") and an
/// optional `--skills-dir <path>` (defaulting to none, i.e. built-ins only).
fn parse_args(args: impl Iterator<Item = String>) -> Result<Args, String> {
    let mut locale = "en".to_string();
    let mut skills_dir: Option<PathBuf> = None;
    let mut it = args;
    while let Some(a) = it.next() {
        match a.as_str() {
            "--locale" => {
                locale = it.next().ok_or_else(|| "--locale requires a value".to_string())?;
            }
            "--skills-dir" => {
                let v = it
                    .next()
                    .ok_or_else(|| "--skills-dir requires a value".to_string())?;
                skills_dir = Some(PathBuf::from(v));
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    Ok(Args { locale, skills_dir })
}

/// Answer the keyword question for each line, preserving order and count.
fn run(locale: &str, skills_dir: Option<&Path>, texts: &[String]) -> Result<Vec<bool>, String> {
    let mut engine = ari_ffi::build_engine_with_builtins();
    engine.set_locale(locale.to_string());
    if let Some(root) = skills_dir {
        let loaded = register_community_skills(&mut engine, root)?;
        eprintln!("keyword-hit: registered {loaded} skill(s) from {}", root.display());
    }
    Ok(texts
        .iter()
        .map(|t| engine.keyword_decision(t).is_some())
        .collect())
}

/// Split raw stdin text into one entry per input line.
///
/// Uses `str::lines()` so the line-exact contract holds at every edge:
/// empty input yields zero lines (not a phantom empty one), a lone `"\n"`
/// yields exactly one empty line, a trailing newline is not a phantom extra
/// line, and CRLF endings don't leave a stray `\r` on the line.
fn split_lines(input: &str) -> Vec<String> {
    input.lines().map(|s| s.to_string()).collect()
}

fn main() {
    use std::io::Read;

    let args = match parse_args(std::env::args().skip(1)) {
        Ok(a) => a,
        Err(msg) => {
            eprintln!("{msg}");
            std::process::exit(2);
        }
    };

    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .expect("read stdin");

    let texts = split_lines(&input);

    let verdicts = match run(&args.locale, args.skills_dir.as_deref(), &texts) {
        Ok(v) => v,
        Err(msg) => {
            eprintln!("{msg}");
            std::process::exit(2);
        }
    };

    for verdict in verdicts {
        println!("{verdict}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(v: &[&str]) -> Result<Args, String> {
        parse_args(v.iter().map(|s| s.to_string()))
    }

    /// The real `ari-skills` checkout, resolved as a sibling of this engine
    /// clone — the layout `generate-dataset.py::find_skills_dir` assumes and
    /// the one the training pipeline actually runs against.
    ///
    /// The oracle's correctness is a property of the REAL manifests, not of
    /// any fixture we could write here, so these tests read them directly. If
    /// the sibling clone is missing we fail loudly with instructions rather
    /// than skipping: a silently-skipped test would let the oracle drift out
    /// of agreement with the manifests it exists to model.
    fn real_skills_root() -> PathBuf {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../ari-skills/skills")
            .canonicalize()
            .unwrap_or_else(|e| {
                panic!(
                    "cannot resolve the ari-skills checkout ({e}). Clone \
                     https://github.com/ari-digital-assistant/ari-skills as a sibling of \
                     this ari-engine checkout — the keyword oracle is defined against \
                     the real community manifests."
                )
            });
        assert!(
            root.join("timer/SKILL.en.md").is_file(),
            "{} is not an ari-skills skills/ root",
            root.display()
        );
        root
    }

    fn verdicts(locale: &str, skills_dir: Option<&Path>, texts: &[&str]) -> Vec<bool> {
        let owned: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        run(locale, skills_dir, &owned).unwrap()
    }

    #[test]
    fn no_flag_defaults_to_en() {
        assert_eq!(parse(&[]).unwrap().locale, "en");
    }

    #[test]
    fn locale_flag_is_parsed() {
        assert_eq!(parse(&["--locale", "it"]).unwrap().locale, "it");
    }

    #[test]
    fn locale_flag_without_value_is_an_error() {
        assert!(parse(&["--locale"]).is_err());
    }

    #[test]
    fn unknown_argument_is_an_error() {
        assert!(parse(&["--nope"]).is_err());
    }

    #[test]
    fn no_skills_dir_flag_means_builtins_only() {
        assert_eq!(parse(&[]).unwrap().skills_dir, None);
    }

    #[test]
    fn skills_dir_flag_is_parsed() {
        assert_eq!(
            parse(&["--skills-dir", "/opt/ari/skills"]).unwrap().skills_dir,
            Some(PathBuf::from("/opt/ari/skills"))
        );
    }

    #[test]
    fn skills_dir_flag_without_value_is_an_error() {
        assert_eq!(
            parse(&["--skills-dir"]).unwrap_err(),
            "--skills-dir requires a value"
        );
    }

    #[test]
    fn both_flags_parse_together() {
        assert_eq!(
            parse(&["--locale", "it", "--skills-dir", "/s"]).unwrap(),
            Args {
                locale: "it".to_string(),
                skills_dir: Some(PathBuf::from("/s")),
            }
        );
    }

    #[test]
    fn english_verdicts_match_the_keyword_scorer() {
        let texts: Vec<String> = [
            "what time is it",      // canonical trigger — keyword scorer wins
            "long time no see",     // oblique greeting — router's job
            "what is the capital of Denmark", // general knowledge — nobody's trigger
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        assert_eq!(run("en", None, &texts).unwrap(), vec![true, false, false]);
    }

    #[test]
    fn italian_verdicts_use_the_italian_scorer() {
        let texts: Vec<String> = ["apri spotify", "che si racconta"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(run("it", None, &texts).unwrap(), vec![true, false]);
    }

    #[test]
    fn empty_input_yields_empty_output() {
        assert_eq!(run("en", None, &[]).unwrap(), Vec::<bool>::new());
    }

    /// The point of `--skills-dir`: these four utterances are won outright by
    /// their OWN skill's `matching.patterns` in production, but the oracle
    /// cannot see that until the community manifests are registered.
    ///
    /// Each one is quoted from the real manifest it belongs to:
    ///   timer      `\b(set|start|create|add)\b.*\btimer\b`      (SKILL.en.md)
    ///   weather    `\bweather\b`                                 (SKILL.en.md)
    ///   navigation `\b(navigate|directions|route) to\b`          (SKILL.en.md)
    ///   reminder   `\bremind me\b`                               (SKILL.en.md)
    #[test]
    fn english_community_patterns_flip_verdicts_from_false_to_true() {
        let owned = [
            "set a timer for 10 minutes",
            "what is the weather",
            "navigate to the airport",
            "remind me to call mum",
        ];
        assert_eq!(
            verdicts("en", None, &owned),
            vec![false, false, false, false],
            "builtins alone must not claim any of these — that is the bug"
        );
        assert_eq!(
            verdicts("en", Some(&real_skills_root()), &owned),
            vec![true, true, true, true],
            "each of these is won by its own skill's patterns in production"
        );
    }

    /// Same contrast in Italian, quoted from the `SKILL.it.md` variants:
    ///   timer   `\b(imposta|avvia|metti|crea) (un )?timer\b`
    ///   weather `\b(tempo|meteo)\b`
    #[test]
    fn italian_community_patterns_flip_verdicts_from_false_to_true() {
        let owned = ["imposta un timer per 10 minuti", "che meteo fa"];
        assert_eq!(verdicts("it", None, &owned), vec![false, false]);
        assert_eq!(
            verdicts("it", Some(&real_skills_root()), &owned),
            vec![true, true]
        );
    }

    /// General knowledge belongs to nobody in either mode. Without this the
    /// contrast tests above could be satisfied by a flag that just answered
    /// `true` to everything.
    #[test]
    fn general_knowledge_is_a_miss_with_and_without_community_skills() {
        let owned = ["what is the capital of Denmark", "who painted the Mona Lisa"];
        assert_eq!(verdicts("en", None, &owned), vec![false, false]);
        assert_eq!(
            verdicts("en", Some(&real_skills_root()), &owned),
            vec![false, false]
        );
    }

    /// Registering community skills must not disturb the built-ins' own
    /// verdicts — `--skills-dir` may only ever add hits, never move existing
    /// ones.
    #[test]
    fn builtin_verdicts_survive_community_registration() {
        let owned = ["what time is it", "long time no see"];
        assert_eq!(verdicts("en", None, &owned), vec![true, false]);
        assert_eq!(
            verdicts("en", Some(&real_skills_root()), &owned),
            vec![true, false]
        );
    }

    #[test]
    fn a_missing_skills_dir_is_an_error() {
        let err = run("en", Some(Path::new("/nonexistent/ari/skills")), &[]).unwrap_err();
        assert!(
            err.starts_with("--skills-dir /nonexistent/ari/skills:"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn a_skill_that_fails_to_load_is_fatal() {
        // A directory with no SKILL.md is a load failure. Silently tolerating
        // it would mean scoring against an incomplete pattern set.
        let dir = tempdir_lite::TempDir::new("ari-keyword-hit-test");
        std::fs::create_dir_all(dir.path().join("not-a-skill")).unwrap();
        let err = run("en", Some(dir.path()), &[]).unwrap_err();
        assert!(
            err.contains("1 skill(s)") && err.contains("no SKILL.md found"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn empty_stdin_splits_to_zero_lines() {
        assert_eq!(split_lines(""), Vec::<String>::new());
    }

    #[test]
    fn single_blank_line_splits_to_exactly_one_line() {
        // Regression: the old strip_suffix + split logic collapsed a lone
        // "\n" down to zero lines, silently mis-pairing the caller's zip().
        assert_eq!(split_lines("\n"), vec!["".to_string()]);
    }

    #[test]
    fn trailing_newline_does_not_add_a_phantom_line() {
        assert_eq!(
            split_lines("what time is it\nlong time no see\n"),
            vec!["what time is it".to_string(), "long time no see".to_string()]
        );
    }

    #[test]
    fn embedded_blank_line_preserves_total_count() {
        assert_eq!(
            split_lines("first\n\nthird"),
            vec!["first".to_string(), "".to_string(), "third".to_string()]
        );
    }

    #[test]
    fn crlf_input_does_not_leave_a_trailing_carriage_return() {
        assert_eq!(
            split_lines("what time is it\r\nlong time no see\r\n"),
            vec!["what time is it".to_string(), "long time no see".to_string()]
        );
    }

    /// Minimal self-deleting temp directory — same pattern the skill-loader
    /// and validator tests use, so the bin picks up no dev-dependency.
    mod tempdir_lite {
        use std::path::{Path, PathBuf};

        pub struct TempDir(PathBuf);

        impl TempDir {
            pub fn new(prefix: &str) -> Self {
                let nanos = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos();
                let p = std::env::temp_dir().join(format!("{prefix}-{nanos}"));
                std::fs::create_dir_all(&p).unwrap();
                TempDir(p)
            }

            pub fn path(&self) -> &Path {
                &self.0
            }
        }

        impl Drop for TempDir {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
    }
}
