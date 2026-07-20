//! Routing-eval promotion gate.
//!
//! Loads a candidate FunctionGemma GGUF as the live router, then routes every
//! case in a held-out eval set through the REAL routing path (the same
//! `Engine`, builtin catalogue, prompt builder, parser and confidence gate
//! production uses) and checks the pick against the expected outcome.
//!
//! Usage: `route-eval [--locale <xx>] [--skills-dir <path>] <gguf-path> <eval-jsonl-path>`
//!
//! Each eval line is JSON: `{"utterance": "...", "expect": "<skill_id>|NONE"}`
//! — `NONE` means the router must abstain (general-knowledge question that
//! belongs to the assistant). Exits non-zero when a gated metric falls below
//! threshold, which the training workflow uses to block promotion of a
//! regressed model.
//!
//! GATE v3 (2026-07-19). The router is a MIDDLE tier: everything it declines
//! falls through to the assistant/LLM layer, so an abstention is graceful
//! degradation while a misroute is the only user-visible failure (weather
//! request → coin flip). The gate therefore scores what users experience:
//!   - abstention on NONE cases  >= 0.90   (unchanged)
//!   - precision when firing     >= 0.90   (of the positive cases it routes,
//!                                          how many go to the RIGHT skill)
//!   - recall (positives routed correctly / all positives) is REPORTED as
//!     the coverage KPI to grow, but NOT gated — a low-recall high-precision
//!     router is a net win (fast path when sure, safe hand-off when not),
//!     whereas the old `positive >= 0.80` bar punished safe abstention as if
//!     this were the last tier. That bar was also calibrated on the polluted
//!     pre-guardrail eval; nobody ever consciously chose it for hard
//!     obliques under a full catalogue.
//!
//! `--skills-dir` points at a directory of installed skill folders (the
//! `skills/` root of an `ari-skills` checkout) and feeds the keyword-pollution
//! guardrail below. It mirrors `keyword-hit`'s flag of the same name for a
//! reason: `keyword-hit` decides what leaves the TRAINING corpus and this
//! decides what may be MEASURED, and the two must agree on what a keyword-hit
//! is. Without it the gate knows only the six built-ins, so the first eval
//! case a community pattern claims — a weather, alarm, timer or navigation
//! positive, or a `NONE` case some skill's patterns win — sails through the
//! guardrail and inflates the score that promotes a model to real devices.
//! Omitting the flag preserves the builtin-only behaviour exactly.

use std::path::PathBuf;

// Abstention is the regression we care most about, so it's gated hard.
const ABSTAIN_MIN: f64 = 0.90;
// Of the positive cases the router FIRES on, at least this share must go to
// the right skill. Firing wrong is the only user-visible failure mode.
const PRECISION_MIN: f64 = 0.90;

/// Parsed command line.
#[derive(Debug, PartialEq)]
struct Args {
    gguf: String,
    eval_path: String,
    locale: String,
    skills_dir: Option<PathBuf>,
}

/// Parse route-eval args: two positionals (gguf, eval-jsonl) plus an optional
/// `--locale <xx>` and an optional `--skills-dir <path>`, either of which may
/// appear anywhere. Locale defaults to "en"; `skills_dir` defaults to none,
/// i.e. built-ins only.
fn parse_args(args: impl Iterator<Item = String>) -> Result<Args, String> {
    let mut positionals: Vec<String> = Vec::new();
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
            _ => positionals.push(a),
        }
    }
    match positionals.as_slice() {
        [gguf, eval] => Ok(Args {
            gguf: gguf.clone(),
            eval_path: eval.clone(),
            locale,
            skills_dir,
        }),
        _ => Err(
            "usage: route-eval [--locale <xx>] [--skills-dir <path>] <gguf-path> <eval-jsonl-path>"
                .to_string(),
        ),
    }
}

#[cfg(feature = "llm")]
fn main() {
    use std::io::BufRead;

    let Args { gguf, eval_path, locale, skills_dir } = match parse_args(std::env::args().skip(1)) {
        Ok(t) => t,
        Err(msg) => {
            eprintln!("{msg}");
            std::process::exit(2);
        }
    };

    let mut engine = ari_ffi::build_engine_with_builtins();
    engine.set_locale(locale);
    // Registered BEFORE the guardrail runs so community `matching.patterns`
    // participate in the keyword-hit question. A bad path is fatal: loading
    // fewer skills than asked for would under-count keyword-hits, which is the
    // exact silent inflation the guardrail exists to catch.
    if let Some(root) = skills_dir.as_deref() {
        match ari_ffi::register_community_skills(&mut engine, root) {
            Ok(loaded) => {
                eprintln!("route-eval: registered {loaded} skill(s) from {}", root.display())
            }
            Err(msg) => {
                eprintln!("{msg}");
                std::process::exit(2);
            }
        }
    }
    let router = ari_llm::FunctionGemmaRouter::new(std::path::Path::new(&gguf));
    engine.set_router(Some(Box::new(router)));

    // Guardrail: the router is the FALLBACK tier — it only fires when the
    // keyword scorer finds nothing. Any eval case the keyword scorer already
    // wins never reaches the router in production, so scoring the router on it
    // measures nothing and silently corrupts the number. Refuse to report.
    {
        let file = std::fs::File::open(&eval_path)
            .unwrap_or_else(|e| panic!("cannot open eval set {eval_path}: {e}"));
        let mut offenders: Vec<String> = Vec::new();
        for line in std::io::BufReader::new(file).lines() {
            let line = line.expect("read eval line");
            let line = line.trim();
            if line.is_empty() || line.starts_with("//") {
                continue;
            }
            let case: serde_json::Value =
                serde_json::from_str(line).unwrap_or_else(|e| panic!("bad eval line {line:?}: {e}"));
            let utterance = case["utterance"].as_str().expect("utterance field");
            let expect = case["expect"].as_str().expect("expect field");
            if let Some(kw) = engine.keyword_decision(utterance) {
                offenders.push(format!(
                    "  {utterance:?}  expect={expect}  but the keyword scorer already routes it to {kw:?}"
                ));
            }
        }
        if !offenders.is_empty() {
            eprintln!(
                "EVAL POLLUTED — {} case(s) are handled by the keyword scorer and never reach the router:",
                offenders.len()
            );
            for o in &offenders {
                eprintln!("{o}");
            }
            eprintln!(
                "A router promotion gate must contain only keyword-MISSES. Remove or replace these cases."
            );
            std::process::exit(3);
        }
    }

    // ROUTE_EVAL_VERBOSE=1 dumps a tab-separated line per case
    // (category, confidence, expect, raw-pick, utterance) for confidence
    // distribution analysis.
    let verbose = std::env::var("ROUTE_EVAL_VERBOSE").is_ok();

    let file = std::fs::File::open(&eval_path)
        .unwrap_or_else(|e| panic!("cannot open eval set {eval_path}: {e}"));

    let (mut abstain_total, mut abstain_pass) = (0u32, 0u32);
    let (mut positive_total, mut positive_pass) = (0u32, 0u32);
    // Positives the post-threshold router actually FIRED on (right or wrong).
    // Precision = positive_pass / positive_fired; misses (abstained
    // positives) cost recall, never precision — they fall through to the
    // assistant tier and the user is still served.
    let mut positive_fired = 0u32;
    let mut failures: Vec<String> = Vec::new();

    for line in std::io::BufReader::new(file).lines() {
        let line = line.expect("read eval line");
        let line = line.trim();
        if line.is_empty() || line.starts_with("//") {
            continue;
        }
        let case: serde_json::Value =
            serde_json::from_str(line).unwrap_or_else(|e| panic!("bad eval line {line:?}: {e}"));
        let utterance = case["utterance"].as_str().expect("utterance field");
        let expect = case["expect"].as_str().expect("expect field");

        let raw = engine.route_raw(utterance);
        // Post-threshold decision — identical to Engine::route_decision.
        let got = raw
            .as_ref()
            .filter(|(_, c)| *c >= ari_core::MIN_ROUTER_CONFIDENCE)
            .map(|(id, _)| id.clone());
        let abstaining = expect.eq_ignore_ascii_case("NONE");
        let pass = if abstaining {
            got.is_none()
        } else {
            got.as_deref() == Some(expect)
        };

        if verbose {
            let (pick, conf) = match &raw {
                Some((id, c)) => (id.as_str(), *c),
                None => ("NONE", f32::NAN),
            };
            let category = if abstaining {
                if raw.is_none() { "ABSTAIN_OK" } else { "MISROUTE" }
            } else if pass {
                "POS_OK"
            } else if raw.is_none() {
                "POS_MISS"
            } else {
                "POS_WRONG"
            };
            println!("VERBOSE\t{category}\t{conf:.4}\t{expect}\t{pick}\t{utterance}");
        }

        if abstaining {
            abstain_total += 1;
            if pass {
                abstain_pass += 1;
            }
        } else {
            positive_total += 1;
            if got.is_some() {
                positive_fired += 1;
            }
            if pass {
                positive_pass += 1;
            }
        }

        if !pass {
            let got_s = got.as_deref().unwrap_or("NONE");
            failures.push(format!("  FAIL  {utterance:?}  expected={expect}  got={got_s}"));
        }
    }

    let rate = |pass: u32, total: u32| if total == 0 { 1.0 } else { pass as f64 / total as f64 };
    let abstain_rate = rate(abstain_pass, abstain_total);
    let recall = rate(positive_pass, positive_total);
    // A router that never fires has vacuously perfect precision — harmless
    // but useless, which the recall line makes visible.
    let precision = rate(positive_pass, positive_fired);

    if !failures.is_empty() {
        eprintln!("Failures:");
        for f in &failures {
            eprintln!("{f}");
        }
    }
    // Every number below is a function of this threshold, and the harness is
    // built from whatever ari-engine the caller checked out. A gate result
    // that doesn't say what it was graded against is unreproducible: a
    // nightly once failed at -0.10 and passed hours later at -0.06 on the
    // same corpus, and nothing in the output showed why.
    println!(
        "threshold:  MIN_ROUTER_CONFIDENCE = {}",
        ari_core::MIN_ROUTER_CONFIDENCE
    );
    println!(
        "abstention: {abstain_pass}/{abstain_total} ({:.0}%, min {:.0}%)",
        abstain_rate * 100.0,
        ABSTAIN_MIN * 100.0
    );
    println!(
        "precision:  {positive_pass}/{positive_fired} fired ({:.0}%, min {:.0}%)",
        precision * 100.0,
        PRECISION_MIN * 100.0
    );
    println!(
        "recall:     {positive_pass}/{positive_total} ({:.0}%) — coverage KPI, tracked not gated",
        recall * 100.0
    );
    if positive_fired == 0 {
        println!("note: router never fired on a positive — precision is vacuous.");
    }

    let ok = abstain_rate >= ABSTAIN_MIN && precision >= PRECISION_MIN;
    if ok {
        println!("GATE PASS — model may be promoted.");
    } else {
        eprintln!("GATE FAIL — model must NOT be promoted.");
        std::process::exit(1);
    }
}

#[cfg(not(feature = "llm"))]
fn main() {
    eprintln!("route-eval requires the 'llm' feature (FunctionGemma router)");
    std::process::exit(2);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(v: &[&str]) -> Result<Args, String> {
        parse_args(v.iter().map(|s| s.to_string()))
    }

    fn args(gguf: &str, eval: &str, locale: &str, skills: Option<&str>) -> Args {
        Args {
            gguf: gguf.to_string(),
            eval_path: eval.to_string(),
            locale: locale.to_string(),
            skills_dir: skills.map(PathBuf::from),
        }
    }

    #[test]
    fn positional_only_defaults_to_en() {
        assert_eq!(
            parse(&["m.gguf", "e.jsonl"]).unwrap(),
            args("m.gguf", "e.jsonl", "en", None)
        );
    }

    #[test]
    fn locale_flag_before_positionals() {
        assert_eq!(
            parse(&["--locale", "it", "m.gguf", "e.jsonl"]).unwrap(),
            args("m.gguf", "e.jsonl", "it", None)
        );
    }

    #[test]
    fn locale_flag_after_positionals() {
        assert_eq!(
            parse(&["m.gguf", "e.jsonl", "--locale", "it"]).unwrap(),
            args("m.gguf", "e.jsonl", "it", None)
        );
    }

    #[test]
    fn missing_positionals_is_an_error() {
        assert!(parse(&["--locale", "it"]).is_err());
    }

    /// Absent `--skills-dir` must stay builtin-only, so every existing manual
    /// invocation keeps its current meaning.
    #[test]
    fn no_skills_dir_flag_means_builtins_only() {
        assert_eq!(parse(&["m.gguf", "e.jsonl"]).unwrap().skills_dir, None);
    }

    #[test]
    fn skills_dir_flag_before_positionals() {
        assert_eq!(
            parse(&["--skills-dir", "/opt/ari/skills", "m.gguf", "e.jsonl"]).unwrap(),
            args("m.gguf", "e.jsonl", "en", Some("/opt/ari/skills"))
        );
    }

    #[test]
    fn skills_dir_flag_after_positionals() {
        assert_eq!(
            parse(&["m.gguf", "e.jsonl", "--skills-dir", "/opt/ari/skills"]).unwrap(),
            args("m.gguf", "e.jsonl", "en", Some("/opt/ari/skills"))
        );
    }

    #[test]
    fn skills_dir_flag_without_value_is_an_error() {
        assert_eq!(
            parse(&["m.gguf", "e.jsonl", "--skills-dir"]).unwrap_err(),
            "--skills-dir requires a value"
        );
    }

    /// The shape the training workflow actually invokes.
    #[test]
    fn both_flags_parse_together() {
        assert_eq!(
            parse(&["--locale", "it", "--skills-dir", "/tmp/skills/skills", "m.gguf", "e.jsonl"])
                .unwrap(),
            args("m.gguf", "e.jsonl", "it", Some("/tmp/skills/skills"))
        );
    }

    /// `--skills-dir`'s value must never be mistaken for a positional — a
    /// path swallowed as the gguf would fail confusingly, or worse, shift the
    /// eval path.
    #[test]
    fn skills_dir_value_is_not_treated_as_a_positional() {
        assert_eq!(
            parse(&["--skills-dir", "m.gguf", "e.jsonl"]).unwrap_err(),
            "usage: route-eval [--locale <xx>] [--skills-dir <path>] <gguf-path> <eval-jsonl-path>"
        );
    }
}
