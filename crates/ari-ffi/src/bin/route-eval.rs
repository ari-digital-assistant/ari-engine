//! Routing-eval promotion gate.
//!
//! Loads a candidate FunctionGemma GGUF as the live router, then routes every
//! case in a held-out eval set through the REAL routing path (the same
//! `Engine`, builtin catalogue, prompt builder, parser and confidence gate
//! production uses) and checks the pick against the expected outcome.
//!
//! Usage: `route-eval <gguf-path> <eval-jsonl-path>`
//!
//! Each eval line is JSON: `{"utterance": "...", "expect": "<skill_id>|NONE"}`
//! — `NONE` means the router must abstain (general-knowledge question that
//! belongs to the assistant). Exits non-zero when the abstention or positive
//! pass rate falls below threshold, which the training workflow uses to block
//! promotion of a regressed model.

// Abstention is the regression we care most about, so it's gated hard.
const ABSTAIN_MIN: f64 = 0.90;
const POSITIVE_MIN: f64 = 0.80;

#[cfg(feature = "llm")]
fn main() {
    use std::io::BufRead;

    let mut args = std::env::args().skip(1);
    let (gguf, eval_path) = match (args.next(), args.next()) {
        (Some(g), Some(e)) => (g, e),
        _ => {
            eprintln!("usage: route-eval <gguf-path> <eval-jsonl-path>");
            std::process::exit(2);
        }
    };

    let mut engine = ari_ffi::build_engine_with_builtins();
    let router = ari_llm::FunctionGemmaRouter::new(std::path::Path::new(&gguf));
    engine.set_router(Some(Box::new(router)));

    // ROUTE_EVAL_VERBOSE=1 dumps a tab-separated line per case
    // (category, confidence, expect, raw-pick, utterance) for confidence
    // distribution analysis.
    let verbose = std::env::var("ROUTE_EVAL_VERBOSE").is_ok();

    let file = std::fs::File::open(&eval_path)
        .unwrap_or_else(|e| panic!("cannot open eval set {eval_path}: {e}"));

    let (mut abstain_total, mut abstain_pass) = (0u32, 0u32);
    let (mut positive_total, mut positive_pass) = (0u32, 0u32);
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
    let positive_rate = rate(positive_pass, positive_total);

    if !failures.is_empty() {
        eprintln!("Failures:");
        for f in &failures {
            eprintln!("{f}");
        }
    }
    println!(
        "abstention: {abstain_pass}/{abstain_total} ({:.0}%, min {:.0}%)",
        abstain_rate * 100.0,
        ABSTAIN_MIN * 100.0
    );
    println!(
        "positive:   {positive_pass}/{positive_total} ({:.0}%, min {:.0}%)",
        positive_rate * 100.0,
        POSITIVE_MIN * 100.0
    );

    let ok = abstain_rate >= ABSTAIN_MIN && positive_rate >= POSITIVE_MIN;
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
