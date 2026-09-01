//! Dump every built-in skill's id, description, and example utterances
//! as a single JSON document on stdout. Used by the FunctionGemma
//! training pipeline to build the dataset from the canonical source —
//! the skills themselves — instead of a hardcoded copy.
//!
//! Schema (one entry per skill):
//!
//! ```json
//! [
//!   {
//!     "id": "current_time",
//!     "description": "Tells the current time...",
//!     "specificity": "high",
//!     "examples": [
//!       {"text": "what time is it", "args": {}},
//!       ...
//!     ]
//!   }
//! ]
//! ```
//!
//! The `args` field is parsed from the `args` JSON literal each
//! `ExampleUtterance` carries. Skills with parameterless examples
//! emit an empty object.

use ari_core::{Skill, Specificity};
use ari_skills::{
    CalculatorSkill, CurrentTimeSkill, DateSkill, GreetingSkill, OpenSkill, SearchSkill,
};
use serde_json::json;

fn specificity_str(s: Specificity) -> &'static str {
    match s {
        Specificity::High => "high",
        Specificity::Medium => "medium",
        Specificity::Low => "low",
    }
}

fn dump_skill(skill: &dyn Skill, locale: &str) -> serde_json::Value {
    let examples: Vec<serde_json::Value> = skill
        .example_utterances_for(locale)
        .iter()
        .map(|e| {
            let args: serde_json::Value = serde_json::from_str(e.args).unwrap_or(json!({}));
            json!({
                "text": e.text,
                "args": args,
            })
        })
        .collect();

    let parameters: serde_json::Value =
        serde_json::from_str(skill.parameters_schema()).unwrap_or(json!({}));

    json!({
        "id": skill.id(),
        "description": skill.description(),
        "specificity": specificity_str(skill.specificity()),
        "router_eligible": skill.router_eligible(),
        "parameters": parameters,
        "examples": examples,
    })
}

/// Build the export JSON for `locale`. Extracted from `main` so it is unit-testable.
fn run(locale: &str) -> String {
    let skills: Vec<Box<dyn Skill>> = vec![
        Box::new(CurrentTimeSkill::new()),
        Box::new(DateSkill::new()),
        Box::new(CalculatorSkill::new()),
        Box::new(GreetingSkill::new()),
        Box::new(OpenSkill::new()),
        Box::new(SearchSkill::new()),
    ];
    let dump: Vec<serde_json::Value> = skills
        .iter()
        .map(|s| dump_skill(s.as_ref(), locale))
        .collect();
    serde_json::to_string_pretty(&dump).unwrap()
}

fn main() {
    // Optional `--locale <xx>` flag; defaults to English.
    let mut locale = "en".to_string();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--locale" => {
                locale = args.next().unwrap_or_else(|| {
                    eprintln!("--locale requires a value");
                    std::process::exit(2);
                });
            }
            other => {
                eprintln!("unknown argument: {other}");
                std::process::exit(2);
            }
        }
    }
    println!("{}", run(&locale));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn en_export_is_valid_json_with_all_builtins() {
        let out = run("en");
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v.as_array().unwrap().len(), 6, "six built-in skills");
        assert!(out.contains("what time is it"), "english example present");
    }

    #[test]
    fn unlocalised_locale_falls_back_to_english() {
        // No built-in localises "fr", so the export is the English one
        // verbatim. ("it" is localised — see the Italian test below.)
        assert_eq!(run("fr"), run("en"));
    }

    #[test]
    fn it_export_uses_italian_examples_for_localised_builtins() {
        let v: serde_json::Value = serde_json::from_str(&run("it")).unwrap();
        let en_v: serde_json::Value = serde_json::from_str(&run("en")).unwrap();
        fn pick<'a>(doc: &'a serde_json::Value, id: &str) -> &'a Vec<serde_json::Value> {
            doc.as_array()
                .unwrap()
                .iter()
                .find(|s| s["id"] == id)
                .expect("skill present in export")["examples"]
                .as_array()
                .unwrap()
        }
        let examples = |id: &str| pick(&v, id);
        let examples_en = |id: &str| pick(&en_v, id);

        // Every router-eligible built-in exports its own Italian phrases.
        // Neither counts nor ordering are asserted: the phrase banks are
        // authored corpora that grow, and the export preserves declaration
        // order, so pinning either turns every corpus edit into a test edit.
        let has = |id: &str, text: &str| {
            examples(id).iter().any(|e| e["text"] == text)
        };
        assert!(has("current_time", "che ora è"));
        assert!(has("current_date", "che giorno è oggi"));
        assert!(has("greeting", "ciao"));

        // `open` and `calculator` carry args. The utterance is Italian but
        // the value is not: app_name feeds app resolution and expression
        // feeds the evaluator, and neither has a locale.
        let arg_for = |id: &str, text: &str| {
            examples(id)
                .iter()
                .find(|e| e["text"] == text)
                .map(|e| e["args"].clone())
                .unwrap_or(json!(null))
        };
        assert_eq!(arg_for("open", "apri la posta"), json!({"app_name": "Email"}));
        assert_eq!(
            arg_for("calculator", "quanto fa {n1} più {n2}"),
            json!({"expression": "{n1} + {n2}"})
        );

        // `search` is router_eligible=false and is deliberately never
        // localised — it stays English in every locale.
        assert_eq!(examples("search"), examples_en("search"));
    }

    #[test]
    fn export_declares_router_eligibility() {
        let out = run("en");
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        let pick = |id: &str| -> bool {
            v.as_array().unwrap().iter()
                .find(|s| s["id"] == id).expect("skill present in export")
                ["router_eligible"].as_bool().expect("router_eligible is a bool")
        };
        // search is keyword-only — router_catalog() filters it out, so the
        // trainer must be able to see that and exclude it from the corpus.
        assert!(!pick("search"), "search must declare router_eligible=false");
        assert!(pick("current_time"), "current_time must be router-eligible");
    }
}
