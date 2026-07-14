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
    let dump: Vec<serde_json::Value> =
        skills.iter().map(|s| dump_skill(s.as_ref(), locale)).collect();
    serde_json::to_string_pretty(&dump).unwrap()
}

fn main() {
    // Optional `--locale <xx>` flag; defaults to English.
    let mut locale = "en".to_string();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--locale" {
            locale = args.next().unwrap_or_else(|| {
                eprintln!("--locale requires a value");
                std::process::exit(2);
            });
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
        // No built-in overrides example_utterances_for yet (Plan 2), so an
        // unlocalised request returns the English export unchanged.
        assert_eq!(run("it"), run("en"));
    }
}
