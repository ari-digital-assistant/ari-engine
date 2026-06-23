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

fn dump_skill(skill: &dyn Skill) -> serde_json::Value {
    let examples: Vec<serde_json::Value> = skill
        .example_utterances()
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

fn main() {
    let skills: Vec<Box<dyn Skill>> = vec![
        Box::new(CurrentTimeSkill::new()),
        Box::new(DateSkill::new()),
        Box::new(CalculatorSkill::new()),
        Box::new(GreetingSkill::new()),
        Box::new(OpenSkill::new()),
        Box::new(SearchSkill::new()),
    ];

    let dump: Vec<serde_json::Value> = skills.iter().map(|s| dump_skill(s.as_ref())).collect();

    println!("{}", serde_json::to_string_pretty(&dump).unwrap());
}
