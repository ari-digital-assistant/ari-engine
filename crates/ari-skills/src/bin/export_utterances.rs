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

        // All five router-eligible built-ins lead with their canonical
        // Italian phrasing and keep count parity with English.
        assert_eq!(examples("current_time").len(), 29);
        assert_eq!(examples("current_time")[0]["text"], "che ora è");
        assert_eq!(examples("current_time")[0]["args"], json!({}));
        assert_eq!(examples("current_date").len(), 30);
        assert_eq!(examples("current_date")[0]["text"], "che giorno è oggi");
        assert_eq!(examples("greeting").len(), 30);
        assert_eq!(examples("greeting")[0]["text"], "ciao");
        // `open` is the first localised skill carrying args: the Italian
        // text is translated but the app_name value stays canonical
        // English, because it feeds app resolution rather than display.
        assert_eq!(examples("open").len(), 40);
        assert_eq!(examples("open")[0]["text"], "apri spotify");
        assert_eq!(examples("open")[0]["args"], json!({"app_name": "Spotify"}));
        assert_eq!(examples("open")[1]["text"], "apri la fotocamera");
        assert_eq!(examples("open")[1]["args"], json!({"app_name": "Camera"}));
        // `calculator` carries args whose value is canonical evaluator
        // syntax: the utterance is Italian, the expression is not, because
        // the evaluator has no locale.
        assert_eq!(examples("calculator").len(), 40);
        assert_eq!(examples("calculator")[0]["text"], "quanto fa 5 più 3");
        assert_eq!(
            examples("calculator")[0]["args"],
            json!({"expression": "5 + 3"})
        );
        assert_eq!(
            examples("calculator")[2]["text"],
            "quanto fa il quindici percento di duecento"
        );
        assert_eq!(
            examples("calculator")[2]["args"],
            json!({"expression": "15% of 200"})
        );

        // `search` is router_eligible=false and is deliberately never
        // localised — it stays English in every locale.
        assert_eq!(examples("search"), examples_en("search"));
    }
}
