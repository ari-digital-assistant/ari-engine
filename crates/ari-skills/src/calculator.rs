use ari_core::{ExampleUtterance, Response, Skill, SkillContext, Specificity};

// English + Italian trigger verbs. The `to_math_expr` step strips
// these from the input before passing the rest to the expression
// evaluator, so adding a trigger doesn't require parser changes.
const TRIGGER_WORDS: &[&str] = &[
    // English
    "calculate", "compute", "eval", "solve",
    // Italian: calcola (calculate), risolvi (solve)
    "calcola", "risolvi",
];

// Math word → operator, per locale. English and Italian only.
// Multi-word entries (e.g. "divided by") MUST come before their
// single-word prefixes are stripped — order matters because
// `replace` runs top to bottom.
const MATH_WORDS_EN: &[(&str, &str)] = &[
    ("multiplied by", "*"),
    ("divided by", "/"),
    ("to the power of", "^"),
    ("plus", "+"),
    ("minus", "-"),
    ("times", "*"),
    ("over", "/"),
    ("mod", "%"),
    ("squared", "^2"),
    ("cubed", "^3"),
];

const MATH_WORDS_IT: &[(&str, &str)] = &[
    // Percent family guards — "per" is a substring of these, and
    // percentage phrases aren't supported yet, so neutralise them to an
    // all-letters placeholder (stripped by the char filter → a non-
    // evaluating expression → graceful error) instead of letting the
    // "per" -> "*" rule below produce a wrong numeric answer.
    ("percentuale", "PCT"),
    ("per cento", "PCT"),
    ("percento", "PCT"),
    ("diviso per", "/"),
    ("elevato alla", "^"),
    ("più", "+"),
    ("meno", "-"),
    ("per", "*"),
    ("diviso", "/"),
    ("al quadrato", "^2"),
    ("al cubo", "^3"),
];

fn math_words(locale: &str) -> &'static [(&'static str, &'static str)] {
    match locale {
        "it" => MATH_WORDS_IT,
        _ => MATH_WORDS_EN,
    }
}

// Leading "what is" style phrases stripped before evaluation, per locale.
const LEADIN_PHRASES_EN: &[&str] = &["what is", "how much is"];
const LEADIN_PHRASES_IT: &[&str] = &["quanto fa", "quanto è", "quanto e"];

fn leadin_phrases(locale: &str) -> &'static [&'static str] {
    match locale {
        "it" => LEADIN_PHRASES_IT,
        _ => LEADIN_PHRASES_EN,
    }
}

pub struct CalculatorSkill;

impl CalculatorSkill {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CalculatorSkill {
    fn default() -> Self {
        Self::new()
    }
}

fn eval_expr(expr: &str) -> Option<f64> {
    let mut ns = fasteval::EmptyNamespace;
    fasteval::ez_eval(expr, &mut ns).ok()
}

fn to_math_expr(input: &str, locale: &str) -> String {
    let mut expr = input.to_string();

    for trigger in TRIGGER_WORDS {
        expr = expr.replace(trigger, "");
    }
    for phrase in leadin_phrases(locale) {
        expr = expr.replace(phrase, "");
    }

    for (word, op) in math_words(locale) {
        expr = expr.replace(word, op);
    }

    expr.chars()
        .filter(|c| c.is_ascii_digit() || "+-*/.%^() ".contains(*c))
        .collect::<String>()
        .trim()
        .to_string()
}

fn has_math_content(input: &str, locale: &str) -> bool {
    let has_digits = input.chars().any(|c| c.is_ascii_digit());
    let has_operators = input.chars().any(|c| "+-*/%^".contains(c))
        || math_words(locale).iter().any(|(word, _)| input.contains(word));
    has_digits && has_operators
}

impl Skill for CalculatorSkill {
    fn id(&self) -> &str {
        "calculator"
    }

    fn description(&self) -> &str {
        "Evaluates math expressions. Use when the user asks to calculate, compute, or figure out any mathematical expression, percentage, division, multiplication, addition, subtraction, or arithmetic."
    }

    fn specificity(&self) -> Specificity {
        Specificity::High
    }

    fn parameters_schema(&self) -> &str {
        r#"{"type": "object", "properties": {"expression": {"type": "string", "description": "The math expression to evaluate."}}, "required": ["expression"]}"#
    }

    fn example_utterances(&self) -> &[ExampleUtterance] {
        &[
            ExampleUtterance { text: "calculate 5 + 3", args: r#"{"expression": "5 + 3"}"# },
            ExampleUtterance { text: "what's 99 divided by 3", args: r#"{"expression": "99 / 3"}"# },
            ExampleUtterance { text: "how much is fifteen percent of two hundred", args: r#"{"expression": "15% of 200"}"# },
            ExampleUtterance { text: "compute 12 times 8", args: r#"{"expression": "12 * 8"}"# },
            ExampleUtterance { text: "what's 100 minus 37", args: r#"{"expression": "100 - 37"}"# },
            ExampleUtterance { text: "figure out 2 to the power of 10", args: r#"{"expression": "2^10"}"# },
            ExampleUtterance { text: "what is 144 divided by 12", args: r#"{"expression": "144 / 12"}"# },
            ExampleUtterance { text: "25 plus 75", args: r#"{"expression": "25 + 75"}"# },
            ExampleUtterance { text: "multiply 9 by 6", args: r#"{"expression": "9 * 6"}"# },
            ExampleUtterance { text: "what's the square root of 81", args: r#"{"expression": "sqrt(81)"}"# },
            ExampleUtterance { text: "how much is 20 percent of 50", args: r#"{"expression": "20% of 50"}"# },
            ExampleUtterance { text: "subtract 15 from 100", args: r#"{"expression": "100 - 15"}"# },
            ExampleUtterance { text: "what does 7 times 7 equal", args: r#"{"expression": "7 * 7"}"# },
            ExampleUtterance { text: "divide 200 by 8", args: r#"{"expression": "200 / 8"}"# },
            ExampleUtterance { text: "add 33 and 67", args: r#"{"expression": "33 + 67"}"# },
            ExampleUtterance { text: "what's 10 percent of 500", args: r#"{"expression": "10% of 500"}"# },
            ExampleUtterance { text: "calculate the sum of 14 and 28", args: r#"{"expression": "14 + 28"}"# },
            ExampleUtterance { text: "how much is 3.14 times 2", args: r#"{"expression": "3.14 * 2"}"# },
            ExampleUtterance { text: "what is 1000 divided by 7", args: r#"{"expression": "1000 / 7"}"# },
            ExampleUtterance { text: "compute 50 plus 50", args: r#"{"expression": "50 + 50"}"# },
            ExampleUtterance { text: "figure out 8 squared", args: r#"{"expression": "8^2"}"# },
            ExampleUtterance { text: "what's half of 246", args: r#"{"expression": "246 / 2"}"# },
            ExampleUtterance { text: "9 plus 10", args: r#"{"expression": "9 + 10"}"# },
            ExampleUtterance { text: "how much is a quarter of 80", args: r#"{"expression": "80 / 4"}"# },
            ExampleUtterance { text: "what's 5 factorial", args: r#"{"expression": "5!"}"# },
            ExampleUtterance { text: "calculate 999 minus 1", args: r#"{"expression": "999 - 1"}"# },
            ExampleUtterance { text: "what is 45 times 3", args: r#"{"expression": "45 * 3"}"# },
            ExampleUtterance { text: "18 divided by 3", args: r#"{"expression": "18 / 3"}"# },
            ExampleUtterance { text: "what's 75 plus 25", args: r#"{"expression": "75 + 25"}"# },
            ExampleUtterance { text: "do the math on 6 times 9", args: r#"{"expression": "6 * 9"}"# },
            // Paraphrases with implicit math intent — no calculate/compute/figure
            // trigger, just bare arithmetic phrasing the keyword scorer might miss.
            ExampleUtterance { text: "twenty three plus seventeen", args: r#"{"expression": "23 + 17"}"# },
            ExampleUtterance { text: "what would 12 multiplied by 7 give me", args: r#"{"expression": "12 * 7"}"# },
            ExampleUtterance { text: "I need the result of 200 minus 47", args: r#"{"expression": "200 - 47"}"# },
            ExampleUtterance { text: "give me 15 percent off 80", args: r#"{"expression": "80 - (80 * 15%)"}"# },
            ExampleUtterance { text: "what does 42 over 6 come to", args: r#"{"expression": "42 / 6"}"# },
            // Italian
            ExampleUtterance { text: "calcola 5 + 3", args: r#"{"expression": "5 + 3"}"# },
            ExampleUtterance { text: "quanto fa 99 diviso 3", args: r#"{"expression": "99 / 3"}"# },
            ExampleUtterance { text: "quanto fa 12 per 8", args: r#"{"expression": "12 * 8"}"# },
            ExampleUtterance { text: "calcola 100 meno 37", args: r#"{"expression": "100 - 37"}"# },
            ExampleUtterance { text: "quanto fa 25 più 75", args: r#"{"expression": "25 + 75"}"# },
        ]
    }

    fn score(&self, input: &str, ctx: &SkillContext) -> f32 {
        let has_trigger = TRIGGER_WORDS.iter().any(|t| input.contains(t));

        if has_trigger && has_math_content(input, ctx.locale.as_str()) {
            return 0.95;
        }

        if has_math_content(input, ctx.locale.as_str()) {
            let expr = to_math_expr(input, ctx.locale.as_str());
            if eval_expr(&expr).is_some() {
                return 0.85;
            }
        }

        if has_trigger {
            return 0.5;
        }

        0.0
    }

    fn execute(&self, input: &str, ctx: &SkillContext) -> Response {
        let expr = to_math_expr(input, ctx.locale.as_str());

        match eval_expr(&expr) {
            Some(result) => {
                if result.fract() == 0.0 && result.abs() < 1e15 {
                    Response::Text(format!("{}", result as i64))
                } else {
                    Response::Text(format!("{:.6}", result).trim_end_matches('0').trim_end_matches('.').to_string())
                }
            }
            None => Response::Text(
                match ctx.locale.as_str() {
                    "it" => "Mi spiace, non sono riuscito a calcolare quell'espressione.",
                    _ => "Sorry, I couldn't evaluate that expression.",
                }
                .to_string(),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> SkillContext {
        SkillContext::default()
    }

    fn exec(input: &str) -> String {
        let skill = CalculatorSkill::new();
        match skill.execute(input, &ctx()) {
            Response::Text(s) => s,
            other => panic!("expected Text, got {other:?}"),
        }
    }

    fn exec_it(input: &str) -> String {
        let skill = CalculatorSkill::new();
        let mut it = SkillContext::default();
        it.locale = "it".to_string();
        match skill.execute(input, &it) {
            Response::Text(s) => s,
            other => panic!("expected Text, got {other:?}"),
        }
    }

    // --- Scoring ---
    // trigger + math = 0.95, bare math that evaluates = 0.85,
    // trigger only = 0.5, nothing = 0.0

    #[test]
    fn score_trigger_plus_math() {
        let skill = CalculatorSkill::new();
        assert_eq!(skill.score("calculate 2 + 2", &ctx()), 0.95);
        assert_eq!(skill.score("compute 10 - 3", &ctx()), 0.95);
        assert_eq!(skill.score("solve 5 * 5", &ctx()), 0.95);
    }

    #[test]
    fn score_bare_evaluable_expression() {
        let skill = CalculatorSkill::new();
        assert_eq!(skill.score("2 + 2", &ctx()), 0.85);
        assert_eq!(skill.score("100 / 5", &ctx()), 0.85);
    }

    #[test]
    fn score_natural_language_math() {
        let skill = CalculatorSkill::new();
        // "what is 5 times 3" — has digits and "times" is a MATH_WORD
        assert_eq!(skill.score("what is 5 times 3", &ctx()), 0.85);
    }

    #[test]
    fn score_trigger_without_math() {
        let skill = CalculatorSkill::new();
        assert_eq!(skill.score("calculate something", &ctx()), 0.5);
    }

    #[test]
    fn score_zero_on_unrelated() {
        let skill = CalculatorSkill::new();
        assert_eq!(skill.score("hello there", &ctx()), 0.0);
        assert_eq!(skill.score("open spotify", &ctx()), 0.0);
    }

    // --- Basic operations ---

    #[test]
    fn addition() {
        assert_eq!(exec("2 + 2"), "4");
        assert_eq!(exec("0 + 0"), "0");
        assert_eq!(exec("999 + 1"), "1000");
    }

    #[test]
    fn subtraction() {
        assert_eq!(exec("10 - 3"), "7");
        assert_eq!(exec("5 - 5"), "0");
    }

    #[test]
    fn multiplication() {
        assert_eq!(exec("6 * 7"), "42");
        assert_eq!(exec("0 * 1000"), "0");
    }

    #[test]
    fn division() {
        assert_eq!(exec("10 / 2"), "5");
        assert_eq!(exec("10 / 3"), "3.333333");
    }

    #[test]
    fn exponentiation() {
        assert_eq!(exec("2 ^ 8"), "256");
        assert_eq!(exec("10 ^ 0"), "1");
    }

    // --- Natural language operations ---

    #[test]
    fn natural_language_plus() {
        assert_eq!(exec("what is 10 plus 5"), "15");
    }

    #[test]
    fn natural_language_minus() {
        assert_eq!(exec("what is 20 minus 7"), "13");
    }

    #[test]
    fn natural_language_times() {
        assert_eq!(exec("what is 10 times 5"), "50");
    }

    #[test]
    fn natural_language_divided_by() {
        assert_eq!(exec("what is 100 divided by 4"), "25");
    }

    // --- Integer vs decimal formatting ---

    #[test]
    fn integer_result_has_no_decimal() {
        assert_eq!(exec("4 + 4"), "8");
        // No trailing ".0"
        assert!(!exec("4 + 4").contains('.'));
    }

    #[test]
    fn decimal_result_trims_trailing_zeros() {
        assert_eq!(exec("1 / 2"), "0.5");
        assert_eq!(exec("1 / 4"), "0.25");
    }

    // --- Edge cases ---

    #[test]
    fn division_by_zero() {
        let result = exec("5 / 0");
        // fasteval returns Inf for division by zero; we format that
        // The important thing: it doesn't panic
        assert!(!result.is_empty());
    }

    #[test]
    fn invalid_expression() {
        assert_eq!(exec("plus plus"), "Sorry, I couldn't evaluate that expression.");
    }

    #[test]
    fn empty_after_stripping() {
        assert_eq!(exec("calculate"), "Sorry, I couldn't evaluate that expression.");
    }

    #[test]
    fn specificity_is_high() {
        assert_eq!(CalculatorSkill::new().specificity(), Specificity::High);
    }

    // --- to_math_expr ---

    #[test]
    fn to_math_expr_strips_trigger_words() {
        assert_eq!(to_math_expr("calculate 5 + 3", "en"), "5 + 3");
        assert_eq!(to_math_expr("compute 10 - 2", "en"), "10 - 2");
    }

    #[test]
    fn to_math_expr_converts_math_words() {
        assert_eq!(to_math_expr("5 plus 3", "en"), "5 + 3");
        assert_eq!(to_math_expr("10 minus 2", "en"), "10 - 2");
        assert_eq!(to_math_expr("4 times 3", "en"), "4 * 3");
        assert_eq!(to_math_expr("10 divided by 2", "en"), "10 / 2");
    }

    #[test]
    fn to_math_expr_strips_what_is() {
        assert_eq!(to_math_expr("what is 5 + 3", "en"), "5 + 3");
        assert_eq!(to_math_expr("how much is 10 * 2", "en"), "10 * 2");
    }

    // --- has_math_content ---

    #[test]
    fn has_math_content_true() {
        assert!(has_math_content("2 + 2", "en"));
        assert!(has_math_content("5 times 3", "en"));
    }

    #[test]
    fn has_math_content_false() {
        assert!(!has_math_content("hello", "en"));
        assert!(!has_math_content("2", "en"));
        assert!(!has_math_content("plus minus", "en"));
    }

    // --- Italian ---

    #[test]
    fn italian_natural_language_operations() {
        assert_eq!(exec_it("quanto fa 10 più 5"), "15");
        assert_eq!(exec_it("quanto fa 20 meno 7"), "13");
        assert_eq!(exec_it("calcola 6 per 7"), "42");
        assert_eq!(exec_it("quanto fa 100 diviso 4"), "25");
    }

    #[test]
    fn italian_error_message() {
        assert_eq!(
            exec_it("calcola"),
            "Mi spiace, non sono riuscito a calcolare quell'espressione."
        );
    }

    #[test]
    fn italian_trigger_scores() {
        let skill = CalculatorSkill::new();
        let mut it = SkillContext::default();
        it.locale = "it".to_string();
        assert_eq!(skill.score("calcola 2 + 2", &it), 0.95);
    }

    #[test]
    fn italian_percent_phrases_do_not_return_wrong_answer() {
        // Percentage phrases aren't supported; they must NOT yield a
        // wrong number via the "per" -> "*" substitution. Graceful
        // error is the correct behaviour.
        let err = "Mi spiace, non sono riuscito a calcolare quell'espressione.";
        assert_eq!(exec_it("quanto fa il 10 per cento di 200"), err);
        assert_eq!(exec_it("quanto fa il 10 percento di 200"), err);
        assert_eq!(exec_it("calcola il 20 percentuale di 50"), err);
    }

}

