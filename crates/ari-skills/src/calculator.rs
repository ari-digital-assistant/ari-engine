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

// Percent-family phrases, neutralised to an all-letters placeholder before
// the operator table runs: "per" is a substring of "per cento", and
// percentage phrases aren't supported yet, so the placeholder (stripped by
// the char filter → a non-evaluating expression → graceful error) beats
// letting the "per" -> "*" rule produce a wrong numeric answer.
//
// Deliberately NOT in MATH_WORDS_IT: `has_math_content` reads every entry
// there as evidence of arithmetic, and "abbassa le luci al 30 percento" is
// a lights command, not a sum.
const PERCENT_PHRASES_IT: &[&str] = &["percentuale", "per cento", "percento"];

fn percent_phrases(locale: &str) -> &'static [&'static str] {
    match locale {
        "it" => PERCENT_PHRASES_IT,
        _ => &[],
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

// Router training examples. Natural raw text as a user would actually
// say it, paired with the canonical expression the router should emit.
//
// The canonical value is locale-agnostic in every locale: it is the
// evaluator's own syntax (ASCII digits and operators), so word-form
// numbers and operations collapse to it -- "fifteen percent of two
// hundred" -> "15% of 200", "8 squared" -> "8^2". It feeds evaluation,
// not display, so it is never translated.
//
// NOTE: the last five English entries are Italian utterances that
// predate the per-locale split. They stay here verbatim so the `en`
// export is unchanged; CALCULATOR_EXAMPLES_IT is the localised list.
const CALCULATOR_EXAMPLES_EN: &[ExampleUtterance] = &[
    ExampleUtterance { text: "what is {n1} plus {n2}", args: r#"{"expression": "{n1} + {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "what is {n1} plus {n2}", args: r#"{"expression": "{n1} + {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "add {n1} and {n2}", args: r#"{"expression": "{n1} + {n2}"}"#, weight: 0.6 },
    ExampleUtterance { text: "{n1} plus {n2}", args: r#"{"expression": "{n1} + {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "calculate {n1} + {n2}", args: r#"{"expression": "{n1} + {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "what do {n1} and {n2} come to", args: r#"{"expression": "{n1} + {n2}"}"#, weight: 0.55 },
    ExampleUtterance { text: "what is the sum of {n1} and {n2}", args: r#"{"expression": "{n1} + {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "add up {n1} and {n2}", args: r#"{"expression": "{n1} + {n2}"}"#, weight: 0.6 },
    ExampleUtterance { text: "{n1} added to {n2}", args: r#"{"expression": "{n1} + {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "what is the total of {n1} and {n2}", args: r#"{"expression": "{n1} + {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "sum {n1} and {n2} for me", args: r#"{"expression": "{n1} + {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "how much is {n1} and {n2} together", args: r#"{"expression": "{n1} + {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "what is {n1} minus {n2}", args: r#"{"expression": "{n1} - {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "what is {n1} minus {n2}", args: r#"{"expression": "{n1} - {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "subtract {n2} from {n1}", args: r#"{"expression": "{n1} - {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "take {n2} away from {n1}", args: r#"{"expression": "{n1} - {n2}"}"#, weight: 0.75 },
    ExampleUtterance { text: "{n1} minus {n2}", args: r#"{"expression": "{n1} - {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "{n1} take away {n2}", args: r#"{"expression": "{n1} - {n2}"}"#, weight: 0.75 },
    ExampleUtterance { text: "what is {n1} less {n2}", args: r#"{"expression": "{n1} - {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "how much is {n1} minus {n2}", args: r#"{"expression": "{n1} - {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "what is the difference between {n1} and {n2}", args: r#"{"expression": "{n1} - {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "knock {n2} off {n1}", args: r#"{"expression": "{n1} - {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "lop {n2} off {n1}", args: r#"{"expression": "{n1} - {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "shave {n2} off {n1}", args: r#"{"expression": "{n1} - {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "take {n2} off {n1}", args: r#"{"expression": "{n1} - {n2}"}"#, weight: 0.6 },
    ExampleUtterance { text: "chop {n2} off {n1}", args: r#"{"expression": "{n1} - {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "trim {n2} off {n1}", args: r#"{"expression": "{n1} - {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "knock {n2} off of {n1}", args: r#"{"expression": "{n1} - {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "minus {n2} off {n1}", args: r#"{"expression": "{n1} - {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "deduct {n2} from {n1}", args: r#"{"expression": "{n1} - {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "what is {n1} times {n2}", args: r#"{"expression": "{n1} * {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "what is {n1} times {n2}", args: r#"{"expression": "{n1} * {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "multiply {n1} by {n2}", args: r#"{"expression": "{n1} * {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "{n1} times {n2}", args: r#"{"expression": "{n1} * {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "{n1} multiplied by {n2}", args: r#"{"expression": "{n1} * {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "compute {n1} * {n2}", args: r#"{"expression": "{n1} * {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "what is the product of {n1} and {n2}", args: r#"{"expression": "{n1} * {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "work out {n1} times {n2}", args: r#"{"expression": "{n1} * {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "what is {n1} divided by {n2}", args: r#"{"expression": "{n1} / {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "what is {n1} divided by {n2}", args: r#"{"expression": "{n1} / {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "divide {n1} by {n2}", args: r#"{"expression": "{n1} / {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "{n1} divided by {n2}", args: r#"{"expression": "{n1} / {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "{n1} over {n2}", args: r#"{"expression": "{n1} / {n2}"}"#, weight: 0.55 },
    ExampleUtterance { text: "how many times does {n2} go into {n1}", args: r#"{"expression": "{n1} / {n2}"}"#, weight: 0.6 },
    ExampleUtterance { text: "split {n1} into {n2}", args: r#"{"expression": "{n1} / {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "what is {n1} divided by {n2} please", args: r#"{"expression": "{n1} / {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "what is {n1} to the power of {exponent}", args: r#"{"expression": "{n1}^{exponent}"}"#, weight: 0.95 },
    ExampleUtterance { text: "{n1} to the power of {exponent}", args: r#"{"expression": "{n1}^{exponent}"}"#, weight: 0.95 },
    ExampleUtterance { text: "{n1} squared", args: r#"{"expression": "{n1}^2"}"#, weight: 0.95 },
    ExampleUtterance { text: "what is {n1} squared", args: r#"{"expression": "{n1}^2"}"#, weight: 0.95 },
    ExampleUtterance { text: "the square of {n1}", args: r#"{"expression": "{n1}^2"}"#, weight: 0.95 },
    ExampleUtterance { text: "{n1} cubed", args: r#"{"expression": "{n1}^3"}"#, weight: 0.95 },
    ExampleUtterance { text: "what is {n1} cubed", args: r#"{"expression": "{n1}^3"}"#, weight: 0.95 },
    ExampleUtterance { text: "what is the square root of {n1}", args: r#"{"expression": "sqrt({n1})"}"#, weight: 0.95 },
    ExampleUtterance { text: "square root of {n1}", args: r#"{"expression": "sqrt({n1})"}"#, weight: 0.95 },
    ExampleUtterance { text: "what is the square root of {n1}", args: r#"{"expression": "sqrt({n1})"}"#, weight: 0.95 },
    ExampleUtterance { text: "what is {percent} percent of {base}", args: r#"{"expression": "{percent}% of {base}"}"#, weight: 0.95 },
    ExampleUtterance { text: "what is {percent} percent of {base}", args: r#"{"expression": "{percent}% of {base}"}"#, weight: 0.95 },
    ExampleUtterance { text: "how much is {percent} percent of {base}", args: r#"{"expression": "{percent}% of {base}"}"#, weight: 0.75 },
    ExampleUtterance { text: "{percent} percent of {base}", args: r#"{"expression": "{percent}% of {base}"}"#, weight: 0.95 },
    ExampleUtterance { text: "calculate {percent}% of {base}", args: r#"{"expression": "{percent}% of {base}"}"#, weight: 0.95 },
    ExampleUtterance { text: "work out {percent} percent of {base}", args: r#"{"expression": "{percent}% of {base}"}"#, weight: 0.95 },
    ExampleUtterance { text: "what is {percent}% of {base}", args: r#"{"expression": "{percent}% of {base}"}"#, weight: 0.55 },
    ExampleUtterance { text: "give me {percent} percent of {base}", args: r#"{"expression": "{percent}% of {base}"}"#, weight: 0.75 },
    ExampleUtterance { text: "what is half of {base}", args: r#"{"expression": "{base} / 2"}"#, weight: 0.95 },
    ExampleUtterance { text: "half of {base}", args: r#"{"expression": "{base} / 2"}"#, weight: 0.95 },
    ExampleUtterance { text: "what is a 3 of {base}", args: r#"{"expression": "{base} / 3"}"#, weight: 0.95 },
    ExampleUtterance { text: "a 3 of {base}", args: r#"{"expression": "{base} / 3"}"#, weight: 0.95 },
    ExampleUtterance { text: "what is a quarter of {base}", args: r#"{"expression": "{base} / 4"}"#, weight: 0.95 },
    ExampleUtterance { text: "a quarter of {base}", args: r#"{"expression": "{base} / 4"}"#, weight: 0.95 },
    ExampleUtterance { text: "double {n1}", args: r#"{"expression": "{n1} * 2"}"#, weight: 0.95 },
    ExampleUtterance { text: "what is double {n1}", args: r#"{"expression": "{n1} * 2"}"#, weight: 0.95 },
    ExampleUtterance { text: "triple {n1}", args: r#"{"expression": "{n1} * 3"}"#, weight: 0.95 },
    ExampleUtterance { text: "what is triple {n1}", args: r#"{"expression": "{n1} * 3"}"#, weight: 0.95 },
    ExampleUtterance { text: "{n1} doubled", args: r#"{"expression": "{n1} * 2"}"#, weight: 0.95 },
    ExampleUtterance { text: "{n1} tripled", args: r#"{"expression": "{n1} * 3"}"#, weight: 0.95 },
    ExampleUtterance { text: "figure out {n1} - {n2}", args: r#"{"expression": "{n1} - {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "solve {n1} + {n2}", args: r#"{"expression": "{n1} + {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "knock {n2} off {n1} for me", args: r#"{"expression": "{n1} - {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "take {n2} off {n1} for me", args: r#"{"expression": "{n1} - {n2}"}"#, weight: 0.6 },
    ExampleUtterance { text: "shave {n2} off {n1} would you", args: r#"{"expression": "{n1} - {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "lop {n2} off {n1} for me", args: r#"{"expression": "{n1} - {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "knock {n2} off {n1} please", args: r#"{"expression": "{n1} - {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "can you knock {n2} off {n1}", args: r#"{"expression": "{n1} - {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "just knock {n2} off {n1}", args: r#"{"expression": "{n1} - {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "chop {n2} off {n1} for me", args: r#"{"expression": "{n1} - {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "take {n2} away from {n1} for me", args: r#"{"expression": "{n1} - {n2}"}"#, weight: 0.75 },
    ExampleUtterance { text: "can you take {n2} away from {n1}", args: r#"{"expression": "{n1} - {n2}"}"#, weight: 0.75 },
    ExampleUtterance { text: "what is {n1} take {n2} away", args: r#"{"expression": "{n1} - {n2}"}"#, weight: 0.75 },
    ExampleUtterance { text: "take away {n2} from {n1}", args: r#"{"expression": "{n1} - {n2}"}"#, weight: 0.75 },
    ExampleUtterance { text: "if i take {n2} away from {n1} what do i get", args: r#"{"expression": "{n1} - {n2}"}"#, weight: 0.6 },
    ExampleUtterance { text: "please take {n2} away from {n1}", args: r#"{"expression": "{n1} - {n2}"}"#, weight: 0.75 },
    ExampleUtterance { text: "work out {n1} take away {n2}", args: r#"{"expression": "{n1} - {n2}"}"#, weight: 0.85 },
    ExampleUtterance { text: "subtract {n2} from {n1} for me", args: r#"{"expression": "{n1} - {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "what do {n1} and {n2} make together", args: r#"{"expression": "{n1} + {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "put {n1} and {n2} together for me", args: r#"{"expression": "{n1} + {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "calculate {expression}", args: r#"{"expression": "{expression}"}"#, weight: 0.95 },
    ExampleUtterance { text: "what is 99 divided by 3", args: r#"{"expression": "99 / 3"}"#, weight: 0.75 },
    ExampleUtterance { text: "how much is 15 percent of 200", args: r#"{"expression": "15% of 200"}"#, weight: 0.85 },
    ExampleUtterance { text: "compute 12 times 8", args: r#"{"expression": "12 * 8"}"#, weight: 0.95 },
    ExampleUtterance { text: "what is 100 minus 37", args: r#"{"expression": "100 - 37"}"#, weight: 0.75 },
    ExampleUtterance { text: "figure out 2 to the power of 10", args: r#"{"expression": "2^10"}"#, weight: 0.95 },
    ExampleUtterance { text: "what is 144 divided by 12", args: r#"{"expression": "144 / 12"}"#, weight: 0.75 },
    ExampleUtterance { text: "25 plus 75", args: r#"{"expression": "25 + 75"}"#, weight: 0.85 },
    ExampleUtterance { text: "multiply 9 by 6", args: r#"{"expression": "9 * 6"}"#, weight: 0.95 },
    ExampleUtterance { text: "what is the square root of 81", args: r#"{"expression": "sqrt(81)"}"#, weight: 0.95 },
    ExampleUtterance { text: "how much is 20 percent of 50", args: r#"{"expression": "20% of 50"}"#, weight: 0.95 },
    ExampleUtterance { text: "subtract 15 from 100", args: r#"{"expression": "100 - 15"}"#, weight: 0.75 },
    ExampleUtterance { text: "what does 7 times 7 equal", args: r#"{"expression": "7 * 7"}"#, weight: 0.75 },
    ExampleUtterance { text: "divide 200 by 8", args: r#"{"expression": "200 / 8"}"#, weight: 0.85 },
    ExampleUtterance { text: "add 33 and 67", args: r#"{"expression": "33 + 67"}"#, weight: 0.6 },
    ExampleUtterance { text: "what is 10 percent of 500", args: r#"{"expression": "10% of 500"}"#, weight: 0.6 },
    ExampleUtterance { text: "calculate the sum of 14 and 28", args: r#"{"expression": "14 + 28"}"#, weight: 0.95 },
    ExampleUtterance { text: "how much is 3.14 times 2", args: r#"{"expression": "3.14 * 2"}"#, weight: 0.75 },
    ExampleUtterance { text: "what is 1000 divided by 7", args: r#"{"expression": "1000 / 7"}"#, weight: 0.85 },
    ExampleUtterance { text: "compute 50 plus 50", args: r#"{"expression": "50 + 50"}"#, weight: 0.95 },
    ExampleUtterance { text: "figure out 8 squared", args: r#"{"expression": "8^2"}"#, weight: 0.95 },
    ExampleUtterance { text: "what is half of 246", args: r#"{"expression": "246 / 2"}"#, weight: 0.75 },
    ExampleUtterance { text: "9 plus 10", args: r#"{"expression": "9 + 10"}"#, weight: 0.95 },
    ExampleUtterance { text: "how much is a quarter of 80", args: r#"{"expression": "80 / 4"}"#, weight: 0.95 },
    ExampleUtterance { text: "what is 5 factorial", args: r#"{"expression": "5!"}"#, weight: 0.6 },
    ExampleUtterance { text: "calculate 999 minus 1", args: r#"{"expression": "999 - 1"}"#, weight: 0.95 },
    ExampleUtterance { text: "what is 45 times 3", args: r#"{"expression": "45 * 3"}"#, weight: 0.75 },
    ExampleUtterance { text: "18 divided by 3", args: r#"{"expression": "18 / 3"}"#, weight: 0.85 },
    ExampleUtterance { text: "what is 75 plus 25", args: r#"{"expression": "75 + 25"}"#, weight: 0.75 },
    ExampleUtterance { text: "do the math on 6 times 9", args: r#"{"expression": "6 * 9"}"#, weight: 0.75 },
    ExampleUtterance { text: "23 plus 17", args: r#"{"expression": "23 + 17"}"#, weight: 0.85 },
    ExampleUtterance { text: "what would 12 multiplied by 7 give me", args: r#"{"expression": "12 * 7"}"#, weight: 0.75 },
    ExampleUtterance { text: "i need the result of 200 minus 47", args: r#"{"expression": "200 - 47"}"#, weight: 0.75 },
    ExampleUtterance { text: "give me 15 percent off 80", args: r#"{"expression": "80 - (80 * 15%)"}"#, weight: 0.85 },
    ExampleUtterance { text: "what does 42 over 6 come to", args: r#"{"expression": "42 / 6"}"#, weight: 0.6 },
    ExampleUtterance { text: "calcola {expression}", args: r#"{"expression": "{expression}"}"#, weight: 0.6 },
    ExampleUtterance { text: "quanto fa 12 per 8", args: r#"{"expression": "12 * 8"}"#, weight: 0.6 },
    ExampleUtterance { text: "calcola 100 meno 37", args: r#"{"expression": "100 - 37"}"#, weight: 0.6 },
    ExampleUtterance { text: "quanto fa 25 più 75", args: r#"{"expression": "25 + 75"}"#, weight: 0.6 },
];

// The same 40 intents in natural Italian — same operation spread (+ - * /
// powers, square root, factorial, percent, halves and quarters) and the
// same phrasing variety (quanto fa / calcola / imperatives / bare
// arithmetic), rather than a line-by-line translation of the English.
//
// The `expression` values are byte-for-byte the same canonical forms the
// English list uses: the evaluator has no locale. That includes the
// decimal point — "3,14" is how an Italian says and writes it, "3.14" is
// what the evaluator parses, so the router is taught that mapping.
const CALCULATOR_EXAMPLES_IT: &[ExampleUtterance] = &[
    ExampleUtterance { text: "quanto fa {n1} più {n2}", args: r#"{"expression": "{n1} + {n2}"}"#, weight: 0.6 },
    ExampleUtterance { text: "quant è {n1} più {n2}", args: r#"{"expression": "{n1} + {n2}"}"#, weight: 0.75 },
    ExampleUtterance { text: "{n1} più {n2}", args: r#"{"expression": "{n1} + {n2}"}"#, weight: 0.6 },
    ExampleUtterance { text: "somma {n1} e {n2}", args: r#"{"expression": "{n1} + {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "fammi la somma di {n1} e {n2}", args: r#"{"expression": "{n1} + {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "qual è la somma tra {n1} e {n2}", args: r#"{"expression": "{n1} + {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "aggiungi {n1} a {n2}", args: r#"{"expression": "{n2} + {n1}"}"#, weight: 0.6 },
    ExampleUtterance { text: "quanto viene {n1} più {n2}", args: r#"{"expression": "{n1} + {n2}"}"#, weight: 0.75 },
    ExampleUtterance { text: "calcolami {n1} più {n2}", args: r#"{"expression": "{n1} + {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "quanto fanno {n1} e {n2} insieme", args: r#"{"expression": "{n1} + {n2}"}"#, weight: 0.85 },
    ExampleUtterance { text: "metti insieme {n1} e {n2}", args: r#"{"expression": "{n1} + {n2}"}"#, weight: 0.75 },
    ExampleUtterance { text: "mi calcoli {n1} più {n2}", args: r#"{"expression": "{n1} + {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "quanto fa {n1} meno {n2}", args: r#"{"expression": "{n1} - {n2}"}"#, weight: 0.6 },
    ExampleUtterance { text: "quant è {n1} meno {n2}", args: r#"{"expression": "{n1} - {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "{n1} meno {n2}", args: r#"{"expression": "{n1} - {n2}"}"#, weight: 0.55 },
    ExampleUtterance { text: "togli {n2} da {n1}", args: r#"{"expression": "{n1} - {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "leva {n2} a {n1}", args: r#"{"expression": "{n1} - {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "sottrai {n2} da {n1}", args: r#"{"expression": "{n1} - {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "qual è la differenza tra {n1} e {n2}", args: r#"{"expression": "{n1} - {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "quanto viene {n1} meno {n2}", args: r#"{"expression": "{n1} - {n2}"}"#, weight: 0.75 },
    ExampleUtterance { text: "quanto resta se da {n1} togli {n2}", args: r#"{"expression": "{n1} - {n2}"}"#, weight: 0.75 },
    ExampleUtterance { text: "mi serve il risultato di {n1} meno {n2}", args: r#"{"expression": "{n1} - {n2}"}"#, weight: 0.75 },
    ExampleUtterance { text: "quanto fa {n1} per {n2}", args: r#"{"expression": "{n1} * {n2}"}"#, weight: 0.6 },
    ExampleUtterance { text: "quant è {n1} per {n2}", args: r#"{"expression": "{n1} * {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "{n1} per {n2}", args: r#"{"expression": "{n1} * {n2}"}"#, weight: 0.55 },
    ExampleUtterance { text: "moltiplica {n1} per {n2}", args: r#"{"expression": "{n1} * {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "calcola {n1} per {n2}", args: r#"{"expression": "{n1} * {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "{n1} volte {n2}", args: r#"{"expression": "{n1} * {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "qual è il prodotto di {n1} e {n2}", args: r#"{"expression": "{n1} * {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "quanto verrebbe {n1} moltiplicato per {n2}", args: r#"{"expression": "{n1} * {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "fammi {n1} per {n2}", args: r#"{"expression": "{n1} * {n2}"}"#, weight: 0.6 },
    ExampleUtterance { text: "quanto fanno {n1} per {n2}", args: r#"{"expression": "{n1} * {n2}"}"#, weight: 0.75 },
    ExampleUtterance { text: "fammi il conto di {n1} per {n2}", args: r#"{"expression": "{n1} * {n2}"}"#, weight: 0.6 },
    ExampleUtterance { text: "quanto fa {n1} diviso {n2}", args: r#"{"expression": "{n1} / {n2}"}"#, weight: 0.85 },
    ExampleUtterance { text: "quant è {n1} diviso {n2}", args: r#"{"expression": "{n1} / {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "{n1} diviso {n2}", args: r#"{"expression": "{n1} / {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "dividi {n1} per {n2}", args: r#"{"expression": "{n1} / {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "quanto viene {n1} diviso {n2}", args: r#"{"expression": "{n1} / {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "puoi calcolare {n1} diviso {n2}", args: r#"{"expression": "{n1} / {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "quante volte {n2} sta in {n1}", args: r#"{"expression": "{n1} / {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "{n1} fratto {n2}", args: r#"{"expression": "{n1} / {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "ho bisogno di sapere quanto fa {n1} diviso {n2}", args: r#"{"expression": "{n1} / {n2}"}"#, weight: 0.75 },
    ExampleUtterance { text: "quanto fa {n1} elevato a {exponent}", args: r#"{"expression": "{n1}^{exponent}"}"#, weight: 0.85 },
    ExampleUtterance { text: "{n1} elevato a {exponent}", args: r#"{"expression": "{n1}^{exponent}"}"#, weight: 0.95 },
    ExampleUtterance { text: "quanto fa {n1} al quadrato", args: r#"{"expression": "{n1}^2"}"#, weight: 0.85 },
    ExampleUtterance { text: "{n1} al quadrato", args: r#"{"expression": "{n1}^2"}"#, weight: 0.95 },
    ExampleUtterance { text: "quanto fa {n1} al cubo", args: r#"{"expression": "{n1}^3"}"#, weight: 0.85 },
    ExampleUtterance { text: "{n1} al cubo", args: r#"{"expression": "{n1}^3"}"#, weight: 0.95 },
    ExampleUtterance { text: "qual è la radice quadrata di {n1}", args: r#"{"expression": "sqrt({n1})"}"#, weight: 0.95 },
    ExampleUtterance { text: "la radice quadrata di {n1}", args: r#"{"expression": "sqrt({n1})"}"#, weight: 0.95 },
    ExampleUtterance { text: "quanto fa la radice di {n1}", args: r#"{"expression": "sqrt({n1})"}"#, weight: 0.85 },
    ExampleUtterance { text: "quanto fa il {percent} percento di {base}", args: r#"{"expression": "{percent}% of {base}"}"#, weight: 0.75 },
    ExampleUtterance { text: "quant è il {percent} per cento di {base}", args: r#"{"expression": "{percent}% of {base}"}"#, weight: 0.95 },
    ExampleUtterance { text: "il {percent} percento di {base}", args: r#"{"expression": "{percent}% of {base}"}"#, weight: 0.95 },
    ExampleUtterance { text: "calcola il {percent} percento di {base}", args: r#"{"expression": "{percent}% of {base}"}"#, weight: 0.95 },
    ExampleUtterance { text: "quanto viene il {percent} per cento di {base}", args: r#"{"expression": "{percent}% of {base}"}"#, weight: 0.95 },
    ExampleUtterance { text: "dammi il {percent} percento di {base}", args: r#"{"expression": "{percent}% of {base}"}"#, weight: 0.75 },
    ExampleUtterance { text: "togli il {percent} percento da {base}", args: r#"{"expression": "{base} - ({base} * {percent}%)"}"#, weight: 0.95 },
    ExampleUtterance { text: "quanto fa {base} meno il {percent} percento", args: r#"{"expression": "{base} - ({base} * {percent}%)"}"#, weight: 0.75 },
    ExampleUtterance { text: "aumenta {base} del {percent} percento", args: r#"{"expression": "{base} + ({base} * {percent}%)"}"#, weight: 0.95 },
    ExampleUtterance { text: "quanto fa la metà di {base}", args: r#"{"expression": "{base} / 2"}"#, weight: 0.75 },
    ExampleUtterance { text: "la metà di {base}", args: r#"{"expression": "{base} / 2"}"#, weight: 0.95 },
    ExampleUtterance { text: "un terzo di {base}", args: r#"{"expression": "{base} / 3"}"#, weight: 0.95 },
    ExampleUtterance { text: "quanto fa un terzo di {base}", args: r#"{"expression": "{base} / 3"}"#, weight: 0.85 },
    ExampleUtterance { text: "un quarto di {base}", args: r#"{"expression": "{base} / 4"}"#, weight: 0.95 },
    ExampleUtterance { text: "quanto fa un quarto di {base}", args: r#"{"expression": "{base} / 4"}"#, weight: 0.85 },
    ExampleUtterance { text: "il doppio di {n1}", args: r#"{"expression": "{n1} * 2"}"#, weight: 0.95 },
    ExampleUtterance { text: "raddoppia {n1}", args: r#"{"expression": "{n1} * 2"}"#, weight: 0.95 },
    ExampleUtterance { text: "il triplo di {n1}", args: r#"{"expression": "{n1} * 3"}"#, weight: 0.95 },
    ExampleUtterance { text: "triplica {n1}", args: r#"{"expression": "{n1} * 3"}"#, weight: 0.95 },
    ExampleUtterance { text: "quanto fanno {n1} e {n2} messi insieme", args: r#"{"expression": "{n1} + {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "quanto fanno {n1} e {n2} in totale", args: r#"{"expression": "{n1} + {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "{n1} e {n2} messi insieme quanto fanno", args: r#"{"expression": "{n1} + {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "quanto viene {n1} e {n2} in tutto", args: r#"{"expression": "{n1} + {n2}"}"#, weight: 0.75 },
    ExampleUtterance { text: "sommati {n1} e {n2} quanto fanno", args: r#"{"expression": "{n1} + {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "{n1} e {n2} sommati", args: r#"{"expression": "{n1} + {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "quanto danno {n1} e {n2} insieme", args: r#"{"expression": "{n1} + {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "levami {n2} da {n1}", args: r#"{"expression": "{n1} - {n2}"}"#, weight: 0.95 },
    ExampleUtterance { text: "quanto fa 5 più 3", args: r#"{"expression": "5 + 3"}"#, weight: 0.6 },
    ExampleUtterance { text: "quanto fa 99 diviso 3", args: r#"{"expression": "99 / 3"}"#, weight: 0.75 },
    ExampleUtterance { text: "quanto fa il quindici percento di duecento", args: r#"{"expression": "15% of 200"}"#, weight: 0.6 },
    ExampleUtterance { text: "calcola 12 per 8", args: r#"{"expression": "12 * 8"}"#, weight: 0.95 },
    ExampleUtterance { text: "quanto fa 100 meno 37", args: r#"{"expression": "100 - 37"}"#, weight: 0.6 },
    ExampleUtterance { text: "calcola 2 elevato alla decima", args: r#"{"expression": "2^10"}"#, weight: 0.95 },
    ExampleUtterance { text: "quanto fa 144 diviso 12", args: r#"{"expression": "144 / 12"}"#, weight: 0.75 },
    ExampleUtterance { text: "25 più 75", args: r#"{"expression": "25 + 75"}"#, weight: 0.6 },
    ExampleUtterance { text: "moltiplica 9 per 6", args: r#"{"expression": "9 * 6"}"#, weight: 0.95 },
    ExampleUtterance { text: "quanto fa la radice quadrata di 81", args: r#"{"expression": "sqrt(81)"}"#, weight: 0.85 },
    ExampleUtterance { text: "quanto fa il 20 per cento di 50", args: r#"{"expression": "20% of 50"}"#, weight: 0.75 },
    ExampleUtterance { text: "sottrai 15 da 100", args: r#"{"expression": "100 - 15"}"#, weight: 0.85 },
    ExampleUtterance { text: "quanto fa 7 per 7", args: r#"{"expression": "7 * 7"}"#, weight: 0.6 },
    ExampleUtterance { text: "dividi 200 per 8", args: r#"{"expression": "200 / 8"}"#, weight: 0.95 },
    ExampleUtterance { text: "somma 33 e 67", args: r#"{"expression": "33 + 67"}"#, weight: 0.85 },
    ExampleUtterance { text: "quanto fa il 10 percento di 500", args: r#"{"expression": "10% of 500"}"#, weight: 0.6 },
    ExampleUtterance { text: "calcola la somma di 14 e 28", args: r#"{"expression": "14 + 28"}"#, weight: 0.95 },
    ExampleUtterance { text: "quanto fa 3,14 per 2", args: r#"{"expression": "3.14 * 2"}"#, weight: 0.6 },
    ExampleUtterance { text: "quanto fa 1000 diviso 7", args: r#"{"expression": "1000 / 7"}"#, weight: 0.75 },
    ExampleUtterance { text: "calcola 50 più 50", args: r#"{"expression": "50 + 50"}"#, weight: 0.75 },
    ExampleUtterance { text: "quanto fa 8 al quadrato", args: r#"{"expression": "8^2"}"#, weight: 0.85 },
    ExampleUtterance { text: "quanto fa la metà di 246", args: r#"{"expression": "246 / 2"}"#, weight: 0.6 },
    ExampleUtterance { text: "9 più 10", args: r#"{"expression": "9 + 10"}"#, weight: 0.6 },
    ExampleUtterance { text: "quanto fa un quarto di 80", args: r#"{"expression": "80 / 4"}"#, weight: 0.75 },
    ExampleUtterance { text: "quanto fa il fattoriale di 5", args: r#"{"expression": "5!"}"#, weight: 0.6 },
    ExampleUtterance { text: "calcola 999 meno 1", args: r#"{"expression": "999 - 1"}"#, weight: 0.95 },
    ExampleUtterance { text: "quanto fa 45 per 3", args: r#"{"expression": "45 * 3"}"#, weight: 0.6 },
    ExampleUtterance { text: "18 diviso 3", args: r#"{"expression": "18 / 3"}"#, weight: 0.95 },
    ExampleUtterance { text: "quanto fa 75 più 25", args: r#"{"expression": "75 + 25"}"#, weight: 0.6 },
    ExampleUtterance { text: "fammi 6 per 9", args: r#"{"expression": "6 * 9"}"#, weight: 0.6 },
    ExampleUtterance { text: "ventitré più diciassette", args: r#"{"expression": "23 + 17"}"#, weight: 0.6 },
    ExampleUtterance { text: "quanto verrebbe 12 moltiplicato per 7", args: r#"{"expression": "12 * 7"}"#, weight: 0.95 },
    ExampleUtterance { text: "mi serve il risultato di 200 meno 47", args: r#"{"expression": "200 - 47"}"#, weight: 0.6 },
    ExampleUtterance { text: "togli il 15 percento da 80", args: r#"{"expression": "80 - (80 * 15%)"}"#, weight: 0.75 },
    ExampleUtterance { text: "quanto viene 42 diviso 6", args: r#"{"expression": "42 / 6"}"#, weight: 0.85 },
    ExampleUtterance { text: "puoi calcolare 250 diviso 5", args: r#"{"expression": "250 / 5"}"#, weight: 0.95 },
    ExampleUtterance { text: "qual è la radice quadrata di 144", args: r#"{"expression": "sqrt(144)"}"#, weight: 0.95 },
    ExampleUtterance { text: "quanto fa 3 elevato alla quarta", args: r#"{"expression": "3^4"}"#, weight: 0.75 },
    ExampleUtterance { text: "aggiungi 40 a 60", args: r#"{"expression": "40 + 60"}"#, weight: 0.85 },
    ExampleUtterance { text: "quanto fa cinquanta meno dodici", args: r#"{"expression": "50 - 12"}"#, weight: 0.6 },
];

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

// fasteval has no `sqrt`, and the router's canonical expressions use it
// ("what's the square root of 81" -> `sqrt(81)`). Supplying it as a
// namespace lookup is cheaper than rewriting the expression, and every
// other name still resolves to `None`, so a bogus expression fails
// instead of silently evaluating.
fn eval_expr(expr: &str) -> Option<f64> {
    let mut ns = |name: &str, args: Vec<f64>| -> Option<f64> {
        match (name, args.as_slice()) {
            ("sqrt", [x]) => Some(x.sqrt()),
            _ => None,
        }
    };
    fasteval::ez_eval(expr, &mut ns).ok()
}

fn format_result(result: f64) -> String {
    if result.fract() == 0.0 && result.abs() < 1e15 {
        format!("{}", result as i64)
    } else {
        format!("{:.6}", result)
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string()
    }
}

// Italian writes decimals with a comma. The char filter below drops it, so
// "3,14 per 2" would evaluate as "3 14 * 2" — convert to the evaluator's
// point first. Only a digit-flanked comma qualifies; anything else is
// sentence punctuation and stays dropped.
fn italian_decimal_point(expr: &str) -> String {
    let chars: Vec<char> = expr.chars().collect();
    chars
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            let decimal = c == ','
                && i > 0
                && chars[i - 1].is_ascii_digit()
                && chars.get(i + 1).is_some_and(|n| n.is_ascii_digit());
            if decimal { '.' } else { c }
        })
        .collect()
}

// Italian "per" is "times" in "12 per 8" but "by" in "dividi 200 per 8".
// The imperative is the case the word table can't see — "diviso per" is
// already a single entry — so rewrite that "per" to the division operator
// before the "per" -> "*" rule reaches it.
fn italian_division_by(expr: &str) -> String {
    const VERB: &str = "dividi";
    const BY: &str = " per ";

    let Some(after_verb) = expr.find(VERB).map(|i| i + VERB.len()) else {
        return expr.to_string();
    };
    match expr[after_verb..].find(BY) {
        Some(offset) => {
            let at = after_verb + offset;
            format!("{} / {}", &expr[..at], &expr[at + BY.len()..])
        }
        None => expr.to_string(),
    }
}

fn to_math_expr(input: &str, locale: &str) -> String {
    let mut expr = input.to_string();

    if locale == "it" {
        expr = italian_decimal_point(&expr);
        expr = italian_division_by(&expr);
    }

    for trigger in TRIGGER_WORDS {
        expr = expr.replace(trigger, "");
    }
    for phrase in leadin_phrases(locale) {
        expr = expr.replace(phrase, "");
    }
    for phrase in percent_phrases(locale) {
        expr = expr.replace(phrase, "PCT");
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

// Whole-word containment. A bare substring check reads the Italian
// operator "per" inside "percento", "aperto", "persona" and friends —
// which is how "abbassa le luci al 30 percento" became a sum.
fn contains_word(haystack: &str, needle: &str) -> bool {
    haystack.match_indices(needle).any(|(i, m)| {
        let before = haystack[..i].chars().next_back();
        let after = haystack[i + m.len()..].chars().next();
        !before.is_some_and(char::is_alphanumeric) && !after.is_some_and(char::is_alphanumeric)
    })
}

fn has_math_content(input: &str, locale: &str) -> bool {
    // Percentages are not an operation this skill performs, and the
    // Italian ones spell "per cento" with the multiplication word in it.
    // Drop them before looking for operators, or every "al 30 percento"
    // command in the house reads as arithmetic.
    let input = percent_phrases(locale)
        .iter()
        .fold(input.to_string(), |acc, phrase| acc.replace(phrase, " "));

    let has_digits = input.chars().any(|c| c.is_ascii_digit());
    let has_operators = input.chars().any(|c| "+-*/%^".contains(c))
        || math_words(locale).iter().any(|(word, _)| contains_word(&input, word));
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
        CALCULATOR_EXAMPLES_EN
    }

    fn example_utterances_for(&self, locale: &str) -> &[ExampleUtterance] {
        match locale {
            "it" => CALCULATOR_EXAMPLES_IT,
            _ => CALCULATOR_EXAMPLES_EN,
        }
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
            Some(result) => Response::Text(format_result(result)),
            None => Response::Text(
                match ctx.locale.as_str() {
                    "it" => "Mi spiace, non sono riuscito a calcolare quell'espressione.",
                    _ => "Sorry, I couldn't evaluate that expression.",
                }
                .to_string(),
            ),
        }
    }

    /// Typed-args path. The router extracts a canonical `expression` in
    /// the evaluator's own syntax, which carries operations the word
    /// tables can't express — "what's the square root of 81" arrives as
    /// `sqrt(81)`, where re-parsing the utterance leaves only `81`.
    /// Falls back to `execute` when the slot is missing or doesn't
    /// evaluate, so an unsupported canonical form (`15% of 200`) is no
    /// worse off than before.
    fn execute_with_args(
        &self,
        input: &str,
        args_json: &str,
        ctx: &SkillContext,
    ) -> Response {
        let result = serde_json::from_str::<serde_json::Value>(args_json)
            .ok()
            .and_then(|v| v.get("expression").and_then(|e| e.as_str()).map(String::from))
            .and_then(|e| eval_expr(e.trim()));

        match result {
            Some(value) => Response::Text(format_result(value)),
            None => self.execute(input, ctx),
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

    #[test]
    fn has_math_content_needs_a_whole_operator_word() {
        // Operator words embedded in longer words are not operators.
        assert!(!has_math_content("3 sometimes", "en"));
        assert!(!has_math_content("discover 4 things", "en"));
        assert!(!has_math_content("2 modern chairs", "en"));
        assert!(!has_math_content("4 persone aperte", "it"));
        assert!(has_math_content("3 times 4", "en"));
        assert!(has_math_content("3 per 4", "it"));
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
    fn italian_router_examples() {
        let skill = CalculatorSkill::new();
        let it = skill.example_utterances_for("it");
        let en = skill.example_utterances_for("en");
        assert!(
            !it.iter().any(|e| en.iter().any(|x| x.text == e.text)),
            "an English phrase leaked into the Italian arm"
        );
        assert_ne!(it, en, "Italian examples are distinct from English");
        assert!(it.iter().any(|e| e.text == "quanto fa {n1} più {n2}"
                && e.args == r#"{"expression": "{n1} + {n2}"}"#),
            "Italian phrase maps its slots onto the evaluator expression");
        assert!(it.iter().all(|e| e.args.contains("expression")), "every calculator example supplies expression");
        assert!(en.iter().any(|e| e.text == "what is {n1} plus {n2}"), "English arm is parametrised too");
        assert_eq!(skill.example_utterances_for("fr"), en, "unknown locale falls back to English");
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

    #[test]
    fn italian_percent_phrase_is_not_math_content() {
        // "abbassa le luci del corridoio al 30 percento" is a lights
        // command. When the percent guards lived in MATH_WORDS_IT the
        // calculator scored it 0.85 at High specificity and won round 0,
        // answering "30" to someone dimming their hallway.
        let skill = CalculatorSkill::new();
        let mut it = SkillContext::default();
        it.locale = "it".to_string();

        assert!(!has_math_content("abbassa le luci del corridoio al 30 percento", "it"));
        assert_eq!(skill.score("abbassa le luci del corridoio al 30 percento", &it), 0.0);
        assert_eq!(skill.score("metti il volume al 50 per cento", &it), 0.0);
        // A trigger word still scores, but only the bare trigger score —
        // a percentage is not the math content that earns 0.95.
        assert_eq!(skill.score("calcola il 20 percentuale di 50", &it), 0.5);
        // Real arithmetic in the same locale is untouched.
        assert_eq!(skill.score("quanto fa 12 per 8", &it), 0.85);
    }

    #[test]
    fn italian_per_after_a_division_verb_divides() {
        // "dividi 200 per 8" is 25. The "per" -> "*" rule answered 1600.
        assert_eq!(exec_it("dividi 200 per 8"), "25");
        assert_eq!(exec_it("dividi 90 per 3"), "30");
        // "per" as multiplication is unaffected when no division verb leads.
        assert_eq!(exec_it("moltiplica 9 per 6"), "54");
        assert_eq!(exec_it("quanto fa 12 per 8"), "96");
        // The pre-existing "diviso per" table entry still resolves.
        assert_eq!(exec_it("quanto fa 99 diviso per 3"), "33");
    }

    #[test]
    fn italian_decimal_comma_is_a_decimal_point() {
        // Italian writes 3,14. Stripping the comma made it "3 14".
        assert_eq!(exec_it("quanto fa 3,14 per 2"), "6.28");
        assert_eq!(exec_it("quanto fa 1,5 più 2,5"), "4");
        assert_eq!(italian_decimal_point("3,14 per 2"), "3.14 per 2");
        // A comma that isn't between digits is not a decimal point.
        assert_eq!(italian_decimal_point("ciao, quanto fa 2 più 2"), "ciao, quanto fa 2 più 2");
    }

    #[test]
    fn execute_with_args_uses_the_routers_expression() {
        // The word tables can't express a square root, so re-parsing the
        // utterance yielded "81". The router's canonical form can.
        let skill = CalculatorSkill::new();
        match skill.execute_with_args(
            "what is the square root of 81",
            r#"{"expression": "sqrt(81)"}"#,
            &ctx(),
        ) {
            Response::Text(s) => assert_eq!(s, "9"),
            other => panic!("expected Text, got {other:?}"),
        }
        // Without args the same utterance still answers wrongly — proof
        // the fix is the args path, not a change to the parser.
        assert_eq!(exec("what is the square root of 81"), "81");
    }

    #[test]
    fn execute_with_args_falls_back_when_the_expression_is_unusable() {
        let skill = CalculatorSkill::new();
        let fallback = |args: &str| match skill.execute_with_args("what is 2 plus 2", args, &ctx()) {
            Response::Text(s) => s,
            other => panic!("expected Text, got {other:?}"),
        };
        assert_eq!(fallback("{}"), "4", "missing slot falls back to the utterance");
        assert_eq!(fallback("not json"), "4", "unparseable args fall back");
        assert_eq!(fallback(r#"{"expression": "15% of 200"}"#), "4",
            "a canonical form fasteval cannot evaluate falls back");
    }

    #[test]
    fn execute_with_args_is_locale_agnostic() {
        // `expression` is the evaluator's syntax in every locale, so the
        // Italian path must not re-interpret it.
        let skill = CalculatorSkill::new();
        let mut it = SkillContext::default();
        it.locale = "it".to_string();
        match skill.execute_with_args("quanto fa 3,14 per 2", r#"{"expression": "3.14 * 2"}"#, &it) {
            Response::Text(s) => assert_eq!(s, "6.28"),
            other => panic!("expected Text, got {other:?}"),
        }
    }
}

