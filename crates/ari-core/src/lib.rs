use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Specificity {
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Response {
    Text(String),
    Action(serde_json::Value),
    Binary { mime: String, data: Vec<u8> },
}

pub struct SkillContext {
    pub locale: String,
}

impl Default for SkillContext {
    fn default() -> Self {
        Self {
            locale: "en".to_string(),
        }
    }
}

/// One example user utterance that should trigger a skill, paired with the
/// JSON arguments the function call should produce.
///
/// Used as training data for the FunctionGemma skill router. The router is
/// the optional second layer of skill matching: when the keyword/regex
/// scorer fails to find a match, the router (a small fine-tuned LLM) gets
/// a chance to pick a skill based on the user's intent rather than literal
/// keywords. The training data teaches it which natural-language phrasings
/// correspond to which skill.
///
/// `text` is the literal user utterance. `args` is a JSON object literal —
/// `"{}"` for parameterless skills, or `r#"{"app_name": "Spotify"}"#` for
/// parameterised ones. The args literal must be valid JSON; the export
/// pipeline parses it directly.
pub struct ExampleUtterance {
    pub text: &'static str,
    pub args: &'static str,
}

/// The core skill trait. Every skill — built-in Rust, declarative, WASM —
/// implements this at the engine boundary.
///
/// # The two layers of skill matching
///
/// 1. **Keyword scorer (always on, fast, free).** Reads `score()`. The
///    engine asks every skill "how confident are you about this input?",
///    runs three ranking rounds with specificity-based thresholds, and
///    executes the winner. This is the baseline that handles most
///    everyday utterances.
///
/// 2. **FunctionGemma router (optional, ~250MB on-device LLM).** Reads
///    `description()`, `parameters_schema()`, and `example_utterances()`.
///    Fires only when the keyword scorer found nothing. Catches
///    paraphrases the keyword patterns missed (e.g. "is it morning or
///    afternoon" routes to `current_time` even though "current_time"
///    doesn't appear in the input).
///
/// You always have to implement `score()` and `execute()`. The router
/// methods are optional but strongly recommended for built-in skills:
/// they cost nothing if the router is disabled, and they massively
/// improve coverage when it's enabled.
///
/// # Implementing for the router
///
/// - **`description()`** — write two sentences. First: what the skill
///   does. Second: when to use it, with semantic keywords. Example:
///   "Tells the current time. Use when the user asks what time it is,
///   what hour it is, whether it is morning or afternoon, or anything
///   about the current time of day." The router pattern-matches on
///   semantic similarity, so the more natural language you put in the
///   description, the better the routing.
///
/// - **`example_utterances()`** — return 20-30 varied phrasings. Cover
///   paraphrases, indirect language, conversational filler ("can you",
///   "please", "I need"). For parameterised skills, include the args
///   the model should produce. These feed directly into the
///   FunctionGemma fine-tuning dataset.
///
/// - **`parameters_schema()`** — for parameterised skills, override
///   this with an OpenAI-style JSON schema. Default is the
///   parameterless `{"type": "object", "properties": {}}`.
pub trait Skill: Send + Sync {
    /// Stable, unique identifier (e.g. `"current_time"`). This is what
    /// the router emits as the function name.
    fn id(&self) -> &str;

    /// Human-readable description. Critical for the FunctionGemma router
    /// — see the trait-level docs.
    fn description(&self) -> &str { "" }

    fn specificity(&self) -> Specificity;
    fn score(&self, input: &str, ctx: &SkillContext) -> f32;
    fn execute(&self, input: &str, ctx: &SkillContext) -> Response;

    /// Example user utterances that should trigger this skill, paired
    /// with the JSON arguments the function call should produce. Used as
    /// training data for the FunctionGemma router fine-tune. Skills that
    /// don't override this contribute nothing to training — keyword
    /// matching still works for them, but the router won't learn
    /// paraphrases for them.
    ///
    /// Aim for 20-30 varied phrasings. Cover paraphrases, indirect
    /// language, and conversational filler. The point is to teach the
    /// router that all the natural ways a user might phrase a request
    /// should land on this skill, not just the rigid ones the keyword
    /// patterns catch.
    fn example_utterances(&self) -> &[ExampleUtterance] { &[] }

    /// JSON schema describing this skill's parameters in OpenAI tool
    /// format. Used by the FunctionGemma router for both training data
    /// and inference. Default is `{"type": "object", "properties": {}}`
    /// for parameterless skills. Override for skills that take args.
    fn parameters_schema(&self) -> &'static str {
        r#"{"type": "object", "properties": {}}"#
    }
}

pub fn words_to_number(word: &str) -> Option<i64> {
    match word {
        "zero" => Some(0),
        "one" => Some(1),
        "two" => Some(2),
        "three" => Some(3),
        "four" => Some(4),
        "five" => Some(5),
        "six" => Some(6),
        "seven" => Some(7),
        "eight" => Some(8),
        "nine" => Some(9),
        "ten" => Some(10),
        "eleven" => Some(11),
        "twelve" => Some(12),
        "thirteen" => Some(13),
        "fourteen" => Some(14),
        "fifteen" => Some(15),
        "sixteen" => Some(16),
        "seventeen" => Some(17),
        "eighteen" => Some(18),
        "nineteen" => Some(19),
        "twenty" => Some(20),
        "thirty" => Some(30),
        "forty" => Some(40),
        "fifty" => Some(50),
        "sixty" => Some(60),
        "seventy" => Some(70),
        "eighty" => Some(80),
        "ninety" => Some(90),
        "hundred" => Some(100),
        "thousand" => Some(1000),
        "million" => Some(1_000_000),
        _ => None,
    }
}

pub fn parse_number_words(words: &[&str]) -> Option<(i64, usize)> {
    if words.is_empty() {
        return None;
    }

    // If first word is already a digit, skip this
    if words[0].parse::<i64>().is_ok() {
        return None;
    }

    let mut total: i64 = 0;
    let mut current: i64 = 0;
    let mut consumed = 0;
    let mut found_any = false;

    for word in words {
        // Handle hyphenated words like "twenty-five"
        let parts: Vec<&str> = word.split('-').collect();
        let mut matched_this_word = false;

        for part in &parts {
            if let Some(val) = words_to_number(part) {
                found_any = true;
                matched_this_word = true;
                match val {
                    1_000_000 => {
                        current = if current == 0 { val } else { current * val };
                        total += current;
                        current = 0;
                    }
                    1000 => {
                        current = if current == 0 { val } else { current * val };
                        total += current;
                        current = 0;
                    }
                    100 => {
                        current = if current == 0 { val } else { current * val };
                    }
                    _ => {
                        current += val;
                    }
                }
            }
        }

        if matched_this_word {
            consumed += 1;
        } else {
            break;
        }
    }

    if found_any {
        total += current;
        Some((total, consumed))
    } else {
        None
    }
}

pub fn replace_number_words(input: &str) -> String {
    let words: Vec<&str> = input.split_whitespace().collect();
    let mut result = Vec::new();
    let mut i = 0;

    while i < words.len() {
        if let Some((num, consumed)) = parse_number_words(&words[i..]) {
            result.push(num.to_string());
            i += consumed;
        } else {
            result.push(words[i].to_string());
            i += 1;
        }
    }

    result.join(" ")
}

pub fn normalize_input(input: &str) -> String {
    let lower = input.to_lowercase();

    let expanded = lower
        .replace("what's", "what is")
        .replace("whats", "what is")
        .replace("it's", "it is")
        .replace("i'm", "i am")
        .replace("don't", "do not")
        .replace("doesn't", "does not")
        .replace("can't", "cannot")
        .replace("won't", "will not")
        .replace("isn't", "is not")
        .replace("aren't", "are not")
        .replace("didn't", "did not")
        .replace("there's", "there is")
        .replace("here's", "here is")
        .replace("that's", "that is")
        .replace("let's", "let us");

    let cleaned: String = expanded
        .chars()
        .map(|c| if c.is_alphanumeric() || c.is_whitespace() || "+-*/.%^".contains(c) { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<&str>>()
        .join(" ");

    replace_number_words(&cleaned)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- normalize_input ---

    #[test]
    fn normalize_lowercases() {
        assert_eq!(normalize_input("HELLO World"), "hello world");
    }

    #[test]
    fn normalize_expands_all_contractions() {
        assert_eq!(normalize_input("what's"), "what is");
        assert_eq!(normalize_input("whats"), "what is");
        assert_eq!(normalize_input("it's"), "it is");
        assert_eq!(normalize_input("i'm"), "i am");
        assert_eq!(normalize_input("don't"), "do not");
        assert_eq!(normalize_input("doesn't"), "does not");
        assert_eq!(normalize_input("can't"), "cannot");
        assert_eq!(normalize_input("won't"), "will not");
        assert_eq!(normalize_input("isn't"), "is not");
        assert_eq!(normalize_input("aren't"), "are not");
        assert_eq!(normalize_input("didn't"), "did not");
        assert_eq!(normalize_input("there's"), "there is");
        assert_eq!(normalize_input("here's"), "here is");
        assert_eq!(normalize_input("that's"), "that is");
        assert_eq!(normalize_input("let's"), "let us");
    }

    #[test]
    fn normalize_strips_punctuation_keeps_math() {
        assert_eq!(normalize_input("hello, world!"), "hello world");
        assert_eq!(normalize_input("what?!"), "what");
        assert_eq!(normalize_input("2 + 2"), "2 + 2");
        assert_eq!(normalize_input("10 * 3.5"), "10 * 3.5");
        assert_eq!(normalize_input("5 % 3"), "5 % 3");
        assert_eq!(normalize_input("2^8"), "2^8");
        assert_eq!(normalize_input("(1 + 2)"), "1 + 2");
    }

    #[test]
    fn normalize_collapses_whitespace() {
        assert_eq!(normalize_input("  hello   world  "), "hello world");
        assert_eq!(normalize_input("\thello\tworld"), "hello world");
    }

    #[test]
    fn normalize_empty_and_whitespace() {
        assert_eq!(normalize_input(""), "");
        assert_eq!(normalize_input("   "), "");
        assert_eq!(normalize_input("!!!"), "");
    }

    #[test]
    fn normalize_combined_contraction_and_number() {
        assert_eq!(normalize_input("what's two plus three"), "what is 2 plus 3");
    }

    // --- words_to_number ---

    #[test]
    fn words_to_number_basics() {
        assert_eq!(words_to_number("zero"), Some(0));
        assert_eq!(words_to_number("one"), Some(1));
        assert_eq!(words_to_number("nineteen"), Some(19));
        assert_eq!(words_to_number("ninety"), Some(90));
        assert_eq!(words_to_number("hundred"), Some(100));
        assert_eq!(words_to_number("thousand"), Some(1000));
        assert_eq!(words_to_number("million"), Some(1_000_000));
    }

    #[test]
    fn words_to_number_rejects_non_numbers() {
        assert_eq!(words_to_number("hello"), None);
        assert_eq!(words_to_number(""), None);
        assert_eq!(words_to_number("42"), None);
    }

    // --- parse_number_words ---

    #[test]
    fn parse_simple_number() {
        assert_eq!(parse_number_words(&["five"]), Some((5, 1)));
        assert_eq!(parse_number_words(&["twenty"]), Some((20, 1)));
    }

    #[test]
    fn parse_compound_number() {
        // "twenty five" = 20 + 5 = 25, consumes 2 words
        assert_eq!(parse_number_words(&["twenty", "five"]), Some((25, 2)));
        // "one hundred" = 1 * 100 = 100, consumes 2 words
        assert_eq!(parse_number_words(&["one", "hundred"]), Some((100, 2)));
        // "three hundred forty two" = 3*100 + 40 + 2 = 342, consumes 4 words
        assert_eq!(parse_number_words(&["three", "hundred", "forty", "two"]), Some((342, 4)));
    }

    #[test]
    fn parse_stops_at_non_number_word() {
        // "five cats" should parse 5, consume 1 word, stop at "cats"
        assert_eq!(parse_number_words(&["five", "cats"]), Some((5, 1)));
    }

    #[test]
    fn parse_skips_digit_strings() {
        assert_eq!(parse_number_words(&["42"]), None);
        assert_eq!(parse_number_words(&["42", "five"]), None);
    }

    #[test]
    fn parse_empty_input() {
        assert_eq!(parse_number_words(&[]), None);
    }

    #[test]
    fn parse_non_number_input() {
        assert_eq!(parse_number_words(&["hello", "world"]), None);
    }

    #[test]
    fn parse_hyphenated_number() {
        assert_eq!(parse_number_words(&["twenty-five"]), Some((25, 1)));
    }

    #[test]
    fn parse_large_compound() {
        // "two thousand three hundred" = 2*1000 + 3*100 = 2300
        assert_eq!(
            parse_number_words(&["two", "thousand", "three", "hundred"]),
            Some((2300, 4))
        );
    }

    // --- replace_number_words ---

    #[test]
    fn replace_converts_scattered_numbers() {
        assert_eq!(replace_number_words("what is five times ten"), "what is 5 times 10");
    }

    #[test]
    fn replace_leaves_non_numbers_alone() {
        assert_eq!(replace_number_words("hello world"), "hello world");
    }

    #[test]
    fn replace_leaves_digit_strings_alone() {
        assert_eq!(replace_number_words("42 plus 8"), "42 plus 8");
    }

    #[test]
    fn replace_handles_adjacent_number_groups() {
        assert_eq!(replace_number_words("twenty plus thirty"), "20 plus 30");
    }
}
