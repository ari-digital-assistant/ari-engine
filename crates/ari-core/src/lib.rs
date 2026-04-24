use serde::{Deserialize, Serialize};

// ── Skill router ──────────────────────────────────────────────────────

/// What the skill router decided.
pub enum RouteResult {
    /// Route to a registered skill by id.
    Skill(String),
    /// Route to a system action (Android intent). The JSON value carries
    /// the action type and parameters for the frontend to dispatch.
    Action(serde_json::Value),
    /// No match — fall through to the assistant.
    NoMatch,
}

/// Trait for an LLM-based skill router that runs after the keyword
/// matcher fails. The router sees the user input and the list of
/// available skills, and either picks one, suggests a system action,
/// or declines.
pub trait SkillRouter: Send + Sync {
    fn route(
        &self,
        input: &str,
        skills: &[(String, String)],
    ) -> RouteResult;
}

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

#[derive(Clone)]
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

    /// Resume skill execution after a Layer C assistant round-trip
    /// (see the `consult_assistant` envelope primitive). The engine
    /// calls this from a background thread once the assistant has
    /// replied.
    ///
    /// `context` is the opaque string the skill previously put in the
    /// `consult_assistant.continuation_context` field — the skill uses
    /// it to carry state (original utterance, settings snapshot, etc.)
    /// into this second invocation. `assistant_response` is the raw
    /// text returned by the assistant.
    ///
    /// Default implementation wraps the arguments in the reserved
    /// `{"_ari_continuation": {...}}` JSON shape and routes through
    /// [`execute`]. This bypasses `normalize_input` — the engine calls
    /// `execute_continuation` directly, not via keyword routing —
    /// so the skill's dispatch function can pattern-match on the
    /// JSON prefix and fork to a continuation handler. Skills that
    /// prefer an explicit second entry-point can override this.
    fn execute_continuation(
        &self,
        context: &str,
        assistant_response: &str,
        ctx: &SkillContext,
    ) -> Response {
        let payload = serde_json::json!({
            "_ari_continuation": {
                "context": context,
                "response": assistant_response,
            }
        });
        self.execute(&payload.to_string(), ctx)
    }
}

pub fn words_to_number(word: &str) -> Option<i64> {
    match word {
        "zero" => Some(0),
        "one" | "first" => Some(1),
        "two" | "second" => Some(2),
        "three" | "third" => Some(3),
        "four" | "fourth" => Some(4),
        "five" | "fifth" => Some(5),
        "six" | "sixth" => Some(6),
        "seven" | "seventh" => Some(7),
        "eight" | "eighth" => Some(8),
        "nine" | "ninth" => Some(9),
        "ten" | "tenth" => Some(10),
        "eleven" | "eleventh" => Some(11),
        "twelve" | "twelfth" => Some(12),
        "thirteen" | "thirteenth" => Some(13),
        "fourteen" | "fourteenth" => Some(14),
        "fifteen" | "fifteenth" => Some(15),
        "sixteen" | "sixteenth" => Some(16),
        "seventeen" | "seventeenth" => Some(17),
        "eighteen" | "eighteenth" => Some(18),
        "nineteen" | "nineteenth" => Some(19),
        "twenty" | "twentieth" => Some(20),
        "thirty" | "thirtieth" => Some(30),
        "forty" | "fortieth" => Some(40),
        "fifty" | "fiftieth" => Some(50),
        "sixty" | "sixtieth" => Some(60),
        "seventy" | "seventieth" => Some(70),
        "eighty" | "eightieth" => Some(80),
        "ninety" | "ninetieth" => Some(90),
        "hundred" | "hundredth" => Some(100),
        "thousand" | "thousandth" => Some(1000),
        "million" | "millionth" => Some(1_000_000),
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
        // Handle hyphenated words like "twenty-five". Apply the whole
        // word tentatively so that a partially-invalid hyphenated word
        // (e.g. "nine-thirty") gets rejected atomically rather than
        // leaving half-mutated state.
        let parts: Vec<&str> = word.split('-').collect();
        let mut t_total = total;
        let mut t_current = current;
        let mut word_ok = false;

        for part in &parts {
            let Some(val) = words_to_number(part) else {
                word_ok = false;
                break;
            };
            word_ok = true;
            match val {
                1_000_000 => {
                    t_current = if t_current == 0 { val } else { t_current * val };
                    t_total += t_current;
                    t_current = 0;
                }
                1000 => {
                    t_current = if t_current == 0 { val } else { t_current * val };
                    t_total += t_current;
                    t_current = 0;
                }
                100 => {
                    t_current = if t_current == 0 { val } else { t_current * val };
                }
                _ => {
                    // English permits exactly one sub-hundred compound:
                    // tens (20..=90 step 10) + ones (1..=9), e.g.
                    // "twenty-five" = 25. Anything else ("nine thirty",
                    // "five six", "ten five") is two separate numbers —
                    // a clock time or adjacent numerals — and must not
                    // be summed into a single value.
                    let sub = t_current % 100;
                    let is_tens_ones_compound =
                        matches!(sub, 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90)
                            && (1..=9).contains(&val);
                    if sub != 0 && !is_tens_ones_compound {
                        word_ok = false;
                        break;
                    }
                    t_current += val;
                }
            }
        }

        if !word_ok {
            break;
        }
        total = t_total;
        current = t_current;
        found_any = true;
        consumed += 1;
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

    // Regression: "nine thirty" is a clock time, not a compound number.
    // The greedy additive parser used to fold it into 39.
    #[test]
    fn replace_keeps_clock_time_as_two_numbers() {
        assert_eq!(replace_number_words("at nine thirty"), "at 9 30");
        assert_eq!(
            replace_number_words("remind me to take out the trash at nine thirty"),
            "remind me to take out the trash at 9 30",
        );
    }

    #[test]
    fn replace_does_not_merge_ones_and_ones() {
        assert_eq!(replace_number_words("five six seven"), "5 6 7");
    }

    #[test]
    fn replace_does_not_merge_teens_and_ones() {
        assert_eq!(replace_number_words("ten five"), "10 5");
    }

    #[test]
    fn replace_preserves_valid_tens_ones_compound() {
        assert_eq!(replace_number_words("twenty five apples"), "25 apples");
        assert_eq!(replace_number_words("thirty two"), "32");
    }

    #[test]
    fn replace_preserves_hundred_tens_ones() {
        assert_eq!(replace_number_words("two hundred thirty five"), "235");
    }

    #[test]
    fn replace_preserves_thousand_compound() {
        assert_eq!(replace_number_words("one thousand nine hundred"), "1900");
    }

    #[test]
    fn parse_rejects_clock_time_compound() {
        // "nine thirty" should be parsed as 9, stopping before "thirty".
        assert_eq!(parse_number_words(&["nine", "thirty"]), Some((9, 1)));
    }

    #[test]
    fn parse_rejects_hyphenated_clock_time() {
        // "nine-thirty" isn't a valid compound either; the whole word
        // is rejected so the outer replacer leaves it untouched.
        assert_eq!(parse_number_words(&["nine-thirty"]), None);
    }

    // ── Ordinal number words ──────────────────────────────────────────
    // Ordinals ("first", "twenty-seventh", "thirtieth") map to the same
    // numeric value as their cardinal counterparts — users writing dates
    // say "the 27th of April" as readily as "April 27", and the
    // normaliser should smooth over that difference before skills see it.

    #[test]
    fn ordinals_resolve_like_cardinals() {
        assert_eq!(words_to_number("first"), Some(1));
        assert_eq!(words_to_number("second"), Some(2));
        assert_eq!(words_to_number("third"), Some(3));
        assert_eq!(words_to_number("fifth"), Some(5));
        assert_eq!(words_to_number("eighth"), Some(8));
        assert_eq!(words_to_number("ninth"), Some(9));
        assert_eq!(words_to_number("twelfth"), Some(12));
        assert_eq!(words_to_number("twentieth"), Some(20));
        assert_eq!(words_to_number("thirtieth"), Some(30));
        assert_eq!(words_to_number("hundredth"), Some(100));
    }

    #[test]
    fn ordinal_compound_day_of_month() {
        // "twenty seventh" is the English month-day ordinal compound.
        // Existing tens+ones compound rule applies when "seventh" maps
        // to 7, so the normaliser returns a single integer just like
        // it does for the cardinal "twenty seven".
        assert_eq!(parse_number_words(&["twenty", "seventh"]), Some((27, 2)));
        assert_eq!(parse_number_words(&["thirty", "first"]), Some((31, 2)));
        assert_eq!(replace_number_words("the twenty seventh of april"), "the 27 of april");
    }

    #[test]
    fn ordinal_hyphenated_day_of_month() {
        assert_eq!(parse_number_words(&["twenty-seventh"]), Some((27, 1)));
        assert_eq!(parse_number_words(&["thirty-first"]), Some((31, 1)));
    }
}
