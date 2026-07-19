//! Batch keyword-scorer oracle for the FunctionGemma training pipeline.
//!
//! Reads utterances on stdin (one per line) and writes `true` or `false` per
//! line: `true` means the keyword scorer already claims that utterance, so the
//! router never sees it in production and training on it is wasted capacity.
//!
//! This exists because the router is the FALLBACK tier. `route-eval` enforces
//! the same rule on the eval sets; this enforces it on the training corpus.
//!
//! Usage: `keyword-hit [--locale <xx>] < utterances.txt`

/// Parse args: an optional `--locale <xx>`, defaulting to "en".
fn parse_args(args: impl Iterator<Item = String>) -> Result<String, String> {
    let mut locale = "en".to_string();
    let mut it = args;
    while let Some(a) = it.next() {
        match a.as_str() {
            "--locale" => {
                locale = it.next().ok_or_else(|| "--locale requires a value".to_string())?;
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    Ok(locale)
}

/// Answer the keyword question for each line, preserving order and count.
fn run(locale: &str, texts: &[String]) -> Vec<bool> {
    let mut engine = ari_ffi::build_engine_with_builtins();
    engine.set_locale(locale.to_string());
    texts
        .iter()
        .map(|t| engine.keyword_decision(t).is_some())
        .collect()
}

/// Split raw stdin text into one entry per input line.
///
/// Uses `str::lines()` so the line-exact contract holds at every edge:
/// empty input yields zero lines (not a phantom empty one), a lone `"\n"`
/// yields exactly one empty line, a trailing newline is not a phantom extra
/// line, and CRLF endings don't leave a stray `\r` on the line.
fn split_lines(input: &str) -> Vec<String> {
    input.lines().map(|s| s.to_string()).collect()
}

fn main() {
    use std::io::Read;

    let locale = match parse_args(std::env::args().skip(1)) {
        Ok(l) => l,
        Err(msg) => {
            eprintln!("{msg}");
            std::process::exit(2);
        }
    };

    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .expect("read stdin");

    let texts = split_lines(&input);

    for verdict in run(&locale, &texts) {
        println!("{verdict}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(v: &[&str]) -> Result<String, String> {
        parse_args(v.iter().map(|s| s.to_string()))
    }

    #[test]
    fn no_flag_defaults_to_en() {
        assert_eq!(parse(&[]).unwrap(), "en");
    }

    #[test]
    fn locale_flag_is_parsed() {
        assert_eq!(parse(&["--locale", "it"]).unwrap(), "it");
    }

    #[test]
    fn locale_flag_without_value_is_an_error() {
        assert!(parse(&["--locale"]).is_err());
    }

    #[test]
    fn unknown_argument_is_an_error() {
        assert!(parse(&["--nope"]).is_err());
    }

    #[test]
    fn english_verdicts_match_the_keyword_scorer() {
        let texts: Vec<String> = [
            "what time is it",      // canonical trigger — keyword scorer wins
            "long time no see",     // oblique greeting — router's job
            "what is the capital of Denmark", // general knowledge — nobody's trigger
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        assert_eq!(run("en", &texts), vec![true, false, false]);
    }

    #[test]
    fn italian_verdicts_use_the_italian_scorer() {
        let texts: Vec<String> = ["apri spotify", "che si racconta"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(run("it", &texts), vec![true, false]);
    }

    #[test]
    fn empty_input_yields_empty_output() {
        assert_eq!(run("en", &[]), Vec::<bool>::new());
    }

    #[test]
    fn empty_stdin_splits_to_zero_lines() {
        assert_eq!(split_lines(""), Vec::<String>::new());
    }

    #[test]
    fn single_blank_line_splits_to_exactly_one_line() {
        // Regression: the old strip_suffix + split logic collapsed a lone
        // "\n" down to zero lines, silently mis-pairing the caller's zip().
        assert_eq!(split_lines("\n"), vec!["".to_string()]);
    }

    #[test]
    fn trailing_newline_does_not_add_a_phantom_line() {
        assert_eq!(
            split_lines("what time is it\nlong time no see\n"),
            vec!["what time is it".to_string(), "long time no see".to_string()]
        );
    }

    #[test]
    fn embedded_blank_line_preserves_total_count() {
        assert_eq!(
            split_lines("first\n\nthird"),
            vec!["first".to_string(), "".to_string(), "third".to_string()]
        );
    }

    #[test]
    fn crlf_input_does_not_leave_a_trailing_carriage_return() {
        assert_eq!(
            split_lines("what time is it\r\nlong time no see\r\n"),
            vec!["what time is it".to_string(), "long time no see".to_string()]
        );
    }
}
