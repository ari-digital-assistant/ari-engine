//! Expose the engine's own `normalize_input` over stdin/stdout so the
//! tooling outside this repo can normalise text through the
//! SAME function the router is served at inference — rather than a Python
//! replica, which would drift and would defeat the point of train/serve parity.
//!
//! Usage: `normalize --locale it < raw.txt > normalised.txt`
//! One line in, one line out, order preserved. `normalize_input` collapses
//! whitespace, so an output line can never contain a newline.

use ari_core::normalize_input;
use std::io::Read;

/// Normalise every line of `input` for `locale`, preserving line count and order.
fn run(input: &str, locale: &str) -> String {
    input
        .lines()
        .map(|line| normalize_input(line, locale))
        .collect::<Vec<String>>()
        .join("\n")
}

fn main() {
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

    let mut input = String::new();
    if let Err(e) = std::io::stdin().read_to_string(&mut input) {
        eprintln!("failed to read stdin: {e}");
        std::process::exit(1);
    }
    println!("{}", run(&input, &locale));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn en_expands_contractions_and_lowercases() {
        assert_eq!(run("what's the time\nI need the TIME", "en"), "what is the time\ni need the time");
    }

    #[test]
    fn it_splits_elisions_and_keeps_accents() {
        assert_eq!(run("dimmi l'ora\nche ora è", "it"), "dimmi l ora\nche ora è");
    }

    #[test]
    fn line_count_is_preserved() {
        // The Python side pairs input lines to output lines by index, so a
        // dropped or added line would silently mis-pair the whole corpus.
        let input = "one\ntwo\nthree";
        assert_eq!(run(input, "en").lines().count(), 3);
    }

    #[test]
    fn punctuation_is_stripped_but_math_chars_survive() {
        assert_eq!(run("quick, what time is it?", "en"), "quick what time is it");
        assert_eq!(run("calculate 5 + 3", "en"), "calculate 5 + 3");
    }
}
