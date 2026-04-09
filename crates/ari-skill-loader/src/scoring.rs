//! Native pattern scorer shared between the declarative and WASM adapters.
//!
//! Both adapters score the same way unless a WASM skill explicitly opts in to
//! a custom `score()` export — keyword patterns require all keywords present
//! as whole words, regex patterns require a match anywhere in the (already
//! normalised) input. The highest matching weight wins.

use crate::manifest::{MatchPattern, Matching};
use ari_core::normalize_input;
use regex::Regex;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ScorerError {
    #[error("invalid regex pattern {pattern:?}: {source}")]
    BadRegex {
        pattern: String,
        #[source]
        source: regex::Error,
    },
}

#[derive(Debug)]
enum CompiledPattern {
    Keywords { words: Vec<String>, weight: f32 },
    Regex { regex: Regex, weight: f32 },
}

impl CompiledPattern {
    fn score(&self, normalised_input: &str) -> Option<f32> {
        match self {
            CompiledPattern::Keywords { words, weight } => {
                if words.iter().all(|w| contains_word(normalised_input, w)) {
                    Some(*weight)
                } else {
                    None
                }
            }
            CompiledPattern::Regex { regex, weight } => {
                if regex.is_match(normalised_input) {
                    Some(*weight)
                } else {
                    None
                }
            }
        }
    }
}

/// A compiled pattern set, ready to score inputs.
#[derive(Debug)]
pub struct PatternScorer {
    patterns: Vec<CompiledPattern>,
}

impl PatternScorer {
    pub fn compile(matching: &Matching) -> Result<Self, ScorerError> {
        let patterns = matching
            .patterns
            .iter()
            .map(compile_pattern)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(PatternScorer { patterns })
    }

    /// Score a raw (un-normalised) input string. Returns the highest matching
    /// weight across all patterns, or `0.0` if nothing matches.
    pub fn score(&self, raw_input: &str) -> f32 {
        let normalised = normalize_input(raw_input);
        self.score_normalised(&normalised)
    }

    /// Score an already-normalised input. Useful when the engine has already
    /// normalised once and doesn't want to do it again per skill.
    pub fn score_normalised(&self, normalised: &str) -> f32 {
        let mut best = 0.0_f32;
        for pat in &self.patterns {
            if let Some(w) = pat.score(normalised) {
                if w > best {
                    best = w;
                }
            }
        }
        best
    }
}

fn compile_pattern(p: &MatchPattern) -> Result<CompiledPattern, ScorerError> {
    match p {
        MatchPattern::Keywords { words, weight } => Ok(CompiledPattern::Keywords {
            words: words.clone(),
            weight: *weight,
        }),
        MatchPattern::Regex { pattern, weight } => {
            let regex = Regex::new(pattern).map_err(|source| ScorerError::BadRegex {
                pattern: pattern.clone(),
                source,
            })?;
            Ok(CompiledPattern::Regex {
                regex,
                weight: *weight,
            })
        }
    }
}

/// Whole-word containment check against an already-normalised input. Avoids
/// `"toss"` matching `"tossed"` and similar — we want intent matching, not
/// substring matching.
fn contains_word(haystack: &str, needle: &str) -> bool {
    haystack.split_whitespace().any(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{MatchPattern, Matching};

    fn matching(patterns: Vec<MatchPattern>) -> Matching {
        Matching {
            patterns,
            custom_score: false,
        }
    }

    #[test]
    fn keywords_require_all_words_as_whole_words() {
        let m = matching(vec![MatchPattern::Keywords {
            words: vec!["flip".to_string(), "coin".to_string()],
            weight: 0.95,
        }]);
        let s = PatternScorer::compile(&m).unwrap();
        assert_eq!(s.score("flip a coin"), 0.95);
        assert_eq!(s.score("toss a coin"), 0.0);
        assert_eq!(s.score("flipping a coin"), 0.0);
        assert_eq!(s.score("flip the pancakes"), 0.0);
    }

    #[test]
    fn highest_weight_wins() {
        let m = matching(vec![
            MatchPattern::Keywords {
                words: vec!["foo".to_string()],
                weight: 0.5,
            },
            MatchPattern::Keywords {
                words: vec!["foo".to_string(), "bar".to_string()],
                weight: 0.9,
            },
        ]);
        let s = PatternScorer::compile(&m).unwrap();
        assert_eq!(s.score("just foo"), 0.5);
        assert_eq!(s.score("foo bar"), 0.9);
    }

    #[test]
    fn regex_compiles_and_matches_normalised_input() {
        let m = matching(vec![MatchPattern::Regex {
            pattern: "what.*weather".to_string(),
            weight: 0.85,
        }]);
        let s = PatternScorer::compile(&m).unwrap();
        // normalize_input lowercases and expands contractions
        assert_eq!(s.score("What's the weather like?"), 0.85);
        assert_eq!(s.score("the weather is nice"), 0.0);
    }

    #[test]
    fn bad_regex_fails_at_compile() {
        let m = matching(vec![MatchPattern::Regex {
            pattern: "[unclosed".to_string(),
            weight: 0.9,
        }]);
        let err = PatternScorer::compile(&m).unwrap_err();
        match err {
            ScorerError::BadRegex { pattern, .. } => assert_eq!(pattern, "[unclosed"),
        }
    }

    #[test]
    fn no_patterns_means_zero_score() {
        let m = matching(vec![]);
        let s = PatternScorer::compile(&m).unwrap();
        assert_eq!(s.score("anything at all"), 0.0);
    }
}
