//! Strips a leading wake phrase from a transcript so the engine never sees it
//! and so endpoint detection can tell "user only said the wake phrase" apart
//! from "user said something real".
//!
//! Patterns covered:
//!   - Optional opener: hey / ok / okay / hi / hello (+ per-locale additions)
//!   - Required name: ari / ary / arie / airy / harry / jarvis / jarviz
//!     (sherpa frequently mishears "ari" as "harry" or "airy"; the list grows
//!     empirically as we see new transcription artefacts, and per-locale
//!     additions get appended via [`mishears_for`])
//!   - Trailing punctuation, comma, full stop, etc., is eaten
//!
//! Up to two words before the name are also dropped, since the pre-roll
//! sometimes catches a stray ambient word and we don't want it polluting the
//! query either. A name later than that is not a wake phrase: in all 93
//! recorded real wakes the name came straight after the opener or fused with
//! it ("Heyari"), and matching it anywhere is how "we're" and "you're" in a
//! lecture and a TV scene came to be answered. If no wake phrase is found at
//! all, returns the input trimmed.
//!
//! The wake-word model is always English regardless of Ari's active locale
//! (per the multi-language plan: per-language wake training is parked), so the
//! baseline patterns are English. Per-locale entries in [`mishears_for`] cover
//! sherpa's locale-specific mishears (e.g. an Italian transcriber might render
//! "hey ari" as something English users never see).
//!
//! The regexes run in Unicode mode, as Android's ICU-backed `java.util.regex`
//! did, so non-ASCII spaces and case folding behave as they did on the phone.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use regex::{Regex, RegexBuilder};

/// Per-locale wake-phrase mishear additions. Empty for every locale at first;
/// populated as native speakers report sherpa mishearing the English wake word
/// in their language. The baseline English patterns apply on top regardless.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WakeMishearEntries {
    pub extra_openers: Vec<&'static str>,
    pub extra_names: Vec<&'static str>,
}

/// The entries for `locale`, a lowercased ISO 639-1 code. Empty for every
/// locale so far: adding one is an edit here, no FFI and no frontend change.
///
/// Italian has no native-speaker mishears collected yet. The wake-word model
/// is still English ("hey ari"), so the baseline patterns already cover the
/// common case. Add Italian-specific transcription artefacts as they are
/// reported.
pub fn mishears_for(_locale: &str) -> WakeMishearEntries {
    WakeMishearEntries::default()
}

// Two-stage strip:
//   1. Try to match opener + name (the strict case).
//   2. If no name token was found, fall back to stripping a bare leading
//      opener ("ok", "okay", "hey", "hi", "hello") because sherpa sometimes
//      drops the name entirely on the way out (e.g. "ok ari what time" ->
//      "Okay what time"). This is mildly risky if the user genuinely starts
//      a query with "ok": accepted cost.
const BASE_OPENERS: &[&str] = &["hey", "ok", "okay", "hi", "hello"];
// No "re": it matched the tail of every "we're" and "you're", and both false
// wakes Ari answered in September were exactly that. "hayari" is Kroko's fused
// "Hey Ari", 2 of 163 recorded wakes on the November bench; two words, "Hay
// Ari", already pass on the stray-word allowance, and "hay" stays out of the
// openers so a command starting "hay fever" keeps its first word.
const BASE_NAMES: &[&str] = &[
    "ari", "ary", "arie", "arrie", "airy", "harry", "hari", "hairy", "ara", "ori", "orie", "ray",
    "rae", "jarvis", "jarviz", "hayari",
];

// Names too short or too common to trust on their own, accepted only straight
// after an opener. Cloud transcription wrote a real "Hey Ari" as "Hey, R" on
// 2026-10-10, and the user had to say it twice; a lone "R" anywhere else is a
// letter, not a name.
const OPENER_ONLY_NAMES: &[&str] = &["r"];

/// `words` with `extra` appended, duplicates dropped, escaped and joined as
/// regex alternatives.
fn alternatives(words: &[&str], extra: &[&str]) -> String {
    let mut seen: Vec<&str> = Vec::new();
    for w in words.iter().chain(extra) {
        if !seen.contains(w) {
            seen.push(w);
        }
    }
    seen.iter()
        .map(|w| regex::escape(w))
        .collect::<Vec<_>>()
        .join("|")
}

fn build_wake_regex(entries: &WakeMishearEntries) -> Regex {
    let openers = alternatives(BASE_OPENERS, &entries.extra_openers);
    let names = alternatives(BASE_NAMES, &entries.extra_names);
    let opener_only = alternatives(OPENER_ONLY_NAMES, &[]);
    RegexBuilder::new(&format!(
        r"^(?:\S+\s+){{0,2}}?\b(?:(?:{openers})?\s*(?:{names})|(?:{openers})[\s,.!?:;]+(?:{opener_only}))\b[\s,.!?:;]*"
    ))
    .case_insensitive(true)
    .build()
    .expect("wake phrase regex is valid")
}

fn build_leading_opener_regex(entries: &WakeMishearEntries) -> Regex {
    let openers = alternatives(BASE_OPENERS, &entries.extra_openers);
    RegexBuilder::new(&format!(r"^\s*(?:{openers})\b[\s,.!?:;]*"))
        .case_insensitive(true)
        .build()
        .expect("leading opener regex is valid")
}

/// The two regexes for a locale: the full wake phrase, and a bare leading opener.
type WakeRegexes = Arc<(Regex, Regex)>;

// Cache compiled regexes per locale: building a Regex from source allocates
// and parses, and this strip is on the hot transcription path.
fn regexes_for(locale: &str) -> WakeRegexes {
    static CACHE: OnceLock<Mutex<HashMap<String, WakeRegexes>>> = OnceLock::new();
    let key = locale.to_lowercase();
    let mut cache = CACHE
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    cache
        .entry(key)
        .or_insert_with_key(|key| {
            let entries = mishears_for(key);
            Arc::new((
                build_wake_regex(&entries),
                build_leading_opener_regex(&entries),
            ))
        })
        .clone()
}

/// Kotlin's `String.trim()`, which the transcripts were always trimmed with:
/// Unicode space separators and ASCII controls `\t`..`\r` and `\u{1c}`..`\u{1f}`.
/// Rust's own `trim` differs from it on exactly two counts, NEL and the
/// separators `\u{1c}`..`\u{1f}`.
fn trim(text: &str) -> &str {
    text.trim_matches(|c: char| {
        (c.is_whitespace() && c != '\u{85}') || ('\u{1c}'..='\u{1f}').contains(&c)
    })
}

/// The outcome of a wake-phrase strip: the remaining query text, and whether a
/// real name token was found (stage 1) rather than just a bare leading opener
/// (stage 2). The flag is the signal used to verify that a wake-word detection
/// was genuinely addressed to Ari: the 64 KB detector fires on phonetic shape
/// alone, sherpa knows what words are.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WakeMatch {
    pub text: String,
    pub name_matched: bool,
}

/// Strip the wake phrase from `text`, using locale-specific mishears stacked on
/// top of the baseline English patterns. `locale` should be an ISO 639-1
/// lowercase code (`"en"`, `"it"`, …); unknown locales fall back to
/// baseline-only behaviour.
pub fn match_wake_phrase(text: &str, locale: &str) -> WakeMatch {
    let trimmed = trim(text);
    if trimmed.is_empty() {
        return WakeMatch {
            text: String::new(),
            name_matched: false,
        };
    }
    let regexes = regexes_for(locale);
    let (full, leading) = &*regexes;
    let after_full = full.replace(trimmed, "");
    if after_full != trimmed {
        return WakeMatch {
            text: trim(&after_full).to_string(),
            name_matched: true,
        };
    }
    // Strict regex didn't match. Fall back to stripping a bare leading
    // opener: sherpa sometimes elides the wake-word name entirely, leaving
    // just "okay what time is it" with no recognisable "ari" token.
    WakeMatch {
        text: trim(&leading.replace(trimmed, "")).to_string(),
        name_matched: false,
    }
}

/// [`match_wake_phrase`] for the call sites that only care about the query text.
pub fn strip_wake_phrase(text: &str, locale: &str) -> String {
    match_wake_phrase(text, locale).text
}

/// The wake-verification verdict a transcript carries: `m`'s verdict for
/// English, `None` for every other locale.
///
/// The wake-word model is English-only whatever Ari's active locale is, but
/// sherpa is not: a non-English recogniser transcribes the English phrase
/// through its own phonotactics. The name list [`match_wake_phrase`] checks was
/// built empirically from ENGLISH sherpa mishears and [`mishears_for`] is still
/// empty for every other language, so outside English we have no evidence
/// about what "hey ari" actually comes out as. Acting on a verdict we can't
/// trust would turn a benign failure (wake phrase left in the query, engine
/// says "not understood") into a hard one (turn silently dismissed), so we
/// don't form one, and the wake gate fails open on `None`. Same call the router
/// already makes in `routerSupportsLocale`.
///
/// The raw transcript is still kept for every locale: the Italian transcripts
/// accruing in the logs are how the mishear table eventually gets filled in.
///
/// This one line carries the entire non-English safety guarantee and needs a
/// test holding it in place.
pub fn wake_verdict(m: &WakeMatch, locale: &str) -> Option<bool> {
    if locale == "en" {
        Some(m.name_matched)
    } else {
        None
    }
}

/// What the early check made of a quiet wake's 2 s clip. The slug goes in the
/// wake log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EarlyNameCheck {
    /// The clip held the name: show Ari now.
    Pass,
    /// It did not. Changes nothing; the whole transcript still decides.
    Miss,
    /// No verdict: a locale that forms none, or no transcript came back.
    None,
}

impl EarlyNameCheck {
    pub fn slug(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Miss => "miss",
            Self::None => "none",
        }
    }
}

/// The name check on the short clip ending at a quiet wake, judged exactly as
/// the whole transcript is.
pub fn early_name_check(transcript: &str, locale: &str) -> EarlyNameCheck {
    match wake_verdict(&match_wake_phrase(transcript, locale), locale) {
        Some(true) => EarlyNameCheck::Pass,
        Some(false) => EarlyNameCheck::Miss,
        None => EarlyNameCheck::None,
    }
}

/// What a streaming partial shows: the text after the wake phrase, and the
/// name verdict so far.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamingPartial {
    pub text: String,
    pub name_matched: Option<bool>,
}

/// The partial to show for `m`, or `None` while there is nothing to show: no
/// text after the wake phrase and a verdict that says no name either. A partial
/// that is only the name still shows, so a bare call can reveal a quiet wake.
pub fn streaming_partial(m: &WakeMatch, locale: &str) -> Option<StreamingPartial> {
    let name_matched = wake_verdict(m, locale);
    if m.text.is_empty() && name_matched == Some(false) {
        return None;
    }
    Some(StreamingPartial {
        text: m.text.clone(),
        name_matched,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn en(text: &str) -> WakeMatch {
        match_wake_phrase(text, "en")
    }

    #[test]
    fn wake_phrase_with_command_reports_a_name_match_and_strips_it() {
        let m = en("hey ari whats the weather");
        assert!(m.name_matched);
        assert_eq!(m.text, "whats the weather");
    }

    #[test]
    fn opener_with_no_name_does_not_report_a_name_match() {
        let m = en("okay so whats the weather");
        assert!(!m.name_matched);
        assert_eq!(m.text, "so whats the weather");
    }

    #[test]
    fn mishear_from_the_name_list_still_reports_a_name_match() {
        let m = en("harry can you set a timer");
        assert!(m.name_matched);
        assert_eq!(m.text, "can you set a timer");
    }

    #[test]
    fn ara_mishear_is_accepted_as_a_name_token() {
        // Real capture 2026-07-29: user said "Hey Ari", sherpa wrote "ARA".
        let m = en("ARA reminds me about business in one hour");
        assert_eq!(m.text, "reminds me about business in one hour");
        assert!(m.name_matched);
    }

    #[test]
    fn fused_wake_and_verb_still_fails_documented_gap() {
        // "Hey Ari, remind" -> "Rind": no separable name token exists.
        // If this starts passing, a list entry got too greedy: investigate.
        assert!(!en("Rind me about business in one hour").name_matched);
    }

    #[test]
    fn the_two_false_wakes_ari_answered_are_no_longer_addressed_to_it() {
        // Real captures, 2026-09: a lecture and a TV scene, matched on "re".
        assert!(!en("So anyway, we're first sorting by country and then by city.").name_matched);
        assert!(!en("Ah, I guess time doesn't mean much when you're dead.").name_matched);
    }

    #[test]
    fn a_name_deep_in_a_sentence_is_not_a_wake_phrase_and_is_left_in_the_text() {
        let m = en("I was telling Harry about the weather");
        assert!(!m.name_matched);
        assert_eq!(m.text, "I was telling Harry about the weather");
    }

    #[test]
    fn a_stray_word_from_the_pre_roll_before_the_wake_phrase_is_still_a_match() {
        let m = en("um hey ari what time is it");
        assert!(m.name_matched);
        assert_eq!(m.text, "what time is it");
    }

    #[test]
    fn cloud_transcriptions_fused_heyari_is_a_match() {
        // Real capture, 2026-09-12: cloud STT writes the phrase as one word.
        let m = en("Heyari, what time is it?");
        assert!(m.name_matched);
        assert_eq!(m.text, "what time is it?");
    }

    #[test]
    fn krokos_fused_hayari_is_a_match() {
        // Real wake, 2026-11-14: the streaming quiet wake turned it away unseen.
        let m = en("Hayari, why is the sky blue");
        assert!(m.name_matched);
        assert_eq!(m.text, "why is the sky blue");
    }

    #[test]
    fn a_lone_r_straight_after_the_opener_is_a_match() {
        // Real capture, 2026-10-10: a genuine wake the old check rejected.
        let m = en("Hey, R, turn on the kitchen lights.");
        assert!(m.name_matched);
        assert_eq!(m.text, "turn on the kitchen lights.");
        assert!(en("ok r what time is it").name_matched);
    }

    #[test]
    fn an_r_anywhere_but_straight_after_the_opener_is_a_letter() {
        assert!(!en("R. Kelly is on the radio").name_matched);
        assert!(!en("I got an R in the test").name_matched);
        assert!(!en("hey r2 turn on the lights").name_matched);
    }

    #[test]
    fn unrelated_speech_reports_no_name_match_and_is_left_alone() {
        let m = en("i was talking to dave about it");
        assert!(!m.name_matched);
        assert_eq!(m.text, "i was talking to dave about it");
    }

    #[test]
    fn bare_wake_phrase_reports_a_name_match_and_empties_the_text() {
        let m = en("hey ari");
        assert!(m.name_matched);
        assert_eq!(m.text, "");
    }

    #[test]
    fn ok_opener_with_name_reports_a_name_match() {
        let m = en("ok ari whats the time");
        assert!(m.name_matched);
        assert_eq!(m.text, "whats the time");
    }

    #[test]
    fn empty_input_reports_no_name_match() {
        assert_eq!(
            en(""),
            WakeMatch {
                text: String::new(),
                name_matched: false
            }
        );
    }

    #[test]
    fn strip_wake_phrase_still_returns_just_the_text() {
        assert_eq!(
            strip_wake_phrase("hey ari whats the weather", "en"),
            "whats the weather"
        );
        assert_eq!(
            strip_wake_phrase("okay so whats the weather", "en"),
            "so whats the weather"
        );
    }

    #[test]
    fn english_forms_a_verdict_from_the_name_match() {
        assert_eq!(
            wake_verdict(&en("hey ari whats the weather"), "en"),
            Some(true)
        );
        assert_eq!(wake_verdict(&en("hey there mate"), "en"), Some(false));
    }

    #[test]
    fn non_english_forms_no_verdict_at_all() {
        // The name list was built from English sherpa mishears and the mishear
        // table is still empty everywhere else, so outside English we have no
        // evidence to reject on. None makes the wake gate fail open. Deleting
        // the locale check here would silently start dismissing Italian turns:
        // that is what this test exists to stop.
        assert_eq!(
            wake_verdict(&match_wake_phrase("hey there mate", "it"), "it"),
            None
        );
        assert_eq!(
            wake_verdict(&match_wake_phrase("hey ari che tempo fa", "it"), "it"),
            None
        );
    }

    #[test]
    fn the_short_clip_passes_when_the_name_check_would() {
        // Early transcripts from the offline test, 2026-10-17.
        assert_eq!(early_name_check("Hey Ari.", "en"), EarlyNameCheck::Pass);
        assert_eq!(early_name_check("Hei ari.", "en"), EarlyNameCheck::Pass);
        assert_eq!(early_name_check("Hey, R.", "en"), EarlyNameCheck::Pass);
    }

    #[test]
    fn a_clip_without_the_name_misses_and_changes_nothing() {
        assert_eq!(early_name_check("Eli.", "en"), EarlyNameCheck::Miss);
        assert_eq!(early_name_check("Hey, Addy.", "en"), EarlyNameCheck::Miss);
        assert_eq!(
            early_name_check("Hey Arya, what time?", "en"),
            EarlyNameCheck::Miss
        );
    }

    #[test]
    fn no_early_verdict_outside_english_as_for_the_whole_transcript() {
        assert_eq!(early_name_check("Ehi Ari.", "it"), EarlyNameCheck::None);
    }

    #[test]
    fn the_wake_log_slugs_are_unchanged() {
        // Bug reports and ari-tools read these out of the wake log.
        assert_eq!(EarlyNameCheck::Pass.slug(), "pass");
        assert_eq!(EarlyNameCheck::Miss.slug(), "miss");
        assert_eq!(EarlyNameCheck::None.slug(), "none");
    }

    #[test]
    fn a_streaming_partial_carries_the_name_verdict_with_the_text_after_it() {
        assert_eq!(
            streaming_partial(&en("Hey Ari what time is it"), "en"),
            Some(StreamingPartial {
                text: "what time is it".into(),
                name_matched: Some(true)
            }),
        );
        assert_eq!(
            streaming_partial(&en("we're going out"), "en"),
            Some(StreamingPartial {
                text: "we're going out".into(),
                name_matched: Some(false)
            }),
        );
    }

    #[test]
    fn a_partial_that_is_only_the_name_still_shows_so_a_bare_call_can_reveal() {
        assert_eq!(
            streaming_partial(&en("Hey Ari"), "en"),
            Some(StreamingPartial {
                text: String::new(),
                name_matched: Some(true)
            }),
        );
        assert_eq!(
            streaming_partial(&match_wake_phrase("Ehi Ari", "it"), "it"),
            Some(StreamingPartial {
                text: String::new(),
                name_matched: None
            }),
        );
    }

    #[test]
    fn a_partial_with_neither_text_nor_name_shows_nothing() {
        assert_eq!(streaming_partial(&en(""), "en"), None);
        assert_eq!(streaming_partial(&en("Hey"), "en"), None);
    }

    #[test]
    fn trimming_matches_kotlin_not_rust() {
        // Kotlin trimmed the information separators and kept NEL; Rust's own
        // trim does the opposite on both.
        assert_eq!(trim("\u{1c} hey \u{1f}"), "hey");
        assert_eq!(trim("\u{85}hey\u{85}"), "\u{85}hey\u{85}");
        assert_eq!(trim("\u{a0}\u{3000} hey\t\n"), "hey");
    }
}
