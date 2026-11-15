//! An on-device assistant answer, handed over while the model is still
//! writing it (ari-android#37): the words so far, for the host to show, and
//! each finished sentence, for it to speak.
//!
//! Words go out whole, never inside emphasis, a code span or a link, so no
//! markdown flickers onto the screen. A sentence goes out once it can't
//! change any more: up to the last newline, or the last sentence end once the
//! next sentence has begun. Both get the same cleanup as the finished answer
//! (reasoning removed, markdown reduced to plain text, continuation marker
//! split off), so what is shown and spoken early reads as the whole answer.

use crate::{assistant_answer, AnswerSink};

/// One answer on its way to an [`AnswerSink`].
pub(crate) struct AnswerStream<'a> {
    sink: &'a dyn AnswerSink,
    /// The words last shown.
    shown: String,
    /// The cleaned answer up to the end of the last sentence released.
    released: String,
    /// The cleaned text stopped starting with what was already released,
    /// as when a stray `</think>` turns everything before it into a
    /// discarded draft. Nothing more goes out until [`Self::finish`].
    diverged: bool,
}

impl<'a> AnswerStream<'a> {
    pub(crate) fn new(sink: &'a dyn AnswerSink) -> Self {
        Self { sink, shown: String::new(), released: String::new(), diverged: false }
    }

    /// Show the words and release any sentences the model has finished,
    /// given its raw output so far.
    pub(crate) fn feed(&mut self, raw: &str) {
        let (words, _) = assistant_answer(&ari_llm::strip_thinking(&raw[..shown_len(raw)]));
        if words != self.shown && words.chars().any(char::is_alphanumeric) {
            self.sink.text(&words);
            self.shown = words;
        }
        if self.diverged {
            return;
        }
        let settled = &raw[..settled_len(raw)];
        let (clean, _) = assistant_answer(&ari_llm::strip_thinking(settled));
        match clean.strip_prefix(self.released.as_str()) {
            Some(new) => {
                if release(self.sink, new) {
                    self.released = clean;
                }
            }
            None => self.diverged = true,
        }
    }

    /// Release the rest of the finished, cleaned `answer`, so a host that
    /// got any of it gets all of it. An answer that released nothing early
    /// releases nothing here either: the host speaks the response as usual.
    pub(crate) fn finish(&self, answer: &str) {
        if self.released.is_empty() {
            return;
        }
        let rest = match answer.strip_prefix(self.released.as_str()) {
            Some(rest) => rest,
            None => after_spoken(answer, &self.released),
        };
        release(self.sink, rest);
    }
}

/// Hand `text` to the sink if it has any words in it, with the space or
/// line break before it, so the sentences joined end to end read as the
/// answer does.
fn release(sink: &dyn AnswerSink, text: &str) -> bool {
    if !text.chars().any(char::is_alphanumeric) {
        return false;
    }
    sink.sentence(text);
    true
}

/// What's left of `answer` after as many letters and digits as `spoken`
/// holds: where to pick up when the answer no longer starts with exactly
/// what was spoken, so the rest of it is still said.
fn after_spoken<'s>(answer: &'s str, spoken: &str) -> &'s str {
    let mut left = spoken.chars().filter(|c| c.is_alphanumeric()).count();
    for (at, c) in answer.char_indices() {
        if left == 0 {
            return answer[at..].trim_start_matches(|c: char| !c.is_alphanumeric());
        }
        if c.is_alphanumeric() {
            left -= 1;
        }
    }
    ""
}

/// Byte length of the part of `text` that can be shown: whole words, with no
/// emphasis, code span or link left open.
fn shown_len(text: &str) -> usize {
    let mut shown = 0;
    let mut line_start = 0;
    for (at, c) in text.char_indices() {
        if c.is_whitespace() && balanced(&text[line_start..at]) {
            shown = at;
        }
        if c == '\n' {
            line_start = at + 1;
        }
    }
    shown
}

/// Byte length of the part of `text` that is finished: up to the last
/// newline, or the last sentence end once the next sentence has begun.
fn settled_len(text: &str) -> usize {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut settled = 0;
    let mut line_start = 0;
    for (i, &(at, c)) in chars.iter().enumerate() {
        if c == '\n' {
            settled = at + 1;
            line_start = i + 1;
        } else if let Some(end) = sentence_end(&chars, line_start, i) {
            settled = end;
        }
    }
    settled
}

/// If the punctuation at `i` ends a sentence and the next one has begun,
/// the byte offset just past it and any closing quote, bracket or emphasis.
fn sentence_end(chars: &[(usize, char)], line_start: usize, i: usize) -> Option<usize> {
    let c = chars[i].1;
    if !matches!(c, '.' | '!' | '?' | '…') {
        return None;
    }
    let mut end = i + 1;
    while end < chars.len() && matches!(chars[end].1, '"' | '\'' | ')' | ']' | '»' | '”' | '’' | '*' | '_') {
        end += 1;
    }
    // A space, then the next sentence's first character. A newline instead
    // settles the line by itself.
    let mut next = end;
    while next < chars.len() && matches!(chars[next].1, ' ' | '\t') {
        next += 1;
    }
    if next == end || next == chars.len() || chars[next].1 == '\n' {
        return None;
    }
    if c == '.' && (chars[next].1.is_lowercase() || is_abbreviation(chars, line_start, i)) {
        return None;
    }
    let line: String = chars[line_start..end].iter().map(|&(_, c)| c).collect();
    if !balanced(&line) {
        return None;
    }
    Some(chars[end].0)
}

/// Whether the full stop at `dot` closes an initial ("J."), an
/// abbreviation with stops in it ("e.g."), or a numbered list item's
/// number ("1." at the start of the line).
fn is_abbreviation(chars: &[(usize, char)], line_start: usize, dot: usize) -> bool {
    let mut start = dot;
    while start > line_start && !chars[start - 1].1.is_whitespace() {
        start -= 1;
    }
    let word: Vec<char> = chars[start..dot].iter().map(|&(_, c)| c).collect();
    let starts_line = chars[line_start..start].iter().all(|&(_, c)| c.is_whitespace());
    (word.len() == 1 && word[0].is_alphabetic())
        || (word.contains(&'.') && word.split(|&c| c == '.').all(|piece| piece.len() <= 2))
        || (starts_line && !word.is_empty() && word.iter().all(char::is_ascii_digit))
}

/// Whether a line cut here leaves no emphasis, code span or link half
/// open: cut inside one, the markdown cleanup reads the two halves
/// differently from the whole. A leading `* ` bullet doesn't count.
fn balanced(line: &str) -> bool {
    let body = line.trim_start();
    let body = body.strip_prefix("* ").unwrap_or(body);
    // Runs, not characters: `**` opens bold just as `*` opens italics.
    let runs = |mark: char| {
        let mut previous = None;
        body.chars()
            .filter(|&c| {
                let starts_run = c == mark && previous != Some(mark);
                previous = Some(c);
                starts_run
            })
            .count()
    };
    let count = |mark: char| body.chars().filter(|&c| c == mark).count();
    runs('*') % 2 == 0 && runs('_') % 2 == 0 && runs('`') % 2 == 0 && count('[') == count(']')
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[derive(Default)]
    struct Recorder {
        shown: Mutex<Vec<String>>,
        sentences: Mutex<Vec<String>>,
    }

    impl AnswerSink for Recorder {
        fn text(&self, so_far: &str) {
            self.shown.lock().unwrap().push(so_far.to_string());
        }
        fn sentence(&self, text: &str) {
            self.sentences.lock().unwrap().push(text.to_string());
        }
    }

    /// Feed `raw` the way the model writes it, a character at a time, then
    /// finish with the cleaned whole, and return everything shown and every
    /// sentence released.
    fn stream_all(raw: &str) -> (Vec<String>, Vec<String>) {
        let sink = Recorder::default();
        let mut stream = AnswerStream::new(&sink);
        for (at, c) in raw.char_indices() {
            stream.feed(&raw[..at + c.len_utf8()]);
        }
        let (answer, _) = assistant_answer(&ari_llm::strip_thinking(raw));
        stream.finish(&answer);
        (sink.shown.into_inner().unwrap(), sink.sentences.into_inner().unwrap())
    }

    fn stream(raw: &str) -> Vec<String> {
        stream_all(raw).1
    }

    fn shown(raw: &str) -> Vec<String> {
        stream_all(raw).0
    }

    fn settled(text: &str) -> &str {
        &text[..settled_len(text)]
    }

    #[test]
    fn words_show_whole_and_outside_open_markdown() {
        assert_eq!(&"Paris is the cap"[..shown_len("Paris is the cap")], "Paris is the");
        assert_eq!(&"Paris is **the big"[..shown_len("Paris is **the big")], "Paris is");
        assert_eq!(&"See [the map"[..shown_len("See [the map")], "See");
    }

    #[test]
    fn a_one_sentence_answer_shows_its_words_as_they_come() {
        assert_eq!(
            shown("Some often named are Lionel Messi and Cristiano Ronaldo. [continuation]"),
            [
                "Some",
                "Some often",
                "Some often named",
                "Some often named are",
                "Some often named are Lionel",
                "Some often named are Lionel Messi",
                "Some often named are Lionel Messi and",
                "Some often named are Lionel Messi and Cristiano",
                "Some often named are Lionel Messi and Cristiano Ronaldo.",
            ]
        );
    }

    #[test]
    fn shown_words_never_carry_markdown_or_the_marker() {
        for words in shown("Top two:\n* **Lionel Messi**: Argentina.\n* **Cristiano Ronaldo**: Portugal. [new]") {
            assert!(!words.contains(['*', '[']), "{words:?}");
        }
    }

    #[test]
    fn a_sentence_settles_once_the_next_begins() {
        assert_eq!(settled("Paris is the capital. It"), "Paris is the capital.");
        assert_eq!(settled("Paris is the capital. "), "");
        assert_eq!(settled("Paris is the capital."), "");
    }

    #[test]
    fn a_newline_settles_the_line() {
        assert_eq!(settled("Top five:\n1. Messi"), "Top five:\n");
    }

    #[test]
    fn numbers_initials_and_abbreviations_are_not_sentence_ends() {
        assert_eq!(settled("1. Lionel Messi"), "");
        assert_eq!(settled("Pi is 3.14 or so"), "");
        assert_eq!(settled("J. K. Rowling wrote it"), "");
        assert_eq!(settled("Fruit, e.g. apples"), "");
        assert_eq!(settled("Made in the U.S. Then"), "");
        assert_eq!(settled("Born in 1987. He"), "Born in 1987.");
    }

    #[test]
    fn a_sentence_inside_emphasis_waits_for_it_to_close() {
        assert_eq!(settled("**Messi. He** is"), "");
        assert_eq!(settled("**Messi.** He"), "**Messi.**");
        assert_eq!(settled("* Messi. He won"), "* Messi.");
    }

    #[test]
    fn sentences_go_out_as_they_finish_and_the_marker_never_does() {
        assert_eq!(
            stream("Paris is the capital of France. It has about two million people. [continuation]"),
            ["Paris is the capital of France.", " It has about two million people."]
        );
    }

    #[test]
    fn a_list_goes_out_a_line_at_a_time_as_plain_text() {
        assert_eq!(
            stream("Top five:\n* **Lionel Messi**: Argentina.\n* **Cristiano Ronaldo**: Portugal. [new]"),
            ["Top five:", "\n• Lionel Messi: Argentina.", "\n• Cristiano Ronaldo: Portugal."]
        );
    }

    #[test]
    fn a_one_sentence_answer_releases_nothing() {
        assert!(stream("Paris.").is_empty());
    }

    #[test]
    fn reasoning_is_never_released() {
        assert_eq!(
            stream("<think>They want a city. Easy.</think>Canberra. It is in the ACT."),
            ["Canberra.", " It is in the ACT."]
        );
    }

    #[test]
    fn a_discarded_draft_still_gets_the_rest_of_the_answer_said() {
        // LFM3 with reasoning off writes a draft, a stray closer, then the
        // answer again: the draft was already spoken, so pick up after as
        // many words as it held.
        assert_eq!(
            stream("Paris is the capital. It is big.</think>Paris is the capital of France. It is big."),
            ["Paris is the capital.", "of France. It is big."]
        );
    }
}
