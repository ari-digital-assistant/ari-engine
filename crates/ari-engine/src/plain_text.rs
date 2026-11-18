//! Markdown to plain text, for assistant replies.
//!
//! Language models format with markdown whether or not anything renders
//! it. Ari shows replies as plain text and speaks them, so a list came
//! through as `**Cheetah**` with the asterisks on screen (ari-android#30).
//! This keeps the words and the line structure and drops the syntax. It
//! covers what models write (emphasis, headings, bullets, code, links,
//! quotes, rules, tables), not the whole of CommonMark.

/// `markdown` as plain text. Emphasis, code and link syntax go, headings
/// and quotes keep their text, bullets become "•", table rows become
/// comma-separated cells, and numbered items keep their numbers.
pub(crate) fn plain_text(markdown: &str) -> String {
    let mut lines: Vec<String> = Vec::new();
    let mut fence: Option<&str> = None;
    for line in markdown.lines() {
        let trimmed = line.trim_start();
        if let Some(marker) = fence {
            if trimmed.starts_with(marker) {
                fence = None;
            } else {
                lines.push(line.to_string());
            }
            continue;
        }
        if let Some(marker) = ["```", "~~~"].into_iter().find(|m| trimmed.starts_with(m)) {
            fence = Some(marker);
            continue;
        }
        if let Some(text) = block_line(line) {
            // One blank line between paragraphs is plenty.
            if text.trim().is_empty() && lines.last().is_some_and(|l| l.trim().is_empty()) {
                continue;
            }
            lines.push(text);
        }
    }
    lines.join("\n").trim().to_string()
}

/// One line outside a code block, or `None` for a line that is only
/// syntax (a rule, a table's divider).
fn block_line(line: &str) -> Option<String> {
    let rest = line.trim_start();
    let indent = &line[..line.len() - rest.len()];
    if is_rule(rest) || is_table_divider(rest) {
        return None;
    }
    let mut rest = rest;
    while let Some(quoted) = rest.strip_prefix('>') {
        rest = quoted.trim_start();
    }
    if let Some(heading) = heading_text(rest) {
        return Some(inline(heading));
    }
    if let Some(item) = bullet_item(rest) {
        return Some(format!("{indent}• {}", inline(item)));
    }
    if let Some(row) = table_row(rest) {
        return Some(row);
    }
    Some(format!("{indent}{}", inline(rest)))
}

/// `---`, `***`, `___`, and a setext heading's `===` underline.
fn is_rule(line: &str) -> bool {
    let marks: Vec<char> = line.chars().filter(|c| !c.is_whitespace()).collect();
    marks.len() >= 3 && matches!(marks[0], '-' | '*' | '_' | '=') && marks.iter().all(|&c| c == marks[0])
}

/// `|---|:---:|`, the line between a table's header and its rows.
fn is_table_divider(line: &str) -> bool {
    let line = line.trim();
    line.starts_with('|') && line.contains('-') && line.chars().all(|c| matches!(c, '|' | '-' | ':' | ' '))
}

/// The text of an ATX heading (`## Title`). `#hashtag` is not one.
fn heading_text(line: &str) -> Option<&str> {
    let hashes = line.len() - line.trim_start_matches('#').len();
    if !(1..=6).contains(&hashes) {
        return None;
    }
    let rest = &line[hashes..];
    if !rest.is_empty() && !rest.starts_with([' ', '\t']) {
        return None;
    }
    Some(rest.trim().trim_end_matches('#').trim_end())
}

/// The text of a `-`, `*` or `+` bullet item.
fn bullet_item(line: &str) -> Option<&str> {
    let rest = line.strip_prefix(['-', '*', '+'])?;
    let item = rest.strip_prefix([' ', '\t'])?;
    Some(item.trim_start())
}

/// `| a | b |` as `a, b`.
fn table_row(line: &str) -> Option<String> {
    let inner = line.trim().strip_prefix('|')?.strip_suffix('|')?;
    let cells: Vec<String> = inner
        .split('|')
        .map(|cell| inline(cell.trim()))
        .filter(|cell| !cell.is_empty())
        .collect();
    Some(cells.join(", "))
}

/// A character of a line, and whether it is literal text that markup must
/// leave alone: code, or a character escaped with a backslash.
#[derive(Clone, Copy)]
struct Ch {
    c: char,
    literal: bool,
}

fn inline(text: &str) -> String {
    let chars = emphasis(links(protect(text)));
    chars.into_iter().map(|ch| ch.c).collect()
}

/// Mark code spans and backslash escapes literal, dropping their syntax.
fn protect(text: &str) -> Vec<Ch> {
    let src: Vec<char> = text.chars().collect();
    let mut out = Vec::with_capacity(src.len());
    let mut i = 0;
    while i < src.len() {
        let c = src[i];
        if c == '\\' && src.get(i + 1).is_some_and(|n| n.is_ascii_punctuation()) {
            out.push(Ch { c: src[i + 1], literal: true });
            i += 2;
            continue;
        }
        if c == '`' {
            let run = backtick_run(&src, i);
            let literal = |c: &char| Ch { c: *c, literal: true };
            match closing_backticks(&src, i + run, run) {
                Some(end) => {
                    let code: String = src[i + run..end].iter().collect();
                    out.extend(code.trim().chars().map(|c| literal(&c)));
                    i = end + run;
                }
                None => {
                    out.extend(src[i..i + run].iter().map(literal));
                    i += run;
                }
            }
            continue;
        }
        out.push(Ch { c, literal: false });
        i += 1;
    }
    out
}

fn backtick_run(src: &[char], from: usize) -> usize {
    src[from..].iter().take_while(|&&c| c == '`').count()
}

/// Where a run of exactly `len` backticks starts, at or after `from`.
fn closing_backticks(src: &[char], from: usize, len: usize) -> Option<usize> {
    let mut i = from;
    while i < src.len() {
        if src[i] == '`' {
            let run = backtick_run(src, i);
            if run == len {
                return Some(i);
            }
            i += run;
        } else {
            i += 1;
        }
    }
    None
}

fn is(chars: &[Ch], i: usize, c: char) -> bool {
    chars.get(i).is_some_and(|ch| !ch.literal && ch.c == c)
}

/// `[text](url)` and `![alt](url)` become their text; `<https://…>` its URL.
fn links(chars: Vec<Ch>) -> Vec<Ch> {
    let mut out: Vec<Ch> = Vec::with_capacity(chars.len());
    let mut i = 0;
    while i < chars.len() {
        if is(&chars, i, '[') {
            if let Some((close, end)) = link_at(&chars, i) {
                if out.last().is_some_and(|ch| !ch.literal && ch.c == '!') {
                    out.pop();
                }
                out.extend_from_slice(&chars[i + 1..close]);
                i = end + 1;
                continue;
            }
        }
        if is(&chars, i, '<') {
            if let Some(end) = autolink_at(&chars, i) {
                out.extend_from_slice(&chars[i + 1..end]);
                i = end + 1;
                continue;
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

/// For a `[text](url)` opening at `open`, the positions of its `]` and `)`.
fn link_at(chars: &[Ch], open: usize) -> Option<(usize, usize)> {
    let close = (open + 1..chars.len()).find(|&j| is(chars, j, ']') || is(chars, j, '['))?;
    if !is(chars, close, ']') || !is(chars, close + 1, '(') {
        return None;
    }
    let end = (close + 2..chars.len()).find(|&j| is(chars, j, ')'))?;
    Some((close, end))
}

/// For a `<https://…>` opening at `open`, the position of its `>`.
fn autolink_at(chars: &[Ch], open: usize) -> Option<usize> {
    let rest: String = chars[open + 1..].iter().take(8).map(|ch| ch.c).collect();
    if !["http://", "https://", "mailto:"].iter().any(|scheme| rest.starts_with(scheme)) {
        return None;
    }
    let end = (open + 1..chars.len()).find(|&j| chars[j].c == '>' || chars[j].c.is_whitespace())?;
    is(chars, end, '>').then_some(end)
}

/// Drop paired `*`, `_` and `~~` delimiters, CommonMark-style: a run opens
/// when a word follows it and closes when one precedes it. Unpaired runs
/// stay as text, so `2 * 3` and `~70 mph` survive. An underscore inside a
/// word (`snake_case`) is text.
fn emphasis(chars: Vec<Ch>) -> Vec<Ch> {
    let mut keep = vec![true; chars.len()];
    // Runs waiting for a closer: (delimiter, start, length).
    let mut openers: Vec<(char, usize, usize)> = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i].c;
        if chars[i].literal || !matches!(c, '*' | '_' | '~') {
            i += 1;
            continue;
        }
        let len = chars[i..].iter().take_while(|ch| !ch.literal && ch.c == c).count();
        let before = i.checked_sub(1).map(|j| chars[j].c);
        let after = chars.get(i + len).map(|ch| ch.c);
        let word_after = after.is_some_and(|a| !a.is_whitespace());
        let word_before = before.is_some_and(|b| !b.is_whitespace());
        let (can_open, can_close) = match c {
            '_' => (
                word_after && !before.is_some_and(char::is_alphanumeric),
                word_before && !after.is_some_and(char::is_alphanumeric),
            ),
            '~' if len != 2 => (false, false),
            _ => (word_after, word_before),
        };
        if can_close {
            if let Some(at) = openers.iter().rposition(|&(open, _, _)| open == c) {
                let (_, start, open_len) = openers[at];
                openers.truncate(at);
                keep[start..start + open_len].fill(false);
                keep[i..i + len].fill(false);
                i += len;
                continue;
            }
        }
        if can_open {
            openers.push((c, i, len));
        }
        i += len;
    }
    chars.into_iter().zip(keep).filter_map(|(ch, k)| k.then_some(ch)).collect()
}

#[cfg(test)]
mod tests {
    use super::plain_text;

    #[test]
    fn a_gemma_style_list_comes_out_clean() {
        let reply = "Okay, here are the top 5 fastest animals:\n\n\
                     1. **Peregrine Falcon:** Up to 240 mph in a dive.\n\
                     2. **Cheetah:** Around 70 mph.\n\n\
                     * *Sailfish*: about 68 mph\n\
                     - __Pronghorn__";
        assert_eq!(
            plain_text(reply),
            "Okay, here are the top 5 fastest animals:\n\n\
             1. Peregrine Falcon: Up to 240 mph in a dive.\n\
             2. Cheetah: Around 70 mph.\n\n\
             • Sailfish: about 68 mph\n\
             • Pronghorn"
        );
    }

    #[test]
    fn plain_text_is_left_alone() {
        let text = "The sky is blue because air scatters blue light the most.";
        assert_eq!(plain_text(text), text);
    }

    #[test]
    fn headings_rules_and_quotes_keep_only_their_text() {
        assert_eq!(plain_text("## Small models\n---\n> Gemma 3 1B"), "Small models\nGemma 3 1B");
        assert_eq!(plain_text("#hashtag stays"), "#hashtag stays");
    }

    #[test]
    fn nested_bullets_keep_their_indent() {
        assert_eq!(plain_text("- Cats\n  - Lions\n  - Tigers"), "• Cats\n  • Lions\n  • Tigers");
    }

    #[test]
    fn code_loses_its_backticks_but_not_its_contents() {
        assert_eq!(plain_text("Run `cargo test` now."), "Run cargo test now.");
        assert_eq!(plain_text("Use `**kwargs`."), "Use **kwargs.");
        assert_eq!(plain_text("```rust\nlet x = 1;\n```\nDone."), "let x = 1;\nDone.");
    }

    #[test]
    fn links_keep_their_text() {
        assert_eq!(plain_text("See [the docs](https://example.com)."), "See the docs.");
        assert_eq!(plain_text("![a logo](logo.png)"), "a logo");
        assert_eq!(plain_text("Go to <https://heyari.dev>."), "Go to https://heyari.dev.");
    }

    #[test]
    fn asterisks_and_underscores_that_are_not_markup_survive() {
        assert_eq!(plain_text("2 * 3 is 6"), "2 * 3 is 6");
        assert_eq!(plain_text("2*3 is 6"), "2*3 is 6");
        assert_eq!(plain_text("set snake_case_name"), "set snake_case_name");
        assert_eq!(plain_text("about ~70 mph"), "about ~70 mph");
        assert_eq!(plain_text(r"a literal \*star\*"), "a literal *star*");
    }

    #[test]
    fn strikethrough_and_nesting() {
        assert_eq!(plain_text("~~old~~ new"), "old new");
        assert_eq!(plain_text("***very*** **bold *and* italic**"), "very bold and italic");
    }

    #[test]
    fn tables_flatten_to_rows() {
        let table = "| Animal | Speed |\n|---|:---:|\n| Cheetah | 70 mph |";
        assert_eq!(plain_text(table), "Animal, Speed\nCheetah, 70 mph");
    }
}
