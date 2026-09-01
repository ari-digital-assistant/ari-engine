use serde::{Deserialize, Serialize};

/// Appended to the assistant system prompt when prior conversation turns
/// are supplied, instructing the model to self-classify the turn. The
/// engine parses and strips the trailing marker (see `ContinuationFlag`).
/// Internal control directive — English source, NOT user-facing copy, so
/// it is not localized. The marker is requested INLINE (same line) because
/// the on-device path stops generation at the first newline.
pub const CONTINUATION_INSTRUCTION: &str = "You may be shown earlier turns of this \
conversation as prior messages. If the user's latest message refers to them, use that \
context to answer. At the very end of your reply, append a single space and then exactly \
[continuation] if this message continues the earlier conversation, or [new] if it begins \
a new, unrelated topic. Keep the marker on the same line as your answer.";

/// A compact block of durable user facts for injection into an assistant
/// system prompt. `None` when there are no facts so callers emit nothing.
/// Format is fixed (see the plan Global Constraints): a header line then one
/// `- <fact>` bullet per fact.
pub fn remembered_facts_block(facts: &[String]) -> Option<String> {
    if facts.is_empty() {
        return None;
    }
    let mut block = String::from("Things you know about the user:");
    for fact in facts {
        block.push_str("\n- ");
        block.push_str(fact);
    }
    Some(block)
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

/// Result of a settings-time `settings_query` invocation. Mirrors the JSON the
/// skill returns: success with options (dynamic_select), success with a message
/// (validate), or failure with an error.
#[derive(Debug, Clone, PartialEq)]
pub struct SettingsQueryResult {
    pub ok: bool,
    pub error: Option<String>,
    pub options: Vec<SettingsOption>,
    pub message: Option<String>,
    /// When true, the frontend should re-run its dependent settings queries
    /// (dynamic_select / validate) — e.g. after an action mints credentials
    /// that those queries need. Defaults false.
    pub refresh: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SettingsOption {
    pub value: String,
    pub label: String,
}

impl SettingsQueryResult {
    pub fn unsupported() -> Self {
        SettingsQueryResult {
            ok: false,
            error: Some("settings_query unsupported by this skill".to_string()),
            options: Vec::new(),
            message: None,
            refresh: false,
        }
    }
}

/// One installed launchable app the frontend knows about, pushed into the
/// engine so scoring can distinguish "open <installed app>" from
/// "open <a physical thing>". `label` is the user-visible name; `package` is
/// the platform package id. Both feed resolution (mirroring the frontend
/// launcher's matcher), never display — so neither is translated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppEntry {
    pub label: String,
    pub package: String,
}

#[derive(Clone)]
pub struct SkillContext {
    pub locale: String,
    /// Installed launchable apps, pushed by the frontend via
    /// `Engine::set_installed_apps`. Empty on platforms that don't supply it
    /// (Linux, headless) or before the first push, which preserves the legacy
    /// "any target is an app" behaviour in `open`'s scoring.
    pub installed_apps: Vec<AppEntry>,
    /// The utterance as the user said it, before `normalize_input` lowercased
    /// it, expanded contractions and stripped punctuation. For skills that
    /// quote the user back to somebody else — a message body, a note — the
    /// normalised text is unusable: "I'll be home soon" becomes "i will be
    /// home soon", and that is what the other person reads.
    ///
    /// Populated for `execute` only. Scoring and matching run against
    /// normalised text by design, so it is empty during `score` and callers
    /// must not reach for it there.
    pub raw_input: String,
}

impl Default for SkillContext {
    fn default() -> Self {
        Self {
            locale: "en".to_string(),
            installed_apps: Vec::new(),
            raw_input: String::new(),
        }
    }
}

/// One example user utterance that should trigger a skill, paired with the
/// JSON arguments the function call should produce.
///
/// Matched directly against the utterance by the phrase tier — the second
/// layer of skill matching, which runs when the keyword/regex scorer finds
/// nothing. The phrase may carry `{slot}` placeholders, each binding one or
/// more words; `weight` is what a full match contributes.
///
/// `text` is the literal user utterance. `args` is a JSON object literal —
/// `"{}"` for parameterless skills, or `r#"{"app_name": "Spotify"}"#` for
/// parameterised ones. The args literal must be valid JSON; the export
/// pipeline parses it directly.
#[derive(Debug, Clone, PartialEq)]
pub struct ExampleUtterance {
    pub text: &'static str,
    pub args: &'static str,
    /// Score a full match contributes, on the same 0..=1 scale as a
    /// declarative skill's `matching.patterns`. Oblique phrasings that
    /// could plausibly belong to another skill sit lower than explicit ones.
    pub weight: f32,
}

/// A skill's opt-in declaration that it acts as a fallback NLU tier: when the
/// router and keyword scorers all miss, the engine forwards the raw utterance
/// to it. Parsed from `metadata.ari.fallback` in the skill manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FallbackTier {
    /// When `Some(key)`, the engine engages this fallback only while the
    /// skill's own setting `key` is non-empty (generic config gate). `None`
    /// means always engage.
    pub requires_setting: Option<String>,
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
/// 2. **Phrase matching (no model, still free).** Reads
///    `example_utterances_for()`. Fires only when the keyword scorer found
///    nothing. Catches paraphrases the keyword patterns missed (e.g. "is it
///    morning or afternoon" routes to `current_time` even though
///    "current_time" appears nowhere in the input).
///
/// A configured cloud assistant gets a third go at anything both decline,
/// reading `description()` and `parameters_schema()`.
///
/// You always have to implement `score()` and `execute()`. The rest are
/// optional but strongly recommended: phrases cost nothing at rest and are
/// the only thing serving users with no cloud assistant.
///
/// # Implementing for matching
///
/// - **`description()`** — write two sentences. First: what the skill
///   does. Second: when to use it, with semantic keywords. Example:
///   "Tells the current time. Use when the user asks what time it is,
///   what hour it is, whether it is morning or afternoon, or anything
///   about the current time of day." Nothing on the device reads this; a
///   cloud assistant does, matching on meaning, so the more natural
///   language you put here the better it routes.
///
/// - **`example_utterances()`** — return 20-30 varied phrasings. Cover
///   paraphrases, indirect language, conversational filler ("can you",
///   "please", "I need"). These are matched verbatim (modulo `{slot}`
///   placeholders), so write what you expect a user to actually say, and
///   store them normalised — see [`normalize_phrase`].
///
/// - **`parameters_schema()`** — for parameterised skills, override
///   this with an OpenAI-style JSON schema. Default is the
///   parameterless `{"type": "object", "properties": {}}`.
pub trait Skill: Send + Sync {
    /// Stable, unique identifier (e.g. `"current_time"`). This is what a
    /// cloud assistant names when it routes.
    fn id(&self) -> &str;

    /// Human-readable description. Read by a cloud assistant when it
    /// routes — see the trait-level docs.
    fn description(&self) -> &str { "" }

    /// Whether this skill may be reached by anything other than its own
    /// keyword triggers — the phrase tier and the assistant-routing
    /// catalogue. Skills that should only ever fire on explicit triggers —
    /// e.g. web search, which competes with the configured assistant for
    /// any "what is X" question — override this to `false`. They remain
    /// fully reachable through the keyword scorer; they just can't be
    /// picked by something guessing at intent.
    fn router_eligible(&self) -> bool { true }

    /// Whether this skill holds the named host capability (snake_case, as
    /// spelled in the manifest — e.g. `"critical_alert"`). The engine uses
    /// this to refuse privileged envelope content a skill never declared:
    /// e.g. only a skill that declared `critical_alert` may emit a
    /// critical, full-takeover alert — anything else gets clamped down to
    /// an ordinary alert. Anything that doesn't override this holds no
    /// capabilities (builtins, declarative skills), which is fail-closed.
    fn has_capability(&self, _name: &str) -> bool { false }

    fn specificity(&self) -> Specificity;
    fn score(&self, input: &str, ctx: &SkillContext) -> f32;
    fn execute(&self, input: &str, ctx: &SkillContext) -> Response;

    /// Variant of [`execute`] for a dispatch that carries typed arguments
    /// extracted from the utterance. `args_json` is a JSON object string
    /// matching the skill's [`parameters_schema`] — e.g.
    /// `{"app_name":"Spotify"}` for the `open` skill, or
    /// `{"title":"call mum","when":"tomorrow at 3pm"}` for reminder.
    /// `input` is the raw (post-normalise) utterance, kept available for
    /// skills that want both the args and the original wording.
    ///
    /// **Nothing supplies args today**, so this is never called: parse what
    /// you need from `input`. It survives because the phrase banks record
    /// which `{slot}` fills which argument, so the phrase tier could start
    /// supplying them without any skill changing shape.
    ///
    /// Default impl ignores `args_json` and delegates to [`execute`]
    /// so existing skills are unaffected. Skills that want typed args
    /// override this method and read the JSON directly. The keyword
    /// scorer's matched skills always go through [`execute`] (no args
    /// to pass); only router-with-args matches call this entry point.
    fn execute_with_args(
        &self,
        input: &str,
        _args_json: &str,
        ctx: &SkillContext,
    ) -> Response {
        self.execute(input, ctx)
    }

    /// Example user utterances that should trigger this skill, paired with
    /// the arguments each `{slot}` fills. The phrase tier matches these
    /// against anything the keyword scorer didn't claim. A skill that
    /// doesn't override this is reachable by its keywords alone.
    ///
    /// Aim for 20-30 varied phrasings. Cover paraphrases, indirect
    /// language, and conversational filler. The point is that all the
    /// natural ways a user might phrase a request land on this skill, not
    /// just the rigid ones the keyword patterns catch.
    ///
    /// Store them normalised — see [`normalize_phrase`]. An un-normalised
    /// phrase is not an error anywhere; it simply never matches.
    fn example_utterances(&self) -> &[ExampleUtterance] { &[] }

    /// Locale-aware example utterances.
    ///
    /// Built-in skills override this to return per-locale phrases (English,
    /// Italian, …) via a `match locale`. The default returns the
    /// locale-agnostic [`example_utterances`](Skill::example_utterances), so a
    /// skill that has not localised its phrases keeps working (English).
    fn example_utterances_for(&self, _locale: &str) -> &[ExampleUtterance] {
        self.example_utterances()
    }

    /// Score from this skill's example phrases against already-normalised
    /// input, on the same 0..=1 scale as `score()`. The engine consults this
    /// only after the keyword tier found no winner, so a phrase never
    /// outranks an explicit trigger.
    ///
    /// The default matches [`example_utterances_for`](Skill::example_utterances_for),
    /// which covers the built-in skills. Declarative and WASM skills override
    /// it to read the phrases from their manifest. Skills that opted out of
    /// semantic routing keep their explicit triggers as the only way in.
    fn phrase_score(&self, normalized: &str, locale: &str) -> f32 {
        if !self.router_eligible() {
            return 0.0;
        }
        let phrases = self
            .example_utterances_for(locale)
            .iter()
            .map(|e| (e.text, e.weight));
        best_phrase_weight(phrases, normalized)
    }

    /// JSON schema describing this skill's parameters in OpenAI tool
    /// format. Shown to a cloud assistant when it routes. Default is
    /// `{"type": "object", "properties": {}}` for parameterless skills.
    /// Override for skills that take args.
    fn parameters_schema(&self) -> &str {
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

    /// Deliver the user's spoken reply to a question this skill previously
    /// asked (via an `await_reply` envelope). `context` is the opaque blob
    /// the skill stored on its question; `user_text` is the (normalized)
    /// reply. Mirrors `execute_continuation`: wraps both strings into a
    /// reserved JSON envelope routed through the single `execute` export, so
    /// WASM skills need no extra export and the host needs no new symbol.
    fn execute_reply(
        &self,
        context: &str,
        user_text: &str,
        ctx: &SkillContext,
    ) -> Response {
        let payload = serde_json::json!({
            "_ari_reply": {
                "context": context,
                "text": user_text,
            }
        });
        self.execute(&payload.to_string(), ctx)
    }

    /// Settings-time invocation (outside the utterance pipeline). `field` is the
    /// settings field key being queried; `values_json` is `{ "<dep_key>": "<val>", ... }`.
    /// Default: unsupported. WASM skills override to call their `settings_query` export.
    fn settings_query(&self, _field: &str, _values_json: &str) -> SettingsQueryResult {
        SettingsQueryResult::unsupported()
    }

    /// Effectful settings action (button press). Default: unsupported.
    fn settings_action(&self, _action: &str, _values_json: &str) -> SettingsQueryResult {
        SettingsQueryResult::unsupported()
    }

    /// Returns this skill's fallback-tier declaration, or `None` if it is not
    /// a fallback skill. Default: not a fallback.
    fn fallback_tier(&self) -> Option<FallbackTier> {
        None
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

/// Lowercase, locale-specific contraction/elision handling, punctuation
/// strip, locale-specific number-word replacement.
///
/// A comma flanked by digits survives the punctuation strip: it is a
/// number's own punctuation, not a sentence's. Italian writes decimals
/// that way ("3,14") and English writes thousands that way ("1,000"), so
/// blanking it turned both into two separate numbers.
///
/// Per-locale dispatch:
/// - `"en"` — expand English contractions, both the irregular whole words
///   (`"don't"` → `"do not"`, `"what's"` → `"what is"`) and the suffixes that
///   attach to any stem (`"I'll"` → `"i will"`, `"they've"` → `"they have"`);
///   fold dotted meridiems (`"p.m."` → `"pm"`) and replace English number
///   words (`"five"` → `"5"`).
/// - `"it"` — strip Italian apostrophe-elisions (`"l'ora"` → `"l ora"`,
///   `"c'è"` → `"c è"`). No contraction expansion or number words yet
///   (Italian number words are a Phase-7 polish item, alongside the
///   first Italian skill).
/// - Any other locale — lowercase + punctuation strip only.
///
/// Adding a new locale: extend the `match locale` block — the rest of
/// the pipeline (lowercase, punctuation strip) is locale-neutral.
pub fn normalize_input(input: &str, locale: &str) -> String {
    let lower = input.to_lowercase();

    let pre_clean = match locale {
        "en" => expand_english_contractions(&lower),
        "it" => strip_italian_elisions(&lower),
        _ => lower,
    };

    let chars: Vec<char> = pre_clean.chars().collect();
    let cleaned: String = chars
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            if c.is_alphanumeric() || c.is_whitespace() || "+-*/.%^".contains(c) {
                c
            } else if c == ',' && is_between_digits(&chars, i) {
                c
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<&str>>()
        .join(" ");

    match locale {
        "en" => replace_number_words(&collapse_meridiem(&cleaned)),
        _ => cleaned,
    }
}

/// Fold dotted meridiem tokens onto their bare forms: `"p.m."` → `"pm"`.
///
/// The punctuation strip deliberately keeps `.` so decimals survive, which
/// means STT output like "4 p.m." reaches skills with the dots intact. The
/// reminder skill's clock parser only knows `am`/`pm`, so "remind me to call
/// Penny Blue tomorrow at 4 p.m." filed at 4am with "p.m." stranded in the
/// title — and nothing in the residue scan flagged it, so it reported high
/// confidence and never asked.
fn collapse_meridiem(cleaned: &str) -> String {
    cleaned
        .split_whitespace()
        .map(|w| match w {
            "a.m." | "a.m" | "am." => "am",
            "p.m." | "p.m" | "pm." => "pm",
            _ => w,
        })
        .collect::<Vec<&str>>()
        .join(" ")
}

fn is_between_digits(chars: &[char], i: usize) -> bool {
    i > 0
        && chars[i - 1].is_ascii_digit()
        && chars.get(i + 1).is_some_and(|c| c.is_ascii_digit())
}

const CONTRACTIONS: &[(&str, &str)] = &[
    ("what's", "what is"),
    ("whats", "what is"),
    ("it's", "it is"),
    ("i'm", "i am"),
    ("don't", "do not"),
    ("doesn't", "does not"),
    ("can't", "cannot"),
    ("won't", "will not"),
    ("isn't", "is not"),
    ("aren't", "are not"),
    ("didn't", "did not"),
    ("there's", "there is"),
    ("here's", "here is"),
    ("that's", "that is"),
    ("let's", "let us"),
];

/// Contraction suffixes that attach to any stem, expanded wherever they close
/// a word.
///
/// These are listed as suffixes rather than whole words because the stem is
/// open-ended: "they'll", "that'll" and "mario'll" are one rule, and a table
/// of whole words would only ever cover the pronouns somebody thought of.
///
/// `'d` is the ambiguous one — it stands for "would" ("I'd like a reminder")
/// and for "had" ("I'd forgotten"), and telling them apart needs a parser we
/// do not have. Leaving it alone was worse: "i d" is wrong under both
/// readings. What an assistant is told is overwhelmingly the "would" sense,
/// so that is the guess, pinned by `apostrophe_d_always_reads_as_would`.
///
/// `'s` is deliberately absent: it is a possessive at least as often as it is
/// "is", and the whole words above already cover the ones worth guessing at.
const CONTRACTION_SUFFIXES: &[(&str, &str)] = &[
    ("'ll", " will"),
    ("'ve", " have"),
    ("'re", " are"),
    ("'d", " would"),
];

/// Whole words first, then the open-ended suffixes — the irregulars ("won't",
/// "can't") do not end in a suffix, so the two passes never fight over the
/// same word.
///
/// Every needle below is spelled with a plain apostrophe, and phone and
/// desktop keyboards autocorrect one into a typographic `\u{2019}`. Folding
/// that first is what makes the rules fire on text somebody typed rather than
/// spoke; without it "don\u{2019}t" fell through to the punctuation strip and
/// reached the skills as "don t".
fn expand_english_contractions(lower: &str) -> String {
    let mut out = lower.replace('\u{2019}', "'");
    for (from, to) in CONTRACTIONS {
        out = replace_whole_word(&out, from, to);
    }
    for (suffix, expansion) in CONTRACTION_SUFFIXES {
        out = expand_suffix(&out, suffix, expansion);
    }
    out
}

/// Expand a suffix wherever it closes a word: `"they'll"` → `"they will"`.
///
/// Two guards, and both earn their keep. The suffix needs an alphanumeric
/// stem, so a stray `"'d"` on its own is left alone. It also has to *end* the
/// word, or the `'d` rule cuts "o'donnell" down to "o would onnell" — and a
/// skill that messages people meets those names.
fn expand_suffix(hay: &str, suffix: &str, replacement: &str) -> String {
    let is_word = |c: char| c.is_alphanumeric() || c == '\'';
    let mut out = String::with_capacity(hay.len());
    let mut from = 0usize;
    while let Some(rel) = hay[from..].find(suffix) {
        let at = from + rel;
        let end = at + suffix.len();
        let has_stem = hay[..at]
            .chars()
            .next_back()
            .is_some_and(char::is_alphanumeric);
        let ends_word = hay[end..].chars().next().is_none_or(|c| !is_word(c));
        out.push_str(&hay[from..at]);
        if has_stem && ends_word {
            out.push_str(replacement);
        } else {
            out.push_str(suffix);
        }
        from = end;
    }
    out.push_str(&hay[from..]);
    out
}

/// Replace `needle` only where it stands as a whole word.
///
/// A plain `str::replace` here corrupts any word that merely *contains* a
/// contraction: the `whats` rule turned "whatsapp" into "what isapp", so every
/// utterance naming WhatsApp reached the skills mangled. Boundaries are
/// alphanumeric-or-apostrophe, so "what's" is one word rather than three.
fn replace_whole_word(hay: &str, needle: &str, replacement: &str) -> String {
    let is_word = |c: char| c.is_alphanumeric() || c == '\'';
    let mut out = String::with_capacity(hay.len());
    let mut from = 0usize;
    while let Some(rel) = hay[from..].find(needle) {
        let at = from + rel;
        let end = at + needle.len();
        let before_ok = hay[..at].chars().next_back().is_none_or(|c| !is_word(c));
        let after_ok = hay[end..].chars().next().is_none_or(|c| !is_word(c));
        out.push_str(&hay[from..at]);
        if before_ok && after_ok {
            out.push_str(replacement);
        } else {
            out.push_str(needle);
        }
        from = end;
    }
    out.push_str(&hay[from..]);
    out
}

/// Replace any apostrophe (or unicode right-single-quote) that's
/// flanked by alphabetic chars with a space. Handles Italian
/// elisions cleanly: `"l'ora"` → `"l ora"`, `"c'è"` → `"c è"`,
/// `"dell'amico"` → `"dell amico"`. The elided form would otherwise
/// fail keyword matches against `"ora"` or `"amico"`.
fn strip_italian_elisions(lower: &str) -> String {
    let chars: Vec<char> = lower.chars().collect();
    let mut out = String::with_capacity(lower.len());
    for (i, c) in chars.iter().enumerate() {
        let is_apostrophe = *c == '\'' || *c == '\u{2019}';
        if is_apostrophe
            && i > 0
            && i + 1 < chars.len()
            && chars[i - 1].is_alphabetic()
            && chars[i + 1].is_alphabetic()
        {
            out.push(' ');
        } else {
            out.push(*c);
        }
    }
    out
}

/// Split a `{slot}`-templated phrase into its literal parts. `n` slots yield
/// `n + 1` literals, any of which may be empty.
fn literal_parts(phrase: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut rest = phrase;
    while let Some(open) = rest.find('{') {
        let Some(close) = rest[open..].find('}') else { break };
        parts.push(&rest[..open]);
        rest = &rest[open + close + 1..];
    }
    parts.push(rest);
    parts
}

/// Whether `slot` swallowed something real — a slot must bind at least one
/// whole word, so `play {song}` does not match a bare "play".
fn slot_is_filled(slot: &str) -> bool {
    !slot.trim().is_empty()
}

/// Normalise an example phrase the way [`normalize_input`] normalises user
/// input, but leaving `{slot}` placeholders intact — plain normalisation
/// strips the braces, which would turn every slot into a literal word.
///
/// Phrases must be stored in this form for [`phrase_matches`] to fire:
/// the input it is matched against has already been through
/// `normalize_input`, so an unnormalised "whats the time" could never
/// meet the normalised "what is the time".
pub fn normalize_phrase(phrase: &str, locale: &str) -> String {
    let mut out = String::with_capacity(phrase.len());
    let mut rest = phrase;
    loop {
        let (literal, slot, tail) = match rest.find('{') {
            Some(open) => match rest[open..].find('}') {
                Some(close) => (
                    &rest[..open],
                    Some(&rest[open..open + close + 1]),
                    &rest[open + close + 1..],
                ),
                None => (rest, None, ""),
            },
            None => (rest, None, ""),
        };
        // Normalising drops the boundary spaces that separate a literal from
        // its neighbouring slot, so put them back.
        let lead = literal.starts_with(char::is_whitespace);
        let trail = literal.ends_with(char::is_whitespace);
        let body = normalize_input(literal, locale);
        if lead && !body.is_empty() {
            out.push(' ');
        }
        out.push_str(&body);
        if trail && !body.is_empty() {
            out.push(' ');
        } else if trail && body.is_empty() && !out.ends_with(' ') && !out.is_empty() {
            out.push(' ');
        }
        match slot {
            Some(s) => out.push_str(s),
            None => break,
        }
        rest = tail;
    }
    out
}

/// Match a `{slot}`-templated example phrase against already-normalised input.
///
/// Literals must appear in order and the match is anchored at both ends;
/// each `{slot}` binds one or more words. `play {song}` matches
/// "play hotel california" but neither "play" nor "shall i play something".
/// Phrases with no slots match the whole input exactly.
pub fn phrase_matches(phrase: &str, normalised: &str) -> bool {
    let parts = literal_parts(phrase);
    let (first, rest) = match parts.split_first() {
        Some(pair) => pair,
        None => return false,
    };
    if rest.is_empty() {
        return normalised == *first;
    }
    let mut cur = match normalised.strip_prefix(first) {
        Some(tail) => tail,
        None => return false,
    };
    let (last, mids) = rest.split_last().expect("rest is non-empty");
    for mid in mids {
        // Search past the first byte so the preceding slot cannot bind empty.
        let found = match cur.match_indices(*mid).find(|(i, _)| slot_is_filled(&cur[..*i])) {
            Some((i, _)) => i,
            None => return false,
        };
        cur = &cur[found + mid.len()..];
    }
    if last.is_empty() {
        slot_is_filled(cur)
    } else {
        match cur.strip_suffix(last) {
            Some(slot) => slot_is_filled(slot),
            None => false,
        }
    }
}

/// Highest weight among the example phrases matching `normalised`, or `0.0`.
/// Callers pass `(phrase, weight)` pairs so both the static built-in tables
/// and a manifest's owned strings can share one matcher.
pub fn best_phrase_weight<'a, I>(examples: I, normalised: &str) -> f32
where
    I: IntoIterator<Item = (&'a str, f32)>,
{
    examples
        .into_iter()
        .filter(|(phrase, _)| phrase_matches(phrase, normalised))
        .map(|(_, weight)| weight)
        .fold(0.0_f32, f32::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_phrase_expands_literals_and_keeps_slots() {
        assert_eq!(normalize_phrase("play {song}", "en"), "play {song}");
        assert_eq!(
            normalize_phrase("whats it saying on the clock", "en"),
            "what is it saying on the clock"
        );
        assert_eq!(
            normalize_phrase("i'd like {artist} on {service}", "en"),
            "i would like {artist} on {service}"
        );
        assert_eq!(normalize_phrase("{app} please", "en"), "{app} please");
    }

    #[test]
    fn normalized_phrase_matches_normalized_input() {
        let phrase = normalize_phrase("whats playing on {service}", "en");
        let input = normalize_input("What's playing on Spotify", "en");
        assert!(phrase_matches(&phrase, &input));
    }

    #[test]
    fn phrase_without_slots_matches_only_the_exact_input() {
        assert!(phrase_matches("pause the music", "pause the music"));
        assert!(!phrase_matches("pause the music", "pause the music now"));
        assert!(!phrase_matches("pause the music", "pause"));
    }

    #[test]
    fn trailing_slot_binds_one_or_more_words() {
        assert!(phrase_matches("play {song}", "play hotel california"));
        assert!(phrase_matches("play {song}", "play thriller"));
        assert!(!phrase_matches("play {song}", "play"));
        assert!(!phrase_matches("play {song}", "play "));
    }

    #[test]
    fn phrase_is_anchored_at_both_ends() {
        assert!(!phrase_matches("play {song}", "can you play thriller"));
        assert!(phrase_matches("can you play {song}", "can you play thriller"));
    }

    #[test]
    fn interior_slots_bind_between_literals() {
        assert!(phrase_matches(
            "play {song} on {service}",
            "play abbey road on spotify"
        ));
        assert!(!phrase_matches("play {song} on {service}", "play abbey road on"));
        assert!(!phrase_matches("play {song} on {service}", "play on spotify"));
    }

    #[test]
    fn leading_slot_binds_before_a_literal() {
        assert!(phrase_matches("{app} please open", "spotify please open"));
        assert!(!phrase_matches("{app} please open", "please open"));
    }

    #[test]
    fn best_phrase_weight_returns_the_highest_match_and_zero_for_none() {
        let examples = [("play {song}", 0.95_f32), ("play {artist}", 0.6_f32)];
        assert_eq!(best_phrase_weight(examples, "play thriller"), 0.95);
        assert_eq!(best_phrase_weight(examples, "what time is it"), 0.0);
    }

    // --- SkillContext / AppEntry ---

    #[test]
    fn skill_context_default_has_no_installed_apps() {
        let ctx = SkillContext::default();
        assert_eq!(ctx.locale, "en");
        assert!(ctx.installed_apps.is_empty());
    }

    #[test]
    fn app_entry_holds_label_and_package() {
        let a = AppEntry { label: "Spotify".to_string(), package: "com.spotify.music".to_string() };
        assert_eq!(a.label, "Spotify");
        assert_eq!(a.package, "com.spotify.music");
    }

    // --- remembered_facts_block ---

    #[test]
    fn remembered_facts_block_formats_bullets() {
        let facts = vec!["i am vegetarian".to_string(), "my wife is sara".to_string()];
        assert_eq!(
            crate::remembered_facts_block(&facts),
            Some("Things you know about the user:\n- i am vegetarian\n- my wife is sara".to_string())
        );
    }

    #[test]
    fn remembered_facts_block_empty_is_none() {
        assert_eq!(crate::remembered_facts_block(&[]), None);
    }

    // --- settings_query ---

    #[test]
    fn settings_query_result_has_refresh_default_false() {
        let r = SettingsQueryResult::unsupported();
        assert_eq!(r.refresh, false);
    }

    #[test]
    fn settings_query_default_is_unsupported() {
        struct S;
        impl Skill for S {
            fn id(&self) -> &str { "x" }
            fn specificity(&self) -> Specificity { Specificity::Low }
            fn score(&self, _: &str, _: &SkillContext) -> f32 { 0.0 }
            fn execute(&self, _: &str, _: &SkillContext) -> Response {
                Response::Text(String::new())
            }
        }
        let r = S.settings_query("agent_id", "{}");
        assert_eq!(r.ok, false);
        assert!(r.error.as_deref().unwrap_or("").contains("unsupported"));
    }

    // --- fallback_tier ---

    #[test]
    fn fallback_tier_default_is_none() {
        struct Bare;
        impl Skill for Bare {
            fn id(&self) -> &str { "bare" }
            fn specificity(&self) -> Specificity { Specificity::Low }
            fn score(&self, _: &str, _: &SkillContext) -> f32 { 0.0 }
            fn execute(&self, _: &str, _: &SkillContext) -> Response {
                Response::Text(String::new())
            }
        }
        assert!(Bare.fallback_tier().is_none());
    }

    // --- normalize_input ---

    #[test]
    fn normalize_lowercases() {
        assert_eq!(normalize_input("HELLO World", "en"), "hello world");
    }

    #[test]
    fn normalize_expands_all_contractions() {
        assert_eq!(normalize_input("what's", "en"), "what is");
        assert_eq!(normalize_input("whats", "en"), "what is");
        assert_eq!(normalize_input("it's", "en"), "it is");
        assert_eq!(normalize_input("i'm", "en"), "i am");
        assert_eq!(normalize_input("don't", "en"), "do not");
        assert_eq!(normalize_input("doesn't", "en"), "does not");
        assert_eq!(normalize_input("can't", "en"), "cannot");
        assert_eq!(normalize_input("won't", "en"), "will not");
        assert_eq!(normalize_input("isn't", "en"), "is not");
        assert_eq!(normalize_input("aren't", "en"), "are not");
        assert_eq!(normalize_input("didn't", "en"), "did not");
        assert_eq!(normalize_input("there's", "en"), "there is");
        assert_eq!(normalize_input("here's", "en"), "here is");
        assert_eq!(normalize_input("that's", "en"), "that is");
        assert_eq!(normalize_input("let's", "en"), "let us");
    }

    #[test]
    fn contraction_rules_only_fire_on_whole_words() {
        // "whats" → "what is" as a plain substring replace turned "whatsapp"
        // into "what isapp", so every utterance naming WhatsApp reached the
        // skills corrupted — and the message skill could never see which
        // service the user asked for.
        assert_eq!(
            normalize_input("tell mario hello on WhatsApp", "en"),
            "tell mario hello on whatsapp",
        );
        assert_eq!(normalize_input("whatsapp mario", "en"), "whatsapp mario");
    }

    #[test]
    fn whole_word_guard_does_not_break_real_contractions() {
        assert_eq!(normalize_input("whats the time", "en"), "what is the time");
        assert_eq!(normalize_input("what's the time", "en"), "what is the time");
        // Trailing punctuation is not a word character, so the rule still fires.
        assert_eq!(normalize_input("what's up?", "en"), "what is up");
    }

    #[test]
    fn normalize_expands_were_contraction() {
        assert_eq!(normalize_input("we're done", "en"), "we are done");
    }

    #[test]
    fn normalize_expands_contraction_suffixes() {
        assert_eq!(normalize_input("i'll be home soon", "en"), "i will be home soon");
        assert_eq!(normalize_input("they'll wait", "en"), "they will wait");
        assert_eq!(normalize_input("that'll do", "en"), "that will do");
        assert_eq!(normalize_input("i've eaten", "en"), "i have eaten");
        assert_eq!(normalize_input("should've asked", "en"), "should have asked");
        assert_eq!(normalize_input("you're late", "en"), "you are late");
        assert_eq!(normalize_input("they're here", "en"), "they are here");
    }

    #[test]
    fn suffix_rules_need_a_stem_that_ends_there() {
        // "o'donnell" contains "'d", and expanding it mid-word would hand the
        // message skill "o would onnell" as somebody's name.
        assert_eq!(normalize_input("message o'donnell", "en"), "message o donnell");
        assert_eq!(normalize_input("call o'brien", "en"), "call o brien");
        // No stem: nothing to expand, and the punctuation strip takes it.
        assert_eq!(normalize_input("'ll", "en"), "ll");
    }

    #[test]
    fn apostrophe_d_always_reads_as_would() {
        assert_eq!(normalize_input("i'd like a reminder", "en"), "i would like a reminder");
        // The "had" sense is expanded wrongly and knowingly: telling the two
        // apart needs a parser, and "i d forgotten" was wrong either way.
        assert_eq!(normalize_input("i'd forgotten", "en"), "i would forgotten");
    }

    #[test]
    fn typographic_apostrophes_expand_too() {
        // What a phone keyboard produces when somebody types instead of speaks.
        assert_eq!(normalize_input("I\u{2019}ll be home soon", "en"), "i will be home soon");
        assert_eq!(normalize_input("don\u{2019}t", "en"), "do not");
    }

    #[test]
    fn normalize_strips_punctuation_keeps_math() {
        assert_eq!(normalize_input("hello, world!", "en"), "hello world");
        assert_eq!(normalize_input("what?!", "en"), "what");
        assert_eq!(normalize_input("2 + 2", "en"), "2 + 2");
        assert_eq!(normalize_input("10 * 3.5", "en"), "10 * 3.5");
        assert_eq!(normalize_input("5 % 3", "en"), "5 % 3");
        assert_eq!(normalize_input("2^8", "en"), "2^8");
        assert_eq!(normalize_input("(1 + 2)", "en"), "1 + 2");
    }

    #[test]
    fn normalize_keeps_a_comma_between_digits() {
        // Italian decimals and English thousands both survive; the comma
        // strip used to split them into two numbers ("3 14", "1 000").
        assert_eq!(normalize_input("quanto fa 3,14 per 2", "it"), "quanto fa 3,14 per 2");
        assert_eq!(normalize_input("what is 1,000 plus 5", "en"), "what is 1,000 plus 5");
        // Sentence punctuation is still punctuation.
        assert_eq!(normalize_input("hello, world", "en"), "hello world");
        assert_eq!(normalize_input("ciao, 5 more", "en"), "ciao 5 more");
        assert_eq!(normalize_input("5, 6 and 7", "en"), "5 6 and 7");
        assert_eq!(normalize_input(",5", "en"), "5");
        assert_eq!(normalize_input("5,", "en"), "5");
    }

    #[test]
    fn normalize_folds_dotted_meridiems() {
        assert_eq!(
            normalize_input("remind me to call Penny Blue tomorrow at 4 p.m.", "en"),
            "remind me to call penny blue tomorrow at 4 pm"
        );
        assert_eq!(normalize_input("wake me at 6 a.m.", "en"), "wake me at 6 am");
        assert_eq!(normalize_input("at four p.m", "en"), "at 4 pm");
        assert_eq!(normalize_input("at 9 am.", "en"), "at 9 am");
        // Bare forms and non-meridiem words are untouched.
        assert_eq!(normalize_input("at 4 pm", "en"), "at 4 pm");
        assert_eq!(normalize_input("yes i am", "en"), "yes i am");
        assert_eq!(normalize_input("what is 3.5 times 2", "en"), "what is 3.5 times 2");
        // English-only dispatch — Italian keeps whatever it was given.
        assert_eq!(normalize_input("alle 4 p.m.", "it"), "alle 4 p.m.");
    }

    #[test]
    fn normalize_collapses_whitespace() {
        assert_eq!(normalize_input("  hello   world  ", "en"), "hello world");
        assert_eq!(normalize_input("\thello\tworld", "en"), "hello world");
    }

    #[test]
    fn normalize_empty_and_whitespace() {
        assert_eq!(normalize_input("", "en"), "");
        assert_eq!(normalize_input("   ", "en"), "");
        assert_eq!(normalize_input("!!!", "en"), "");
    }

    #[test]
    fn normalize_combined_contraction_and_number() {
        assert_eq!(normalize_input("what's two plus three", "en"), "what is 2 plus 3");
    }

    #[test]
    fn normalize_italian_strips_apostrophe_elisions() {
        // Definite article "l'" + vowel-initial noun → "l ora" so the
        // pattern matcher's "ora" keyword catches this.
        assert_eq!(normalize_input("l'ora", "it"), "l ora");
        // Multi-letter contraction: "dell'amico" → "dell amico".
        assert_eq!(normalize_input("dell'amico", "it"), "dell amico");
        // "c'è" with a non-ASCII grave-accented vowel should pass
        // through unchanged after the elision split.
        assert_eq!(normalize_input("c'è", "it"), "c è");
        // Right single quote (U+2019) is treated identically to ASCII apostrophe.
        assert_eq!(normalize_input("l\u{2019}ora", "it"), "l ora");
    }

    #[test]
    fn normalize_italian_does_not_expand_english_contractions() {
        // English contraction expansion is a per-locale concern;
        // an Italian utterance shouldn't get "what is" rewriting.
        assert_eq!(normalize_input("what's", "it"), "what s");
    }

    #[test]
    fn normalize_italian_does_not_run_english_number_words() {
        // Italian number-word handling is deliberately not implemented
        // yet (Phase 7 polish item); for now the cleaned string passes
        // through. English number-word replacement must NOT fire.
        assert_eq!(normalize_input("ho due ore", "it"), "ho due ore");
    }

    #[test]
    fn normalize_unknown_locale_falls_through_to_lowercase_and_clean() {
        // Some future locale we don't have a normaliser for yet:
        // lowercase + punctuation strip only, no contraction or
        // elision logic.
        assert_eq!(normalize_input("Hello, World!", "es"), "hello world");
        // English contractions do NOT expand for unknown locales.
        assert_eq!(normalize_input("what's", "es"), "what s");
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

    // --- execute_reply ---

    #[test]
    fn execute_reply_wraps_context_and_text_into_reserved_envelope() {
        use std::sync::Mutex;
        struct Capture {
            seen: Mutex<String>,
        }
        impl Skill for Capture {
            fn id(&self) -> &str { "cap" }
            fn specificity(&self) -> Specificity { Specificity::Low }
            fn score(&self, _: &str, _: &SkillContext) -> f32 { 0.0 }
            fn execute(&self, input: &str, _: &SkillContext) -> Response {
                *self.seen.lock().unwrap() = input.to_string();
                Response::Text(String::new())
            }
        }
        let s = Capture { seen: Mutex::new(String::new()) };
        let ctx = SkillContext::default();
        s.execute_reply("ctx-blob", "spotify", &ctx);
        let seen = s.seen.lock().unwrap().clone();
        let v: serde_json::Value = serde_json::from_str(&seen).unwrap();
        assert_eq!(v["_ari_reply"]["context"], "ctx-blob");
        assert_eq!(v["_ari_reply"]["text"], "spotify");
    }

    #[test]
    fn continuation_instruction_mentions_both_markers() {
        assert!(crate::CONTINUATION_INSTRUCTION.contains("[continuation]"));
        assert!(crate::CONTINUATION_INSTRUCTION.contains("[new]"));
        assert!(!crate::CONTINUATION_INSTRUCTION.is_empty());
    }
}
