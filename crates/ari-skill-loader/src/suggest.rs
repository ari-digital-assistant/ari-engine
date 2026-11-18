//! Suggests a store skill the user hasn't installed, when what they said
//! matches one of the phrases the skill lists as clearly needing it
//! (ari-android#44). Without this, "remind me to call Mum" on a phone without
//! the reminder skill goes to the assistant, which can only say it can't set
//! reminders.
//!
//! Opt-in: only a skill's `suggest:` phrases count, never its matching
//! patterns or examples. Those are written broad, because an installed skill
//! turns down what isn't for it; an uninstalled one can't, so the message
//! skill's "tell …" pattern would offer it for "tell me a joke".
//!
//! The phrases come from the registry's `index.json`, which the store caches
//! on the phone each time it fetches it, so suggesting works offline and costs
//! no network round trip.

use crate::registry::{Index, IndexEntry};
use ari_core::{normalize_phrase, phrase_matches_in, SkillContext};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::SystemTime;

/// A store skill worth offering.
#[derive(Debug, Clone, PartialEq)]
pub struct SkillSuggestion {
    pub id: String,
    /// In the user's language when the skill ships a translation.
    pub name: String,
}

struct Candidate {
    id: String,
    name: String,
    localized_names: HashMap<String, String>,
    /// Per locale, the suggest phrases normalised as input is.
    phrases: HashMap<String, Vec<String>>,
}

/// The suggestible skills in one registry index: regular skills (never
/// assistants) that list suggest phrases in some language.
pub struct SkillSuggester {
    candidates: Vec<Candidate>,
}

impl SkillSuggester {
    pub fn from_index(index: &Index) -> Self {
        let candidates = index
            .skills
            .iter()
            .filter(|e| e.skill_type == "skill")
            .filter_map(candidate)
            .collect();
        SkillSuggester { candidates }
    }

    /// How many skills could be suggested in some language.
    pub fn len(&self) -> usize {
        self.candidates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }

    /// The skill one of whose phrases matches the whole of `normalised`
    /// input in `ctx`'s language, skipping any that `installed` says the
    /// phone already has. When several match, the phrase with the most
    /// words of its own wins, as the more specific. A skill with no
    /// phrases in the user's language is never suggested in it.
    pub fn suggest(
        &self,
        normalised: &str,
        ctx: &SkillContext,
        installed: impl Fn(&str) -> bool,
    ) -> Option<SkillSuggestion> {
        let mut best: Option<(usize, &Candidate)> = None;
        for c in &self.candidates {
            let Some(phrases) = c.phrases.get(&ctx.locale) else { continue };
            let Some(specific) = phrases
                .iter()
                .filter(|p| phrase_matches_in(p, normalised, ctx))
                .map(|p| literal_len(p))
                .max()
            else {
                continue;
            };
            if best.is_none_or(|(b, _)| specific > b) && !installed(&c.id) {
                best = Some((specific, c));
            }
        }
        best.map(|(_, c)| SkillSuggestion {
            id: c.id.clone(),
            name: c
                .localized_names
                .get(&ctx.locale)
                .cloned()
                .unwrap_or_else(|| c.name.clone()),
        })
    }
}

/// Characters outside the `{slots}`: how much of the phrase is fixed words.
fn literal_len(phrase: &str) -> usize {
    let mut depth = 0u32;
    phrase
        .chars()
        .filter(|c| match c {
            '{' => {
                depth += 1;
                false
            }
            '}' => {
                depth = depth.saturating_sub(1);
                false
            }
            _ => depth == 0,
        })
        .count()
}

fn candidate(entry: &IndexEntry) -> Option<Candidate> {
    let phrases: HashMap<String, Vec<String>> = entry
        .suggest
        .iter()
        .map(|(locale, phrases)| {
            let normalised = phrases
                .iter()
                // The manifest refuses all-slot phrases; an index is only
                // as careful as whoever built it, so check again.
                .filter(|p| literal_len(p.trim()) > 0 && p.chars().any(char::is_alphanumeric))
                .map(|p| normalize_phrase(p, locale))
                .collect::<Vec<_>>();
            (locale.clone(), normalised)
        })
        .filter(|(_, phrases)| !phrases.is_empty())
        .collect();
    if phrases.is_empty() {
        return None;
    }
    Some(Candidate {
        id: entry.id.clone(),
        name: entry.name.clone(),
        localized_names: entry
            .localizations
            .iter()
            .map(|(locale, d)| (locale.clone(), d.name.clone()))
            .collect(),
        phrases,
    })
}

/// A [`SkillSuggester`] over the index file the store caches, read again
/// whenever the file changes, so a Browse visit or the daily update check
/// refreshes suggestions without anyone telling the engine. No file, or one
/// that won't parse, means no suggestions.
pub struct CachedSkillSuggester {
    path: PathBuf,
    loaded: Mutex<(Option<SystemTime>, Option<SkillSuggester>)>,
}

impl CachedSkillSuggester {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        CachedSkillSuggester { path: path.into(), loaded: Mutex::new((None, None)) }
    }

    /// [`SkillSuggester::suggest`] over the cached index as it is now.
    pub fn suggest(
        &self,
        normalised: &str,
        ctx: &SkillContext,
        installed: impl Fn(&str) -> bool,
    ) -> Option<SkillSuggestion> {
        let modified = std::fs::metadata(&self.path).and_then(|m| m.modified()).ok();
        let mut loaded = self.loaded.lock().ok()?;
        if loaded.0 != modified {
            loaded.1 = modified
                .and_then(|_| std::fs::read(&self.path).ok())
                .and_then(|bytes| serde_json::from_slice::<Index>(&bytes).ok())
                .map(|index| SkillSuggester::from_index(&index));
            loaded.0 = modified;
        }
        loaded.1.as_ref()?.suggest(normalised, ctx, installed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ari_core::normalize_input;
    use std::io::Write;

    fn index_json(skills: &str) -> String {
        format!(r#"{{"index_version": 1, "generated_at": "now", "skills": [{skills}]}}"#)
    }

    fn entry(id: &str, kind: &str, suggest: &str) -> String {
        format!(
            r#"{{"id": "{id}", "version": "1.0.0", "name": "{name}", "description": "", "type": "{kind}",
                "bundle": "b", "signature": "s", "sha256": "h",
                "localizations": {{"it": {{"name": "promemoria", "description": ""}}}},
                "suggest": {suggest}}}"#,
            name = id.rsplit('.').next().unwrap()
        )
    }

    const REMINDER: &str = r#"{
        "en": ["remind me", "remind me to {text}", "set a reminder"],
        "it": ["ricordami di {text}"]
    }"#;

    fn suggester(skills: &str) -> SkillSuggester {
        let index: Index = serde_json::from_str(&index_json(skills)).unwrap();
        SkillSuggester::from_index(&index)
    }

    fn reminder_index() -> SkillSuggester {
        suggester(&entry("dev.heyari.reminder", "skill", REMINDER))
    }

    fn ctx(locale: &str) -> SkillContext {
        SkillContext { locale: locale.to_string(), ..SkillContext::default() }
    }

    fn suggest(s: &SkillSuggester, said: &str, locale: &str) -> Option<SkillSuggestion> {
        s.suggest(&normalize_input(said, locale), &ctx(locale), |_| false)
    }

    #[test]
    fn suggests_a_skill_whose_phrase_matches_the_whole_utterance() {
        let s = reminder_index();
        let hit = suggest(&s, "Remind me to call Mum.", "en").unwrap();
        assert_eq!(hit.id, "dev.heyari.reminder");
        assert_eq!(hit.name, "reminder");
        // The bare request the issue started from.
        assert!(suggest(&s, "remind me", "en").is_some());
    }

    #[test]
    fn a_phrase_must_cover_the_whole_utterance() {
        // "remind me" appearing inside something else is not a request for it.
        assert_eq!(suggest(&reminder_index(), "can you remind me what we said", "en"), None);
        assert_eq!(suggest(&reminder_index(), "tell me a joke", "en"), None);
    }

    #[test]
    fn installed_skills_are_not_suggested() {
        let s = reminder_index();
        let said = normalize_input("remind me to call mum", "en");
        assert_eq!(s.suggest(&said, &ctx("en"), |id| id == "dev.heyari.reminder"), None);
    }

    #[test]
    fn uses_the_users_language_and_its_name() {
        let s = reminder_index();
        assert_eq!(suggest(&s, "ricordami di chiamare la mamma", "it").unwrap().name, "promemoria");
        assert_eq!(suggest(&s, "remind me to call mum", "it"), None);
        assert_eq!(suggest(&s, "remind me to call mum", "fr"), None);
    }

    #[test]
    fn the_more_specific_phrase_wins() {
        let s = suggester(&format!(
            "{},{}",
            entry("dev.example.vague", "skill", r#"{"en": ["set a {thing}"]}"#),
            entry("dev.heyari.reminder", "skill", REMINDER)
        ));
        assert_eq!(suggest(&s, "set a reminder", "en").unwrap().id, "dev.heyari.reminder");
    }

    #[test]
    fn skills_without_phrases_and_assistants_are_never_suggested() {
        let s = suggester(&format!(
            "{},{}",
            entry("dev.heyari.message", "skill", "{}"),
            entry("dev.heyari.assistant.chatgpt", "assistant", r#"{"en": ["remind me"]}"#)
        ));
        assert!(s.is_empty());
    }

    #[test]
    fn an_all_slot_phrase_in_the_index_is_ignored() {
        let s = suggester(&entry("dev.example.greedy", "skill", r#"{"en": ["{anything}"]}"#));
        assert!(s.is_empty());
    }

    #[test]
    fn an_old_index_without_suggest_suggests_nothing() {
        let old = r#"{"id": "dev.heyari.reminder", "version": "1", "name": "reminder", "description": "",
                      "bundle": "b", "signature": "s", "sha256": "h"}"#;
        assert!(suggester(old).is_empty());
    }

    /// Prints what a real index suggests for everyday utterances, for eyeballing
    /// a skill's list before publishing it:
    /// `SUGGEST_INDEX=index.json cargo test -p ari-skill-loader real_index -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn real_index_suggestions() {
        let Ok(path) = std::env::var("SUGGEST_INDEX") else { return };
        let index: Index = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let s = SkillSuggester::from_index(&index);
        let said = [
            ("en", "remind me"), ("en", "remind me to buy milk"), ("en", "Remind me tomorrow to call Mum"),
            ("en", "tell me a joke"), ("en", "tell me about the roman empire"), ("en", "why is the sky blue"),
            ("en", "what is the difference between weather and climate"), ("en", "what's the weather like?"),
            ("en", "will it rain tomorrow"), ("en", "set a timer for 5 minutes"), ("en", "play some music"),
            ("en", "play twenty questions with me"), ("en", "text Mum I'm running late"), ("en", "send a message to Dad"),
            ("en", "turn on the kitchen lights"), ("en", "turn on bluetooth"), ("en", "take me home"),
            ("en", "flip a coin"), ("en", "wake me up at 7"), ("en", "can you remind me what we talked about"),
            ("en", "add cream to the coffee"), ("en", "add milk to my shopping list"),
            ("it", "ricordami di comprare il latte"), ("it", "quanto tempo ci vuole per arrivare a Roma"),
            ("it", "che tempo fa"), ("it", "che tempo fa a Milano?"), ("it", "raccontami una barzelletta"),
            ("it", "manda un messaggio a Marco"), ("it", "accendi le luci"), ("it", "portami a casa"),
            ("it", "metti un po' di musica"), ("it", "imposta un timer di 10 minuti"), ("it", "svegliami alle 7"),
        ];
        for (locale, text) in said {
            let ctx = SkillContext { locale: locale.into(), ..SkillContext::default() };
            let hit = s.suggest(&normalize_input(text, locale), &ctx, |_| false);
            eprintln!("{locale} {text:52} -> {}", hit.map(|h| h.id).unwrap_or_else(|| "-".into()));
        }
    }

    #[test]
    fn cached_suggester_reads_the_file_and_copes_without_one() {
        let nanos = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos();
        let dir = std::env::temp_dir().join(format!("ari-suggest-{nanos}"));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("index.json");
        let cached = CachedSkillSuggester::new(&path);
        assert_eq!(cached.suggest("remind me", &ctx("en"), |_| false), None);

        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(index_json(&entry("dev.heyari.reminder", "skill", REMINDER)).as_bytes()).unwrap();
        drop(f);
        assert_eq!(cached.suggest("remind me", &ctx("en"), |_| false).unwrap().id, "dev.heyari.reminder");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
