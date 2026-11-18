//! Suggests a store skill the user hasn't installed, when what they said
//! matches one of its strong trigger patterns (ari-android#44). Without
//! this, "remind me to call Mum" on a phone without the reminder skill goes
//! to the assistant, which can only say it can't set reminders.
//!
//! The patterns come from the registry's `index.json`, which the store
//! caches on the phone each time it fetches it, so suggesting works offline
//! and costs no network round trip.

use crate::manifest::{MatchPattern, Matching};
use crate::registry::{Index, IndexEntry};
use crate::scoring::PatternScorer;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::SystemTime;

/// Only a skill's strong patterns suggest it. The loose ones exist to win a
/// late ranking round among installed skills: reminder's `^add .+ to .+`
/// (0.65) would suggest a reminder for "add cream to the coffee".
pub const MIN_SUGGEST_WEIGHT: f32 = 0.85;

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
    /// Per locale, only the strong patterns.
    scorers: HashMap<String, PatternScorer>,
}

/// The suggestible skills in one registry index: regular skills (never
/// assistants) with at least one strong pattern in some language.
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

    /// The skill whose strong patterns match `normalised` input best in
    /// `locale`, skipping any that `installed` says the phone already has.
    /// A skill with no patterns in `locale` doesn't speak it and is never
    /// suggested there.
    pub fn suggest(
        &self,
        normalised: &str,
        locale: &str,
        installed: impl Fn(&str) -> bool,
    ) -> Option<SkillSuggestion> {
        let mut best: Option<(f32, &Candidate)> = None;
        for c in &self.candidates {
            let Some(scorer) = c.scorers.get(locale) else { continue };
            let score = scorer.score_normalised(normalised);
            if score >= MIN_SUGGEST_WEIGHT
                && best.is_none_or(|(b, _)| score > b)
                && !installed(&c.id)
            {
                best = Some((score, c));
            }
        }
        best.map(|(_, c)| SkillSuggestion {
            id: c.id.clone(),
            name: c.localized_names.get(locale).cloned().unwrap_or_else(|| c.name.clone()),
        })
    }
}

fn candidate(entry: &IndexEntry) -> Option<Candidate> {
    let mut scorers = HashMap::new();
    for (locale, patterns) in &entry.matching {
        // Each pattern vetted on its own, so one bad regex costs only
        // itself rather than the skill's whole language.
        let strong: Vec<MatchPattern> = patterns
            .iter()
            .filter(|p| p.weight >= MIN_SUGGEST_WEIGHT)
            .filter_map(|p| p.to_match_pattern())
            .filter(|p| PatternScorer::compile(&matching_of(vec![p.clone()]), &[], locale).is_ok())
            .collect();
        if strong.is_empty() {
            continue;
        }
        if let Ok(scorer) = PatternScorer::compile(&matching_of(strong), &[], locale) {
            scorers.insert(locale.clone(), scorer);
        }
    }
    if scorers.is_empty() {
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
        scorers,
    })
}

fn matching_of(patterns: Vec<MatchPattern>) -> Matching {
    Matching { patterns, custom_score: false }
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
        locale: &str,
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
        loaded.1.as_ref()?.suggest(normalised, locale, installed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn index_json(skills: &str) -> String {
        format!(r#"{{"index_version": 1, "generated_at": "now", "skills": [{skills}]}}"#)
    }

    fn entry(id: &str, kind: &str, matching: &str) -> String {
        format!(
            r#"{{"id": "{id}", "version": "1.0.0", "name": "{name}", "description": "", "type": "{kind}",
                "bundle": "b", "signature": "s", "sha256": "h",
                "localizations": {{"it": {{"name": "promemoria", "description": ""}}}},
                "matching": {matching}}}"#,
            name = id.rsplit('.').next().unwrap()
        )
    }

    const REMINDER: &str = r#"{
        "en": [{"regex": "\\bremind me\\b", "weight": 0.95}, {"regex": "^add .+ to .+", "weight": 0.65}],
        "it": [{"regex": "\\bricordami\\b", "weight": 0.95}]
    }"#;

    fn suggester(skills: &str) -> SkillSuggester {
        let index: Index = serde_json::from_str(&index_json(skills)).unwrap();
        SkillSuggester::from_index(&index)
    }

    fn reminder_index() -> SkillSuggester {
        suggester(&entry("dev.heyari.reminder", "skill", REMINDER))
    }

    #[test]
    fn suggests_a_skill_its_strong_pattern_matches() {
        let s = reminder_index();
        let hit = s.suggest("remind me to call mum", "en", |_| false).unwrap();
        assert_eq!(hit.id, "dev.heyari.reminder");
        assert_eq!(hit.name, "reminder");
    }

    #[test]
    fn weak_patterns_never_suggest() {
        // ari-android#44: the 0.65 catch-all would offer a reminder for this.
        assert_eq!(reminder_index().suggest("add cream to the coffee", "en", |_| false), None);
    }

    #[test]
    fn installed_skills_are_not_suggested() {
        let s = reminder_index();
        assert_eq!(s.suggest("remind me to call mum", "en", |id| id == "dev.heyari.reminder"), None);
    }

    #[test]
    fn uses_the_users_language_and_its_name() {
        let s = reminder_index();
        assert_eq!(s.suggest("ricordami di chiamare la mamma", "it", |_| false).unwrap().name, "promemoria");
        // English patterns don't fire for an Italian user, or the reverse.
        assert_eq!(s.suggest("remind me to call mum", "it", |_| false), None);
        assert_eq!(s.suggest("remind me to call mum", "fr", |_| false), None);
    }

    #[test]
    fn assistants_are_never_suggested() {
        let s = suggester(&entry("dev.heyari.assistant.chatgpt", "assistant", r#"{"en": [{"regex": "\\bremind me\\b", "weight": 1.0}]}"#));
        assert!(s.is_empty());
    }

    #[test]
    fn a_bad_regex_costs_only_itself() {
        let s = suggester(&entry(
            "dev.heyari.reminder",
            "skill",
            r#"{"en": [{"regex": "(unclosed", "weight": 0.95}, {"regex": "\\bremind me\\b", "weight": 0.95}]}"#,
        ));
        assert!(s.suggest("remind me", "en", |_| false).is_some());
    }

    #[test]
    fn an_old_index_without_matching_suggests_nothing() {
        let old = r#"{"id": "dev.heyari.reminder", "version": "1", "name": "reminder", "description": "",
                      "bundle": "b", "signature": "s", "sha256": "h"}"#;
        assert!(suggester(old).is_empty());
    }

    #[test]
    fn cached_suggester_reads_the_file_and_copes_without_one() {
        let nanos = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos();
        let dir = std::env::temp_dir().join(format!("ari-suggest-{nanos}"));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("index.json");
        let cached = CachedSkillSuggester::new(&path);
        assert_eq!(cached.suggest("remind me", "en", |_| false), None);

        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(index_json(&entry("dev.heyari.reminder", "skill", REMINDER)).as_bytes()).unwrap();
        drop(f);
        assert_eq!(cached.suggest("remind me", "en", |_| false).unwrap().id, "dev.heyari.reminder");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
