use ari_core::{Response, Skill, SkillContext, Specificity};

/// Trigger phrases, longest first so a multi-word phrase wins over any
/// single word it might contain. Matched against post-normalised input
/// (lowercase, no punctuation). EN + IT source strings only.
const TRIGGERS: &[&str] = &[
    // multi-word
    "listen to", "put on", "fai partire",
    // single word — EN "play"; IT metti/riproduci/ascolta/suona
    "play", "riproduci", "metti", "ascolta", "suona",
];

/// True when any trigger appears as a whole word/phrase (even with nothing
/// after it — that's the "play" → clarification case).
fn has_trigger(input: &str) -> bool {
    let bytes = input.as_bytes();
    for trig in TRIGGERS {
        let mut from = 0;
        while let Some(pos) = input[from..].find(trig) {
            let abs = from + pos;
            let before_ok = abs == 0 || bytes[abs - 1] == b' ';
            let end = abs + trig.len();
            let after_ok = end == input.len() || bytes[end] == b' ';
            if before_ok && after_ok {
                return true;
            }
            from = abs + 1;
        }
    }
    false
}

/// The raw text after the first trigger phrase (trimmed), or `None` when no
/// trigger is followed by content. This is the query *before* service
/// stripping (Task 3 refines it).
fn after_trigger(input: &str) -> Option<&str> {
    let bytes = input.as_bytes();
    for trig in TRIGGERS {
        let mut from = 0;
        while let Some(pos) = input[from..].find(trig) {
            let abs = from + pos;
            let before_ok = abs == 0 || bytes[abs - 1] == b' ';
            let end = abs + trig.len();
            let followed_by_space = bytes.get(end) == Some(&b' ');
            if before_ok && followed_by_space {
                let rest = input[end + 1..].trim();
                if !rest.is_empty() {
                    return Some(rest);
                }
            }
            from = abs + 1;
        }
    }
    None
}

/// Build the `play_media` action envelope. `service` is omitted entirely
/// when `None` (default-service play).
fn play_action(query: &str, service: Option<&str>) -> Response {
    let mut media = serde_json::json!({ "query": query });
    if let Some(s) = service {
        media["service"] = serde_json::Value::String(s.to_string());
    }
    Response::Action(serde_json::json!({ "v": 1, "play_media": media }))
}

fn clarify(ctx: &SkillContext) -> Response {
    Response::Text(
        match ctx.locale.as_str() {
            "it" => "Cosa vuoi ascoltare?",
            _ => "What would you like me to play?",
        }
        .to_string(),
    )
}

/// Closed set of canonical service ids emitted in the action.
const VALID_SERVICE_IDS: &[&str] = &[
    "spotify", "apple_music", "youtube_music", "tidal", "deezer", "youtube", "amazon_music",
];

/// (canonical id, recognised alias). Longest aliases first so
/// "youtube music" wins over "youtube".
const SERVICE_ALIASES: &[(&str, &str)] = &[
    ("youtube_music", "youtube music"),
    ("youtube_music", "yt music"),
    ("amazon_music", "amazon music"),
    ("apple_music", "apple music"),
    ("spotify", "spotify"),
    ("tidal", "tidal"),
    ("deezer", "deezer"),
    ("youtube", "youtube"),
];

/// Connector words before a service name: EN "on", IT "su".
const SERVICE_CONNECTORS: &[&str] = &["on", "su"];

/// Resolve a free-text service name (alias or canonical id, any case) to a
/// canonical id. `None` for anything not in the closed set.
fn canonical_service(s: &str) -> Option<String> {
    let s = s.trim().to_lowercase();
    if VALID_SERVICE_IDS.contains(&s.as_str()) {
        return Some(s);
    }
    SERVICE_ALIASES
        .iter()
        .find(|(_, alias)| *alias == s)
        .map(|(id, _)| (*id).to_string())
}

/// Split a raw query into `(query, service_id)`. Only strips a trailing
/// "<connector> <known alias>" — so a song title containing "on" survives.
/// Also handles the case where raw is exactly "<connector> <alias>" (no
/// preceding query), returning an empty query so the caller can clarify.
fn split_service(raw: &str) -> (String, Option<String>) {
    for conn in SERVICE_CONNECTORS {
        for (id, alias) in SERVICE_ALIASES {
            // Case 1: query before the connector — " on <alias>" suffix.
            let suffix = format!(" {conn} {alias}");
            if let Some(stripped) = raw.strip_suffix(&suffix) {
                return (stripped.trim().to_string(), Some((*id).to_string()));
            }
            // Case 2: raw IS "<connector> <alias>" with no preceding query.
            let exact = format!("{conn} {alias}");
            if raw == exact {
                return (String::new(), Some((*id).to_string()));
            }
        }
    }
    (raw.trim().to_string(), None)
}

pub struct MusicSkill;

impl MusicSkill {
    pub fn new() -> Self { Self }
}
impl Default for MusicSkill {
    fn default() -> Self { Self::new() }
}

impl Skill for MusicSkill {
    fn id(&self) -> &str {
        "music"
    }

    fn description(&self) -> &str {
        "Plays music by name in a music app. Use when the user wants to play, put on, or listen to a song, artist, album, or playlist — e.g. 'play hotel california', 'put on some pink floyd', 'listen to jazz'. The user may name a service with 'on <service>' (Spotify, Apple Music, YouTube Music, Tidal, Deezer, YouTube, Amazon Music). The query is what to play; service is the optional app to play it on."
    }

    fn specificity(&self) -> Specificity {
        Specificity::Medium
    }

    fn score(&self, input: &str, _ctx: &SkillContext) -> f32 {
        if !has_trigger(input) {
            return 0.0;
        }
        match after_trigger(input) {
            Some(raw) if !split_service(raw).0.is_empty() => 0.9,
            _ => 0.3,
        }
    }

    fn execute(&self, input: &str, ctx: &SkillContext) -> Response {
        match after_trigger(input) {
            Some(raw) => {
                let (query, service) = split_service(raw);
                if query.is_empty() {
                    clarify(ctx)
                } else {
                    play_action(&query, service.as_deref())
                }
            }
            None => clarify(ctx),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> SkillContext { SkillContext::default() }

    #[test]
    fn score_trigger_plus_query_is_high() {
        assert_eq!(MusicSkill::new().score("play hotel california", &ctx()), 0.9);
        assert_eq!(MusicSkill::new().score("put on comfortably numb", &ctx()), 0.9);
        assert_eq!(MusicSkill::new().score("listen to the beatles", &ctx()), 0.9);
    }

    #[test]
    fn score_trigger_alone_is_low() {
        assert_eq!(MusicSkill::new().score("play", &ctx()), 0.3);
    }

    #[test]
    fn score_no_trigger_is_zero() {
        assert_eq!(MusicSkill::new().score("what time is it", &ctx()), 0.0);
        assert_eq!(MusicSkill::new().score("hotel california", &ctx()), 0.0);
    }

    #[test]
    fn execute_default_service_emits_query_only() {
        match MusicSkill::new().execute("play hotel california", &ctx()) {
            Response::Action(v) => {
                assert_eq!(v["v"], 1);
                assert_eq!(v["play_media"]["query"], "hotel california");
                assert!(v["play_media"].get("service").is_none());
                assert!(v.get("speak").is_none());
            }
            other => panic!("expected Action, got {other:?}"),
        }
    }

    #[test]
    fn execute_no_query_asks_clarification() {
        match MusicSkill::new().execute("play", &ctx()) {
            Response::Text(s) => assert_eq!(s, "What would you like me to play?"),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn execute_named_service_splits_query_and_service() {
        match MusicSkill::new().execute("play hotel california on spotify", &ctx()) {
            Response::Action(v) => {
                assert_eq!(v["play_media"]["query"], "hotel california");
                assert_eq!(v["play_media"]["service"], "spotify");
            }
            other => panic!("expected Action, got {other:?}"),
        }
    }

    #[test]
    fn youtube_music_beats_bare_youtube() {
        match MusicSkill::new().execute("play lofi beats on youtube music", &ctx()) {
            Response::Action(v) => {
                assert_eq!(v["play_media"]["query"], "lofi beats");
                assert_eq!(v["play_media"]["service"], "youtube_music");
            }
            other => panic!("expected Action, got {other:?}"),
        }
    }

    #[test]
    fn song_title_containing_on_is_not_split() {
        // "heavens door" is not a known service → no split.
        match MusicSkill::new().execute("play knockin on heavens door", &ctx()) {
            Response::Action(v) => {
                assert_eq!(v["play_media"]["query"], "knockin on heavens door");
                assert!(v["play_media"].get("service").is_none());
            }
            other => panic!("expected Action, got {other:?}"),
        }
    }

    #[test]
    fn italian_su_connector_splits_service() {
        match MusicSkill::new().execute("metti hotel california su spotify", &ctx()) {
            Response::Action(v) => {
                assert_eq!(v["play_media"]["query"], "hotel california");
                assert_eq!(v["play_media"]["service"], "spotify");
            }
            other => panic!("expected Action, got {other:?}"),
        }
    }

    #[test]
    fn service_only_no_query_asks_clarification() {
        match MusicSkill::new().execute("play on spotify", &ctx()) {
            Response::Text(s) => assert_eq!(s, "What would you like me to play?"),
            other => panic!("expected Text, got {other:?}"),
        }
        assert_eq!(MusicSkill::new().score("play on spotify", &ctx()), 0.3);
    }

    #[test]
    fn canonical_service_resolves_aliases_and_case() {
        assert_eq!(canonical_service("Spotify"), Some("spotify".to_string()));
        assert_eq!(canonical_service("apple music"), Some("apple_music".to_string()));
        assert_eq!(canonical_service("yt music"), Some("youtube_music".to_string()));
        assert_eq!(canonical_service("pandora"), None);
    }
}
