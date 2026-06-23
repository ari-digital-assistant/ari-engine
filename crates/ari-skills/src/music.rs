use ari_core::{ExampleUtterance, Response, Skill, SkillContext, Specificity};

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
        // Task 3 swaps this for the service-aware query check; for now any
        // post-trigger content counts as a query.
        match after_trigger(input) {
            Some(_) => 0.9,
            None => 0.3,
        }
    }

    fn execute(&self, input: &str, ctx: &SkillContext) -> Response {
        match after_trigger(input) {
            Some(query) => play_action(query, None),
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
}
