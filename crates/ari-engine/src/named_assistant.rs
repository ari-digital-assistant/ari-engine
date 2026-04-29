//! "Ask <assistant> X" prefix routing.
//!
//! Sits at the very top of the engine pipeline (before keyword scoring,
//! router, and active-assistant fallback). Lets a user address any
//! installed assistant skill by name regardless of which assistant is
//! currently active. Deterministic, no model round-trip.
//!
//! All matching runs against post-`normalize_input` text — lowercase,
//! contractions expanded, punctuation stripped — so aliases must
//! likewise be lowercase ASCII alphanum + spaces (validated at manifest
//! parse time in `ari_skill_loader::manifest::validate_alias`).

use ari_skill_loader::assistant::ConfigStore;
use ari_skill_loader::manifest::ApiConfig;
use std::sync::Arc;

/// Words skipped at the start of an utterance before we look for a
/// verb. Keeps polite/conversational openings from blocking the match.
const FLUFF_WORDS: &[&str] = &[
    "hey", "ok", "okay", "please", "can", "could", "you", "would",
];

/// Verbs that introduce an assistant request. Optional — bare-prefix
/// utterances ("claude what is X") work too. At most one verb is
/// consumed per match.
const VERBS: &[&str] = &["ask", "tell", "use", "get"];

/// Connectors swallowed *after* the alias and before the prompt
/// remainder. "ask claude **to** do X" → prompt is "do X".
const CONNECTORS: &[&str] = &["to", "for", "about"];

/// One named-assistant binding the engine can dispatch to. Built from
/// an installed `metadata.ari.type: assistant` skill that declares
/// `metadata.ari.assistant.aliases`.
#[derive(Clone)]
pub struct NamedAssistantBinding {
    pub skill_id: String,
    pub aliases: Vec<String>,
    pub config: ApiConfig,
    pub config_store: Arc<dyn ConfigStore>,
}

/// Result of a successful match: which binding fired, and the prompt
/// to send to its API (everything after the alias + optional connector).
#[derive(Clone)]
pub struct NamedAssistantMatch<'a> {
    pub binding: &'a NamedAssistantBinding,
    pub remainder: String,
}

/// Try to match `normalized` against any of `bindings`. Returns
/// `Some(match)` on the first hit (iteration order = registry order),
/// `None` if no alias matched or the remainder would be empty.
pub fn match_named<'a>(
    normalized: &str,
    bindings: &'a [NamedAssistantBinding],
) -> Option<NamedAssistantMatch<'a>> {
    if bindings.is_empty() {
        return None;
    }
    let words: Vec<&str> = normalized.split_whitespace().collect();
    if words.is_empty() {
        return None;
    }

    for binding in bindings {
        for alias in &binding.aliases {
            if let Some(remainder) = try_match_alias(&words, alias) {
                return Some(NamedAssistantMatch {
                    binding,
                    remainder,
                });
            }
        }
    }
    None
}

/// Try one alias against the tokenised input. Returns the remainder
/// (joined with single spaces) when matched, or `None`. Allows:
///
/// 1. Optional leading fluff words (any number — "hey", "can", "you" …).
/// 2. Optional single verb ("ask", "tell", "use", "get").
/// 3. The alias itself (1+ tokens, must match consecutively).
/// 4. Optional connector ("to", "for", "about").
/// 5. Non-empty prompt remainder.
fn try_match_alias(words: &[&str], alias: &str) -> Option<String> {
    let alias_tokens: Vec<&str> = alias.split_whitespace().collect();
    if alias_tokens.is_empty() {
        return None;
    }

    let mut idx = 0;

    while idx < words.len() && FLUFF_WORDS.contains(&words[idx]) {
        idx += 1;
    }

    if idx < words.len() && VERBS.contains(&words[idx]) {
        idx += 1;
    }

    if idx + alias_tokens.len() > words.len() {
        return None;
    }
    if &words[idx..idx + alias_tokens.len()] != alias_tokens.as_slice() {
        return None;
    }
    idx += alias_tokens.len();

    if idx < words.len() && CONNECTORS.contains(&words[idx]) {
        idx += 1;
    }

    if idx >= words.len() {
        return None;
    }
    Some(words[idx..].join(" "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ari_skill_loader::manifest::{ApiConfig, AuthScheme, RequestFormat};

    fn dummy_config() -> ApiConfig {
        ApiConfig {
            endpoint: Some("https://example.com/v1".into()),
            endpoint_config_key: None,
            default_endpoint: None,
            auth: AuthScheme::None,
            auth_header: None,
            auth_config_key: None,
            model_config_key: None,
            default_model: "test-model".into(),
            system_prompt: "test".into(),
            request_format: RequestFormat::Openai,
            response_path: "choices[0].message.content".into(),
            api_version: None,
            api_version_header: None,
            max_tokens: 256,
            temperature: 0.5,
        }
    }

    struct NoopStore;
    impl ConfigStore for NoopStore {
        fn get(&self, _skill_id: &str, _key: &str) -> Option<String> {
            None
        }
    }

    fn make_binding(skill_id: &str, aliases: &[&str]) -> NamedAssistantBinding {
        NamedAssistantBinding {
            skill_id: skill_id.into(),
            aliases: aliases.iter().map(|s| (*s).to_string()).collect(),
            config: dummy_config(),
            config_store: Arc::new(NoopStore) as Arc<dyn ConfigStore>,
        }
    }

    #[test]
    fn matches_ask_verb_prefix() {
        let bindings = vec![make_binding("claude", &["claude", "anthropic"])];
        let m = match_named("ask claude what is rust", &bindings).unwrap();
        assert_eq!(m.binding.skill_id, "claude");
        assert_eq!(m.remainder, "what is rust");
    }

    #[test]
    fn matches_bare_alias_prefix() {
        let bindings = vec![make_binding("claude", &["claude"])];
        let m = match_named("claude what time is it", &bindings).unwrap();
        assert_eq!(m.remainder, "what time is it");
    }

    #[test]
    fn matches_fluff_then_verb() {
        let bindings = vec![make_binding("claude", &["claude"])];
        let m = match_named("hey can you ask claude for a joke", &bindings).unwrap();
        assert_eq!(m.binding.skill_id, "claude");
        assert_eq!(m.remainder, "a joke");
    }

    #[test]
    fn matches_multiword_alias() {
        let bindings = vec![make_binding("chatgpt", &["chatgpt", "chat gpt"])];
        let m = match_named("tell chat gpt to write a haiku", &bindings).unwrap();
        assert_eq!(m.binding.skill_id, "chatgpt");
        assert_eq!(m.remainder, "write a haiku");
    }

    #[test]
    fn matches_connector_to() {
        let bindings = vec![make_binding("claude", &["claude"])];
        let m = match_named("get claude to help me with this", &bindings).unwrap();
        assert_eq!(m.remainder, "help me with this");
    }

    #[test]
    fn matches_connector_about() {
        let bindings = vec![make_binding("claude", &["claude"])];
        let m = match_named("ask claude about quantum computing", &bindings).unwrap();
        assert_eq!(m.remainder, "quantum computing");
    }

    #[test]
    fn no_match_unrelated_input() {
        let bindings = vec![make_binding("claude", &["claude", "anthropic"])];
        assert!(match_named("what time is it", &bindings).is_none());
        assert!(match_named("set a timer for five minutes", &bindings).is_none());
    }

    #[test]
    fn no_match_empty_remainder() {
        let bindings = vec![make_binding("claude", &["claude"])];
        assert!(match_named("ask claude", &bindings).is_none());
        assert!(match_named("hey claude", &bindings).is_none());
        assert!(match_named("claude", &bindings).is_none());
    }

    #[test]
    fn no_match_empty_remainder_after_connector() {
        let bindings = vec![make_binding("claude", &["claude"])];
        assert!(match_named("ask claude to", &bindings).is_none());
        assert!(match_named("ask claude about", &bindings).is_none());
    }

    #[test]
    fn no_match_alias_mid_sentence() {
        let bindings = vec![make_binding("claude", &["claude"])];
        // Consumes only fluff/verb prefix — "i", "think" aren't fluff,
        // so the alias never gets a shot at words[0].
        assert!(match_named("i think claude is great", &bindings).is_none());
        assert!(match_named("the model claude is smart", &bindings).is_none());
    }

    #[test]
    fn no_match_when_no_bindings() {
        let bindings: Vec<NamedAssistantBinding> = Vec::new();
        assert!(match_named("ask claude anything", &bindings).is_none());
    }

    #[test]
    fn first_binding_wins_on_alias_collision() {
        // If two assistants both claim the alias "ai", the first in the
        // list wins — registry load order is the tiebreaker.
        let bindings = vec![
            make_binding("first", &["ai"]),
            make_binding("second", &["ai"]),
        ];
        let m = match_named("ask ai what time is it", &bindings).unwrap();
        assert_eq!(m.binding.skill_id, "first");
    }

    #[test]
    fn picks_first_alias_match_across_bindings() {
        let bindings = vec![
            make_binding("claude", &["claude", "anthropic"]),
            make_binding("chatgpt", &["chatgpt", "openai"]),
        ];
        let m = match_named("ask openai what is rust", &bindings).unwrap();
        assert_eq!(m.binding.skill_id, "chatgpt");
        assert_eq!(m.remainder, "what is rust");
    }
}
