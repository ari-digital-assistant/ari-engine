//! API adapter for assistant skills with `provider: api`.
//!
//! Builds an HTTP request from the manifest's [`ApiConfig`], resolves
//! runtime values (API key, model, endpoint) from the [`ConfigStore`],
//! and extracts the response text via the manifest's `response_path`.

use crate::manifest::{
    ApiConfig, AuthScheme, RequestFormat, extract_by_path, parse_response_path,
};
use crate::models::ModelCatalog;
use crate::tls;
use thiserror::Error;

/// Sampling fields dropped when a tier falls back to its manifest pin. The
/// catalog is what tells us whether a model accepts them, so without it we
/// don't know — and guessing wrong costs a 400 on every request, whereas
/// omitting them just means provider-default sampling. Only the tier path uses
/// this; skills still storing a model ID verbatim keep sending `temperature`.
const UNKNOWN_MODEL_OMIT_PARAMS: [&str; 3] = ["temperature", "top_p", "top_k"];

// ── ConfigStore trait ──────────────────────────────────────────────────

/// Abstraction over platform-specific config/secret storage. The engine
/// reads config values through this trait; each frontend provides its own
/// implementation (Android: DataStore + EncryptedSharedPreferences,
/// Linux: GSettings + libsecret, CLI: env vars).
pub trait ConfigStore: Send + Sync {
    /// Read a config value for a given assistant skill. Returns `None` if
    /// the key hasn't been set yet.
    fn get(&self, skill_id: &str, key: &str) -> Option<String>;
}

/// In-memory config store for tests and CLI.
pub struct MemoryConfigStore {
    entries: std::collections::HashMap<(String, String), String>,
}

impl MemoryConfigStore {
    pub fn new() -> Self {
        Self {
            entries: std::collections::HashMap::new(),
        }
    }

    pub fn set(&mut self, skill_id: &str, key: &str, value: &str) {
        self.entries
            .insert((skill_id.to_string(), key.to_string()), value.to_string());
    }
}

impl ConfigStore for MemoryConfigStore {
    fn get(&self, skill_id: &str, key: &str) -> Option<String> {
        self.entries
            .get(&(skill_id.to_string(), key.to_string()))
            .cloned()
    }
}

// ── Resolved config ───────────────────────────────────────────────────

/// Runtime-resolved values needed to make an API call.
#[derive(Debug)]
struct ResolvedConfig {
    endpoint: String,
    model: String,
    api_key: Option<String>,
    omit_params: Vec<String>,
}

impl ResolvedConfig {
    fn omits(&self, param: &str) -> bool {
        self.omit_params.iter().any(|p| p == param)
    }
}

fn resolve_config(
    config: &ApiConfig,
    skill_id: &str,
    store: &dyn ConfigStore,
    catalog: Option<&ModelCatalog>,
) -> Result<ResolvedConfig, AssistantApiError> {
    let endpoint = if let Some(ref fixed) = config.endpoint {
        fixed.clone()
    } else if let Some(ref key) = config.endpoint_config_key {
        store
            .get(skill_id, key)
            .or_else(|| config.default_endpoint.clone())
            .ok_or(AssistantApiError::MissingConfig {
                key: key.clone(),
            })?
    } else {
        return Err(AssistantApiError::MissingConfig {
            key: "endpoint".into(),
        });
    };

    // Tier skills resolve through the registry catalog; the rest still store a
    // concrete model ID in config.
    let (model, omit_params) = if let Some(ref tier_key) = config.tier_config_key {
        let tier = store
            .get(skill_id, tier_key)
            .or_else(|| config.default_tier.clone())
            .unwrap_or_default();
        let provider = config.model_provider.as_deref().unwrap_or_default();

        match catalog.and_then(|c| c.lookup(provider, &tier)) {
            Some(resolved) => (resolved.id.clone(), resolved.omit_params.clone()),
            None => {
                let pinned = config
                    .default_models
                    .get(&tier)
                    .cloned()
                    .unwrap_or_else(|| config.default_model.clone());
                (
                    pinned,
                    UNKNOWN_MODEL_OMIT_PARAMS.iter().map(|s| s.to_string()).collect(),
                )
            }
        }
    } else if let Some(ref key) = config.model_config_key {
        (
            store
                .get(skill_id, key)
                .unwrap_or_else(|| config.default_model.clone()),
            Vec::new(),
        )
    } else {
        (config.default_model.clone(), Vec::new())
    };

    let api_key = if let Some(ref key) = config.auth_config_key {
        let val = store.get(skill_id, key).ok_or(AssistantApiError::MissingConfig {
            key: key.clone(),
        })?;
        Some(val)
    } else {
        None
    };

    Ok(ResolvedConfig {
        endpoint,
        model,
        api_key,
        omit_params,
    })
}

// ── Errors ─────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum AssistantApiError {
    #[error("missing config value for key `{key}`")]
    MissingConfig { key: String },

    #[error("HTTP request failed: {0}")]
    Http(String),

    #[error("could not parse API response: {0}")]
    ResponseParse(String),

    #[error("API returned error status {status}: {body}")]
    ApiError { status: u16, body: String },

    #[error("request timed out")]
    Timeout,
}

// ── API adapter ───────────────────────────────────────────────────────

const REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Call an assistant API and return the response text.
pub fn call_assistant_api(
    config: &ApiConfig,
    skill_id: &str,
    store: &dyn ConfigStore,
    user_input: &str,
    locale: &str,
    history: &[(String, String)],
    facts: &[String],
    catalog: Option<&ModelCatalog>,
) -> Result<String, AssistantApiError> {
    let resolved = resolve_config(config, skill_id, store, catalog)?;

    let body = build_request_body(config, &resolved, user_input, locale, history, facts);

    let tls_config = tls::webpki_roots_config();
    let client = reqwest::blocking::Client::builder()
        .use_preconfigured_tls(tls_config)
        .timeout(REQUEST_TIMEOUT)
        .build()
        .map_err(|e| AssistantApiError::Http(e.to_string()))?;

    let mut req = client.post(&resolved.endpoint);

    req = match config.auth {
        AuthScheme::Bearer => {
            let key = resolved
                .api_key
                .as_ref()
                .ok_or(AssistantApiError::MissingConfig {
                    key: "api_key".into(),
                })?;
            req.bearer_auth(key)
        }
        AuthScheme::Header => {
            let key = resolved
                .api_key
                .as_ref()
                .ok_or(AssistantApiError::MissingConfig {
                    key: "api_key".into(),
                })?;
            let header_name = config
                .auth_header
                .as_deref()
                .unwrap_or("Authorization");
            req.header(header_name, key)
        }
        AuthScheme::None => req,
    };

    if let Some(ref version) = config.api_version {
        let header_name = config
            .api_version_header
            .as_deref()
            .unwrap_or("api-version");
        req = req.header(header_name, version);
    }

    req = req.header("content-type", "application/json");

    let response = req
        .body(body)
        .send()
        .map_err(|e| {
            if e.is_timeout() {
                AssistantApiError::Timeout
            } else {
                AssistantApiError::Http(e.to_string())
            }
        })?;

    let status = response.status().as_u16();
    let response_body = response
        .text()
        .map_err(|e| AssistantApiError::Http(e.to_string()))?;

    if status >= 400 {
        return Err(AssistantApiError::ApiError {
            status,
            body: response_body,
        });
    }

    let json: serde_json::Value = serde_json::from_str(&response_body)
        .map_err(|e| AssistantApiError::ResponseParse(e.to_string()))?;

    let segments = parse_response_path(&config.response_path)
        .map_err(|e| AssistantApiError::ResponseParse(e.to_string()))?;

    extract_by_path(&json, &segments).ok_or_else(|| {
        AssistantApiError::ResponseParse(format!(
            "response_path `{}` did not match the API response",
            config.response_path,
        ))
    })
}

/// Map an ISO 639-1 locale code to its English language name, for the
/// per-request "Please reply in X" hint. Returns `None` for English
/// (no hint needed) and for locales we don't yet ship — those fall
/// back to the canonical-English system prompt without a language
/// override, matching the existing behaviour. Keep the entries here
/// in lockstep with `SupportedLocales` on each frontend.
fn english_language_name(locale: &str) -> Option<&'static str> {
    match locale {
        "it" => Some("Italian"),
        "es" => Some("Spanish"),
        "fr" => Some("French"),
        "de" => Some("German"),
        _ => None,
    }
}

fn build_request_body(
    config: &ApiConfig,
    resolved: &ResolvedConfig,
    user_input: &str,
    locale: &str,
    history: &[(String, String)],
    facts: &[String],
) -> String {
    // Two-tier locale handling for cloud assistants:
    //   1. If the skill ships a `system_prompt` translation for this
    //      locale, use it verbatim.
    //   2. Otherwise — the common case for community skills authored
    //      in English — we fall back to the English prompt and append
    //      a one-line "Please reply in <Language>." hint. Cloud LLMs
    //      reliably honour this without needing the rest of the prompt
    //      translated.
    // English locale or unknown-to-us locales just use the prompt as-is.
    let base_prompt = config.system_prompt.for_locale(locale);
    let has_translation = config
        .system_prompt
        .supported_locales()
        .iter()
        .any(|l| l == locale);
    let system_prompt: String = if !has_translation {
        if let Some(language) = english_language_name(locale) {
            format!("{}\n\nPlease reply in {}.", base_prompt, language)
        } else {
            base_prompt.to_string()
        }
    } else {
        base_prompt.to_string()
    };
    let mut system_prompt = system_prompt;
    if !history.is_empty() {
        system_prompt = format!("{system_prompt}\n\n{}", ari_core::CONTINUATION_INSTRUCTION);
    }
    if let Some(block) = ari_core::remembered_facts_block(facts) {
        system_prompt = format!("{system_prompt}\n\n{block}");
    }
    let system_prompt = system_prompt.as_str();
    let body = match config.request_format {
        RequestFormat::Openai => {
            // `max_completion_tokens` replaced `max_tokens` in the
            // Chat Completions API; newer families (o1/o3/gpt-5) hard-
            // reject `max_tokens` with HTTP 400, and the older ones
            // (gpt-4o, gpt-4o-mini, gpt-3.5-turbo) silently accept the
            // new name. One field name works across the current range.
            let mut messages = vec![serde_json::json!({"role": "system", "content": system_prompt})];
            for (role, content) in history {
                messages.push(serde_json::json!({"role": role, "content": content}));
            }
            messages.push(serde_json::json!({"role": "user", "content": user_input}));
            let mut obj = serde_json::json!({
                "model": resolved.model,
                "max_completion_tokens": config.max_tokens,
                "messages": messages,
            });
            if !resolved.omits("temperature") {
                obj["temperature"] = serde_json::json!(config.temperature);
            }
            obj
        }
        RequestFormat::Anthropic => {
            let mut messages = Vec::new();
            for (role, content) in history {
                messages.push(serde_json::json!({"role": role, "content": content}));
            }
            messages.push(serde_json::json!({"role": "user", "content": user_input}));
            let mut obj = serde_json::json!({
                "model": resolved.model,
                "max_tokens": config.max_tokens,
                "system": system_prompt,
                "messages": messages,
            });
            if !resolved.omits("temperature") {
                obj["temperature"] = serde_json::json!(config.temperature);
            }
            obj
        }
    };
    serde_json::to_string(&body).expect("json serialisation cannot fail")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{PathSegment, parse_response_path, extract_by_path};

    #[test]
    fn parse_openai_response_path() {
        let segments = parse_response_path("choices[0].message.content").unwrap();
        assert_eq!(
            segments,
            vec![
                PathSegment::Field("choices".into()),
                PathSegment::Index(0),
                PathSegment::Field("message".into()),
                PathSegment::Field("content".into()),
            ]
        );
    }

    #[test]
    fn parse_anthropic_response_path() {
        let segments = parse_response_path("content[0].text").unwrap();
        assert_eq!(
            segments,
            vec![
                PathSegment::Field("content".into()),
                PathSegment::Index(0),
                PathSegment::Field("text".into()),
            ]
        );
    }

    #[test]
    fn extract_openai_response() {
        let json: serde_json::Value = serde_json::json!({
            "choices": [{
                "message": {
                    "content": "The capital of Malta is Valletta."
                }
            }]
        });
        let segments = parse_response_path("choices[0].message.content").unwrap();
        let result = extract_by_path(&json, &segments);
        assert_eq!(result.as_deref(), Some("The capital of Malta is Valletta."));
    }

    #[test]
    fn extract_anthropic_response() {
        let json: serde_json::Value = serde_json::json!({
            "content": [{
                "type": "text",
                "text": "Valletta is the capital."
            }]
        });
        let segments = parse_response_path("content[0].text").unwrap();
        let result = extract_by_path(&json, &segments);
        assert_eq!(result.as_deref(), Some("Valletta is the capital."));
    }

    #[test]
    fn extract_returns_none_for_missing_path() {
        let json: serde_json::Value = serde_json::json!({"foo": "bar"});
        let segments = parse_response_path("choices[0].message.content").unwrap();
        assert!(extract_by_path(&json, &segments).is_none());
    }

    #[test]
    fn build_openai_request_body() {
        let config = ApiConfig {
            endpoint: Some("https://api.example.com".into()),
            endpoint_config_key: None,
            default_endpoint: None,
            auth: AuthScheme::Bearer,
            auth_header: None,
            auth_config_key: Some("api_key".into()),
            model_config_key: None,
            model_provider: None,
            tier_config_key: None,
            default_tier: None,
            default_models: Default::default(),
            default_model: "gpt-4o-mini".into(),
            system_prompt: "You are Ari.".into(),
            request_format: RequestFormat::Openai,
            response_path: "choices[0].message.content".into(),
            api_version: None,
            api_version_header: None,
            max_tokens: 256,
            temperature: 0.7,
        };
        let resolved = ResolvedConfig {
            endpoint: "https://api.example.com".into(),
            model: "gpt-4o-mini".into(),
            api_key: Some("sk-test".into()),
            omit_params: Vec::new(),
        };
        let body = build_request_body(&config, &resolved, "What is 2+2?", "en", &[], &[]);
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed["model"], "gpt-4o-mini");
        assert_eq!(parsed["max_completion_tokens"], 256);
        // Newer OpenAI models reject `max_tokens`; field must NOT appear
        // in the outgoing body even though `ApiConfig.max_tokens` is
        // the source of the value.
        assert!(parsed.get("max_tokens").is_none());
        assert_eq!(parsed["messages"][0]["role"], "system");
        assert_eq!(parsed["messages"][0]["content"], "You are Ari.");
        assert_eq!(parsed["messages"][1]["role"], "user");
        assert_eq!(parsed["messages"][1]["content"], "What is 2+2?");
    }

    #[test]
    fn build_request_body_injects_facts_block() {
        let config = ApiConfig {
            endpoint: Some("https://api.example.com".into()),
            endpoint_config_key: None,
            default_endpoint: None,
            auth: AuthScheme::Bearer,
            auth_header: None,
            auth_config_key: Some("api_key".into()),
            model_config_key: None,
            model_provider: None,
            tier_config_key: None,
            default_tier: None,
            default_models: Default::default(),
            default_model: "gpt-4o-mini".into(),
            system_prompt: "You are Ari.".into(),
            request_format: RequestFormat::Openai,
            response_path: "choices[0].message.content".into(),
            api_version: None,
            api_version_header: None,
            max_tokens: 256,
            temperature: 0.7,
        };
        let resolved = ResolvedConfig {
            endpoint: "https://api.example.com".into(),
            model: "gpt-4o-mini".into(),
            api_key: Some("sk-test".into()),
            omit_params: Vec::new(),
        };
        let facts = vec!["i am vegetarian".to_string()];
        let body = build_request_body(&config, &resolved, "what should i cook", "en", &[], &facts);
        assert!(body.contains("Things you know about the user:\\n- i am vegetarian"));
    }

    #[test]
    fn build_request_body_no_facts_block_when_empty() {
        let config = ApiConfig {
            endpoint: Some("https://api.example.com".into()),
            endpoint_config_key: None,
            default_endpoint: None,
            auth: AuthScheme::Bearer,
            auth_header: None,
            auth_config_key: Some("api_key".into()),
            model_config_key: None,
            model_provider: None,
            tier_config_key: None,
            default_tier: None,
            default_models: Default::default(),
            default_model: "gpt-4o-mini".into(),
            system_prompt: "You are Ari.".into(),
            request_format: RequestFormat::Openai,
            response_path: "choices[0].message.content".into(),
            api_version: None,
            api_version_header: None,
            max_tokens: 256,
            temperature: 0.7,
        };
        let resolved = ResolvedConfig {
            endpoint: "https://api.example.com".into(),
            model: "gpt-4o-mini".into(),
            api_key: Some("sk-test".into()),
            omit_params: Vec::new(),
        };
        let body = build_request_body(&config, &resolved, "hi", "en", &[], &[]);
        assert!(!body.contains("Things you know about the user"));
    }

    #[test]
    fn build_request_body_appends_locale_hint_for_untranslated_locale() {
        // English-only system prompt + Italian user → engine should
        // append "Please reply in Italian." so the LLM doesn't default
        // to English. This is the path taken by every cloud assistant
        // skill in the registry today (none yet ship per-locale
        // translations of their system_prompt).
        let config = ApiConfig {
            endpoint: Some("https://api.example.com".into()),
            endpoint_config_key: None,
            default_endpoint: None,
            auth: AuthScheme::Bearer,
            auth_header: None,
            auth_config_key: Some("api_key".into()),
            model_config_key: None,
            model_provider: None,
            tier_config_key: None,
            default_tier: None,
            default_models: Default::default(),
            default_model: "gpt-4o-mini".into(),
            system_prompt: "You are Ari.".into(),
            request_format: RequestFormat::Openai,
            response_path: "choices[0].message.content".into(),
            api_version: None,
            api_version_header: None,
            max_tokens: 256,
            temperature: 0.7,
        };
        let resolved = ResolvedConfig {
            endpoint: "https://api.example.com".into(),
            model: "gpt-4o-mini".into(),
            api_key: Some("sk-test".into()),
            omit_params: Vec::new(),
        };
        let body = build_request_body(&config, &resolved, "che ora è?", "it", &[], &[]);
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(
            parsed["messages"][0]["content"],
            "You are Ari.\n\nPlease reply in Italian.",
        );
    }

    #[test]
    fn build_request_body_no_hint_when_locale_translation_present() {
        use std::collections::BTreeMap;
        // Skill ships its own Italian system_prompt → engine must use
        // it verbatim, without appending a redundant "Please reply in
        // Italian." instruction.
        let mut prompts = BTreeMap::new();
        prompts.insert("en".to_string(), "You are Ari.".to_string());
        prompts.insert("it".to_string(), "Sei Ari.".to_string());
        let config = ApiConfig {
            endpoint: Some("https://api.example.com".into()),
            endpoint_config_key: None,
            default_endpoint: None,
            auth: AuthScheme::Bearer,
            auth_header: None,
            auth_config_key: Some("api_key".into()),
            model_config_key: None,
            model_provider: None,
            tier_config_key: None,
            default_tier: None,
            default_models: Default::default(),
            default_model: "gpt-4o-mini".into(),
            system_prompt: crate::manifest::LocalizedPrompt::from_map(prompts).unwrap(),
            request_format: RequestFormat::Openai,
            response_path: "choices[0].message.content".into(),
            api_version: None,
            api_version_header: None,
            max_tokens: 256,
            temperature: 0.7,
        };
        let resolved = ResolvedConfig {
            endpoint: "https://api.example.com".into(),
            model: "gpt-4o-mini".into(),
            api_key: Some("sk-test".into()),
            omit_params: Vec::new(),
        };
        let body = build_request_body(&config, &resolved, "ciao", "it", &[], &[]);
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed["messages"][0]["content"], "Sei Ari.");
    }

    #[test]
    fn build_request_body_no_hint_for_english() {
        let config = ApiConfig {
            endpoint: Some("https://api.example.com".into()),
            endpoint_config_key: None,
            default_endpoint: None,
            auth: AuthScheme::Bearer,
            auth_header: None,
            auth_config_key: Some("api_key".into()),
            model_config_key: None,
            model_provider: None,
            tier_config_key: None,
            default_tier: None,
            default_models: Default::default(),
            default_model: "gpt-4o-mini".into(),
            system_prompt: "You are Ari.".into(),
            request_format: RequestFormat::Openai,
            response_path: "choices[0].message.content".into(),
            api_version: None,
            api_version_header: None,
            max_tokens: 256,
            temperature: 0.7,
        };
        let resolved = ResolvedConfig {
            endpoint: "https://api.example.com".into(),
            model: "gpt-4o-mini".into(),
            api_key: Some("sk-test".into()),
            omit_params: Vec::new(),
        };
        let body = build_request_body(&config, &resolved, "what time?", "en", &[], &[]);
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed["messages"][0]["content"], "You are Ari.");
    }

    #[test]
    fn build_anthropic_request_body() {
        let config = ApiConfig {
            endpoint: Some("https://api.anthropic.com/v1/messages".into()),
            endpoint_config_key: None,
            default_endpoint: None,
            auth: AuthScheme::Header,
            auth_header: Some("x-api-key".into()),
            auth_config_key: Some("api_key".into()),
            model_config_key: None,
            model_provider: None,
            tier_config_key: None,
            default_tier: None,
            default_models: Default::default(),
            default_model: "claude-sonnet-4-6".into(),
            system_prompt: "You are Ari.".into(),
            request_format: RequestFormat::Anthropic,
            response_path: "content[0].text".into(),
            api_version: Some("2023-06-01".into()),
            api_version_header: Some("anthropic-version".into()),
            max_tokens: 256,
            temperature: 0.7,
        };
        let resolved = ResolvedConfig {
            endpoint: "https://api.anthropic.com/v1/messages".into(),
            model: "claude-sonnet-4-6".into(),
            api_key: Some("sk-ant-test".into()),
            omit_params: Vec::new(),
        };
        let body = build_request_body(&config, &resolved, "Hello", "en", &[], &[]);
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed["model"], "claude-sonnet-4-6");
        assert_eq!(parsed["system"], "You are Ari.");
        assert_eq!(parsed["messages"][0]["role"], "user");
        assert!(parsed["messages"].as_array().unwrap().len() == 1);
    }

    #[test]
    fn build_request_body_inserts_history_before_current_turn_openai() {
        let config = ApiConfig {
            endpoint: Some("https://api.example.com".into()),
            endpoint_config_key: None,
            default_endpoint: None,
            auth: AuthScheme::Bearer,
            auth_header: None,
            auth_config_key: Some("api_key".into()),
            model_config_key: None,
            model_provider: None,
            tier_config_key: None,
            default_tier: None,
            default_models: Default::default(),
            default_model: "gpt-4o-mini".into(),
            system_prompt: "You are Ari.".into(),
            request_format: RequestFormat::Openai,
            response_path: "choices[0].message.content".into(),
            api_version: None,
            api_version_header: None,
            max_tokens: 256,
            temperature: 0.7,
        };
        let resolved = ResolvedConfig {
            endpoint: "https://api.example.com".into(),
            model: "gpt-4o-mini".into(),
            api_key: Some("sk-test".into()),
            omit_params: Vec::new(),
        };
        let history = vec![
            ("user".to_string(), "what is the capital of uae?".to_string()),
            ("assistant".to_string(), "Abu Dhabi.".to_string()),
        ];
        let body = build_request_body(&config, &resolved, "what is the population?", "en", &history, &[]);
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        let msgs = v["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 4); // system + 2 history + current user
        assert_eq!(msgs[0]["role"], "system");
        assert!(msgs[0]["content"].as_str().unwrap().contains("[continuation]"));
        assert_eq!(msgs[1]["role"], "user");
        assert_eq!(msgs[1]["content"], "what is the capital of uae?");
        assert_eq!(msgs[2]["role"], "assistant");
        assert_eq!(msgs[2]["content"], "Abu Dhabi.");
        assert_eq!(msgs[3]["role"], "user");
        assert_eq!(msgs[3]["content"], "what is the population?");
    }

    #[test]
    fn build_request_body_no_history_omits_instruction_openai() {
        let config = ApiConfig {
            endpoint: Some("https://api.example.com".into()),
            endpoint_config_key: None,
            default_endpoint: None,
            auth: AuthScheme::Bearer,
            auth_header: None,
            auth_config_key: Some("api_key".into()),
            model_config_key: None,
            model_provider: None,
            tier_config_key: None,
            default_tier: None,
            default_models: Default::default(),
            default_model: "gpt-4o-mini".into(),
            system_prompt: "You are Ari.".into(),
            request_format: RequestFormat::Openai,
            response_path: "choices[0].message.content".into(),
            api_version: None,
            api_version_header: None,
            max_tokens: 256,
            temperature: 0.7,
        };
        let resolved = ResolvedConfig {
            endpoint: "https://api.example.com".into(),
            model: "gpt-4o-mini".into(),
            api_key: Some("sk-test".into()),
            omit_params: Vec::new(),
        };
        let body = build_request_body(&config, &resolved, "hello", "en", &[], &[]);
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        let msgs = v["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 2); // system + user only
        assert!(!msgs[0]["content"].as_str().unwrap().contains("[continuation]"));
    }

    #[test]
    fn memory_config_store_basic() {
        let mut store = MemoryConfigStore::new();
        assert!(store.get("x", "y").is_none());
        store.set("x", "y", "val");
        assert_eq!(store.get("x", "y").as_deref(), Some("val"));
    }

    #[test]
    fn resolve_config_uses_defaults() {
        let mut store = MemoryConfigStore::new();
        store.set("test.id", "api_key", "sk-123");

        let config = ApiConfig {
            endpoint: Some("https://api.example.com".into()),
            endpoint_config_key: None,
            default_endpoint: None,
            auth: AuthScheme::Bearer,
            auth_header: None,
            auth_config_key: Some("api_key".into()),
            model_config_key: Some("model".into()),
            model_provider: None,
            tier_config_key: None,
            default_tier: None,
            default_models: Default::default(),
            default_model: "default-model".into(),
            system_prompt: "test".into(),
            request_format: RequestFormat::Openai,
            response_path: "choices[0].message.content".into(),
            api_version: None,
            api_version_header: None,
            max_tokens: 256,
            temperature: 0.7,
        };

        let resolved = resolve_config(&config, "test.id", &store, None).unwrap();
        assert_eq!(resolved.endpoint, "https://api.example.com");
        assert_eq!(resolved.model, "default-model");
        assert_eq!(resolved.api_key.as_deref(), Some("sk-123"));
    }

    #[test]
    fn resolve_config_overrides_model_from_store() {
        let mut store = MemoryConfigStore::new();
        store.set("test.id", "api_key", "sk-123");
        store.set("test.id", "model", "gpt-4o");

        let config = ApiConfig {
            endpoint: Some("https://api.example.com".into()),
            endpoint_config_key: None,
            default_endpoint: None,
            auth: AuthScheme::Bearer,
            auth_header: None,
            auth_config_key: Some("api_key".into()),
            model_config_key: Some("model".into()),
            model_provider: None,
            tier_config_key: None,
            default_tier: None,
            default_models: Default::default(),
            default_model: "gpt-4o-mini".into(),
            system_prompt: "test".into(),
            request_format: RequestFormat::Openai,
            response_path: "choices[0].message.content".into(),
            api_version: None,
            api_version_header: None,
            max_tokens: 256,
            temperature: 0.7,
        };

        let resolved = resolve_config(&config, "test.id", &store, None).unwrap();
        assert_eq!(resolved.model, "gpt-4o");
    }

    #[test]
    fn loads_chatgpt_skill_from_disk() {
        let skill_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../ari-skills/skills");
        let chatgpt_dir = skill_dir.join("chatgpt");
        if !chatgpt_dir.exists() {
            // Skill not present in this checkout — skip.
            return;
        }
        let report = crate::loader::load_single_skill_dir(&chatgpt_dir);
        assert_eq!(report.skills.len(), 0, "assistant should not be in skills");
        assert_eq!(report.failures.len(), 0, "no failures: {:?}", report.failures);
        assert_eq!(report.assistants.len(), 1);
        let entry = &report.assistants[0];
        assert_eq!(entry.id, "dev.heyari.assistant.chatgpt");
        assert_eq!(entry.name, "chatgpt");
        let api = entry.manifest.api.as_ref().expect("api config present");
        assert_eq!(api.endpoint.as_deref(), Some("https://api.openai.com/v1/chat/completions"));
        assert_eq!(api.default_model, "gpt-4o-mini");
        assert_eq!(api.response_path, "choices[0].message.content");
    }

    #[test]
    #[ignore] // requires OPENAI_API_KEY env var
    fn chatgpt_real_api_call() {
        let api_key = match std::env::var("OPENAI_API_KEY") {
            Ok(k) if !k.is_empty() => k,
            _ => {
                eprintln!("OPENAI_API_KEY not set, skipping");
                return;
            }
        };

        let skill_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../ari-skills/skills/chatgpt");
        let report = crate::loader::load_single_skill_dir(&skill_dir);
        let entry = &report.assistants[0];
        let api = entry.manifest.api.as_ref().unwrap();

        let mut store = MemoryConfigStore::new();
        store.set(&entry.id, "api_key", &api_key);

        let result = call_assistant_api(api, &entry.id, &store, "what is the capital of malta", "en", &[], &[], None);
        match result {
            Ok(text) => {
                eprintln!("ChatGPT response: {text}");
                assert!(text.to_lowercase().contains("valletta"), "expected Valletta in: {text}");
            }
            Err(e) => panic!("API call failed: {e}"),
        }
    }

    #[test]
    fn resolve_config_missing_api_key_errors() {
        let store = MemoryConfigStore::new();
        let config = ApiConfig {
            endpoint: Some("https://api.example.com".into()),
            endpoint_config_key: None,
            default_endpoint: None,
            auth: AuthScheme::Bearer,
            auth_header: None,
            auth_config_key: Some("api_key".into()),
            model_config_key: None,
            model_provider: None,
            tier_config_key: None,
            default_tier: None,
            default_models: Default::default(),
            default_model: "model".into(),
            system_prompt: "test".into(),
            request_format: RequestFormat::Openai,
            response_path: "choices[0].message.content".into(),
            api_version: None,
            api_version_header: None,
            max_tokens: 256,
            temperature: 0.7,
        };

        let err = resolve_config(&config, "test.id", &store, None).unwrap_err();
        assert!(matches!(err, AssistantApiError::MissingConfig { key } if key == "api_key"));
    }

    #[test]
    fn build_request_body_inserts_history_before_current_turn_anthropic() {
        let config = ApiConfig {
            endpoint: Some("https://api.anthropic.com/v1/messages".into()),
            endpoint_config_key: None,
            default_endpoint: None,
            auth: AuthScheme::Header,
            auth_header: Some("x-api-key".into()),
            auth_config_key: Some("api_key".into()),
            model_config_key: None,
            model_provider: None,
            tier_config_key: None,
            default_tier: None,
            default_models: Default::default(),
            default_model: "claude-sonnet-4-6".into(),
            system_prompt: "You are Ari.".into(),
            request_format: RequestFormat::Anthropic,
            response_path: "content[0].text".into(),
            api_version: Some("2023-06-01".into()),
            api_version_header: Some("anthropic-version".into()),
            max_tokens: 256,
            temperature: 0.7,
        };
        let resolved = ResolvedConfig {
            endpoint: "https://api.anthropic.com/v1/messages".into(),
            model: "claude-sonnet-4-6".into(),
            api_key: Some("sk-ant-test".into()),
            omit_params: Vec::new(),
        };
        let history = vec![
            ("user".to_string(), "what is the capital of uae?".to_string()),
            ("assistant".to_string(), "Abu Dhabi.".to_string()),
        ];
        let body = build_request_body(&config, &resolved, "what is the population?", "en", &history, &[]);
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        let msgs = v["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 3); // 2 history + current user
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"], "what is the capital of uae?");
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[1]["content"], "Abu Dhabi.");
        assert_eq!(msgs[2]["role"], "user");
        assert_eq!(msgs[2]["content"], "what is the population?");
        assert!(v["system"].as_str().unwrap().contains("[continuation]"));
    }

    const TIER_CATALOG: &str = r#"{
      "schema_version": 1,
      "providers": {
        "openai": {
          "fast": {"id": "gpt-5.6-luna", "omit_params": ["temperature", "top_p", "top_k"]},
          "balanced": {"id": "gpt-5.6-terra", "omit_params": []}
        },
        "anthropic": {
          "smartest": {"id": "claude-opus-5", "omit_params": ["temperature"]}
        }
      }
    }"#;

    fn tier_config(provider: &str, format: RequestFormat) -> ApiConfig {
        let mut default_models = std::collections::BTreeMap::new();
        default_models.insert("fast".to_string(), "pinned-fast".to_string());
        default_models.insert("balanced".to_string(), "pinned-balanced".to_string());
        default_models.insert("smartest".to_string(), "pinned-smartest".to_string());
        ApiConfig {
            endpoint: Some("https://api.example.com".into()),
            endpoint_config_key: None,
            default_endpoint: None,
            auth: AuthScheme::None,
            auth_header: None,
            auth_config_key: None,
            model_config_key: None,
            model_provider: Some(provider.into()),
            tier_config_key: Some("tier".into()),
            default_tier: Some("balanced".into()),
            default_models,
            default_model: "last-resort".into(),
            system_prompt: "You are Ari.".into(),
            request_format: format,
            response_path: "choices[0].message.content".into(),
            api_version: None,
            api_version_header: None,
            max_tokens: 256,
            temperature: 0.7,
        }
    }

    #[test]
    fn tier_resolves_the_users_choice_through_the_catalog() {
        let catalog = ModelCatalog::from_json_bytes(TIER_CATALOG.as_bytes()).unwrap();
        let config = tier_config("openai", RequestFormat::Openai);
        let mut store = MemoryConfigStore::new();
        store.set("test.id", "tier", "fast");

        let resolved = resolve_config(&config, "test.id", &store, Some(&catalog)).unwrap();
        assert_eq!(resolved.model, "gpt-5.6-luna");
        assert_eq!(
            resolved.omit_params,
            vec!["temperature".to_string(), "top_p".to_string(), "top_k".to_string()]
        );
    }

    #[test]
    fn tier_falls_back_to_the_selects_default_when_unset() {
        let catalog = ModelCatalog::from_json_bytes(TIER_CATALOG.as_bytes()).unwrap();
        let config = tier_config("openai", RequestFormat::Openai);
        let store = MemoryConfigStore::new();

        let resolved = resolve_config(&config, "test.id", &store, Some(&catalog)).unwrap();
        assert_eq!(resolved.model, "gpt-5.6-terra");
        assert!(resolved.omit_params.is_empty());
    }

    #[test]
    fn no_catalog_uses_the_pin_for_that_tier_and_drops_sampling_params() {
        let config = tier_config("openai", RequestFormat::Openai);
        let mut store = MemoryConfigStore::new();
        store.set("test.id", "tier", "smartest");

        let resolved = resolve_config(&config, "test.id", &store, None).unwrap();
        assert_eq!(resolved.model, "pinned-smartest");
        // Without the catalog we don't know whether the pin accepts sampling
        // params, and a wrong guess is a 400 on every request.
        assert_eq!(
            resolved.omit_params,
            vec!["temperature".to_string(), "top_p".to_string(), "top_k".to_string()]
        );
    }

    #[test]
    fn a_tier_missing_from_the_catalog_uses_its_pin() {
        let catalog = ModelCatalog::from_json_bytes(TIER_CATALOG.as_bytes()).unwrap();
        // The catalog has no openai/smartest entry.
        let config = tier_config("openai", RequestFormat::Openai);
        let mut store = MemoryConfigStore::new();
        store.set("test.id", "tier", "smartest");

        let resolved = resolve_config(&config, "test.id", &store, Some(&catalog)).unwrap();
        assert_eq!(resolved.model, "pinned-smartest");
    }

    #[test]
    fn an_unrecognised_tier_value_falls_all_the_way_to_default_model() {
        let catalog = ModelCatalog::from_json_bytes(TIER_CATALOG.as_bytes()).unwrap();
        let config = tier_config("openai", RequestFormat::Openai);
        let mut store = MemoryConfigStore::new();
        store.set("test.id", "tier", "turbo");

        let resolved = resolve_config(&config, "test.id", &store, Some(&catalog)).unwrap();
        assert_eq!(resolved.model, "last-resort");
    }

    #[test]
    fn omit_params_removes_temperature_from_an_openai_body() {
        let catalog = ModelCatalog::from_json_bytes(TIER_CATALOG.as_bytes()).unwrap();
        let config = tier_config("openai", RequestFormat::Openai);
        let mut store = MemoryConfigStore::new();
        store.set("test.id", "tier", "fast");
        let resolved = resolve_config(&config, "test.id", &store, Some(&catalog)).unwrap();

        let body = build_request_body(&config, &resolved, "hello", "en", &[], &[]);
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed["model"], "gpt-5.6-luna");
        assert_eq!(parsed["max_completion_tokens"], 256);
        assert!(parsed.get("temperature").is_none());
    }

    #[test]
    fn a_tier_that_accepts_temperature_still_gets_it() {
        let catalog = ModelCatalog::from_json_bytes(TIER_CATALOG.as_bytes()).unwrap();
        let config = tier_config("openai", RequestFormat::Openai);
        let mut store = MemoryConfigStore::new();
        store.set("test.id", "tier", "balanced");
        let resolved = resolve_config(&config, "test.id", &store, Some(&catalog)).unwrap();

        let body = build_request_body(&config, &resolved, "hello", "en", &[], &[]);
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed["model"], "gpt-5.6-terra");
        assert_eq!(parsed["temperature"], serde_json::json!(0.7f32));
    }

    #[test]
    fn omit_params_removes_temperature_from_an_anthropic_body() {
        let catalog = ModelCatalog::from_json_bytes(TIER_CATALOG.as_bytes()).unwrap();
        let config = tier_config("anthropic", RequestFormat::Anthropic);
        let mut store = MemoryConfigStore::new();
        store.set("test.id", "tier", "smartest");
        let resolved = resolve_config(&config, "test.id", &store, Some(&catalog)).unwrap();

        let body = build_request_body(&config, &resolved, "hello", "en", &[], &[]);
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed["model"], "claude-opus-5");
        assert_eq!(parsed["max_tokens"], 256);
        assert!(parsed.get("temperature").is_none());
        assert_eq!(parsed["system"], "You are Ari.");
    }

    #[test]
    fn a_stored_model_id_skill_is_unaffected_by_the_tier_path() {
        let mut config = tier_config("openai", RequestFormat::Openai);
        config.tier_config_key = None;
        config.model_provider = None;
        config.default_tier = None;
        config.model_config_key = Some("model".into());
        let mut store = MemoryConfigStore::new();
        store.set("test.id", "model", "gpt-4o-mini");

        let resolved = resolve_config(&config, "test.id", &store, None).unwrap();
        assert_eq!(resolved.model, "gpt-4o-mini");
        assert!(resolved.omit_params.is_empty());

        let body = build_request_body(&config, &resolved, "hello", "en", &[], &[]);
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed["temperature"], serde_json::json!(0.7f32));
    }
}
