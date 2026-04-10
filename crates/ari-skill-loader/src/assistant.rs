//! API adapter for assistant skills with `provider: api`.
//!
//! Builds an HTTP request from the manifest's [`ApiConfig`], resolves
//! runtime values (API key, model, endpoint) from the [`ConfigStore`],
//! and extracts the response text via the manifest's `response_path`.

use crate::manifest::{
    ApiConfig, AuthScheme, RequestFormat, extract_by_path, parse_response_path,
};
use crate::tls;
use thiserror::Error;

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
}

fn resolve_config(
    config: &ApiConfig,
    skill_id: &str,
    store: &dyn ConfigStore,
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

    let model = if let Some(ref key) = config.model_config_key {
        store
            .get(skill_id, key)
            .unwrap_or_else(|| config.default_model.clone())
    } else {
        config.default_model.clone()
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
) -> Result<String, AssistantApiError> {
    let resolved = resolve_config(config, skill_id, store)?;

    let body = build_request_body(config, &resolved, user_input);

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

fn build_request_body(
    config: &ApiConfig,
    resolved: &ResolvedConfig,
    user_input: &str,
) -> String {
    let body = match config.request_format {
        RequestFormat::Openai => {
            serde_json::json!({
                "model": resolved.model,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "messages": [
                    {"role": "system", "content": config.system_prompt},
                    {"role": "user", "content": user_input}
                ]
            })
        }
        RequestFormat::Anthropic => {
            let mut obj = serde_json::json!({
                "model": resolved.model,
                "max_tokens": config.max_tokens,
                "system": config.system_prompt,
                "messages": [
                    {"role": "user", "content": user_input}
                ]
            });
            if let Some(ref temp) = Some(config.temperature) {
                obj["temperature"] = serde_json::json!(temp);
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
        };
        let body = build_request_body(&config, &resolved, "What is 2+2?");
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed["model"], "gpt-4o-mini");
        assert_eq!(parsed["max_tokens"], 256);
        assert_eq!(parsed["messages"][0]["role"], "system");
        assert_eq!(parsed["messages"][0]["content"], "You are Ari.");
        assert_eq!(parsed["messages"][1]["role"], "user");
        assert_eq!(parsed["messages"][1]["content"], "What is 2+2?");
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
        };
        let body = build_request_body(&config, &resolved, "Hello");
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed["model"], "claude-sonnet-4-6");
        assert_eq!(parsed["system"], "You are Ari.");
        assert_eq!(parsed["messages"][0]["role"], "user");
        assert!(parsed["messages"].as_array().unwrap().len() == 1);
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
            default_model: "default-model".into(),
            system_prompt: "test".into(),
            request_format: RequestFormat::Openai,
            response_path: "choices[0].message.content".into(),
            api_version: None,
            api_version_header: None,
            max_tokens: 256,
            temperature: 0.7,
        };

        let resolved = resolve_config(&config, "test.id", &store).unwrap();
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
            default_model: "gpt-4o-mini".into(),
            system_prompt: "test".into(),
            request_format: RequestFormat::Openai,
            response_path: "choices[0].message.content".into(),
            api_version: None,
            api_version_header: None,
            max_tokens: 256,
            temperature: 0.7,
        };

        let resolved = resolve_config(&config, "test.id", &store).unwrap();
        assert_eq!(resolved.model, "gpt-4o");
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
            default_model: "model".into(),
            system_prompt: "test".into(),
            request_format: RequestFormat::Openai,
            response_path: "choices[0].message.content".into(),
            api_version: None,
            api_version_header: None,
            max_tokens: 256,
            temperature: 0.7,
        };

        let err = resolve_config(&config, "test.id", &store).unwrap_err();
        assert!(matches!(err, AssistantApiError::MissingConfig { key } if key == "api_key"));
    }
}
