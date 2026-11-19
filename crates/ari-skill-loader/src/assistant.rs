//! API adapter for assistant skills with `provider: api`.
//!
//! Builds an HTTP request from the manifest's [`ApiConfig`], resolves
//! runtime values (API key, model, endpoint) from the [`ConfigStore`],
//! and extracts the response text via the manifest's `response_path`.

use crate::manifest::{
    ApiConfig, AuthScheme, RequestFormat, WebSearchConfig, extract_by_path, parse_response_path,
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
        endpoint: endpoint.replace("{model}", &model),
        model,
        api_key,
        omit_params,
    })
}

/// `config` as its `web_search` block changes it.
fn searching(config: &ApiConfig, search: &WebSearchConfig) -> ApiConfig {
    let mut searching = config.clone();
    if let Some(ref endpoint) = search.endpoint {
        searching.endpoint = Some(endpoint.clone());
        searching.endpoint_config_key = None;
        searching.default_endpoint = None;
    }
    if let Some(format) = search.request_format {
        searching.request_format = format;
    }
    if let Some(auth) = search.auth {
        searching.auth = auth;
    }
    if let Some(ref header) = search.auth_header {
        searching.auth_header = Some(header.clone());
    }
    if let Some(ref path) = search.response_path {
        searching.response_path = path.clone();
    }
    if let Some(ref prompt) = search.system_prompt {
        searching.system_prompt = prompt.clone();
    }
    searching
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

/// Whether an assistant call may search the web. Only a skill whose
/// `web_search` block says how can; for any other, every call goes out as
/// `Off`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebSearch {
    /// Routing and structured replies: the skill's own call, without the tool.
    Off,
    /// An answer: the assistant searches when the question needs it.
    Allowed,
    /// A question the on-device model handed over because it needs current
    /// information: the assistant is told to search before it answers, and
    /// made to where the provider allows it.
    Required,
}

/// Added to the instructions of a call that must search. Left to decide,
/// ChatGPT answered a handed-over "who won the most recent Formula 1 race?"
/// from memory, with a race from weeks before.
pub const SEARCH_FIRST_INSTRUCTION: &str = "This question needs current information: \
search the web before you answer, and answer from what you find.";

/// Call an assistant API and return the response text. A skill that declares
/// a `web_search` block may search the web first, as `web_search` allows;
/// leave it `Off` for calls that want structured output rather than an
/// answer.
pub fn call_assistant_api(
    config: &ApiConfig,
    skill_id: &str,
    store: &dyn ConfigStore,
    user_input: &str,
    locale: &str,
    history: &[(String, String)],
    facts: &[String],
    personality: &str,
    catalog: Option<&ModelCatalog>,
    web_search: WebSearch,
) -> Result<String, AssistantApiError> {
    let search = config.web_search.as_ref().filter(|_| web_search != WebSearch::Off);
    let searching_config;
    let config = match search {
        Some(search) => {
            searching_config = searching(config, search);
            &searching_config
        }
        None => config,
    };
    let resolved = resolve_config(config, skill_id, store, catalog)?;

    let body = build_request_body(
        config,
        &resolved,
        user_input,
        locale,
        history,
        facts,
        personality,
        search.map(|s| (&s.tool, web_search)),
    );

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

    let mut json: serde_json::Value = serde_json::from_str(&response_body)
        .map_err(|e| AssistantApiError::ResponseParse(e.to_string()))?;

    gather_answer(config.request_format, &mut json);

    let segments = parse_response_path(&config.response_path)
        .map_err(|e| AssistantApiError::ResponseParse(e.to_string()))?;

    extract_by_path(&json, &segments).ok_or_else(|| {
        AssistantApiError::ResponseParse(format!(
            "response_path `{}` did not match the API response",
            config.response_path,
        ))
    })
}

/// Put the whole answer where a simple `response_path` finds it.
///
/// Reasoning and searching scatter it: thinking blocks or reasoning items
/// come first, a search adds tool blocks, and the answer after a search
/// arrives in pieces, one per citation. Leaves a response with nothing to
/// gather alone, so the path fails on it and the error names the path the
/// author wrote.
fn gather_answer(format: RequestFormat, json: &mut serde_json::Value) {
    match format {
        RequestFormat::Openai => {}
        RequestFormat::Anthropic => gather_anthropic_answer(json),
        RequestFormat::OpenaiResponses => gather_openai_responses_answer(json),
        RequestFormat::Gemini => gather_gemini_answer(json),
    }
}

/// `content` becomes one text block: the text after the last block that
/// isn't text. What comes before a search is Claude saying it will search.
/// A turn that stopped mid-search (`pause_turn`) ends on a tool block, has
/// nothing after it, and fails as having no answer: a voice reply can't wait
/// for another round trip, and a search limit of a few keeps it rare.
fn gather_anthropic_answer(json: &mut serde_json::Value) {
    let Some(blocks) = json.get_mut("content").and_then(|c| c.as_array_mut()) else {
        return;
    };
    let is_text = |b: &serde_json::Value| b.get("type").and_then(|t| t.as_str()) == Some("text");
    let answer_start = blocks.iter().rposition(|b| !is_text(b)).map_or(0, |i| i + 1);
    let answer: String = blocks[answer_start..]
        .iter()
        .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
        .collect();
    *blocks = if answer.is_empty() {
        Vec::new()
    } else {
        vec![serde_json::json!({"type": "text", "text": answer})]
    };
}

/// Sets `output_text`, as OpenAI's own libraries do: the text of every
/// message in `output`, skipping reasoning and search items and any message
/// marked as commentary rather than the answer.
fn gather_openai_responses_answer(json: &mut serde_json::Value) {
    let Some(items) = json.get("output").and_then(|o| o.as_array()) else {
        return;
    };
    let answer: String = items
        .iter()
        .filter(|item| item.get("type").and_then(|t| t.as_str()) == Some("message"))
        .filter(|item| item.get("phase").and_then(|p| p.as_str()) != Some("commentary"))
        .filter_map(|item| item.get("content").and_then(|c| c.as_array()))
        .flatten()
        .filter(|part| part.get("type").and_then(|t| t.as_str()) == Some("output_text"))
        .filter_map(|part| part.get("text").and_then(|t| t.as_str()))
        .collect();
    if !answer.is_empty() {
        json["output_text"] = serde_json::Value::String(answer);
    }
}

/// The first candidate's `parts` become one part holding all its text,
/// leaving out thought summaries.
fn gather_gemini_answer(json: &mut serde_json::Value) {
    let Some(parts) = json
        .pointer_mut("/candidates/0/content/parts")
        .and_then(|p| p.as_array_mut())
    else {
        return;
    };
    let answer: String = parts
        .iter()
        .filter(|p| p.get("thought").and_then(|t| t.as_bool()) != Some(true))
        .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
        .collect();
    *parts = if answer.is_empty() {
        Vec::new()
    } else {
        vec![serde_json::json!({"text": answer})]
    };
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
    personality: &str,
    search: Option<(&serde_json::Value, WebSearch)>,
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
    let must_search = matches!(search, Some((_, WebSearch::Required)));
    if must_search {
        system_prompt = format!("{system_prompt}\n\n{SEARCH_FIRST_INSTRUCTION}");
    }
    if !history.is_empty() {
        system_prompt = format!("{system_prompt}\n\n{}", ari_core::HISTORY_INSTRUCTION);
    }
    if let Some(block) = ari_core::personality_block(personality) {
        system_prompt = format!("{system_prompt}\n\n{block}");
    }
    if let Some(block) = ari_core::remembered_facts_block(facts) {
        system_prompt = format!("{system_prompt}\n\n{block}");
    }
    let system_prompt = system_prompt.as_str();
    let mut body = match config.request_format {
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
        RequestFormat::OpenaiResponses => {
            let mut input = Vec::new();
            for (role, content) in history {
                input.push(serde_json::json!({"role": role, "content": content}));
            }
            input.push(serde_json::json!({"role": "user", "content": user_input}));
            // Responses are kept on OpenAI's side for later retrieval unless
            // told otherwise; Chat Completions never kept them. No
            // `temperature`: GPT-5.6 and GPT-6 reject it here (HTTP 400,
            // November 2026) though the catalog, which describes Chat
            // Completions, says they take it.
            serde_json::json!({
                "model": resolved.model,
                "instructions": system_prompt,
                "input": input,
                "max_output_tokens": config.max_tokens,
                "store": false,
            })
        }
        RequestFormat::Gemini => {
            let mut contents = Vec::new();
            for (role, content) in history {
                let role = if role == "assistant" { "model" } else { role.as_str() };
                contents.push(serde_json::json!({"role": role, "parts": [{"text": content}]}));
            }
            contents.push(serde_json::json!({"role": "user", "parts": [{"text": user_input}]}));
            let mut generation_config = serde_json::json!({"maxOutputTokens": config.max_tokens});
            if !resolved.omits("temperature") {
                generation_config["temperature"] = serde_json::json!(config.temperature);
            }
            serde_json::json!({
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": contents,
                "generationConfig": generation_config,
            })
        }
    };
    if let Some((tool, _)) = search {
        body["tools"] = serde_json::json!([tool]);
    }
    // Only OpenAI's Responses API can be made to call it. Anthropic's current
    // models reject a forced `tool_choice` (HTTP 400) and Gemini has no way to
    // force its search, so for those the instruction above is all there is.
    if must_search && config.request_format == RequestFormat::OpenaiResponses {
        body["tool_choice"] = serde_json::json!("required");
    }
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
    fn extract_anthropic_response_past_a_thinking_block() {
        // Reasoning models put `thinking` at content[0], which has no `text`
        // field — without the filter the manifest path resolves to nothing and
        // the user gets the engine's "didn't understand that" instead.
        let mut json: serde_json::Value = serde_json::json!({
            "content": [
                {"type": "thinking", "thinking": ""},
                {"type": "text", "text": "The Dead Sea."}
            ]
        });
        gather_answer(RequestFormat::Anthropic, &mut json);
        let segments = parse_response_path("content[0].text").unwrap();
        let result = extract_by_path(&json, &segments);
        assert_eq!(result.as_deref(), Some("The Dead Sea."));
    }

    #[test]
    fn gather_answer_leaves_a_chat_completions_body_alone() {
        let body = serde_json::json!({
            "choices": [{"message": {"content": "Valletta."}}]
        });
        let mut json = body.clone();
        gather_answer(RequestFormat::Openai, &mut json);
        assert_eq!(json, body);
    }

    #[test]
    fn a_response_with_no_text_block_still_fails_the_path() {
        let mut json: serde_json::Value = serde_json::json!({
            "content": [{"type": "thinking", "thinking": ""}]
        });
        gather_answer(RequestFormat::Anthropic, &mut json);
        assert_eq!(json["content"].as_array().unwrap().len(), 0);
        let segments = parse_response_path("content[0].text").unwrap();
        assert!(extract_by_path(&json, &segments).is_none());
    }

    fn gathered(format: RequestFormat, mut json: serde_json::Value, path: &str) -> Option<String> {
        gather_answer(format, &mut json);
        extract_by_path(&json, &parse_response_path(path).unwrap())
    }

    #[test]
    fn a_searched_claude_answer_comes_back_whole_without_its_preamble() {
        // The shape Anthropic's web search docs show: a sentence before the
        // search, the search and its results, then the answer in pieces.
        let json = serde_json::json!({
            "content": [
                {"type": "text", "text": "I'll search for when Claude Shannon was born."},
                {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search",
                 "input": {"query": "claude shannon birth date"}},
                {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1",
                 "content": [{"type": "web_search_result", "url": "https://en.wikipedia.org/wiki/Claude_Shannon",
                              "title": "Claude Shannon", "encrypted_content": "x"}]},
                {"type": "text", "text": "Based on the search results, "},
                {"type": "text", "text": "Claude Shannon was born on April 30, 1916, in Petoskey, Michigan",
                 "citations": [{"type": "web_search_result_location", "cited_text": "Claude Elwood Shannon"}]},
                {"type": "text", "text": "."}
            ],
            "stop_reason": "end_turn"
        });
        assert_eq!(
            gathered(RequestFormat::Anthropic, json, "content[0].text").as_deref(),
            Some("Based on the search results, Claude Shannon was born on April 30, 1916, in Petoskey, Michigan.")
        );
    }

    #[test]
    fn a_claude_turn_paused_mid_search_has_no_answer() {
        let json = serde_json::json!({
            "content": [
                {"type": "text", "text": "Let me look that up."},
                {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "q"}}
            ],
            "stop_reason": "pause_turn"
        });
        assert_eq!(gathered(RequestFormat::Anthropic, json, "content[0].text"), None);
    }

    #[test]
    fn a_responses_answer_skips_reasoning_searches_and_commentary() {
        let json = serde_json::json!({
            "status": "completed",
            "output": [
                {"type": "reasoning", "id": "rs_1", "summary": []},
                {"type": "message", "role": "assistant", "phase": "commentary",
                 "content": [{"type": "output_text", "text": "Searching the BBC.", "annotations": []}]},
                {"type": "web_search_call", "id": "ws_1", "status": "completed",
                 "action": {"type": "search", "query": "bbc news top story"}},
                {"type": "message", "role": "assistant", "phase": "final_answer",
                 "content": [
                     {"type": "output_text", "text": "The top story is about ", "annotations": []},
                     {"type": "output_text", "text": "the budget.",
                      "annotations": [{"type": "url_citation", "url": "https://www.bbc.co.uk/news"}]}
                 ]}
            ]
        });
        assert_eq!(
            gathered(RequestFormat::OpenaiResponses, json, "output_text").as_deref(),
            Some("The top story is about the budget.")
        );
    }

    #[test]
    fn a_responses_answer_without_a_phase_is_kept() {
        let json = serde_json::json!({
            "output": [
                {"type": "message", "role": "assistant",
                 "content": [{"type": "output_text", "text": "Valletta.", "annotations": []}]}
            ]
        });
        assert_eq!(gathered(RequestFormat::OpenaiResponses, json, "output_text").as_deref(), Some("Valletta."));
    }

    #[test]
    fn a_responses_reply_with_no_message_has_no_answer() {
        // What running out of output tokens while reasoning looks like.
        let json = serde_json::json!({
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [{"type": "reasoning", "id": "rs_1", "summary": []}]
        });
        assert_eq!(gathered(RequestFormat::OpenaiResponses, json, "output_text"), None);
    }

    #[test]
    fn a_gemini_answer_joins_its_parts_without_thoughts() {
        let json = serde_json::json!({
            "candidates": [{
                "content": {"role": "model", "parts": [
                    {"text": "The user wants the Euro 2024 winner.", "thought": true},
                    {"text": "Spain won Euro 2024, "},
                    {"text": "beating England 2-1 in the final."}
                ]},
                "groundingMetadata": {"webSearchQueries": ["who won euro 2024"]}
            }]
        });
        assert_eq!(
            gathered(RequestFormat::Gemini, json, "candidates[0].content.parts[0].text").as_deref(),
            Some("Spain won Euro 2024, beating England 2-1 in the final.")
        );
    }

    #[test]
    fn a_gemini_reply_with_only_thoughts_has_no_answer() {
        let json = serde_json::json!({
            "candidates": [{"content": {"parts": [{"text": "thinking", "thought": true}]},
                            "finishReason": "MAX_TOKENS"}]
        });
        assert_eq!(gathered(RequestFormat::Gemini, json, "candidates[0].content.parts[0].text"), None);
    }

    fn plain_config(format: RequestFormat) -> ApiConfig {
        ApiConfig {
            endpoint: Some("https://api.example.com/v1/chat/completions".into()),
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
            default_model: "m-1".into(),
            system_prompt: "No live data.".into(),
            request_format: format,
            response_path: "choices[0].message.content".into(),
            api_version: None,
            api_version_header: None,
            max_tokens: 2048,
            temperature: 0.7,
            web_search: None,
        }
    }

    fn resolved_for(model: &str, omit_params: Vec<String>) -> ResolvedConfig {
        ResolvedConfig {
            endpoint: "https://api.example.com".into(),
            model: model.into(),
            api_key: Some("k".into()),
            omit_params,
        }
    }

    #[test]
    fn a_responses_request_carries_instructions_history_and_the_search_tool() {
        let config = plain_config(RequestFormat::OpenaiResponses);
        let history = vec![
            ("user".to_string(), "who won the league?".to_string()),
            ("assistant".to_string(), "Liverpool.".to_string()),
        ];
        let tool = serde_json::json!({"type": "web_search"});
        let body = build_request_body(
            &config,
            &resolved_for("gpt-x", Vec::new()),
            "and the cup?",
            "en",
            &history,
            &[],
            "",
            Some((&tool, WebSearch::Allowed)),
        );
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(
            parsed,
            serde_json::json!({
                "model": "gpt-x",
                "instructions": format!("No live data.\n\n{}", ari_core::HISTORY_INSTRUCTION),
                "input": [
                    {"role": "user", "content": "who won the league?"},
                    {"role": "assistant", "content": "Liverpool."},
                    {"role": "user", "content": "and the cup?"}
                ],
                "max_output_tokens": 2048,
                "store": false,
                "tools": [{"type": "web_search"}]
            })
        );
    }

    #[test]
    fn a_gemini_request_names_the_assistant_model_and_drops_omitted_temperature() {
        let config = plain_config(RequestFormat::Gemini);
        let history = vec![
            ("user".to_string(), "who won euro 2024?".to_string()),
            ("assistant".to_string(), "Spain.".to_string()),
        ];
        let tool = serde_json::json!({"google_search": {}});
        let body = build_request_body(
            &config,
            &resolved_for("gemini-x", vec!["temperature".into()]),
            "who scored?",
            "en",
            &history,
            &[],
            "",
            Some((&tool, WebSearch::Allowed)),
        );
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(
            parsed,
            serde_json::json!({
                "systemInstruction": {"parts": [{"text": format!("No live data.\n\n{}", ari_core::HISTORY_INSTRUCTION)}]},
                "contents": [
                    {"role": "user", "parts": [{"text": "who won euro 2024?"}]},
                    {"role": "model", "parts": [{"text": "Spain."}]},
                    {"role": "user", "parts": [{"text": "who scored?"}]}
                ],
                "generationConfig": {"maxOutputTokens": 2048},
                "tools": [{"google_search": {}}]
            })
        );
    }

    #[test]
    fn an_anthropic_request_with_search_lists_the_tool() {
        let config = plain_config(RequestFormat::Anthropic);
        let tool = serde_json::json!({"type": "web_search_20250305", "name": "web_search", "max_uses": 3});
        let body = build_request_body(&config, &resolved_for("claude-x", Vec::new()), "hi", "en", &[], &[], "", Some((&tool, WebSearch::Allowed)));
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed["tools"], serde_json::json!([tool]));
        assert_eq!(parsed["system"], "No live data.");
    }

    #[test]
    fn a_required_search_is_forced_on_the_responses_api() {
        let config = plain_config(RequestFormat::OpenaiResponses);
        let tool = serde_json::json!({"type": "web_search"});
        let body = build_request_body(
            &config,
            &resolved_for("gpt-x", Vec::new()),
            "who came second?",
            "en",
            &[],
            &[],
            "",
            Some((&tool, WebSearch::Required)),
        );
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed["instructions"], format!("No live data.\n\n{SEARCH_FIRST_INSTRUCTION}"));
        assert_eq!(parsed["tools"], serde_json::json!([tool]));
        assert_eq!(parsed["tool_choice"], "required");
    }

    #[test]
    fn a_required_search_is_only_asked_for_where_it_cannot_be_forced() {
        for (format, tool, instructions) in [
            (RequestFormat::Anthropic, serde_json::json!({"type": "web_search_20250305", "name": "web_search"}), "/system"),
            (RequestFormat::Gemini, serde_json::json!({"google_search": {}}), "/systemInstruction/parts/0/text"),
        ] {
            let config = plain_config(format);
            let body = build_request_body(
                &config,
                &resolved_for("m", Vec::new()),
                "who came second?",
                "en",
                &[],
                &[],
                "",
                Some((&tool, WebSearch::Required)),
            );
            let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
            assert_eq!(
                parsed.pointer(instructions),
                Some(&serde_json::json!(format!("No live data.\n\n{SEARCH_FIRST_INSTRUCTION}"))),
                "{format:?}"
            );
            assert_eq!(parsed["tools"], serde_json::json!([tool]), "{format:?}");
            assert!(parsed.get("tool_choice").is_none(), "{format:?}");
        }
    }

    #[test]
    fn a_request_without_search_lists_no_tools() {
        let config = plain_config(RequestFormat::Anthropic);
        let body = build_request_body(&config, &resolved_for("claude-x", Vec::new()), "hi", "en", &[], &[], "", None);
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert!(parsed.get("tools").is_none());
    }

    #[test]
    fn searching_swaps_in_every_field_the_block_sets() {
        let mut config = plain_config(RequestFormat::Openai);
        config.endpoint_config_key = Some("endpoint".into());
        config.default_endpoint = Some("https://default.example.com".into());
        let search = WebSearchConfig {
            endpoint: Some("https://g.example.com/v1beta/models/{model}:generateContent".into()),
            request_format: Some(RequestFormat::Gemini),
            auth: Some(AuthScheme::Header),
            auth_header: Some("x-goog-api-key".into()),
            response_path: Some("candidates[0].content.parts[0].text".into()),
            system_prompt: Some("Search when you need to.".into()),
            tool: serde_json::json!({"google_search": {}}),
        };
        let searched = searching(&config, &search);
        assert_eq!(searched.endpoint.as_deref(), Some("https://g.example.com/v1beta/models/{model}:generateContent"));
        assert_eq!(searched.endpoint_config_key, None);
        assert_eq!(searched.default_endpoint, None);
        assert_eq!(searched.request_format, RequestFormat::Gemini);
        assert_eq!(searched.auth, AuthScheme::Header);
        assert_eq!(searched.auth_header.as_deref(), Some("x-goog-api-key"));
        assert_eq!(searched.response_path, "candidates[0].content.parts[0].text");
        assert_eq!(searched.system_prompt.for_locale("en"), "Search when you need to.");
        assert_eq!(searched.auth_config_key.as_deref(), Some("api_key"));
        assert_eq!(searched.max_tokens, 2048);
    }

    #[test]
    fn searching_keeps_every_field_the_block_leaves_out() {
        let config = plain_config(RequestFormat::Anthropic);
        let search = WebSearchConfig {
            endpoint: None,
            request_format: None,
            auth: None,
            auth_header: None,
            response_path: None,
            system_prompt: None,
            tool: serde_json::json!({"type": "web_search_20250305", "name": "web_search"}),
        };
        assert_eq!(searching(&config, &search), config);
    }

    #[test]
    fn the_model_goes_into_an_endpoint_that_asks_for_it() {
        let mut config = plain_config(RequestFormat::Gemini);
        config.endpoint = Some("https://g.example.com/v1beta/models/{model}:generateContent".into());
        let mut store = MemoryConfigStore::new();
        store.set("s", "api_key", "k");
        let resolved = resolve_config(&config, "s", &store, None).unwrap();
        assert_eq!(resolved.endpoint, "https://g.example.com/v1beta/models/m-1:generateContent");
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
            web_search: None,
        };
        let resolved = ResolvedConfig {
            endpoint: "https://api.example.com".into(),
            model: "gpt-4o-mini".into(),
            api_key: Some("sk-test".into()),
            omit_params: Vec::new(),
        };
        let body = build_request_body(&config, &resolved, "What is 2+2?", "en", &[], &[], "", None);
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
            web_search: None,
        };
        let resolved = ResolvedConfig {
            endpoint: "https://api.example.com".into(),
            model: "gpt-4o-mini".into(),
            api_key: Some("sk-test".into()),
            omit_params: Vec::new(),
        };
        let facts = vec!["i am vegetarian".to_string()];
        let body = build_request_body(&config, &resolved, "what should i cook", "en", &[], &facts, "", None);
        assert!(body.contains("Things you know about the user:\\n- i am vegetarian"));

        let body = build_request_body(&config, &resolved, "what should i cook", "en", &[], &facts, "Talk like a chef.", None);
        let personality = body.find("Talk like a chef.").expect("personality missing from the system prompt");
        assert!(personality < body.find("Things you know about the user").unwrap());
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
            web_search: None,
        };
        let resolved = ResolvedConfig {
            endpoint: "https://api.example.com".into(),
            model: "gpt-4o-mini".into(),
            api_key: Some("sk-test".into()),
            omit_params: Vec::new(),
        };
        let body = build_request_body(&config, &resolved, "hi", "en", &[], &[], "", None);
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
            web_search: None,
        };
        let resolved = ResolvedConfig {
            endpoint: "https://api.example.com".into(),
            model: "gpt-4o-mini".into(),
            api_key: Some("sk-test".into()),
            omit_params: Vec::new(),
        };
        let body = build_request_body(&config, &resolved, "che ora è?", "it", &[], &[], "", None);
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
            web_search: None,
        };
        let resolved = ResolvedConfig {
            endpoint: "https://api.example.com".into(),
            model: "gpt-4o-mini".into(),
            api_key: Some("sk-test".into()),
            omit_params: Vec::new(),
        };
        let body = build_request_body(&config, &resolved, "ciao", "it", &[], &[], "", None);
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
            web_search: None,
        };
        let resolved = ResolvedConfig {
            endpoint: "https://api.example.com".into(),
            model: "gpt-4o-mini".into(),
            api_key: Some("sk-test".into()),
            omit_params: Vec::new(),
        };
        let body = build_request_body(&config, &resolved, "what time?", "en", &[], &[], "", None);
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
            web_search: None,
        };
        let resolved = ResolvedConfig {
            endpoint: "https://api.anthropic.com/v1/messages".into(),
            model: "claude-sonnet-4-6".into(),
            api_key: Some("sk-ant-test".into()),
            omit_params: Vec::new(),
        };
        let body = build_request_body(&config, &resolved, "Hello", "en", &[], &[], "", None);
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
            web_search: None,
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
        let body = build_request_body(&config, &resolved, "what is the population?", "en", &history, &[], "", None);
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        let msgs = v["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 4); // system + 2 history + current user
        assert_eq!(msgs[0]["role"], "system");
        assert!(msgs[0]["content"].as_str().unwrap().contains(ari_core::HISTORY_INSTRUCTION));
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
            web_search: None,
        };
        let resolved = ResolvedConfig {
            endpoint: "https://api.example.com".into(),
            model: "gpt-4o-mini".into(),
            api_key: Some("sk-test".into()),
            omit_params: Vec::new(),
        };
        let body = build_request_body(&config, &resolved, "hello", "en", &[], &[], "", None);
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        let msgs = v["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 2); // system + user only
        assert!(!msgs[0]["content"].as_str().unwrap().contains(ari_core::HISTORY_INSTRUCTION));
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
            web_search: None,
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
            web_search: None,
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
        let Some(answer) = ask_real_chatgpt("what is the capital of malta", WebSearch::Off) else {
            return;
        };
        eprintln!("ChatGPT response: {answer}");
        assert!(answer.to_lowercase().contains("valletta"), "expected Valletta in: {answer}");
    }

    #[test]
    #[ignore] // requires OPENAI_API_KEY env var
    fn chatgpt_real_api_call_with_web_search() {
        let Some(answer) = ask_real_chatgpt("who won the most recent formula 1 grand prix?", WebSearch::Allowed) else {
            return;
        };
        // A live answer can't be pinned down; this proves the Responses
        // request is accepted and its answer comes back whole.
        eprintln!("ChatGPT response: {answer}");
        assert!(!answer.trim().is_empty());
    }

    #[test]
    #[ignore] // requires OPENAI_API_KEY env var
    fn chatgpt_real_api_call_with_required_search() {
        let Some(answer) = ask_real_chatgpt("who came second in the most recent formula 1 grand prix?", WebSearch::Required) else {
            return;
        };
        // Proves OpenAI accepts the forced `tool_choice` with the skill's
        // search tool on the model the catalog picks.
        eprintln!("ChatGPT response: {answer}");
        assert!(!answer.trim().is_empty());
    }

    /// The answer the ChatGPT skill in a sibling ari-skills checkout gives,
    /// or `None` when there's no key to ask with.
    fn ask_real_chatgpt(question: &str, web_search: WebSearch) -> Option<String> {
        let api_key = match std::env::var("OPENAI_API_KEY") {
            Ok(k) if !k.is_empty() => k,
            _ => {
                eprintln!("OPENAI_API_KEY not set, skipping");
                return None;
            }
        };

        let skill_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../ari-skills/skills/chatgpt");
        let report = crate::loader::load_single_skill_dir(&skill_dir);
        let entry = &report.assistants[0];
        let api = entry.manifest.api.as_ref().unwrap();

        let mut store = MemoryConfigStore::new();
        store.set(&entry.id, "api_key", &api_key);

        // The registry's catalog, so the tier resolves to the model a phone
        // would use rather than the manifest's offline pin.
        let catalog_bytes = std::fs::read(skill_dir.join("../../models.json")).unwrap();
        let catalog = ModelCatalog::from_json_bytes(&catalog_bytes).unwrap();

        match call_assistant_api(api, &entry.id, &store, question, "en", &[], &[], "", Some(&catalog), web_search) {
            Ok(text) => Some(text),
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
            web_search: None,
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
            web_search: None,
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
        let body = build_request_body(&config, &resolved, "what is the population?", "en", &history, &[], "", None);
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        let msgs = v["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 3); // 2 history + current user
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"], "what is the capital of uae?");
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[1]["content"], "Abu Dhabi.");
        assert_eq!(msgs[2]["role"], "user");
        assert_eq!(msgs[2]["content"], "what is the population?");
        assert!(v["system"].as_str().unwrap().contains(ari_core::HISTORY_INSTRUCTION));
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
            web_search: None,
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

        let body = build_request_body(&config, &resolved, "hello", "en", &[], &[], "", None);
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

        let body = build_request_body(&config, &resolved, "hello", "en", &[], &[], "", None);
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

        let body = build_request_body(&config, &resolved, "hello", "en", &[], &[], "", None);
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

        let body = build_request_body(&config, &resolved, "hello", "en", &[], &[], "", None);
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed["temperature"], serde_json::json!(0.7f32));
    }
}
