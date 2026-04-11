# Assistant Skills — Design Document

## Problem

Ari's on-device LLM fallback is hardwired into the engine. It works, but it's a closed system — there's no way for a developer or user to swap in an alternative (ChatGPT, Claude, Ollama, a different local model framework) without modifying the engine source. The LLM settings page is bespoke Android code that only knows about the three bundled Gemma tiers.

We want any developer to be able to ship an alternative "brain" for Ari as a community skill, and have it appear as a first-class option in Settings alongside the built-in local LLM — with zero code, just a SKILL.md.

## Solution

A new skill type: **assistant**. An assistant skill doesn't compete in the ranking rounds. It answers when no regular skill can. Only one assistant can be active at a time. If none are active, Ari behaves exactly as it does today with "None" selected — ranking rounds fail, the user gets "Sorry, I didn't understand that."

## What an assistant skill is NOT

- It is **not** a regular skill. It never enters the three-round ranking pipeline.
- It is **not** a replacement for the engine's `Skill` trait. Regular skills (built-in, declarative, WASM) are unchanged.
- It is **not** a conversation engine. It answers one question at a time. Multi-turn context is a future enhancement.

## Manifest format

Assistant skills use the same `SKILL.md` envelope as every other Ari skill. The distinguishing field is `metadata.ari.type: assistant`. When `type` is `assistant`, the `matching` and `behaviour` (declarative/wasm) blocks are absent — replaced by `metadata.ari.assistant`.

### Built-in local LLM

```yaml
---
name: local-llm
description: >
  On-device language model. Answers general questions privately
  with no internet connection. Requires downloading a model.
metadata:
  ari:
    id: dev.heyari.assistant.local
    version: "0.1.0"
    type: assistant
    author: Ari Project
    homepage: https://github.com/ari-digital-assistant/ari
    languages: [en]
    assistant:
      provider: builtin
      privacy: local
      config:
        - key: model_tier
          label: Model
          type: select
          required: true
          options:
            - value: small
              label: "Gemma 3 1B (~769 MB)"
              download_url: "https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf"
              download_bytes: 806354944
            - value: medium
              label: "Gemma 4 E2B (~3.1 GB)"
              download_url: "https://huggingface.co/unsloth/gemma-4-e2b-it-GGUF/resolve/main/gemma-4-e2b-it-Q4_K_M.gguf"
              download_bytes: 3326083072
            - value: large
              label: "Gemma 4 E4B (~5.0 GB)"
              download_url: "https://huggingface.co/unsloth/gemma-4-e4b-it-GGUF/resolve/main/gemma-4-e4b-it-Q4_K_M.gguf"
              download_bytes: 5368709120
---
Runs a language model entirely on your device. No data leaves your phone.
Choose a model size based on your available storage and how much RAM
your device has. Smaller models are faster but less capable.
```

`provider: builtin` tells the engine: "I know what to do natively — don't make HTTP calls." The engine sees this ID as active, and routes to the existing `LazyLlmFallback` machinery. The `config` block with `type: select` and `download_url` fields drives the model download UI — same functionality as the current `LlmModelRegistry` and `LlmDownloadManager`, but now data-driven from the manifest instead of hardcoded in Kotlin.

### Cloud provider (ChatGPT)

```yaml
---
name: chatgpt
description: >
  Use OpenAI's ChatGPT to answer general questions.
  Requires an API key from platform.openai.com.
  Your questions are sent to OpenAI's servers.
metadata:
  ari:
    id: dev.heyari.assistant.chatgpt
    version: "0.1.0"
    type: assistant
    author: Ari Project
    homepage: https://github.com/ari-digital-assistant/ari
    languages: [en]
    assistant:
      provider: api
      privacy: cloud
      api:
        endpoint: https://api.openai.com/v1/chat/completions
        auth: bearer
        auth_config_key: api_key
        model_config_key: model
        default_model: gpt-4o-mini
        system_prompt: >
          You are Ari, a helpful voice assistant.
          Answer the user's question in one short sentence.
        response_path: "choices[0].message.content"
      config:
        - key: api_key
          label: API Key
          type: secret
          required: true
        - key: model
          label: Model
          type: text
          default: gpt-4o-mini
---
Uses OpenAI's ChatGPT API. You need an API key — get one at
https://platform.openai.com/api-keys. Queries are sent to
OpenAI's servers; see their privacy policy for data handling.
```

### Cloud provider (Claude)

```yaml
---
name: claude
description: >
  Use Anthropic's Claude to answer general questions.
  Requires an API key from console.anthropic.com.
  Your questions are sent to Anthropic's servers.
metadata:
  ari:
    id: dev.heyari.assistant.claude
    version: "0.1.0"
    type: assistant
    author: Ari Project
    homepage: https://github.com/ari-digital-assistant/ari
    languages: [en]
    assistant:
      provider: api
      privacy: cloud
      api:
        endpoint: https://api.anthropic.com/v1/messages
        auth: header
        auth_header: x-api-key
        auth_config_key: api_key
        model_config_key: model
        default_model: claude-sonnet-4-6
        api_version: "2023-06-01"
        api_version_header: anthropic-version
        system_prompt: >
          You are Ari, a helpful voice assistant.
          Answer the user's question in one short sentence.
        request_format: anthropic
        response_path: "content[0].text"
      config:
        - key: api_key
          label: API Key
          type: secret
          required: true
        - key: model
          label: Model
          type: text
          default: claude-sonnet-4-6
---
Uses Anthropic's Claude API. You need an API key — get one at
https://console.anthropic.com. Queries are sent to Anthropic's
servers; see their privacy policy for data handling.
```

### Self-hosted (Ollama)

```yaml
---
name: ollama
description: >
  Connect to a self-hosted Ollama instance on your local network.
  Private — traffic stays on your network.
metadata:
  ari:
    id: dev.heyari.assistant.ollama
    version: "0.1.0"
    type: assistant
    author: Ari Project
    homepage: https://github.com/ari-digital-assistant/ari
    languages: [en]
    assistant:
      provider: api
      privacy: local
      api:
        endpoint_config_key: endpoint
        default_endpoint: http://localhost:11434/v1/chat/completions
        auth: none
        model_config_key: model
        default_model: llama3
        system_prompt: >
          You are Ari, a helpful voice assistant.
          Answer the user's question in one short sentence.
        response_path: "choices[0].message.content"
      config:
        - key: endpoint
          label: Server URL
          type: text
          default: http://localhost:11434/v1/chat/completions
          required: true
        - key: model
          label: Model
          type: text
          default: llama3
          required: true
---
Connects to an Ollama server running on your local network.
Install Ollama at https://ollama.com, pull a model, and point
Ari at the server URL.
```

## Manifest schema: `metadata.ari.assistant`

### Top-level fields

| Field | Type | Required | Description |
|---|---|---|---|
| `provider` | `builtin` or `api` | yes | Routing hint. `builtin` = engine handles natively. `api` = engine uses the generic HTTP adapter. |
| `privacy` | `local` or `cloud` | yes | Displayed in the UI. `local` means data stays on-device or local network. `cloud` means queries are sent to a third-party server. |
| `api` | object | if `provider: api` | API adapter configuration. Absent for `builtin`. |
| `config` | array | no | User-configurable fields rendered in Settings. |

### `api` object

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `endpoint` | string | no* | — | Fixed API endpoint URL. |
| `endpoint_config_key` | string | no* | — | Config key whose value is the endpoint URL (for user-configurable endpoints). |
| `default_endpoint` | string | no | — | Default value when using `endpoint_config_key`. |
| `auth` | `bearer` / `header` / `none` | yes | — | Authentication scheme. |
| `auth_header` | string | if `auth: header` | `Authorization` | Custom header name for the API key. |
| `auth_config_key` | string | if auth != `none` | — | Config key that holds the API key/token. |
| `model_config_key` | string | no | — | Config key whose value overrides the model. |
| `default_model` | string | yes | — | Model identifier sent in the request body. |
| `system_prompt` | string | yes | — | System message prepended to the conversation. |
| `request_format` | `openai` / `anthropic` | no | `openai` | Request body shape. |
| `response_path` | string | yes | — | JSONPath-like expression to extract the response text. |
| `api_version` | string | no | — | Version string for APIs that require it (e.g. Anthropic). |
| `api_version_header` | string | no | — | Header name for the version string. |
| `max_tokens` | int | no | 256 | Maximum tokens in the response. |
| `temperature` | float | no | 0.7 | Sampling temperature. |

*One of `endpoint` or `endpoint_config_key` must be present.

### `request_format` shapes

**`openai`** (default) — covers OpenAI, Groq, Together, Mistral, Ollama, LM Studio, vLLM:

```json
{
  "model": "<model>",
  "max_tokens": 256,
  "temperature": 0.7,
  "messages": [
    {"role": "system", "content": "<system_prompt>"},
    {"role": "user", "content": "<user input>"}
  ]
}
```

**`anthropic`** — covers Anthropic's native API:

```json
{
  "model": "<model>",
  "max_tokens": 256,
  "system": "<system_prompt>",
  "messages": [
    {"role": "user", "content": "<user input>"}
  ]
}
```

These two formats cover every major provider as of April 2026. A WASM escape hatch exists for anything truly exotic — a WASM skill with `[http]` that builds its own request. But that's not the expected path.

### `config` array entries

| Field | Type | Required | Description |
|---|---|---|---|
| `key` | string | yes | Unique identifier. Referenced by `auth_config_key`, `model_config_key`, `endpoint_config_key`. |
| `label` | string | yes | Display label in Settings. |
| `type` | `text` / `secret` / `select` | yes | Determines UI widget and storage backend. |
| `required` | bool | no | Default false. If true, the assistant can't be activated until this field has a value. |
| `default` | string | no | Pre-filled value. |
| `options` | array | if `type: select` | List of `{ value, label }` objects. For built-in LLM, each option may include `download_url` and `download_bytes`. |

## Config storage

Config values are scoped per assistant skill ID.

| Type | Android | Linux | CLI |
|---|---|---|---|
| `text` | DataStore (existing `ari_settings`) | GSettings or file | Environment variable or dotfile |
| `secret` | EncryptedSharedPreferences (Jetpack Security) | libsecret / GNOME Keyring | Environment variable |
| `select` | DataStore | GSettings or file | CLI flag |

DataStore key format: `assistant_config_<skill_id>_<config_key>` — e.g. `assistant_config_dev.heyari.assistant.chatgpt_api_key`. This avoids collisions with existing preferences and naturally scopes to the skill.

Secrets (API keys) never appear in plain text in DataStore. On Android, `EncryptedSharedPreferences` from Jetpack Security handles encryption at rest. On Linux, `libsecret` talks to the user's keyring. The CLI is a dev tool — env vars are acceptable.

## Engine changes

### New type field in AriExtension

`AriExtension` gains an optional `type` field. When absent or `"skill"`, the existing parsing and loading logic applies unchanged. When `"assistant"`, the loader expects `metadata.ari.assistant` instead of `matching` + `declarative`/`wasm`.

```rust
pub enum SkillType {
    Skill,      // default — regular skill, enters ranking rounds
    Assistant,  // assistant provider — never enters ranking rounds
}
```

### AssistantManifest struct

Parsed from `metadata.ari.assistant`:

```rust
pub struct AssistantManifest {
    pub provider: AssistantProvider,
    pub privacy: Privacy,
    pub api: Option<ApiConfig>,
    pub config: Vec<ConfigField>,
}

pub enum AssistantProvider {
    Builtin,
    Api,
}

pub enum Privacy {
    Local,
    Cloud,
}

pub struct ApiConfig {
    pub endpoint: Option<String>,
    pub endpoint_config_key: Option<String>,
    pub default_endpoint: Option<String>,
    pub auth: AuthScheme,
    pub auth_header: Option<String>,
    pub auth_config_key: Option<String>,
    pub model_config_key: Option<String>,
    pub default_model: String,
    pub system_prompt: String,
    pub request_format: RequestFormat,
    pub response_path: String,
    pub api_version: Option<String>,
    pub api_version_header: Option<String>,
    pub max_tokens: u32,        // default 256
    pub temperature: f32,       // default 0.7
}

pub enum AuthScheme { Bearer, Header, None }
pub enum RequestFormat { OpenAi, Anthropic }

pub struct ConfigField {
    pub key: String,
    pub label: String,
    pub field_type: ConfigFieldType,
    pub required: bool,
    pub default: Option<String>,
}

pub enum ConfigFieldType {
    Text,
    Secret,
    Select { options: Vec<SelectOption> },
}

pub struct SelectOption {
    pub value: String,
    pub label: String,
    pub download_url: Option<String>,
    pub download_bytes: Option<u64>,
}
```

### Loader changes

`load_skill_directory` (and `load_single_skill_dir`) currently return `LoadReport { skills, failures }`. Assistant-type skills are **not** added to `skills` — they don't implement `Skill` and don't enter the ranking pipeline. Instead, `LoadReport` gains a new field:

```rust
pub struct LoadReport {
    pub skills: Vec<Box<dyn Skill>>,
    pub assistants: Vec<AssistantEntry>,
    pub failures: Vec<LoadFailure>,
}

pub struct AssistantEntry {
    pub id: String,
    pub name: String,
    pub description: String,
    pub manifest: AssistantManifest,
    pub body: String,
}
```

Validation at load time:
- `type: assistant` + `provider: api` requires `api` block with all mandatory fields.
- `type: assistant` + `provider: builtin` rejects `api` block.
- Config keys referenced by `auth_config_key` / `model_config_key` / `endpoint_config_key` must exist in `config`.
- `response_path` syntax is validated (simple bracket-path, not full JSONPath).
- `endpoint` or `endpoint_config_key` — exactly one must be present.

### Engine integration

The engine's `process_input_traced` flow changes from:

```
ranking rounds fail → LLM fallback (hardcoded) → FALLBACK_RESPONSE
```

to:

```
ranking rounds fail → active assistant? → route by provider → FALLBACK_RESPONSE
```

Concretely, `Engine` gains:

```rust
pub struct Engine {
    skills: Vec<Box<dyn Skill>>,
    ctx: SkillContext,
    debug: bool,
    #[cfg(feature = "llm")]
    llm: Option<Box<dyn ari_llm::Fallback>>,
    active_assistant: Option<ActiveAssistant>,
}

enum ActiveAssistant {
    Builtin,    // route to self.llm
    Api(ApiConfig, ConfigStore),
}
```

When `active_assistant` is `Some(Builtin)`, the existing `self.llm.try_answer()` path fires — zero behaviour change from today. When it's `Some(Api(..))`, the engine builds the HTTP request from the `ApiConfig`, resolves config values (model, endpoint, API key) from the `ConfigStore`, makes the call, and extracts the response via `response_path`.

When `active_assistant` is `None`, the engine returns `FALLBACK_RESPONSE` immediately. No LLM, no API call.

### ConfigStore trait

The engine needs access to config values (API keys, model names, endpoint URLs) but doesn't own the storage backend — each frontend does. Solution: a trait the frontend injects.

```rust
pub trait ConfigStore: Send + Sync {
    fn get(&self, skill_id: &str, key: &str) -> Option<String>;
}
```

Android implements this by reading from DataStore / EncryptedSharedPreferences. Linux implements it with GSettings / libsecret. The CLI implements it with env vars. The engine doesn't care where the values come from.

### API adapter

Lives in `ari-skill-loader` (reuses `tls::webpki_roots_config()` for TLS, same as the registry client and WASM `http_fetch`). Single function:

```rust
pub fn call_assistant_api(
    config: &ApiConfig,
    resolved: &ResolvedConfig,  // endpoint, model, api_key after config lookup
    user_input: &str,
) -> Result<String, AssistantApiError>
```

Builds the request body based on `request_format`, sets auth headers, POSTs, extracts response text via `response_path`. Timeout: 30 seconds (hardcoded, not configurable — antislop rule 13). Error type carries enough info for a user-facing message: timeout, auth failure, network error, unexpected response shape.

### Response path extraction

`response_path` is a minimal bracket-path syntax, not full JSONPath. Supported:

- `choices[0].message.content` — array index + field access
- `content[0].text` — same pattern
- `result.answer` — nested field access

Parsed at manifest load time into a `Vec<PathSegment>`. Applied at runtime via sequential `serde_json::Value` traversal. If any segment fails to resolve, the call returns `AssistantApiError::ResponseParse`.

## FFI surface (new exports)

```rust
// List all installed assistant skills (built-in + community)
fn list_assistants() -> Vec<FfiAssistantEntry>;

// Get/set the active assistant (None = disabled)
fn get_active_assistant() -> Option<String>;
fn set_active_assistant(id: Option<String>);

// Config management
fn get_assistant_config_schema(id: String) -> Vec<FfiConfigField>;
fn get_assistant_config_value(id: String, key: String) -> Option<String>;
fn set_assistant_config_value(id: String, key: String, value: String);

// For secret fields — frontend stores the actual secret and passes
// a retrieval callback, or the engine reads from ConfigStore at call time.
// Secrets never transit through FFI as return values.
```

```rust
pub struct FfiAssistantEntry {
    pub id: String,
    pub name: String,
    pub description: String,
    pub provider: String,   // "builtin" or "api"
    pub privacy: String,    // "local" or "cloud"
    pub active: bool,
    pub body: String,       // markdown body for detail view
}

pub struct FfiConfigField {
    pub key: String,
    pub label: String,
    pub field_type: String,     // "text", "secret", "select"
    pub required: bool,
    pub default_value: Option<String>,
    pub current_value: Option<String>,  // None for secrets
    pub options: Vec<FfiSelectOption>,  // empty for non-select
}

pub struct FfiSelectOption {
    pub value: String,
    pub label: String,
    pub download_url: Option<String>,
    pub download_bytes: Option<u64>,
}
```

## Android changes

### Settings UI

The current "LLM" settings page is replaced by an "Assistant" page. The hardcoded `LlmModelRegistry` is replaced by data from `list_assistants()` + `get_assistant_config_schema()`.

```
Settings
  └── Assistant
        ├── ○ None
        │     No assistant. Ari only answers via matched skills.
        │
        ├── ● On-Device LLM                          [local]
        │     Answers questions privately on your device.
        │     Model: [Gemma 3 1B (~769 MB) ▾]
        │            [Download]  or  [Downloaded ✓]
        │
        ├── ○ ChatGPT                          [cloud ☁]
        │     Uses OpenAI's servers.
        │     API Key: [••••••••••]
        │     Model:   [gpt-4o-mini]
        │
        └── ○ Ollama                            [local]
              Connect to your own server.
              Server URL: [http://localhost:11434/...]
              Model:      [llama3]
```

Radio buttons for selection. Expanding a provider shows its config fields, rendered generically from the `config` array. The `[cloud]` badge comes from `privacy: cloud` with a brief note that queries leave the device.

A `select` field with `download_url` entries renders the existing download UI (progress bar, cancel, delete) — the `LlmDownloadManager` is reused, just driven by manifest data instead of hardcoded constants.

### What gets deleted

- `LlmModel.kt` and `LlmModelRegistry` — replaced by the built-in assistant's manifest config.
- The bespoke `LlmSettingsPage.kt` — replaced by the generic assistant settings page.
- `KEY_ACTIVE_LLM_MODEL` in `SettingsRepository` — replaced by `KEY_ACTIVE_ASSISTANT`.

### What stays

- `LlmDownloadManager` — reused for downloading models from `download_url` in select options.
- `AriApplication.onTrimMemory` — still calls `engine.unloadLlmModel()` when memory is low.
- The engine's `load_llm_model(path)` / `unload_llm_model()` FFI — still used internally when the active assistant is the built-in.
- `EngineModule` startup logic — reads active assistant ID, if it's the built-in and a model is downloaded, calls `load_llm_model(path)`.

### Secrets storage (new)

Add `androidx.security:security-crypto` to the Android dependency tree. Create `AssistantSecretStore` that wraps `EncryptedSharedPreferences` for `type: secret` config values. Implements the `ConfigStore` trait on the FFI boundary — the engine calls `config_store.get(skill_id, key)` and the Android implementation reads from the encrypted store.

## Built-in assistant: bundled, not installed

The built-in local LLM assistant is **not** installed via the skill registry. Its `SKILL.md` is bundled in the engine (compiled in as a const string, or shipped in Android assets). It always appears in `list_assistants()` regardless of what's in the `skills/` directory. This means:

- It can't be uninstalled.
- It's always available, even offline, even on first launch.
- It doesn't go through the bundle signing pipeline.
- It updates with the engine/app, not via the registry.

Community assistant skills (ChatGPT, Claude, Ollama, etc.) are installed from the registry like any other community skill. They appear in `list_assistants()` only when installed.

## Registry and distribution

Assistant skills are distributed through the existing `ari-digital-assistant/ari-skills` registry. They go through the same PR review, validation, signing, and publishing pipeline as regular skills. The validator (`ari-skill-validate`) gains awareness of `type: assistant` manifests — it validates the `assistant` block schema, checks config key references, and validates `response_path` syntax.

Assistant skills don't have WASM modules or pattern matching, so they're always lightweight — just a `SKILL.md` in a tarball. The signing and bundle machinery handles this already (declarative skills are the same shape).

## Security considerations

1. **API keys are user-provided secrets.** They're stored in platform-appropriate secure storage, never in plain text, never logged, never sent over FFI as return values.

2. **Cloud assistant skills send user queries to third parties.** The `privacy: cloud` badge makes this visible. The first time a user activates a cloud assistant, the UI should show a one-time confirmation: "This assistant sends your questions to [provider name]'s servers. Continue?"

3. **Endpoint URLs in community skills.** A malicious skill could point to `https://evil.example.com/steal-your-queries`. Mitigations: HTTPS-only enforcement for cloud providers (same `HttpConfig::strict()` used by WASM skills), registry PR review catches obviously malicious endpoints, the user explicitly chooses to activate the assistant.

4. **The built-in assistant has no network access.** `provider: builtin` never makes HTTP calls. The engine routes directly to the local LLM. No data leaves the device.

## Future extensions (not in scope now)

- **Multi-turn conversation.** The `messages` array in the API request grows from 2 entries to N. The engine holds a conversation buffer scoped to the active assistant. Additive change — no manifest schema change needed.
- **Streaming responses.** Server-sent events for cloud providers, token-by-token for local LLM. Requires FFI callback mechanism.
- **Custom system prompt.** A user-facing config field that overrides the manifest's `system_prompt`. Already achievable by adding a `system_prompt` config key of type `text` — no engine change needed.
- **WASM assistant provider.** For APIs with genuinely incompatible request/response shapes. A WASM module that implements an `assistant_execute(input) -> string` export instead of the regular skill ABI. Only build this if a real provider can't be served by the declarative format.
- **First-run wizard.** The wtf.md already tracks this. The assistant selection would be a step in the wizard: "How should Ari answer general questions?" with the installed options.
