//! Parser for `SKILL.md` manifests.
//!
//! A `SKILL.md` is an [AgentSkills](https://agentskills.io/specification.md)
//! document: YAML frontmatter (delimited by `---` lines) followed by a Markdown
//! body. Ari-specific configuration lives entirely under `metadata.ari` in the
//! frontmatter; the body is preserved verbatim and ignored by the deterministic
//! router (it's reserved for future LLM-routing and AgentSkills tooling).
//!
//! A document is only treated as an Ari skill if `metadata.ari` is present.
//! AgentSkills documents without it parse cleanly but yield `ari_extension =
//! None`, so the loader can skip them without error.

use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;

const FRONTMATTER_DELIM: &str = "---";
const NAME_MAX_LEN: usize = 64;
const DESCRIPTION_MAX_LEN: usize = 1024;

// ── Skill type ────────────────────────────────────────────────────────

/// Distinguishes regular skills (enter ranking rounds) from assistant
/// providers (fire after all rounds fail).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SkillType {
    /// Default — a regular skill that competes in ranking rounds.
    Skill,
    /// An assistant provider that answers when no skill matches.
    Assistant,
}

impl Default for SkillType {
    fn default() -> Self {
        SkillType::Skill
    }
}

// ── Assistant manifest types ──────────────────────────────────────────

/// Parsed from `metadata.ari.assistant`.
#[derive(Debug, Clone, PartialEq)]
pub struct AssistantManifest {
    pub provider: AssistantProvider,
    pub privacy: Privacy,
    pub api: Option<ApiConfig>,
    pub config: Vec<ConfigField>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssistantProvider {
    Builtin,
    Api,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Privacy {
    Local,
    Cloud,
}

/// Configuration for `provider: api` assistant skills.
#[derive(Debug, Clone, PartialEq)]
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
    pub max_tokens: u32,
    pub temperature: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthScheme {
    Bearer,
    Header,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestFormat {
    Openai,
    Anthropic,
}

impl Default for RequestFormat {
    fn default() -> Self {
        RequestFormat::Openai
    }
}

/// A user-configurable field rendered in Settings.
#[derive(Debug, Clone, PartialEq)]
pub struct ConfigField {
    pub key: String,
    pub label: String,
    pub field_type: ConfigFieldType,
    pub required: bool,
    pub default: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConfigFieldType {
    Text,
    Secret,
    Select { options: Vec<SelectOption> },
}

#[derive(Debug, Clone, PartialEq)]
pub struct SelectOption {
    pub value: String,
    pub label: String,
    pub download_url: Option<String>,
    pub download_bytes: Option<u64>,
}

// ── Skill examples (for FunctionGemma training) ──────────────────────

/// One example utterance in a SKILL.md manifest. Used as training data
/// for the FunctionGemma skill router.
#[derive(Debug, Clone, PartialEq)]
pub struct SkillExample {
    pub text: String,
    pub args: Option<serde_json::Value>,
}

const MIN_EXAMPLES: usize = 5;

// ── Errors ─────────────────────────────────────────────────────────────

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ManifestError {
    #[error("file is missing the opening `---` frontmatter delimiter")]
    MissingFrontmatterStart,

    #[error("file is missing the closing `---` frontmatter delimiter")]
    MissingFrontmatterEnd,

    #[error("YAML frontmatter could not be parsed: {0}")]
    YamlParse(String),

    #[error("`name` field is required")]
    MissingName,

    #[error("`name` must be 1-{NAME_MAX_LEN} characters")]
    NameLength,

    #[error("`name` may only contain lowercase letters, digits, and hyphens")]
    NameCharset,

    #[error("`name` must not start or end with a hyphen")]
    NameHyphenEdge,

    #[error("`name` must not contain consecutive hyphens")]
    NameDoubleHyphen,

    #[error("`name` ({name:?}) must match the parent directory name ({dir:?})")]
    NameDirMismatch { name: String, dir: String },

    #[error("`description` field is required")]
    MissingDescription,

    #[error("`description` must be 1-{DESCRIPTION_MAX_LEN} characters")]
    DescriptionLength,

    #[error("`metadata.ari.id` is required")]
    MissingAriId,

    #[error("`metadata.ari.version` is required")]
    MissingAriVersion,

    #[error("`metadata.ari.engine` is required")]
    MissingAriEngine,

    #[error("`metadata.ari.matching.patterns` must contain at least one entry")]
    EmptyPatterns,

    #[error("a matching pattern must contain either `keywords` or `regex`")]
    PatternMissingMatcher,

    #[error("a matching pattern must not contain both `keywords` and `regex`")]
    PatternConflictingMatcher,

    #[error("`keywords` list must not be empty")]
    EmptyKeywords,

    #[error(
        "exactly one of `metadata.ari.declarative` or `metadata.ari.wasm` must be present \
         (found {found})"
    )]
    BehaviourCardinality { found: &'static str },

    #[error(
        "declarative skill must set exactly one of `response`, `response_pick`, or \
         `response_template`"
    )]
    DeclarativeResponseCardinality,

    #[error("`response_pick` list must not be empty")]
    EmptyResponsePick,

    #[error(
        "`metadata.ari.matching.custom_score = true` is only valid for WASM skills"
    )]
    CustomScoreWithoutWasm,

    #[error("`metadata.ari.assistant` is required when `type` is `assistant`")]
    MissingAssistantBlock,

    #[error("`metadata.ari.assistant` must not be present for regular skills")]
    UnexpectedAssistantBlock,

    #[error("assistant skill must not have `matching`, `declarative`, or `wasm` blocks")]
    AssistantHasSkillFields,

    #[error("`metadata.ari.assistant.provider` is required")]
    MissingAssistantProvider,

    #[error("`metadata.ari.assistant.privacy` is required")]
    MissingAssistantPrivacy,

    #[error("`metadata.ari.assistant.api` is required when provider is `api`")]
    MissingApiBlock,

    #[error("`metadata.ari.assistant.api` must not be present when provider is `builtin`")]
    UnexpectedApiBlock,

    #[error("`metadata.ari.assistant.api` must have either `endpoint` or `endpoint_config_key`")]
    MissingEndpoint,

    #[error("`metadata.ari.assistant.api` must not have both `endpoint` and `endpoint_config_key`")]
    ConflictingEndpoint,

    #[error("`metadata.ari.assistant.api.default_model` is required")]
    MissingDefaultModel,

    #[error("`metadata.ari.assistant.api.system_prompt` is required")]
    MissingSystemPrompt,

    #[error("`metadata.ari.assistant.api.response_path` is required")]
    MissingResponsePath,

    #[error("`metadata.ari.assistant.api.auth_config_key` is required when auth is not `none`")]
    MissingAuthConfigKey,

    #[error("config key `{key}` referenced by `{field}` not found in config array")]
    ConfigKeyNotFound { key: String, field: String },

    #[error("invalid response_path syntax: `{path}`")]
    InvalidResponsePath { path: String },

    #[error("`metadata.ari.examples` must contain at least {MIN_EXAMPLES} entries (found {found})")]
    TooFewExamples { found: usize },

    #[error("example utterance missing `text` field")]
    ExampleMissingText,

    #[error("`metadata.ari.wasm.memory_limit_mb` must be 1..=16 (found {found})")]
    MemoryLimitOutOfRange { found: u32 },
}

/// A fully parsed `SKILL.md`. Holds both the raw AgentSkills frontmatter fields
/// and (when present) the Ari-specific extension.
#[derive(Debug, Clone, PartialEq)]
pub struct Skillfile {
    pub name: String,
    pub description: String,
    pub license: Option<String>,
    pub compatibility: Option<String>,
    pub ari_extension: Option<AriExtension>,
    pub body: String,
}

/// Everything that lives under `metadata.ari` in the frontmatter.
#[derive(Debug, Clone, PartialEq)]
pub struct AriExtension {
    pub id: String,
    pub version: String,
    pub author: Option<String>,
    pub homepage: Option<String>,
    pub engine: String,
    pub capabilities: Vec<Capability>,
    pub platforms: Option<Vec<String>>,
    pub languages: Vec<String>,
    pub skill_type: SkillType,
    /// Present only for regular skills (`skill_type == Skill`).
    pub specificity: SpecificityLevel,
    /// Present only for regular skills (`skill_type == Skill`).
    pub matching: Option<Matching>,
    /// Present only for regular skills (`skill_type == Skill`).
    pub behaviour: Option<Behaviour>,
    /// Present only for assistant skills (`skill_type == Assistant`).
    pub assistant: Option<AssistantManifest>,
    /// Example utterances for FunctionGemma training. Required for
    /// regular skills (minimum 5), optional for assistant skills.
    pub examples: Vec<SkillExample>,
    /// User-configurable fields for this skill, rendered as a settings
    /// panel in the frontend. Any skill type may declare these —
    /// WASM skills read current values at runtime via the `storage_kv`
    /// host imports, declarative skills via `{{config.<key>}}` template
    /// interpolation, and assistant skills via the `auth_config_key` /
    /// `model_config_key` cross-references inside `assistant.api`.
    ///
    /// For legacy assistant manifests that still declare
    /// `metadata.ari.assistant.config`, the parser copies those entries
    /// into this field so old and new frontmatter produce the same
    /// runtime shape. The deprecated sub-location stays readable but
    /// new skills should put settings at the top level.
    pub settings: Vec<ConfigField>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpecificityLevel {
    Low,
    Medium,
    High,
}

impl SpecificityLevel {
    pub fn as_core(self) -> ari_core::Specificity {
        match self {
            SpecificityLevel::Low => ari_core::Specificity::Low,
            SpecificityLevel::Medium => ari_core::Specificity::Medium,
            SpecificityLevel::High => ari_core::Specificity::High,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    Http,
    Location,
    Notifications,
    LaunchApp,
    Clipboard,
    Tts,
    StorageKv,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Matching {
    pub patterns: Vec<MatchPattern>,
    pub custom_score: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MatchPattern {
    Keywords { words: Vec<String>, weight: f32 },
    Regex { pattern: String, weight: f32 },
}

impl MatchPattern {
    pub fn weight(&self) -> f32 {
        match self {
            MatchPattern::Keywords { weight, .. } | MatchPattern::Regex { weight, .. } => *weight,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Behaviour {
    Declarative(DeclarativeBehaviour),
    Wasm(WasmBehaviour),
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeclarativeBehaviour {
    pub response: ResponseSpec,
    pub action: Option<serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ResponseSpec {
    Fixed(String),
    Pick(Vec<String>),
    Template(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct WasmBehaviour {
    pub module: String,
    pub memory_limit_mb: u32,
}

impl Skillfile {
    /// Parse a `SKILL.md` from raw bytes. `parent_dir_name` is the name of the
    /// directory that contains the file (used to enforce the AgentSkills rule
    /// that `name` must match the parent dir). Pass `None` to skip that check —
    /// useful for unit tests that don't care about on-disk layout.
    pub fn parse(source: &str, parent_dir_name: Option<&str>) -> Result<Self, ManifestError> {
        let (frontmatter, body) = split_frontmatter(source)?;
        let raw: RawDocument = serde_yaml_ng::from_str(frontmatter)
            .map_err(|e| ManifestError::YamlParse(e.to_string()))?;

        let name = raw.name.ok_or(ManifestError::MissingName)?;
        validate_name(&name)?;
        if let Some(dir) = parent_dir_name {
            if dir != name {
                return Err(ManifestError::NameDirMismatch {
                    name: name.clone(),
                    dir: dir.to_string(),
                });
            }
        }

        let description = raw.description.ok_or(ManifestError::MissingDescription)?;
        if description.is_empty() || description.len() > DESCRIPTION_MAX_LEN {
            return Err(ManifestError::DescriptionLength);
        }

        let ari_extension = raw
            .metadata
            .as_ref()
            .and_then(|m| m.ari.as_ref())
            .map(AriExtension::from_raw)
            .transpose()?;

        Ok(Skillfile {
            name,
            description,
            license: raw.license,
            compatibility: raw.compatibility,
            ari_extension,
            body: body.to_string(),
        })
    }

    /// Convenience: parse a file on disk, deriving `parent_dir_name` from the
    /// path.
    pub fn parse_file(path: &Path) -> Result<Self, ManifestError> {
        let source = std::fs::read_to_string(path)
            .map_err(|e| ManifestError::YamlParse(format!("could not read {path:?}: {e}")))?;
        let parent = path
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str());
        Self::parse(&source, parent)
    }
}

impl AriExtension {
    /// Check that the skill has enough example utterances. Called by
    /// the validator at PR review time, not at install time — existing
    /// skills without examples must still load.
    pub fn validate_examples(&self) -> Result<(), ManifestError> {
        if self.skill_type == SkillType::Skill && self.examples.len() < MIN_EXAMPLES {
            return Err(ManifestError::TooFewExamples {
                found: self.examples.len(),
            });
        }
        Ok(())
    }

    fn from_raw(raw: &RawAriExtension) -> Result<Self, ManifestError> {
        let id = raw
            .id
            .clone()
            .filter(|s| !s.is_empty())
            .ok_or(ManifestError::MissingAriId)?;
        let version = raw
            .version
            .clone()
            .filter(|s| !s.is_empty())
            .ok_or(ManifestError::MissingAriVersion)?;
        let engine = raw
            .engine
            .clone()
            .filter(|s| !s.is_empty())
            .ok_or(ManifestError::MissingAriEngine)?;

        let skill_type = raw.skill_type.unwrap_or_default();

        // Resolve user-facing settings. Top-level `metadata.ari.settings`
        // is the canonical location. For back-compat with pre-migration
        // assistant manifests we also accept `metadata.ari.assistant.config`
        // — but only when `settings` is empty. Mixing both is a bug we'd
        // rather surface than silently merge, so if both are populated we
        // error out.
        let settings_raw: &[RawConfigField] = if !raw.settings.is_empty() {
            if let Some(asst) = raw.assistant.as_ref() {
                if !asst.config.is_empty() {
                    return Err(ManifestError::YamlParse(
                        "skill declares both top-level `settings` and legacy \
                         `assistant.config` — pick one (prefer top-level `settings`)"
                            .to_string(),
                    ));
                }
            }
            &raw.settings
        } else {
            raw.assistant
                .as_ref()
                .map(|a| a.config.as_slice())
                .unwrap_or(&[])
        };
        let settings = settings_raw
            .iter()
            .map(ConfigField::from_raw)
            .collect::<Result<Vec<_>, _>>()?;

        let (matching, behaviour, assistant) = match skill_type {
            SkillType::Skill => {
                if raw.assistant.is_some() {
                    return Err(ManifestError::UnexpectedAssistantBlock);
                }
                let raw_matching =
                    raw.matching.as_ref().ok_or(ManifestError::EmptyPatterns)?;
                let matching = Matching::from_raw(raw_matching)?;

                let behaviour = match (raw.declarative.as_ref(), raw.wasm.as_ref()) {
                    (Some(d), None) => {
                        Behaviour::Declarative(DeclarativeBehaviour::from_raw(d)?)
                    }
                    (None, Some(w)) => Behaviour::Wasm(WasmBehaviour::from_raw(w)?),
                    (None, None) => {
                        return Err(ManifestError::BehaviourCardinality {
                            found: "neither",
                        })
                    }
                    (Some(_), Some(_)) => {
                        return Err(ManifestError::BehaviourCardinality { found: "both" })
                    }
                };

                if matching.custom_score && !matches!(behaviour, Behaviour::Wasm(_)) {
                    return Err(ManifestError::CustomScoreWithoutWasm);
                }

                (Some(matching), Some(behaviour), None)
            }
            SkillType::Assistant => {
                if raw.matching.is_some()
                    || raw.declarative.is_some()
                    || raw.wasm.is_some()
                {
                    return Err(ManifestError::AssistantHasSkillFields);
                }
                let raw_asst = raw
                    .assistant
                    .as_ref()
                    .ok_or(ManifestError::MissingAssistantBlock)?;
                let assistant = AssistantManifest::from_raw(raw_asst, &settings)?;
                (None, None, Some(assistant))
            }
        };

        // Parse examples.
        let examples: Vec<SkillExample> = raw
            .examples
            .iter()
            .map(|e| {
                let text = e
                    .text
                    .clone()
                    .filter(|s| !s.is_empty())
                    .ok_or(ManifestError::ExampleMissingText)?;
                Ok(SkillExample {
                    text,
                    args: e.args.clone(),
                })
            })
            .collect::<Result<Vec<_>, ManifestError>>()?;

        Ok(AriExtension {
            id,
            version,
            author: raw.author.clone(),
            homepage: raw.homepage.clone(),
            engine,
            capabilities: raw.capabilities.clone().unwrap_or_default(),
            platforms: raw.platforms.clone(),
            languages: raw.languages.clone().unwrap_or_default(),
            skill_type,
            specificity: raw.specificity.unwrap_or(SpecificityLevel::Medium),
            matching,
            behaviour,
            assistant,
            examples,
            settings,
        })
    }
}

impl Matching {
    fn from_raw(raw: &RawMatching) -> Result<Self, ManifestError> {
        if raw.patterns.is_empty() {
            return Err(ManifestError::EmptyPatterns);
        }
        let patterns = raw
            .patterns
            .iter()
            .map(MatchPattern::from_raw)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Matching {
            patterns,
            custom_score: raw.custom_score.unwrap_or(false),
        })
    }
}

impl MatchPattern {
    fn from_raw(raw: &RawPattern) -> Result<Self, ManifestError> {
        let weight = raw.weight.unwrap_or(1.0);
        match (raw.keywords.as_ref(), raw.regex.as_ref()) {
            (Some(words), None) => {
                if words.is_empty() {
                    return Err(ManifestError::EmptyKeywords);
                }
                Ok(MatchPattern::Keywords {
                    words: words.iter().map(|w| w.to_lowercase()).collect(),
                    weight,
                })
            }
            (None, Some(pattern)) => Ok(MatchPattern::Regex {
                pattern: pattern.clone(),
                weight,
            }),
            (None, None) => Err(ManifestError::PatternMissingMatcher),
            (Some(_), Some(_)) => Err(ManifestError::PatternConflictingMatcher),
        }
    }
}

impl DeclarativeBehaviour {
    fn from_raw(raw: &RawDeclarative) -> Result<Self, ManifestError> {
        let response = match (
            raw.response.as_ref(),
            raw.response_pick.as_ref(),
            raw.response_template.as_ref(),
        ) {
            (Some(r), None, None) => ResponseSpec::Fixed(r.clone()),
            (None, Some(p), None) => {
                if p.is_empty() {
                    return Err(ManifestError::EmptyResponsePick);
                }
                ResponseSpec::Pick(p.clone())
            }
            (None, None, Some(t)) => ResponseSpec::Template(t.clone()),
            _ => return Err(ManifestError::DeclarativeResponseCardinality),
        };
        Ok(DeclarativeBehaviour {
            response,
            action: raw.action.clone(),
        })
    }
}

impl WasmBehaviour {
    fn from_raw(raw: &RawWasm) -> Result<Self, ManifestError> {
        let mem = raw.memory_limit_mb.unwrap_or(16);
        if !(1..=16).contains(&mem) {
            return Err(ManifestError::MemoryLimitOutOfRange { found: mem });
        }
        Ok(WasmBehaviour {
            module: raw.module.clone(),
            memory_limit_mb: mem,
        })
    }
}

impl AssistantManifest {
    /// `settings` is the already-merged settings list from
    /// [`AriExtension::from_raw`] — the top-level `metadata.ari.settings`
    /// (new) or the legacy `metadata.ari.assistant.config` (old),
    /// whichever was present. This struct mirrors it into its own
    /// `config` field so callers that still read `manifest.config`
    /// stay working during the transition.
    fn from_raw(raw: &RawAssistant, settings: &[ConfigField]) -> Result<Self, ManifestError> {
        let provider = raw.provider.ok_or(ManifestError::MissingAssistantProvider)?;
        let privacy = raw.privacy.ok_or(ManifestError::MissingAssistantPrivacy)?;

        let api = match provider {
            AssistantProvider::Api => {
                let raw_api = raw.api.as_ref().ok_or(ManifestError::MissingApiBlock)?;
                Some(ApiConfig::from_raw(raw_api, settings)?)
            }
            AssistantProvider::Builtin => {
                if raw.api.is_some() {
                    return Err(ManifestError::UnexpectedApiBlock);
                }
                None
            }
        };

        Ok(AssistantManifest {
            provider,
            privacy,
            api,
            config: settings.to_vec(),
        })
    }
}

impl ApiConfig {
    fn from_raw(
        raw: &RawApiConfig,
        settings: &[ConfigField],
    ) -> Result<Self, ManifestError> {
        // Endpoint: exactly one of fixed or config-key.
        match (&raw.endpoint, &raw.endpoint_config_key) {
            (None, None) => return Err(ManifestError::MissingEndpoint),
            (Some(_), Some(_)) => return Err(ManifestError::ConflictingEndpoint),
            _ => {}
        }

        let auth = raw.auth.unwrap_or(AuthScheme::None);

        // Auth config key required when auth is not none.
        if !matches!(auth, AuthScheme::None) && raw.auth_config_key.is_none() {
            return Err(ManifestError::MissingAuthConfigKey);
        }

        let default_model = raw
            .default_model
            .clone()
            .filter(|s| !s.is_empty())
            .ok_or(ManifestError::MissingDefaultModel)?;

        let system_prompt = raw
            .system_prompt
            .clone()
            .filter(|s| !s.is_empty())
            .ok_or(ManifestError::MissingSystemPrompt)?;

        let response_path = raw
            .response_path
            .clone()
            .filter(|s| !s.is_empty())
            .ok_or(ManifestError::MissingResponsePath)?;

        validate_response_path(&response_path)?;

        // Validate that referenced config keys exist. Resolves against
        // the unified settings list (top-level `metadata.ari.settings` —
        // or the legacy `assistant.config` location, since
        // `AriExtension::from_raw` merges them before we get here).
        let config_keys: Vec<&str> = settings.iter().map(|f| f.key.as_str()).collect();

        if let Some(ref key) = raw.auth_config_key {
            if !config_keys.contains(&key.as_str()) {
                return Err(ManifestError::ConfigKeyNotFound {
                    key: key.clone(),
                    field: "auth_config_key".to_string(),
                });
            }
        }
        if let Some(ref key) = raw.model_config_key {
            if !config_keys.contains(&key.as_str()) {
                return Err(ManifestError::ConfigKeyNotFound {
                    key: key.clone(),
                    field: "model_config_key".to_string(),
                });
            }
        }
        if let Some(ref key) = raw.endpoint_config_key {
            if !config_keys.contains(&key.as_str()) {
                return Err(ManifestError::ConfigKeyNotFound {
                    key: key.clone(),
                    field: "endpoint_config_key".to_string(),
                });
            }
        }

        Ok(ApiConfig {
            endpoint: raw.endpoint.clone(),
            endpoint_config_key: raw.endpoint_config_key.clone(),
            default_endpoint: raw.default_endpoint.clone(),
            auth,
            auth_header: raw.auth_header.clone(),
            auth_config_key: raw.auth_config_key.clone(),
            model_config_key: raw.model_config_key.clone(),
            default_model,
            system_prompt,
            request_format: raw.request_format.unwrap_or_default(),
            response_path,
            api_version: raw.api_version.clone(),
            api_version_header: raw.api_version_header.clone(),
            max_tokens: raw.max_tokens.unwrap_or(256),
            temperature: raw.temperature.unwrap_or(0.7),
        })
    }
}

impl ConfigField {
    fn from_raw(raw: &RawConfigField) -> Result<Self, ManifestError> {
        let key = raw
            .key
            .clone()
            .filter(|s| !s.is_empty())
            .ok_or(ManifestError::YamlParse("config field missing `key`".into()))?;
        let label = raw
            .label
            .clone()
            .filter(|s| !s.is_empty())
            .ok_or(ManifestError::YamlParse(
                "config field missing `label`".into(),
            ))?;
        let field_type = match raw.field_type.as_deref() {
            Some("text") => ConfigFieldType::Text,
            Some("secret") => ConfigFieldType::Secret,
            Some("select") => {
                let options = raw
                    .options
                    .iter()
                    .map(|o| {
                        Ok(SelectOption {
                            value: o
                                .value
                                .clone()
                                .filter(|s| !s.is_empty())
                                .ok_or(ManifestError::YamlParse(
                                    "select option missing `value`".into(),
                                ))?,
                            label: o
                                .label
                                .clone()
                                .filter(|s| !s.is_empty())
                                .ok_or(ManifestError::YamlParse(
                                    "select option missing `label`".into(),
                                ))?,
                            download_url: o.download_url.clone(),
                            download_bytes: o.download_bytes,
                        })
                    })
                    .collect::<Result<Vec<_>, ManifestError>>()?;
                ConfigFieldType::Select { options }
            }
            Some(other) => {
                return Err(ManifestError::YamlParse(format!(
                    "unknown config field type: {other}"
                )))
            }
            None => {
                return Err(ManifestError::YamlParse(
                    "config field missing `type`".into(),
                ))
            }
        };
        Ok(ConfigField {
            key,
            label,
            field_type,
            required: raw.required,
            default: raw.default.clone(),
        })
    }
}

// ── Response path parsing ─────────────────────────────────────────────

/// A segment of a parsed response path like `choices[0].message.content`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PathSegment {
    Field(String),
    Index(usize),
}

/// Parse a response path string into segments. Supports `field`,
/// `field[N]`, and dotted chains like `choices[0].message.content`.
pub fn parse_response_path(path: &str) -> Result<Vec<PathSegment>, ManifestError> {
    let mut segments = Vec::new();
    for part in path.split('.') {
        if part.is_empty() {
            return Err(ManifestError::InvalidResponsePath {
                path: path.to_string(),
            });
        }
        if let Some(bracket) = part.find('[') {
            let field = &part[..bracket];
            if !field.is_empty() {
                segments.push(PathSegment::Field(field.to_string()));
            }
            let rest = &part[bracket..];
            if !rest.ends_with(']') {
                return Err(ManifestError::InvalidResponsePath {
                    path: path.to_string(),
                });
            }
            let idx_str = &rest[1..rest.len() - 1];
            let idx: usize =
                idx_str
                    .parse()
                    .map_err(|_| ManifestError::InvalidResponsePath {
                        path: path.to_string(),
                    })?;
            segments.push(PathSegment::Index(idx));
        } else {
            segments.push(PathSegment::Field(part.to_string()));
        }
    }
    if segments.is_empty() {
        return Err(ManifestError::InvalidResponsePath {
            path: path.to_string(),
        });
    }
    Ok(segments)
}

/// Extract a string value from a JSON value using a parsed path.
pub fn extract_by_path(value: &serde_json::Value, segments: &[PathSegment]) -> Option<String> {
    let mut current = value;
    for seg in segments {
        match seg {
            PathSegment::Field(name) => {
                current = current.get(name)?;
            }
            PathSegment::Index(idx) => {
                current = current.get(idx)?;
            }
        }
    }
    current.as_str().map(|s| s.to_string())
}

fn validate_response_path(path: &str) -> Result<(), ManifestError> {
    parse_response_path(path)?;
    Ok(())
}

fn split_frontmatter(source: &str) -> Result<(&str, &str), ManifestError> {
    let trimmed = source.trim_start_matches('\u{feff}');
    let after_first = trimmed
        .strip_prefix(FRONTMATTER_DELIM)
        .and_then(|rest| rest.strip_prefix('\n').or_else(|| rest.strip_prefix("\r\n")))
        .ok_or(ManifestError::MissingFrontmatterStart)?;

    let end_idx = find_closing_delim(after_first).ok_or(ManifestError::MissingFrontmatterEnd)?;
    let frontmatter = &after_first[..end_idx];
    let body_start = end_idx + FRONTMATTER_DELIM.len();
    let body = after_first[body_start..]
        .trim_start_matches('\n')
        .trim_start_matches("\r\n");
    Ok((frontmatter, body))
}

fn find_closing_delim(s: &str) -> Option<usize> {
    let mut idx = 0;
    for line in s.split_inclusive('\n') {
        let stripped = line.trim_end_matches('\n').trim_end_matches('\r');
        if stripped == FRONTMATTER_DELIM {
            return Some(idx);
        }
        idx += line.len();
    }
    None
}

fn validate_name(name: &str) -> Result<(), ManifestError> {
    if name.is_empty() || name.len() > NAME_MAX_LEN {
        return Err(ManifestError::NameLength);
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
    {
        return Err(ManifestError::NameCharset);
    }
    if name.starts_with('-') || name.ends_with('-') {
        return Err(ManifestError::NameHyphenEdge);
    }
    if name.contains("--") {
        return Err(ManifestError::NameDoubleHyphen);
    }
    Ok(())
}

// --- raw deserialisation types -------------------------------------------------

#[derive(Debug, Deserialize)]
struct RawDocument {
    name: Option<String>,
    description: Option<String>,
    license: Option<String>,
    compatibility: Option<String>,
    metadata: Option<RawMetadata>,
}

#[derive(Debug, Deserialize)]
struct RawMetadata {
    ari: Option<RawAriExtension>,
}

#[derive(Debug, Deserialize)]
struct RawAriExtension {
    id: Option<String>,
    version: Option<String>,
    author: Option<String>,
    homepage: Option<String>,
    engine: Option<String>,
    capabilities: Option<Vec<Capability>>,
    platforms: Option<Vec<String>>,
    languages: Option<Vec<String>>,
    #[serde(rename = "type")]
    skill_type: Option<SkillType>,
    specificity: Option<SpecificityLevel>,
    matching: Option<RawMatching>,
    declarative: Option<RawDeclarative>,
    wasm: Option<RawWasm>,
    assistant: Option<RawAssistant>,
    #[serde(default)]
    examples: Vec<RawExample>,
    /// Top-level user-configurable fields. Replaces the legacy
    /// `metadata.ari.assistant.config` location (which still parses
    /// for back-compat — see [`AriExtension::from_raw`]).
    #[serde(default)]
    settings: Vec<RawConfigField>,
}

#[derive(Debug, Deserialize)]
struct RawExample {
    text: Option<String>,
    #[serde(default)]
    args: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct RawMatching {
    patterns: Vec<RawPattern>,
    custom_score: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct RawPattern {
    keywords: Option<Vec<String>>,
    regex: Option<String>,
    weight: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct RawDeclarative {
    response: Option<String>,
    response_pick: Option<Vec<String>>,
    response_template: Option<String>,
    action: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct RawWasm {
    module: String,
    memory_limit_mb: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct RawAssistant {
    provider: Option<AssistantProvider>,
    privacy: Option<Privacy>,
    api: Option<RawApiConfig>,
    #[serde(default)]
    config: Vec<RawConfigField>,
}

#[derive(Debug, Deserialize)]
struct RawApiConfig {
    endpoint: Option<String>,
    endpoint_config_key: Option<String>,
    default_endpoint: Option<String>,
    auth: Option<AuthScheme>,
    auth_header: Option<String>,
    auth_config_key: Option<String>,
    model_config_key: Option<String>,
    default_model: Option<String>,
    system_prompt: Option<String>,
    request_format: Option<RequestFormat>,
    response_path: Option<String>,
    api_version: Option<String>,
    api_version_header: Option<String>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct RawConfigField {
    key: Option<String>,
    label: Option<String>,
    #[serde(rename = "type")]
    field_type: Option<String>,
    #[serde(default)]
    required: bool,
    default: Option<String>,
    #[serde(default)]
    options: Vec<RawSelectOption>,
}

#[derive(Debug, Deserialize)]
struct RawSelectOption {
    value: Option<String>,
    label: Option<String>,
    download_url: Option<String>,
    download_bytes: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn coin_flip_source() -> &'static str {
        r#"---
name: coin-flip
description: Flips a virtual coin and returns heads or tails. Use when the user asks to flip a coin.
license: MIT
metadata:
  ari:
    id: ai.example.coinflip
    version: "0.1.0"
    author: Mira <mira@example.com>
    engine: ">=0.3,<0.4"
    capabilities: []
    languages: [en]
    specificity: high
    matching:
      patterns:
        - keywords: [flip, coin]
          weight: 0.95
        - keywords: [toss, coin]
          weight: 0.95
    declarative:
      response_pick: ["Heads.", "Tails."]
---

# Coin Flip
Flips a virtual coin.
"#
    }

    #[test]
    fn parses_canonical_coin_flip() {
        let sf = Skillfile::parse(coin_flip_source(), Some("coin-flip")).unwrap();
        assert_eq!(sf.name, "coin-flip");
        assert_eq!(sf.license.as_deref(), Some("MIT"));
        assert!(sf.body.starts_with("# Coin Flip"));

        let ari = sf.ari_extension.expect("ari extension present");
        assert_eq!(ari.id, "ai.example.coinflip");
        assert_eq!(ari.version, "0.1.0");
        assert_eq!(ari.engine, ">=0.3,<0.4");
        assert_eq!(ari.specificity, SpecificityLevel::High);
        assert_eq!(ari.languages, vec!["en"]);
        assert!(ari.capabilities.is_empty());
        let matching = ari.matching.as_ref().expect("matching present for skill");
        assert_eq!(matching.patterns.len(), 2);
        assert!(!matching.custom_score);

        match &matching.patterns[0] {
            MatchPattern::Keywords { words, weight } => {
                assert_eq!(words, &vec!["flip".to_string(), "coin".to_string()]);
                assert_eq!(*weight, 0.95);
            }
            _ => panic!("expected keywords pattern"),
        }

        match ari.behaviour {
            Some(Behaviour::Declarative(DeclarativeBehaviour {
                response: ResponseSpec::Pick(picks),
                action: None,
            })) => {
                assert_eq!(picks, vec!["Heads.".to_string(), "Tails.".to_string()]);
            }
            _ => panic!("expected declarative response_pick with no action"),
        }
    }

    #[test]
    fn agentskills_doc_without_ari_extension_parses_but_yields_none() {
        let source = "---\nname: pdf-tools\ndescription: Helps with PDFs.\n---\nbody";
        let sf = Skillfile::parse(source, Some("pdf-tools")).unwrap();
        assert_eq!(sf.name, "pdf-tools");
        assert!(sf.ari_extension.is_none());
    }

    #[test]
    fn rejects_missing_frontmatter_start() {
        let err = Skillfile::parse("no frontmatter here", None).unwrap_err();
        assert_eq!(err, ManifestError::MissingFrontmatterStart);
    }

    #[test]
    fn rejects_missing_frontmatter_end() {
        let err = Skillfile::parse("---\nname: foo\ndescription: bar\n", None).unwrap_err();
        assert_eq!(err, ManifestError::MissingFrontmatterEnd);
    }

    #[test]
    fn rejects_uppercase_name() {
        let src = "---\nname: Coin-Flip\ndescription: x\n---\n";
        let err = Skillfile::parse(src, None).unwrap_err();
        assert_eq!(err, ManifestError::NameCharset);
    }

    #[test]
    fn rejects_leading_hyphen_name() {
        let src = "---\nname: -coin\ndescription: x\n---\n";
        assert_eq!(
            Skillfile::parse(src, None).unwrap_err(),
            ManifestError::NameHyphenEdge
        );
    }

    #[test]
    fn rejects_trailing_hyphen_name() {
        let src = "---\nname: coin-\ndescription: x\n---\n";
        assert_eq!(
            Skillfile::parse(src, None).unwrap_err(),
            ManifestError::NameHyphenEdge
        );
    }

    #[test]
    fn rejects_double_hyphen_name() {
        let src = "---\nname: coin--flip\ndescription: x\n---\n";
        assert_eq!(
            Skillfile::parse(src, None).unwrap_err(),
            ManifestError::NameDoubleHyphen
        );
    }

    #[test]
    fn rejects_overly_long_name() {
        let long = "a".repeat(65);
        let src = format!("---\nname: {long}\ndescription: x\n---\n");
        assert_eq!(
            Skillfile::parse(&src, None).unwrap_err(),
            ManifestError::NameLength
        );
    }

    #[test]
    fn rejects_name_dir_mismatch() {
        let src = "---\nname: coin-flip\ndescription: x\n---\n";
        let err = Skillfile::parse(src, Some("something-else")).unwrap_err();
        assert_eq!(
            err,
            ManifestError::NameDirMismatch {
                name: "coin-flip".to_string(),
                dir: "something-else".to_string()
            }
        );
    }

    #[test]
    fn rejects_overly_long_description() {
        let long = "a".repeat(1025);
        let src = format!("---\nname: x\ndescription: {long}\n---\n");
        assert_eq!(
            Skillfile::parse(&src, None).unwrap_err(),
            ManifestError::DescriptionLength
        );
    }

    #[test]
    fn rejects_both_declarative_and_wasm() {
        let src = r#"---
name: x
description: x
metadata:
  ari:
    id: a.b.c
    version: "1"
    engine: ">=0.3"
    matching:
      patterns:
        - keywords: [foo]
    declarative:
      response: hi
    wasm:
      module: x.wasm
---
"#;
        let err = Skillfile::parse(src, None).unwrap_err();
        assert_eq!(err, ManifestError::BehaviourCardinality { found: "both" });
    }

    #[test]
    fn rejects_neither_declarative_nor_wasm() {
        let src = r#"---
name: x
description: x
metadata:
  ari:
    id: a.b.c
    version: "1"
    engine: ">=0.3"
    matching:
      patterns:
        - keywords: [foo]
---
"#;
        let err = Skillfile::parse(src, None).unwrap_err();
        assert_eq!(err, ManifestError::BehaviourCardinality { found: "neither" });
    }

    #[test]
    fn rejects_pattern_with_both_keywords_and_regex() {
        let src = r#"---
name: x
description: x
metadata:
  ari:
    id: a.b.c
    version: "1"
    engine: ">=0.3"
    matching:
      patterns:
        - keywords: [foo]
          regex: "bar"
    declarative:
      response: hi
---
"#;
        assert_eq!(
            Skillfile::parse(src, None).unwrap_err(),
            ManifestError::PatternConflictingMatcher
        );
    }

    #[test]
    fn rejects_pattern_with_neither_keywords_nor_regex() {
        let src = r#"---
name: x
description: x
metadata:
  ari:
    id: a.b.c
    version: "1"
    engine: ">=0.3"
    matching:
      patterns:
        - weight: 0.9
    declarative:
      response: hi
---
"#;
        assert_eq!(
            Skillfile::parse(src, None).unwrap_err(),
            ManifestError::PatternMissingMatcher
        );
    }

    #[test]
    fn rejects_empty_keywords_list() {
        let src = r#"---
name: x
description: x
metadata:
  ari:
    id: a.b.c
    version: "1"
    engine: ">=0.3"
    matching:
      patterns:
        - keywords: []
    declarative:
      response: hi
---
"#;
        assert_eq!(
            Skillfile::parse(src, None).unwrap_err(),
            ManifestError::EmptyKeywords
        );
    }

    #[test]
    fn rejects_empty_response_pick() {
        let src = r#"---
name: x
description: x
metadata:
  ari:
    id: a.b.c
    version: "1"
    engine: ">=0.3"
    matching:
      patterns:
        - keywords: [foo]
    declarative:
      response_pick: []
---
"#;
        assert_eq!(
            Skillfile::parse(src, None).unwrap_err(),
            ManifestError::EmptyResponsePick
        );
    }

    #[test]
    fn rejects_multiple_response_kinds() {
        let src = r#"---
name: x
description: x
metadata:
  ari:
    id: a.b.c
    version: "1"
    engine: ">=0.3"
    matching:
      patterns:
        - keywords: [foo]
    declarative:
      response: hi
      response_pick: ["a", "b"]
---
"#;
        assert_eq!(
            Skillfile::parse(src, None).unwrap_err(),
            ManifestError::DeclarativeResponseCardinality
        );
    }

    #[test]
    fn rejects_custom_score_on_declarative_skill() {
        let src = r#"---
name: x
description: x
metadata:
  ari:
    id: a.b.c
    version: "1"
    engine: ">=0.3"
    matching:
      patterns:
        - keywords: [foo]
      custom_score: true
    declarative:
      response: hi
---
"#;
        assert_eq!(
            Skillfile::parse(src, None).unwrap_err(),
            ManifestError::CustomScoreWithoutWasm
        );
    }

    #[test]
    fn keywords_are_lowercased_on_load() {
        let src = r#"---
name: x
description: x
metadata:
  ari:
    id: a.b.c
    version: "1"
    engine: ">=0.3"
    matching:
      patterns:
        - keywords: [Flip, COIN]
    declarative:
      response: hi
---
"#;
        let sf = Skillfile::parse(src, None).unwrap();
        let ari = sf.ari_extension.unwrap();
        match &ari.matching.as_ref().unwrap().patterns[0] {
            MatchPattern::Keywords { words, .. } => {
                assert_eq!(words, &vec!["flip".to_string(), "coin".to_string()]);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn wasm_skill_with_defaults() {
        let src = r#"---
name: weather
description: Weather lookup.
metadata:
  ari:
    id: ai.example.weather
    version: "1.0.0"
    engine: ">=0.3"
    capabilities: [http]
    matching:
      patterns:
        - keywords: [weather]
    wasm:
      module: skill.wasm
---
"#;
        let sf = Skillfile::parse(src, Some("weather")).unwrap();
        let ari = sf.ari_extension.unwrap();
        assert_eq!(ari.capabilities, vec![Capability::Http]);
        match ari.behaviour {
            Some(Behaviour::Wasm(w)) => {
                assert_eq!(w.module, "skill.wasm");
                assert_eq!(w.memory_limit_mb, 16);
            }
            _ => panic!("expected wasm behaviour"),
        }
    }

    #[test]
    fn parses_carriage_return_line_endings() {
        let src = "---\r\nname: x\r\ndescription: y\r\n---\r\nbody\r\n";
        let sf = Skillfile::parse(src, None).unwrap();
        assert_eq!(sf.name, "x");
        assert_eq!(sf.description, "y");
    }

    #[test]
    fn missing_name_is_explicit_error() {
        let src = "---\ndescription: x\n---\n";
        assert_eq!(
            Skillfile::parse(src, None).unwrap_err(),
            ManifestError::MissingName
        );
    }

    #[test]
    fn missing_description_is_explicit_error() {
        let src = "---\nname: x\n---\n";
        assert_eq!(
            Skillfile::parse(src, None).unwrap_err(),
            ManifestError::MissingDescription
        );
    }

    #[test]
    fn pattern_default_weight_is_one() {
        let src = r#"---
name: x
description: x
metadata:
  ari:
    id: a.b.c
    version: "1"
    engine: ">=0.3"
    matching:
      patterns:
        - keywords: [foo]
    declarative:
      response: hi
---
"#;
        let sf = Skillfile::parse(src, None).unwrap();
        let ari = sf.ari_extension.unwrap();
        assert_eq!(ari.matching.as_ref().unwrap().patterns[0].weight(), 1.0);
    }

    #[test]
    fn specificity_defaults_to_medium_when_omitted() {
        let src = r#"---
name: x
description: x
metadata:
  ari:
    id: a.b.c
    version: "1"
    engine: ">=0.3"
    matching:
      patterns:
        - keywords: [foo]
    declarative:
      response: hi
---
"#;
        let sf = Skillfile::parse(src, None).unwrap();
        assert_eq!(
            sf.ari_extension.unwrap().specificity,
            SpecificityLevel::Medium
        );
    }

    // --- Assistant skill parsing ---

    fn chatgpt_assistant_source() -> &'static str {
        r#"---
name: chatgpt
description: Use OpenAI's ChatGPT to answer general questions.
metadata:
  ari:
    id: dev.heyari.assistant.chatgpt
    version: "0.1.0"
    type: assistant
    author: Ari Project
    engine: ">=0.3"
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
        system_prompt: You are Ari, a helpful voice assistant. Answer in one short sentence.
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
Uses OpenAI's ChatGPT API.
"#
    }

    fn builtin_assistant_source() -> &'static str {
        r#"---
name: local-llm
description: On-device language model.
metadata:
  ari:
    id: dev.heyari.assistant.local
    version: "0.1.0"
    type: assistant
    engine: ">=0.3"
    languages: [en]
    assistant:
      provider: builtin
      privacy: local
      config:
        - key: model_tier
          label: Model
          type: select
          options:
            - value: small
              label: "Gemma 3 1B"
              download_url: "https://example.com/small.gguf"
              download_bytes: 769000000
---
Runs locally.
"#
    }

    #[test]
    fn parses_api_assistant_skill() {
        let sf = Skillfile::parse(chatgpt_assistant_source(), Some("chatgpt")).unwrap();
        let ari = sf.ari_extension.expect("ari extension present");
        assert_eq!(ari.skill_type, SkillType::Assistant);
        assert!(ari.matching.is_none());
        assert!(ari.behaviour.is_none());

        let asst = ari.assistant.expect("assistant block present");
        assert_eq!(asst.provider, AssistantProvider::Api);
        assert_eq!(asst.privacy, Privacy::Cloud);
        assert_eq!(asst.config.len(), 2);
        assert_eq!(asst.config[0].key, "api_key");
        assert!(matches!(asst.config[0].field_type, ConfigFieldType::Secret));
        assert!(asst.config[0].required);
        assert_eq!(asst.config[1].key, "model");
        assert!(matches!(asst.config[1].field_type, ConfigFieldType::Text));
        assert_eq!(asst.config[1].default.as_deref(), Some("gpt-4o-mini"));

        let api = asst.api.expect("api block present");
        assert_eq!(api.endpoint.as_deref(), Some("https://api.openai.com/v1/chat/completions"));
        assert_eq!(api.auth, AuthScheme::Bearer);
        assert_eq!(api.auth_config_key.as_deref(), Some("api_key"));
        assert_eq!(api.default_model, "gpt-4o-mini");
        assert_eq!(api.response_path, "choices[0].message.content");
        assert_eq!(api.request_format, RequestFormat::Openai);
        assert_eq!(api.max_tokens, 256);
    }

    #[test]
    fn parses_builtin_assistant_skill() {
        let sf = Skillfile::parse(builtin_assistant_source(), Some("local-llm")).unwrap();
        let ari = sf.ari_extension.expect("ari extension present");
        assert_eq!(ari.skill_type, SkillType::Assistant);

        let asst = ari.assistant.expect("assistant block present");
        assert_eq!(asst.provider, AssistantProvider::Builtin);
        assert_eq!(asst.privacy, Privacy::Local);
        assert!(asst.api.is_none());
        assert_eq!(asst.config.len(), 1);
        assert_eq!(asst.config[0].key, "model_tier");
        match &asst.config[0].field_type {
            ConfigFieldType::Select { options } => {
                assert_eq!(options.len(), 1);
                assert_eq!(options[0].value, "small");
                assert_eq!(options[0].download_url.as_deref(), Some("https://example.com/small.gguf"));
                assert_eq!(options[0].download_bytes, Some(769000000));
            }
            _ => panic!("expected select config field"),
        }
    }

    #[test]
    fn rejects_assistant_with_matching_block() {
        let src = r#"---
name: x
description: x
metadata:
  ari:
    id: a.b.c
    version: "1"
    engine: ">=0.3"
    type: assistant
    matching:
      patterns:
        - keywords: [foo]
    assistant:
      provider: builtin
      privacy: local
---
"#;
        assert_eq!(
            Skillfile::parse(src, None).unwrap_err(),
            ManifestError::AssistantHasSkillFields
        );
    }

    #[test]
    fn rejects_assistant_with_declarative_block() {
        let src = r#"---
name: x
description: x
metadata:
  ari:
    id: a.b.c
    version: "1"
    engine: ">=0.3"
    type: assistant
    declarative:
      response: hi
    assistant:
      provider: builtin
      privacy: local
---
"#;
        assert_eq!(
            Skillfile::parse(src, None).unwrap_err(),
            ManifestError::AssistantHasSkillFields
        );
    }

    #[test]
    fn rejects_regular_skill_with_assistant_block() {
        let src = r#"---
name: x
description: x
metadata:
  ari:
    id: a.b.c
    version: "1"
    engine: ">=0.3"
    matching:
      patterns:
        - keywords: [foo]
    declarative:
      response: hi
    assistant:
      provider: builtin
      privacy: local
---
"#;
        assert_eq!(
            Skillfile::parse(src, None).unwrap_err(),
            ManifestError::UnexpectedAssistantBlock
        );
    }

    #[test]
    fn rejects_api_assistant_missing_api_block() {
        let src = r#"---
name: x
description: x
metadata:
  ari:
    id: a.b.c
    version: "1"
    engine: ">=0.3"
    type: assistant
    assistant:
      provider: api
      privacy: cloud
---
"#;
        assert_eq!(
            Skillfile::parse(src, None).unwrap_err(),
            ManifestError::MissingApiBlock
        );
    }

    #[test]
    fn rejects_builtin_assistant_with_api_block() {
        let src = r#"---
name: x
description: x
metadata:
  ari:
    id: a.b.c
    version: "1"
    engine: ">=0.3"
    type: assistant
    assistant:
      provider: builtin
      privacy: local
      api:
        endpoint: https://example.com
        default_model: x
        system_prompt: x
        response_path: "x"
---
"#;
        assert_eq!(
            Skillfile::parse(src, None).unwrap_err(),
            ManifestError::UnexpectedApiBlock
        );
    }

    #[test]
    fn rejects_api_assistant_with_missing_endpoint() {
        let src = r#"---
name: x
description: x
metadata:
  ari:
    id: a.b.c
    version: "1"
    engine: ">=0.3"
    type: assistant
    assistant:
      provider: api
      privacy: cloud
      api:
        default_model: x
        system_prompt: x
        response_path: "x"
      config:
        - key: api_key
          label: Key
          type: secret
---
"#;
        assert_eq!(
            Skillfile::parse(src, None).unwrap_err(),
            ManifestError::MissingEndpoint
        );
    }

    #[test]
    fn rejects_api_assistant_with_bad_response_path() {
        let src = r#"---
name: x
description: x
metadata:
  ari:
    id: a.b.c
    version: "1"
    engine: ">=0.3"
    type: assistant
    assistant:
      provider: api
      privacy: cloud
      api:
        endpoint: https://example.com
        default_model: x
        system_prompt: x
        response_path: "choices[abc].content"
      config:
        - key: api_key
          label: Key
          type: secret
---
"#;
        let err = Skillfile::parse(src, None).unwrap_err();
        assert!(matches!(err, ManifestError::InvalidResponsePath { .. }));
    }

    #[test]
    fn rejects_api_with_auth_but_missing_config_key() {
        let src = r#"---
name: x
description: x
metadata:
  ari:
    id: a.b.c
    version: "1"
    engine: ">=0.3"
    type: assistant
    assistant:
      provider: api
      privacy: cloud
      api:
        endpoint: https://example.com
        auth: bearer
        auth_config_key: api_key
        default_model: x
        system_prompt: x
        response_path: "x"
---
"#;
        let err = Skillfile::parse(src, None).unwrap_err();
        assert!(matches!(err, ManifestError::ConfigKeyNotFound { key, field } if key == "api_key" && field == "auth_config_key"));
    }

    #[test]
    fn top_level_settings_populates_extension_and_assistant_config_mirror() {
        // New canonical location: settings live at metadata.ari.settings.
        // The parser must surface them on AriExtension.settings AND mirror
        // them onto AssistantManifest.config so the existing FFI code that
        // reads manifest.config keeps working untouched.
        let src = r#"---
name: claude
description: Talk to Claude
metadata:
  ari:
    id: dev.heyari.assistant.claude
    version: "0.1.0"
    type: assistant
    engine: ">=0.3"
    settings:
      - key: api_key
        label: API Key
        type: secret
        required: true
      - key: model
        label: Model
        type: select
        options:
          - { value: claude-sonnet-4-6, label: "Sonnet 4.6" }
          - { value: claude-opus-4-6, label: "Opus 4.6" }
    assistant:
      provider: api
      privacy: cloud
      api:
        endpoint: https://api.anthropic.com/v1/messages
        auth: bearer
        auth_config_key: api_key
        model_config_key: model
        default_model: claude-sonnet-4-6
        system_prompt: "you are claude"
        response_path: "content[0].text"
---
"#;
        let sf = Skillfile::parse(src, None).expect("must parse");
        let ari = sf.ari_extension.expect("ari extension required");
        assert_eq!(ari.settings.len(), 2, "top-level settings must be populated");
        assert_eq!(ari.settings[0].key, "api_key");
        assert_eq!(ari.settings[1].key, "model");
        let asst = ari.assistant.expect("assistant block required");
        assert_eq!(
            asst.config, ari.settings,
            "AssistantManifest.config must mirror AriExtension.settings exactly"
        );
    }

    #[test]
    fn legacy_assistant_config_still_parses_for_back_compat() {
        // Old manifests put settings under metadata.ari.assistant.config.
        // We promote them to AriExtension.settings transparently so any
        // skill that hasn't been migrated yet still works.
        let src = r#"---
name: claude
description: Talk to Claude
metadata:
  ari:
    id: dev.heyari.assistant.claude
    version: "0.1.0"
    type: assistant
    engine: ">=0.3"
    assistant:
      provider: api
      privacy: cloud
      api:
        endpoint: https://api.anthropic.com/v1/messages
        auth: bearer
        auth_config_key: api_key
        default_model: claude-sonnet-4-6
        system_prompt: "you are claude"
        response_path: "content[0].text"
      config:
        - key: api_key
          label: API Key
          type: secret
          required: true
---
"#;
        let sf = Skillfile::parse(src, None).expect("legacy manifest must parse");
        let ari = sf.ari_extension.unwrap();
        assert_eq!(ari.settings.len(), 1);
        assert_eq!(ari.settings[0].key, "api_key");
        let asst = ari.assistant.unwrap();
        assert_eq!(asst.config, ari.settings);
    }

    #[test]
    fn rejects_skill_declaring_both_settings_locations() {
        // Mixing top-level and legacy locations is almost certainly a
        // mistake mid-migration — fail loudly rather than silently
        // picking one.
        let src = r#"---
name: claude
description: x
metadata:
  ari:
    id: a.b.c
    version: "1"
    engine: ">=0.3"
    type: assistant
    settings:
      - { key: api_key, label: API Key, type: secret }
    assistant:
      provider: api
      privacy: cloud
      api:
        endpoint: https://x
        auth: bearer
        auth_config_key: api_key
        default_model: x
        system_prompt: x
        response_path: "x"
      config:
        - { key: api_key, label: API Key, type: secret }
---
"#;
        let err = Skillfile::parse(src, None).unwrap_err();
        match err {
            ManifestError::YamlParse(msg) => {
                assert!(
                    msg.contains("both") && msg.contains("settings"),
                    "error message should explain the conflict, got: {msg}"
                );
            }
            other => panic!("expected YamlParse error, got {other:?}"),
        }
    }

    #[test]
    fn top_level_settings_visible_for_regular_wasm_skills() {
        // The whole point of moving settings out of `assistant.config` is
        // that any skill type can declare them. Verify a WASM skill with
        // settings parses cleanly and exposes them on AriExtension.
        let src = r#"---
name: reminder
description: Sets reminders.
metadata:
  ari:
    id: dev.heyari.reminder
    version: "0.1.0"
    engine: ">=0.3"
    matching:
      patterns:
        - keywords: [remind]
          weight: 0.95
    examples:
      - text: "remind me to feed the cat at 5pm"
      - text: "remind me about laundry tomorrow"
      - text: "remind me to call mum"
      - text: "remind me to take out bins"
      - text: "remind me to drink water"
    settings:
      - key: default_calendar
        label: Default calendar
        type: text
    wasm:
      module: skill.wasm
      memory_limit_mb: 1
---
"#;
        let sf = Skillfile::parse(src, None).expect("must parse");
        let ari = sf.ari_extension.unwrap();
        assert_eq!(ari.skill_type, SkillType::Skill);
        assert_eq!(ari.settings.len(), 1);
        assert_eq!(ari.settings[0].key, "default_calendar");
        assert!(
            ari.assistant.is_none(),
            "non-assistant skill must not have assistant block"
        );
    }

    #[test]
    fn response_path_parse_simple_field() {
        let segs = parse_response_path("result").unwrap();
        assert_eq!(segs, vec![PathSegment::Field("result".into())]);
    }

    #[test]
    fn response_path_parse_nested() {
        let segs = parse_response_path("a.b.c").unwrap();
        assert_eq!(
            segs,
            vec![
                PathSegment::Field("a".into()),
                PathSegment::Field("b".into()),
                PathSegment::Field("c".into()),
            ]
        );
    }

    #[test]
    fn response_path_rejects_empty() {
        assert!(parse_response_path("").is_err());
    }

    #[test]
    fn response_path_rejects_double_dot() {
        assert!(parse_response_path("a..b").is_err());
    }
}
