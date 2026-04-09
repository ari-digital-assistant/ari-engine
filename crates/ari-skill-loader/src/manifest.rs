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
    pub specificity: SpecificityLevel,
    pub matching: Matching,
    pub behaviour: Behaviour,
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

        let raw_matching = raw.matching.as_ref().ok_or(ManifestError::EmptyPatterns)?;
        let matching = Matching::from_raw(raw_matching)?;

        let behaviour = match (raw.declarative.as_ref(), raw.wasm.as_ref()) {
            (Some(d), None) => Behaviour::Declarative(DeclarativeBehaviour::from_raw(d)?),
            (None, Some(w)) => Behaviour::Wasm(WasmBehaviour::from_raw(w)),
            (None, None) => {
                return Err(ManifestError::BehaviourCardinality { found: "neither" })
            }
            (Some(_), Some(_)) => {
                return Err(ManifestError::BehaviourCardinality { found: "both" })
            }
        };

        if matching.custom_score && !matches!(behaviour, Behaviour::Wasm(_)) {
            return Err(ManifestError::CustomScoreWithoutWasm);
        }

        Ok(AriExtension {
            id,
            version,
            author: raw.author.clone(),
            homepage: raw.homepage.clone(),
            engine,
            capabilities: raw.capabilities.clone().unwrap_or_default(),
            platforms: raw.platforms.clone(),
            languages: raw.languages.clone().unwrap_or_default(),
            specificity: raw.specificity.unwrap_or(SpecificityLevel::Medium),
            matching,
            behaviour,
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
    fn from_raw(raw: &RawWasm) -> Self {
        WasmBehaviour {
            module: raw.module.clone(),
            memory_limit_mb: raw.memory_limit_mb.unwrap_or(16),
        }
    }
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
    specificity: Option<SpecificityLevel>,
    matching: Option<RawMatching>,
    declarative: Option<RawDeclarative>,
    wasm: Option<RawWasm>,
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
        assert_eq!(ari.matching.patterns.len(), 2);
        assert!(!ari.matching.custom_score);

        match &ari.matching.patterns[0] {
            MatchPattern::Keywords { words, weight } => {
                assert_eq!(words, &vec!["flip".to_string(), "coin".to_string()]);
                assert_eq!(*weight, 0.95);
            }
            _ => panic!("expected keywords pattern"),
        }

        match ari.behaviour {
            Behaviour::Declarative(DeclarativeBehaviour {
                response: ResponseSpec::Pick(picks),
                action: None,
            }) => {
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
        match &ari.matching.patterns[0] {
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
            Behaviour::Wasm(w) => {
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
        assert_eq!(ari.matching.patterns[0].weight(), 1.0);
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
}
