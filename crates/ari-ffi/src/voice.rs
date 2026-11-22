//! UniFFI wrapper around [`ari_voice`]: the wake phrase rules, the wake word
//! models and their ladders.
//!
//! A thin shim, like `skill_registry.rs`. The rules live in `ari-voice`; this
//! file mirrors its types as UniFFI records and enums and converts between
//! them.

use ari_voice::ladder::{self, FalseWakeRemedy, LadderFit, TunedLadder, TunedSession};
use ari_voice::wake_model::{self, OperatingPoint, WakeEngine, WakeSensitivity, WakeWordModel};
use ari_voice::wake_phrase::{self, EarlyNameCheck, StreamingPartial, WakeMatch};

#[derive(Debug, Clone, PartialEq, Eq, uniffi::Record)]
pub struct FfiWakeMatch {
    pub text: String,
    pub name_matched: bool,
}

impl From<WakeMatch> for FfiWakeMatch {
    fn from(m: WakeMatch) -> Self {
        Self {
            text: m.text,
            name_matched: m.name_matched,
        }
    }
}

impl From<FfiWakeMatch> for WakeMatch {
    fn from(m: FfiWakeMatch) -> Self {
        Self {
            text: m.text,
            name_matched: m.name_matched,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum FfiEarlyNameCheck {
    Pass,
    Miss,
    None,
}

impl From<EarlyNameCheck> for FfiEarlyNameCheck {
    fn from(c: EarlyNameCheck) -> Self {
        match c {
            EarlyNameCheck::Pass => Self::Pass,
            EarlyNameCheck::Miss => Self::Miss,
            EarlyNameCheck::None => Self::None,
        }
    }
}

impl From<FfiEarlyNameCheck> for EarlyNameCheck {
    fn from(c: FfiEarlyNameCheck) -> Self {
        match c {
            FfiEarlyNameCheck::Pass => Self::Pass,
            FfiEarlyNameCheck::Miss => Self::Miss,
            FfiEarlyNameCheck::None => Self::None,
        }
    }
}

#[uniffi::export]
impl FfiEarlyNameCheck {
    /// The wake log's name for this result.
    pub fn slug(&self) -> String {
        EarlyNameCheck::from(*self).slug().to_string()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, uniffi::Record)]
pub struct FfiStreamingPartial {
    pub text: String,
    pub name_matched: Option<bool>,
}

impl From<StreamingPartial> for FfiStreamingPartial {
    fn from(p: StreamingPartial) -> Self {
        Self {
            text: p.text,
            name_matched: p.name_matched,
        }
    }
}

/// See [`wake_phrase::match_wake_phrase`].
#[uniffi::export]
pub fn match_wake_phrase(text: String, locale: String) -> FfiWakeMatch {
    wake_phrase::match_wake_phrase(&text, &locale).into()
}

/// See [`wake_phrase::strip_wake_phrase`].
#[uniffi::export]
pub fn strip_wake_phrase(text: String, locale: String) -> String {
    wake_phrase::strip_wake_phrase(&text, &locale)
}

/// See [`wake_phrase::wake_verdict`].
#[uniffi::export]
pub fn wake_verdict(wake_match: FfiWakeMatch, locale: String) -> Option<bool> {
    wake_phrase::wake_verdict(&wake_match.into(), &locale)
}

/// See [`wake_phrase::early_name_check`].
#[uniffi::export]
pub fn early_name_check(transcript: String, locale: String) -> FfiEarlyNameCheck {
    wake_phrase::early_name_check(&transcript, &locale).into()
}

/// See [`wake_phrase::streaming_partial`].
#[uniffi::export]
pub fn streaming_partial(wake_match: FfiWakeMatch, locale: String) -> Option<FfiStreamingPartial> {
    wake_phrase::streaming_partial(&wake_match.into(), &locale).map(Into::into)
}

#[derive(Debug, Clone, Copy, PartialEq, uniffi::Record)]
pub struct FfiOperatingPoint {
    pub probability_cutoff: f32,
    pub sliding_window_size: i32,
}

impl From<OperatingPoint> for FfiOperatingPoint {
    fn from(p: OperatingPoint) -> Self {
        Self {
            probability_cutoff: p.probability_cutoff,
            sliding_window_size: p.sliding_window_size,
        }
    }
}

impl From<FfiOperatingPoint> for OperatingPoint {
    fn from(p: FfiOperatingPoint) -> Self {
        Self {
            probability_cutoff: p.probability_cutoff,
            sliding_window_size: p.sliding_window_size,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum FfiWakeEngine {
    MicroWakeWord,
    OpenWakeWord,
}

impl From<WakeEngine> for FfiWakeEngine {
    fn from(e: WakeEngine) -> Self {
        match e {
            WakeEngine::MicroWakeWord => Self::MicroWakeWord,
            WakeEngine::OpenWakeWord => Self::OpenWakeWord,
        }
    }
}

impl From<FfiWakeEngine> for WakeEngine {
    fn from(e: FfiWakeEngine) -> Self {
        match e {
            FfiWakeEngine::MicroWakeWord => Self::MicroWakeWord,
            FfiWakeEngine::OpenWakeWord => Self::OpenWakeWord,
        }
    }
}

/// Kotlin names these `HIGH`, `MEDIUM` and `LOW`, which are the names
/// [`wake_sensitivity_from_name`] reads back.
#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum FfiWakeSensitivity {
    High,
    Medium,
    Low,
}

impl From<WakeSensitivity> for FfiWakeSensitivity {
    fn from(s: WakeSensitivity) -> Self {
        match s {
            WakeSensitivity::High => Self::High,
            WakeSensitivity::Medium => Self::Medium,
            WakeSensitivity::Low => Self::Low,
        }
    }
}

impl From<FfiWakeSensitivity> for WakeSensitivity {
    fn from(s: FfiWakeSensitivity) -> Self {
        match s {
            FfiWakeSensitivity::High => Self::High,
            FfiWakeSensitivity::Medium => Self::Medium,
            FfiWakeSensitivity::Low => Self::Low,
        }
    }
}

/// See [`WakeSensitivity::DEFAULT`].
#[uniffi::export]
pub fn wake_sensitivity_default() -> FfiWakeSensitivity {
    WakeSensitivity::DEFAULT.into()
}

/// See [`WakeSensitivity::from_name`].
#[uniffi::export]
pub fn wake_sensitivity_from_name(name: Option<String>) -> FfiWakeSensitivity {
    WakeSensitivity::from_name(name.as_deref()).into()
}

#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct FfiWakeWordModel {
    pub id: String,
    pub display_name: String,
    pub asset_filename: String,
    pub feature_step_size_ms: i32,
    pub high: FfiOperatingPoint,
    pub medium: FfiOperatingPoint,
    pub low: FfiOperatingPoint,
    pub engine: FfiWakeEngine,
}

impl From<WakeWordModel> for FfiWakeWordModel {
    fn from(m: WakeWordModel) -> Self {
        Self {
            id: m.id,
            display_name: m.display_name,
            asset_filename: m.asset_filename,
            feature_step_size_ms: m.feature_step_size_ms,
            high: m.high.into(),
            medium: m.medium.into(),
            low: m.low.into(),
            engine: m.engine.into(),
        }
    }
}

impl From<FfiWakeWordModel> for WakeWordModel {
    fn from(m: FfiWakeWordModel) -> Self {
        Self {
            id: m.id,
            display_name: m.display_name,
            asset_filename: m.asset_filename,
            feature_step_size_ms: m.feature_step_size_ms,
            high: m.high.into(),
            medium: m.medium.into(),
            low: m.low.into(),
            engine: m.engine.into(),
        }
    }
}

#[uniffi::export]
impl FfiWakeWordModel {
    /// See [`WakeWordModel::operating_point`].
    pub fn operating_point(&self, sensitivity: FfiWakeSensitivity) -> FfiOperatingPoint {
        WakeWordModel::from(self.clone())
            .operating_point(sensitivity.into())
            .into()
    }

    /// See [`WakeWordModel::with_ladder`].
    pub fn with_ladder(&self, ladder: Option<FfiTunedLadder>) -> FfiWakeWordModel {
        let ladder = ladder.map(TunedLadder::from);
        WakeWordModel::from(self.clone())
            .with_ladder(ladder.as_ref())
            .into()
    }
}

/// Every bundled wake word model, the default first.
#[uniffi::export]
pub fn wake_word_models() -> Vec<FfiWakeWordModel> {
    wake_model::all().into_iter().map(Into::into).collect()
}

#[uniffi::export]
pub fn default_wake_word_model() -> FfiWakeWordModel {
    wake_model::default_model().into()
}

/// The model with this id, or the default for an unknown or missing one.
#[uniffi::export]
pub fn wake_word_model_by_id(id: Option<String>) -> FfiWakeWordModel {
    wake_model::by_id(id.as_deref()).into()
}

#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct FfiTunedLadder {
    pub model_id: String,
    pub high: f32,
    pub medium: f32,
    pub low: f32,
    pub sliding_window_size: i32,
}

impl From<TunedLadder> for FfiTunedLadder {
    fn from(l: TunedLadder) -> Self {
        Self {
            model_id: l.model_id,
            high: l.high,
            medium: l.medium,
            low: l.low,
            sliding_window_size: l.sliding_window_size,
        }
    }
}

impl From<FfiTunedLadder> for TunedLadder {
    fn from(l: FfiTunedLadder) -> Self {
        Self {
            model_id: l.model_id,
            high: l.high,
            medium: l.medium,
            low: l.low,
            sliding_window_size: l.sliding_window_size,
        }
    }
}

#[uniffi::export]
impl FfiTunedLadder {
    /// The stored form, read back by [`tuned_ladder_parse`].
    pub fn format(&self) -> String {
        TunedLadder::from(self.clone()).format()
    }
}

/// See [`TunedLadder::parse`].
#[uniffi::export]
pub fn tuned_ladder_parse(stored: Option<String>) -> Option<FfiTunedLadder> {
    TunedLadder::parse(stored.as_deref()).map(Into::into)
}

/// The range a tuned cutoff may take.
#[derive(Debug, Clone, Copy, PartialEq, uniffi::Record)]
pub struct FfiCutoffRange {
    pub min: f32,
    pub max: f32,
}

/// [`TunedLadder::MIN_CUTOFF`] and [`TunedLadder::MAX_CUTOFF`], the bounds of
/// the tuner's search. Interim: the search moves into the engine in step 3 of
/// the audio-in plan, and this goes with it.
#[uniffi::export]
pub fn tuned_ladder_cutoff_range() -> FfiCutoffRange {
    FfiCutoffRange {
        min: TunedLadder::MIN_CUTOFF,
        max: TunedLadder::MAX_CUTOFF,
    }
}

/// See [`ladder::LADDER_REVIEW_DELAY_HOURS`].
#[uniffi::export]
pub fn ladder_review_delay_hours() -> i64 {
    ladder::LADDER_REVIEW_DELAY_HOURS
}

#[derive(Debug, Clone, PartialEq, uniffi::Enum)]
pub enum FfiFalseWakeRemedy {
    StepDown { to: FfiWakeSensitivity },
    Tighten { ladder: FfiTunedLadder },
    Exhausted,
}

impl From<FalseWakeRemedy> for FfiFalseWakeRemedy {
    fn from(r: FalseWakeRemedy) -> Self {
        match r {
            FalseWakeRemedy::StepDown { to } => Self::StepDown { to: to.into() },
            FalseWakeRemedy::Tighten { ladder } => Self::Tighten {
                ladder: ladder.into(),
            },
            FalseWakeRemedy::Exhausted => Self::Exhausted,
        }
    }
}

/// See [`FalseWakeRemedy::for_level`].
#[uniffi::export]
pub fn false_wake_remedy(level: FfiWakeSensitivity, model: FfiWakeWordModel) -> FfiFalseWakeRemedy {
    FalseWakeRemedy::for_level(level.into(), &model.into()).into()
}

#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct FfiTunedSession {
    pub id: i32,
    pub phrase_scores: Vec<f32>,
    pub decoy_scores: Vec<f32>,
}

impl From<FfiTunedSession> for TunedSession {
    fn from(s: FfiTunedSession) -> Self {
        Self {
            id: s.id,
            phrase_scores: s.phrase_scores,
            decoy_scores: s.decoy_scores,
        }
    }
}

#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct FfiLadderFit {
    pub ladder: FfiTunedLadder,
    pub phrases_heard: i32,
    pub phrases_total: i32,
    pub worst_decoy: f32,
    pub overlapping: bool,
}

impl From<LadderFit> for FfiLadderFit {
    fn from(f: LadderFit) -> Self {
        Self {
            ladder: f.ladder.into(),
            phrases_heard: f.phrases_heard,
            phrases_total: f.phrases_total,
            worst_decoy: f.worst_decoy,
            overlapping: f.overlapping,
        }
    }
}

/// See [`ladder::fit`].
#[uniffi::export]
pub fn fit_tuned_ladder(
    sessions: Vec<FfiTunedSession>,
    model: FfiWakeWordModel,
) -> Option<FfiLadderFit> {
    let sessions: Vec<TunedSession> = sessions.into_iter().map(Into::into).collect();
    ladder::fit(&sessions, &model.into()).map(Into::into)
}
