//! The bundled wake word models, and the three sensitivities a user picks
//! between on each.

use crate::ladder::TunedLadder;

/// The threshold the engine compares against, and how many recent
/// probabilities it averages first. Together they are one point on a model's
/// recall / false wake curve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OperatingPoint {
    pub probability_cutoff: f32,
    pub sliding_window_size: i32,
}

impl OperatingPoint {
    pub const fn new(probability_cutoff: f32, sliding_window_size: i32) -> Self {
        Self {
            probability_cutoff,
            sliding_window_size,
        }
    }
}

/// Which engine runs a model: microWakeWord or openWakeWord.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WakeEngine {
    MicroWakeWord,
    OpenWakeWord,
}

/// User-adjustable wake word sensitivity: `High` fires more readily, `Low` is
/// strictest. Follows the user's intuition and matches how Alexa and Google
/// Home label the same setting. Easy to get backwards: don't.
///
/// The numbers live on [`WakeWordModel`], not here. A cutoff means nothing on
/// its own (0.5 is the sweet spot for one model and fires on almost anything
/// for another), so each model carries its own three points and this enum is
/// only the name of which one to use. Holding absolute values here is what
/// made a retrained model silently run on its predecessor's ladder, where it
/// scored worse than the model it replaced.
///
/// See `docs/superpowers/specs/2026-09-10-wake-word-retrain-design.md` §7.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WakeSensitivity {
    High,
    Medium,
    Low,
}

impl WakeSensitivity {
    pub const DEFAULT: Self = Self::Medium;

    /// Loosest first. The order the ladder is presented and reasoned about in.
    pub const LOOSEST_FIRST: [Self; 3] = [Self::High, Self::Medium, Self::Low];

    /// The level a frontend stored by name: `"HIGH"`, `"MEDIUM"` or `"LOW"`,
    /// the names Kotlin gives these variants. Anything else is the default,
    /// never an error.
    pub fn from_name(name: Option<&str>) -> Self {
        match name {
            Some("HIGH") => Self::High,
            Some("MEDIUM") => Self::Medium,
            Some("LOW") => Self::Low,
            _ => Self::DEFAULT,
        }
    }

    /// The name [`from_name`](Self::from_name) reads, and the wake log writes.
    pub fn name(self) -> &'static str {
        match self {
            Self::High => "HIGH",
            Self::Medium => "MEDIUM",
            Self::Low => "LOW",
        }
    }
}

/// A wake word model bundled with the app.
///
/// Each model carries its own three operating points rather than sharing one
/// ladder, because a cutoff means nothing on its own: it is a point on one
/// model's curve and says nothing about another's.
///
/// A cutoff also means nothing without the noise it has to clear. Quote every
/// one of these against the loudest window mean a real room ever reached, and
/// pick it for the gap, not for how it scores on the recording you happen to
/// have. A threshold that clears twenty minutes of kitchen by 0.002 is fitted
/// to that kitchen's noise and will fire in the twenty-first minute.
///
/// The numbers are measured, not guessed. See `ari-tools/wakeword/baseline/`
/// and `docs/superpowers/specs/2026-09-10-wake-word-retrain-design.md` §7.
#[derive(Debug, Clone, PartialEq)]
pub struct WakeWordModel {
    pub id: String,
    pub display_name: String,
    pub asset_filename: String,
    pub feature_step_size_ms: i32,
    pub high: OperatingPoint,
    pub medium: OperatingPoint,
    pub low: OperatingPoint,
    pub engine: WakeEngine,
}

impl WakeWordModel {
    pub fn operating_point(&self, sensitivity: WakeSensitivity) -> OperatingPoint {
        match sensitivity {
            WakeSensitivity::High => self.high,
            WakeSensitivity::Medium => self.medium,
            WakeSensitivity::Low => self.low,
        }
    }

    /// This model running on a ladder measured on the user's own phone.
    ///
    /// A ladder for another model is ignored rather than applied: the numbers
    /// are points on one model's curve and mean nothing on another's, which is
    /// the mistake this type's documentation exists to record. Someone
    /// switching wake word gets the built-in ladder back until they tune again.
    pub fn with_ladder(&self, ladder: Option<&TunedLadder>) -> WakeWordModel {
        match ladder {
            Some(ladder) if ladder.model_id == self.id => WakeWordModel {
                high: OperatingPoint::new(ladder.high, ladder.sliding_window_size),
                medium: OperatingPoint::new(ladder.medium, ladder.sliding_window_size),
                low: OperatingPoint::new(ladder.low, ladder.sliding_window_size),
                ..self.clone()
            },
            _ => self.clone(),
        }
    }
}

/// An openWakeWord model (`ari-tools/wakeword/models/openwakeword-2026-10-28/`,
/// `rung2_half`), trained by Keith on openwakeword.com, whose models are for
/// personal and non-commercial use unless a commercial licence is bought. Its
/// classifier sits on a speech embedding pretrained on real speech, which is
/// the likely reason it hears this household where every microWakeWord
/// retrain went deaf to some of it.
///
/// It replaced the microWakeWord `hey_ari`, called the old model below.
///
/// It fires on a single 80 ms score, so the window is 1. Over the twenty
/// minutes of household ambient the highest score is 0.350. MEDIUM is 0.55,
/// the cutoff its offline exam ran at: 82 of 92 marked takes, none of the 34
/// held-out household false wakes, 0.5 false fires an hour on English speech
/// against the old model's 52, none on hum. HIGH is 0.50, the loosest swept
/// cutoff that clears the ambient by the exam's minimum of 0.10, and LOW is
/// 0.65, the old model's smallest gap of +0.30. Recall barely moves across
/// them: 83 / 82 / 81 of the 92 takes (Gail 45 / 44 / 43 of 46; Joe 10 of
/// 15, Keith 16 of 16 and Maryanne 12 of 15 at every level), against the old
/// model's 64 / 62 / 58.
///
/// The gaps, +0.15 / +0.20 / +0.30, are narrower than the old model's, and
/// +0.10 has false-fired in this house before. A week live in the house (31
/// Oct to 7 Nov) answered for +0.20: none of its false wakes was ever shown,
/// and 64 reached the transcriber, against 82 for the old model running
/// beside it.
fn hey_ari_open_wake_word() -> WakeWordModel {
    WakeWordModel {
        id: "hey_ari_oww".into(),
        display_name: "Hey Ari".into(),
        asset_filename: "hey_ari_oww.onnx".into(),
        // openWakeWord's melspectrogram steps 10 ms; only microWakeWord reads this.
        feature_step_size_ms: 10,
        high: OperatingPoint::new(0.50, 1),
        medium: OperatingPoint::new(0.55, 1),
        low: OperatingPoint::new(0.65, 1),
        engine: WakeEngine::OpenWakeWord,
    }
}

/// Unmeasured, so deliberately left on the ladder the app shipped before
/// operating points became per-model. Neither has been retrained and neither
/// has a household recording to score against; changing their behaviour on
/// the strength of `hey_ari`'s numbers would be a guess dressed as a fix.
fn unmeasured(id: &str, display_name: &str) -> WakeWordModel {
    WakeWordModel {
        id: id.into(),
        display_name: display_name.into(),
        asset_filename: format!("{id}.tflite"),
        feature_step_size_ms: 10,
        high: OperatingPoint::new(0.90, 5),
        medium: OperatingPoint::new(0.95, 5),
        low: OperatingPoint::new(0.985, 10),
        engine: WakeEngine::MicroWakeWord,
    }
}

/// Every bundled model, the default first.
pub fn all() -> Vec<WakeWordModel> {
    vec![
        hey_ari_open_wake_word(),
        unmeasured("ok_ari", "OK Ari"),
        unmeasured("hey_jarvis", "Hey Jarvis"),
    ]
}

pub fn default_model() -> WakeWordModel {
    hey_ari_open_wake_word()
}

/// The model with this id, or the default for an unknown or missing one.
pub fn by_id(id: Option<&str>) -> WakeWordModel {
    all()
        .into_iter()
        .find(|m| Some(m.id.as_str()) == id)
        .unwrap_or_else(default_model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_model_gets_stricter_from_high_through_medium_to_low() {
        // The engine fires when sum(recent probabilities) > cutoff * window, so
        // raising either value makes it stricter. Labelled backwards the setting
        // does the opposite of what it says and nothing else in the app notices.
        // Checked per model now that each carries its own numbers: one model
        // ordered correctly says nothing about the next.
        for model in all() {
            let ladder: Vec<_> = WakeSensitivity::LOOSEST_FIRST
                .iter()
                .map(|s| model.operating_point(*s))
                .collect();
            for pair in ladder.windows(2) {
                let (looser, stricter) = (pair[0], pair[1]);
                assert!(
                    looser.probability_cutoff <= stricter.probability_cutoff
                        && looser.sliding_window_size <= stricter.sliding_window_size,
                    "{} is not monotonic: {looser:?} then {stricter:?}",
                    model.id,
                );
            }
        }
    }

    #[test]
    fn every_level_survives_int8_quantisation_as_a_distinct_threshold() {
        // MicroWakeWordEngine compares against (uint8)(cutoff * 255), so cutoffs
        // that round to the same byte are the same setting wearing two names.
        // There is no room at all near the ceiling: anything above 0.9981 is 254
        // until it becomes 255 and can never be exceeded.
        for model in all() {
            let quantised: Vec<(i32, i32)> = WakeSensitivity::LOOSEST_FIRST
                .iter()
                .map(|s| {
                    let p = model.operating_point(*s);
                    ((p.probability_cutoff * 255.0) as i32, p.sliding_window_size)
                })
                .collect();
            let distinct: std::collections::HashSet<_> = quantised.iter().collect();
            assert_eq!(
                quantised.len(),
                distinct.len(),
                "{} has duplicate levels",
                model.id
            );
            assert!(
                quantised.iter().all(|q| q.0 < 255),
                "{} has a cutoff that quantises to 255 and can never fire",
                model.id,
            );
        }
    }

    #[test]
    fn the_default_runs_at_the_operating_points_its_own_recordings_were_scored_at() {
        // Not a restatement of the constants: these three numbers are what the
        // cutoff sweep over the 92 marked takes measured, and a change here
        // without a new measurement is the exact mistake that made a retrained
        // model run on its predecessor's ladder.
        let model = default_model();

        assert_eq!(model.id, "hey_ari_oww");
        assert_eq!(
            model.operating_point(WakeSensitivity::High),
            OperatingPoint::new(0.50, 1)
        );
        assert_eq!(
            model.operating_point(WakeSensitivity::Medium),
            OperatingPoint::new(0.55, 1)
        );
        assert_eq!(
            model.operating_point(WakeSensitivity::Low),
            OperatingPoint::new(0.65, 1)
        );
    }

    #[test]
    fn every_default_cutoff_clears_the_loudest_household_ambient_we_have_measured() {
        // Twenty minutes of household ambient peak at a single score of 0.350
        // for this model. A cutoff at or under that fires on conversation by
        // construction, and no amount of recall makes up for it. The floor can
        // only rise as more rooms are recorded; if it passes a cutoff, the
        // cutoff moves.
        let ambient_peak = 0.350_f32;
        // HIGH's gap, the smallest the ladder has. The exam's own rule was 0.10.
        let minimum_margin = 0.15_f32;
        let model = default_model();

        for level in WakeSensitivity::LOOSEST_FIRST {
            let point = model.operating_point(level);
            assert!(
                point.probability_cutoff >= ambient_peak + minimum_margin - 1e-6,
                "{} {level:?} sits at {}, too close to household ambient ({ambient_peak})",
                model.id,
                point.probability_cutoff,
            );
        }
    }

    #[test]
    fn the_unmeasured_models_keep_the_ladder_the_app_shipped_before_this_change() {
        // ok_ari and hey_jarvis have no household recordings to score against.
        // Moving them on hey_ari's evidence would be a guess dressed as a fix.
        for id in ["ok_ari", "hey_jarvis"] {
            let model = by_id(Some(id));
            assert_eq!(model.id, id);
            assert_eq!(model.engine, WakeEngine::MicroWakeWord);
            assert_eq!(model.asset_filename, format!("{id}.tflite"));
            assert_eq!(
                model.operating_point(WakeSensitivity::High),
                OperatingPoint::new(0.90, 5)
            );
            assert_eq!(
                model.operating_point(WakeSensitivity::Medium),
                OperatingPoint::new(0.95, 5)
            );
            assert_eq!(
                model.operating_point(WakeSensitivity::Low),
                OperatingPoint::new(0.985, 10)
            );
        }
    }

    #[test]
    fn the_defaults_points_would_be_nonsense_on_any_other_model() {
        // The reason operating points are per model. The default's strictest
        // setting is far below every other model's loosest, so a shared ladder
        // cannot serve both.
        let default = default_model();
        let jarvis = by_id(Some("hey_jarvis"));

        assert!(
            default
                .operating_point(WakeSensitivity::Low)
                .probability_cutoff
                < jarvis
                    .operating_point(WakeSensitivity::High)
                    .probability_cutoff
        );
    }

    #[test]
    fn the_registry_lists_every_bundled_model_default_first() {
        let ids: Vec<_> = all()
            .into_iter()
            .map(|m| (m.id, m.display_name, m.asset_filename))
            .collect();
        assert_eq!(
            ids,
            vec![
                (
                    "hey_ari_oww".into(),
                    "Hey Ari".into(),
                    "hey_ari_oww.onnx".into()
                ),
                ("ok_ari".into(), "OK Ari".into(), "ok_ari.tflite".into()),
                (
                    "hey_jarvis".into(),
                    "Hey Jarvis".into(),
                    "hey_jarvis.tflite".into()
                ),
            ],
        );
        assert_eq!(default_model().engine, WakeEngine::OpenWakeWord);
    }

    #[test]
    fn an_unknown_or_missing_id_is_the_default() {
        assert_eq!(by_id(Some("hey_ari")), default_model());
        assert_eq!(by_id(None), default_model());
    }

    #[test]
    fn the_default_is_the_middle_of_the_ladder() {
        assert_eq!(WakeSensitivity::DEFAULT, WakeSensitivity::Medium);
    }

    #[test]
    fn an_unknown_stored_name_falls_back_to_the_default_rather_than_throwing() {
        assert_eq!(
            WakeSensitivity::from_name(Some("ENTHUSIASTIC")),
            WakeSensitivity::DEFAULT
        );
        assert_eq!(WakeSensitivity::from_name(None), WakeSensitivity::DEFAULT);
        assert_eq!(
            WakeSensitivity::from_name(Some("LOW")),
            WakeSensitivity::Low
        );
        assert_eq!(
            WakeSensitivity::from_name(Some("HIGH")),
            WakeSensitivity::High
        );
        assert_eq!(
            WakeSensitivity::from_name(Some("MEDIUM")),
            WakeSensitivity::Medium
        );
    }

    #[test]
    fn the_ladder_is_declared_strictest_last_which_is_what_stepping_down_relies_on() {
        // for_level walks this list rather than comparing cutoffs, because the
        // numbers behind each rung differ per model and cannot be compared
        // across one. Inverting this list silently inverts the remedy.
        assert_eq!(
            WakeSensitivity::LOOSEST_FIRST,
            [
                WakeSensitivity::High,
                WakeSensitivity::Medium,
                WakeSensitivity::Low
            ],
        );
    }
}
