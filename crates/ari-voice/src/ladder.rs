//! Sensitivity ladders measured on the user's own phone, the remedy Ari offers
//! for false wakes, and the arithmetic that turns a tuning session into a
//! ladder.

use crate::wake_model::{WakeSensitivity, WakeWordModel};

/// A ladder measured on this phone, in this household, which stands in for the
/// one the model ships with.
///
/// The built-in ladder is the best answer available before anybody has used the
/// app: it comes from one developer's recordings, four voices and twenty
/// minutes of one house. It cannot know what this phone's microphone does, what
/// this room sounds like, or which words the people here say that happen to
/// resemble the wake phrase. Everything here exists to let the phone find that
/// out and write it down.
///
/// Two rules keep this from becoming the fault it was built to fix:
///
/// A stored ladder may only ever be STRICTER than the built-in one
/// ([`TunedLadder::at_least`]). A tuning session is thirty seconds in whatever
/// room the user is standing in; the built-in ladder was measured against
/// twenty minutes of a real kitchen. Letting the short measurement loosen the
/// long one is exactly how a ladder ended up at 0.45 and false-fired within two
/// days. Tightening on local evidence is sound; loosening on it is not.
///
/// The rungs stay [`TunedLadder::STEP`] apart and inside
/// [`TunedLadder::MIN_CUTOFF`]..[`TunedLadder::MAX_CUTOFF`], so the three
/// levels a user can pick never collapse into each other. Above roughly 0.998
/// the int8 quantisation the engine does makes every cutoff the same number,
/// and a picker with three settings that behave identically is a lie told
/// three ways.
#[derive(Debug, Clone, PartialEq)]
pub struct TunedLadder {
    pub model_id: String,
    pub high: f32,
    pub medium: f32,
    pub low: f32,
    pub sliding_window_size: i32,
}

impl TunedLadder {
    /// Below this a cutoff is in the range ordinary conversation reaches.
    pub const MIN_CUTOFF: f32 = 0.30;

    /// Above this the levels stop being distinct once quantised to int8.
    pub const MAX_CUTOFF: f32 = 0.95;

    /// Gap between rungs. Wide enough that picking a different one does something.
    pub const STEP: f32 = 0.10;

    /// The narrowest a rung may sit from its neighbour once the ladder is
    /// pressed against the ceiling. About five quantisation levels, so the
    /// three settings stay distinct after the engine rounds them to int8.
    pub const MIN_GAP: f32 = 0.02;

    /// The stored form, read back by [`TunedLadder::parse`].
    pub fn format(&self) -> String {
        format!(
            "{}|{}|{}|{}|{}",
            self.model_id, self.high, self.medium, self.low, self.sliding_window_size
        )
    }

    /// A ladder centred on `medium`, spaced by [`TunedLadder::STEP`] and
    /// squeezed rather than shifted where it meets a limit.
    ///
    /// MEDIUM is the rung the tuner solves for and the rung the app defaults
    /// to, so it is the one a measurement is allowed to move; HIGH and LOW
    /// follow it. Near the ceiling they compress toward it instead of dragging
    /// MEDIUM back down, because a ladder asked for a strict middle must
    /// deliver a strict middle: quietly loosening it to keep the rungs evenly
    /// spaced would answer a question nobody asked.
    pub fn centred_on(model_id: &str, medium: f32, sliding_window_size: i32) -> TunedLadder {
        let centre = medium.clamp(
            Self::MIN_CUTOFF + Self::MIN_GAP,
            Self::MAX_CUTOFF - Self::MIN_GAP,
        );
        TunedLadder {
            model_id: model_id.to_string(),
            high: (centre - Self::STEP).clamp(Self::MIN_CUTOFF, centre - Self::MIN_GAP),
            medium: centre,
            low: (centre + Self::STEP).clamp(centre + Self::MIN_GAP, Self::MAX_CUTOFF),
            sliding_window_size,
        }
    }

    /// Parses [`TunedLadder::format`]. `None` on anything it does not recognise.
    pub fn parse(stored: Option<&str>) -> Option<TunedLadder> {
        let parts: Vec<&str> = stored?.split('|').collect();
        if parts.len() != 5 {
            return None;
        }
        let high: f32 = parts[1].parse().ok()?;
        let medium: f32 = parts[2].parse().ok()?;
        let low: f32 = parts[3].parse().ok()?;
        let window: i32 = parts[4].parse().ok()?;
        // Order is the one invariant the rest of the app relies on without
        // checking: WakeSensitivity's whole meaning is that LOW is the
        // strictest. A stored ladder that breaks it is corrupt, not old.
        if high >= medium || medium >= low {
            return None;
        }
        if high < Self::MIN_CUTOFF || low > Self::MAX_CUTOFF || window <= 0 {
            return None;
        }
        Some(TunedLadder {
            model_id: parts[0].to_string(),
            high,
            medium,
            low,
            sliding_window_size: window,
        })
    }

    /// This ladder, or the model's built-in one wherever that is stricter.
    ///
    /// Applied rung by rung rather than to the ladder as a whole: a local
    /// measurement that tightens HIGH but would loosen LOW is half right, and
    /// there is no reason to throw away the half that helps.
    pub fn at_least(&self, model: &WakeWordModel) -> TunedLadder {
        TunedLadder {
            high: self.high.max(model.high.probability_cutoff),
            medium: self.medium.max(model.medium.probability_cutoff),
            low: self.low.max(model.low.probability_cutoff),
            ..self.clone()
        }
    }
}

/// How long to live with a self-tightening before asking about it.
///
/// Long enough for everybody in a household to have tried to use the thing at
/// least once (a person who only talks to Ari at breakfast needs two mornings
/// to notice) and short enough that they have not yet given up and stopped
/// trying, which is the failure this question is racing.
///
/// Deliberately driven by the clock and not by wake activity. Counting
/// successful wakes as the signal to ask would be measuring the people the
/// tightening did not affect: if it went deaf to one member of the household,
/// the wakes still arriving belong to everybody else and the counter fills up
/// looking healthy.
pub const LADDER_REVIEW_DELAY_HOURS: i64 = 48;

/// What Ari can offer somebody whose wake word keeps firing at nothing.
///
/// Always exactly one suggestion beside doing nothing. Handing over three
/// options and a probability curve moves the decision to the person least
/// equipped to make it: they reported a symptom, not a preference about
/// thresholds.
#[derive(Debug, Clone, PartialEq)]
pub enum FalseWakeRemedy {
    /// Move down the ladder the app already offers. Costs nothing to undo and
    /// the user can see what changed, so it comes before touching the numbers.
    StepDown { to: WakeSensitivity },

    /// Already on the strictest rung, so the rung itself has to move: rebuild
    /// the ladder around a cutoff [`FalseWakeRemedy::TIGHTEN_STEP`] above the
    /// one in force and land the user on its middle.
    Tighten { ladder: TunedLadder },

    /// Nothing useful left. The ladder is already against the ceiling, where
    /// further cutoffs stop being distinguishable once quantised: saying so is
    /// the honest answer, and it means the model is the problem, not the number.
    Exhausted,
}

impl FalseWakeRemedy {
    /// How much stricter a [`FalseWakeRemedy::Tighten`] makes the wake word.
    ///
    /// Small on purpose. The obvious move (jump the whole ladder so its
    /// loosest rung clears the current strictest) is roughly a 0.25 step, and
    /// on the household recordings that costs the quietest speaker a quarter of
    /// the wakes she has left. One nudge that might need repeating beats one
    /// leap that cannot be felt until somebody has given up on the thing.
    pub const TIGHTEN_STEP: f32 = 0.05;

    /// The single remedy for somebody currently on `level`.
    ///
    /// `model` must already carry whatever ladder is actually in force
    /// ([`WakeWordModel::with_ladder`]), because the step is measured from the
    /// cutoff the user is running, not from the one the app shipped with.
    pub fn for_level(level: WakeSensitivity, model: &WakeWordModel) -> FalseWakeRemedy {
        if level != WakeSensitivity::Low {
            // LOOSEST_FIRST is HIGH, MEDIUM, LOW, so the next one along is the
            // next one stricter. Going through the declared order rather than a
            // match means a fourth rung would not silently skip itself here.
            let order = WakeSensitivity::LOOSEST_FIRST;
            let next = order
                .iter()
                .position(|s| *s == level)
                .and_then(|i| order.get(i + 1));
            if let Some(next) = next {
                return FalseWakeRemedy::StepDown { to: *next };
            }
        }
        let point = model.operating_point(level);
        let current = point.probability_cutoff;
        let ladder = TunedLadder::centred_on(
            &model.id,
            current + Self::TIGHTEN_STEP,
            point.sliding_window_size,
        );
        if ladder.medium > current {
            FalseWakeRemedy::Tighten { ladder }
        } else {
            FalseWakeRemedy::Exhausted
        }
    }
}

/// One tuning session, scored.
///
/// Each number is the highest cutoff at which that take still fired, 0 when it
/// never fired at all. Storing the threshold rather than a hit at three preset
/// levels is the whole point: a hit count can only say "this level worked",
/// which is true of a wide range of levels and picks none of them. A threshold
/// says where the take actually sits, so a ladder can be solved for rather than
/// chosen from three guesses.
#[derive(Debug, Clone, PartialEq)]
pub struct TunedSession {
    pub id: i32,
    pub phrase_scores: Vec<f32>,
    pub decoy_scores: Vec<f32>,
}

/// A ladder solved from one or more sessions, and how well it separates.
///
/// `overlapping` is the answer nobody wants and the one that has to be said out
/// loud. When the things a person says that are NOT the wake phrase score as
/// high as the ones that are, no threshold divides them, and a screen that
/// reports a cheerful number anyway is how a ladder ships that nothing measured.
#[derive(Debug, Clone, PartialEq)]
pub struct LadderFit {
    pub ladder: TunedLadder,
    pub phrases_heard: i32,
    pub phrases_total: i32,
    pub worst_decoy: f32,
    pub overlapping: bool,
}

/// How far MEDIUM sits above the loudest thing the user said that was not the
/// wake phrase.
///
/// Small, because a decoy is speech from the same mouth into the same
/// microphone: it is already the hardest negative this room can offer, unlike
/// room noise, which is why this margin is a tenth of the one the built-in
/// ladder quotes against ambient.
const DECOY_MARGIN: f32 = 0.10;

/// Below this share of phrases still heard, the sets have not separated.
const MIN_PHRASES_KEPT: f32 = 0.6;

/// The ladder these sessions imply, or `None` before anybody has spoken.
///
/// MEDIUM lands [`DECOY_MARGIN`] above the loudest decoy anyone produced (the
/// whole household's worst, not each person's own), because one setting serves
/// the house and the person whose voice defeats it is the one who decides
/// where it sits.
///
/// [`TunedLadder::at_least`] then floors it at the model's built-in ladder, so
/// a session held in a quiet room can tighten the setting and can never loosen
/// it. That asymmetry is deliberate and it is the guard rail: the built-in
/// numbers were measured against twenty minutes of a working kitchen, and
/// thirty seconds of somebody standing still in a hallway is not evidence that
/// beats them.
pub fn fit(sessions: &[TunedSession], model: &WakeWordModel) -> Option<LadderFit> {
    let phrases: Vec<f32> = sessions
        .iter()
        .flat_map(|s| s.phrase_scores.iter().copied())
        .collect();
    if phrases.is_empty() {
        return None;
    }
    let worst_decoy = sessions
        .iter()
        .flat_map(|s| s.decoy_scores.iter().copied())
        .reduce(f32::max)
        .unwrap_or(0.0);
    let ladder = TunedLadder::centred_on(
        &model.id,
        worst_decoy + DECOY_MARGIN,
        model.medium.sliding_window_size,
    )
    .at_least(model);
    let heard = phrases.iter().filter(|p| **p >= ladder.medium).count();
    Some(LadderFit {
        overlapping: (heard as f32) / (phrases.len() as f32) < MIN_PHRASES_KEPT,
        ladder,
        phrases_heard: heard as i32,
        phrases_total: phrases.len() as i32,
        worst_decoy,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wake_model::{OperatingPoint, default_model};

    const TOLERANCE: f32 = 0.0001;

    fn close(expected: f32, actual: f32) -> bool {
        (expected - actual).abs() <= TOLERANCE
    }

    #[test]
    fn a_ladder_is_spaced_a_step_either_side_of_its_middle() {
        let ladder = TunedLadder::centred_on("hey_ari", 0.70, 10);

        assert!(close(0.60, ladder.high), "{ladder:?}");
        assert!(close(0.70, ladder.medium), "{ladder:?}");
        assert!(close(0.80, ladder.low), "{ladder:?}");
    }

    #[test]
    fn near_the_ceiling_the_rungs_compress_rather_than_dragging_the_middle_down() {
        // The remedy path asks for a middle of 0.90 when somebody on the
        // strictest setting is still getting false wakes. Answering 0.85
        // because 0.90 + a step will not fit would quietly hand back a LOOSER
        // setting than the one they complained about.
        let ladder = TunedLadder::centred_on("hey_ari", 0.90, 10);

        assert!(close(0.90, ladder.medium), "{ladder:?}");
        assert!(close(0.80, ladder.high), "{ladder:?}");
        assert!(close(TunedLadder::MAX_CUTOFF, ladder.low), "{ladder:?}");
        assert!(ladder.medium < ladder.low, "rungs must stay distinct");
    }

    #[test]
    fn the_rungs_stay_ordered_and_distinct_at_both_limits() {
        for asked in [0.0, 0.1, 0.5, 0.94, 0.99, 1.0] {
            let ladder = TunedLadder::centred_on("hey_ari", asked, 10);
            assert!(ladder.high < ladder.medium, "{asked}: high < medium");
            assert!(ladder.medium < ladder.low, "{asked}: medium < low");
            assert!(
                ladder.high >= TunedLadder::MIN_CUTOFF,
                "{asked}: within bounds"
            );
            assert!(
                ladder.low <= TunedLadder::MAX_CUTOFF,
                "{asked}: within bounds"
            );
        }
    }

    #[test]
    fn a_measured_ladder_may_tighten_the_built_in_one() {
        let model = default_model();
        let ladder = TunedLadder::centred_on("hey_ari", 0.90, 10).at_least(&model);

        assert!(ladder.medium > model.medium.probability_cutoff);
    }

    #[test]
    fn a_measured_ladder_may_never_loosen_the_built_in_one() {
        // The guard rail. A session recorded in a silent hallway produces no
        // decoys worth clearing and would otherwise solve for a very low
        // cutoff, which is precisely the ladder that shipped in September and
        // false-fired within two days.
        let model = default_model();
        let ladder = TunedLadder::centred_on("hey_ari", 0.35, 10).at_least(&model);

        assert!(close(model.high.probability_cutoff, ladder.high));
        assert!(close(model.medium.probability_cutoff, ladder.medium));
        assert!(close(model.low.probability_cutoff, ladder.low));
    }

    #[test]
    fn a_ladder_survives_being_written_down_and_read_back() {
        let ladder = TunedLadder::centred_on("hey_ari", 0.72, 10);

        assert_eq!(TunedLadder::parse(Some(&ladder.format())), Some(ladder));
    }

    #[test]
    fn the_stored_form_is_the_one_the_app_has_always_written() {
        // Ladders already on phones were written by Kotlin; they must keep
        // reading back, and a new one must look the same.
        let stored = "hey_ari_oww|0.5|0.6|0.7|1";
        let ladder = TunedLadder::parse(Some(stored)).expect("parses");
        assert_eq!(
            ladder,
            TunedLadder {
                model_id: "hey_ari_oww".into(),
                high: 0.5,
                medium: 0.6,
                low: 0.7,
                sliding_window_size: 1
            },
        );
        assert_eq!(ladder.format(), stored);
    }

    #[test]
    fn nonsense_on_the_way_back_in_is_refused_rather_than_half_applied() {
        assert_eq!(TunedLadder::parse(None), None);
        assert_eq!(TunedLadder::parse(Some("")), None);
        assert_eq!(TunedLadder::parse(Some("hey_ari|0.6|0.7")), None);
        assert_eq!(TunedLadder::parse(Some("hey_ari|0.6|0.7|0.8|nope")), None);
        // Out of order: LOW must be the strictest or the whole picker inverts.
        assert_eq!(TunedLadder::parse(Some("hey_ari|0.9|0.7|0.5|10")), None);
        // Equal rungs quantise to the same byte, so three settings do one thing.
        assert_eq!(TunedLadder::parse(Some("hey_ari|0.7|0.7|0.8|10")), None);
        assert_eq!(TunedLadder::parse(Some("hey_ari|0.1|0.2|0.3|10")), None);
    }

    #[test]
    fn a_ladder_belonging_to_another_model_is_ignored_not_applied() {
        let model = default_model();
        let ladder = TunedLadder::centred_on("ok_ari", 0.50, 10);

        assert_eq!(model.with_ladder(Some(&ladder)), model);
        assert_eq!(model.with_ladder(None), model);
    }

    #[test]
    fn a_ladder_for_this_model_replaces_every_rung_and_the_window() {
        let model = default_model();
        let ladder = TunedLadder {
            model_id: model.id.clone(),
            high: 0.60,
            medium: 0.70,
            low: 0.80,
            sliding_window_size: 15,
        };

        let tuned = model.with_ladder(Some(&ladder));

        assert_eq!(
            tuned.operating_point(WakeSensitivity::High),
            OperatingPoint::new(0.60, 15)
        );
        assert_eq!(
            tuned.operating_point(WakeSensitivity::Medium),
            OperatingPoint::new(0.70, 15)
        );
        assert_eq!(
            tuned.operating_point(WakeSensitivity::Low),
            OperatingPoint::new(0.80, 15)
        );
        assert_eq!(tuned.id, model.id);
        assert_eq!(tuned.engine, model.engine);
    }

    #[test]
    fn the_loose_settings_just_step_down_the_ladder_they_already_have() {
        let model = default_model();
        assert_eq!(
            FalseWakeRemedy::for_level(WakeSensitivity::High, &model),
            FalseWakeRemedy::StepDown {
                to: WakeSensitivity::Medium
            },
        );
        assert_eq!(
            FalseWakeRemedy::for_level(WakeSensitivity::Medium, &model),
            FalseWakeRemedy::StepDown {
                to: WakeSensitivity::Low
            },
        );
    }

    #[test]
    fn on_the_strictest_setting_the_ladder_itself_moves() {
        let model = default_model();
        let FalseWakeRemedy::Tighten { ladder } =
            FalseWakeRemedy::for_level(WakeSensitivity::Low, &model)
        else {
            panic!("expected Tighten");
        };
        assert!(close(
            model.low.probability_cutoff + FalseWakeRemedy::TIGHTEN_STEP,
            ladder.medium
        ));
    }

    #[test]
    fn tightening_from_the_strictest_setting_really_is_stricter() {
        // The trap in the obvious implementation: shift the ladder up a step
        // and then land the user on its MIDDLE, which is below the rung they
        // were already on. They report false wakes and get a looser phone.
        let model = default_model();
        let before = model
            .operating_point(WakeSensitivity::Low)
            .probability_cutoff;

        let FalseWakeRemedy::Tighten { ladder } =
            FalseWakeRemedy::for_level(WakeSensitivity::Low, &model)
        else {
            panic!("expected Tighten");
        };
        assert!(
            ladder.medium > before,
            "{} should be stricter than {before}",
            ladder.medium
        );
    }

    #[test]
    fn repeated_tightening_stops_rather_than_pretending() {
        // Walk it up the way a household would: complain, accept, complain.
        let mut current = default_model();
        for _ in 0..8 {
            match FalseWakeRemedy::for_level(WakeSensitivity::Medium, &current) {
                FalseWakeRemedy::Tighten { ladder } => current = current.with_ladder(Some(&ladder)),
                FalseWakeRemedy::StepDown { to } => {
                    let ladder = TunedLadder::centred_on(
                        &current.id,
                        current.operating_point(to).probability_cutoff,
                        current.medium.sliding_window_size,
                    );
                    current = current.with_ladder(Some(&ladder));
                }
                FalseWakeRemedy::Exhausted => continue,
            }
            assert!(
                current.low.probability_cutoff <= TunedLadder::MAX_CUTOFF,
                "never past the point where cutoffs stop being distinguishable",
            );
        }
    }

    #[test]
    fn a_phone_already_at_the_ceiling_is_told_so_instead_of_being_nudged() {
        let model = default_model();
        let pinned = model.with_ladder(Some(&TunedLadder {
            model_id: model.id.clone(),
            high: 0.90,
            medium: 0.93,
            low: TunedLadder::MAX_CUTOFF,
            sliding_window_size: model.medium.sliding_window_size,
        }));

        assert_eq!(
            FalseWakeRemedy::for_level(WakeSensitivity::Low, &pinned),
            FalseWakeRemedy::Exhausted
        );
    }

    fn session(phrases: &[f32], decoys: &[f32]) -> TunedSession {
        TunedSession {
            id: 1,
            phrase_scores: phrases.to_vec(),
            decoy_scores: decoys.to_vec(),
        }
    }

    #[test]
    fn nothing_spoken_yet_is_none_not_a_ladder_built_from_no_evidence() {
        let model = default_model();
        assert_eq!(fit(&[], &model), None);
        assert_eq!(fit(&[session(&[], &[0.0])], &model), None);
    }

    #[test]
    fn the_middle_rung_clears_the_loudest_decoy() {
        let f = fit(&[session(&[0.99, 0.98], &[0.70, 0.40])], &default_model()).unwrap();

        assert!(close(0.80, f.ladder.medium), "{f:?}");
        assert!(close(0.70, f.worst_decoy), "{f:?}");
    }

    #[test]
    fn the_households_worst_decoy_sets_it_not_each_sessions_own() {
        // One setting serves the house, so the person whose voice defeats it
        // is the person who decides where it sits. Averaging the sessions would
        // let a quiet one hide a loud one.
        let f = fit(
            &[session(&[0.99], &[0.30]), session(&[0.99], &[0.82])],
            &default_model(),
        )
        .unwrap();

        assert!(close(0.92, f.ladder.medium), "{f:?}");
    }

    #[test]
    fn a_silent_room_cannot_loosen_the_built_in_ladder() {
        // Decoys that never fired at all. Solved naively this asks for a
        // cutoff of 0.10, which is below anything a kitchen does.
        let model = default_model();
        let f = fit(&[session(&[0.99, 0.99], &[0.0, 0.0])], &model).unwrap();

        assert!(
            close(model.medium.probability_cutoff, f.ladder.medium),
            "{f:?}"
        );
    }

    #[test]
    fn phrases_that_still_land_at_the_new_setting_are_counted() {
        let f = fit(&[session(&[0.99, 0.95, 0.10], &[0.60])], &default_model()).unwrap();

        assert_eq!(f.phrases_heard, 2);
        assert_eq!(f.phrases_total, 3);
    }

    #[test]
    fn decoys_that_score_like_the_wake_phrase_are_reported_as_overlapping() {
        // The result the screen must not dress up: ordinary speech reaching the
        // same scores as the phrase means no cutoff separates them.
        let f = fit(&[session(&[0.80, 0.75, 0.70], &[0.88])], &default_model()).unwrap();

        assert!(f.overlapping);
        assert_eq!(f.phrases_heard, 0);
    }

    #[test]
    fn a_clean_separation_is_not_reported_as_overlapping() {
        let f = fit(
            &[session(&[0.99, 0.98, 0.97, 0.99], &[0.30])],
            &default_model(),
        )
        .unwrap();

        assert!(!f.overlapping);
        assert_eq!(f.phrases_heard, 4);
    }

    #[test]
    fn a_session_with_no_decoys_measures_from_zero() {
        let model = default_model();
        let f = fit(&[session(&[0.99], &[])], &model).unwrap();

        assert_eq!(f.worst_decoy, 0.0);
        assert!(
            close(model.medium.probability_cutoff, f.ladder.medium),
            "{f:?}"
        );
    }

    #[test]
    fn the_review_waits_two_days() {
        assert_eq!(LADDER_REVIEW_DELAY_HOURS, 48);
    }
}
