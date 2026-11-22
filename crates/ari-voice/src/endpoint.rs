//! When a listen ends: the streaming path's partial-stability rule with its
//! VAD veto, and the buffered path's loudness-and-detector rule. Times are the
//! caller's clock, in milliseconds.

/// How long the cleaned partial must hold steady before the user is taken to
/// be done speaking. 1500 ms is the empirical sweet spot: longer doesn't help
/// (sherpa's streaming decoder either commits late tokens within ~1s or never
/// does) and slows responses.
pub const STABILITY_WINDOW_MS: i64 = 1_500;

/// VAD veto: the partial-stability endpoint may only fire when silero has
/// also heard no speech for this long. Stability measures the DECODER going
/// quiet; in heavy noise sherpa stalls mid-utterance and the two diverge. The
/// two-day capture set has a clip where the user spoke for 2.3s past the last
/// committed token and got amputated.
pub const VETO_SPEECH_WINDOW_MS: i64 = 1_000;

/// Upper bound on the veto itself. Continuous speech-like noise (a
/// television, a conversation at the next table) never lets the time since
/// speech reach [`VETO_SPEECH_WINDOW_MS`], which would starve the stability
/// arm and leave [`MAX_ONLINE_UTTERANCE_MS`] as the only way out: 30s of dead
/// air where today there is 1.5s. The veto exists to outlast a stalled
/// decoder, not to hand the endpoint over to the telly. 4s is a human-decided
/// UX bound (2026-07-30 review), not a measured constant.
pub const VETO_OVERRIDE_STABILITY_MS: i64 = 4_000;

/// Hard cap on a single streaming utterance, matching the buffered path's
/// 30s. Without it, steady noise that silero scores as speech would veto the
/// endpoint indefinitely.
pub const MAX_ONLINE_UTTERANCE_MS: i64 = 30_000;

/// Whether the streaming decode loop should end the utterance now.
///
/// `partial_stable_for_ms` is how long the cleaned partial has held its
/// current value, and callers pass 0 while it is empty: an empty partial can
/// never satisfy the stability arm, but it must still be able to hit the cap.
/// `ms_since_last_speech` is the VAD's, or `i64::MAX` when no VAD is
/// available, which reduces this to the stability-only endpoint shipped
/// before the veto.
pub fn should_endpoint(partial_stable_for_ms: i64, ms_since_last_speech: i64, listening_for_ms: i64) -> bool {
    if listening_for_ms >= MAX_ONLINE_UTTERANCE_MS {
        return true;
    }
    if partial_stable_for_ms < STABILITY_WINDOW_MS {
        return false;
    }
    ms_since_last_speech >= VETO_SPEECH_WINDOW_MS || partial_stable_for_ms >= VETO_OVERRIDE_STABILITY_MS
}

/// How a buffered (cloud or Whisper) listen ended. The slug goes in the wake log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ListenEnd {
    /// Speech, then [`MIN_SILENCE_AFTER_SPEECH_MS`] without any.
    Endpoint,
    /// The speech detector alone held it open [`DETECTOR_HOLD_MS`] past the last loud sound.
    Bound,
    /// A quiet wake's listen reached its limit.
    Limit,
    /// [`MAX_SAMPLES`] of audio.
    Cap,
    /// A quiet wake heard nothing at all in time, and nothing is transcribed.
    GiveUp,
}

impl ListenEnd {
    pub fn slug(self) -> &'static str {
        match self {
            Self::Endpoint => "endpoint",
            Self::Bound => "bound",
            Self::Limit => "limit",
            Self::Cap => "cap",
            Self::GiveUp => "give-up",
        }
    }
}

/// Once speech has been heard, this much without any ends the utterance.
/// 800 ms is roughly the window the streaming path uses for partial-text
/// stability, which keeps the response time alike on both paths.
pub const MIN_SILENCE_AFTER_SPEECH_MS: i64 = 800;

/// Don't fire the endpoint until the utterance is at least this long. Stops a
/// single noise burst followed by silence from ending the session before the
/// user has actually said anything.
pub const MIN_UTTERANCE_MS: i64 = 500;

/// The same limit the streaming path puts on its detector veto: long enough
/// to outlast a soft speaker, short enough not to hand the endpoint over to
/// the telly.
pub const DETECTOR_HOLD_MS: i64 = VETO_OVERRIDE_STABILITY_MS;

/// Whisper's training window is 30 s; longer audio gets chunked internally
/// and the first segment dominates accuracy, so it caps at the same point as a
/// safety net rather than an expected case.
pub const MAX_SAMPLES: usize = 16_000 * 30;

/// When a buffered listen should stop, decided chunk by chunk.
///
/// Speech is a chunk loud enough ([`SPEECH_RMS_THRESHOLD`] on its loudest
/// frame) or one in which the speech detector heard speech. Loudness alone
/// missed soft speakers: on 2026-10-14 it ignored two whole commands said
/// quietly, and cut a third off once its "Hey Ari" had passed, taking the
/// softer command after it for silence. The detector hears them, so either
/// counts, the pre-roll included.
///
/// The detector also hears a television and people talking in the next room,
/// and would hold a listen open as long as they go on. So it can carry a
/// listen at most [`DETECTOR_HOLD_MS`] past the last loud chunk, and a quiet
/// wake, which shows nothing while it listens and leaves the wake word off
/// meanwhile, stops at its own limit. Tested offline before it was built:
/// `ari-tools/wakeword/trained/2026-10-17-bounded-listen/`.
///
/// The pre-roll arrives all at once at the start.
#[derive(Debug)]
pub struct BufferedEndpoint {
    started_at: i64,
    /// A quiet wake's: give up untranscribed if nothing counts as speech by then.
    give_up_without_speech_ms: Option<i64>,
    /// A quiet wake's: transcribe whatever there is by then.
    limit_ms: Option<i64>,
    first_speech_at: Option<i64>,
    last_speech_at: i64,
    last_loud_at: Option<i64>,
}

impl BufferedEndpoint {
    pub fn new(started_at: i64, give_up_without_speech_ms: Option<i64>, limit_ms: Option<i64>) -> Self {
        Self {
            started_at,
            give_up_without_speech_ms,
            limit_ms,
            first_speech_at: None,
            last_speech_at: started_at,
            last_loud_at: None,
        }
    }

    /// `None` to keep listening. Call once per chunk, after adding it to `total_samples`.
    pub fn next(&mut self, now: i64, loud: bool, detector_speech: bool, total_samples: usize) -> Option<ListenEnd> {
        if loud {
            self.last_loud_at = Some(now);
        }
        if loud || detector_speech {
            self.first_speech_at.get_or_insert(now);
            self.last_speech_at = now;
        }
        let first = self.first_speech_at;
        if let Some(first) = first
            && now - self.last_speech_at >= MIN_SILENCE_AFTER_SPEECH_MS
            && now - first >= MIN_UTTERANCE_MS
        {
            return Some(ListenEnd::Endpoint);
        }
        if first.is_some() && now - self.last_loud_at.unwrap_or(self.started_at) >= DETECTOR_HOLD_MS {
            return Some(ListenEnd::Bound);
        }
        if self.limit_ms.is_some_and(|limit| now - self.started_at >= limit) {
            return Some(ListenEnd::Limit);
        }
        if total_samples >= MAX_SAMPLES {
            return Some(ListenEnd::Cap);
        }
        if first.is_none() && self.give_up_without_speech_ms.is_some_and(|ms| now - self.started_at >= ms) {
            return Some(ListenEnd::GiveUp);
        }
        None
    }
}

/// 30 ms, the size of one capture chunk, so a live chunk scores exactly as it
/// did before frames were looked at separately.
const RMS_FRAME_SAMPLES: usize = 480;

/// RMS for "this chunk is loud enough to be speech". int16 PCM samples span
/// [-32768, 32767]; ambient room noise is typically under 300. Soft speakers
/// sit below it too, which is why the speech detector counts as well
/// ([`BufferedEndpoint`]); loudness still decides how long the detector may
/// carry a listen on its own.
pub const SPEECH_RMS_THRESHOLD: f32 = 600.0;

/// The RMS of the loudest [`RMS_FRAME_SAMPLES`] frame in `samples`. The
/// loudest frame rather than the whole chunk: a "Hey Ari" inside a pre-roll
/// slice, averaged over the silence around it, can sit below the speech level.
pub fn loudest_frame_rms(samples: &[i16]) -> f32 {
    samples
        .chunks(RMS_FRAME_SAMPLES)
        .map(|frame| {
            let sum: f64 = frame.iter().map(|s| (*s as f64) * (*s as f64)).sum();
            (sum / frame.len() as f64).sqrt() as f32
        })
        .fold(0.0, f32::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- EndpointDecisionTest ---

    #[test]
    fn stable_partial_with_quiet_vad_fires() {
        assert!(should_endpoint(1500, 1000, 5_000));
    }

    #[test]
    fn stable_partial_is_vetoed_while_vad_still_hears_speech() {
        // The reminder-clip failure: decoder stalled at "business" for 1.5s
        // while the user was saying "in one hour". VAD heard speech 200ms ago
        // -> must NOT endpoint.
        assert!(!should_endpoint(1500, 200, 5_000));
    }

    #[test]
    fn unstable_partial_never_fires_even_when_quiet() {
        assert!(!should_endpoint(800, 2_000, 5_000));
    }

    #[test]
    fn hard_cap_fires_regardless_of_vad() {
        assert!(should_endpoint(0, 0, 30_000));
    }

    #[test]
    fn long_stability_overrides_the_veto() {
        // Continuous speech-like noise never lets the time since speech reach
        // the veto window, which would starve the stability arm and leave the
        // 30s cap as the only exit.
        assert!(should_endpoint(4000, 0, 5_000));
    }

    #[test]
    fn just_under_the_override_still_respects_the_veto() {
        assert!(!should_endpoint(3999, 0, 5_000));
    }

    #[test]
    fn empty_partial_cannot_fire_before_the_hard_cap() {
        // Callers report zero stability while the partial is empty, so neither
        // the stability arm nor the override may fire on it.
        assert!(!should_endpoint(0, 0, 5_000));
        assert!(!should_endpoint(0, i64::MAX, 29_999));
    }

    #[test]
    fn boundary_values_fire_exactly_at_thresholds() {
        assert!(should_endpoint(1500, 1000, 0));
        assert!(!should_endpoint(1500, 999, 29_999));
    }

    // --- BufferedEndpointTest ---

    const PREROLL: usize = 32_000;
    const CHUNK: usize = 480;

    /// Feeds a listen as the phone does: the pre-roll at 0, then a 30 ms chunk
    /// at a time, `loud` and `heard` saying which live times are speech.
    fn run(
        mut endpoint: BufferedEndpoint,
        loud_preroll: bool,
        heard_preroll: bool,
        loud: impl Fn(i64) -> bool,
        heard: impl Fn(i64) -> bool,
    ) -> (ListenEnd, i64) {
        let mut samples = PREROLL;
        if let Some(end) = endpoint.next(0, loud_preroll, heard_preroll, samples) {
            return (end, 0);
        }
        let mut t = 0;
        loop {
            t += 30;
            samples += CHUNK;
            if let Some(end) = endpoint.next(t, loud(t), heard(t), samples) {
                return (end, t);
            }
        }
    }

    fn quiet_wake() -> BufferedEndpoint {
        BufferedEndpoint::new(0, Some(3_000), Some(6_000))
    }

    fn ordinary() -> BufferedEndpoint {
        BufferedEndpoint::new(0, None, None)
    }

    #[test]
    fn a_loud_bare_hey_ari_ends_800_ms_later_as_before() {
        assert_eq!((ListenEnd::Endpoint, 810), run(quiet_wake(), true, false, |_| false, |_| false));
    }

    #[test]
    fn a_call_too_soft_to_be_loud_is_heard_instead_of_given_up_on() {
        // 2026-10-14, call 2: nothing reached the speech level, the detector heard it all.
        let ended = run(quiet_wake(), false, true, |_| false, |t| (250..=1_250).contains(&t));
        assert_eq!((ListenEnd::Endpoint, 2_040), ended);
    }

    #[test]
    fn a_soft_command_after_a_loud_hey_ari_is_not_cut_off() {
        // Call 3: only the phrase was loud, and loudness alone ended it at 0.8 s.
        let ended = run(quiet_wake(), true, false, |_| false, |t| (200..=1_900).contains(&t));
        assert_eq!((ListenEnd::Endpoint, 2_700), ended);
    }

    #[test]
    fn the_detector_alone_holds_a_listen_at_most_4_s_past_the_last_loud_sound() {
        let telly = run(ordinary(), false, false, |t| t == 1_500, |_| true);
        assert_eq!((ListenEnd::Bound, 5_520), telly);
    }

    #[test]
    fn a_quiet_wake_transcribes_what_it_has_at_its_limit_however_loud_the_room() {
        let loud_room = run(quiet_wake(), true, false, |_| true, |_| false);
        assert_eq!((ListenEnd::Limit, 6_000), loud_room);
    }

    #[test]
    fn an_ordinary_listen_in_a_loud_room_runs_to_the_cap() {
        // The 934th live chunk takes the pre-roll past 30 s of audio.
        assert_eq!((ListenEnd::Cap, 28_020), run(ordinary(), true, false, |_| true, |_| false));
    }

    #[test]
    fn a_quiet_wake_that_hears_nothing_gives_up_at_3_s_an_ordinary_listen_never_does() {
        assert_eq!((ListenEnd::GiveUp, 3_000), run(quiet_wake(), false, false, |_| false, |_| false));
        assert_eq!(ListenEnd::Cap, run(ordinary(), false, false, |_| false, |_| false).0);
    }

    // --- QuietListenTest ---

    #[test]
    fn a_short_loud_stretch_inside_a_long_quiet_preroll_is_heard() {
        // 2 s of room tone at 100 with 0.6 s of speech at 2000: averaged over
        // the whole chunk it sits under the 600 speech level, its loudest
        // frame does not.
        let preroll: Vec<i16> = (0..32_000)
            .map(|i| {
                let level = if (20_000..29_600).contains(&i) { 2000 } else { 100 };
                if i % 2 == 0 { level } else { -level }
            })
            .collect();
        assert!((loudest_frame_rms(&preroll) - 2000.0).abs() < 0.01);
    }

    #[test]
    fn a_30_ms_live_chunk_scores_the_same_as_before() {
        let chunk: Vec<i16> = (0..480).map(|i| if i % 2 == 0 { 700 } else { -700 }).collect();
        assert!((loudest_frame_rms(&chunk) - 700.0).abs() < 0.01);
    }
}
