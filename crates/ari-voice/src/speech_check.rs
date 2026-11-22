//! The wake word's speech check: a detection only opens Ari if there was
//! speech in the second before it, on the rule measured offline
//! (`ari-tools/wakeword/trained/2026-10-07-vad-gate/`). Measured, not tuned;
//! changing any number here means measuring again.

use crate::native::{SileroVad, VAD_WINDOW_SAMPLES};

/// A window at or over this Silero probability is speech.
pub const SPEECH_PROBABILITY: f32 = 0.5;

/// The speech check's view: the ring buffer's 2 s ending at the detection...
pub const SPEECH_CHECK_BUFFER_SAMPLES: usize = 2 * 16_000;

/// ...with speech required in its last second.
pub const SPEECH_CHECK_TAIL_SAMPLES: usize = 16_000;

/// Silero asking "has anyone been speaking recently?", one 512-sample window
/// at a time against [`SPEECH_PROBABILITY`]; sherpa's segment machinery is not
/// used.
///
/// Two kinds of owner, each with its own instance: a listen, for the streaming
/// endpoint's veto through [`feed`](Self::feed) and the buffered listen's
/// speech marks through [`heard_speech`](Self::heard_speech); and the wake
/// word's check for speech before a wake, through
/// [`speech_in_tail`](Self::speech_in_tail). One instance belongs to one thread.
pub struct SpeechGate {
    vad: SileroVad,
    // Carry-over so 1600-sample batches don't drop the 3.125-window tail.
    pending: Vec<f32>,
    last_speech_at: Option<i64>,
}

impl SpeechGate {
    pub fn new(model: &[u8]) -> Result<Self, String> {
        Ok(Self { vad: SileroVad::new(model)?, pending: Vec::new(), last_speech_at: None })
    }

    /// Reset per-utterance state at each listen start: the gate is reused
    /// across utterances, since the model is not cheap to rebuild.
    pub fn begin_utterance(&mut self) {
        self.pending.clear();
        self.last_speech_at = None;
        self.vad.reset();
    }

    /// Feed one decode-loop batch; `now_ms` is the caller's clock.
    pub fn feed(&mut self, samples: &[f32], now_ms: i64) -> Result<(), String> {
        if self.scan(samples)? {
            self.last_speech_at = Some(now_ms);
        }
        Ok(())
    }

    /// Feed one chunk; whether a window it completed was speech.
    pub fn heard_speech(&mut self, samples: &[i16]) -> Result<bool, String> {
        let floats: Vec<f32> = samples.iter().map(|s| *s as f32 / 32768.0).collect();
        self.scan(&floats)
    }

    fn scan(&mut self, samples: &[f32]) -> Result<bool, String> {
        self.pending.extend_from_slice(samples);
        let mut heard = false;
        let mut offset = 0;
        while self.pending.len() - offset >= VAD_WINDOW_SAMPLES {
            let window: &[f32; VAD_WINDOW_SAMPLES] =
                self.pending[offset..offset + VAD_WINDOW_SAMPLES].try_into().expect("a whole window");
            if self.vad.compute(window).ok_or("Silero failed to run")? >= SPEECH_PROBABILITY {
                heard = true;
            }
            offset += VAD_WINDOW_SAMPLES;
        }
        self.pending.drain(..offset);
        Ok(heard)
    }

    /// `i64::MAX` until the first speech window: arming silence must not read
    /// as "recent speech".
    pub fn ms_since_last_speech(&self, now_ms: i64) -> i64 {
        self.last_speech_at.map_or(i64::MAX, |at| now_ms - at)
    }

    /// Whether `pcm` holds speech in a window ending within its last
    /// `tail_samples`, scored from a fresh state with the windows aligned so
    /// the last ends where `pcm` does. An error when the model fails to run.
    pub fn speech_in_tail(&mut self, pcm: &[i16], tail_samples: usize) -> Result<bool, String> {
        // Clears the state feed() keeps, so the two uses never share an instance.
        self.vad.reset();
        let mut start = pcm.len() % VAD_WINDOW_SAMPLES;
        while start + VAD_WINDOW_SAMPLES <= pcm.len() {
            let window: [f32; VAD_WINDOW_SAMPLES] = std::array::from_fn(|i| pcm[start + i] as f32 / 32768.0);
            let probability = self.vad.compute(&window).ok_or("Silero failed to run")?;
            if start + VAD_WINDOW_SAMPLES > pcm.len().saturating_sub(tail_samples) && probability >= SPEECH_PROBABILITY {
                return Ok(true);
            }
            start += VAD_WINDOW_SAMPLES;
        }
        Ok(false)
    }
}

/// When to run the speech check on a detection the engine keeps reporting, and
/// which of the resulting drops the wake log records.
///
/// A detection the check drops is left standing in the engine (see
/// [`WakeDetector`](crate::detector::WakeDetector)'s `reset_on_detection`),
/// which then reports again on every inference while its window stays above
/// the cutoff. That is what lets a wake model that fires a moment before the
/// phrase still catch the phrase: the check runs again once the speech has
/// arrived. Running it on every report would cost a speech-model pass every
/// 20-30 ms, so it runs at most every [`RECHECK_MS`](Self::RECHECK_MS).
///
/// A run of reports with no gap longer than
/// [`EPISODE_GAP_MS`](Self::EPISODE_GAP_MS) is one episode, and one episode is
/// one dropped wake in the log however many checks it took, so the count stays
/// comparable with the wakes that do open the overlay.
///
/// Measured offline before it was built: `ari-tools/wakeword/trained/
/// 2026-10-07-vad-gate/RESULTS.md`.
#[derive(Debug)]
pub struct SpeechCheckPacing {
    last_report_at: i64,
    next_check_at: i64,
    episode_logged: bool,
}

impl Default for SpeechCheckPacing {
    fn default() -> Self {
        Self { last_report_at: i64::MIN / 2, next_check_at: i64::MIN, episode_logged: false }
    }
}

impl SpeechCheckPacing {
    pub const RECHECK_MS: i64 = 250;
    pub const EPISODE_GAP_MS: i64 = 1_000;

    /// Called on every detection the engine reports. True when the check should run now.
    pub fn should_check(&mut self, now: i64) -> bool {
        if now - self.last_report_at > Self::EPISODE_GAP_MS {
            self.episode_logged = false;
            self.next_check_at = now;
        }
        self.last_report_at = now;
        now >= self.next_check_at
    }

    /// The check found no speech. True for the episode's first drop, the one to log.
    pub fn dropped(&mut self, now: i64) -> bool {
        self.next_check_at = now + Self::RECHECK_MS;
        let first = !self.episode_logged;
        self.episode_logged = true;
        first
    }

    /// The check found speech. True when this episode already logged a drop:
    /// the wake was let through after all, so that drop prevented nothing.
    pub fn passed(&self) -> bool {
        self.episode_logged
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_report_of_an_episode_is_checked_at_once() {
        assert!(SpeechCheckPacing::default().should_check(10_000));
    }

    #[test]
    fn after_a_drop_the_check_waits_250_ms_while_the_engine_keeps_reporting() {
        let mut pacing = SpeechCheckPacing::default();
        pacing.should_check(10_000);
        pacing.dropped(10_000);
        // The engine reports every 20 ms on Hey Ari.
        let checked: Vec<i64> = (10_020..=10_400).step_by(20).filter(|t| pacing.should_check(*t)).collect();
        assert_eq!(vec![10_260], checked[..1]);
    }

    #[test]
    fn one_episode_logs_one_drop_however_many_checks_it_takes() {
        let mut pacing = SpeechCheckPacing::default();
        let mut logged = vec![];
        for t in (10_000..=11_000).step_by(20) {
            if pacing.should_check(t) && pacing.dropped(t) {
                logged.push(t);
            }
        }
        assert_eq!(vec![10_000], logged);
    }

    #[test]
    fn a_pass_after_a_logged_drop_in_the_same_episode_says_the_drop_prevented_nothing() {
        let mut pacing = SpeechCheckPacing::default();
        pacing.should_check(10_000);
        assert!(!pacing.passed());
        pacing.dropped(10_000);
        pacing.should_check(10_260);
        assert!(pacing.passed());
    }

    #[test]
    fn a_pass_in_a_fresh_episode_is_a_plain_wake() {
        let mut pacing = SpeechCheckPacing::default();
        pacing.should_check(10_000);
        pacing.dropped(10_000);
        pacing.should_check(12_000);
        assert!(!pacing.passed());
    }

    #[test]
    fn a_gap_of_over_a_second_starts_a_new_episode_checked_and_logged_afresh() {
        let mut pacing = SpeechCheckPacing::default();
        pacing.should_check(10_000);
        pacing.dropped(10_000);
        assert!(!pacing.should_check(10_100));
        // The engine fell quiet, then reported again 1.2 s later.
        assert!(pacing.should_check(11_300));
        assert!(pacing.dropped(11_300));
    }
}
