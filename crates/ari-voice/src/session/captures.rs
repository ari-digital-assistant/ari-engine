//! The debug captures a voice turn can leave behind, moved from ari-android's
//! `AudioClipStore.kt`, `WakeCaptureStore.kt` and `UtteranceCaptureStore.kt`:
//! the audio of a wake or an utterance as a WAV, with a `.txt` sidecar beside
//! it, in bounded directories under the frontend's files directory.
//!
//! Every one is off by default behind a setting the session checks, not this.
//! The frontend lists, shares and clears the directories itself.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use chrono::TimeZone;

use crate::stt::wav_bytes;
use crate::wake_log::iso_offset_date_time;

/// Which path caught a wake clip. Recorded in its filename.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WakeCaptureHook {
    /// The transcript contained speech but no wake-phrase name token.
    Rejected,
    /// The wake fired and nobody said anything before the silence timeout.
    Silent,
    /// The wake fired and the transcriber found no speech in what followed.
    /// The same label as `Silent`, reached on the paths that endpoint and
    /// transcribe instead of timing out, which on cloud transcription is all
    /// of them.
    NoSpeech,
    /// The wake fired and the turn was accepted: a true positive. Captured by
    /// the keep-wake setting (or keep-everything), never by the false-trigger
    /// one: the two are opposite labels and must not share a switch.
    Accepted,
}

impl WakeCaptureHook {
    pub fn slug(self) -> &'static str {
        match self {
            Self::Rejected => "rejected",
            Self::Silent => "silent",
            Self::NoSpeech => "no-speech",
            Self::Accepted => "accepted",
        }
    }
}

/// Which kind of turn produced an utterance. Recorded in the sidecar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UtteranceTurn {
    /// The first turn of a session: a wake word, or a tap on the mic.
    Opening,
    /// An answer to a question a skill asked, on a mic re-armed without a wake word.
    Reply,
    /// A turn inside continuous "let's talk" mode.
    Talk,
    /// Composer dictation: transcribed into the text field, never dispatched.
    Dictation,
}

impl UtteranceTurn {
    pub fn slug(self) -> &'static str {
        match self {
            Self::Opening => "opening",
            Self::Reply => "reply",
            Self::Talk => "talk",
            Self::Dictation => "dictation",
        }
    }
}

/// What became of the transcript. Also the last segment of the filename.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UtteranceOutcome {
    /// The engine acted on the first transcript.
    Answered,
    /// The engine only acted after a parallel or offline retry corrected the transcript.
    Rescued,
    /// Every transcript this turn produced left the engine none the wiser.
    NotUnderstood,
    /// The words matched a button on the card on screen, so no engine dispatch happened.
    Card,
    /// The words stopped a ringing alert, so no engine dispatch happened.
    Alert,
    /// The recogniser endpointed with nothing to show for it.
    Blank,
    /// Dictation, which has no engine outcome to report.
    Dictated,
}

impl UtteranceOutcome {
    pub fn slug(self) -> &'static str {
        match self {
            Self::Answered => "answered",
            Self::Rescued => "rescued",
            Self::NotUnderstood => "not-understood",
            Self::Card => "card",
            Self::Alert => "alert",
            Self::Blank => "blank",
            Self::Dictated => "dictated",
        }
    }
}

/// Everything known about one utterance except the audio itself. The
/// transcript fields are the three the retry ladder can produce plus the raw
/// text before wake-phrase stripping: together they show which layer heard
/// what, which is the whole point of keeping the recording.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UtteranceCapture {
    pub turn: UtteranceTurn,
    pub outcome: UtteranceOutcome,
    /// How the engine replied, e.g. `action(dev.heyari.weather)`. `None` off the engine path.
    pub response: Option<String>,
    pub raw: Option<String>,
    pub transcript: String,
    pub parallel: Option<String>,
    pub offline: Option<String>,
    /// The transcript Ari actually acted on. Differs from `transcript` on a rescue.
    pub used: String,
    pub locale: String,
    pub model: Option<String>,
}

const NONE: &str = "(none)";

fn or_none(value: &str) -> &str {
    if value.trim().is_empty() { NONE } else { value }
}

/// `capture` as the clip's `.txt` sidecar: one `key: value` line per field,
/// in a fixed order, so a directory of exports greps and diffs cleanly.
pub fn utterance_sidecar<Tz: TimeZone>(capture: &UtteranceCapture, timestamp_ms: i64, zone: &Tz) -> String {
    [
        ("when", iso_offset_date_time(timestamp_ms, zone).as_str()),
        ("turn", capture.turn.slug()),
        ("outcome", capture.outcome.slug()),
        ("locale", &capture.locale),
        ("model", capture.model.as_deref().unwrap_or(NONE)),
        ("response", capture.response.as_deref().unwrap_or(NONE)),
        ("raw", capture.raw.as_deref().unwrap_or(NONE)),
        ("transcript", or_none(&capture.transcript)),
        ("parallel", capture.parallel.as_deref().unwrap_or(NONE)),
        ("offline", capture.offline.as_deref().unwrap_or(NONE)),
        ("used", or_none(&capture.used)),
    ]
    .iter()
    .map(|(key, value)| format!("{key}: {value}\n"))
    .collect()
}

/// `<prefix>-<zero-padded epoch ms>-<slug>`, the filename stem every clip and
/// its sidecar share. Fixed width, because [`evict_oldest`] treats
/// lexicographic order as chronological order.
pub fn clip_stem(prefix: &str, timestamp_ms: i64, slug: &str) -> String {
    format!("{prefix}-{timestamp_ms:017}-{slug}")
}

fn clips(dir: &Path) -> Vec<(PathBuf, u64)> {
    let Ok(entries) = fs::read_dir(dir) else { return Vec::new() };
    let mut clips: Vec<(PathBuf, u64)> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "wav"))
        .map(|p| {
            let len = fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
            (p, len)
        })
        .collect();
    clips.sort_by(|a, b| a.0.file_name().cmp(&b.0.file_name()));
    clips
}

/// Delete the oldest clip and sidecar pairs until `dir` is within both caps.
/// Filenames are timestamp-prefixed and fixed-width, so lexicographic order is
/// chronological order.
pub fn evict_oldest(dir: &Path, max_files: usize, max_bytes: u64) {
    let clips = clips(dir);
    let mut count = clips.len();
    let mut bytes: u64 = clips.iter().map(|(_, len)| len).sum();
    for (clip, len) in clips {
        if count <= max_files && bytes <= max_bytes {
            return;
        }
        bytes -= len;
        count -= 1;
        let _ = fs::remove_file(&clip);
        let _ = fs::remove_file(clip.with_extension("txt"));
    }
}

/// Write `pcm` as `<stem>.wav` and `sidecar` as `<stem>.txt` in `dir`, then
/// evict the oldest pairs past the caps. Nothing is written for empty audio.
pub fn save_clip(dir: &Path, stem: &str, pcm: &[i16], sidecar: &str, max_files: usize, max_bytes: u64) -> io::Result<()> {
    if pcm.is_empty() {
        return Ok(());
    }
    fs::create_dir_all(dir)?;
    fs::write(dir.join(format!("{stem}.wav")), wav_bytes(pcm))?;
    fs::write(dir.join(format!("{stem}.txt")), sidecar)?;
    evict_oldest(dir, max_files, max_bytes);
    Ok(())
}

/// Hard-negative candidates: the wake fired and nobody spoke. Safe retrain feed.
pub const WAKE_SILENT_DIR: &str = "wake-captures";
/// QUARANTINE: the wake fired but the transcript carried no name token. The
/// 2026-07-29 captures proved these are frequently genuine wakes that sherpa
/// misheard ("ARA", "Rind…"): training on them as negatives teaches the
/// model to ignore the user. A human reviews them before any clip in here
/// enters a retrain set, which is why they never share the silent directory.
pub const WAKE_REJECTED_DIR: &str = "wake-captures-rejected";
/// TRUE POSITIVES: accepted wakes. Kept apart from both directories above:
/// mixing confirmed positives into either the retrain feed or the quarantine
/// would poison the labels in the other direction.
pub const WAKE_ACCEPTED_DIR: &str = "wake-captures-accepted";
pub const WAKE_MAX_FILES: usize = 50;
pub const WAKE_MAX_BYTES: u64 = 20 * 1024 * 1024;

/// Larger than the wake caps: this records every turn, not the rare
/// containment path.
pub const UTTERANCE_DIR: &str = "utterance-captures";
pub const UTTERANCE_MAX_FILES: usize = 100;
pub const UTTERANCE_MAX_BYTES: u64 = 50 * 1024 * 1024;

/// The directory a wake clip caught by `hook` goes to. Exhaustive on purpose:
/// a new hook must decide its directory here, not silently inherit the
/// retrain feed.
pub fn wake_capture_dir(hook: WakeCaptureHook) -> &'static str {
    match hook {
        WakeCaptureHook::Rejected => WAKE_REJECTED_DIR,
        WakeCaptureHook::Silent | WakeCaptureHook::NoSpeech => WAKE_SILENT_DIR,
        WakeCaptureHook::Accepted => WAKE_ACCEPTED_DIR,
    }
}

/// Keep a wake's audio under `base`, with the transcript that decided it as
/// its sidecar.
pub fn save_wake_capture(base: &Path, pcm: &[i16], raw_transcript: &str, hook: WakeCaptureHook, timestamp_ms: i64) -> io::Result<PathBuf> {
    let dir = base.join(wake_capture_dir(hook));
    let stem = clip_stem("wake", timestamp_ms, hook.slug());
    save_clip(&dir, &stem, pcm, raw_transcript, WAKE_MAX_FILES, WAKE_MAX_BYTES)?;
    Ok(dir.join(format!("{stem}.wav")))
}

/// Keep what the user said under `base`, with every transcript it produced.
pub fn save_utterance_capture<Tz: TimeZone>(
    base: &Path,
    pcm: &[i16],
    capture: &UtteranceCapture,
    timestamp_ms: i64,
    zone: &Tz,
) -> io::Result<PathBuf> {
    let dir = base.join(UTTERANCE_DIR);
    let stem = clip_stem("utterance", timestamp_ms, capture.outcome.slug());
    save_clip(&dir, &stem, pcm, &utterance_sidecar(capture, timestamp_ms, zone), UTTERANCE_MAX_FILES, UTTERANCE_MAX_BYTES)?;
    Ok(dir.join(format!("{stem}.wav")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{FixedOffset, Utc};

    fn le(bytes: &[u8], at: usize, len: usize) -> i64 {
        match len {
            2 => i16::from_le_bytes([bytes[at], bytes[at + 1]]) as i64,
            _ => i32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]) as i64,
        }
    }

    #[test]
    fn the_wav_header_describes_16_bit_mono_16_khz_pcm() {
        let pcm: Vec<i16> = (0..800).collect();
        let out = wav_bytes(&pcm);
        assert_eq!(out.len(), 44 + 1600);
        assert_eq!(&out[0..4], b"RIFF");
        assert_eq!(le(&out, 4, 4), 36 + 1600);
        assert_eq!(&out[8..12], b"WAVE");
        assert_eq!(&out[12..16], b"fmt ");
        assert_eq!(le(&out, 16, 4), 16);
        assert_eq!(le(&out, 20, 2), 1);
        assert_eq!(le(&out, 22, 2), 1);
        assert_eq!(le(&out, 24, 4), 16000);
        assert_eq!(le(&out, 28, 4), 32000);
        assert_eq!(le(&out, 32, 2), 2);
        assert_eq!(le(&out, 34, 2), 16);
        assert_eq!(&out[36..40], b"data");
        assert_eq!(le(&out, 40, 4), 1600);
    }

    #[test]
    fn wav_samples_are_little_endian() {
        let out = wav_bytes(&[0, 1, -1, 258]);
        assert_eq!(le(&out, 44, 2), 0);
        assert_eq!(le(&out, 46, 2), 1);
        assert_eq!(le(&out, 48, 2), -1);
        assert_eq!(le(&out, 50, 2), 258);
    }

    fn wavs(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .filter(|n| n.ends_with(".wav"))
            .collect();
        names.sort();
        names
    }

    fn names(dir: &Path) -> Vec<String> {
        let Ok(entries) = fs::read_dir(dir) else { return Vec::new() };
        let mut names: Vec<String> = entries.map(|e| e.unwrap().file_name().into_string().unwrap()).collect();
        names.sort();
        names
    }

    #[test]
    fn eviction_drops_the_oldest_clips_past_the_file_cap() {
        let dir = tempfile::tempdir().unwrap();
        for i in 1..=5 {
            fs::write(dir.path().join(format!("wake-0000000000000000{i}-rejected.wav")), [0u8; 10]).unwrap();
            fs::write(dir.path().join(format!("wake-0000000000000000{i}-rejected.txt")), format!("clip {i}")).unwrap();
        }
        evict_oldest(dir.path(), 3, u64::MAX);
        assert_eq!(
            wavs(dir.path()),
            [
                "wake-00000000000000003-rejected.wav",
                "wake-00000000000000004-rejected.wav",
                "wake-00000000000000005-rejected.wav"
            ]
        );
        assert!(!names(dir.path()).iter().any(|n| n.starts_with("wake-00000000000000001")));
    }

    #[test]
    fn eviction_drops_the_oldest_clips_past_the_byte_cap() {
        let dir = tempfile::tempdir().unwrap();
        for i in 1..=4 {
            fs::write(dir.path().join(format!("wake-0000000000000000{i}-silent.wav")), [0u8; 100]).unwrap();
            fs::write(dir.path().join(format!("wake-0000000000000000{i}-silent.txt")), "").unwrap();
        }
        evict_oldest(dir.path(), usize::MAX, 250);
        assert_eq!(wavs(dir.path()), ["wake-00000000000000003-silent.wav", "wake-00000000000000004-silent.wav"]);
    }

    #[test]
    fn eviction_leaves_an_under_cap_directory_alone() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("wake-00000000000000001-rejected.wav"), [0u8; 10]).unwrap();
        fs::write(dir.path().join("wake-00000000000000001-rejected.txt"), "only one").unwrap();
        evict_oldest(dir.path(), 50, 20 * 1024 * 1024);
        assert_eq!(wavs(dir.path()).len(), 1);
        assert_eq!(fs::read_to_string(dir.path().join("wake-00000000000000001-rejected.txt")).unwrap(), "only one");
    }

    #[test]
    fn stems_are_fixed_width_so_lexicographic_order_is_chronological() {
        assert_eq!(clip_stem("utterance", 1_769_000_000, "rescued"), "utterance-00000001769000000-rescued");
        let older = clip_stem("wake", 999, "silent");
        let newer = clip_stem("wake", 1_000, "silent");
        assert_eq!(older, "wake-00000000000000999-silent");
        assert!(older < newer);
        assert_eq!(clip_stem("wake", 42, "rejected"), "wake-00000000000000042-rejected");
    }

    const PCM: [i16; 1600] = [100; 1600];

    #[test]
    fn rejected_clips_land_in_the_quarantine_and_silent_ones_in_the_retrain_feed() {
        let base = tempfile::tempdir().unwrap();
        save_wake_capture(base.path(), &PCM, "ARA reminds me", WakeCaptureHook::Rejected, 1_000).unwrap();
        save_wake_capture(base.path(), &PCM, "", WakeCaptureHook::Silent, 2_000).unwrap();
        assert_eq!(
            names(&base.path().join(WAKE_REJECTED_DIR)),
            ["wake-00000000000001000-rejected.txt", "wake-00000000000001000-rejected.wav"]
        );
        assert_eq!(
            names(&base.path().join(WAKE_SILENT_DIR)),
            ["wake-00000000000002000-silent.txt", "wake-00000000000002000-silent.wav"]
        );
        // The transcript that got the clip rejected travels with it, so a
        // human reviewing the quarantine can see what was heard.
        assert_eq!(
            fs::read_to_string(base.path().join(WAKE_REJECTED_DIR).join("wake-00000000000001000-rejected.txt")).unwrap(),
            "ARA reminds me"
        );
    }

    #[test]
    fn no_speech_clips_join_the_silent_ones_in_the_retrain_feed() {
        let base = tempfile::tempdir().unwrap();
        save_wake_capture(base.path(), &PCM, "", WakeCaptureHook::NoSpeech, 4_000).unwrap();
        assert_eq!(
            names(&base.path().join(WAKE_SILENT_DIR)),
            ["wake-00000000000004000-no-speech.txt", "wake-00000000000004000-no-speech.wav"]
        );
        assert!(names(&base.path().join(WAKE_REJECTED_DIR)).is_empty());
    }

    #[test]
    fn accepted_clips_land_in_their_own_directory() {
        let base = tempfile::tempdir().unwrap();
        let wav = save_wake_capture(base.path(), &PCM, "hey ari how s the weather", WakeCaptureHook::Accepted, 3_000).unwrap();
        assert_eq!(
            names(&base.path().join(WAKE_ACCEPTED_DIR)),
            ["wake-00000000000003000-accepted.txt", "wake-00000000000003000-accepted.wav"]
        );
        // 44 header bytes and 3,200 of samples.
        assert_eq!(fs::metadata(wav).unwrap().len(), 3_244);
        assert!(names(&base.path().join(WAKE_SILENT_DIR)).is_empty());
        assert!(names(&base.path().join(WAKE_REJECTED_DIR)).is_empty());
    }

    #[test]
    fn empty_audio_writes_nothing() {
        let base = tempfile::tempdir().unwrap();
        save_wake_capture(base.path(), &[], "", WakeCaptureHook::Silent, 1).unwrap();
        assert!(!base.path().join(WAKE_SILENT_DIR).exists());
    }

    // 2026-07-28T09:15:30Z
    const TIMESTAMP_MS: i64 = 1_785_230_130_000;

    fn rescued() -> UtteranceCapture {
        UtteranceCapture {
            turn: UtteranceTurn::Opening,
            outcome: UtteranceOutcome::Rescued,
            response: Some("action(dev.heyari.weather)".into()),
            raw: Some("hey ari whats the weather".into()),
            transcript: "whats the leather".into(),
            parallel: Some("whats the feather".into()),
            offline: Some("whats the weather".into()),
            used: "whats the weather".into(),
            locale: "en".into(),
            model: Some("kroko-en-int8".into()),
        }
    }

    #[test]
    fn a_rescued_turn_records_every_layer_that_had_a_go() {
        assert_eq!(
            utterance_sidecar(&rescued(), TIMESTAMP_MS, &Utc),
            "when: 2026-07-28T09:15:30Z\n\
             turn: opening\n\
             outcome: rescued\n\
             locale: en\n\
             model: kroko-en-int8\n\
             response: action(dev.heyari.weather)\n\
             raw: hey ari whats the weather\n\
             transcript: whats the leather\n\
             parallel: whats the feather\n\
             offline: whats the weather\n\
             used: whats the weather\n"
        );
    }

    #[test]
    fn the_timestamp_is_rendered_in_the_supplied_zone() {
        let malta_summer = FixedOffset::east_opt(2 * 3600).unwrap();
        let sidecar = utterance_sidecar(&rescued(), TIMESTAMP_MS, &malta_summer);
        assert_eq!(sidecar.lines().next(), Some("when: 2026-07-28T11:15:30+02:00"));
    }

    #[test]
    fn layers_that_produced_nothing_are_marked_not_left_blank() {
        let blank = UtteranceCapture {
            turn: UtteranceTurn::Reply,
            outcome: UtteranceOutcome::Blank,
            response: None,
            raw: None,
            transcript: String::new(),
            parallel: None,
            offline: None,
            used: String::new(),
            locale: "it".into(),
            model: None,
        };
        assert_eq!(
            utterance_sidecar(&blank, TIMESTAMP_MS, &Utc),
            "when: 2026-07-28T09:15:30Z\n\
             turn: reply\n\
             outcome: blank\n\
             locale: it\n\
             model: (none)\n\
             response: (none)\n\
             raw: (none)\n\
             transcript: (none)\n\
             parallel: (none)\n\
             offline: (none)\n\
             used: (none)\n"
        );
    }

    #[test]
    fn an_utterance_is_filed_under_its_outcome() {
        let base = tempfile::tempdir().unwrap();
        save_utterance_capture(base.path(), &PCM, &rescued(), TIMESTAMP_MS, &Utc).unwrap();
        assert_eq!(
            names(&base.path().join(UTTERANCE_DIR)),
            ["utterance-00001785230130000-rescued.txt", "utterance-00001785230130000-rescued.wav"]
        );
    }
}
