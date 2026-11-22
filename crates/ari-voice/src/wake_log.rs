//! Every wake, how it ended, and when the microphone was open, on this device.
//!
//! Exists because the app could not answer the question it most needed to: how
//! often does Ari wake for nothing? The recordings only ever caught some kinds
//! of false wake, and an overlay that closed on a transcription error left no
//! trace anywhere. One line per event, so a tester's bug report carries real
//! numbers rather than an impression.
//!
//! No audio and no transcripts, only times, model names and outcomes, so it is
//! kept whatever the recording settings say. It stays on the device except in
//! a bug report the user chooses to send.
//!
//! The lines are byte for byte what `WakeLog.kt` wrote: bug reports and
//! ari-tools read them, and a log written before the move reads the same.

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::fs::{self, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use chrono::{Local, Offset, TimeZone};
use sha2::{Digest, Sha256};

use crate::wake_model::{OperatingPoint, WakeSensitivity, WakeWordModel};

/// How a wake the user could see ended.
///
/// [`false_wake`](Self::false_wake) marks the endings where the overlay opened
/// and nobody had called Ari. Those are what the false-wake monitor counts and
/// what the log's hourly rate is made of. An error is not one of them: a
/// network failure can sink a request somebody really made.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum WakeOutcome {
    /// Passed the name check and went to the engine.
    Accepted,
    /// Speech, but no "Ari" in it.
    Rejected,
    /// The transcriber heard no speech at all: a knock, a buzz, a hum.
    NoSpeech,
    /// Nothing came back before the silence timeout.
    Silent,
    /// Transcription failed, or the turn itself did.
    Error,
    /// Closed before anything else happened, by a tap outside the overlay or by the system.
    Dismissed,
    /// The speech model was still loading, so the user was asked to say it again.
    ColdStart,
    /// The overlay could not be opened at all.
    NotShown,
}

impl WakeOutcome {
    /// Declaration order, which is the order a summary lists them in.
    pub const ALL: [Self; 8] = [
        Self::Accepted,
        Self::Rejected,
        Self::NoSpeech,
        Self::Silent,
        Self::Error,
        Self::Dismissed,
        Self::ColdStart,
        Self::NotShown,
    ];

    pub fn slug(self) -> &'static str {
        match self {
            Self::Accepted => "accepted",
            Self::Rejected => "rejected",
            Self::NoSpeech => "no-speech",
            Self::Silent => "silent",
            Self::Error => "error",
            Self::Dismissed => "dismissed",
            Self::ColdStart => "cold-start",
            Self::NotShown => "not-shown",
        }
    }

    pub fn false_wake(self) -> bool {
        matches!(self, Self::Rejected | Self::NoSpeech | Self::Silent)
    }

    pub fn from_slug(slug: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|o| o.slug() == slug)
    }
}

pub const LOG_LISTEN_ON: &str = "listen-on";
pub const LOG_LISTEN_OFF: &str = "listen-off";
pub const LOG_ALIVE: &str = "alive";
pub const LOG_WAKE: &str = "wake";
pub const LOG_OUTCOME: &str = "outcome";
pub const LOG_DROPPED: &str = "dropped";
pub const LOG_PASSED_AFTER_DROP: &str = "passed-after-drop";
pub const LOG_CHECK_FAILED: &str = "speech-check-failed";
pub const LOG_LISTEN_END: &str = "listen-end";
pub const LOG_EARLY_NAME: &str = "early-name";

/// How often a listening capture loop writes [`LOG_ALIVE`].
pub const ALIVE_INTERVAL_MS: i64 = 15 * 60 * 1000;

/// Big enough for about a month of heartbeats, which are most of the lines; a
/// wake-heavy month trims sooner, and the week a report reads is what matters.
const MAX_BYTES: u64 = 512 * 1024;

const DAY_MS: i64 = 24 * 60 * 60 * 1000;
const WEEK_MS: i64 = 7 * DAY_MS;

/// One line of the wake log: when, what, and the words that follow.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WakeLogEntry {
    pub at: i64,
    pub kind: String,
    pub args: Vec<String>,
}

pub fn format_wake_log_line(at: i64, kind: &str, args: &[&str]) -> String {
    let mut line = format!("{at} {kind}");
    for arg in args {
        line.push(' ');
        line.push_str(arg);
    }
    line
}

/// `None` for anything that is not a line this log wrote, a torn last line included.
pub fn parse_wake_log_line(line: &str) -> Option<WakeLogEntry> {
    // Kotlin's trim() and split(" "): every space separates, empty words kept.
    let parts: Vec<&str> = line.trim_matches(|c: char| c <= ' ').split(' ').collect();
    if parts.len() < 2 {
        return None;
    }
    let at = parts[0].parse().ok()?;
    Some(WakeLogEntry {
        at,
        kind: parts[1].to_string(),
        args: parts[2..].iter().map(|s| s.to_string()).collect(),
    })
}

/// What one wake model did over a stretch of the log.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelTally {
    pub listening_ms: i64,
    pub wakes: i32,
    pub outcomes: BTreeMap<WakeOutcome, i32>,
    /// Detections the speech check dropped before anything opened. Not wakes, not false wakes.
    pub dropped: i32,
    /// Drops whose episode was let through on a re-check after all: prevented nothing.
    pub passed_after_drop: i32,
    /// Wakes let through unchecked because the check could not load or threw.
    pub check_failed: i32,
    /// How quiet wakes ended that never showed anything. Not in `outcomes`.
    pub hidden: BTreeMap<WakeOutcome, i32>,
}

impl ModelTally {
    /// False wakes somebody saw. The ones a quiet wake hid are [`hidden_false_wakes`](Self::hidden_false_wakes).
    pub fn false_wakes(&self) -> i32 {
        self.outcomes.iter().filter(|(o, _)| o.false_wake()).map(|(_, n)| n).sum()
    }

    pub fn hidden_false_wakes(&self) -> i32 {
        self.hidden.iter().filter(|(o, _)| o.false_wake()).map(|(_, n)| n).sum()
    }

    /// Wakes with no outcome line: the overlay was replaced by another wake, or the app died.
    pub fn unresolved(&self) -> i32 {
        self.wakes - self.outcomes.values().sum::<i32>() - self.hidden.values().sum::<i32>()
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WakeLogSummary {
    pub by_model: BTreeMap<String, ModelTally>,
    /// Listening spells that never logged a stop, because the process was killed.
    pub unclosed: i32,
}

/// Tally the log between `from` and `to`, per wake model.
///
/// Listening time is what makes a false-wake count comparable between phones:
/// ten a day means one thing on a phone that listens all day and another on one
/// that listens for an hour. A spell that never logged a stop ended when its
/// process was killed, and is counted up to the last line written while it was
/// open, which the capture loop's heartbeat keeps within [`ALIVE_INTERVAL_MS`]
/// of the truth. Only the last spell can still be running, and `listening_now`
/// says whether it is.
///
/// An outcome belongs to the wake before it. The two are written from
/// different places, the capture loop and the voice session, and pairing them
/// by order is what keeps the voice session from needing to know which model
/// woke it.
pub fn summarise_wake_log(entries: &[WakeLogEntry], from: i64, to: i64, listening_now: bool) -> WakeLogSummary {
    let mut summary = WakeLogSummary::default();
    let mut open_model: Option<String> = None;
    let mut open_at = 0;
    let mut last_sign = 0;
    let mut pending_wake: Option<String> = None;
    let mut pending_in_window = false;
    // Kept apart: as in WakeLog.kt, a model these alone mention is not listed.
    let mut passed_after_drop: HashMap<String, i32> = HashMap::new();

    // True when the spell overlapped the window at all.
    let close = |summary: &mut WakeLogSummary, open_model: &mut Option<String>, open_at: i64, end: i64| -> bool {
        let Some(model) = open_model.take() else { return false };
        let overlap = end.min(to) - open_at.max(from);
        if overlap <= 0 {
            return false;
        }
        summary.by_model.entry(model).or_default().listening_ms += overlap;
        true
    };

    let mut sorted: Vec<&WakeLogEntry> = entries.iter().collect();
    sorted.sort_by_key(|e| e.at);
    for entry in sorted {
        let first_arg = entry.args.first().cloned();
        match entry.kind.as_str() {
            LOG_LISTEN_ON => {
                if close(&mut summary, &mut open_model, open_at, last_sign) {
                    summary.unclosed += 1;
                }
                open_model = first_arg;
                open_at = entry.at;
            }
            LOG_LISTEN_OFF => {
                close(&mut summary, &mut open_model, open_at, entry.at);
            }
            LOG_WAKE => {
                if let Some(model) = first_arg {
                    pending_in_window = (from..=to).contains(&entry.at);
                    if pending_in_window {
                        summary.by_model.entry(model.clone()).or_default().wakes += 1;
                    }
                    pending_wake = Some(model);
                }
            }
            LOG_DROPPED | LOG_PASSED_AFTER_DROP | LOG_CHECK_FAILED => {
                if let Some(model) = first_arg
                    && (from..=to).contains(&entry.at)
                {
                    match entry.kind.as_str() {
                        LOG_DROPPED => summary.by_model.entry(model).or_default().dropped += 1,
                        LOG_PASSED_AFTER_DROP => *passed_after_drop.entry(model).or_default() += 1,
                        _ => summary.by_model.entry(model).or_default().check_failed += 1,
                    }
                }
            }
            LOG_OUTCOME => {
                if let Some(model) = pending_wake.take()
                    && let Some(outcome) = first_arg.as_deref().and_then(WakeOutcome::from_slug)
                    && pending_in_window
                {
                    let tally = summary.by_model.entry(model).or_default();
                    let into = if entry.args.get(1).map(String::as_str) == Some("hidden") {
                        &mut tally.hidden
                    } else {
                        &mut tally.outcomes
                    };
                    *into.entry(outcome).or_default() += 1;
                }
            }
            _ => {}
        }
        if open_model.is_some() {
            last_sign = entry.at;
        }
    }
    let end = if listening_now { to } else { last_sign };
    if close(&mut summary, &mut open_model, open_at, end) && !listening_now {
        summary.unclosed += 1;
    }
    for (model, count) in passed_after_drop {
        if let Some(tally) = summary.by_model.get_mut(&model) {
            tally.passed_after_drop = count;
        }
    }
    summary
}

/// `summary` as the lines a bug report carries, under a heading naming `span`.
pub fn render_wake_log_summary(span: &str, summary: &WakeLogSummary) -> String {
    let mut out = format!("--------- wake log, {span}\n");
    if summary.by_model.is_empty() {
        out.push_str("nothing recorded\n");
        return out;
    }
    for (model, tally) in &summary.by_model {
        let hours = tally.listening_ms as f64 / 3_600_000.0;
        let _ = write!(
            out,
            "{model}: listening {hours:.1} h, {} wakes, {} false",
            tally.wakes,
            tally.false_wakes()
        );
        if tally.listening_ms > 0 {
            let _ = write!(out, " ({:.2} per listening hour)", tally.false_wakes() as f64 / hours);
        }
        out.push('\n');
        let mut parts: Vec<String> = WakeOutcome::ALL
            .iter()
            .filter_map(|o| tally.outcomes.get(o).map(|n| format!("{} {n}", o.slug())))
            .collect();
        if tally.unresolved() > 0 {
            parts.push(format!("no outcome {}", tally.unresolved()));
        }
        if tally.dropped > 0 {
            parts.push(format!("dropped for no speech {}", tally.dropped));
        }
        if tally.passed_after_drop > 0 {
            parts.push(format!("of those, let through on a re-check {}", tally.passed_after_drop));
        }
        if tally.check_failed > 0 {
            parts.push(format!("let through unchecked {}", tally.check_failed));
        }
        if !parts.is_empty() {
            let _ = writeln!(out, "  {}", parts.join(", "));
        }
        let quiet: Vec<String> = WakeOutcome::ALL
            .iter()
            .filter_map(|o| tally.hidden.get(o).map(|n| format!("{} {n}", o.slug())))
            .collect();
        if !quiet.is_empty() {
            let _ = writeln!(out, "  never shown (quiet wake): {}", quiet.join(", "));
        }
    }
    if summary.unclosed > 0 {
        let _ = writeln!(
            out,
            "{} listening spells ended without a stop (app killed), counted to their last line",
            summary.unclosed
        );
    }
    out
}

/// `f` as Kotlin's `Float.toString()` writes it, which is what the log has
/// always held: the shortest form that reads back as `f`, with a `.0` on whole
/// numbers.
fn kotlin_float(f: f32) -> String {
    let s = f.to_string();
    if s.contains(['.', 'e', 'i', 'N']) { s } else { s + ".0" }
}

/// `at` as Java's `ISO_OFFSET_DATE_TIME` writes it: seconds always, the
/// fraction only when there is one and without trailing zeros, `Z` for UTC.
fn iso_offset_date_time<Tz: TimeZone>(at: i64, zone: &Tz) -> String {
    let Some(time) = zone.timestamp_millis_opt(at).single() else { return at.to_string() };
    let mut out = time.naive_local().format("%Y-%m-%dT%H:%M:%S").to_string();
    let millis = at.rem_euclid(1000);
    if millis != 0 {
        out.push_str(format!(".{millis:03}").trim_end_matches('0'));
    }
    let offset = time.offset().fix().local_minus_utc();
    if offset == 0 {
        out.push('Z');
    } else {
        let sign = if offset < 0 { '-' } else { '+' };
        let abs = offset.abs();
        let _ = write!(out, "{sign}{:02}:{:02}", abs / 3600, abs % 3600 / 60);
        if abs % 60 != 0 {
            let _ = write!(out, ":{:02}", abs % 60);
        }
    }
    out
}

/// The first 8 hex digits of `model`'s SHA-256, so a result can be traced to
/// exactly the file that ran.
pub fn model_fingerprint(model: &[u8]) -> String {
    Sha256::digest(model)[..4].iter().map(|b| format!("{b:02x}")).collect()
}

fn now_ms() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis() as i64)
}

/// The log file. Writes go to a thread of their own and are timestamped at the
/// call, so callers on the audio thread and the UI thread never wait on the
/// disk, and the reader sorts by time rather than trusting the order lines
/// landed in.
pub struct WakeLog {
    file: Arc<LogFile>,
    writer: Mutex<Sender<String>>,
}

struct LogFile {
    path: PathBuf,
    lock: Mutex<()>,
}

impl LogFile {
    fn append(&self, line: &str) {
        let _guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        let result = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .and_then(|mut f| f.write_all(format!("{line}\n").as_bytes()));
        if let Err(e) = result {
            eprintln!("could not write the wake log: {e}");
            return;
        }
        if fs::metadata(&self.path).is_ok_and(|m| m.len() > MAX_BYTES) {
            self.trim();
        }
    }

    /// Keep the newest half. Written beside the log and renamed over it, so a
    /// process killed halfway through leaves the old log rather than none.
    fn trim(&self) {
        let Ok(text) = fs::read_to_string(&self.path) else { return };
        let mut kept = 0u64;
        let mut tail: Vec<&str> = text
            .lines()
            .rev()
            .take_while(|l| {
                kept += l.len() as u64 + 1;
                kept <= MAX_BYTES / 2
            })
            .collect();
        tail.reverse();
        let next = PathBuf::from(format!("{}.tmp", self.path.display()));
        let body: String = tail.iter().map(|l| format!("{l}\n")).collect();
        if fs::write(&next, body).and_then(|_| fs::rename(&next, &self.path)).is_err() {
            eprintln!("could not replace the wake log after trimming");
        }
    }

    fn entries(&self) -> Vec<WakeLogEntry> {
        let _guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        match fs::read_to_string(&self.path) {
            Ok(text) => text.lines().filter_map(parse_wake_log_line).collect(),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(e) => {
                eprintln!("could not read the wake log: {e}");
                Vec::new()
            }
        }
    }
}

impl WakeLog {
    pub fn new(path: impl AsRef<Path>) -> Self {
        let file = Arc::new(LogFile { path: path.as_ref().to_path_buf(), lock: Mutex::new(()) });
        let (sender, receiver) = mpsc::channel::<String>();
        let writer = Arc::clone(&file);
        std::thread::Builder::new()
            .name("AriWakeLog".into())
            .spawn(move || {
                for line in receiver {
                    writer.append(&line);
                }
            })
            .expect("the wake log's writer thread");
        Self { file, writer: Mutex::new(sender) }
    }

    fn record(&self, kind: &str, args: &[&str]) {
        let line = format_wake_log_line(now_ms(), kind, args);
        let _ = self.writer.lock().unwrap_or_else(|e| e.into_inner()).send(line);
    }

    /// `fingerprint` is [`model_fingerprint`] of the model file.
    pub fn listening_started(
        &self,
        model: &WakeWordModel,
        sensitivity: WakeSensitivity,
        point: OperatingPoint,
        speech_check: bool,
        quiet_wake: bool,
        fingerprint: &str,
    ) {
        self.record(
            LOG_LISTEN_ON,
            &[
                &model.id,
                sensitivity.name(),
                &kotlin_float(point.probability_cutoff),
                &point.sliding_window_size.to_string(),
                if speech_check { "speech-check=on" } else { "speech-check=off" },
                if quiet_wake { "quiet-wake=on" } else { "quiet-wake=off" },
                &format!("file={fingerprint}"),
            ],
        );
    }

    pub fn listening_stopped(&self) {
        self.record(LOG_LISTEN_OFF, &[]);
    }

    /// The heartbeat that bounds a spell whose process is killed before it can say so.
    pub fn still_listening(&self) {
        self.record(LOG_ALIVE, &[]);
    }

    pub fn wake(&self, model_id: &str) {
        self.record(LOG_WAKE, &[model_id]);
    }

    /// `hidden`: a quiet wake that ended without ever showing anything.
    pub fn outcome(&self, outcome: WakeOutcome, hidden: bool) {
        if hidden {
            self.record(LOG_OUTCOME, &[outcome.slug(), "hidden"]);
        } else {
            self.record(LOG_OUTCOME, &[outcome.slug()]);
        }
    }

    /// A detection the speech check dropped: no overlay, no chime, only this line.
    pub fn dropped(&self, model_id: &str) {
        self.record(LOG_DROPPED, &[model_id]);
    }

    /// The check found speech in an episode it had already dropped once.
    pub fn passed_after_drop(&self, model_id: &str) {
        self.record(LOG_PASSED_AFTER_DROP, &[model_id]);
    }

    /// A wake let through because the check could not run: `reason` is
    /// `unavailable` (the model would not load) or `error` (it failed). Kept
    /// apart from drops so a trial can show which protection actually ran.
    pub fn speech_check_failed(&self, model_id: &str, reason: &str) {
        self.record(LOG_CHECK_FAILED, &[model_id, reason]);
    }

    /// How a wake's listen ended (`end` is a `ListenEnd` slug) and `after_ms`
    /// from its start, written just before the wake's outcome.
    pub fn listen_ended(&self, end: &str, after_ms: i64) {
        self.record(LOG_LISTEN_END, &[end, &after_ms.to_string()]);
    }

    /// A quiet wake's early name check: `result` is an `EarlyNameCheck` slug,
    /// `after_ms` how long it took, and `late` follows when the wake had
    /// already been decided, so the result changed nothing.
    pub fn early_name_check(&self, result: &str, after_ms: i64, in_time: bool) {
        let after = after_ms.to_string();
        if in_time {
            self.record(LOG_EARLY_NAME, &[result, &after]);
        } else {
            self.record(LOG_EARLY_NAME, &[result, &after, "late"]);
        }
    }

    /// Write `line` now, on this thread. For tests.
    pub fn append(&self, line: &str) {
        self.file.append(line);
    }

    pub fn entries(&self) -> Vec<WakeLogEntry> {
        self.file.entries()
    }

    /// The last day and week summarised, then the week's lines in full for
    /// anything the summary did not think to ask, in the device's time zone.
    pub fn report(&self, now: i64, listening_now: bool) -> String {
        self.report_in(now, listening_now, &Local)
    }

    /// [`report`](Self::report) in `zone`. The heartbeat lines are left out of
    /// the listing: the summary has already used them.
    pub fn report_in<Tz: TimeZone>(&self, now: i64, listening_now: bool, zone: &Tz) -> String {
        let entries = self.entries();
        let week_ago = now - WEEK_MS;
        let mut out = render_wake_log_summary(
            "last 24 hours",
            &summarise_wake_log(&entries, now - DAY_MS, now, listening_now),
        );
        out += &render_wake_log_summary("last 7 days", &summarise_wake_log(&entries, week_ago, now, listening_now));
        out.push_str("--------- wake log lines, last 7 days\n");
        let mut listed: Vec<&WakeLogEntry> =
            entries.iter().filter(|e| e.at >= week_ago && e.kind != LOG_ALIVE).collect();
        listed.sort_by_key(|e| e.at);
        for entry in listed {
            let _ = write!(out, "{} {}", iso_offset_date_time(entry.at, zone), entry.kind);
            for arg in &entry.args {
                out.push(' ');
                out.push_str(arg);
            }
            out.push('\n');
        }
        out.push('\n');
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    const HOUR: i64 = 3_600_000;

    fn entry(at: i64, kind: &str, args: &[&str]) -> WakeLogEntry {
        WakeLogEntry { at, kind: kind.into(), args: args.iter().map(|s| s.to_string()).collect() }
    }
    fn on(at: i64) -> WakeLogEntry {
        on_model(at, "hey_ari")
    }
    fn on_model(at: i64, model: &str) -> WakeLogEntry {
        entry(at, LOG_LISTEN_ON, &[model, "MEDIUM", "0.93", "10"])
    }
    fn off(at: i64) -> WakeLogEntry {
        entry(at, LOG_LISTEN_OFF, &[])
    }
    fn alive(at: i64) -> WakeLogEntry {
        entry(at, LOG_ALIVE, &[])
    }
    fn wake(at: i64) -> WakeLogEntry {
        wake_model(at, "hey_ari")
    }
    fn wake_model(at: i64, model: &str) -> WakeLogEntry {
        entry(at, LOG_WAKE, &[model])
    }
    fn outcome(at: i64, outcome: WakeOutcome) -> WakeLogEntry {
        entry(at, LOG_OUTCOME, &[outcome.slug()])
    }
    fn ari(summary: &WakeLogSummary) -> &ModelTally {
        &summary.by_model["hey_ari"]
    }
    fn counts(pairs: &[(WakeOutcome, i32)]) -> BTreeMap<WakeOutcome, i32> {
        pairs.iter().cloned().collect()
    }

    #[test]
    fn listening_time_is_clipped_to_the_window() {
        let entries = [on(0), off(10 * HOUR), on(20 * HOUR), off(22 * HOUR)];
        let summary = summarise_wake_log(&entries, 5 * HOUR, 21 * HOUR, false);
        assert_eq!(6 * HOUR, ari(&summary).listening_ms);
        assert_eq!(0, summary.unclosed);
    }

    #[test]
    fn a_spell_its_process_never_closed_ends_at_its_last_line() {
        // Killed at some point after the 3h heartbeat; the next process starts at 5h.
        let entries = [on(0), alive(HOUR), alive(3 * HOUR), on(5 * HOUR), off(6 * HOUR)];
        let summary = summarise_wake_log(&entries, 0, 10 * HOUR, false);
        assert_eq!(4 * HOUR, ari(&summary).listening_ms);
        assert_eq!(1, summary.unclosed);
    }

    #[test]
    fn the_last_spell_runs_to_now_while_the_microphone_is_still_open() {
        let entries = [on(0), alive(HOUR)];
        let listening = summarise_wake_log(&entries, 0, 4 * HOUR, true);
        let killed = summarise_wake_log(&entries, 0, 4 * HOUR, false);
        assert_eq!(4 * HOUR, ari(&listening).listening_ms);
        assert_eq!(0, listening.unclosed);
        assert_eq!(HOUR, ari(&killed).listening_ms);
        assert_eq!(1, killed.unclosed);
    }

    #[test]
    fn each_outcome_belongs_to_the_wake_before_it_per_model() {
        let entries = [
            on_model(0, "hey_ari"),
            wake_model(10, "hey_ari"),
            outcome(11, WakeOutcome::NoSpeech),
            wake_model(20, "hey_ari"),
            outcome(21, WakeOutcome::Accepted),
            // Replaced by another wake before it ended: no outcome of its own.
            wake_model(30, "hey_ari"),
            wake_model(40, "hey_ari"),
            outcome(41, WakeOutcome::Rejected),
            off(50),
            on_model(60, "hey_jarvis"),
            wake_model(70, "hey_jarvis"),
            outcome(71, WakeOutcome::Silent),
            off(80),
        ];
        let summary = summarise_wake_log(&entries, 0, 100, false);
        let a = ari(&summary);
        assert_eq!(4, a.wakes);
        assert_eq!(
            counts(&[(WakeOutcome::NoSpeech, 1), (WakeOutcome::Accepted, 1), (WakeOutcome::Rejected, 1)]),
            a.outcomes
        );
        assert_eq!(2, a.false_wakes());
        assert_eq!(1, a.unresolved());
        let jarvis = &summary.by_model["hey_jarvis"];
        assert_eq!(1, jarvis.wakes);
        assert_eq!(1, jarvis.false_wakes());
    }

    #[test]
    fn dropped_detections_are_counted_apart_and_never_as_wakes_or_false_wakes() {
        let entries = [
            on(0),
            entry(10, LOG_DROPPED, &["hey_ari"]),
            entry(20, LOG_DROPPED, &["hey_ari"]),
            wake(30),
            outcome(31, WakeOutcome::Accepted),
            entry(200, LOG_DROPPED, &["hey_ari"]),
        ];
        let summary = summarise_wake_log(&entries, 0, 100, true);
        let a = ari(&summary);
        assert_eq!(2, a.dropped);
        assert_eq!(0, a.passed_after_drop);
        assert_eq!(1, a.wakes);
        assert_eq!(0, a.false_wakes());
        assert_eq!(0, a.unresolved());
    }

    #[test]
    fn drops_let_through_later_and_checks_that_could_not_run_are_counted_apart() {
        let entries = [
            on(0),
            entry(10, LOG_DROPPED, &["hey_ari"]),
            entry(12, LOG_PASSED_AFTER_DROP, &["hey_ari"]),
            wake(12),
            outcome(14, WakeOutcome::Accepted),
            entry(20, LOG_CHECK_FAILED, &["hey_ari", "error"]),
            wake(20),
            outcome(22, WakeOutcome::NoSpeech),
        ];
        let summary = summarise_wake_log(&entries, 0, 100, true);
        let a = ari(&summary);
        assert_eq!(1, a.dropped);
        assert_eq!(1, a.passed_after_drop);
        assert_eq!(1, a.check_failed);
        assert_eq!(2, a.wakes);
        assert_eq!(1, a.false_wakes());
    }

    #[test]
    fn quiet_wakes_that_never_showed_are_not_false_wakes_anyone_saw() {
        let entries = [
            on(0),
            wake(10),
            entry(12, LOG_OUTCOME, &["rejected", "hidden"]),
            wake(20),
            entry(22, LOG_OUTCOME, &["no-speech", "hidden"]),
            wake(30),
            outcome(32, WakeOutcome::Accepted),
            wake(40),
            outcome(42, WakeOutcome::Rejected),
        ];
        let summary = summarise_wake_log(&entries, 0, 100, true);
        let a = ari(&summary);
        assert_eq!(4, a.wakes);
        assert_eq!(1, a.false_wakes());
        assert_eq!(2, a.hidden_false_wakes());
        assert_eq!(0, a.unresolved());
        let shown = WakeLogSummary {
            by_model: [("hey_ari".to_string(), ModelTally { listening_ms: 10 * HOUR, ..a.clone() })].into(),
            unclosed: 0,
        };
        assert_eq!(
            "--------- wake log, x\n\
             hey_ari: listening 10.0 h, 4 wakes, 1 false (0.10 per listening hour)\n  \
             accepted 1, rejected 1\n  \
             never shown (quiet wake): rejected 1, no-speech 1\n",
            render_wake_log_summary("x", &shown)
        );
    }

    #[test]
    fn listen_end_and_early_name_lines_between_a_wake_and_its_outcome_change_no_count() {
        let entries = [
            on(0),
            wake(10),
            entry(11, LOG_EARLY_NAME, &["pass", "640"]),
            entry(12, LOG_LISTEN_END, &["endpoint", "1860"]),
            outcome(13, WakeOutcome::Accepted),
            wake(20),
            entry(22, LOG_LISTEN_END, &["give-up", "3000"]),
            entry(22, LOG_OUTCOME, &["no-speech", "hidden"]),
        ];
        let summary = summarise_wake_log(&entries, 0, 100, true);
        let a = ari(&summary);
        assert_eq!(2, a.wakes);
        assert_eq!(counts(&[(WakeOutcome::Accepted, 1)]), a.outcomes);
        assert_eq!(1, a.hidden_false_wakes());
        assert_eq!(0, a.unresolved());
    }

    #[test]
    fn a_wake_before_the_window_takes_its_outcome_with_it() {
        let entries = [wake(10), outcome(20, WakeOutcome::Rejected), wake(30), outcome(31, WakeOutcome::Accepted)];
        let summary = summarise_wake_log(&entries, 15, 100, false);
        assert_eq!(1, ari(&summary).wakes);
        assert_eq!(counts(&[(WakeOutcome::Accepted, 1)]), ari(&summary).outcomes);
    }

    #[test]
    fn only_the_visible_endings_with_nobody_calling_count_as_false_wakes() {
        let false_wakes: Vec<WakeOutcome> = WakeOutcome::ALL.into_iter().filter(|o| o.false_wake()).collect();
        assert_eq!(vec![WakeOutcome::Rejected, WakeOutcome::NoSpeech, WakeOutcome::Silent], false_wakes);
    }

    #[test]
    fn lines_round_trip_and_anything_else_is_skipped() {
        let line = format_wake_log_line(1_234, LOG_LISTEN_ON, &["hey_ari", "LOW", "0.95", "10"]);
        assert_eq!("1234 listen-on hey_ari LOW 0.95 10", line);
        assert_eq!(Some(entry(1_234, LOG_LISTEN_ON, &["hey_ari", "LOW", "0.95", "10"])), parse_wake_log_line(&line));
        assert_eq!(None, parse_wake_log_line("12"));
        assert_eq!(None, parse_wake_log_line("not-a-time wake hey_ari"));
    }

    #[test]
    fn the_summary_reads_as_counts_per_model_and_an_hourly_rate() {
        let summary = WakeLogSummary {
            by_model: [(
                "hey_ari".to_string(),
                ModelTally {
                    listening_ms: 8 * HOUR,
                    wakes: 5,
                    outcomes: counts(&[(WakeOutcome::Accepted, 1), (WakeOutcome::NoSpeech, 3)]),
                    dropped: 7,
                    ..Default::default()
                },
            )]
            .into(),
            unclosed: 2,
        };
        assert_eq!(
            "--------- wake log, last day\n\
             hey_ari: listening 8.0 h, 5 wakes, 3 false (0.38 per listening hour)\n  \
             accepted 1, no-speech 3, no outcome 1, dropped for no speech 7\n\
             2 listening spells ended without a stop (app killed), counted to their last line\n",
            render_wake_log_summary("last day", &summary)
        );
        assert_eq!(
            "--------- wake log, last day\nnothing recorded\n",
            render_wake_log_summary("last day", &WakeLogSummary::default())
        );
    }

    #[test]
    fn a_report_summarises_the_log_and_lists_it_without_the_heartbeats() {
        let dir = tempfile::tempdir().unwrap();
        let log = WakeLog::new(dir.path().join("wake-log.txt"));
        let now = 10 * HOUR;
        log.append(&format_wake_log_line(now - 2 * HOUR, LOG_LISTEN_ON, &["hey_ari", "MEDIUM", "0.93", "10"]));
        log.append(&format_wake_log_line(now - HOUR, LOG_ALIVE, &[]));
        log.append(&format_wake_log_line(now - 30 * 60_000, LOG_WAKE, &["hey_ari"]));
        log.append(&format_wake_log_line(now - 30 * 60_000 + 3_000, LOG_OUTCOME, &["no-speech"]));
        assert_eq!(
            "--------- wake log, last 24 hours\n\
             hey_ari: listening 2.0 h, 1 wakes, 1 false (0.50 per listening hour)\n  \
             no-speech 1\n\
             --------- wake log, last 7 days\n\
             hey_ari: listening 2.0 h, 1 wakes, 1 false (0.50 per listening hour)\n  \
             no-speech 1\n\
             --------- wake log lines, last 7 days\n\
             1970-01-01T08:00:00Z listen-on hey_ari MEDIUM 0.93 10\n\
             1970-01-01T09:30:00Z wake hey_ari\n\
             1970-01-01T09:30:03Z outcome no-speech\n\
             \n",
            log.report_in(now, true, &Utc)
        );
    }

    #[test]
    fn a_log_past_its_cap_keeps_its_newest_lines() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wake-log.txt");
        let log = WakeLog::new(&path);
        // Each line is 26 bytes with its newline, so about 20,200 fill 512 KB.
        for i in 0..25_000 {
            log.append(&format_wake_log_line(1_000_000_000 + i, LOG_WAKE, &["hey_ari_x"]));
        }
        let entries = log.entries();
        assert!(fs::metadata(&path).unwrap().len() <= 512 * 1024);
        assert_eq!(1_000_000_000 + 24_999, entries.last().unwrap().at);
        // Contiguous: the trim drops from the front and nowhere else.
        let first = entries.first().unwrap().at;
        assert!(entries.iter().enumerate().all(|(i, e)| e.at == first + i as i64));
    }

    #[test]
    fn a_models_fingerprint_is_the_start_of_its_sha256() {
        // SHA-256("abc") begins ba7816bf.
        assert_eq!("ba7816bf", model_fingerprint(b"abc"));
    }

    #[test]
    fn times_and_cutoffs_are_written_as_kotlin_wrote_them() {
        assert_eq!("0.55", kotlin_float(0.55));
        assert_eq!("0.93", kotlin_float(0.93));
        assert_eq!("1.0", kotlin_float(1.0));
        assert_eq!("1970-01-01T00:00:00.877Z", iso_offset_date_time(877, &Utc));
        assert_eq!("1970-01-01T00:00:00.5Z", iso_offset_date_time(500, &Utc));
        let bst = chrono::FixedOffset::east_opt(3600).unwrap();
        assert_eq!("1970-01-01T01:00:01.12+01:00", iso_offset_date_time(1_120, &bst));
    }

    #[test]
    fn writes_land_in_order_on_the_writer_thread() {
        let dir = tempfile::tempdir().unwrap();
        let log = WakeLog::new(dir.path().join("wake-log.txt"));
        log.wake("hey_ari_oww");
        log.outcome(WakeOutcome::NoSpeech, true);
        log.early_name_check("pass", 804, false);
        for _ in 0..200 {
            if log.entries().len() == 3 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let lines: Vec<(String, Vec<String>)> = log.entries().into_iter().map(|e| (e.kind, e.args)).collect();
        assert_eq!(
            vec![
                ("wake".to_string(), vec!["hey_ari_oww".to_string()]),
                ("outcome".into(), vec!["no-speech".into(), "hidden".into()]),
                ("early-name".into(), vec!["pass".into(), "804".into(), "late".into()]),
            ],
            lines
        );
    }
}
