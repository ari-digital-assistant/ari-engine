//! Host-provided platform capabilities.
//!
//! Skills reach these through WASM host imports (`ari::tasks_*`,
//! `ari::calendar_*`, `ari::local_now_components`, …). The engine
//! doesn't implement any of them — frontends do. On Android the
//! implementations are content-provider wrappers; on Linux desktop
//! they'll be EDS bindings; on the CLI engine they default to the
//! `Null*` no-op impls in this module.
//!
//! One trait per OS subsystem, named for the capability the host
//! wraps rather than the skill that happens to use it first. A
//! reminder skill, a shopping-list skill, a recipe skill, a
//! meeting-scheduler — anything that reads or writes tasks / calendar
//! events — all go through the same trait. The skill author only
//! writes WASM; the frontend author only writes platform wrappers.
//!
//! Additive: new capabilities join this module without touching
//! existing skills. Removing a trait is a breaking change to the
//! skill ABI and must be treated as such.

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Debug;

// `alloc` is used so downstream `no_std`-ish crates can still consume
// these types; the engine workspace itself is std, but keeping the
// import shape consistent with the rest of the crate makes re-using
// these types in skill-side code cleaner.
extern crate alloc;

// ── Tasks ──────────────────────────────────────────────────────────────

/// One row from the host's task-list table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskList {
    /// Opaque provider id, stable across calls within a session.
    pub id: u64,
    pub display_name: String,
    /// Disambiguator when two lists share a display name (e.g. "Personal"
    /// under two CalDAV accounts). Empty string if the host has no such
    /// concept.
    pub account_name: String,
}

/// Parameters for [`TasksProvider::insert`]. Separate struct rather than
/// positional args so future fields don't break the ABI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InsertTaskParams {
    pub list_id: u64,
    pub title: String,
    /// UTC epoch ms for the due datetime. `None` for untimed tasks
    /// (shopping-list use case).
    pub due_ms: Option<i64>,
    /// When true, `due_ms` is interpreted as a wall-clock date and the
    /// time portion is discarded by the provider. When false, it's a
    /// precise instant.
    pub due_all_day: bool,
    /// IANA timezone id (e.g. `"Europe/London"`). Required by some
    /// providers (OpenTasks / Tasks.org) when `due_ms` is an instant;
    /// ignored for all-day tasks.
    pub tz_id: Option<String>,
}

/// One row from a `query_in_range` lookup. Strict subset of the
/// info `insert` puts in: enough to render a "what's on my list"
/// summary, not enough to round-trip back into another insert. If a
/// future skill needs richer data (notes, recurrence info, …) it goes
/// on a sibling row type, not this one — kept narrow so the host
/// query stays cheap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskRow {
    /// Provider row id (same identifier `delete` accepts).
    pub id: u64,
    pub title: String,
    /// UTC epoch ms for the task's due time.
    pub due_ms: i64,
    /// True when only the date portion of `due_ms` is meaningful.
    pub due_all_day: bool,
    pub list_id: u64,
}

/// Host-supplied access to the platform's task provider. On Android
/// this wraps the OpenTasks `ContentResolver`; on Linux it'll wrap EDS.
pub trait TasksProvider: Send + Sync {
    /// Is any tasks provider available on this host right now? False
    /// means the host has no suitable backend installed — the skill
    /// should degrade gracefully rather than calling other methods.
    fn is_provider_installed(&self) -> bool;

    /// All user-visible task lists the skill can write to.
    fn list_lists(&self) -> Vec<TaskList>;

    /// Insert a task. Returns the provider's row id on success; `None`
    /// on permission failure / invalid list / IO error. Failures are
    /// expected to be logged by the host-side impl.
    fn insert(&self, params: InsertTaskParams) -> Option<u64>;

    /// Hard-delete a task by its row id. Returns true if the row
    /// existed and was removed.
    fn delete(&self, id: u64) -> bool;

    /// Tasks with a due time in `[start_ms, end_ms)`, ordered by due
    /// time ascending and capped at `limit` rows. Untimed tasks (no
    /// due date) are deliberately excluded — they don't fit a date-
    /// range query. Returns an empty Vec when the provider isn't
    /// installed or the range is empty.
    fn query_in_range(
        &self,
        start_ms: i64,
        end_ms: i64,
        limit: u32,
    ) -> Vec<TaskRow>;
}

/// No-op default used by the CLI engine and tests. Reports no provider
/// available and drops every write.
pub struct NullTasksProvider;

impl TasksProvider for NullTasksProvider {
    fn is_provider_installed(&self) -> bool {
        false
    }
    fn list_lists(&self) -> Vec<TaskList> {
        Vec::new()
    }
    fn insert(&self, _params: InsertTaskParams) -> Option<u64> {
        None
    }
    fn delete(&self, _id: u64) -> bool {
        false
    }
    fn query_in_range(&self, _start_ms: i64, _end_ms: i64, _limit: u32) -> Vec<TaskRow> {
        Vec::new()
    }
}

// ── Calendar ───────────────────────────────────────────────────────────

/// One writable calendar the user can target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Calendar {
    pub id: u64,
    pub display_name: String,
    pub account_name: String,
    /// Calendar colour as ARGB (Android's native format). `None` when
    /// the host doesn't expose one.
    pub color_argb: Option<i32>,
}

/// Parameters for [`CalendarProvider::insert`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InsertCalendarEventParams {
    pub calendar_id: u64,
    pub title: String,
    /// UTC epoch ms the event starts at.
    pub start_ms: i64,
    /// Event length. Providers usually require a non-zero duration.
    pub duration_minutes: u32,
    /// Reminder offset — the event fires a notification this many
    /// minutes before `start_ms`. 0 = no reminder; the provider still
    /// stores the event.
    pub reminder_minutes_before: u32,
    /// IANA timezone id for the provider's `EVENT_TIMEZONE` field.
    pub tz_id: String,
}

/// One event instance from a `query_in_range` lookup. Recurring
/// events expand into multiple rows — one per concrete instance whose
/// start falls within the queried window — matching the way
/// `CalendarContract.Instances` works on Android.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalendarEventRow {
    pub id: u64,
    pub title: String,
    /// UTC epoch ms the instance starts at.
    pub start_ms: i64,
    /// UTC epoch ms the instance ends at. May equal `start_ms` for
    /// providers that don't store a duration.
    pub end_ms: i64,
    pub all_day: bool,
    pub calendar_id: u64,
}

/// Host-supplied access to the platform's calendar provider.
pub trait CalendarProvider: Send + Sync {
    /// Has the host been granted WRITE access? Implementations that
    /// use a runtime permission (Android `WRITE_CALENDAR`) return the
    /// live permission state; implementations that can't fail to
    /// write (tests, a hypothetical user-owned store) return true.
    fn has_write_permission(&self) -> bool;

    /// All writable calendars the skill can target.
    fn list_calendars(&self) -> Vec<Calendar>;

    /// Insert an event + its reminder row. Returns the event id on
    /// success, `None` otherwise.
    fn insert(&self, params: InsertCalendarEventParams) -> Option<u64>;

    /// Delete an event by id. Cascades to the provider's reminder
    /// rows where applicable (CalendarContract does this
    /// automatically). Returns true if the row existed.
    fn delete(&self, id: u64) -> bool;

    /// Event instances starting in `[start_ms, end_ms)`, ordered by
    /// start time ascending and capped at `limit`. Recurring events
    /// expand to one row per instance in range. Returns an empty Vec
    /// when read permission is missing or the range is empty.
    fn query_in_range(
        &self,
        start_ms: i64,
        end_ms: i64,
        limit: u32,
    ) -> Vec<CalendarEventRow>;
}

/// No-op default used by the CLI engine and tests.
pub struct NullCalendarProvider;

impl CalendarProvider for NullCalendarProvider {
    fn has_write_permission(&self) -> bool {
        false
    }
    fn list_calendars(&self) -> Vec<Calendar> {
        Vec::new()
    }
    fn insert(&self, _params: InsertCalendarEventParams) -> Option<u64> {
        None
    }
    fn delete(&self, _id: u64) -> bool {
        false
    }
    fn query_in_range(
        &self,
        _start_ms: i64,
        _end_ms: i64,
        _limit: u32,
    ) -> Vec<CalendarEventRow> {
        Vec::new()
    }
}

// ── Local clock ────────────────────────────────────────────────────────

/// Components of the current local datetime, as seen by the host.
///
/// A skill needs this to interpret "today", "next Friday", "on the
/// 27th" relative to the user's timezone — which the skill can't
/// compute itself from `ari::now_ms()` alone because WASM has no TZ
/// database. The host reads its own locale and hands back fully
/// resolved components.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalTimeComponents {
    pub year: i32,
    /// 1..=12
    pub month: u8,
    /// 1..=31
    pub day: u8,
    /// 0..=23
    pub hour: u8,
    /// 0..=59
    pub minute: u8,
    /// 0..=59. Rarely needed but cheap to include.
    pub second: u8,
    /// ISO weekday: 0=Monday..6=Sunday.
    pub weekday: u8,
    /// IANA timezone id (`"Europe/London"`, `"America/New_York"`).
    pub tz_id: String,
}

/// Host-supplied wall-clock reader. Separate from `ari::now_ms()`
/// (which is always UTC epoch ms and needs no host-specific logic)
/// because breaking a timestamp into local components requires the
/// host's TZ database.
pub trait LocalClock: Send + Sync {
    fn now_components(&self) -> LocalTimeComponents;
    fn timezone_id(&self) -> String;
}

/// UTC-only fallback used when the host doesn't supply a real clock.
/// Returns components computed against UTC (so skills still function
/// deterministically in tests), and reports "UTC" as the zone id.
pub struct UtcLocalClock;

impl LocalClock for UtcLocalClock {
    fn now_components(&self) -> LocalTimeComponents {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        let (year, month, day) = days_to_ymd((secs.div_euclid(86_400)) as i64);
        let tod = secs.rem_euclid(86_400);
        let hour = (tod / 3_600) as u8;
        let minute = ((tod % 3_600) / 60) as u8;
        let second = (tod % 60) as u8;
        // 1970-01-01 was a Thursday → Unix day 0. ISO weekday 3 (Thu).
        // `day_of_week = (unix_day + 3) mod 7`.
        let weekday = ((secs.div_euclid(86_400) + 3).rem_euclid(7)) as u8;
        LocalTimeComponents {
            year,
            month,
            day,
            hour,
            minute,
            second,
            weekday,
            tz_id: "UTC".into(),
        }
    }

    fn timezone_id(&self) -> String {
        "UTC".into()
    }
}

/// Convert a Unix-day count (days since 1970-01-01) to a
/// `(year, month, day)` triple in the proleptic Gregorian calendar.
/// Shamelessly lifted from the "civil_from_days" algorithm in Howard
/// Hinnant's date library — public-domain, branchless, leap-year-safe
/// across the full range of i64 days.
pub(crate) fn days_to_ymd(z: i64) -> (i32, u8, u8) {
    // Shift so day 0 is 0000-03-01 (the "era" boundary — start of a
    // 400-year cycle). Every era has exactly 146_097 days.
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097) as u64;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp.wrapping_sub(9) };
    let year = if m <= 2 { y + 1 } else { y };
    (year as i32, m as u8, d as u8)
}

// ── Locale ─────────────────────────────────────────────────────────────

/// Host-supplied access to the user's currently-active language.
///
/// The frontend's settings store is the single source of truth for
/// locale (per the multi-language architecture decision). Engine code
/// that needs to dispatch on language — text normalisers, LLM prompt
/// selection, skill regex filtering — reads through this trait rather
/// than threading a parameter manually.
///
/// Implementations are read on every utterance, so they should be
/// cheap. The host typically caches the latest value via a `StateFlow`
/// or atomic, updated by a background coroutine that watches the
/// settings store.
pub trait LocaleProvider: Send + Sync {
    /// ISO 639-1 lowercase language code (e.g. `"en"`, `"it"`). The
    /// engine treats this as opaque; mapping rules between region
    /// variants ("en-GB" → "en") live on the host side.
    fn current_locale(&self) -> String;
}

/// Default locale provider used when no host has supplied one. Always
/// returns `"en"`. Used by the CLI engine and tests so engine code can
/// call `current_locale()` unconditionally without first checking that
/// a real provider has been wired up.
pub struct EnglishLocaleProvider;

impl LocaleProvider for EnglishLocaleProvider {
    fn current_locale(&self) -> String {
        "en".into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_tasks_provider_is_empty() {
        let p = NullTasksProvider;
        assert!(!p.is_provider_installed());
        assert!(p.list_lists().is_empty());
        assert!(p.insert(InsertTaskParams {
            list_id: 1,
            title: "x".into(),
            due_ms: None,
            due_all_day: false,
            tz_id: None,
        })
        .is_none());
        assert!(!p.delete(42));
    }

    #[test]
    fn null_calendar_provider_is_empty() {
        let p = NullCalendarProvider;
        assert!(!p.has_write_permission());
        assert!(p.list_calendars().is_empty());
        assert!(p
            .insert(InsertCalendarEventParams {
                calendar_id: 1,
                title: "x".into(),
                start_ms: 0,
                duration_minutes: 30,
                reminder_minutes_before: 5,
                tz_id: "UTC".into(),
            })
            .is_none());
        assert!(!p.delete(42));
    }

    #[test]
    fn utc_clock_returns_plausible_components() {
        let c = UtcLocalClock.now_components();
        // Anything from 2020 to 2100 is "plausible enough" without
        // hardcoding a timestamp that'd make the test flaky.
        assert!(c.year >= 2020 && c.year <= 2100, "year={}", c.year);
        assert!((1..=12).contains(&c.month), "month={}", c.month);
        assert!((1..=31).contains(&c.day), "day={}", c.day);
        assert!(c.hour <= 23);
        assert!(c.minute <= 59);
        assert!(c.second <= 59);
        assert!(c.weekday <= 6);
        assert_eq!(c.tz_id, "UTC");
    }

    #[test]
    fn utc_clock_weekday_on_known_date() {
        // Not testing "now" (flaky) but the helper on known dates.
        // 1970-01-01 was a Thursday → weekday 3.
        let (_, _, _) = days_to_ymd(0);
        let weekday_1970_01_01 = ((0_i64 + 3).rem_euclid(7)) as u8;
        assert_eq!(weekday_1970_01_01, 3);
        // 2026-04-22 is a Wednesday → weekday 2.
        // Unix day = (2026-04-22) - (1970-01-01).
        let unix_day_2026_04_22 = 20_565_i64; // precomputed
        assert_eq!(days_to_ymd(unix_day_2026_04_22), (2026, 4, 22));
        assert_eq!(((unix_day_2026_04_22 + 3).rem_euclid(7)) as u8, 2);
    }

    #[test]
    fn english_locale_provider_returns_en() {
        assert_eq!(EnglishLocaleProvider.current_locale(), "en");
    }
}
