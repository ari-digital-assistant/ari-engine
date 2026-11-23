//! When false wakes come often enough to offer to do something about them,
//! moved from ari-android's `FalseWakeMonitor`. The voice session decides
//! what a false wake is; the frontend keeps the timestamps and shows the
//! notification.

/// How long a false wake stays on the books.
///
/// A day rather than an hour, because the complaint that brings someone to
/// this feature is "it keeps going off", which is a thing people notice
/// across an evening, not inside one.
pub const WINDOW_MS: i64 = 24 * 60 * 60 * 1000;

/// False wakes inside [`WINDOW_MS`] before Ari offers to intervene.
///
/// PROVISIONAL, and honestly so: nobody has ever measured what a normal
/// false-wake rate looks like on a real phone, so this is the number of times
/// a day that sounded annoying rather than a number anything produced. It is
/// deliberately on the high side: a notification that arrives when nothing is
/// wrong teaches people to dismiss the one that matters. Replace it once the
/// false-wake log has said what real phones do.
pub const BURST_THRESHOLD: usize = 5;

/// Drops anything older than [`WINDOW_MS`]; also the pruning on every write.
/// A stamp from the future goes too: a clock that has gone backwards would
/// otherwise leave it on the books for good.
pub fn prune(timestamps: &[i64], now: i64) -> Vec<i64> {
    timestamps.iter().copied().filter(|t| (0..=WINDOW_MS).contains(&(now - t))).collect()
}

/// Whether `recent`, already pruned to the window, is a burst worth a
/// notification. Only false wakes after the last notification count, so the
/// next one needs a fresh burst rather than arriving again on the very next
/// false wake.
pub fn burst_due(recent: &[i64], notified_at: i64) -> bool {
    recent.iter().filter(|t| **t > notified_at).count() >= BURST_THRESHOLD
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_false_wakes_inside_the_window_count_toward_a_burst() {
        let now = 1_000_000_000;
        let timestamps = [now - WINDOW_MS - 1, now - WINDOW_MS + 1, now - 1_000, now];
        assert_eq!(prune(&timestamps, now), [now - WINDOW_MS + 1, now - 1_000, now]);
    }

    #[test]
    fn a_clock_that_has_gone_backwards_drops_the_future_rather_than_counting_it() {
        // Timezone changes and NTP corrections both do this. A stamp from the
        // future would otherwise sit in the log forever, one short of a burst,
        // and tip the next real false wake into a notification early.
        let now = 1_000_000_000;
        assert!(prune(&[now + 60_000], now).is_empty());
    }

    #[test]
    fn five_false_wakes_in_the_window_are_a_burst_four_are_not() {
        let four = [1_000, 2_000, 3_000, 4_000];
        assert!(!burst_due(&four, 0));
        assert!(burst_due(&[1_000, 2_000, 3_000, 4_000, 5_000], 0));
    }

    #[test]
    fn after_a_warning_only_the_false_wakes_since_it_count_toward_the_next_one() {
        // The log keeps all six for the screen the warning opens; the next
        // warning still needs five new ones rather than arriving on the next.
        let recent = [1_000, 2_000, 3_000, 4_000, 5_000, 6_000];
        assert!(!burst_due(&recent, 5_000));
        let more = [1_000, 2_000, 3_000, 4_000, 5_000, 6_000, 7_000, 8_000, 9_000, 10_000];
        assert!(burst_due(&more, 5_000));
    }
}
