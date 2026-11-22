//! A fixed-capacity PCM ring with a monotonic sample counter: the last two
//! seconds of the microphone, which the speech check reads back at a
//! detection. Written and read by the capture thread alone.

/// Two seconds at 16 kHz.
pub const RING_CAPACITY_SAMPLES: usize = 2 * 16_000;

#[derive(Debug)]
pub struct AudioRing {
    buffer: Vec<i16>,
    written: u64,
}

impl AudioRing {
    pub fn new(capacity: usize) -> Self {
        Self { buffer: vec![0; capacity], written: 0 }
    }

    /// Total samples ever written (monotonic, never wraps).
    pub fn samples_written(&self) -> u64 {
        self.written
    }

    /// Append `samples`, overwriting the oldest on wrap.
    pub fn write(&mut self, samples: &[i16]) {
        let capacity = self.buffer.len();
        if samples.is_empty() || capacity == 0 {
            return;
        }
        // More than the ring holds: only the newest fit, at the positions they
        // would have reached.
        let skip = samples.len().saturating_sub(capacity);
        let start = ((self.written + skip as u64) % capacity as u64) as usize;
        let kept = &samples[skip..];
        let first = kept.len().min(capacity - start);
        self.buffer[start..start + first].copy_from_slice(&kept[..first]);
        self.buffer[..kept.len() - first].copy_from_slice(&kept[first..]);
        self.written += samples.len() as u64;
    }

    /// Samples in the absolute range `from..to` (positions in the monotonic
    /// stream), clamped to what the ring still holds and what has been written.
    pub fn snapshot(&self, from: u64, to: u64) -> Vec<i16> {
        let capacity = self.buffer.len() as u64;
        let end = to.min(self.written);
        let start = from.max(self.written.saturating_sub(capacity));
        if start >= end {
            return Vec::new();
        }
        let length = (end - start) as usize;
        let ring_start = (start % capacity) as usize;
        let first = length.min(self.buffer.len() - ring_start);
        let mut out = Vec::with_capacity(length);
        out.extend_from_slice(&self.buffer[ring_start..ring_start + first]);
        out.extend_from_slice(&self.buffer[..length - first]);
        out
    }

    /// The most recent `samples`, or all there are.
    pub fn recent(&self, samples: usize) -> Vec<i16> {
        self.snapshot(self.written.saturating_sub(samples as u64), self.written)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(from: i16, count: usize) -> Vec<i16> {
        (0..count).map(|i| from + i as i16).collect()
    }

    #[test]
    fn recent_returns_the_newest_samples_in_order() {
        let mut ring = AudioRing::new(8);
        ring.write(&ramp(0, 5));
        ring.write(&ramp(5, 5));
        assert_eq!(ramp(4, 6), ring.recent(6));
        assert_eq!(ramp(2, 8), ring.recent(100));
        assert_eq!(10, ring.samples_written());
    }

    #[test]
    fn a_snapshot_is_clamped_to_what_the_ring_still_holds() {
        let mut ring = AudioRing::new(8);
        ring.write(&ramp(0, 12));
        assert_eq!(ramp(4, 6), ring.snapshot(0, 10));
        assert!(ring.snapshot(10, 10).is_empty());
        assert_eq!(ramp(10, 2), ring.snapshot(10, 99));
    }

    #[test]
    fn a_write_longer_than_the_ring_keeps_its_newest_samples() {
        let mut ring = AudioRing::new(4);
        ring.write(&ramp(0, 3));
        ring.write(&ramp(3, 10));
        assert_eq!(ramp(9, 4), ring.recent(4));
        assert_eq!(13, ring.samples_written());
    }
}
