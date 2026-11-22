//! The wake engines and the VAD through `libari_voice_native`, against
//! references made outside it. Needs the host build: run
//! `native/voice/build-host.sh` first.
//!
//! `fixtures/oww_parity.wav` is a Common Voice English clip (CC0) and three
//! synthetic "Hey Ari"s from piper-sample-generator, behind the 2 s lead
//! exam_oww puts in front of every clip. Made by
//! `ari-tools/wakeword/corpus/tater/scripts/oww_parity_fixture.py`.

use std::path::{Path, PathBuf};

use ari_voice::detector::{OWW_EMBEDDING_FILE, OWW_MELSPECTROGRAM_FILE, SILERO_VAD_FILE, WakeDetector};
use ari_voice::native::{self, OWW_FRAME_SAMPLES, OpenWakeWord, SileroVad, VAD_WINDOW_SAMPLES};
use ari_voice::wake_model::{OperatingPoint, by_id};

/// The capture loop's read size (WakeWordService.CHUNK_SIZE).
const READ: usize = 480;
const WARMUP_FRAMES: usize = 24;
const FIRST_EXPECTED_FRAME: usize = 25;

/// Set before the first run on the phone. Float rounding across ONNX Runtime
/// versions and CPUs moves a score by far less; a port that gets the pipeline
/// wrong moves it by far more.
const OWW_TOLERANCE: f32 = 1e-3;

/// The reference is plain ONNX Runtime, which is all sherpa-onnx's
/// `Vad.compute` is; the spike found differences below 1e-9.
const VAD_TOLERANCE: f32 = 1e-5;

fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn use_host_library() {
    let library = workspace().join("target/voice-native/libari_voice_native.so");
    assert!(
        library.exists(),
        "{} is missing: run native/voice/build-host.sh first",
        library.display()
    );
    native::set_library_path(library);
}

fn asset(name: &str) -> Result<Vec<u8>, String> {
    std::fs::read(workspace().join("assets/voice").join(name)).map_err(|e| format!("{name}: {e}"))
}

fn fixture(name: &str) -> String {
    std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)).unwrap()
}

/// The samples of a 16-bit mono WAV's data chunk.
fn wav(name: &str) -> Vec<i16> {
    let bytes = std::fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)).unwrap();
    let mut offset = 12;
    while offset + 8 <= bytes.len() {
        let size = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap()) as usize;
        if &bytes[offset..offset + 4] == b"data" {
            return bytes[offset + 8..offset + 8 + size]
                .chunks_exact(2)
                .map(|b| i16::from_le_bytes([b[0], b[1]]))
                .collect();
        }
        offset += 8 + size + (size & 1);
    }
    panic!("{name} has no data chunk");
}

fn expected_oww_scores() -> Vec<f32> {
    fixture("oww_parity_scores.txt").lines().map(|l| l.parse().unwrap()).collect()
}

fn hey_ari(cutoff: f32, reset_on_detection: bool) -> OpenWakeWord {
    use_host_library();
    OpenWakeWord::new(
        &asset(OWW_MELSPECTROGRAM_FILE).unwrap(),
        &asset(OWW_EMBEDDING_FILE).unwrap(),
        &asset(&by_id(Some("hey_ari_oww")).asset_filename).unwrap(),
        cutoff,
        reset_on_detection,
    )
    .unwrap()
}

#[test]
fn open_wake_word_scores_match_the_laptop_scorer() {
    let audio = wav("oww_parity.wav");
    let expected = expected_oww_scores();
    // The fixture has to span the cutoffs, or agreeing on it proves little.
    assert!(expected.iter().cloned().fold(f32::MIN, f32::max) > 0.95);
    assert!(expected.iter().cloned().fold(f32::MAX, f32::min) < 0.001);
    assert!(expected.iter().any(|s| (0.3..=0.7).contains(s)));

    let mut engine = hey_ari(1.0, true);
    let scores: Vec<f32> = audio.chunks(READ).flat_map(|read| engine.scores(read)).collect();

    assert_eq!(audio.len() / OWW_FRAME_SAMPLES, scores.len());
    // Nothing until the classifier's 16 embeddings are all audio.
    assert_eq!(vec![0.0; WARMUP_FRAMES], scores[..WARMUP_FRAMES]);
    let compared = &scores[FIRST_EXPECTED_FRAME..];
    assert_eq!(expected.len(), compared.len());
    let (worst, diff) = compared
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .enumerate()
        .fold((0, 0.0), |best, (i, d)| if d > best.1 { (i, d) } else { best });
    assert!(
        diff <= OWW_TOLERANCE,
        "frame {}: engine {}, laptop {}",
        worst + FIRST_EXPECTED_FRAME,
        compared[worst],
        expected[worst]
    );
}

#[test]
fn open_wake_word_without_reset_reports_every_frame_over_the_cutoff() {
    let audio = wav("oww_parity.wav");
    let cutoff = by_id(Some("hey_ari_oww")).medium.probability_cutoff;
    let mut engine = hey_ari(cutoff, false);
    let detections = audio.chunks(READ).filter(|read| engine.process(read)).count();

    // Each read is shorter than a frame, so completes at most one.
    assert_eq!(expected_oww_scores().iter().filter(|s| **s > cutoff).count(), detections);
}

#[test]
fn open_wake_word_reset_on_detection_goes_deaf_for_the_warm_up() {
    let audio = wav("oww_parity.wav");
    let expected = expected_oww_scores();
    let cutoff = by_id(Some("hey_ari_oww")).medium.probability_cutoff;
    let mut engine = hey_ari(cutoff, true);
    // Whole frames, so the samples a detection discards are none and the
    // scores after each reset line up with the laptop's.
    let detections = audio.chunks(OWW_FRAME_SAMPLES).filter(|frame| engine.process(frame)).count();

    let mut expected_detections = 0;
    let mut frame = 0;
    while frame < expected.len() {
        if expected[frame] > cutoff {
            expected_detections += 1;
            frame += WARMUP_FRAMES + 1;
        } else {
            frame += 1;
        }
    }
    assert_eq!(expected_detections, detections);
    let over = expected.iter().filter(|s| **s > cutoff).count();
    assert!((2..over).contains(&expected_detections));
}

#[test]
fn micro_wake_word_detects_exactly_where_the_android_engine_did() {
    use_host_library();
    let audio = wav("oww_parity.wav");
    let recorded = fixture("mww_parity_detections.txt");
    let mut checked = 0;
    for line in recorded.lines().filter(|l| !l.starts_with('#')) {
        let (head, reads) = line.split_once(':').unwrap();
        let mut head = head.split_whitespace();
        let model = by_id(head.next());
        let point = OperatingPoint {
            probability_cutoff: head.next().unwrap().parse().unwrap(),
            sliding_window_size: head.next().unwrap().parse().unwrap(),
        };
        let expected: Vec<usize> = reads.split_whitespace().map(|i| i.parse().unwrap()).collect();

        let mut detector = WakeDetector::new(&model, point, false, &asset).unwrap();
        let detected: Vec<usize> = audio
            .chunks(READ)
            .enumerate()
            .filter(|(_, read)| detector.process_audio(read))
            .map(|(i, _)| i)
            .collect();
        assert_eq!(expected, detected, "{line}");
        checked += 1;
    }
    assert!(checked >= 6);
}

#[test]
fn silero_vad_matches_plain_onnx_runtime_window_by_window() {
    use_host_library();
    let audio = wav("oww_parity.wav");
    let expected: Vec<f32> = fixture("vad_parity_probabilities.txt").lines().map(|l| l.parse().unwrap()).collect();
    // Speech and silence both, or agreeing proves little.
    assert!(expected.iter().any(|p| *p > 0.9) && expected.iter().any(|p| *p < 0.1));

    let mut vad = SileroVad::new(&asset(SILERO_VAD_FILE).unwrap()).unwrap();
    let probabilities: Vec<f32> = audio
        .chunks_exact(VAD_WINDOW_SAMPLES)
        .map(|w| {
            let window: [f32; VAD_WINDOW_SAMPLES] = std::array::from_fn(|i| w[i] as f32 / 32768.0);
            vad.compute(&window).unwrap()
        })
        .collect();

    assert_eq!(expected.len(), probabilities.len());
    for (i, (got, want)) in probabilities.iter().zip(&expected).enumerate() {
        assert!((got - want).abs() <= VAD_TOLERANCE, "window {i}: {got} vs {want}");
    }
}
