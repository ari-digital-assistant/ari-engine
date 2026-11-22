//! sherpa-onnx through the C API, with real models. Ignored by default, as the
//! LiteRT model tests are: they need `native/voice/build-host.sh` and the
//! models in `$ARI_STT_MODELS` (default `~/.cache/ari/stt/<model id>/`).
//!
//! ```sh
//! cargo test -p ari-voice --test sherpa_models -- --ignored --nocapture
//! ```

mod common;

use std::path::PathBuf;

use ari_voice::sherpa::{self, OnlineRecognizer, TransducerFiles, to_float};
use common::{wav, workspace};

fn models() -> PathBuf {
    std::env::var_os("ARI_STT_MODELS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/ari/stt"))
}

fn use_host_sherpa() {
    sherpa::set_library_path(workspace().join("target/voice-native/libsherpa-onnx-c-api.so"));
}

#[test]
#[ignore = "needs the host build and the Kroko model"]
fn kroko_streams_the_parity_clip() {
    use_host_sherpa();
    let dir = models().join("kroko-2025-08-06");
    let files = TransducerFiles {
        encoder: dir.join("encoder.onnx"),
        decoder: dir.join("decoder.onnx"),
        joiner: dir.join("joiner.onnx"),
        tokens: dir.join("tokens.txt"),
    };
    let recognizer = OnlineRecognizer::new(&files, "zipformer2").unwrap();
    let mut stream = recognizer.create_stream().unwrap();
    for batch in wav("oww_parity.wav").chunks(1600) {
        stream.accept_waveform(&to_float(batch));
        stream.decode_ready();
    }
    stream.input_finished();
    stream.decode_ready();
    let text = stream.text();
    println!("Kroko heard: {text:?}");
    assert!(text.to_lowercase().contains("ari"), "{text}");
}
