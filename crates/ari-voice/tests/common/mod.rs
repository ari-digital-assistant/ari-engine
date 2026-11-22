//! What the native tests share: the host build of libari_voice_native, the
//! engine's bundled models, and the fixtures.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

use ari_voice::native;

pub fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

pub fn use_host_library() {
    let library = workspace().join("target/voice-native/libari_voice_native.so");
    assert!(
        library.exists(),
        "{} is missing: run native/voice/build-host.sh first",
        library.display()
    );
    native::set_library_path(library);
}

pub fn asset(name: &str) -> Result<Vec<u8>, String> {
    std::fs::read(workspace().join("assets/voice").join(name)).map_err(|e| format!("{name}: {e}"))
}

pub fn fixture(name: &str) -> String {
    std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)).unwrap()
}

/// The samples of a 16-bit mono WAV's data chunk.
pub fn wav(name: &str) -> Vec<i16> {
    let bytes = std::fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)).unwrap();
    let mut offset = 12;
    while offset + 8 <= bytes.len() {
        let size = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap()) as usize;
        if &bytes[offset..offset + 4] == b"data" {
            return bytes[offset + 8..offset + 8 + size]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| i16::from_le_bytes(*b))
                .collect();
        }
        offset += 8 + size + (size & 1);
    }
    panic!("{name} has no data chunk");
}
