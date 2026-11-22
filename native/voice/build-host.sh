#!/bin/sh
# Builds libari_voice_native.so for this machine into target/voice-native/,
# with sherpa-onnx's C API and the ONNX Runtime both load beside it.
# ari-voice's tests load them from there, so run this before `cargo test`
# whenever anything in native/voice changes. The first run downloads TFLite
# Micro, ONNX Runtime's headers and sherpa-onnx.
set -eu
root="$(cd "$(dirname "$0")/../.." && pwd)"
out="$root/target/voice-native"
# Clang, as Android's NDK uses: GCC rejects TFLite Micro's error reporter.
cmake -S "$root/native/voice" -B "$out/build" \
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$out"
# Only the library: the fetched projects bring targets of their own.
cmake --build "$out/build" --parallel --target ari_voice_native
echo "Built $out/libari_voice_native.so"
