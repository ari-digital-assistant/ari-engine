//! The voice pipeline every frontend shares: the wake word, speech-to-text,
//! and the rules that decide what a voice turn does with them.
//!
//! The frontend opens the microphone and pushes its audio in; the pipeline
//! detects the wake word, transcribes, and runs the voice session, telling the
//! frontend what to show, play and run. It moved here from `ari-android`
//! (`docs/superpowers/plans/2026-11-22-audio-in-engine.md`).
//!
//! No UniFFI here. A Rust frontend links this crate directly; `ari-ffi`
//! wraps it for Kotlin.

pub mod cloud_stt;
pub mod detector;
pub mod endpoint;
pub mod false_wake;
pub mod ladder;
pub mod native;
pub mod pipeline;
pub mod ring;
pub mod session;
pub mod sherpa;
pub mod speech_check;
pub mod stt;
pub mod stt_models;
pub mod wake_log;
pub mod wake_model;
pub mod wake_phrase;
