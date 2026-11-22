//! The voice pipeline every frontend shares: the wake word, speech-to-text,
//! and the rules that decide what a voice turn does with them.
//!
//! It is moving here from `ari-android` a step at a time
//! (`docs/superpowers/plans/2026-11-22-audio-in-engine.md`). What is here so
//! far is the rules and tables: matching the wake phrase in a transcript, the
//! wake word models and their sensitivity ladders, tuning those ladders, and
//! the cloud speech-to-text client.
//!
//! No UniFFI here. A Rust frontend links this crate directly; `ari-ffi`
//! wraps it for Kotlin.

pub mod cloud_stt;
pub mod detector;
pub mod endpoint;
pub mod ladder;
pub mod native;
pub mod pipeline;
pub mod ring;
pub mod sherpa;
pub mod speech_check;
pub mod wake_log;
pub mod wake_model;
pub mod wake_phrase;
