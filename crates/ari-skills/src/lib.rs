//! Built-in Rust skills shipped with the Ari engine.
//!
//! Each skill implements [`ari_core::Skill`]. To add a new built-in skill:
//!
//! 1. Create a new module here (e.g. `weather.rs`).
//! 2. Implement the [`ari_core::Skill`] trait. Required: `id()`,
//!    `specificity()`, `score()`, `execute()`. Strongly recommended for
//!    the FunctionGemma router: `description()`, `example_utterances()`,
//!    and `parameters_schema()` if your skill takes args.
//! 3. Add the module declaration and `pub use` below.
//! 4. Add the skill to `bin/export_utterances.rs` so it appears in the
//!    training data dump.
//! 5. Register it in `ari-engine/crates/ari-ffi/src/lib.rs::build_engine_with_builtins`.
//!
//! See the trait-level docs on [`ari_core::Skill`] for guidance on writing
//! a description and example utterances that work well with the router.

mod calculator;
mod current_time;
mod date;
mod greeting;
mod open;
mod search;

pub use calculator::CalculatorSkill;
pub use current_time::CurrentTimeSkill;
pub use date::DateSkill;
pub use greeting::GreetingSkill;
pub use open::OpenSkill;
pub use search::SearchSkill;
