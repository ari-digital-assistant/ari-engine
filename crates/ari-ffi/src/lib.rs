#![allow(clippy::new_without_default)]

use ari_engine::{Engine, FALLBACK_RESPONSE};
use ari_skill_loader::{
    load_skill_directory_with, Calendar, CalendarProvider, Capability, HostCapabilities,
    HttpConfig, InsertCalendarEventParams, InsertTaskParams, LoadOptions, LocalClock,
    LocalTimeComponents, LogLevel, LogSink, NullCalendarProvider, NullLogSink, NullTasksProvider,
    StorageConfig, TaskList, TasksProvider, UtcLocalClock,
};
use ari_skills::{
    CalculatorSkill, CurrentTimeSkill, DateSkill, GreetingSkill, OpenSkill, SearchSkill,
};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

mod assistant_registry;
mod settings_store;
mod skill_registry;

pub use assistant_registry::{
    AssistantRegistry, FfiAssistantEntry, FfiConfigField, FfiSelectOption,
};
pub use settings_store::SkillSettingsStore;
pub use skill_registry::{
    FfiBrowseEntry, FfiInstalledSkill, FfiRegistryError, FfiSkillUpdate, SkillRegistry,
};

/// Build the [`LoadOptions`] the Android host uses for every install and
/// every reload. Grants `pure_frontend` caps (frontend-mediated actions),
/// `http` (backed by reqwest with bundled webpki-roots — see `tls.rs`),
/// and `storage_kv` (backed by per-skill JSON files under `storage_dir`).
///
/// Keep this in one place so every loader entry point in the FFI crate
/// sees the same grants. A mismatch — e.g. install granting `http` but
/// reload not — would let a skill install cleanly and then silently drop
/// off the conversation engine on the next app start.
pub(crate) fn android_load_options(storage_dir: &str) -> LoadOptions {
    let host_caps = HostCapabilities::pure_frontend()
        .with(Capability::Http)
        .with(Capability::StorageKv)
        .with(Capability::Tasks)
        .with(Capability::Calendar);
    LoadOptions {
        log_sink: Arc::new(NullLogSink),
        host_capabilities: host_caps,
        http_config: HttpConfig::strict(),
        storage_config: StorageConfig::new(PathBuf::from(storage_dir)),
        tasks_provider: Arc::new(NullTasksProvider),
        calendar_provider: Arc::new(NullCalendarProvider),
        local_clock: Arc::new(UtcLocalClock),
    }
}

uniffi::setup_scaffolding!();

/// WASM-skill log level, mirrored from [`ari_skill_loader::LogLevel`] for
/// the UniFFI boundary. The engine's own `LogLevel` isn't exportable
/// directly because UniFFI types can't derive outside the FFI crate.
#[derive(Debug, Clone, Copy, uniffi::Enum)]
pub enum FfiLogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl From<LogLevel> for FfiLogLevel {
    fn from(level: LogLevel) -> Self {
        match level {
            LogLevel::Trace => FfiLogLevel::Trace,
            LogLevel::Debug => FfiLogLevel::Debug,
            LogLevel::Info => FfiLogLevel::Info,
            LogLevel::Warn => FfiLogLevel::Warn,
            LogLevel::Error => FfiLogLevel::Error,
        }
    }
}

/// Callback interface the host implements to receive log lines from WASM
/// skills. Rust calls `log` whenever a skill invokes `ari::log(...)` via
/// the SDK's `host_log` import. On Android this is wired to
/// `android.util.Log`; on other hosts (CLI, tests) it defaults to a
/// no-op sink constructed internally.
#[uniffi::export(with_foreign)]
pub trait FfiLogSink: Send + Sync {
    fn log(&self, skill_id: String, level: FfiLogLevel, message: String);
}

/// Wraps a foreign [`FfiLogSink`] so it can satisfy the engine's internal
/// [`LogSink`] trait. The engine's trait takes borrowed `&str`s; we own
/// them across the FFI boundary, so the adapter copies into `String` on
/// every call. Logging isn't on the hot path, so the allocation is fine.
struct ForeignLogSinkAdapter(Arc<dyn FfiLogSink>);

impl LogSink for ForeignLogSinkAdapter {
    fn log(&self, skill_id: &str, level: LogLevel, message: &str) {
        self.0
            .log(skill_id.to_string(), level.into(), message.to_string());
    }
}

// ── Platform capability FFI surface ─────────────────────────────────
//
// Android (and, in future, the Linux frontend) implement these traits
// to expose the platform's tasks / calendar / clock APIs to skills.
// No skill-specific knowledge lives on either side of the boundary —
// every skill that declares the right capability can use the whole
// surface.

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiTaskList {
    pub id: u64,
    pub display_name: String,
    pub account_name: String,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiInsertTaskParams {
    pub list_id: u64,
    pub title: String,
    pub due_ms: Option<i64>,
    pub due_all_day: bool,
    pub tz_id: Option<String>,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiCalendar {
    pub id: u64,
    pub display_name: String,
    pub account_name: String,
    pub color_argb: Option<i32>,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiInsertCalendarEventParams {
    pub calendar_id: u64,
    pub title: String,
    pub start_ms: i64,
    pub duration_minutes: u32,
    pub reminder_minutes_before: u32,
    pub tz_id: String,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiLocalTimeComponents {
    pub year: i32,
    pub month: u8,
    pub day: u8,
    pub hour: u8,
    pub minute: u8,
    pub second: u8,
    /// 0=Monday..6=Sunday
    pub weekday: u8,
    pub tz_id: String,
}

/// Foreign-implemented tasks provider. The host wraps whatever
/// platform API gives it read/write access to user tasks — on Android,
/// the OpenTasks ContentResolver; on Linux, EDS.
#[uniffi::export(with_foreign)]
pub trait FfiTasksProvider: Send + Sync {
    fn is_provider_installed(&self) -> bool;
    fn list_lists(&self) -> Vec<FfiTaskList>;
    /// Returns 0 on failure; the provider row id otherwise. UniFFI
    /// over JNI marshals `Option<u64>` awkwardly, so the sentinel-0
    /// convention matches what the host-side WASM ABI already uses.
    fn insert(&self, params: FfiInsertTaskParams) -> u64;
    fn delete(&self, id: u64) -> bool;
}

/// Foreign-implemented calendar provider.
#[uniffi::export(with_foreign)]
pub trait FfiCalendarProvider: Send + Sync {
    fn has_write_permission(&self) -> bool;
    fn list_calendars(&self) -> Vec<FfiCalendar>;
    fn insert(&self, params: FfiInsertCalendarEventParams) -> u64;
    fn delete(&self, id: u64) -> bool;
}

/// Foreign-implemented wall-clock reader. Needed so skills can
/// resolve weekdays / "today" / local dates — WASM has no TZ
/// database, the host does.
#[uniffi::export(with_foreign)]
pub trait FfiLocalClock: Send + Sync {
    fn now_components(&self) -> FfiLocalTimeComponents;
    fn timezone_id(&self) -> String;
}

// Adapters from the foreign FFI traits to the engine's internal
// traits. Engine code only sees the internal trait object; these
// adapters handle the `Arc<dyn FfiFoo>` → `Arc<dyn Foo>` conversion
// so the engine doesn't need to know UniFFI exists.

struct ForeignTasksProviderAdapter(Arc<dyn FfiTasksProvider>);

impl TasksProvider for ForeignTasksProviderAdapter {
    fn is_provider_installed(&self) -> bool {
        self.0.is_provider_installed()
    }
    fn list_lists(&self) -> Vec<TaskList> {
        self.0
            .list_lists()
            .into_iter()
            .map(|l| TaskList {
                id: l.id,
                display_name: l.display_name,
                account_name: l.account_name,
            })
            .collect()
    }
    fn insert(&self, params: InsertTaskParams) -> Option<u64> {
        let ffi = FfiInsertTaskParams {
            list_id: params.list_id,
            title: params.title,
            due_ms: params.due_ms,
            due_all_day: params.due_all_day,
            tz_id: params.tz_id,
        };
        match self.0.insert(ffi) {
            0 => None,
            id => Some(id),
        }
    }
    fn delete(&self, id: u64) -> bool {
        self.0.delete(id)
    }
}

struct ForeignCalendarProviderAdapter(Arc<dyn FfiCalendarProvider>);

impl CalendarProvider for ForeignCalendarProviderAdapter {
    fn has_write_permission(&self) -> bool {
        self.0.has_write_permission()
    }
    fn list_calendars(&self) -> Vec<Calendar> {
        self.0
            .list_calendars()
            .into_iter()
            .map(|c| Calendar {
                id: c.id,
                display_name: c.display_name,
                account_name: c.account_name,
                color_argb: c.color_argb,
            })
            .collect()
    }
    fn insert(&self, params: InsertCalendarEventParams) -> Option<u64> {
        let ffi = FfiInsertCalendarEventParams {
            calendar_id: params.calendar_id,
            title: params.title,
            start_ms: params.start_ms,
            duration_minutes: params.duration_minutes,
            reminder_minutes_before: params.reminder_minutes_before,
            tz_id: params.tz_id,
        };
        match self.0.insert(ffi) {
            0 => None,
            id => Some(id),
        }
    }
    fn delete(&self, id: u64) -> bool {
        self.0.delete(id)
    }
}

struct ForeignLocalClockAdapter(Arc<dyn FfiLocalClock>);

impl LocalClock for ForeignLocalClockAdapter {
    fn now_components(&self) -> LocalTimeComponents {
        let c = self.0.now_components();
        LocalTimeComponents {
            year: c.year,
            month: c.month,
            day: c.day,
            hour: c.hour,
            minute: c.minute,
            second: c.second,
            weekday: c.weekday,
            tz_id: c.tz_id,
        }
    }
    fn timezone_id(&self) -> String {
        self.0.timezone_id()
    }
}

#[derive(uniffi::Enum)]
pub enum FfiResponse {
    Text { body: String },
    /// `skill_id` is the manifest id of the emitting skill (e.g.
    /// `dev.heyari.timer`), used by the frontend to resolve `asset:<path>`
    /// references back to the skill's bundle directory. Empty string if
    /// the engine couldn't attribute the response to a specific skill
    /// (router-direct actions, fallbacks) — treat that as "no bundle,
    /// asset references will fail to resolve".
    Action { json: String, skill_id: String },
    Binary { mime: String, data: Vec<u8> },
    /// The engine couldn't match any skill to the input. The host can use
    /// this signal to retry the upstream STT (e.g. with a fresh sherpa
    /// stream on the buffered audio) before falling back to the apology.
    /// `body` carries the apology text the host should say if the retry
    /// also fails — kept here so the host doesn't have to hardcode it.
    NotUnderstood { body: String },
}

#[derive(uniffi::Object)]
pub struct AriEngine {
    // Wrapped in Mutex because `reload_community_skills` mutates the
    // skill set after construction. `process_input` only needs a shared
    // lock in practice but the Engine trait takes `&self` anyway.
    pub(crate) inner: Mutex<Engine>,
    // Log sink handed to every WASM skill loaded via `reload_community_skills`.
    // Defaults to NullLogSink for callers that use the no-arg constructor
    // (tests, CLI). The Android host passes a real sink via `with_log_sink`
    // so skill `ari::log(...)` calls surface in `adb logcat`.
    pub(crate) log_sink: Arc<dyn LogSink>,
    // Platform capability providers. Defaults to the Null/UTC impls
    // from ari_skill_loader for callers that don't supply real ones
    // (tests, CLI). The Android host supplies real implementations
    // via [`AriEngine::with_platform_providers`].
    pub(crate) tasks_provider: Arc<dyn TasksProvider>,
    pub(crate) calendar_provider: Arc<dyn CalendarProvider>,
    pub(crate) local_clock: Arc<dyn LocalClock>,
}

fn build_engine_with_builtins() -> Engine {
    let mut engine = Engine::new();
    engine.register_skill(Box::new(CurrentTimeSkill::new()));
    engine.register_skill(Box::new(DateSkill::new()));
    engine.register_skill(Box::new(CalculatorSkill::new()));
    engine.register_skill(Box::new(GreetingSkill::new()));
    engine.register_skill(Box::new(OpenSkill::new()));
    engine.register_skill(Box::new(SearchSkill::new()));
    engine
}

#[uniffi::export]
impl AriEngine {
    #[uniffi::constructor]
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(build_engine_with_builtins()),
            log_sink: Arc::new(NullLogSink),
            tasks_provider: Arc::new(NullTasksProvider),
            calendar_provider: Arc::new(NullCalendarProvider),
            local_clock: Arc::new(UtcLocalClock),
        }
    }

    /// Construct with a host-supplied log sink for WASM skill output.
    /// Android wires a sink that forwards to `android.util.Log`; callers
    /// that don't care about skill logs (tests, CLI smoke tests) use
    /// [`AriEngine::new`] instead.
    #[uniffi::constructor]
    pub fn with_log_sink(sink: Arc<dyn FfiLogSink>) -> Self {
        Self {
            inner: Mutex::new(build_engine_with_builtins()),
            log_sink: Arc::new(ForeignLogSinkAdapter(sink)),
            tasks_provider: Arc::new(NullTasksProvider),
            calendar_provider: Arc::new(NullCalendarProvider),
            local_clock: Arc::new(UtcLocalClock),
        }
    }

    /// Construct with the full set of host-supplied platform
    /// providers. This is the constructor the Android frontend uses
    /// at startup so any skill that declares the `tasks`, `calendar`
    /// or clock capabilities gets real implementations rather than
    /// the Null defaults. Any provider argument can be left `None`
    /// to fall back to the corresponding Null/UTC default — useful
    /// for frontends that only wire up part of the surface.
    #[uniffi::constructor]
    pub fn with_platform_providers(
        sink: Option<Arc<dyn FfiLogSink>>,
        tasks: Option<Arc<dyn FfiTasksProvider>>,
        calendar: Option<Arc<dyn FfiCalendarProvider>>,
        clock: Option<Arc<dyn FfiLocalClock>>,
    ) -> Self {
        let log_sink: Arc<dyn LogSink> = match sink {
            Some(s) => Arc::new(ForeignLogSinkAdapter(s)),
            None => Arc::new(NullLogSink),
        };
        let tasks_provider: Arc<dyn TasksProvider> = match tasks {
            Some(t) => Arc::new(ForeignTasksProviderAdapter(t)),
            None => Arc::new(NullTasksProvider),
        };
        let calendar_provider: Arc<dyn CalendarProvider> = match calendar {
            Some(c) => Arc::new(ForeignCalendarProviderAdapter(c)),
            None => Arc::new(NullCalendarProvider),
        };
        let local_clock: Arc<dyn LocalClock> = match clock {
            Some(c) => Arc::new(ForeignLocalClockAdapter(c)),
            None => Arc::new(UtcLocalClock),
        };
        Self {
            inner: Mutex::new(build_engine_with_builtins()),
            log_sink,
            tasks_provider,
            calendar_provider,
            local_clock,
        }
    }

    pub fn process_input(&self, input: String) -> FfiResponse {
        let engine = self.inner.lock().expect("engine mutex poisoned");
        let (response, skill_id) = engine.process_input_with_skill(&input);
        match response {
            ari_core::Response::Text(s) => {
                if s == FALLBACK_RESPONSE {
                    FfiResponse::NotUnderstood { body: s }
                } else {
                    FfiResponse::Text { body: s }
                }
            }
            ari_core::Response::Action(v) => FfiResponse::Action {
                json: serde_json::to_string(&v).unwrap_or_default(),
                skill_id: skill_id.unwrap_or_default(),
            },
            ari_core::Response::Binary { mime, data } => FfiResponse::Binary { mime, data },
        }
    }

    /// Set the GGUF model path for the LLM fallback. The model is NOT
    /// loaded immediately — it loads on demand when the first unmatched
    /// query arrives, and unloads after 60 seconds of idle to free RAM.
    ///
    /// Returns `true` if the path exists, `false` otherwise.
    /// Call at app startup if a model file is available on disk.
    #[cfg(feature = "llm")]
    pub fn load_llm_model(&self, model_path: String) -> bool {
        let path = std::path::Path::new(&model_path);
        if !path.is_file() {
            return false;
        }
        let lazy = ari_llm::LazyLlmFallback::new(path);
        let mut engine = self.inner.lock().expect("engine mutex poisoned");
        engine.set_llm(Box::new(lazy));
        true
    }

    /// Remove the LLM fallback. If a model is currently loaded in RAM,
    /// it is dropped and the memory is freed.
    #[cfg(feature = "llm")]
    pub fn unload_llm_model(&self) {
        let mut engine = self.inner.lock().expect("engine mutex poisoned");
        engine.set_llm_none();
    }

    /// Set the FunctionGemma router model path. Like the LLM fallback,
    /// the model loads lazily on first use and unloads after 60s idle.
    /// Returns `true` if the path exists, `false` otherwise.
    #[cfg(feature = "llm")]
    pub fn load_router_model(&self, model_path: String) -> bool {
        let path = std::path::Path::new(&model_path);
        if !path.is_file() {
            return false;
        }
        let router = ari_llm::FunctionGemmaRouter::new(path);
        let mut engine = self.inner.lock().expect("engine mutex poisoned");
        engine.set_router(Some(Box::new(router)));
        true
    }

    /// Remove the FunctionGemma router. Keyword scoring still works;
    /// unmatched queries go straight to the assistant.
    #[cfg(feature = "llm")]
    pub fn unload_router_model(&self) {
        let mut engine = self.inner.lock().expect("engine mutex poisoned");
        engine.set_router(None);
    }

    /// Rebuild the engine's skill set from scratch: the 6 built-in Rust
    /// skills plus every community skill on disk under `skill_store_dir`.
    ///
    /// `storage_dir` is where per-skill `storage_kv` JSON files live —
    /// must match what `SkillRegistry` was constructed with, otherwise a
    /// skill's installed state (on-disk JSON) will be invisible at
    /// conversation time. Both dirs should sit under the app's private
    /// files directory on Android (`context.filesDir`).
    ///
    /// Call once at app startup (after constructing `SkillRegistry` so the
    /// store dir exists) and again after every successful install / update
    /// / uninstall so the next `process_input` can see the new state.
    ///
    /// Silently ignores skills that fail to load — individual failures are
    /// recorded in the loader's `LoadReport.failures`, which we currently
    /// discard at this boundary. A broken skill should not take the
    /// conversation engine down with it. Returns the number of community
    /// skills successfully registered so the caller can log / surface it.
    pub fn reload_community_skills(
        &self,
        skill_store_dir: String,
        storage_dir: String,
    ) -> u32 {
        let mut fresh = build_engine_with_builtins();
        // Start from the shared default LoadOptions (host caps, HTTP, storage)
        // and override the log sink with whatever the host installed at
        // construction time. Install/validation paths elsewhere keep the
        // NullLogSink default — those paths don't execute skills, so the
        // sink there only ever sees load-time diagnostics the loader
        // currently doesn't emit.
        let mut options = android_load_options(&storage_dir);
        options.log_sink = self.log_sink.clone();
        options.tasks_provider = self.tasks_provider.clone();
        options.calendar_provider = self.calendar_provider.clone();
        options.local_clock = self.local_clock.clone();
        let loaded: u32 =
            match load_skill_directory_with(&PathBuf::from(&skill_store_dir), &options) {
                Ok(report) => {
                    let n = report.skills.len() as u32;
                    for skill in report.skills {
                        fresh.register_skill(skill);
                    }
                    n
                }
                Err(_) => 0,
            };
        *self.inner.lock().expect("engine mutex poisoned") = fresh;
        loaded
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_creates_and_responds_to_greeting() {
        let engine = AriEngine::new();
        let resp = engine.process_input("hello".to_string());
        match resp {
            FfiResponse::Text { body } => {
                assert!(!body.is_empty());
                assert_ne!(body, "Sorry, I didn't understand that.");
            }
            _ => panic!("expected Text response for greeting"),
        }
    }

    #[test]
    fn engine_returns_time() {
        let engine = AriEngine::new();
        let resp = engine.process_input("what time is it".to_string());
        match resp {
            FfiResponse::Text { body } => {
                assert!(body.starts_with("It's "), "response was: {body}");
            }
            _ => panic!("expected Text response for time"),
        }
    }

    #[test]
    fn engine_returns_calculation() {
        let engine = AriEngine::new();
        let resp = engine.process_input("calculate 5 + 3".to_string());
        match resp {
            FfiResponse::Text { body } => assert_eq!(body, "8"),
            _ => panic!("expected Text response for calculation"),
        }
    }

    #[test]
    fn engine_returns_action_for_open() {
        let engine = AriEngine::new();
        let resp = engine.process_input("open spotify".to_string());
        match resp {
            FfiResponse::Action { json, skill_id } => {
                let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
                assert_eq!(parsed["v"], 1);
                assert_eq!(parsed["launch_app"], "spotify");
                assert_eq!(skill_id, "open");
            }
            _ => panic!("expected Action response for open"),
        }
    }

    #[test]
    fn engine_returns_not_understood_for_gibberish() {
        let engine = AriEngine::new();
        let resp = engine.process_input("asdfghjkl".to_string());
        match resp {
            FfiResponse::NotUnderstood { body } => {
                assert_eq!(body, "Sorry, I didn't understand that.");
            }
            _ => panic!("expected NotUnderstood fallback"),
        }
    }
}
