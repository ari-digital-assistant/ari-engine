#![allow(clippy::new_without_default)]

use ari_engine::{fallback_response_for, Engine, EnvelopeSink, FALLBACK_RESPONSE};
use ari_skill_loader::assistant::{ConfigStore, MemoryConfigStore};
use ari_skill_loader::{
    load_skill_directory_with, AuthorizeInput, AuthorizeOutput, AuthorizeProvider, Calendar,
    CalendarEventRow, CalendarProvider, Capability, EnglishLocaleProvider, HostCapabilities,
    HttpConfig, InsertCalendarEventParams, InsertTaskParams, LoadOptions, LocalClock,
    LocalTimeComponents, LocaleProvider, LocationProvider, LocationResult, LocationStatus, LogLevel,
    LogSink, NullAuthorizeProvider, NullCalendarProvider, NullLocationProvider, NullLogSink,
    MediaServicesProvider, NullMediaServicesProvider, NullSettingWriter, NullTasksProvider,
    SettingWriter, StorageConfig, TaskList, TaskRow, TasksProvider, UtcLocalClock,
};
use ari_skills::{
    CalculatorSkill, CurrentTimeSkill, DateSkill, GreetingSkill, MusicSkill, OpenSkill, SearchSkill,
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
        .with(Capability::Calendar)
        .with(Capability::Location)
        .with(Capability::Authorize)
        .with(Capability::MediaServices);
    LoadOptions {
        log_sink: Arc::new(NullLogSink),
        host_capabilities: host_caps,
        http_config: HttpConfig::strict(),
        storage_config: StorageConfig::new(PathBuf::from(storage_dir)),
        tasks_provider: Arc::new(NullTasksProvider),
        calendar_provider: Arc::new(NullCalendarProvider),
        location_provider: Arc::new(NullLocationProvider),
        media_services_provider: Arc::new(NullMediaServicesProvider),
        local_clock: Arc::new(UtcLocalClock),
        config_store: Arc::new(MemoryConfigStore::new()),
        locale_provider: Arc::new(EnglishLocaleProvider),
        setting_writer: Arc::new(NullSettingWriter),
        authorize_provider: Arc::new(NullAuthorizeProvider),
        // Persist cranelift output so skill compilation is a one-time cost
        // per skill version, not a per-launch one. Without this, every
        // process start recompiled every installed WASM skill on the main
        // thread — the root of the startup ANR. The dir sits under the
        // app-private storage root the host already hands us.
        compile_cache_dir: Some(PathBuf::from(storage_dir).join("wasm-cache")),
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

/// Callback interface the host implements to receive envelopes the
/// engine produces outside the synchronous `process_input` flow —
/// currently only the phase-2 envelope from a Layer C assistant
/// round-trip. Rust calls this from a background thread, so the host
/// implementation must be safe to invoke off the UI thread and is
/// responsible for dispatching back to the UI/conversation pipeline
/// itself (e.g. by emitting on a `SharedFlow` the viewmodel observes).
#[uniffi::export(with_foreign)]
pub trait FfiEnvelopeSink: Send + Sync {
    /// Push a JSON-serialised envelope plus the emitting skill id
    /// (`None` for engine-origin envelopes). The skill id is what the
    /// frontend uses to resolve `asset:<path>` references inside the
    /// envelope back to the emitting skill's bundle directory — same
    /// contract as synchronous `FfiResponse::Action.skill_id`.
    fn push(&self, envelope_json: String, skill_id: Option<String>);
}

/// Wraps a foreign [`FfiEnvelopeSink`] so it satisfies the engine's
/// internal [`EnvelopeSink`] trait. Same `&str`→`String` copy pattern
/// as [`ForeignLogSinkAdapter`] — envelope push isn't on the hot path.
struct ForeignEnvelopeSinkAdapter(Arc<dyn FfiEnvelopeSink>);

impl EnvelopeSink for ForeignEnvelopeSinkAdapter {
    fn push(&self, envelope_json: &str, skill_id: Option<&str>) {
        self.0
            .push(envelope_json.to_string(), skill_id.map(|s| s.to_string()));
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

/// Row returned by [`FfiTasksProvider::query_in_range`]. Mirrors
/// [`ari_skill_loader::TaskRow`] across the UniFFI boundary.
#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiTaskRow {
    pub id: u64,
    pub title: String,
    pub due_ms: i64,
    pub due_all_day: bool,
    pub list_id: u64,
}

/// Row returned by [`FfiCalendarProvider::query_in_range`]. Mirrors
/// [`ari_skill_loader::CalendarEventRow`].
#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiCalendarEventRow {
    pub id: u64,
    pub title: String,
    pub start_ms: i64,
    pub end_ms: i64,
    pub all_day: bool,
    pub calendar_id: u64,
}

#[derive(Debug, Clone, Copy, uniffi::Enum)]
pub enum FfiLocationStatus {
    Ok,
    PermissionDenied,
    Unavailable,
    Timeout,
}

#[derive(Debug, Clone, Copy, uniffi::Record)]
pub struct FfiLocationResult {
    pub status: FfiLocationStatus,
    pub lat: f64,
    pub lon: f64,
    pub accuracy_m: f64,
    pub timestamp_ms: i64,
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
    /// Tasks with due time in `[start_ms, end_ms)`, ordered by due
    /// time ascending and capped at `limit`. Implementers must
    /// exclude untimed tasks (no due date set).
    fn query_in_range(
        &self,
        start_ms: i64,
        end_ms: i64,
        limit: u32,
    ) -> Vec<FfiTaskRow>;
}

/// Foreign-implemented calendar provider.
#[uniffi::export(with_foreign)]
pub trait FfiCalendarProvider: Send + Sync {
    fn has_write_permission(&self) -> bool;
    fn list_calendars(&self) -> Vec<FfiCalendar>;
    fn insert(&self, params: FfiInsertCalendarEventParams) -> u64;
    fn delete(&self, id: u64) -> bool;
    /// Event instances starting in `[start_ms, end_ms)`, ordered by
    /// start time ascending and capped at `limit`. Recurring events
    /// expand to one row per instance whose start lands in range.
    fn query_in_range(
        &self,
        start_ms: i64,
        end_ms: i64,
        limit: u32,
    ) -> Vec<FfiCalendarEventRow>;
}

/// Foreign-implemented coarse location provider.
#[uniffi::export(with_foreign)]
pub trait FfiLocationProvider: Send + Sync {
    /// Coarse fix; returns a cached last-known location no older than
    /// `max_age_ms` else a single fresh fix, giving up after `timeout_ms`.
    fn current(&self, max_age_ms: i64, timeout_ms: i64) -> FfiLocationResult;
}

/// Foreign-implemented wall-clock reader. Needed so skills can
/// resolve weekdays / "today" / local dates — WASM has no TZ
/// database, the host does.
#[uniffi::export(with_foreign)]
pub trait FfiLocalClock: Send + Sync {
    fn now_components(&self) -> FfiLocalTimeComponents;
    fn timezone_id(&self) -> String;
}

/// Foreign-implemented locale reader. The host's settings store is the
/// single source of truth for the user's currently-active language.
/// Engine code reads through this trait whenever it needs to dispatch
/// on locale (text normalisers, prompt selection, skill regex
/// filtering). Implementations must be cheap — called on every utterance.
#[uniffi::export(with_foreign)]
pub trait FfiLocaleProvider: Send + Sync {
    /// ISO 639-1 lowercase language code (e.g. `"en"`, `"it"`).
    fn current_locale(&self) -> String;
}

/// Foreign-implemented setting writer. The frontend persists the value
/// durably (encrypted for secret fields) and updates the in-memory
/// settings mirror so a later `setting_get` sees it.
#[uniffi::export(with_foreign)]
pub trait FfiSettingWriter: Send + Sync {
    fn set_value(&self, skill_id: String, key: String, value: String, is_secret: bool) -> bool;
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
    fn query_in_range(&self, start_ms: i64, end_ms: i64, limit: u32) -> Vec<TaskRow> {
        self.0
            .query_in_range(start_ms, end_ms, limit)
            .into_iter()
            .map(|r| TaskRow {
                id: r.id,
                title: r.title,
                due_ms: r.due_ms,
                due_all_day: r.due_all_day,
                list_id: r.list_id,
            })
            .collect()
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
    fn query_in_range(&self, start_ms: i64, end_ms: i64, limit: u32) -> Vec<CalendarEventRow> {
        self.0
            .query_in_range(start_ms, end_ms, limit)
            .into_iter()
            .map(|r| CalendarEventRow {
                id: r.id,
                title: r.title,
                start_ms: r.start_ms,
                end_ms: r.end_ms,
                all_day: r.all_day,
                calendar_id: r.calendar_id,
            })
            .collect()
    }
}

struct ForeignLocationProviderAdapter(Arc<dyn FfiLocationProvider>);

impl LocationProvider for ForeignLocationProviderAdapter {
    fn current(&self, max_age_ms: i64, timeout_ms: i64) -> LocationResult {
        let r = self.0.current(max_age_ms, timeout_ms);
        let status = match r.status {
            FfiLocationStatus::Ok => LocationStatus::Ok,
            FfiLocationStatus::PermissionDenied => LocationStatus::PermissionDenied,
            FfiLocationStatus::Unavailable => LocationStatus::Unavailable,
            FfiLocationStatus::Timeout => LocationStatus::Timeout,
        };
        LocationResult {
            status,
            lat: r.lat,
            lon: r.lon,
            accuracy_m: r.accuracy_m,
            timestamp_ms: r.timestamp_ms,
        }
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

struct ForeignLocaleProviderAdapter(Arc<dyn FfiLocaleProvider>);

impl LocaleProvider for ForeignLocaleProviderAdapter {
    fn current_locale(&self) -> String {
        self.0.current_locale()
    }
}

struct ForeignSettingWriterAdapter(Arc<dyn FfiSettingWriter>);

impl SettingWriter for ForeignSettingWriterAdapter {
    fn set_value(&self, skill_id: &str, key: &str, value: &str, is_secret: bool) -> bool {
        self.0
            .set_value(skill_id.to_string(), key.to_string(), value.to_string(), is_secret)
    }
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiAuthorizeRequest {
    pub auth_url: String,
    pub redirect_uri: String,
    pub timeout_ms: u64,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiAuthorizeParam {
    pub key: String,
    pub value: String,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiAuthorizeResult {
    pub ok: bool,
    pub params: Vec<FfiAuthorizeParam>,
    pub error: Option<String>,
}

/// Foreign-implemented browser round-trip. Opens `auth_url`, waits for the
/// redirect to `redirect_uri`, returns the callback params.
#[uniffi::export(with_foreign)]
pub trait FfiAuthorizeProvider: Send + Sync {
    fn authorize(&self, req: FfiAuthorizeRequest) -> FfiAuthorizeResult;
    fn redirect_uri(&self) -> String;
}

struct ForeignAuthorizeProviderAdapter(Arc<dyn FfiAuthorizeProvider>);

impl AuthorizeProvider for ForeignAuthorizeProviderAdapter {
    fn authorize(&self, input: AuthorizeInput) -> AuthorizeOutput {
        let res = self.0.authorize(FfiAuthorizeRequest {
            auth_url: input.auth_url,
            redirect_uri: input.redirect_uri,
            timeout_ms: input.timeout_ms,
        });
        AuthorizeOutput {
            ok: res.ok,
            params: res.params.into_iter().map(|p| (p.key, p.value)).collect(),
            error: res.error,
        }
    }
    fn redirect_uri(&self) -> String {
        self.0.redirect_uri()
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

/// Result of a settings-time skill invocation, mirrored from
/// [`ari_core::SettingsQueryResult`] across the UniFFI boundary. `options`
/// reuses [`FfiSelectOption`] (the same record `dynamic_select` config
/// fields expose), so the frontend can render query results with the same
/// option-list UI it already uses for static selects.
#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiSettingsQueryResult {
    pub ok: bool,
    pub error: Option<String>,
    pub options: Vec<FfiSelectOption>,
    pub message: Option<String>,
    pub refresh: bool,
}

/// Convert an engine-side [`ari_core::SettingsQueryResult`] into the
/// UniFFI-exportable [`FfiSettingsQueryResult`]. The engine's
/// `SettingsOption` only carries `value`/`label`; the richer
/// `FfiSelectOption` download fields don't apply to query results, so
/// they're `None`.
pub(crate) fn map_settings_result(
    r: ari_core::SettingsQueryResult,
) -> FfiSettingsQueryResult {
    FfiSettingsQueryResult {
        ok: r.ok,
        error: r.error,
        message: r.message,
        refresh: r.refresh,
        options: r
            .options
            .into_iter()
            .map(|o| FfiSelectOption {
                value: o.value,
                label: o.label,
                download_url: None,
                download_bytes: None,
            })
            .collect(),
    }
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
    pub(crate) location_provider: Arc<dyn LocationProvider>,
    pub(crate) media_services_provider: Arc<dyn MediaServicesProvider>,
    pub(crate) local_clock: Arc<dyn LocalClock>,
    /// Locale source of truth — engine reads through this whenever it
    /// needs to dispatch on language. Defaults to [`EnglishLocaleProvider`]
    /// for callers that don't supply a real one (CLI, tests). The
    /// Android host wires a real provider that reads the user's
    /// chosen language from the frontend DataStore.
    pub(crate) locale_provider: Arc<dyn LocaleProvider>,
    /// Config store backing `ari::setting_get` in WASM skills.
    /// Defaults to an empty in-memory map; the Android host passes
    /// the shared `SkillSettingsStore`'s inner map so skills see
    /// live UI-written values.
    pub(crate) config_store: Arc<dyn ConfigStore>,
    /// Setting writer backing `ari::setting_set` in WASM skills.
    /// Defaults to [`NullSettingWriter`] (no-op) for callers that don't
    /// supply a real one (CLI, tests). The Android host wires a writer
    /// that persists durably and updates the in-memory settings mirror.
    pub(crate) setting_writer: Arc<dyn SettingWriter>,
    /// Authorize provider backing `ari::authorize` in WASM skills.
    /// Defaults to [`NullAuthorizeProvider`] (no browser) for callers that
    /// don't supply a real one (CLI, tests). The Android host wires a
    /// provider that opens the system browser and returns callback params.
    pub(crate) authorize_provider: Arc<dyn AuthorizeProvider>,
    /// Envelope sink the engine uses to push phase-2 Layer C envelopes
    /// asynchronously. Stored here (not just on [`Engine`]) so
    /// `reload_community_skills` can re-attach it to the fresh engine
    /// it swaps in — otherwise the first community-skill reload would
    /// silently disable Layer C for every session afterwards.
    pub(crate) envelope_sink: Option<Arc<dyn EnvelopeSink>>,
}

fn build_engine_with_builtins() -> Engine {
    let mut engine = Engine::new();
    engine.register_skill(Box::new(CurrentTimeSkill::new()));
    engine.register_skill(Box::new(DateSkill::new()));
    engine.register_skill(Box::new(CalculatorSkill::new()));
    engine.register_skill(Box::new(GreetingSkill::new()));
    engine.register_skill(Box::new(MusicSkill::new()));
    engine.register_skill(Box::new(OpenSkill::new()));
    engine.register_skill(Box::new(SearchSkill::new()));
    engine
}

#[uniffi::export]
impl AriEngine {
    #[uniffi::constructor]
    pub fn new() -> Self {
        let config_store: Arc<dyn ConfigStore> = Arc::new(MemoryConfigStore::new());
        let mut engine = build_engine_with_builtins();
        engine.set_config_store(Some(config_store.clone()));
        Self {
            inner: Mutex::new(engine),
            log_sink: Arc::new(NullLogSink),
            tasks_provider: Arc::new(NullTasksProvider),
            calendar_provider: Arc::new(NullCalendarProvider),
            location_provider: Arc::new(NullLocationProvider),
            media_services_provider: Arc::new(NullMediaServicesProvider),
            local_clock: Arc::new(UtcLocalClock),
            locale_provider: Arc::new(EnglishLocaleProvider),
            config_store,
            setting_writer: Arc::new(NullSettingWriter),
            authorize_provider: Arc::new(NullAuthorizeProvider),
            envelope_sink: None,
        }
    }

    /// Construct with a host-supplied log sink for WASM skill output.
    /// Android wires a sink that forwards to `android.util.Log`; callers
    /// that don't care about skill logs (tests, CLI smoke tests) use
    /// [`AriEngine::new`] instead.
    #[uniffi::constructor]
    pub fn with_log_sink(sink: Arc<dyn FfiLogSink>) -> Self {
        let log_sink: Arc<dyn LogSink> = Arc::new(ForeignLogSinkAdapter(sink));
        let config_store: Arc<dyn ConfigStore> = Arc::new(MemoryConfigStore::new());
        let mut engine = build_engine_with_builtins();
        engine.set_log_sink(Some(log_sink.clone()));
        engine.set_config_store(Some(config_store.clone()));
        Self {
            inner: Mutex::new(engine),
            log_sink,
            tasks_provider: Arc::new(NullTasksProvider),
            calendar_provider: Arc::new(NullCalendarProvider),
            location_provider: Arc::new(NullLocationProvider),
            media_services_provider: Arc::new(NullMediaServicesProvider),
            local_clock: Arc::new(UtcLocalClock),
            locale_provider: Arc::new(EnglishLocaleProvider),
            config_store,
            setting_writer: Arc::new(NullSettingWriter),
            authorize_provider: Arc::new(NullAuthorizeProvider),
            envelope_sink: None,
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
        location: Option<Arc<dyn FfiLocationProvider>>,
        clock: Option<Arc<dyn FfiLocalClock>>,
        settings: Option<Arc<SkillSettingsStore>>,
        envelope_sink: Option<Arc<dyn FfiEnvelopeSink>>,
        locale: Option<Arc<dyn FfiLocaleProvider>>,
        setting_writer: Option<Arc<dyn FfiSettingWriter>>,
        authorize: Option<Arc<dyn FfiAuthorizeProvider>>,
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
        let location_provider: Arc<dyn LocationProvider> = match location {
            Some(l) => Arc::new(ForeignLocationProviderAdapter(l)),
            None => Arc::new(NullLocationProvider),
        };
        let local_clock: Arc<dyn LocalClock> = match clock {
            Some(c) => Arc::new(ForeignLocalClockAdapter(c)),
            None => Arc::new(UtcLocalClock),
        };
        let locale_provider: Arc<dyn LocaleProvider> = match locale {
            Some(l) => Arc::new(ForeignLocaleProviderAdapter(l)),
            None => Arc::new(EnglishLocaleProvider),
        };
        let config_store: Arc<dyn ConfigStore> = match settings {
            Some(s) => s.as_config_store(),
            None => Arc::new(MemoryConfigStore::new()),
        };
        let setting_writer: Arc<dyn SettingWriter> = match setting_writer {
            Some(w) => Arc::new(ForeignSettingWriterAdapter(w)),
            None => Arc::new(NullSettingWriter),
        };
        let authorize_provider: Arc<dyn AuthorizeProvider> = match authorize {
            Some(a) => Arc::new(ForeignAuthorizeProviderAdapter(a)),
            None => Arc::new(NullAuthorizeProvider),
        };
        let adapted_envelope_sink: Option<Arc<dyn EnvelopeSink>> = envelope_sink
            .map(|es| Arc::new(ForeignEnvelopeSinkAdapter(es)) as Arc<dyn EnvelopeSink>);
        let mut engine = build_engine_with_builtins();
        engine.set_log_sink(Some(log_sink.clone()));
        engine.set_config_store(Some(config_store.clone()));
        if let Some(ref es) = adapted_envelope_sink {
            engine.set_envelope_sink(Some(es.clone()));
        }
        Self {
            inner: Mutex::new(engine),
            log_sink,
            tasks_provider,
            calendar_provider,
            location_provider,
            media_services_provider: Arc::new(NullMediaServicesProvider),
            local_clock,
            locale_provider,
            config_store,
            setting_writer,
            authorize_provider,
            envelope_sink: adapted_envelope_sink,
        }
    }

    /// The user's currently-active language, as seen by the engine.
    /// Reads through the [`LocaleProvider`] the host wired up at
    /// construction time. ISO 639-1 lowercase (e.g. `"en"`, `"it"`).
    ///
    /// Cheap to call — DataStore-backed implementations cache the
    /// latest value and read it without blocking.
    pub fn current_locale(&self) -> String {
        self.locale_provider.current_locale()
    }

    /// Settings-time skill invocation: run `skill_id`'s `settings_query` for
    /// `field`, passing the current `values` (the field's `depends_on`
    /// siblings). The host calls this from the settings UI to populate a
    /// `dynamic_select` field's options or to validate a field whose value
    /// depends on a server round-trip.
    pub fn query_skill_setting(
        &self,
        skill_id: String,
        field: String,
        values: std::collections::HashMap<String, String>,
    ) -> FfiSettingsQueryResult {
        // The UI masks `secret` fields as a bullet placeholder and never
        // round-trips their real value across the FFI (see
        // `SkillRegistry::get_skill_settings`). So `values` carries
        // "••••••••" for a token, which the skill would otherwise forward
        // upstream verbatim and get rejected. Resolve each dep from the
        // shared config store — the same source `ari::setting_get` and the
        // execute path read — so settings-time queries see the real
        // committed values. UI-passed values stay as a fallback for keys
        // not yet persisted to the store.
        let mut merged = values;
        for (key, slot) in merged.iter_mut() {
            if let Some(real) = self.config_store.get(&skill_id, key.as_str()) {
                *slot = real;
            }
        }
        let values_json = serde_json::to_string(&merged).unwrap_or_else(|_| "{}".to_string());
        let engine = self.inner.lock().expect("engine mutex poisoned");
        map_settings_result(engine.query_skill_setting(&skill_id, &field, &values_json))
    }

    /// Effectful settings-time skill invocation: run `skill_id`'s `settings_action`
    /// for `action`, passing the current `values` (sibling field values the
    /// skill reads during the action — e.g. `base_url`/`token` for HA sign-in).
    /// Secret-masked values are resolved from the config store before the call,
    /// identical to the resolution step in [`AriEngine::query_skill_setting`].
    pub fn settings_action(
        &self,
        skill_id: String,
        action: String,
        values: std::collections::HashMap<String, String>,
    ) -> FfiSettingsQueryResult {
        let mut merged = values;
        for (key, slot) in merged.iter_mut() {
            if let Some(real) = self.config_store.get(&skill_id, key.as_str()) {
                *slot = real;
            }
        }
        let values_json = serde_json::to_string(&merged).unwrap_or_else(|_| "{}".to_string());
        let engine = self.inner.lock().expect("engine mutex poisoned");
        map_settings_result(engine.settings_action(&skill_id, &action, &values_json))
    }

    pub fn process_input(&self, input: String) -> FfiResponse {
        // Refresh the engine's locale on every call so per-locale
        // skill scorers and responses pick up live changes from the
        // frontend's settings store. The locale provider's read is a
        // synchronous AtomicReference lookup on Android (see
        // AriFfiLocaleProvider), so this is essentially free.
        let locale = self.locale_provider.current_locale();
        let mut engine = self.inner.lock().expect("engine mutex poisoned");
        engine.set_locale(locale.clone());
        let (response, skill_id) = engine.process_input_with_skill(&input);
        match response {
            ari_core::Response::Text(s) => {
                // The engine's fallback text is locale-specific
                // (Phase 5 step 2). Compare against the locale-
                // appropriate version so the Android NotUnderstood
                // retry path fires for Italian / French / etc.
                // users too — not just English.
                let is_fallback =
                    s == fallback_response_for(&locale) || s == FALLBACK_RESPONSE;
                if is_fallback {
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
        engine.set_llm(std::sync::Arc::new(lazy));
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
        // Re-attach the engine-level sinks the host installed at
        // construction time. Without this, the fresh Engine starts
        // with `log_sink = None` and `envelope_sink = None`, which
        // silently disables both the engine's diagnostic log stream
        // and Layer C phase-2 push for every session after the first
        // reload_community_skills call (that is: for every session
        // at all on Android, since EngineModule always reloads).
        fresh.set_log_sink(Some(self.log_sink.clone()));
        fresh.set_config_store(Some(self.config_store.clone()));
        if let Some(ref es) = self.envelope_sink {
            fresh.set_envelope_sink(Some(es.clone()));
        }
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
        options.location_provider = self.location_provider.clone();
        options.media_services_provider = self.media_services_provider.clone();
        options.local_clock = self.local_clock.clone();
        options.config_store = self.config_store.clone();
        options.locale_provider = self.locale_provider.clone();
        options.setting_writer = self.setting_writer.clone();
        options.authorize_provider = self.authorize_provider.clone();
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

    struct OkWriter;
    impl FfiSettingWriter for OkWriter {
        fn set_value(&self, _skill_id: String, _key: String, _value: String, _is_secret: bool) -> bool { true }
    }

    #[test]
    fn setting_writer_adapter_delegates_with_secret_flag() {
        use ari_skill_loader::platform_capabilities::SettingWriter;
        let adapter = ForeignSettingWriterAdapter(std::sync::Arc::new(OkWriter));
        assert!(adapter.set_value("s", "k", "v", true));
    }

    struct CancelProvider;
    impl FfiAuthorizeProvider for CancelProvider {
        fn authorize(&self, _req: FfiAuthorizeRequest) -> FfiAuthorizeResult {
            FfiAuthorizeResult {
                ok: false,
                params: vec![],
                error: Some("cancelled".into()),
            }
        }
        fn redirect_uri(&self) -> String {
            "https://heyari.dev/oauth/callback".into()
        }
    }

    #[test]
    fn authorize_adapter_maps_result() {
        use ari_skill_loader::platform_capabilities::{AuthorizeInput, AuthorizeProvider};
        let adapter = ForeignAuthorizeProviderAdapter(std::sync::Arc::new(CancelProvider));
        let out = adapter.authorize(AuthorizeInput {
            auth_url: "u".into(),
            redirect_uri: "r".into(),
            timeout_ms: 5,
        });
        assert_eq!(out.ok, false);
        assert_eq!(out.error.as_deref(), Some("cancelled"));
    }

    struct DeniedLocation;
    impl FfiLocationProvider for DeniedLocation {
        fn current(&self, _max_age_ms: i64, _timeout_ms: i64) -> FfiLocationResult {
            FfiLocationResult {
                status: FfiLocationStatus::PermissionDenied,
                lat: 0.0,
                lon: 0.0,
                accuracy_m: 0.0,
                timestamp_ms: 0,
            }
        }
    }

    #[test]
    fn location_adapter_maps_status_and_coords() {
        use ari_skill_loader::platform_capabilities::{LocationProvider, LocationStatus};
        let adapter = ForeignLocationProviderAdapter(std::sync::Arc::new(DeniedLocation));
        let r = adapter.current(600_000, 5_000);
        assert_eq!(r.status, LocationStatus::PermissionDenied);
        assert_eq!(r.lat, 0.0);
        assert_eq!(r.timestamp_ms, 0);
    }

    #[test]
    fn map_settings_result_carries_options_and_error() {
        let r = ari_core::SettingsQueryResult {
            ok: true,
            error: None,
            message: None,
            refresh: false,
            options: vec![ari_core::SettingsOption {
                value: "v".into(),
                label: "L".into(),
            }],
        };
        let f = super::map_settings_result(r);
        assert_eq!(f.ok, true);
        assert_eq!(f.options[0].value, "v");
        assert_eq!(f.options[0].label, "L");
    }

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

    #[test]
    fn ffi_settings_action_unknown_skill_is_error() {
        let engine = AriEngine::new();
        let r = engine.settings_action(
            "nope".into(),
            "sign_in".into(),
            std::collections::HashMap::new(),
        );
        assert_eq!(r.ok, false);
        assert!(r.error.is_some());
    }

    #[test]
    fn android_host_grants_authorize_capability() {
        let opts = android_load_options("/tmp/ignored");
        assert!(opts.host_capabilities.provides(Capability::Authorize));
    }
}
