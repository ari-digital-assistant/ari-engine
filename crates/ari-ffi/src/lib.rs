#![allow(clippy::new_without_default)]

use ari_engine::{AnswerSink, ConversationLength, ConversationMemory, EnvelopeSink};
use ari_runtime::{Providers, Reply, Runtime};
use ari_skill_loader::{
    AuthorizeInput, AuthorizeOutput, AuthorizeProvider, Calendar,
    CalendarEventRow, CalendarProvider, Capability, Contact, ContactChannel, ContactsProvider,
    LiveConversationsProvider, HostCapabilities,
    InsertCalendarEventParams, InsertTaskParams, LoadOptions, LocalClock,
    LocalTimeComponents, LocaleProvider, LocationProvider, LocationResult, LocationStatus, LogLevel,
    LogSink, MediaServicesProvider,
    SettingWriter, TaskList, TaskRow, TasksProvider,
};
use std::path::Path;
use std::sync::{Arc, Mutex};

mod assistant_registry;
mod envelope;
mod settings_store;
mod skill_registry;
mod voice;
mod voice_pipeline;

pub use assistant_registry::{
    AssistantRegistry, FfiAssistantEntry, FfiConfigField, FfiSelectOption,
};
pub use settings_store::SkillSettingsStore;
pub use skill_registry::{
    FfiBrowseEntry, FfiInstalledSkill, FfiRegistryError, FfiSkillUpdate, SkillRegistry,
};

/// What Android can run: every capability there is.
fn android_host_capabilities() -> HostCapabilities {
    HostCapabilities::pure_frontend()
        .with(Capability::Http)
        .with(Capability::StorageKv)
        .with(Capability::Tasks)
        .with(Capability::Calendar)
        .with(Capability::Location)
        .with(Capability::Authorize)
        .with(Capability::MediaServices)
        .with(Capability::Contacts)
        .with(Capability::Reply)
}

/// The [`LoadOptions`] the Android host uses for every install and every
/// reload (see [`ari_runtime::load_options`]).
pub(crate) fn android_load_options(storage_dir: &str) -> LoadOptions {
    ari_runtime::load_options(Path::new(storage_dir), android_host_capabilities())
}

uniffi::setup_scaffolding!();

/// The engine version a frontend should quote in a bug report.
///
/// This is `ari-ffi`'s own crate version, and that is deliberate: the engine's
/// crates are not versioned in lockstep, so there is no single number that
/// describes "the engine". What every frontend actually links is this crate,
/// so its version is the one that identifies the boundary a bug was reported
/// against. Bump it when the FFI surface changes.
#[uniffi::export]
pub fn engine_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// The CPU the on-device assistant runs on, for bug reports and the debug
/// screen.
#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiCpuInfo {
    /// The CPU features this device reports that LiteRT-LM's CPU kernels
    /// can use.
    pub features: Vec<String>,
}

/// This CPU's features (ari-android#32).
#[uniffi::export]
pub fn on_device_cpu_info() -> FfiCpuInfo {
    #[cfg(feature = "llm")]
    {
        FfiCpuInfo { features: ari_llm::cpu_features() }
    }
    #[cfg(not(feature = "llm"))]
    {
        FfiCpuInfo { features: Vec::new() }
    }
}

/// One of Ari's built-in commands, which typed input runs as `/name`.
#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiSlashCommand {
    /// What follows the `/`, the same in every language.
    pub name: String,
    pub description: String,
    /// Whether the command needs words after it ("/remember I like tea").
    pub takes_argument: bool,
}

/// The built-in `/` commands, described in `locale`, for a frontend to offer
/// as the user types.
#[uniffi::export]
pub fn slash_commands(locale: String) -> Vec<FfiSlashCommand> {
    ari_engine::slash_command::SlashCommand::ALL
        .into_iter()
        .map(|c| FfiSlashCommand {
            name: c.name().to_string(),
            description: c.description_for(&locale).to_string(),
            takes_argument: c.takes_argument(),
        })
        .collect()
}

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

/// Callback interface the host implements to get an on-device assistant
/// answer while the model is still writing it, via
/// [`AriEngine::process_input_streaming`]: the words so far to show, and each
/// finished sentence to speak. Same contract as the engine's
/// [`ari_engine::AnswerSink`]: a host that gets any sentence gets the whole
/// answer as sentences, so it speaks those instead of the response text.
#[uniffi::export(with_foreign)]
pub trait FfiAnswerSink: Send + Sync {
    /// The answer so far, in whole words, and as markdown once it formats
    /// anything. Each call replaces the last.
    fn text(&self, so_far: String, markdown: Option<String>);
    /// The next finished sentence.
    fn sentence(&self, text: String);
}

/// Wraps a foreign [`FfiAnswerSink`] so it satisfies the engine's
/// [`AnswerSink`]. Same `&str`→`String` copy as the other adapters.
struct ForeignAnswerSinkAdapter(Arc<dyn FfiAnswerSink>);

impl AnswerSink for ForeignAnswerSinkAdapter {
    fn text(&self, so_far: &str, markdown: Option<&str>) {
        self.0.text(so_far.to_string(), markdown.map(str::to_string));
    }
    fn sentence(&self, text: &str) {
        self.0.sentence(text.to_string());
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

/// One installed launchable app, pushed into the engine so scoring can tell
/// "open <app>" from "open <smart-home device>". Mirrors
/// [`ari_core::AppEntry`] across the UniFFI boundary.
#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiAppEntry {
    pub label: String,
    pub package: String,
}

impl From<FfiAppEntry> for ari_core::AppEntry {
    fn from(a: FfiAppEntry) -> Self {
        ari_core::AppEntry { label: a.label, package: a.package }
    }
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

/// Whether Ari still remembers the conversation; see [`ConversationMemory`].
#[derive(Debug, Clone, Copy, uniffi::Enum)]
pub enum FfiConversationMemory {
    None,
    Live { forgotten_in_ms: u64 },
    Lapsed,
}

impl From<ConversationMemory> for FfiConversationMemory {
    fn from(memory: ConversationMemory) -> Self {
        match memory {
            ConversationMemory::None => FfiConversationMemory::None,
            ConversationMemory::Live(left) => FfiConversationMemory::Live {
                forgotten_in_ms: left.as_millis() as u64,
            },
            ConversationMemory::Lapsed => FfiConversationMemory::Lapsed,
        }
    }
}

/// The user's conversation-length setting; see [`ConversationLength`].
#[derive(Debug, Clone, Copy, uniffi::Enum)]
pub enum FfiConversationLength {
    Short,
    Medium,
    Long,
}

impl From<FfiConversationLength> for ConversationLength {
    fn from(length: FfiConversationLength) -> Self {
        match length {
            FfiConversationLength::Short => ConversationLength::Short,
            FfiConversationLength::Medium => ConversationLength::Medium,
            FfiConversationLength::Long => ConversationLength::Long,
        }
    }
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

/// Foreign-implemented media-services provider. The host lists
/// whatever music/streaming apps are actually installed so the
/// music skill can validate play requests against reality.
#[uniffi::export(with_foreign)]
pub trait FfiMediaServicesProvider: Send + Sync {
    fn installed_services(&self) -> Vec<String>;
}

/// One way to reach a contact — a canonical service id and whatever
/// identifier that service addresses the person by.
#[derive(uniffi::Record)]
pub struct FfiContactChannel {
    pub service: String,
    pub id: String,
}

/// A person the user could message, and the ways they can be reached.
#[derive(uniffi::Record)]
pub struct FfiContact {
    pub display_name: String,
    pub channels: Vec<FfiContactChannel>,
}

/// Foreign-implemented address-book reader. **Lookup only** — there is no
/// "list every contact", by design: a skill asks about a name the user
/// already said and gets the matches, and can never walk the address book.
#[uniffi::export(with_foreign)]
pub trait FfiContactsProvider: Send + Sync {
    fn has_permission(&self) -> bool;
    fn lookup(&self, query: String) -> Vec<FfiContact>;
}

/// Foreign-implemented reader for conversations that can be replied into now.
///
/// Names only, deliberately. The frontend reads notifications to know this;
/// nothing it learns doing so — the message, the app, the pending intent —
/// crosses this boundary.
#[uniffi::export(with_foreign)]
pub trait FfiLiveConversationsProvider: Send + Sync {
    fn names(&self) -> Vec<String>;
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

struct ForeignMediaServicesProviderAdapter(Arc<dyn FfiMediaServicesProvider>);

impl MediaServicesProvider for ForeignMediaServicesProviderAdapter {
    fn installed_services(&self) -> Vec<String> {
        self.0.installed_services()
    }
}

struct ForeignContactsProviderAdapter(Arc<dyn FfiContactsProvider>);

impl ContactsProvider for ForeignContactsProviderAdapter {
    fn has_permission(&self) -> bool {
        self.0.has_permission()
    }

    fn lookup(&self, query: &str) -> Vec<Contact> {
        self.0
            .lookup(query.to_string())
            .into_iter()
            .map(|c| Contact {
                display_name: c.display_name,
                channels: c
                    .channels
                    .into_iter()
                    .map(|ch| ContactChannel { service: ch.service, id: ch.id })
                    .collect(),
            })
            .collect()
    }
}

struct ForeignLiveConversationsProviderAdapter(Arc<dyn FfiLiveConversationsProvider>);

impl LiveConversationsProvider for ForeignLiveConversationsProviderAdapter {
    fn names(&self) -> Vec<String> {
        self.0.names()
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

#[derive(Debug, uniffi::Enum)]
pub enum FfiResponse {
    /// `rearm` true means the engine is awaiting a spoken reply — the host
    /// should re-arm the mic without a wake word (see multi-turn design).
    /// `enter_conversation` true means the engine entered "let's talk"
    /// continuous-conversation mode this turn; `exit_conversation` true means
    /// it left the mode this turn — the host mirrors that state.
    /// `facts_changed` true means this turn mutated the engine's durable
    /// personal facts (a remember/forget) — the host should re-read
    /// `remembered_facts()` and persist the snapshot.
    /// `clear_conversation` true means the user cleared the conversation this
    /// turn ("clear history") and the engine has already forgotten it; the
    /// host clears its chat log to match.
    /// `markdown` is an assistant answer as written, for a host that renders
    /// markdown to show; `body` is the same answer as plain text, for
    /// speaking. `None` for anything else, and for an answer that formats
    /// nothing.
    Text { body: String, markdown: Option<String>, rearm: bool, enter_conversation: bool, exit_conversation: bool, facts_changed: bool, clear_conversation: bool },
    /// `skill_id` is the manifest id of the emitting skill (e.g.
    /// `dev.heyari.timer`), used by the frontend to resolve `asset:<path>`
    /// references back to the skill's bundle directory. Empty string if
    /// the engine couldn't attribute the response to a specific skill
    /// (phrase-tier actions, fallbacks) — treat that as "no bundle,
    /// asset references will fail to resolve".
    /// `rearm` true means the engine is awaiting a spoken reply — the host
    /// should re-arm the mic without a wake word (see multi-turn design).
    /// `enter_conversation` true means the engine entered "let's talk"
    /// continuous-conversation mode this turn; `exit_conversation` true means
    /// it left the mode this turn — the host mirrors that state.
    Action { json: String, skill_id: String, rearm: bool, enter_conversation: bool, exit_conversation: bool, facts_changed: bool, clear_conversation: bool },
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

/// Collected platform providers for [`AriEngineBuilder`]. Each is optional;
/// unset ones fall back to the Null/UTC defaults in `assemble_with_providers`.
#[derive(Default)]
struct EngineBuilderState {
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
    media_services: Option<Arc<dyn FfiMediaServicesProvider>>,
    contacts: Option<Arc<dyn FfiContactsProvider>>,
    live_conversations: Option<Arc<dyn FfiLiveConversationsProvider>>,
}

/// Builds an [`AriEngine`] one provider at a time. This exists specifically to
/// avoid a UniFFI/JNA arm64 calling-convention bug: a single constructor taking
/// all 11 providers passes 11 by-value `RustBuffer` structs, and JNA mishandles
/// the ones that spill onto the stack on AArch64 (benign on x86_64), crashing at
/// startup on real devices. Each setter here is one FFI call with <=2 args, so
/// nothing ever spills to the stack.
#[derive(uniffi::Object)]
pub struct AriEngineBuilder {
    state: Mutex<EngineBuilderState>,
}

#[uniffi::export]
impl AriEngineBuilder {
    #[uniffi::constructor]
    pub fn new() -> Arc<Self> {
        Arc::new(Self { state: Mutex::new(EngineBuilderState::default()) })
    }

    pub fn sink(&self, v: Arc<dyn FfiLogSink>) {
        self.state.lock().unwrap().sink = Some(v);
    }
    pub fn tasks(&self, v: Arc<dyn FfiTasksProvider>) {
        self.state.lock().unwrap().tasks = Some(v);
    }
    pub fn calendar(&self, v: Arc<dyn FfiCalendarProvider>) {
        self.state.lock().unwrap().calendar = Some(v);
    }
    pub fn location(&self, v: Arc<dyn FfiLocationProvider>) {
        self.state.lock().unwrap().location = Some(v);
    }
    pub fn clock(&self, v: Arc<dyn FfiLocalClock>) {
        self.state.lock().unwrap().clock = Some(v);
    }
    pub fn settings(&self, v: Arc<SkillSettingsStore>) {
        self.state.lock().unwrap().settings = Some(v);
    }
    pub fn envelope_sink(&self, v: Arc<dyn FfiEnvelopeSink>) {
        self.state.lock().unwrap().envelope_sink = Some(v);
    }
    pub fn locale(&self, v: Arc<dyn FfiLocaleProvider>) {
        self.state.lock().unwrap().locale = Some(v);
    }
    pub fn setting_writer(&self, v: Arc<dyn FfiSettingWriter>) {
        self.state.lock().unwrap().setting_writer = Some(v);
    }
    pub fn authorize(&self, v: Arc<dyn FfiAuthorizeProvider>) {
        self.state.lock().unwrap().authorize = Some(v);
    }
    pub fn media_services(&self, v: Arc<dyn FfiMediaServicesProvider>) {
        self.state.lock().unwrap().media_services = Some(v);
    }
    pub fn contacts(&self, v: Arc<dyn FfiContactsProvider>) {
        self.state.lock().unwrap().contacts = Some(v);
    }
    pub fn live_conversations(&self, v: Arc<dyn FfiLiveConversationsProvider>) {
        self.state.lock().unwrap().live_conversations = Some(v);
    }

    pub fn build(&self) -> Arc<AriEngine> {
        let mut s = self.state.lock().unwrap();
        Arc::new(assemble_with_providers(
            s.sink.take(),
            s.tasks.take(),
            s.calendar.take(),
            s.location.take(),
            s.clock.take(),
            s.settings.take(),
            s.envelope_sink.take(),
            s.locale.take(),
            s.setting_writer.take(),
            s.authorize.take(),
            s.media_services.take(),
            s.contacts.take(),
            s.live_conversations.take(),
        ))
    }
}

#[derive(uniffi::Object)]
pub struct AriEngine {
    // An Arc: the voice pipeline holds the runtime too, as its TurnEngine.
    pub(crate) runtime: Arc<Runtime>,
}

/// Internal assembler shared by [`AriEngineBuilder::build`]. NOT a UniFFI
/// entry point: exposing all 11 providers as one FFI call passes 11 by-value
/// `RustBuffer` structs, which JNA mis-marshals on arm64 (args spill to the
/// stack and get corrupted -> SIGSEGV at startup on real devices). The builder
/// sets providers one per call (<=2 args each) to stay within the register arg
/// budget; this free function does the actual assembly.
#[allow(clippy::too_many_arguments)]
fn assemble_with_providers(
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
    media_services: Option<Arc<dyn FfiMediaServicesProvider>>,
    contacts: Option<Arc<dyn FfiContactsProvider>>,
    live_conversations: Option<Arc<dyn FfiLiveConversationsProvider>>,
) -> AriEngine {
    let defaults = Providers::default();
    let providers = Providers {
        log_sink: match sink {
            Some(s) => Arc::new(ForeignLogSinkAdapter(s)),
            None => defaults.log_sink,
        },
        tasks: match tasks {
            Some(t) => Arc::new(ForeignTasksProviderAdapter(t)),
            None => defaults.tasks,
        },
        calendar: match calendar {
            Some(c) => Arc::new(ForeignCalendarProviderAdapter(c)),
            None => defaults.calendar,
        },
        location: match location {
            Some(l) => Arc::new(ForeignLocationProviderAdapter(l)),
            None => defaults.location,
        },
        media_services: match media_services {
            Some(p) => Arc::new(ForeignMediaServicesProviderAdapter(p)),
            None => defaults.media_services,
        },
        contacts: match contacts {
            Some(p) => Arc::new(ForeignContactsProviderAdapter(p)),
            None => defaults.contacts,
        },
        live_conversations: match live_conversations {
            Some(p) => Arc::new(ForeignLiveConversationsProviderAdapter(p)),
            None => defaults.live_conversations,
        },
        local_clock: match clock {
            Some(c) => Arc::new(ForeignLocalClockAdapter(c)),
            None => defaults.local_clock,
        },
        locale: match locale {
            Some(l) => Arc::new(ForeignLocaleProviderAdapter(l)),
            None => defaults.locale,
        },
        config_store: match settings {
            Some(s) => s.as_config_store(),
            None => defaults.config_store,
        },
        setting_writer: match setting_writer {
            Some(w) => Arc::new(ForeignSettingWriterAdapter(w)),
            None => defaults.setting_writer,
        },
        authorize: match authorize {
            Some(a) => Arc::new(ForeignAuthorizeProviderAdapter(a)),
            None => defaults.authorize,
        },
        envelope_sink: envelope_sink.map(|es| Arc::new(ForeignEnvelopeSinkAdapter(es)) as Arc<dyn EnvelopeSink>),
    };
    AriEngine { runtime: Arc::new(Runtime::new(providers)) }
}

#[uniffi::export]
impl AriEngine {
    #[uniffi::constructor]
    pub fn new() -> Self {
        Self { runtime: Arc::new(Runtime::new(Providers::default())) }
    }

    /// Construct with a host-supplied log sink for WASM skill output.
    /// Android wires a sink that forwards to `android.util.Log`; callers
    /// that don't care about skill logs (tests, CLI smoke tests) use
    /// [`AriEngine::new`] instead.
    #[uniffi::constructor]
    pub fn with_log_sink(sink: Arc<dyn FfiLogSink>) -> Self {
        let log_sink: Arc<dyn LogSink> = Arc::new(ForeignLogSinkAdapter(sink));
        Self { runtime: Arc::new(Runtime::new(Providers { log_sink, ..Providers::default() })) }
    }

    /// Construct with the full set of host-supplied platform
    /// providers. This is the constructor the Android frontend uses
    /// at startup so any skill that declares the `tasks`, `calendar`
    /// or clock capabilities gets real implementations rather than
    /// the Null defaults. Any provider argument can be left `None`
    /// to fall back to the corresponding Null/UTC default — useful
    /// for frontends that only wire up part of the surface.
    /// The user's currently-active language, as seen by the engine.
    /// Reads through the [`LocaleProvider`] the host wired up at
    /// construction time. ISO 639-1 lowercase (e.g. `"en"`, `"it"`).
    ///
    /// Cheap to call — DataStore-backed implementations cache the
    /// latest value and read it without blocking.
    pub fn current_locale(&self) -> String {
        self.runtime.current_locale()
    }

    /// Load the cached tier→model catalog written by
    /// [`crate::SkillRegistry::refresh_model_catalog`], so cloud assistant
    /// skills resolve their `fast`/`balanced`/`smartest` setting to a current
    /// model ID. Call this at startup with the path that method returned, and
    /// again after each refresh.
    ///
    /// Returns whether a catalog was installed. `false` is a normal state, not
    /// a failure — a first run has nothing cached yet, and skills fall back to
    /// the per-tier pins in their own manifests. The reason is logged either
    /// way, since a catalog that stops loading is otherwise invisible: the
    /// skills keep working, just on ageing pinned models.
    pub fn load_model_catalog(&self, path: String) -> bool {
        self.runtime.load_model_catalog(&path)
    }

    /// Where the store caches the registry index
    /// ([`crate::skill_registry::SkillRegistry::index_cache_path`]). Ari then
    /// offers a store skill that isn't installed, ahead of the assistant,
    /// when what the user says matches one of its strong trigger patterns
    /// (ari-android#44). The file is read again whenever it changes; until
    /// it exists, nothing is suggested.
    pub fn set_skill_suggestions_path(&self, path: String) {
        let mut engine = self.runtime.engine();
        engine.set_skill_suggestions_path(path, android_host_capabilities());
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
        map_settings_result(self.runtime.query_skill_setting(&skill_id, &field, values))
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
        map_settings_result(self.runtime.settings_action(&skill_id, &action, values))
    }

    pub fn process_input(&self, input: String) -> FfiResponse {
        self.respond(input, None)
    }

    /// [`Self::process_input`], handing an on-device assistant answer to
    /// `sink` while the model is still writing it: the words so far to show,
    /// and each finished sentence so the host can start speaking
    /// (ari-android#37). The sink is called on this thread, under the engine
    /// lock, so it must not call back into the engine.
    pub fn process_input_streaming(&self, input: String, sink: Arc<dyn FfiAnswerSink>) -> FfiResponse {
        self.respond(input, Some(Arc::new(ForeignAnswerSinkAdapter(sink))))
    }

    /// Discard any pending question the engine is awaiting a reply to. Called
    /// by the host when the re-armed mic times out, the user dismisses, or a
    /// fresh wake word starts a new session. No-op when nothing is pending.
    pub fn cancel_pending_reply(&self) {
        let engine = self.runtime.engine();
        engine.clear_pending_turn();
    }

    /// Tell the engine whether "let's talk" continuous-conversation mode is
    /// currently active. The frontend `VoiceSession` loop calls this — `true`
    /// on entry, `false` on every exit route (exit phrase, silence timeout, or
    /// error). While active the engine interprets exit phrases (a bare "stop"
    /// ends the mode instead of routing to a skill) and records skill turns
    /// into the conversation buffer.
    pub fn set_conversation_active(&self, active: bool) {
        self.runtime
            .engine()
            .set_conversation_active(active);
    }

    /// Master switch for conversation memory (cross-turn context + "Let's
    /// Talk" mode). Mirrors the Android `conversationMemoryEnabled` setting;
    /// hydrated at engine build and written through when the user toggles it.
    /// When `false` the engine retains no conversation buffer and refuses
    /// "let's talk" entry (guiding the user to the toggle instead).
    pub fn set_conversation_memory_enabled(&self, enabled: bool) {
        self.runtime
            .engine()
            .set_conversation_memory_enabled(enabled);
    }

    /// Forget the conversation so far, as a spoken "clear history" does. The
    /// frontend calls this when the user clears the chat by other means, so
    /// the assistant doesn't answer follow-ups about what was just cleared.
    pub fn clear_conversation(&self) {
        self.runtime
            .engine()
            .clear_conversation();
    }

    /// Whether Ari still remembers the conversation, and for how much longer,
    /// so the chat can say when it has been forgotten (ari-android#51).
    /// Reading it doesn't keep the conversation alive.
    pub fn conversation_memory(&self) -> FfiConversationMemory {
        self.runtime
            .engine()
            .conversation_memory()
            .into()
    }

    /// How much conversation the engine keeps, and for how long. Mirrors
    /// the Android `conversationLength` setting; hydrated at engine build and
    /// written through when the user changes it.
    pub fn set_conversation_length(&self, length: FfiConversationLength) {
        self.runtime
            .engine()
            .set_conversation_length(length.into());
    }

    /// How assistant answers should sound, in the user's words; blank for
    /// Ari's own voice. Mirrors the Android `personality` setting; hydrated
    /// at engine build and written through when the user edits it.
    pub fn set_personality(&self, personality: String) {
        self.runtime
            .engine()
            .set_personality(&personality);
    }

    /// Whether the on-device assistant may offer to ask a cloud assistant
    /// when it doesn't know. Mirrors the Android `handoffEnabled` setting;
    /// hydrated at engine build and written through when the user toggles it.
    pub fn set_handoff_enabled(&self, enabled: bool) {
        self.runtime
            .engine()
            .set_handoff_enabled(enabled);
    }

    /// Skill ids of the installed cloud assistants a handoff could offer,
    /// those with an API key, the one it would offer first leading. Empty
    /// means the handoff setting can't do anything yet.
    pub fn handoff_assistants(&self) -> Vec<String> {
        self.runtime
            .engine()
            .handoff_assistants()
    }

    /// The cloud assistant that answered last, for the frontend to persist
    /// after a turn: a handoff offers it first.
    pub fn last_cloud_assistant(&self) -> Option<String> {
        self.runtime
            .engine()
            .last_cloud_assistant()
    }

    /// Hydrate [`Self::last_cloud_assistant`] at engine build.
    pub fn set_last_cloud_assistant(&self, skill_id: Option<String>) {
        self.runtime
            .engine()
            .set_last_cloud_assistant(skill_id);
    }

    /// Replace the engine's durable personal facts. Mirrors the Android
    /// persisted store; hydrated at engine build and written through after a
    /// settings-screen edit.
    pub fn set_remembered_facts(&self, facts: Vec<String>) {
        self.runtime
            .engine()
            .set_remembered_facts(facts);
    }

    /// Replace the engine's snapshot of installed launchable apps. The frontend
    /// pushes this at build time and refreshes it on app install/uninstall.
    pub fn set_installed_apps(&self, apps: Vec<FfiAppEntry>) {
        self.runtime
            .engine()
            .set_installed_apps(apps.into_iter().map(Into::into).collect());
    }

    /// Replace one runtime vocabulary a skill's `{slot:name}` phrases can bind
    /// against — the user's task lists, rooms, contacts. The frontend re-pushes
    /// whenever the underlying set changes. Push an empty `values` when the
    /// user has none — phrases naming that vocabulary then match nothing.
    pub fn set_vocabulary(&self, name: String, values: Vec<String>) {
        self.runtime
            .engine()
            .set_vocabulary(name, values);
    }

    /// Snapshot of the engine's durable personal facts (oldest first). The
    /// frontend reads this after a turn that signalled `facts_changed` and
    /// persists the result.
    pub fn remembered_facts(&self) -> Vec<String> {
        self.runtime
            .engine()
            .remembered_facts()
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
        self.runtime.reload_community_skills(Path::new(&skill_store_dir), android_load_options(&storage_dir))
    }
}

impl AriEngine {
    /// The body of [`Self::process_input`], with an on-device answer going to
    /// `answer` while it's written, when there is one.
    fn respond(&self, input: String, answer: Option<Arc<dyn AnswerSink>>) -> FfiResponse {
        self.runtime.reply(&input, None, answer).0.into()
    }
}

impl From<Reply> for FfiResponse {
    fn from(reply: Reply) -> Self {
        match reply {
            Reply::Text { body, markdown, flags } => FfiResponse::Text {
                body,
                markdown,
                rearm: flags.rearm,
                enter_conversation: flags.enter_conversation,
                exit_conversation: flags.exit_conversation,
                facts_changed: flags.facts_changed,
                clear_conversation: flags.clear_conversation,
            },
            Reply::Action { json, skill_id, flags } => FfiResponse::Action {
                json,
                skill_id,
                rearm: flags.rearm,
                enter_conversation: flags.enter_conversation,
                exit_conversation: flags.exit_conversation,
                facts_changed: flags.facts_changed,
                clear_conversation: flags.clear_conversation,
            },
            Reply::Binary { mime, data } => FfiResponse::Binary { mime, data },
            Reply::NotUnderstood { body } => FfiResponse::NotUnderstood { body },
        }
    }
}

// ── On-device model loading ────────────────────────────────────────────
//
// These four methods live in their own `impl` blocks rather than in the
// main one above because `#[uniffi::export]` is a proc macro: it sees the
// method tokens *before* the compiler strips `#[cfg]`, so a per-method
// `#[cfg(feature = "llm")]` still emits FFI scaffolding that calls a
// method the cfg just deleted. A `#[cfg]` on the whole `impl` is honoured
// (item-level cfg is evaluated before the attribute macro runs), so we
// gate at that level and supply a `not(llm)` twin instead.
//
// The twin keeps both the exported symbols AND the UniFFI checksums
// identical across builds (docstrings and argument names are folded into
// the per-method checksum UniFFI generates, so the stubs must match the
// real methods word-for-word in those respects, not just in signature) —
// which is why no Kotlin regeneration is needed when the `llm` feature is
// toggled. Dropping a UniFFI method, or letting a stub's docstring or
// argument names drift from its counterpart, would break the frontend.
// Callers already treat `false` from the `load_*` methods as "no model is
// active", which is exactly the truth in a build with no LLM support
// compiled in.

#[cfg(feature = "llm")]
#[uniffi::export]
impl AriEngine {
    /// Set the model file for the LLM fallback: a .litertlm or .task file,
    /// which LiteRT-LM runs on the GPU when `use_gpu` is set and the GPU
    /// loads it, else on the CPU. The model is NOT
    /// loaded immediately — it loads on demand when the first unmatched
    /// query arrives, and unloads after 60 seconds of idle to free RAM.
    ///
    /// Returns `true` if the path exists, `false` otherwise.
    /// Call at app startup if a model file is available on disk.
    pub fn load_llm_model(&self, model_path: String, use_gpu: bool) -> bool {
        self.runtime.load_llm_model(Path::new(&model_path), use_gpu)
    }

    /// Remove the LLM fallback. If a model is currently loaded in RAM,
    /// it is dropped and the memory is freed. The on-device assistant
    /// stops answering until [`Self::load_llm_model`] is called again, so
    /// this is for switching or deleting models: to free memory, use
    /// [`Self::release_llm_memory`].
    pub fn unload_llm_model(&self) {
        let mut engine = self.runtime.engine();
        engine.set_llm_none();
    }

    /// Free the loaded model's memory but keep it set, so the next question
    /// loads it again. Blocks until any answer in progress finishes, so
    /// don't call it on the main thread.
    pub fn release_llm_memory(&self) {
        let engine = self.runtime.engine();
        engine.release_llm_memory();
    }

}

/// Stubs for builds without the `llm` feature (e.g. the `keyword-hit`
/// oracle the training pipeline builds). Signatures match the real methods exactly; there is
/// no model to load, so loading always reports failure and unloading is a
/// no-op.
#[cfg(not(feature = "llm"))]
#[uniffi::export]
impl AriEngine {
    /// Set the model file for the LLM fallback: a .litertlm or .task file,
    /// which LiteRT-LM runs on the GPU when `use_gpu` is set and the GPU
    /// loads it, else on the CPU. The model is NOT
    /// loaded immediately — it loads on demand when the first unmatched
    /// query arrives, and unloads after 60 seconds of idle to free RAM.
    ///
    /// Returns `true` if the path exists, `false` otherwise.
    /// Call at app startup if a model file is available on disk.
    pub fn load_llm_model(&self, model_path: String, use_gpu: bool) -> bool {
        let _ = (model_path, use_gpu);
        false
    }

    /// Remove the LLM fallback. If a model is currently loaded in RAM,
    /// it is dropped and the memory is freed. The on-device assistant
    /// stops answering until [`Self::load_llm_model`] is called again, so
    /// this is for switching or deleting models: to free memory, use
    /// [`Self::release_llm_memory`].
    pub fn unload_llm_model(&self) {}

    /// Free the loaded model's memory but keep it set, so the next question
    /// loads it again. Blocks until any answer in progress finishes, so
    /// don't call it on the main thread.
    pub fn release_llm_memory(&self) {}

}

/// Mechanical guard for the `llm` / `not(llm)` twin `impl` blocks above.
///
/// UniFFI folds each method's signature, argument names AND docstring into a
/// per-method checksum that the generated Kotlin asserts at load time. If the
/// twins drift on any of those, a build with the other feature setting fails
/// at RUNTIME on device with a checksum mismatch — long after CI, and nowhere
/// near the edit that caused it. The prose comment above the blocks asked
/// nicely; this enforces it.
///
/// Necessarily a source-text test: the two blocks are never compiled together,
/// so no amount of type-level cleverness can compare them. It reads this very
/// file and compares what UniFFI hashes.
#[cfg(test)]
mod uniffi_twin_guard {
    /// This file's own source. The blocks under test are the ones directly
    /// above, so reading `lib.rs` is reading the thing being guarded.
    const SRC: &str = include_str!("lib.rs");

    const LLM_MARKER: &str = "#[cfg(feature = \"llm\")]\n#[uniffi::export]\nimpl AriEngine {";
    const NO_LLM_MARKER: &str =
        "#[cfg(not(feature = \"llm\"))]\n#[uniffi::export]\nimpl AriEngine {";

    /// The methods that must exist in both twins. Hardcoded so deleting a
    /// method from BOTH blocks — which would keep them trivially equal while
    /// dropping an exported FFI symbol the frontend calls — still fails here.
    const EXPECTED: [&str; 3] = ["load_llm_model", "unload_llm_model", "release_llm_memory"];

    /// Body of the `impl AriEngine` block introduced by `marker`, ending at
    /// the first column-0 `}`.
    fn block(marker: &str) -> &'static str {
        let start = SRC
            .find(marker)
            .unwrap_or_else(|| panic!("lib.rs no longer contains the block introduced by:\n{marker}"))
            + marker.len();
        let len = SRC[start..]
            .find("\n}\n")
            .unwrap_or_else(|| panic!("unterminated impl block after:\n{marker}"));
        &SRC[start..start + len]
    }

    /// Every `pub fn` in `block` as `(name, signature, docstring)`, where the
    /// docstring is the run of `///` lines immediately preceding it.
    ///
    /// Per-line trimming makes this indifferent to indentation, but nothing
    /// else: reword a docstring, rename an argument, change a type or reorder
    /// the methods in one twin and the comparison fails.
    fn methods(block: &str) -> Vec<(String, String, String)> {
        let mut out = Vec::new();
        let mut doc: Vec<String> = Vec::new();
        for line in block.lines() {
            let t = line.trim();
            if let Some(d) = t.strip_prefix("///") {
                doc.push(d.trim().to_string());
            } else if let Some(sig) = t.strip_prefix("pub fn ") {
                let sig = sig.split('{').next().unwrap().trim();
                let name = sig.split('(').next().unwrap().trim().to_string();
                out.push((name, sig.to_string(), doc.join("\n")));
                doc.clear();
            } else if !t.is_empty() {
                // Any other code line breaks the doc-comment run.
                doc.clear();
            }
        }
        out
    }

    #[test]
    fn llm_twins_match_in_everything_uniffi_hashes() {
        let with = methods(block(LLM_MARKER));
        let without = methods(block(NO_LLM_MARKER));

        let names: Vec<&str> = with.iter().map(|(n, _, _)| n.as_str()).collect();
        assert_eq!(
            names, EXPECTED,
            "the #[cfg(feature = \"llm\")] impl block no longer exports exactly the four \
             expected methods in order — update EXPECTED here only if you also updated \
             both twins and regenerated the Kotlin bindings"
        );

        assert_eq!(
            with.len(),
            without.len(),
            "the twin impl blocks export {} and {} methods — every UniFFI method must \
             exist in both, or the bindings break in whichever build lacks it",
            with.len(),
            without.len()
        );

        for (a, b) in with.iter().zip(without.iter()) {
            assert_eq!(
                a.0, b.0,
                "twin methods are out of order: llm has `{}` where not(llm) has `{}`",
                a.0, b.0
            );
            assert_eq!(
                a.1, b.1,
                "signature drift in `{}` — UniFFI folds signatures and argument names into \
                 its per-method checksum, so this breaks the Kotlin bindings at runtime.\n\
                 \x20 llm:      {}\n\x20 not(llm): {}",
                a.0, a.1, b.1
            );
            assert_eq!(
                a.2, b.2,
                "docstring drift in `{}` — UniFFI folds docstrings into its per-method \
                 checksum too, so this breaks the Kotlin bindings at runtime.\n\
                 \x20 llm:\n{}\n\x20 not(llm):\n{}",
                a.0, a.2, b.2
            );
        }
    }

    /// The guard is only worth anything if the extractor actually sees the
    /// docstrings and signatures. Pin that against a known method rather than
    /// letting a parser that silently returns nothing pass the test above.
    #[test]
    fn extractor_reads_real_signatures_and_docstrings() {
        let with = methods(block(LLM_MARKER));
        let (name, sig, doc) = &with[0];
        assert_eq!(name, "load_llm_model");
        assert_eq!(sig, "load_llm_model(&self, model_path: String, use_gpu: bool) -> bool");
        assert_eq!(
            doc,
            "Set the model file for the LLM fallback: a .litertlm or .task file,\n\
             which LiteRT-LM runs on the GPU when `use_gpu` is set and the GPU\n\
             loads it, else on the CPU. The model is NOT\n\
             loaded immediately — it loads on demand when the first unmatched\n\
             query arrives, and unloads after 60 seconds of idle to free RAM.\n\
             \n\
             Returns `true` if the path exists, `false` otherwise.\n\
             Call at app startup if a model file is available on disk."
        );
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
    fn android_host_grants_every_capability_that_exists() {
        // Android is the reference frontend and implements the lot. Adding a
        // capability to the enum without adding it here rejects the skill at
        // install on a real phone while the CLI happily loads it — which is
        // exactly how `contacts` was missed.
        let opts = android_load_options("/tmp/ignored");
        let missing: Vec<&str> = ari_skill_loader::ALL_CAPABILITIES
            .iter()
            .filter(|c| !opts.host_capabilities.provides(**c))
            .map(|c| ari_skill_loader::capability_name(*c))
            .collect();
        assert!(missing.is_empty(), "android_load_options is missing: {missing:?}");
    }

    #[test]
    fn media_services_adapter_passes_ids_through() {
        struct Fake;
        impl FfiMediaServicesProvider for Fake {
            fn installed_services(&self) -> Vec<String> {
                vec!["spotify".to_string(), "youtube_music".to_string()]
            }
        }
        let adapter = ForeignMediaServicesProviderAdapter(Arc::new(Fake));
        assert_eq!(
            adapter.installed_services(),
            vec!["spotify".to_string(), "youtube_music".to_string()]
        );
    }

    #[test]
    fn ffi_response_action_carries_rearm_field() {
        // Compile-level guarantee the variant has the expected fields.
        let r = FfiResponse::Action {
            json: "{}".to_string(),
            skill_id: "x".to_string(),
            rearm: true,
            enter_conversation: false,
            exit_conversation: false,
            facts_changed: false,
            clear_conversation: false,
        };
        match r {
            FfiResponse::Action { rearm, .. } => assert!(rearm),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn ffi_engine_set_and_get_remembered_facts() {
        let engine = AriEngine::new();
        engine.set_remembered_facts(vec!["i am vegetarian".to_string()]);
        assert_eq!(engine.remembered_facts(), vec!["i am vegetarian".to_string()]);
    }

    #[test]
    fn ffi_app_entry_converts_to_core() {
        let core: ari_core::AppEntry = FfiAppEntry {
            label: "Spotify".to_string(),
            package: "com.spotify.music".to_string(),
        }
        .into();
        assert_eq!(core.label, "Spotify");
        assert_eq!(core.package, "com.spotify.music");
    }

    #[test]
    fn set_installed_apps_marshals_and_open_still_launches() {
        let engine = AriEngine::new(); // built-ins incl. `open`
        engine.set_installed_apps(vec![FfiAppEntry {
            label: "Spotify".to_string(),
            package: "com.spotify.music".to_string(),
        }]);
        match engine.process_input("open spotify".to_string()) {
            FfiResponse::Action { json, .. } => {
                assert!(json.contains("launch_app"), "expected a launch_app envelope: {json}");
                assert!(json.to_lowercase().contains("spotify"), "target preserved: {json}");
            }
            other => panic!("expected a launch action for an installed app, got {other:?}"),
        }
    }
}
