pub mod assistant;
pub mod bundle;
pub mod declarative;
pub mod host_capabilities;
pub mod http_config;
pub mod loader;
pub mod manifest;
pub mod platform_capabilities;
pub mod scoring;
pub mod registry;
pub mod signature;
pub mod storage_config;
pub mod store;
mod tls;
pub mod wasm;

pub use assistant::{AssistantApiError, ConfigStore, MemoryConfigStore, call_assistant_api};
pub use bundle::{install_from_bytes, sha256_hex, BundleError, InstalledBundle};
pub use declarative::{AdapterError, DeclarativeSkill};
pub use signature::{SignatureError, TrustRoot, VERIFYING_KEY_LENGTH};
pub use host_capabilities::{capability_name, parse_capability, HostCapabilities};
pub use http_config::HttpConfig;
pub use loader::{
    load_single_skill_dir, load_single_skill_dir_with, load_skill_directory,
    load_skill_directory_with, AssistantEntry, LoadFailure, LoadFailureKind, LoadOptions,
    LoadReport,
};
pub use manifest::{
    ApiConfig, AriExtension, AssistantManifest, AssistantProvider, AuthScheme, Behaviour,
    Capability, ConfigField, ConfigFieldType, DeclarativeBehaviour, ManifestError, MatchPattern,
    Matching, PathSegment, Privacy, RequestFormat, ResponseSpec, SelectOption, SkillExample,
    SkillType, Skillfile, SpecificityLevel, WasmBehaviour, extract_by_path, parse_response_path,
};
pub use registry::{
    check_updates, install_by_id, install_update, AvailableUpdate, Index, IndexEntry,
    RegistryClient, RegistryError, REGISTRY_BASE_URL, REGISTRY_INDEX_URL, REGISTRY_TRUST_KEY,
};
pub use platform_capabilities::{
    Calendar, CalendarProvider, InsertCalendarEventParams, InsertTaskParams,
    LocalClock, LocalTimeComponents, NullCalendarProvider, NullTasksProvider,
    TaskList, TasksProvider, UtcLocalClock,
};
pub use storage_config::StorageConfig;
pub use store::{InstalledSkill, SkillStore, StoreError};
pub use wasm::{CapturingLogSink, LogLevel, LogSink, NullLogSink, WasmError, WasmSkill};
