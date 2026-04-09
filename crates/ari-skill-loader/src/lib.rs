pub mod bundle;
pub mod declarative;
pub mod host_capabilities;
pub mod http_config;
pub mod loader;
pub mod manifest;
pub mod scoring;
pub mod registry;
pub mod signature;
pub mod storage_config;
pub mod store;
mod tls;
pub mod wasm;

pub use bundle::{install_from_bytes, sha256_hex, BundleError, InstalledBundle};
pub use declarative::{AdapterError, DeclarativeSkill};
pub use signature::{SignatureError, TrustRoot, VERIFYING_KEY_LENGTH};
pub use host_capabilities::{capability_name, parse_capability, HostCapabilities};
pub use http_config::HttpConfig;
pub use loader::{
    load_single_skill_dir, load_single_skill_dir_with, load_skill_directory,
    load_skill_directory_with, LoadFailure, LoadFailureKind, LoadOptions, LoadReport,
};
pub use manifest::{
    AriExtension, Behaviour, Capability, DeclarativeBehaviour, ManifestError, MatchPattern,
    Matching, ResponseSpec, Skillfile, SpecificityLevel, WasmBehaviour,
};
pub use registry::{
    check_updates, install_by_id, install_update, AvailableUpdate, Index, IndexEntry,
    RegistryClient, RegistryError, REGISTRY_BASE_URL, REGISTRY_INDEX_URL, REGISTRY_TRUST_KEY,
};
pub use storage_config::StorageConfig;
pub use store::{InstalledSkill, SkillStore, StoreError};
pub use wasm::{CapturingLogSink, LogLevel, LogSink, NullLogSink, WasmError, WasmSkill};
