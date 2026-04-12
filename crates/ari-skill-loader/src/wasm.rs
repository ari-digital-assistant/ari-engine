//! WASM skill adapter (step 5a).
//!
//! ## ABI v1
//!
//! A skill module **must** export:
//!
//! - `memory` — its linear memory
//! - `ari_alloc(size: i32) -> i32` — bump-style allocator. Host calls this to
//!   reserve space for the input string before each call. Skill owns the
//!   allocator implementation; host treats the returned pointer as opaque.
//! - `score(input_ptr: i32, input_len: i32) -> f32` — return a score in
//!   [0.0, 1.0] for the given UTF-8 input.
//! - `execute(input_ptr: i32, input_len: i32) -> i64` — return a packed
//!   `(ptr << 32) | len` pointing to a UTF-8 response string in the skill's
//!   linear memory. The host wraps the result in `Response::Text`. Richer
//!   response variants will arrive in a later ABI revision.
//!
//! A skill module **may** import:
//!
//! - `ari::log(level: i32, ptr: i32, len: i32)` — emit a log line. `level` is
//!   `0=trace, 1=debug, 2=info, 3=warn, 4=error`. Anything else is clamped.
//! - `ari::get_capability(name_ptr: i32, name_len: i32) -> i32` — returns 1
//!   if the capability is **both** declared by the skill in
//!   `metadata.ari.capabilities` **and** granted by the host, else 0. Skills
//!   cannot use this to detect undeclared capabilities — the loader checks
//!   declared caps against the host set at install time, and `get_capability`
//!   only ever answers truthfully for caps the skill is allowed to know about.
//! - `ari::http_fetch(url_ptr: i32, url_len: i32) -> i64` — perform a GET
//!   request. The url must use a scheme allowed by the host's [`HttpConfig`]
//!   (default: `https` only). Response is encoded as a JSON string written
//!   into the skill's linear memory via the skill's own `ari_alloc`, and the
//!   call returns a packed `(ptr << 32) | len`. Response shape:
//!   `{"status": <int>, "body": "<utf-8>"}` for any HTTP response (incl. 4xx
//!   and 5xx, where `status` is the real code), or `{"status": 0, "body":
//!   null, "error": "<msg>"}` for network/timeout/scheme/body-too-large
//!   failures. Requires the `http` capability.
//! - `ari::storage_get(key_ptr: i32, key_len: i32) -> i64` — read a UTF-8
//!   value from the skill's per-skill key-value store. Returns a packed
//!   `(ptr << 32) | len` of the value (allocated via the skill's own
//!   `ari_alloc`), or `0` if the key is not present or the read fails.
//!   Requires the `storage_kv` capability.
//! - `ari::storage_set(key_ptr: i32, key_len: i32, val_ptr: i32, val_len: i32) -> i32`
//!   — write a UTF-8 value into the skill's store. Returns `0` on success,
//!   non-zero on any failure (key/value/total length cap exceeded, IO error,
//!   bad UTF-8). The on-disk state is unchanged on failure. Requires the
//!   `storage_kv` capability.
//!
//! ## Sneak guard
//!
//! At install time the loader scans the WASM module's `ari::*` imports and
//! checks each one against a known import→capability table. If a module
//! imports something whose required capability isn't in the manifest's
//! `metadata.ari.capabilities` list, install fails with
//! [`WasmError::UndeclaredCapability`]. Combined with the manifest-vs-host
//! check that already happens, this means a malicious skill cannot quietly
//! pull in `http_fetch` without the user's knowledge.
//!
//! ## Sandbox guarantees (5a)
//!
//! - **Memory cap**: enforced via `StoreLimits`. Default 16 MiB; configurable
//!   from the manifest's `metadata.ari.wasm.memory_limit_mb`.
//! - **Per-call fuel**: a fixed amount of execution fuel is added to the store
//!   before each call. Skills that bust the budget are killed and the call
//!   returns score `0.0` or a fallback `Response::Text("(skill error)")`.
//!   Default 50_000_000 units, ~tens of milliseconds of pure compute on
//!   modern hardware.
//! - **Fresh store per call**: each `score`/`execute` invocation creates a
//!   brand-new `Store` and `Instance`. Skills cannot retain state between
//!   calls (storage_kv arrives in step 5d).
//!
//! Anything else (capabilities, http, persistent storage) is **not** in 5a.

use crate::host_capabilities::{capability_name, HostCapabilities};
use crate::http_config::HttpConfig;
use crate::manifest::{AriExtension, Behaviour, Capability, Skillfile, WasmBehaviour};
use crate::scoring::{PatternScorer, ScorerError};
use crate::storage_config::StorageConfig;
use ari_core::{Response, Skill, SkillContext, Specificity};
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use thiserror::Error;
use wasmtime::{Caller, Engine, Linker, Memory, Module, Store, StoreLimits, StoreLimitsBuilder};

/// Static table mapping every `ari::*` host import to the capability a skill
/// must declare in order to use it. The install-time sneak guard scans the
/// module's imports against this table.
const HOST_IMPORT_CAPABILITY_TABLE: &[(&str, Option<Capability>)] = &[
    // log and get_capability are unconditional — every skill is allowed to
    // log and to query its own capabilities. Marked `None` so they pass the
    // sneak guard without requiring any declaration.
    ("log", None),
    ("get_capability", None),
    ("http_fetch", Some(Capability::Http)),
    ("storage_get", Some(Capability::StorageKv)),
    ("storage_set", Some(Capability::StorageKv)),
];

/// Default fuel budget per call. Tuned for "tens of milliseconds of compute
/// on modern hardware". Configurable via a future manifest field if needed.
const DEFAULT_FUEL_PER_CALL: u64 = 50_000_000;

#[derive(Debug, Error)]
pub enum WasmError {
    #[error("skillfile has no `metadata.ari` extension")]
    NotAnAriSkill,

    #[error("skillfile is a declarative skill; use the declarative adapter instead")]
    NotWasm,

    #[error("could not read WASM module {path:?}: {source}")]
    ReadModule {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("could not compile WASM module: {0}")]
    Compile(String),

    #[error("WASM module is missing required export: {0}")]
    MissingExport(&'static str),

    #[error("WASM module export {0} has the wrong signature")]
    BadExportSignature(&'static str),

    #[error("scorer compile failed: {0}")]
    Scorer(#[from] ScorerError),

    #[error("skill needs host capabilities not provided by this host: {missing:?}")]
    MissingCapabilities { missing: Vec<Capability> },

    #[error(
        "WASM module imports `ari::{import}` which requires the `{required:?}` capability, \
         but the skill manifest does not declare it"
    )]
    UndeclaredCapability {
        import: &'static str,
        required: Capability,
    },

    #[error("WASM module imports an unknown host symbol `ari::{0}`")]
    UnknownHostImport(String),
}

/// Levels match the table in the module doc comment.
#[derive(Debug, Clone, Copy)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl LogLevel {
    fn from_i32(n: i32) -> LogLevel {
        match n {
            0 => LogLevel::Trace,
            1 => LogLevel::Debug,
            2 => LogLevel::Info,
            3 => LogLevel::Warn,
            _ => LogLevel::Error,
        }
    }
}

/// Trait the host implements to receive log messages from a skill.
pub trait LogSink: Send + Sync {
    fn log(&self, skill_id: &str, level: LogLevel, message: &str);
}

/// A no-op log sink. Useful for tests and as a default.
pub struct NullLogSink;

impl LogSink for NullLogSink {
    fn log(&self, _skill_id: &str, _level: LogLevel, _message: &str) {}
}

/// A log sink that captures every line into a `Vec`. Test-only convenience.
#[derive(Default, Clone)]
pub struct CapturingLogSink {
    lines: Arc<Mutex<Vec<(LogLevel, String)>>>,
}

impl CapturingLogSink {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn lines(&self) -> Vec<(LogLevel, String)> {
        self.lines
            .lock()
            .unwrap()
            .iter()
            .map(|(l, m)| (*l, m.clone()))
            .collect()
    }
}

impl LogSink for CapturingLogSink {
    fn log(&self, _skill_id: &str, level: LogLevel, message: &str) {
        self.lines.lock().unwrap().push((level, message.to_string()));
    }
}

/// State that travels with each `Store`. Re-created per call.
struct StoreData {
    skill_id: String,
    log_sink: Arc<dyn LogSink>,
    limits: StoreLimits,
    /// The intersection of (skill-declared capabilities) ∩ (host-granted
    /// capabilities). `get_capability` returns 1 iff the requested name is
    /// in this set. Computed once per `WasmSkill` and cloned into each store.
    granted_capabilities: Arc<HashSet<Capability>>,
    /// Reqwest blocking client and policy. Only present when the skill has
    /// the http capability granted; otherwise None and the http_fetch import
    /// is never wired into the linker.
    http_client: Option<Arc<HttpClientCtx>>,
    /// Per-skill key-value storage context. Only present when the skill has
    /// the storage_kv capability granted.
    storage: Option<Arc<StorageCtx>>,
}

/// Bundle of (configured client, config) so the host import has everything it
/// needs without poking back into `WasmSkill`.
struct HttpClientCtx {
    client: reqwest::blocking::Client,
    config: HttpConfig,
}

/// Per-skill storage state. The mutex protects the on-disk file: every
/// load/store goes through it so concurrent calls (which shouldn't happen in
/// practice today, but we don't enforce single-threaded skill execution at
/// the engine level) don't trample each other.
struct StorageCtx {
    file: PathBuf,
    config: StorageConfig,
    lock: Mutex<()>,
}

/// A WASM skill, ready to plug into the engine. Owns a compiled `Module` and
/// the configuration needed to instantiate it on demand.
pub struct WasmSkill {
    id: String,
    description: String,
    specificity: Specificity,
    custom_score: bool,
    /// Native pattern scorer used when `custom_score = false`. Same code path
    /// as the declarative adapter, so a WASM skill that doesn't override
    /// scoring behaves identically to a declarative one with the same
    /// `metadata.ari.matching` block.
    scorer: PatternScorer,
    engine: Engine,
    module: Module,
    memory_limit_bytes: usize,
    log_sink: Arc<dyn LogSink>,
    /// Intersection of declared and host-granted capabilities. Cloned into
    /// every store so the `get_capability` host import can answer queries.
    granted_capabilities: Arc<HashSet<Capability>>,
    /// Pre-built reqwest client honouring `HttpConfig`. Only `Some` when the
    /// skill has the `http` capability granted. Cloned into each store.
    http_client: Option<Arc<HttpClientCtx>>,
    /// Per-skill kv storage context. Only `Some` when the skill has the
    /// `storage_kv` capability granted.
    storage: Option<Arc<StorageCtx>>,
}

impl std::fmt::Debug for WasmSkill {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WasmSkill")
            .field("id", &self.id)
            .field("specificity", &self.specificity)
            .field("custom_score", &self.custom_score)
            .field("memory_limit_bytes", &self.memory_limit_bytes)
            .finish_non_exhaustive()
    }
}

impl WasmSkill {
    /// Build a WASM skill from a parsed `SKILL.md` and the directory the
    /// manifest lives in (used to resolve the relative `wasm.module` path).
    /// `host_caps` is the set of capabilities the host provides; the skill's
    /// declared capabilities are checked against it and the intersection is
    /// what `get_capability` will report at runtime. `http_config` controls
    /// the `ari::http_fetch` import for skills that have the `http` cap.
    pub fn from_skillfile(
        sf: &Skillfile,
        skill_dir: &Path,
        log_sink: Arc<dyn LogSink>,
        host_caps: &HostCapabilities,
        http_config: &HttpConfig,
        storage_config: &StorageConfig,
    ) -> Result<Self, WasmError> {
        let ari = sf.ari_extension.as_ref().ok_or(WasmError::NotAnAriSkill)?;
        let wasm = match &ari.behaviour {
            Some(Behaviour::Wasm(w)) => w,
            Some(Behaviour::Declarative(_)) | None => return Err(WasmError::NotWasm),
        };
        Self::build(
            ari,
            &sf.description,
            wasm,
            skill_dir,
            log_sink,
            host_caps,
            http_config,
            storage_config,
        )
    }

    fn build(
        ari: &AriExtension,
        description: &str,
        wasm: &WasmBehaviour,
        skill_dir: &Path,
        log_sink: Arc<dyn LogSink>,
        host_caps: &HostCapabilities,
        http_config: &HttpConfig,
        storage_config: &StorageConfig,
    ) -> Result<Self, WasmError> {
        let module_path = skill_dir.join(&wasm.module);
        let bytes = std::fs::read(&module_path).map_err(|source| WasmError::ReadModule {
            path: module_path.clone(),
            source,
        })?;
        Self::from_module_bytes(
            ari,
            description,
            wasm,
            &bytes,
            log_sink,
            host_caps,
            http_config,
            storage_config,
        )
    }

    /// Test seam: build directly from in-memory module bytes (WASM or WAT).
    pub fn from_module_bytes(
        ari: &AriExtension,
        description: &str,
        wasm: &WasmBehaviour,
        bytes: &[u8],
        log_sink: Arc<dyn LogSink>,
        host_caps: &HostCapabilities,
        http_config: &HttpConfig,
        storage_config: &StorageConfig,
    ) -> Result<Self, WasmError> {
        let mut config = wasmtime::Config::new();
        config.consume_fuel(true);
        let engine = Engine::new(&config).map_err(|e| WasmError::Compile(e.to_string()))?;

        let module = compile_module_on_big_stack(&engine, bytes)?;

        let memory_limit_bytes = wasm.memory_limit_mb.max(1) as usize * 1024 * 1024;
        let matching = ari.matching.as_ref().ok_or(WasmError::NotWasm)?;
        let scorer = PatternScorer::compile(matching)?;

        // Capability check at install time. The grant set is the intersection
        // of (declared) ∩ (host-provided). Anything declared but not provided
        // is a hard install failure — we never let a skill load knowing it'll
        // hit a missing host import on first call.
        let missing = host_caps.missing_for(&ari.capabilities);
        if !missing.is_empty() {
            return Err(WasmError::MissingCapabilities { missing });
        }
        let granted: HashSet<Capability> = ari.capabilities.iter().copied().collect();

        // Sneak guard: scan the module's `ari::*` imports against the
        // declared capability list. A module that imports a capability-bound
        // host function without declaring the corresponding capability is
        // rejected before we ever instantiate.
        validate_module_imports(&module, &granted)?;

        // Build the http client only if the skill has the http capability.
        // The linker only wires up `ari::http_fetch` when this is `Some`, so
        // a skill that didn't declare http (and therefore can't be granted
        // it) will fail at link time if it tries to import http_fetch — but
        // the import scan above catches that case earlier with a friendlier
        // error.
        let http_client = if granted.contains(&Capability::Http) {
            Some(Arc::new(HttpClientCtx {
                client: build_reqwest_client(http_config)?,
                config: http_config.clone(),
            }))
        } else {
            None
        };

        let storage = if granted.contains(&Capability::StorageKv) {
            // Make sure the storage root exists. Skills should never see a
            // missing-directory error from a get/set call.
            std::fs::create_dir_all(&storage_config.root).map_err(|e| {
                WasmError::Compile(format!(
                    "could not create storage root {:?}: {e}",
                    storage_config.root
                ))
            })?;
            Some(Arc::new(StorageCtx {
                file: storage_config.file_for(&ari.id),
                config: storage_config.clone(),
                lock: Mutex::new(()),
            }))
        } else {
            None
        };

        let skill = WasmSkill {
            id: ari.id.clone(),
            description: description.to_string(),
            specificity: ari.specificity.as_core(),
            custom_score: matching.custom_score,
            scorer,
            engine,
            module,
            memory_limit_bytes,
            log_sink,
            granted_capabilities: Arc::new(granted),
            http_client,
            storage,
        };
        skill.validate_exports()?;
        Ok(skill)
    }

    /// Throwaway instantiation just to verify the module exposes the expected
    /// exports with the expected signatures, before we ever call it for real.
    /// This catches authoring mistakes at load time, not on first utterance.
    fn validate_exports(&self) -> Result<(), WasmError> {
        let mut store = self.fresh_store();
        let linker = self.fresh_linker()?;
        let instance = linker
            .instantiate(&mut store, &self.module)
            .map_err(|e| WasmError::Compile(e.to_string()))?;

        if instance.get_memory(&mut store, "memory").is_none() {
            return Err(WasmError::MissingExport("memory"));
        }
        instance
            .get_typed_func::<i32, i32>(&mut store, "ari_alloc")
            .map_err(|_| WasmError::BadExportSignature("ari_alloc"))?;
        instance
            .get_typed_func::<(i32, i32), f32>(&mut store, "score")
            .map_err(|_| WasmError::BadExportSignature("score"))?;
        instance
            .get_typed_func::<(i32, i32), i64>(&mut store, "execute")
            .map_err(|_| WasmError::BadExportSignature("execute"))?;
        Ok(())
    }

    fn fresh_store(&self) -> Store<StoreData> {
        let limits = StoreLimitsBuilder::new()
            .memory_size(self.memory_limit_bytes)
            .build();
        let mut store = Store::new(
            &self.engine,
            StoreData {
                skill_id: self.id.clone(),
                log_sink: self.log_sink.clone(),
                limits,
                granted_capabilities: self.granted_capabilities.clone(),
                http_client: self.http_client.clone(),
                storage: self.storage.clone(),
            },
        );
        store.limiter(|data| &mut data.limits);
        // Ignore the error path: with `consume_fuel(true)` set in the
        // engine config, set_fuel always succeeds.
        let _ = store.set_fuel(DEFAULT_FUEL_PER_CALL);
        store
    }

    fn fresh_linker(&self) -> Result<Linker<StoreData>, WasmError> {
        let mut linker = Linker::new(&self.engine);
        linker
            .func_wrap(
                "ari",
                "log",
                |mut caller: Caller<'_, StoreData>, level: i32, ptr: i32, len: i32| {
                    let memory = match caller.get_export("memory") {
                        Some(wasmtime::Extern::Memory(m)) => m,
                        _ => return,
                    };
                    let msg = read_utf8(&memory, &caller, ptr, len).unwrap_or_default();
                    let data = caller.data();
                    data.log_sink
                        .log(&data.skill_id, LogLevel::from_i32(level), &msg);
                },
            )
            .map_err(|e| WasmError::Compile(e.to_string()))?;
        linker
            .func_wrap(
                "ari",
                "get_capability",
                |mut caller: Caller<'_, StoreData>, name_ptr: i32, name_len: i32| -> i32 {
                    let memory = match caller.get_export("memory") {
                        Some(wasmtime::Extern::Memory(m)) => m,
                        _ => return 0,
                    };
                    let Some(name) = read_utf8(&memory, &caller, name_ptr, name_len) else {
                        return 0;
                    };
                    let Some(cap) = crate::host_capabilities::parse_capability(&name) else {
                        return 0;
                    };
                    if caller.data().granted_capabilities.contains(&cap) {
                        1
                    } else {
                        0
                    }
                },
            )
            .map_err(|e| WasmError::Compile(e.to_string()))?;

        // Wire up http_fetch only if the skill is allowed to use it. The
        // sneak guard already rejected at install time any module that
        // imported ari::http_fetch without declaring [http], so a skill
        // reaching this point with `http_client = None` cannot possibly
        // import http_fetch.
        if self.http_client.is_some() {
            linker
                .func_wrap(
                    "ari",
                    "http_fetch",
                    |mut caller: Caller<'_, StoreData>,
                     url_ptr: i32,
                     url_len: i32|
                     -> i64 {
                        http_fetch_impl(&mut caller, url_ptr, url_len)
                    },
                )
                .map_err(|e| WasmError::Compile(e.to_string()))?;
        }

        // Wire up storage_get/storage_set only if the skill is allowed.
        // Same sneak-guard logic as http_fetch.
        if self.storage.is_some() {
            linker
                .func_wrap(
                    "ari",
                    "storage_get",
                    |mut caller: Caller<'_, StoreData>,
                     key_ptr: i32,
                     key_len: i32|
                     -> i64 { storage_get_impl(&mut caller, key_ptr, key_len) },
                )
                .map_err(|e| WasmError::Compile(e.to_string()))?;
            linker
                .func_wrap(
                    "ari",
                    "storage_set",
                    |mut caller: Caller<'_, StoreData>,
                     key_ptr: i32,
                     key_len: i32,
                     val_ptr: i32,
                     val_len: i32|
                     -> i32 {
                        storage_set_impl(&mut caller, key_ptr, key_len, val_ptr, val_len)
                    },
                )
                .map_err(|e| WasmError::Compile(e.to_string()))?;
        }
        Ok(linker)
    }

    /// The capabilities this skill is allowed to use (intersection of
    /// declared and host-granted). Test/debug helper.
    pub fn granted_capabilities(&self) -> Vec<Capability> {
        let mut v: Vec<Capability> = self.granted_capabilities.iter().copied().collect();
        v.sort_by_key(|c| capability_name(*c));
        v
    }

    /// Run a closure with a freshly instantiated module. Centralises the
    /// per-call setup so `score` and `execute` don't drift apart.
    fn with_instance<R>(
        &self,
        f: impl FnOnce(&mut Store<StoreData>, wasmtime::Instance) -> R,
        on_error: R,
    ) -> R {
        let mut store = self.fresh_store();
        let linker = match self.fresh_linker() {
            Ok(l) => l,
            Err(_) => return on_error,
        };
        let instance = match linker.instantiate(&mut store, &self.module) {
            Ok(i) => i,
            Err(_) => return on_error,
        };
        f(&mut store, instance)
    }

    fn write_input(
        store: &mut Store<StoreData>,
        instance: wasmtime::Instance,
        input: &str,
    ) -> Option<(Memory, i32, i32)> {
        let memory = instance.get_memory(&mut *store, "memory")?;
        let alloc = instance
            .get_typed_func::<i32, i32>(&mut *store, "ari_alloc")
            .ok()?;
        let len = input.len() as i32;
        let ptr = alloc.call(&mut *store, len).ok()?;
        if ptr <= 0 {
            return None;
        }
        let mem_data = memory.data_mut(&mut *store);
        let start = ptr as usize;
        let end = start.checked_add(input.len())?;
        if end > mem_data.len() {
            return None;
        }
        mem_data[start..end].copy_from_slice(input.as_bytes());
        Some((memory, ptr, len))
    }
}

fn read_utf8(memory: &Memory, store: &impl wasmtime::AsContext, ptr: i32, len: i32) -> Option<String> {
    if ptr < 0 || len < 0 {
        return None;
    }
    let data = memory.data(store);
    let start = ptr as usize;
    let end = start.checked_add(len as usize)?;
    if end > data.len() {
        return None;
    }
    std::str::from_utf8(&data[start..end])
        .ok()
        .map(|s| s.to_string())
}

/// Scan the WASM module's `ari::*` imports and verify each one is satisfied
/// by the skill's declared capability set. Anything imported but not declared
/// is rejected; anything imported that we don't know about is also rejected.
fn validate_module_imports(
    module: &Module,
    granted: &HashSet<Capability>,
) -> Result<(), WasmError> {
    for import in module.imports() {
        if import.module() != "ari" {
            // Other namespaces are forbidden — if a module imports anything
            // outside `ari::`, instantiation will fail at link time anyway,
            // but a clearer error here would be nice. For now we just let
            // wasmtime emit its own diagnostic.
            continue;
        }
        let name = import.name();
        let entry = HOST_IMPORT_CAPABILITY_TABLE
            .iter()
            .find(|(n, _)| *n == name);
        match entry {
            None => return Err(WasmError::UnknownHostImport(name.to_string())),
            Some((_, None)) => {
                // Unconditional import (log, get_capability) — always allowed.
            }
            Some((_, Some(required))) => {
                if !granted.contains(required) {
                    // Find the table entry's static name str so the error
                    // can carry it as a `&'static str`.
                    let static_name = HOST_IMPORT_CAPABILITY_TABLE
                        .iter()
                        .find(|(n, _)| *n == name)
                        .map(|(n, _)| *n)
                        .unwrap();
                    return Err(WasmError::UndeclaredCapability {
                        import: static_name,
                        required: *required,
                    });
                }
            }
        }
    }
    Ok(())
}

/// Compile a WASM module on a dedicated thread with an 8 MB stack.
///
/// **Why:** `wasmtime::Module::new` runs cranelift synchronously on the
/// calling thread. Cranelift allocates significant stack frames for its IR
/// transforms — desktop threads have ~8 MB and handle it fine, but Android
/// coroutine workers (`Dispatchers.IO` etc.) can have as little as a few
/// hundred KB and blow the stack with SIGSEGV / "stack pointer is not in
/// a rw map". We hit this trying to install a WASM skill through the
/// Android UI and it killed the app instantly.
///
/// Rather than telling every caller to be mindful of their stack, we
/// always compile on a fresh `std::thread` sized to match desktop. The
/// overhead is one thread spawn + join per WasmSkill construction, which
/// is measured in microseconds — cranelift itself dominates by orders of
/// magnitude, so the wrapper cost is invisible.
fn compile_module_on_big_stack(engine: &Engine, bytes: &[u8]) -> Result<Module, WasmError> {
    let engine_clone = engine.clone();
    let bytes_vec = bytes.to_vec();
    let handle = std::thread::Builder::new()
        .name("ari-wasm-compile".to_string())
        .stack_size(8 * 1024 * 1024)
        .spawn(move || {
            Module::new(&engine_clone, &bytes_vec).map_err(|e| WasmError::Compile(e.to_string()))
        })
        .map_err(|e| WasmError::Compile(format!("could not spawn wasm compile thread: {e}")))?;
    handle
        .join()
        .map_err(|_| WasmError::Compile("wasm compile thread panicked".to_string()))?
}

fn build_reqwest_client(config: &HttpConfig) -> Result<reqwest::blocking::Client, WasmError> {
    reqwest::blocking::Client::builder()
        .timeout(config.timeout)
        .redirect(reqwest::redirect::Policy::limited(config.max_redirects as usize))
        .user_agent(&config.user_agent)
        .use_preconfigured_tls(crate::tls::webpki_roots_config())
        .build()
        .map_err(|e| WasmError::Compile(format!("could not build http client: {e}")))
}

/// Implementation of `ari::http_fetch`. Reads the URL from wasm memory,
/// validates the scheme, performs a blocking GET, encodes the result as a
/// JSON string, allocates space in the skill's linear memory via its own
/// `ari_alloc` export, copies the JSON in, and returns the packed pointer.
///
/// On any failure (network, timeout, scheme, body too large, host out of
/// memory), returns a JSON document with `status: 0` and a non-null `error`
/// field. The skill always gets *some* response — never traps.
fn http_fetch_impl(caller: &mut Caller<'_, StoreData>, url_ptr: i32, url_len: i32) -> i64 {
    // Step 1: read the URL.
    let memory = match caller.get_export("memory") {
        Some(wasmtime::Extern::Memory(m)) => m,
        _ => return 0,
    };
    let url = match read_utf8(&memory, &*caller, url_ptr, url_len) {
        Some(u) => u,
        None => {
            return write_response(
                caller,
                memory,
                &error_json("could not read url from wasm memory"),
            );
        }
    };

    // Step 2: pull the http context out of the store.
    let http = match caller.data().http_client.clone() {
        Some(h) => h,
        None => {
            // Should be impossible — we only wire up http_fetch when this is
            // Some — but defensive.
            return write_response(
                caller,
                memory,
                &error_json("http capability not available"),
            );
        }
    };

    // Step 3: scheme check.
    let parsed = match url::Url::parse(&url) {
        Ok(u) => u,
        Err(e) => {
            return write_response(
                caller,
                memory,
                &error_json(&format!("invalid url: {e}")),
            );
        }
    };
    if !http.config.allows_scheme(parsed.scheme()) {
        return write_response(
            caller,
            memory,
            &error_json(&format!("scheme not allowed: {}", parsed.scheme())),
        );
    }

    // Step 4: do the request. Blocking.
    let json = match http.client.get(parsed).send() {
        Ok(resp) => {
            let status = resp.status().as_u16();
            // Read body up to the limit. We can't use `Response::bytes` directly
            // with a cap, so read the underlying reader manually.
            use std::io::Read;
            let mut reader = resp.take(http.config.max_body_bytes as u64 + 1);
            let mut buf: Vec<u8> = Vec::new();
            match reader.read_to_end(&mut buf) {
                Ok(_) if buf.len() > http.config.max_body_bytes => {
                    error_json(&format!(
                        "response body exceeds {} byte limit",
                        http.config.max_body_bytes
                    ))
                }
                Ok(_) => {
                    let body = String::from_utf8_lossy(&buf).into_owned();
                    success_json(status, &body)
                }
                Err(e) => error_json(&format!("body read failed: {e}")),
            }
        }
        Err(e) => error_json(&format!("request failed: {e}")),
    };

    write_response(caller, memory, &json)
}

fn success_json(status: u16, body: &str) -> String {
    serde_json::json!({
        "status": status,
        "body": body,
    })
    .to_string()
}

fn error_json(message: &str) -> String {
    serde_json::json!({
        "status": 0,
        "body": serde_json::Value::Null,
        "error": message,
    })
    .to_string()
}

// --- storage_get / storage_set host imports --------------------------------

fn storage_get_impl(caller: &mut Caller<'_, StoreData>, key_ptr: i32, key_len: i32) -> i64 {
    let memory = match caller.get_export("memory") {
        Some(wasmtime::Extern::Memory(m)) => m,
        _ => return 0,
    };
    let key = match read_utf8(&memory, &*caller, key_ptr, key_len) {
        Some(k) => k,
        None => return 0,
    };

    let storage = match caller.data().storage.clone() {
        Some(s) => s,
        None => return 0, // unreachable: import only wired when storage is Some
    };

    let _guard = storage.lock.lock().unwrap_or_else(|e| e.into_inner());
    let map = match load_storage(&storage.file) {
        Ok(m) => m,
        Err(_) => return 0,
    };
    drop(_guard);

    match map.get(&key) {
        Some(value) => write_response(caller, memory, value),
        None => 0,
    }
}

fn storage_set_impl(
    caller: &mut Caller<'_, StoreData>,
    key_ptr: i32,
    key_len: i32,
    val_ptr: i32,
    val_len: i32,
) -> i32 {
    let memory = match caller.get_export("memory") {
        Some(wasmtime::Extern::Memory(m)) => m,
        _ => return 1,
    };
    let key = match read_utf8(&memory, &*caller, key_ptr, key_len) {
        Some(k) => k,
        None => return 2,
    };
    let value = match read_utf8(&memory, &*caller, val_ptr, val_len) {
        Some(v) => v,
        None => return 3,
    };

    let storage = match caller.data().storage.clone() {
        Some(s) => s,
        None => return 4,
    };

    if key.len() > storage.config.max_key_bytes {
        return 5;
    }
    if value.len() > storage.config.max_value_bytes {
        return 6;
    }

    let _guard = storage.lock.lock().unwrap_or_else(|e| e.into_inner());
    let mut map = load_storage(&storage.file).unwrap_or_default();
    map.insert(key, value);

    // Total-byte cap, computed after the insert.
    let total: usize = map.iter().map(|(k, v)| k.len() + v.len()).sum();
    if total > storage.config.max_total_bytes {
        return 7;
    }

    if save_storage_atomic(&storage.file, &map).is_err() {
        return 8;
    }
    0
}

fn load_storage(path: &Path) -> std::io::Result<BTreeMap<String, String>> {
    match std::fs::read(path) {
        Ok(bytes) => serde_json::from_slice::<BTreeMap<String, String>>(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(BTreeMap::new()),
        Err(e) => Err(e),
    }
}

fn save_storage_atomic(
    path: &Path,
    map: &BTreeMap<String, String>,
) -> std::io::Result<()> {
    let bytes = serde_json::to_vec(map)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let parent = path.parent().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "storage path has no parent")
    })?;
    std::fs::create_dir_all(parent)?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, &bytes)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Allocate space in the skill's linear memory via its own `ari_alloc`
/// export, copy `s` in, and return the packed `(ptr << 32) | len`. Returns
/// 0 if the allocation or copy fails (the skill will read 0 as a sentinel).
fn write_response(caller: &mut Caller<'_, StoreData>, memory: Memory, s: &str) -> i64 {
    let alloc = match caller.get_export("ari_alloc") {
        Some(wasmtime::Extern::Func(f)) => f,
        _ => return 0,
    };
    let alloc = match alloc.typed::<i32, i32>(&*caller) {
        Ok(t) => t,
        Err(_) => return 0,
    };
    let bytes = s.as_bytes();
    let len = bytes.len() as i32;
    let ptr = match alloc.call(&mut *caller, len) {
        Ok(p) if p > 0 => p,
        _ => return 0,
    };
    let mem_data = memory.data_mut(&mut *caller);
    let start = ptr as usize;
    let end = start.saturating_add(bytes.len());
    if end > mem_data.len() {
        return 0;
    }
    mem_data[start..end].copy_from_slice(bytes);
    ((ptr as i64) << 32) | (len as i64)
}

impl Skill for WasmSkill {
    fn id(&self) -> &str {
        &self.id
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn specificity(&self) -> Specificity {
        self.specificity
    }

    fn score(&self, input: &str, _ctx: &SkillContext) -> f32 {
        // Default path: same native pattern scorer the declarative adapter
        // uses, applied to the manifest's `metadata.ari.matching` block. The
        // WASM module is never invoked.
        //
        // Custom path: when `custom_score = true` the manifest grants the
        // module its own `score()` export, called for every input. Documented
        // as a power-user feature with a perf warning.
        if !self.custom_score {
            return self.scorer.score(input);
        }

        self.with_instance(
            |store, instance| {
                let Some((_mem, ptr, len)) = WasmSkill::write_input(store, instance, input) else {
                    return 0.0;
                };
                let score_fn = match instance.get_typed_func::<(i32, i32), f32>(&mut *store, "score")
                {
                    Ok(f) => f,
                    Err(_) => return 0.0,
                };
                score_fn.call(store, (ptr, len)).unwrap_or(0.0).clamp(0.0, 1.0)
            },
            0.0,
        )
    }

    fn execute(&self, input: &str, _ctx: &SkillContext) -> Response {
        let fallback = || Response::Text("(skill error)".to_string());
        self.with_instance(
            |store, instance| {
                let Some((memory, ptr, len)) = WasmSkill::write_input(store, instance, input) else {
                    return fallback();
                };
                let exec_fn = match instance.get_typed_func::<(i32, i32), i64>(&mut *store, "execute")
                {
                    Ok(f) => f,
                    Err(_) => return fallback(),
                };
                let packed = match exec_fn.call(&mut *store, (ptr, len)) {
                    Ok(v) => v,
                    Err(_) => return fallback(),
                };
                let resp_ptr = (packed >> 32) as i32;
                let resp_len = (packed & 0xffff_ffff) as i32;
                match read_utf8(&memory, &*store, resp_ptr, resp_len) {
                    Some(text) => Response::Text(text),
                    None => fallback(),
                }
            },
            fallback(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{Capability, MatchPattern, Matching, SkillType, SpecificityLevel};
    use std::sync::atomic::{AtomicU64, Ordering};

    /// A unique storage root per test invocation, so concurrent tests don't
    /// share state. Cleaned up by `Drop`-ing the helper if needed; for now we
    /// rely on the OS to reap /tmp.
    fn test_storage_config() -> StorageConfig {
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let mut root = std::env::temp_dir();
        root.push(format!("ari-skill-loader-test-{nanos}-{n}"));
        StorageConfig::new(root)
    }

    /// Build a minimal `AriExtension` for tests without going through YAML.
    fn fake_ari(custom_score: bool) -> AriExtension {
        AriExtension {
            id: "ai.example.test".to_string(),
            version: "0.1.0".to_string(),
            author: None,
            homepage: None,
            engine: ">=0.3".to_string(),
            capabilities: Vec::<Capability>::new(),
            platforms: None,
            languages: vec!["en".to_string()],
            skill_type: SkillType::Skill,
            specificity: SpecificityLevel::High,
            matching: Some(Matching {
                patterns: vec![MatchPattern::Keywords {
                    words: vec!["hex".to_string()],
                    weight: 0.9,
                }],
                custom_score,
            }),
            behaviour: Some(Behaviour::Wasm(WasmBehaviour {
                module: "skill.wasm".to_string(),
                memory_limit_mb: 1,
            })),
            assistant: None,
            examples: Vec::new(),
        }
    }

    fn behaviour(ext: &AriExtension) -> &WasmBehaviour {
        match &ext.behaviour {
            Some(Behaviour::Wasm(w)) => w,
            _ => unreachable!(),
        }
    }

    /// A WAT module that:
    /// - exports memory, ari_alloc (one-shot bump @ 1024), score, execute
    /// - score returns 0.42 regardless of input
    /// - execute writes "hello from wasm" at offset 2048 and returns its
    ///   packed pointer/length
    /// - logs once via the imported ari::log
    fn echo_wat() -> &'static str {
        r#"(module
  (import "ari" "log" (func $log (param i32 i32 i32)))
  (memory (export "memory") 1)
  (data (i32.const 2048) "hello from wasm")
  (data (i32.const 4096) "wasm log line")
  (global $bump (mut i32) (i32.const 1024))

  (func (export "ari_alloc") (param $size i32) (result i32)
    (local $p i32)
    local.get $size
    i32.const 0
    i32.lt_s
    if (result i32)
      i32.const 0
    else
      global.get $bump
      local.set $p
      global.get $bump
      local.get $size
      i32.add
      global.set $bump
      local.get $p
    end)

  (func (export "score") (param $ptr i32) (param $len i32) (result f32)
    f32.const 0.42)

  (func (export "execute") (param $ptr i32) (param $len i32) (result i64)
    ;; log "wasm log line" at info level
    i32.const 2
    i32.const 4096
    i32.const 13
    call $log
    ;; pack (2048 << 32) | 15
    i64.const 2048
    i64.const 32
    i64.shl
    i64.const 15
    i64.or)
)"#
    }

    fn build_skill(custom_score: bool, sink: Arc<dyn LogSink>) -> WasmSkill {
        let bytes = wat::parse_str(echo_wat()).unwrap();
        let ari = fake_ari(custom_score);
        WasmSkill::from_module_bytes(
            &ari,
            "",
            behaviour(&ari),
            &bytes,
            sink,
            &HostCapabilities::all(),
            &HttpConfig::strict(),
            &test_storage_config(),
        )
        .unwrap()
    }

    #[test]
    fn validates_required_exports_at_construction() {
        let _ = build_skill(false, Arc::new(NullLogSink));
        // No assertion beyond "build_skill did not panic" — the validator runs
        // inside from_module_bytes and would error if any export was missing.
    }

    #[test]
    fn rejects_module_missing_memory() {
        let wat = r#"(module
          (func (export "ari_alloc") (param i32) (result i32) i32.const 0)
          (func (export "score") (param i32 i32) (result f32) f32.const 0)
          (func (export "execute") (param i32 i32) (result i64) i64.const 0)
        )"#;
        let bytes = wat::parse_str(wat).unwrap();
        let ari = fake_ari(false);
        let err = WasmSkill::from_module_bytes(
            &ari,
            "",
            behaviour(&ari),
            &bytes,
            Arc::new(NullLogSink),
            &HostCapabilities::all(),
            &HttpConfig::strict(),
            &test_storage_config(),
        )
        .unwrap_err();
        assert!(matches!(err, WasmError::MissingExport("memory")));
    }

    #[test]
    fn rejects_module_with_wrong_score_signature() {
        let wat = r#"(module
          (memory (export "memory") 1)
          (func (export "ari_alloc") (param i32) (result i32) i32.const 0)
          ;; score returns i32 instead of f32
          (func (export "score") (param i32 i32) (result i32) i32.const 0)
          (func (export "execute") (param i32 i32) (result i64) i64.const 0)
        )"#;
        let bytes = wat::parse_str(wat).unwrap();
        let ari = fake_ari(false);
        let err = WasmSkill::from_module_bytes(
            &ari,
            "",
            behaviour(&ari),
            &bytes,
            Arc::new(NullLogSink),
            &HostCapabilities::all(),
            &HttpConfig::strict(),
            &test_storage_config(),
        )
        .unwrap_err();
        assert!(matches!(err, WasmError::BadExportSignature("score")));
    }

    #[test]
    fn declarative_score_path_uses_native_pattern_matcher() {
        // custom_score=false → score() does NOT enter the WASM module. It runs
        // the same native keyword scorer the declarative adapter uses, against
        // the manifest's `metadata.ari.matching` block.
        let skill = build_skill(false, Arc::new(NullLogSink));
        let ctx = SkillContext::default();
        // fake_ari has one keyword pattern: ["hex"] @ 0.9
        assert_eq!(skill.score("convert ff to hex", &ctx), 0.9);
        assert_eq!(skill.score("anything else", &ctx), 0.0);
        // whole-word matching, "hexagonal" must NOT match "hex"
        assert_eq!(skill.score("hexagonal numbers", &ctx), 0.0);
    }

    #[test]
    fn custom_score_path_calls_into_wasm() {
        // custom_score=true → score() goes through the module, which returns 0.42
        let skill = build_skill(true, Arc::new(NullLogSink));
        let ctx = SkillContext::default();
        assert_eq!(skill.score("anything", &ctx), 0.42);
    }

    #[test]
    fn execute_returns_text_from_wasm_memory() {
        let skill = build_skill(false, Arc::new(NullLogSink));
        let ctx = SkillContext::default();
        match skill.execute("convert ff to decimal", &ctx) {
            Response::Text(t) => assert_eq!(t, "hello from wasm"),
            _ => panic!("expected text response"),
        }
    }

    #[test]
    fn host_log_import_delivers_to_sink() {
        let sink = CapturingLogSink::new();
        let skill = build_skill(false, Arc::new(sink.clone()));
        let _ = skill.execute("anything", &SkillContext::default());
        let lines = sink.lines();
        assert_eq!(lines.len(), 1);
        assert!(matches!(lines[0].0, LogLevel::Info));
        assert_eq!(lines[0].1, "wasm log line");
    }

    #[test]
    fn id_and_specificity_are_propagated() {
        let skill = build_skill(false, Arc::new(NullLogSink));
        assert_eq!(skill.id(), "ai.example.test");
        assert_eq!(skill.specificity(), Specificity::High);
    }

    #[test]
    fn fuel_exhaustion_returns_fallback_response() {
        // Module whose execute() loops forever via an unbounded br loop.
        let wat = r#"(module
          (memory (export "memory") 1)
          (func (export "ari_alloc") (param i32) (result i32) i32.const 0)
          (func (export "score") (param i32 i32) (result f32) f32.const 0)
          (func (export "execute") (param i32 i32) (result i64)
            (loop $forever
              br $forever)
            i64.const 0)
        )"#;
        let bytes = wat::parse_str(wat).unwrap();
        let ari = fake_ari(false);
        let skill = WasmSkill::from_module_bytes(
            &ari,
            "",
            behaviour(&ari),
            &bytes,
            Arc::new(NullLogSink),
            &HostCapabilities::all(),
            &HttpConfig::strict(),
            &test_storage_config(),
        )
        .unwrap();
        match skill.execute("anything", &SkillContext::default()) {
            Response::Text(t) => assert_eq!(t, "(skill error)"),
            _ => panic!("expected fallback text"),
        }
    }

    #[test]
    fn install_rejects_skill_needing_unprovided_capability() {
        let bytes = wat::parse_str(echo_wat()).unwrap();
        let mut ari = fake_ari(false);
        ari.capabilities = vec![Capability::Http, Capability::Notifications];
        let host = HostCapabilities::pure_frontend(); // grants notifications, not http
        let err = WasmSkill::from_module_bytes(
            &ari,
            "",
            behaviour(&ari),
            &bytes,
            Arc::new(NullLogSink),
            &host,
            &HttpConfig::strict(),
            &test_storage_config(),
        )
        .unwrap_err();
        match err {
            WasmError::MissingCapabilities { missing } => {
                assert_eq!(missing, vec![Capability::Http]);
            }
            other => panic!("expected MissingCapabilities, got {other:?}"),
        }
    }

    #[test]
    fn install_succeeds_when_all_declared_caps_are_granted() {
        let bytes = wat::parse_str(echo_wat()).unwrap();
        let mut ari = fake_ari(false);
        ari.capabilities = vec![Capability::Notifications, Capability::LaunchApp];
        let skill = WasmSkill::from_module_bytes(
            &ari,
            "",
            behaviour(&ari),
            &bytes,
            Arc::new(NullLogSink),
            &HostCapabilities::pure_frontend(),
            &HttpConfig::strict(),
            &test_storage_config(),
        )
        .unwrap();
        let caps = skill.granted_capabilities();
        assert!(caps.contains(&Capability::Notifications));
        assert!(caps.contains(&Capability::LaunchApp));
    }

    /// A WAT module that on `execute` queries `ari::get_capability` for
    /// "notifications" and returns "yes" or "no" as a UTF-8 string.
    fn capability_query_wat() -> &'static str {
        r#"(module
  (import "ari" "get_capability" (func $gc (param i32 i32) (result i32)))
  (memory (export "memory") 1)
  (data (i32.const 256) "notifications")
  (data (i32.const 2048) "yes")
  (data (i32.const 2052) "no")
  (global $bump (mut i32) (i32.const 1024))
  (func (export "ari_alloc") (param $size i32) (result i32)
    (local $p i32)
    global.get $bump
    local.set $p
    global.get $bump
    local.get $size
    i32.add
    global.set $bump
    local.get $p)
  (func (export "score") (param i32 i32) (result f32)
    f32.const 0.95)
  (func (export "execute") (param i32 i32) (result i64)
    i32.const 256
    i32.const 13
    call $gc
    if (result i64)
      i64.const 2048
      i64.const 32
      i64.shl
      i64.const 3
      i64.or
    else
      i64.const 2052
      i64.const 32
      i64.shl
      i64.const 2
      i64.or
    end)
)"#
    }

    fn build_capability_skill(declared: Vec<Capability>, host: HostCapabilities) -> WasmSkill {
        let bytes = wat::parse_str(capability_query_wat()).unwrap();
        let mut ari = fake_ari(false);
        ari.capabilities = declared;
        WasmSkill::from_module_bytes(
            &ari,
            "",
            behaviour(&ari),
            &bytes,
            Arc::new(NullLogSink),
            &host,
            &HttpConfig::strict(),
            &test_storage_config(),
        )
        .unwrap()
    }

    #[test]
    fn get_capability_returns_yes_when_declared_and_granted() {
        let skill = build_capability_skill(
            vec![Capability::Notifications],
            HostCapabilities::pure_frontend(),
        );
        match skill.execute("anything", &SkillContext::default()) {
            Response::Text(t) => assert_eq!(t, "yes"),
            _ => panic!(),
        }
    }

    #[test]
    fn get_capability_returns_no_when_skill_did_not_declare_it() {
        // Host grants notifications, but skill didn't declare it. The sneak
        // guard kicks in: get_capability answers 0 ("no") so the skill can't
        // use a capability it never asked the user's consent for.
        let skill = build_capability_skill(vec![], HostCapabilities::pure_frontend());
        match skill.execute("anything", &SkillContext::default()) {
            Response::Text(t) => assert_eq!(t, "no"),
            _ => panic!(),
        }
    }

    /// A WAT module that imports `ari::http_fetch` and uses it on `execute`
    /// to GET a hardcoded URL stored at offset 256.
    fn http_fetch_wat(url: &str) -> String {
        // We bake the URL into the data segment so the WAT is self-contained.
        // The URL goes at offset 256, length is `url.len()`.
        format!(r#"(module
  (import "ari" "http_fetch" (func $fetch (param i32 i32) (result i64)))
  (memory (export "memory") 1)
  (data (i32.const 256) "{url}")
  (global $bump (mut i32) (i32.const 8192))
  (func (export "ari_alloc") (param $size i32) (result i32)
    (local $p i32)
    global.get $bump
    local.set $p
    global.get $bump
    local.get $size
    i32.add
    global.set $bump
    local.get $p)
  (func (export "score") (param i32 i32) (result f32) f32.const 0.95)
  (func (export "execute") (param i32 i32) (result i64)
    i32.const 256
    i32.const {len}
    call $fetch)
)"#, url = url, len = url.len())
    }

    #[test]
    fn sneak_guard_rejects_undeclared_http_import() {
        // Module imports ari::http_fetch but the manifest declares no caps.
        // Even with HostCapabilities::all() granting http, the import scan
        // catches the mismatch and refuses to load.
        let bytes = wat::parse_str(&http_fetch_wat("https://example.com")).unwrap();
        let ari = fake_ari(false); // capabilities: []
        let err = WasmSkill::from_module_bytes(
            &ari,
            "",
            behaviour(&ari),
            &bytes,
            Arc::new(NullLogSink),
            &HostCapabilities::all(),
            &HttpConfig::strict(),
            &test_storage_config(),
        )
        .unwrap_err();
        match err {
            WasmError::UndeclaredCapability { import, required } => {
                assert_eq!(import, "http_fetch");
                assert_eq!(required, Capability::Http);
            }
            other => panic!("expected UndeclaredCapability, got {other:?}"),
        }
    }

    #[test]
    fn unknown_host_import_is_rejected() {
        let wat = r#"(module
          (import "ari" "telepathy" (func $t (param i32) (result i32)))
          (memory (export "memory") 1)
          (func (export "ari_alloc") (param i32) (result i32) i32.const 0)
          (func (export "score") (param i32 i32) (result f32) f32.const 0)
          (func (export "execute") (param i32 i32) (result i64) i64.const 0)
        )"#;
        let bytes = wat::parse_str(wat).unwrap();
        let ari = fake_ari(false);
        let err = WasmSkill::from_module_bytes(
            &ari,
            "",
            behaviour(&ari),
            &bytes,
            Arc::new(NullLogSink),
            &HostCapabilities::all(),
            &HttpConfig::strict(),
            &test_storage_config(),
        )
        .unwrap_err();
        assert!(matches!(err, WasmError::UnknownHostImport(ref n) if n == "telepathy"));
    }

    #[test]
    fn http_fetch_against_local_server_returns_status_and_body() {
        let server = TestServer::start("HTTP/1.1 200 OK\r\nContent-Length: 13\r\n\r\nhello, world!");
        let url = format!("http://{}/", server.addr);

        let bytes = wat::parse_str(&http_fetch_wat(&url)).unwrap();
        let mut ari = fake_ari(false);
        ari.capabilities = vec![Capability::Http];
        let skill = WasmSkill::from_module_bytes(
            &ari,
            "",
            behaviour(&ari),
            &bytes,
            Arc::new(NullLogSink),
            &HostCapabilities::all(),
            &HttpConfig::permissive_for_tests(),
            &test_storage_config(),
        )
        .unwrap();

        let resp = skill.execute("doesn't matter", &SkillContext::default());
        let text = match resp {
            Response::Text(t) => t,
            other => panic!("expected text, got {other:?}"),
        };
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["status"], 200);
        assert_eq!(v["body"], "hello, world!");
    }

    #[test]
    fn http_fetch_rejects_disallowed_scheme() {
        // Server speaks plain HTTP but config only allows https. The error
        // arrives as a JSON envelope with status 0.
        let server = TestServer::start("HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok");
        let url = format!("http://{}/", server.addr);

        let bytes = wat::parse_str(&http_fetch_wat(&url)).unwrap();
        let mut ari = fake_ari(false);
        ari.capabilities = vec![Capability::Http];
        let skill = WasmSkill::from_module_bytes(
            &ari,
            "",
            behaviour(&ari),
            &bytes,
            Arc::new(NullLogSink),
            &HostCapabilities::all(),
            &HttpConfig::strict(), // https only
            &test_storage_config(),
        )
        .unwrap();

        let text = match skill.execute("x", &SkillContext::default()) {
            Response::Text(t) => t,
            other => panic!("expected text, got {other:?}"),
        };
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["status"], 0);
        assert!(v["error"].as_str().unwrap().contains("scheme not allowed"));
    }

    #[test]
    fn http_fetch_propagates_real_status_codes() {
        // 404 should come back as status: 404, not as an error.
        let server = TestServer::start("HTTP/1.1 404 Not Found\r\nContent-Length: 9\r\n\r\nnot here!");
        let url = format!("http://{}/", server.addr);

        let bytes = wat::parse_str(&http_fetch_wat(&url)).unwrap();
        let mut ari = fake_ari(false);
        ari.capabilities = vec![Capability::Http];
        let skill = WasmSkill::from_module_bytes(
            &ari,
            "",
            behaviour(&ari),
            &bytes,
            Arc::new(NullLogSink),
            &HostCapabilities::all(),
            &HttpConfig::permissive_for_tests(),
            &test_storage_config(),
        )
        .unwrap();

        let text = match skill.execute("x", &SkillContext::default()) {
            Response::Text(t) => t,
            _ => panic!(),
        };
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["status"], 404);
        assert_eq!(v["body"], "not here!");
    }

    #[test]
    fn http_fetch_enforces_body_size_cap() {
        let big_body = "X".repeat(2048);
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{}",
            big_body.len(),
            big_body
        );
        let server = TestServer::start(&response);
        let url = format!("http://{}/", server.addr);

        let bytes = wat::parse_str(&http_fetch_wat(&url)).unwrap();
        let mut ari = fake_ari(false);
        ari.capabilities = vec![Capability::Http];
        let mut config = HttpConfig::permissive_for_tests();
        config.max_body_bytes = 1024;
        let skill = WasmSkill::from_module_bytes(
            &ari,
            "",
            behaviour(&ari),
            &bytes,
            Arc::new(NullLogSink),
            &HostCapabilities::all(),
            &config,
            &test_storage_config(),
        )
        .unwrap();

        let text = match skill.execute("x", &SkillContext::default()) {
            Response::Text(t) => t,
            _ => panic!(),
        };
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["status"], 0);
        assert!(v["error"].as_str().unwrap().contains("byte limit"));
    }

    #[test]
    fn http_fetch_import_not_wired_when_skill_lacks_http_grant() {
        // Skill manifest declares nothing. Module imports ari::http_fetch.
        // The sneak guard catches it before we even get to the linker, so
        // this is the same scenario as `sneak_guard_rejects_undeclared_http_import`
        // — kept as a regression test under a friendlier name.
        let bytes = wat::parse_str(&http_fetch_wat("http://localhost/")).unwrap();
        let ari = fake_ari(false);
        assert!(WasmSkill::from_module_bytes(
            &ari,
            "",
            behaviour(&ari),
            &bytes,
            Arc::new(NullLogSink),
            &HostCapabilities::pure_frontend(), // does not grant http
            &HttpConfig::permissive_for_tests(),
            &test_storage_config(),
        )
        .is_err());
    }

    #[test]
    fn fresh_store_per_call_means_no_state_leak_between_calls() {
        // The bump allocator in echo_wat starts at 1024 and grows. If state
        // leaked, repeated calls would push the bump pointer up until it
        // overran the data we hardcoded at offset 2048. With per-call fresh
        // stores, every call is identical.
        let skill = build_skill(false, Arc::new(NullLogSink));
        let ctx = SkillContext::default();
        for _ in 0..50 {
            match skill.execute("ping", &ctx) {
                Response::Text(t) => assert_eq!(t, "hello from wasm"),
                _ => panic!(),
            }
        }
    }

    /// A WAT module that on `execute`:
    ///   1. calls `storage_get("counter")`
    ///   2. if it returned 0 (not found), writes "1" to memory at offset 4096
    ///      and stores it via `storage_set("counter", "1")`
    ///   3. otherwise reads the stored byte (we use a single-digit counter to
    ///      keep WAT brutally simple), increments it, writes the new digit at
    ///      offset 4097, and calls `storage_set("counter", "<digit>")`
    ///   4. returns the new digit as a one-byte string
    ///
    /// Hardcoded data:
    ///   "counter" at offset 256 (length 7)
    ///   bump allocator starts at 8192 to leave the host plenty of room above
    fn counter_wat() -> &'static str {
        r#"(module
  (import "ari" "storage_get" (func $get (param i32 i32) (result i64)))
  (import "ari" "storage_set" (func $set (param i32 i32 i32 i32) (result i32)))
  (memory (export "memory") 1)
  (data (i32.const 256) "counter")
  (global $bump (mut i32) (i32.const 8192))

  (func (export "ari_alloc") (param $size i32) (result i32)
    (local $p i32)
    global.get $bump
    local.set $p
    global.get $bump
    local.get $size
    i32.add
    global.set $bump
    local.get $p)

  (func (export "score") (param i32 i32) (result f32) f32.const 0.95)

  (func (export "execute") (param i32 i32) (result i64)
    (local $packed i64)
    (local $existing_ptr i32)
    (local $byte i32)
    (local $new_byte i32)
    ;; storage_get("counter")
    i32.const 256
    i32.const 7
    call $get
    local.set $packed

    local.get $packed
    i64.const 0
    i64.eq
    if (result i64)
      ;; not found: write '1' at 4097, store it, return (4097 << 32) | 1
      i32.const 4097
      i32.const 49 ;; ASCII '1'
      i32.store8
      i32.const 256 i32.const 7 i32.const 4097 i32.const 1
      call $set
      drop
      i64.const 4097 i64.const 32 i64.shl i64.const 1 i64.or
    else
      ;; found: read existing byte from packed pointer (high 32 bits),
      ;; increment, write new byte at 4097, store, return packed.
      local.get $packed
      i64.const 32
      i64.shr_u
      i32.wrap_i64
      local.set $existing_ptr
      local.get $existing_ptr
      i32.load8_u
      local.set $byte
      ;; if byte >= '9', wrap to '1' so we don't go past ASCII digits
      local.get $byte
      i32.const 57 ;; '9'
      i32.ge_s
      if (result i32)
        i32.const 49 ;; '1'
      else
        local.get $byte
        i32.const 1
        i32.add
      end
      local.set $new_byte
      i32.const 4097
      local.get $new_byte
      i32.store8
      i32.const 256 i32.const 7 i32.const 4097 i32.const 1
      call $set
      drop
      i64.const 4097 i64.const 32 i64.shl i64.const 1 i64.or
    end)
)"#
    }

    fn build_storage_skill(storage: StorageConfig) -> WasmSkill {
        let bytes = wat::parse_str(counter_wat()).unwrap();
        let mut ari = fake_ari(false);
        ari.capabilities = vec![Capability::StorageKv];
        WasmSkill::from_module_bytes(
            &ari,
            "",
            behaviour(&ari),
            &bytes,
            Arc::new(NullLogSink),
            &HostCapabilities::all(),
            &HttpConfig::strict(),
            &storage,
        )
        .unwrap()
    }

    #[test]
    fn storage_round_trip_counter() {
        let storage = test_storage_config();
        let skill = build_storage_skill(storage.clone());
        let ctx = SkillContext::default();

        for expected in &["1", "2", "3", "4"] {
            match skill.execute("tick", &ctx) {
                Response::Text(t) => assert_eq!(&t, *expected),
                _ => panic!(),
            }
        }
    }

    #[test]
    fn storage_persists_across_skill_instances() {
        let storage = test_storage_config();

        // First instance: tick three times.
        {
            let skill = build_storage_skill(storage.clone());
            for _ in 0..3 {
                let _ = skill.execute("tick", &SkillContext::default());
            }
        }

        // Second instance, same storage_config → same on-disk file. Should
        // continue from "4".
        let skill = build_storage_skill(storage);
        match skill.execute("tick", &SkillContext::default()) {
            Response::Text(t) => assert_eq!(t, "4"),
            _ => panic!(),
        }
    }

    #[test]
    fn sneak_guard_rejects_undeclared_storage_import() {
        let bytes = wat::parse_str(counter_wat()).unwrap();
        let ari = fake_ari(false); // capabilities: []
        let err = WasmSkill::from_module_bytes(
            &ari,
            "",
            behaviour(&ari),
            &bytes,
            Arc::new(NullLogSink),
            &HostCapabilities::all(),
            &HttpConfig::strict(),
            &test_storage_config(),
        )
        .unwrap_err();
        match err {
            WasmError::UndeclaredCapability { import, required } => {
                // Either storage_get or storage_set will be the first
                // problematic import the scanner finds.
                assert!(import == "storage_get" || import == "storage_set");
                assert_eq!(required, Capability::StorageKv);
            }
            other => panic!("expected UndeclaredCapability, got {other:?}"),
        }
    }

    #[test]
    fn storage_set_failure_when_value_exceeds_per_value_cap() {
        // A WAT module that calls storage_set with a 100-byte value, against
        // a config with max_value_bytes = 10. Returns the set return code as
        // a one-byte ASCII digit.
        let wat = r#"(module
          (import "ari" "storage_set" (func $set (param i32 i32 i32 i32) (result i32)))
          (memory (export "memory") 1)
          (data (i32.const 256) "k")
          (data (i32.const 512) "abcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghij")
          (global $bump (mut i32) (i32.const 8192))
          (func (export "ari_alloc") (param i32) (result i32)
            (local $p i32) global.get $bump local.set $p
            global.get $bump local.get 0 i32.add global.set $bump local.get $p)
          (func (export "score") (param i32 i32) (result f32) f32.const 0.95)
          (func (export "execute") (param i32 i32) (result i64)
            (local $rc i32)
            i32.const 256 i32.const 1 i32.const 512 i32.const 100
            call $set
            local.set $rc
            ;; write the rc as ASCII digit at 4096
            i32.const 4096
            local.get $rc
            i32.const 48 ;; '0'
            i32.add
            i32.store8
            i64.const 4096 i64.const 32 i64.shl i64.const 1 i64.or)
        )"#;
        let bytes = wat::parse_str(wat).unwrap();
        let mut ari = fake_ari(false);
        ari.capabilities = vec![Capability::StorageKv];
        let storage = test_storage_config().with_max_value_bytes(10);
        let skill = WasmSkill::from_module_bytes(
            &ari,
            "",
            behaviour(&ari),
            &bytes,
            Arc::new(NullLogSink),
            &HostCapabilities::all(),
            &HttpConfig::strict(),
            &storage,
        )
        .unwrap();
        match skill.execute("x", &SkillContext::default()) {
            Response::Text(t) => assert_eq!(t, "6"), // rc 6 = max_value_bytes exceeded
            _ => panic!(),
        }
    }

    #[test]
    fn storage_get_returns_zero_for_missing_key() {
        // Module that just calls storage_get and returns 1 if it found
        // something, 0 otherwise (encoded as ASCII).
        let wat = r#"(module
          (import "ari" "storage_get" (func $get (param i32 i32) (result i64)))
          (memory (export "memory") 1)
          (data (i32.const 256) "nope")
          (global $bump (mut i32) (i32.const 8192))
          (func (export "ari_alloc") (param i32) (result i32)
            (local $p i32) global.get $bump local.set $p
            global.get $bump local.get 0 i32.add global.set $bump local.get $p)
          (func (export "score") (param i32 i32) (result f32) f32.const 0.95)
          (func (export "execute") (param i32 i32) (result i64)
            (local $packed i64)
            i32.const 256 i32.const 4
            call $get
            local.set $packed
            i32.const 4096
            local.get $packed
            i64.const 0
            i64.eq
            if (result i32) i32.const 48 else i32.const 49 end
            i32.store8
            i64.const 4096 i64.const 32 i64.shl i64.const 1 i64.or)
        )"#;
        let bytes = wat::parse_str(wat).unwrap();
        let mut ari = fake_ari(false);
        ari.capabilities = vec![Capability::StorageKv];
        let skill = WasmSkill::from_module_bytes(
            &ari,
            "",
            behaviour(&ari),
            &bytes,
            Arc::new(NullLogSink),
            &HostCapabilities::all(),
            &HttpConfig::strict(),
            &test_storage_config(),
        )
        .unwrap();
        match skill.execute("x", &SkillContext::default()) {
            Response::Text(t) => assert_eq!(t, "0"),
            _ => panic!(),
        }
    }

    /// Tiny single-shot HTTP server for tests. Binds to 127.0.0.1:0, accepts
    /// connections in a background thread, drains the request, and writes the
    /// configured raw HTTP response back. No dependency on a real HTTP crate
    /// — we just need something to point reqwest at.
    struct TestServer {
        addr: std::net::SocketAddr,
        _thread: std::thread::JoinHandle<()>,
    }

    impl TestServer {
        fn start(raw_response: &str) -> Self {
            use std::io::{Read, Write};
            use std::net::TcpListener;
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            listener.set_nonblocking(false).unwrap();
            let addr = listener.local_addr().unwrap();
            let response = raw_response.to_string();
            let thread = std::thread::spawn(move || {
                // Serve a single request, then exit. Good enough for tests
                // that fire one fetch.
                if let Ok((mut stream, _)) = listener.accept() {
                    // Drain the request headers (read until \r\n\r\n).
                    let mut buf = [0u8; 1024];
                    let mut acc: Vec<u8> = Vec::new();
                    loop {
                        let n = match stream.read(&mut buf) {
                            Ok(0) => break,
                            Ok(n) => n,
                            Err(_) => break,
                        };
                        acc.extend_from_slice(&buf[..n]);
                        if acc.windows(4).any(|w| w == b"\r\n\r\n") {
                            break;
                        }
                    }
                    let _ = stream.write_all(response.as_bytes());
                    let _ = stream.flush();
                }
            });
            Self {
                addr,
                _thread: thread,
            }
        }
    }
}
