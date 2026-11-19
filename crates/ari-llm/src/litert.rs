//! LiteRT-LM, Google's on-device LLM runtime, as the second runtime beside
//! llama.cpp. On a Pixel 10 Pro Fold it answered the same questions about
//! three times sooner on Medium, twice on Large and 1.7 times on Small, with
//! the same accuracy (ari-android#52).
//!
//! It is reached through its C API, `liblitert-lm.so`, opened at run time
//! rather than linked: the library has no SONAME, so Android's linker refuses
//! any binary linked against its versioned symbols. On Android the app ships
//! it next to `libari_ffi.so`, where the bare name finds it; elsewhere it has
//! to be on the loader's search path (`LD_LIBRARY_PATH`).
//!
//! Each question runs in a fresh conversation, as each llama.cpp question
//! runs in a fresh context: the system prompt and history go in as the
//! conversation's opening messages, and the model's own chat template, from
//! the model file, wraps them.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::Path;
use std::sync::{mpsc, OnceLock};

use crate::{Accelerator, Generation};

/// Model files LiteRT-LM runs: its own `.litertlm`, and MediaPipe's older
/// `.task`, which it also loads. Small ships as `.task`: Google's
/// `.litertlm` build of Gemma 3 1B scored 20 of 38 against llama.cpp's 31,
/// and its 8-bit `.task` build 31 (ari-android#52).
pub(crate) fn handles(path: &Path) -> bool {
    matches!(path.extension().and_then(|e| e.to_str()), Some("litertlm" | "task"))
}

/// The C API major version these bindings follow. A library from another
/// major version may have changed any signature here, so it isn't used.
const API_MAJOR: c_int = 1;

/// Tokens each message's turn markers add around its text: Gemma's
/// `<start_of_turn>user\n…<end_of_turn>\n` is five, so eight leaves room.
/// LiteRT-LM renders the template itself, so history fitting counts the
/// messages' text and adds this per message.
const TURN_OVERHEAD_TOKENS: usize = 8;

/// `kLiteRtLmSamplerTypeTopP`. Greedy and top-k are "not implemented yet"
/// on the CPU in 0.19.1; top-p after top-k 1 leaves one candidate, the
/// same pick as greedy.
const SAMPLER_TOP_P: c_int = 2;

type Status = c_int;
type Handle = *mut c_void;
type StreamCallback = unsafe extern "C" fn(data: *mut c_void, chunk: *const c_void);

/// Declares the C functions used here and loads them all at once. Each
/// field is the C name without its `litert_lm_` prefix.
macro_rules! c_api {
    ($($name:ident: fn($($arg:ty),*) $(-> $ret:ty)?;)*) => {
        struct Api {
            $($name: unsafe extern "C" fn($($arg),*) $(-> $ret)?,)*
            _library: libloading::Library,
        }

        impl Api {
            /// SAFETY: `library` must be LiteRT-LM's C API at major version
            /// [`API_MAJOR`], whose functions have the signatures above.
            unsafe fn bind(library: libloading::Library) -> Result<Self, String> {
                $(
                    let $name = *library
                        .get::<unsafe extern "C" fn($($arg),*) $(-> $ret)?>(concat!("litert_lm_", stringify!($name)))
                        .map_err(|e| format!("liblitert-lm.so has no litert_lm_{}: {e}", stringify!($name)))?;
                )*
                Ok(Api { $($name,)* _library: library })
            }
        }
    };
}

c_api! {
    get_api_version: fn(*mut c_int, *mut c_int, *mut c_int) -> Status;
    get_last_error_message: fn() -> *const c_char;
    engine_settings_create: fn(*const c_char, *const c_char, *const c_char, *const c_char, *mut Handle) -> Status;
    engine_settings_set_max_num_tokens: fn(Handle, c_int) -> Status;
    engine_settings_delete: fn(Handle);
    engine_create: fn(Handle, *mut Handle) -> Status;
    engine_delete: fn(Handle);
    engine_tokenize: fn(Handle, *const c_char, *mut Handle) -> Status;
    tokenize_result_get_num_tokens: fn(Handle, *mut usize) -> Status;
    tokenize_result_delete: fn(Handle);
    sampler_params_create: fn(c_int, *mut Handle) -> Status;
    sampler_params_set_top_k: fn(Handle, i32) -> Status;
    sampler_params_delete: fn(Handle);
    session_config_create: fn(*mut Handle) -> Status;
    session_config_set_max_output_tokens: fn(Handle, c_int) -> Status;
    session_config_set_sampler_params: fn(Handle, Handle) -> Status;
    session_config_delete: fn(Handle);
    thinking_config_create: fn(*mut Handle) -> Status;
    thinking_config_set_enable_thinking: fn(Handle, bool) -> Status;
    thinking_config_delete: fn(Handle);
    conversation_config_create: fn(*mut Handle) -> Status;
    conversation_config_set_session_config: fn(Handle, Handle) -> Status;
    conversation_config_set_system_message: fn(Handle, *const c_char) -> Status;
    conversation_config_set_messages: fn(Handle, *const c_char) -> Status;
    conversation_config_set_thinking_config: fn(Handle, Handle) -> Status;
    conversation_config_delete: fn(Handle);
    conversation_create: fn(Handle, Handle, *mut Handle) -> Status;
    conversation_delete: fn(Handle);
    conversation_send_message_stream: fn(Handle, *const c_char, *const c_char, Handle, StreamCallback, *mut c_void) -> Status;
    stream_chunk_get_text: fn(*const c_void, *mut *const c_char) -> Status;
    stream_chunk_get_error: fn(*const c_void, *mut *const c_char) -> Status;
    stream_chunk_is_final: fn(*const c_void, *mut bool) -> Status;
}

/// The library, opened once per process. The error says why it couldn't be.
fn api() -> Result<&'static Api, String> {
    static API: OnceLock<Result<Api, String>> = OnceLock::new();
    API.get_or_init(|| {
        // SAFETY: loading runs the library's initialisers, which have no
        // preconditions; the version check below comes before any other call.
        let library = unsafe { libloading::Library::new("liblitert-lm.so") }
            .map_err(|e| format!("can't open liblitert-lm.so: {e}"))?;
        // SAFETY: every major-1 library exports these signatures.
        let api = unsafe { Api::bind(library) }?;
        let (mut major, mut minor, mut patch) = (0, 0, 0);
        // SAFETY: three valid out-pointers.
        api.check("get_api_version", unsafe { (api.get_api_version)(&mut major, &mut minor, &mut patch) })?;
        if major != API_MAJOR {
            return Err(format!("liblitert-lm.so has C API {major}.{minor}.{patch}; Ari needs {API_MAJOR}.x"));
        }
        Ok(api)
    })
    .as_ref()
    .map_err(Clone::clone)
}

impl Api {
    /// `Ok` for a successful call, or the library's own message for the
    /// failure, which it keeps per thread until the next failure.
    fn check(&self, what: &str, status: Status) -> Result<(), String> {
        if status == 0 {
            return Ok(());
        }
        // SAFETY: no arguments; the result is null or a NUL-terminated
        // string the library owns, valid until this thread's next failure.
        let message = unsafe { (self.get_last_error_message)() };
        let message = if message.is_null() {
            String::new()
        } else {
            // SAFETY: as above, and read before any other call.
            unsafe { CStr::from_ptr(message) }.to_string_lossy().into_owned()
        };
        Err(format!("litert_lm_{what} failed (status {status}): {message}"))
    }
}

/// A handle the library created, deleted with its own destructor when this
/// goes out of scope, so an early return can't leak it.
struct Owned {
    handle: Handle,
    delete: unsafe extern "C" fn(Handle),
}

impl Owned {
    /// Runs `create`, which writes a new handle to its out-pointer.
    fn create(
        api: &Api,
        what: &str,
        delete: unsafe extern "C" fn(Handle),
        create: impl FnOnce(*mut Handle) -> Status,
    ) -> Result<Self, String> {
        let mut handle: Handle = std::ptr::null_mut();
        api.check(what, create(&mut handle))?;
        Ok(Owned { handle, delete })
    }
}

impl Drop for Owned {
    fn drop(&mut self) {
        // SAFETY: the handle came from the matching create call and is
        // deleted once; the destructors accept null.
        unsafe { (self.delete)(self.handle) }
    }
}

/// A model loaded into LiteRT-LM, ready for questions.
pub(crate) struct LoadedModel {
    api: &'static Api,
    /// What it runs on, for logs: "LiteRT-LM GPU", "LiteRT-LM CPU", or the
    /// CPU with the reason the GPU didn't load.
    runtime: String,
    // Field order is drop order: the engine before the settings it was made from.
    engine: Owned,
    _settings: Owned,
}

impl LoadedModel {
    /// Load the model at `path`, allowing `context_tokens` for a prompt and
    /// its answer together, on `accelerator`. A model the GPU won't load
    /// goes on the CPU instead, and [`Self::runtime`] says why.
    pub(crate) fn load(path: &Path, context_tokens: usize, accelerator: Accelerator) -> Result<Self, String> {
        match accelerator {
            Accelerator::Cpu => Self::load_on(path, context_tokens, c"cpu", "LiteRT-LM CPU".into()),
            Accelerator::Gpu => Self::load_on(path, context_tokens, c"gpu", "LiteRT-LM GPU".into()).or_else(|gpu| {
                Self::load_on(path, context_tokens, c"cpu", format!("LiteRT-LM CPU (GPU failed: {gpu})"))
            }),
        }
    }

    fn load_on(path: &Path, context_tokens: usize, backend: &CStr, runtime: String) -> Result<Self, String> {
        let api = api()?;
        let model = CString::new(path.to_string_lossy().as_bytes()).map_err(|e| e.to_string())?;
        // SAFETY (all calls below): arguments are live handles from this
        // library, NUL-terminated strings, or valid out-pointers.
        let settings = Owned::create(api, "engine_settings_create", api.engine_settings_delete, |out| unsafe {
            (api.engine_settings_create)(model.as_ptr(), backend.as_ptr(), std::ptr::null(), std::ptr::null(), out)
        })?;
        api.check("engine_settings_set_max_num_tokens", unsafe {
            (api.engine_settings_set_max_num_tokens)(settings.handle, context_tokens as c_int)
        })?;
        let engine = Owned::create(api, "engine_create", api.engine_delete, |out| unsafe {
            (api.engine_create)(settings.handle, out)
        })?;
        remove_unused_caches(path, backend == c"gpu");
        Ok(LoadedModel { api, runtime, engine, _settings: settings })
    }

    pub(crate) fn runtime(&self) -> &str {
        &self.runtime
    }

    /// How many tokens `text` takes, without the start token the tokenizer
    /// may add.
    fn tokens(&self, text: &str) -> Result<usize, String> {
        let count = |text: &CStr| -> Result<usize, String> {
            let api = self.api;
            // SAFETY: a live engine, a NUL-terminated string, out-pointers.
            let result = Owned::create(api, "engine_tokenize", api.tokenize_result_delete, |out| unsafe {
                (api.engine_tokenize)(self.engine.handle, text.as_ptr(), out)
            })?;
            let mut n = 0usize;
            api.check("tokenize_result_get_num_tokens", unsafe {
                (api.tokenize_result_get_num_tokens)(result.handle, &mut n)
            })?;
            Ok(n)
        };
        let text = CString::new(text.replace('\0', "")).map_err(|e| e.to_string())?;
        Ok(count(&text)?.saturating_sub(count(c"")?))
    }

    /// An estimate, from above, of the tokens the prompt for `system`,
    /// `history` and `user` takes once the chat template wraps it.
    pub(crate) fn prompt_tokens(&self, system: &str, history: &[(String, String)], user: &str) -> Result<usize, String> {
        let mut total = 0;
        for text in std::iter::once(system).chain(history.iter().map(|(_, t)| t.as_str())).chain([user]) {
            total += self.tokens(text)? + TURN_OVERHEAD_TOKENS;
        }
        Ok(total)
    }

    /// Answer `user` after `system` (none when empty) and `history`, at
    /// most `max_tokens` long, greedily, with reasoning off, showing
    /// `on_text` the answer so far as it streams in.
    pub(crate) fn generate(
        &self,
        system: &str,
        history: &[(String, String)],
        user: &str,
        max_tokens: usize,
        on_text: &mut dyn FnMut(&str),
    ) -> Result<Generation, String> {
        let api = self.api;
        let json = |value: serde_json::Value| CString::new(value.to_string()).map_err(|e| e.to_string());
        // SAFETY (all calls below): arguments are live handles from this
        // library, NUL-terminated strings, or valid out-pointers. Setters
        // copy what they are given; every handle outlives its last use.
        let sampler = Owned::create(api, "sampler_params_create", api.sampler_params_delete, |out| unsafe {
            (api.sampler_params_create)(SAMPLER_TOP_P, out)
        })?;
        api.check("sampler_params_set_top_k", unsafe { (api.sampler_params_set_top_k)(sampler.handle, 1) })?;
        let session = Owned::create(api, "session_config_create", api.session_config_delete, |out| unsafe {
            (api.session_config_create)(out)
        })?;
        api.check("session_config_set_max_output_tokens", unsafe {
            (api.session_config_set_max_output_tokens)(session.handle, max_tokens as c_int)
        })?;
        api.check("session_config_set_sampler_params", unsafe {
            (api.session_config_set_sampler_params)(session.handle, sampler.handle)
        })?;
        let thinking = Owned::create(api, "thinking_config_create", api.thinking_config_delete, |out| unsafe {
            (api.thinking_config_create)(out)
        })?;
        api.check("thinking_config_set_enable_thinking", unsafe {
            (api.thinking_config_set_enable_thinking)(thinking.handle, false)
        })?;
        let config = Owned::create(api, "conversation_config_create", api.conversation_config_delete, |out| unsafe {
            (api.conversation_config_create)(out)
        })?;
        api.check("conversation_config_set_session_config", unsafe {
            (api.conversation_config_set_session_config)(config.handle, session.handle)
        })?;
        api.check("conversation_config_set_thinking_config", unsafe {
            (api.conversation_config_set_thinking_config)(config.handle, thinking.handle)
        })?;
        if !system.is_empty() {
            let system = json(system_message(system))?;
            api.check("conversation_config_set_system_message", unsafe {
                (api.conversation_config_set_system_message)(config.handle, system.as_ptr())
            })?;
        }
        if !history.is_empty() {
            let history = json(history_messages(history))?;
            api.check("conversation_config_set_messages", unsafe {
                (api.conversation_config_set_messages)(config.handle, history.as_ptr())
            })?;
        }
        let message = json(user_message(user))?;
        let (sender, chunks) = mpsc::channel::<Chunk>();
        // Made before the conversation so it is dropped after it on every
        // path, an early return included: no callback can outlive it.
        let sender = Box::new(sender);
        let conversation = Owned::create(api, "conversation_create", api.conversation_delete, |out| unsafe {
            (api.conversation_create)(self.engine.handle, config.handle, out)
        })?;
        api.check("conversation_send_message_stream", unsafe {
            (api.conversation_send_message_stream)(
                conversation.handle,
                message.as_ptr(),
                std::ptr::null(),
                std::ptr::null_mut(),
                on_chunk,
                &*sender as *const mpsc::Sender<Chunk> as *mut c_void,
            )
        })?;

        let mut text = String::new();
        loop {
            match chunks.recv() {
                Ok(Chunk::Text(piece)) => {
                    text.push_str(&chunk_text(&piece));
                    on_text(&text);
                }
                Ok(Chunk::Done) => break,
                Ok(Chunk::Error(e)) => return Err(format!("generation failed: {e}")),
                Err(_) => return Err("generation stream ended without finishing".into()),
            }
        }
        drop(conversation);
        let cut_off = self.tokens(&text)? >= max_tokens;
        Ok(Generation { text, cut_off })
    }
}

/// What a stream callback passes back to the thread waiting for the answer.
enum Chunk {
    Text(String),
    Error(String),
    Done,
}

/// Called on one of the library's threads for each piece of the answer.
/// Must not unwind and must not call back into the conversation.
unsafe extern "C" fn on_chunk(data: *mut c_void, chunk: *const c_void) {
    // Already open: the stream that calls this was started through it.
    let Ok(api) = api() else { return };
    // SAFETY: `data` is the Sender `generate` keeps alive past the stream.
    let sender = unsafe { &*(data as *const mpsc::Sender<Chunk>) };
    let read = |get: unsafe extern "C" fn(*const c_void, *mut *const c_char) -> Status| {
        let mut out: *const c_char = std::ptr::null();
        // SAFETY: a live chunk and an out-pointer; the string the library
        // writes lives as long as the chunk, and is copied before return.
        let ok = unsafe { get(chunk, &mut out) } == 0 && !out.is_null();
        ok.then(|| unsafe { CStr::from_ptr(out) }.to_string_lossy().into_owned())
    };
    if let Some(error) = read(api.stream_chunk_get_error).filter(|e| !e.is_empty()) {
        let _ = sender.send(Chunk::Error(error));
        return;
    }
    if let Some(text) = read(api.stream_chunk_get_text).filter(|t| !t.is_empty()) {
        let _ = sender.send(Chunk::Text(text));
    }
    let mut last = false;
    // SAFETY: a live chunk and an out-pointer.
    if unsafe { (api.stream_chunk_is_final)(chunk, &mut last) } == 0 && last {
        let _ = sender.send(Chunk::Done);
    }
}

/// LiteRT-LM keeps repacked copies of the weights beside the model, each
/// named after the model file, its mtime and its size: on the CPU
/// `<stamp>.xnnpack_cache`, about as big as the model again (Medium 0.8 GB,
/// Large 2.2 GB), and on the GPU `<stamp>_mldrift_weight_cache.bin` and
/// `<stamp>_mldrift_program_cache.bin` (Medium 0.8 GB together). With them a
/// load takes under a second on the Fold, without them several.
///
/// A model replaced by a newer download gets new caches and leaves the old
/// ones behind, and the GPU's stay after the GPU is switched off. So after a
/// load this removes this model's caches other than the current CPU one and,
/// loaded on the GPU, the current GPU ones. It only does so when the loaded
/// backend's cache is where that naming says, so a change in the naming
/// deletes nothing.
fn remove_unused_caches(model: &Path, on_gpu: bool) {
    let (Some(dir), Some(name), Ok(meta)) = (model.parent(), model.file_name(), std::fs::metadata(model)) else {
        return;
    };
    let name = name.to_string_lossy();
    let Some(mtime) = meta.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()) else {
        return;
    };
    let stamp = format!("{name}_{}_{}", mtime.as_secs(), meta.len());
    let cpu = format!("{stamp}.xnnpack_cache");
    let gpu = [format!("{stamp}_mldrift_weight_cache.bin"), format!("{stamp}_mldrift_program_cache.bin")];
    let (marker, kept) = if on_gpu {
        (&gpu[0], vec![&cpu, &gpu[0], &gpu[1]])
    } else {
        (&cpu, vec![&cpu])
    };
    if !dir.join(marker).is_file() {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let file = entry.file_name().to_string_lossy().into_owned();
        if is_cache(&file, &name) && !kept.contains(&&file) {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

fn is_cache(file: &str, model: &str) -> bool {
    file.starts_with(&format!("{model}_"))
        && (file.ends_with(".xnnpack_cache") || (file.contains("_mldrift_") && file.ends_with(".bin")))
}

fn system_message(text: &str) -> serde_json::Value {
    serde_json::json!({ "type": "text", "text": text })
}

fn history_messages(history: &[(String, String)]) -> serde_json::Value {
    history.iter().map(|(role, text)| serde_json::json!({ "role": role, "content": text })).collect()
}

fn user_message(text: &str) -> serde_json::Value {
    serde_json::json!({ "role": "user", "content": [{ "type": "text", "text": text }] })
}

/// The text in one streamed piece of the answer: a JSON message whose
/// content parts carry it. Text that isn't such a message is taken whole.
fn chunk_text(chunk: &str) -> String {
    let Ok(message) = serde_json::from_str::<serde_json::Value>(chunk) else {
        return chunk.to_string();
    };
    message["content"]
        .as_array()
        .map(|parts| {
            parts
                .iter()
                .filter(|p| p["type"] == "text")
                .filter_map(|p| p["text"].as_str())
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn litert_runs_litertlm_and_task_files_only() {
        assert!(handles(Path::new("/m/gemma-4-E2B-it.litertlm")));
        assert!(handles(Path::new("/m/Gemma3-1B-IT_multi-prefill-seq_q8_ekv2048.task")));
        assert!(!handles(Path::new("/m/gemma-4-E2B-it-Q4_K_M.gguf")));
        assert!(!handles(Path::new("/m/litertlm")));
        assert!(!handles(Path::new("/m/model.task.part")));
    }

    #[test]
    fn a_streamed_piece_yields_its_text_parts() {
        let chunk = r#"{"role":"assistant","content":[{"type":"text","text":" Canberra"},{"type":"text","text":"."}]}"#;
        assert_eq!(chunk_text(chunk), " Canberra.");
    }

    #[test]
    fn a_streamed_piece_skips_parts_that_are_not_text() {
        let chunk = r#"{"role":"assistant","content":[{"type":"thinking","text":"hmm"},{"type":"text","text":"Yes"}]}"#;
        assert_eq!(chunk_text(chunk), "Yes");
    }

    #[test]
    fn a_streamed_piece_that_is_not_json_is_taken_whole() {
        assert_eq!(chunk_text("plain {text"), "plain {text");
    }

    #[test]
    fn messages_are_the_json_litert_expects() {
        assert_eq!(system_message("Be \"brief\".").to_string(), r#"{"text":"Be \"brief\".","type":"text"}"#);
        assert_eq!(
            user_message("Hi\nthere").to_string(),
            r#"{"content":[{"text":"Hi\nthere","type":"text"}],"role":"user"}"#
        );
        let history = [("user".to_string(), "Capital of France?".to_string()), ("assistant".to_string(), "Paris.".to_string())];
        assert_eq!(
            history_messages(&history).to_string(),
            r#"[{"content":"Capital of France?","role":"user"},{"content":"Paris.","role":"assistant"}]"#
        );
    }

    #[test]
    fn a_cache_is_one_of_this_models_cpu_or_gpu_caches() {
        let model = "gemma-4-E2B-it.litertlm";
        assert!(is_cache("gemma-4-E2B-it.litertlm_1795073019_2588147712.xnnpack_cache", model));
        assert!(is_cache("gemma-4-E2B-it.litertlm_1795083487_2588147712_mldrift_weight_cache.bin", model));
        assert!(is_cache("gemma-4-E2B-it.litertlm_1795083487_2588147712_mldrift_program_cache.bin", model));
        assert!(!is_cache(model, model));
        assert!(!is_cache("gemma-4-E4B-it.litertlm_1795075025_3659530240.xnnpack_cache", model));
        assert!(!is_cache("gemma-4-E2B-it.litertlm_notes.txt", model));
        assert!(!is_cache("gemma-4-E2B-it.litertlm_1795073019_2588147712.bin", model));
    }

    /// A model `m.task` in a fresh directory, with a cache file of each kind
    /// for its current stamp, for an older stamp, and for another model.
    struct CacheDir {
        dir: std::path::PathBuf,
        model: std::path::PathBuf,
        files: Vec<(&'static str, std::path::PathBuf)>,
    }

    impl CacheDir {
        fn new(test: &str, with: &[&'static str]) -> Self {
            let dir = std::env::temp_dir().join(format!("ari-litert-{test}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            let model = dir.join("m.task");
            std::fs::write(&model, b"weights").unwrap();
            let mtime = std::fs::metadata(&model).unwrap().modified().unwrap();
            let mtime = mtime.duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
            let all = [
                ("cpu", format!("m.task_{mtime}_7.xnnpack_cache")),
                ("gpu weights", format!("m.task_{mtime}_7_mldrift_weight_cache.bin")),
                ("gpu program", format!("m.task_{mtime}_7_mldrift_program_cache.bin")),
                ("old cpu", "m.task_1_5.xnnpack_cache".to_string()),
                ("old gpu weights", "m.task_1_5_mldrift_weight_cache.bin".to_string()),
                ("old gpu program", "m.task_1_5_mldrift_program_cache.bin".to_string()),
                ("other model", "n.task_1_5.xnnpack_cache".to_string()),
                ("other file", "m.task_notes.txt".to_string()),
            ];
            let files: Vec<_> = all
                .into_iter()
                .filter(|(label, _)| with.contains(label))
                .map(|(label, file)| (label, dir.join(file)))
                .collect();
            for (_, f) in &files {
                std::fs::write(f, b"cache").unwrap();
            }
            CacheDir { dir, model, files }
        }

        /// The labels of the files still there, after checking the model is.
        fn left(&self) -> Vec<&'static str> {
            assert!(self.model.exists());
            self.files.iter().filter(|(_, f)| f.exists()).map(|(label, _)| *label).collect()
        }
    }

    impl Drop for CacheDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    const EVERY_FILE: [&str; 8] = [
        "cpu",
        "gpu weights",
        "gpu program",
        "old cpu",
        "old gpu weights",
        "old gpu program",
        "other model",
        "other file",
    ];

    #[test]
    fn on_the_cpu_only_the_current_cpu_cache_stays() {
        let dir = CacheDir::new("cpu", &EVERY_FILE);
        remove_unused_caches(&dir.model, false);
        assert_eq!(dir.left(), ["cpu", "other model", "other file"]);
    }

    #[test]
    fn on_the_gpu_the_current_cpu_and_gpu_caches_stay() {
        let dir = CacheDir::new("gpu", &EVERY_FILE);
        remove_unused_caches(&dir.model, true);
        assert_eq!(dir.left(), ["cpu", "gpu weights", "gpu program", "other model", "other file"]);
    }

    #[test]
    fn without_the_current_cpu_cache_nothing_is_removed() {
        let with = ["gpu weights", "gpu program", "old cpu", "old gpu weights"];
        let dir = CacheDir::new("nocpu", &with);
        remove_unused_caches(&dir.model, false);
        assert_eq!(dir.left(), with);
    }

    #[test]
    fn without_the_current_gpu_cache_nothing_is_removed() {
        let with = ["cpu", "old cpu", "old gpu weights", "old gpu program"];
        let dir = CacheDir::new("nogpu", &with);
        remove_unused_caches(&dir.model, true);
        assert_eq!(dir.left(), with);
    }
}
