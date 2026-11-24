//! Ari as a frontend runs it: the engine with the built-in and community
//! skills, the frontend's platform providers, and the voice session's way in
//! ([`TurnEngine`]). Moved out of `ari-ffi`, which now wraps it for Android;
//! a frontend written in Rust links this directly.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};

use ari_core::{Response, SettingsQueryResult, Skill};
use ari_engine::{Engine, EnvelopeSink, FALLBACK_RESPONSE, fallback_response_for};
use ari_skill_loader::assistant::{ConfigStore, MemoryConfigStore};
use ari_skill_loader::{
    AuthorizeProvider, CalendarProvider, ContactsProvider, EnglishLocaleProvider, HostCapabilities,
    LiveConversationsProvider, LoadOptions, LocalClock, LocaleProvider, LocationProvider, LogLevel, LogSink,
    MediaServicesProvider, ModelCatalog, NullAuthorizeProvider, NullCalendarProvider, NullContactsProvider,
    NullLiveConversationsProvider, NullLocationProvider, NullLogSink, NullMediaServicesProvider, NullSettingWriter,
    NullTasksProvider, SettingWriter, TasksProvider, UtcLocalClock, load_skill_directory_with,
};
use ari_skills::{CalculatorSkill, CurrentTimeSkill, DateSkill, GreetingSkill, OpenSkill, SearchSkill};
use ari_voice::session::{AnswerSink, EngineReply, ReplyFlags, TurnEngine};

/// The 6 built-in Rust skills, freshly constructed. Kept in one place so the
/// initial engine build and every `reload_community_skills` register exactly
/// the same set — a reload that forgot one would silently drop that skill.
fn builtin_skills() -> Vec<Box<dyn Skill>> {
    vec![
        Box::new(CurrentTimeSkill::new()),
        Box::new(DateSkill::new()),
        Box::new(CalculatorSkill::new()),
        Box::new(GreetingSkill::new()),
        Box::new(OpenSkill::new()),
        Box::new(SearchSkill::new()),
    ]
}

/// Build an [`Engine`] with the full set of built-in skills registered — the
/// same catalogue used at runtime. Exposed so tooling can exercise the real
/// skill catalogue without duplicating the skill list.
pub fn build_engine_with_builtins() -> Engine {
    let mut engine = Engine::new();
    for skill in builtin_skills() {
        engine.register_skill(skill);
    }
    engine
}

/// Load every skill under `root` and register it alongside the built-ins,
/// returning the number registered.
///
/// Shared by the `keyword-hit` oracle and its poaching test, which both ask
/// "does the keyword scorer already claim this utterance?" and so must ask it
/// against the same skill catalogue. Two copies of this logic could drift, and
/// a drifted gate reports a silently inflated score.
///
/// Grants [`HostCapabilities::all`] deliberately. Both callers only ever read
/// `matching.patterns` and `specificity` through `Skill::score` /
/// `Skill::specificity` — they never execute a skill, so the runtime caveat on
/// `all()` (unresolvable WASM imports) cannot bite here. Loading with the
/// default `pure_frontend()` set instead would reject every skill declaring
/// `http`, `location`, `storage_kv`, `authorize` or `media_services` —
/// weather, home-assistant, music, counter and github-zen among them — and a
/// rejected skill takes its patterns with it, which is precisely the silent
/// under-count this exists to fix.
///
/// A per-skill load failure is fatal for the same reason: a missing skill
/// means missing patterns, and missing patterns mean the caller quietly treats
/// keyword-hits as keyword-misses. Better to stop than to emit verdicts that
/// are wrong in the direction nobody would notice.
pub fn register_community_skills(engine: &mut Engine, root: &Path) -> Result<usize, String> {
    let options = LoadOptions {
        host_capabilities: HostCapabilities::all(),
        ..LoadOptions::default()
    };
    let report = load_skill_directory_with(root, &options)
        .map_err(|e| format!("--skills-dir {}: {e}", root.display()))?;

    if !report.failures.is_empty() {
        let details: Vec<String> = report.failures.iter().map(|f| f.to_string()).collect();
        return Err(format!(
            "{} skill(s) under {} failed to load, so their patterns are missing and \
             every verdict below them would be silently wrong:\n  {}",
            report.failures.len(),
            root.display(),
            details.join("\n  ")
        ));
    }

    let loaded = report.skills.len();
    for skill in report.skills {
        engine.register_skill(skill);
    }
    Ok(loaded)
}

/// The frontend's platform providers. [`Default`] gives the Null ones, which
/// have nothing to offer, a UTC clock and English: what tests and the CLI use.
pub struct Providers {
    /// Where WASM skills' `ari::log(...)` calls go.
    pub log_sink: Arc<dyn LogSink>,
    pub tasks: Arc<dyn TasksProvider>,
    pub calendar: Arc<dyn CalendarProvider>,
    pub location: Arc<dyn LocationProvider>,
    pub media_services: Arc<dyn MediaServicesProvider>,
    pub contacts: Arc<dyn ContactsProvider>,
    pub live_conversations: Arc<dyn LiveConversationsProvider>,
    pub local_clock: Arc<dyn LocalClock>,
    /// The user's language, read whenever the engine dispatches on it.
    pub locale: Arc<dyn LocaleProvider>,
    /// Backs `ari::setting_get` in WASM skills, and resolves masked secrets
    /// for settings-time calls.
    pub config_store: Arc<dyn ConfigStore>,
    /// Backs `ari::setting_set` in WASM skills.
    pub setting_writer: Arc<dyn SettingWriter>,
    /// Backs `ari::authorize` in WASM skills.
    pub authorize: Arc<dyn AuthorizeProvider>,
    /// Envelopes produced after a turn has returned, from an assistant
    /// round-trip. `None` keeps the engine strictly synchronous.
    pub envelope_sink: Option<Arc<dyn EnvelopeSink>>,
}

impl Default for Providers {
    fn default() -> Self {
        Self {
            log_sink: Arc::new(NullLogSink),
            tasks: Arc::new(NullTasksProvider),
            calendar: Arc::new(NullCalendarProvider),
            location: Arc::new(NullLocationProvider),
            media_services: Arc::new(NullMediaServicesProvider),
            contacts: Arc::new(NullContactsProvider),
            live_conversations: Arc::new(NullLiveConversationsProvider),
            local_clock: Arc::new(UtcLocalClock),
            locale: Arc::new(EnglishLocaleProvider),
            config_store: Arc::new(MemoryConfigStore::new()),
            setting_writer: Arc::new(NullSettingWriter),
            authorize: Arc::new(NullAuthorizeProvider),
            envelope_sink: None,
        }
    }
}

/// What the engine made of one input.
#[derive(Debug, Clone, PartialEq)]
pub enum Reply {
    /// `markdown` is an assistant answer as written, for a frontend that
    /// renders markdown to show; `body` is the same answer as plain text, for
    /// speaking. `None` for anything else, and for an answer that formats
    /// nothing.
    Text { body: String, markdown: Option<String>, flags: ReplyFlags },
    /// `skill_id` is the manifest id of the emitting skill, which resolves
    /// `asset:<path>` references to the skill's bundle. Empty when the engine
    /// couldn't attribute the response to a skill (phrase-tier actions,
    /// fallbacks).
    Action { json: String, skill_id: String, flags: ReplyFlags },
    Binary { mime: String, data: Vec<u8> },
    /// No skill matched. `body` is the apology to say if the frontend's
    /// retries fail too.
    NotUnderstood { body: String },
}

/// The engine and the providers its skills run against.
pub struct Runtime {
    // Behind a Mutex because `reload_community_skills` swaps the skill set
    // after construction.
    engine: Mutex<Engine>,
    providers: Providers,
}

impl Runtime {
    pub fn new(providers: Providers) -> Self {
        let mut engine = build_engine_with_builtins();
        engine.set_log_sink(Some(providers.log_sink.clone()));
        engine.set_config_store(Some(providers.config_store.clone()));
        if let Some(sink) = &providers.envelope_sink {
            engine.set_envelope_sink(Some(sink.clone()));
        }
        Self { engine: Mutex::new(engine), providers }
    }

    /// The engine, for the calls that need nothing else.
    pub fn engine(&self) -> MutexGuard<'_, Engine> {
        self.engine.lock().expect("engine mutex poisoned")
    }

    /// The user's currently-active language, as seen by the engine.
    pub fn current_locale(&self) -> String {
        self.providers.locale.current_locale()
    }

    /// Answer `input`, with `alternative`, a second transcript of the turn the
    /// engine may act on instead, and an on-device answer going to `answer`
    /// while it is written. Also says whether the engine used the alternative.
    pub fn reply(
        &self,
        input: &str,
        alternative: Option<&str>,
        answer: Option<Arc<dyn ari_engine::AnswerSink>>,
    ) -> (Reply, bool) {
        // Refresh the engine's locale on every call so per-locale
        // skill scorers and responses pick up live changes from the
        // frontend's settings store. The locale provider's read is a
        // synchronous AtomicReference lookup on Android (see
        // AriFfiLocaleProvider), so this is essentially free.
        let locale = self.providers.locale.current_locale();
        let mut engine = self.engine();
        engine.set_locale(locale.clone());
        #[cfg(feature = "llm")]
        engine.set_answer_sink(answer);
        // Without the on-device assistant there are no answers to stream.
        #[cfg(not(feature = "llm"))]
        drop(answer);
        engine.set_alternative_transcript(alternative.map(str::to_string));
        let (response, skill_id) = engine.process_input_with_skill(input);
        let used_alternative = engine.take_used_alternative();
        #[cfg(feature = "llm")]
        engine.set_answer_sink(None);
        let flags = ReplyFlags {
            rearm: engine.has_pending_turn(),
            enter_conversation: engine.take_enter_signal(),
            exit_conversation: engine.take_exit_signal(),
            facts_changed: engine.take_facts_changed_signal(),
            clear_conversation: engine.take_clear_signal(),
        };
        let markdown = engine.take_answer_markdown();
        let reply = match response {
            Response::Text(s) => {
                // The engine's fallback text is locale-specific
                // (Phase 5 step 2). Compare against the locale-
                // appropriate version so the NotUnderstood retry path
                // fires for Italian / French / etc. users too — not
                // just English.
                let is_fallback = s == fallback_response_for(&locale) || s == FALLBACK_RESPONSE;
                if is_fallback {
                    // A fallback never re-arms.
                    Reply::NotUnderstood { body: s }
                } else {
                    Reply::Text { body: s, markdown, flags }
                }
            }
            Response::Action(v) => Reply::Action {
                json: serde_json::to_string(&v).unwrap_or_default(),
                skill_id: skill_id.unwrap_or_default(),
                flags,
            },
            Response::Binary { mime, data } => Reply::Binary { mime, data },
        };
        (reply, used_alternative)
    }

    /// Load the cached tier→model catalog the skill store wrote, so cloud
    /// assistant skills resolve their `fast`/`balanced`/`smartest` setting to
    /// a current model ID. Call this at startup, and again after each refresh.
    ///
    /// Returns whether a catalog was installed. `false` is a normal state, not
    /// a failure — a first run has nothing cached yet, and skills fall back to
    /// the per-tier pins in their own manifests. The reason is logged either
    /// way, since a catalog that stops loading is otherwise invisible: the
    /// skills keep working, just on ageing pinned models.
    pub fn load_model_catalog(&self, path: &str) -> bool {
        let outcome = std::fs::read(path)
            .map_err(|e| e.to_string())
            .and_then(|bytes| ModelCatalog::from_json_bytes(&bytes).map_err(|e| e.to_string()));

        match outcome {
            Ok(catalog) => {
                self.engine().set_model_catalog(Some(Arc::new(catalog)));
                self.providers.log_sink.log("ari-ffi", LogLevel::Info, &format!("model catalog loaded from {path}"));
                true
            }
            Err(message) => {
                self.providers.log_sink.log(
                    "ari-ffi",
                    LogLevel::Warn,
                    &format!(
                        "no model catalog from {path} ({message}) — \
                         cloud assistants will use their pinned models"
                    ),
                );
                false
            }
        }
    }

    /// Settings-time skill invocation: run `skill_id`'s `settings_query` for
    /// `field`, passing the current `values` (the field's `depends_on`
    /// siblings), to populate a `dynamic_select` field's options or to
    /// validate a field whose value depends on a server round-trip.
    pub fn query_skill_setting(&self, skill_id: &str, field: &str, values: HashMap<String, String>) -> SettingsQueryResult {
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
            if let Some(real) = self.providers.config_store.get(skill_id, key.as_str()) {
                *slot = real;
            }
        }
        let values_json = serde_json::to_string(&merged).unwrap_or_else(|_| "{}".to_string());
        // Same reasoning as `settings_action` below — a dynamic_select does
        // HTTP, so it can hang for a while too, just not for minutes.
        let skill = self.engine().skill_by_id(skill_id);
        match skill {
            Some(skill) => skill.settings_query(field, &values_json),
            None => skill_not_loaded(skill_id),
        }
    }

    /// Effectful settings-time skill invocation: run `skill_id`'s `settings_action`
    /// for `action`, passing the current `values` (sibling field values the
    /// skill reads during the action — e.g. `base_url`/`token` for HA sign-in).
    /// Secret-masked values are resolved from the config store before the call,
    /// identical to the resolution step in [`Runtime::query_skill_setting`].
    pub fn settings_action(&self, skill_id: &str, action: &str, values: HashMap<String, String>) -> SettingsQueryResult {
        let mut merged = values;
        for (key, slot) in merged.iter_mut() {
            if let Some(real) = self.providers.config_store.get(skill_id, key.as_str()) {
                *slot = real;
            }
        }
        let values_json = serde_json::to_string(&merged).unwrap_or_else(|_| "{}".to_string());
        // Resolve the skill under the lock, then let go of it before calling.
        //
        // This one waits on a human: the Home Assistant sign-in blocks on the
        // OAuth callback for up to five minutes. Every entry point here shares
        // this mutex, `reply` among them, so holding it across the call
        // left Ari deaf to everything for the duration — and reopening the app
        // did not help, because the process and the held lock survived.
        //
        // Nothing about the call needs exclusivity. `Skill::settings_action`
        // takes `&self`, each WASM invocation builds a fresh store, and holding
        // the Arc keeps the skill alive even if the set is replaced mid-call.
        let skill = self.engine().skill_by_id(skill_id);
        match skill {
            Some(skill) => skill.settings_action(action, &values_json),
            None => skill_not_loaded(skill_id),
        }
    }

    /// Rebuild the engine's skill set: the 6 built-in Rust skills plus every
    /// community skill on disk under `skill_store_dir`, loaded with `options`
    /// (the frontend's capability grants, HTTP and storage) and this runtime's
    /// providers. Returns the number of community skills registered.
    ///
    /// Skills that fail to load are left out: a broken skill should not take
    /// the conversation engine down with it.
    pub fn reload_community_skills(&self, skill_store_dir: &Path, options: LoadOptions) -> u32 {
        let p = &self.providers;
        let options = LoadOptions {
            log_sink: p.log_sink.clone(),
            tasks_provider: p.tasks.clone(),
            calendar_provider: p.calendar.clone(),
            location_provider: p.location.clone(),
            media_services_provider: p.media_services.clone(),
            contacts_provider: p.contacts.clone(),
            live_conversations_provider: p.live_conversations.clone(),
            local_clock: p.local_clock.clone(),
            config_store: p.config_store.clone(),
            locale_provider: p.locale.clone(),
            setting_writer: p.setting_writer.clone(),
            authorize_provider: p.authorize.clone(),
            ..options
        };
        // Load the community skills from disk BEFORE touching the live engine,
        // so a whole-directory read failure leaves the current set intact
        // rather than half-swapped.
        let community = match load_skill_directory_with(skill_store_dir, &options) {
            Ok(report) => report.skills,
            Err(_) => Vec::new(),
        };
        let loaded = community.len() as u32;

        // Swap the skill set IN PLACE. We deliberately do NOT build a fresh
        // Engine: that discarded every other field — remembered facts (and the
        // first later "remember" then clobbered the on-disk list), the
        // conversation-memory toggle, the on-device LLM, the
        // pending turn, let's-talk state and the conversation buffer — none of
        // which the frontend re-applies after a reload. `replace_skills`
        // touches only the skills; the engine's sinks/providers were installed
        // once at construction and persist. Built-ins lead the community set.
        let mut skills = builtin_skills();
        skills.extend(community);
        self.engine().replace_skills(skills);
        loaded
    }

    /// Set the model file for the on-device assistant: a .litertlm or .task
    /// file, which LiteRT-LM runs on the GPU when `use_gpu` is set and the GPU
    /// loads it, else on the CPU. It loads when the first unmatched query
    /// arrives, and unloads after 60 seconds of idle to free RAM. False when
    /// there is no file at `model_path`.
    #[cfg(feature = "llm")]
    pub fn load_llm_model(&self, model_path: &Path, use_gpu: bool) -> bool {
        if !model_path.is_file() {
            return false;
        }
        let accelerator = if use_gpu { ari_llm::Accelerator::Gpu } else { ari_llm::Accelerator::Cpu };
        let lazy = ari_llm::LazyLlmFallback::new(model_path, accelerator);
        self.engine().set_llm(Arc::new(lazy));
        true
    }
}

/// The error result both settings entry points return for an id that isn't
/// loaded.
fn skill_not_loaded(skill_id: &str) -> SettingsQueryResult {
    SettingsQueryResult {
        ok: false,
        error: Some(format!("skill not loaded: {skill_id}")),
        options: Vec::new(),
        message: None,
        refresh: false,
    }
}

impl From<Reply> for EngineReply {
    fn from(reply: Reply) -> Self {
        match reply {
            Reply::Text { body, markdown, flags } => EngineReply::Text { body, markdown, flags },
            Reply::Action { json, skill_id, flags } => EngineReply::Action { json, skill_id, flags },
            Reply::Binary { mime, data } => EngineReply::Binary { mime, len: data.len() },
            Reply::NotUnderstood { body } => EngineReply::NotUnderstood { body },
        }
    }
}

/// An answer the engine writes, handed on to the session.
struct VoiceAnswerSink(Arc<dyn AnswerSink>);

impl ari_engine::AnswerSink for VoiceAnswerSink {
    fn text(&self, so_far: &str, markdown: Option<&str>) {
        self.0.text(so_far, markdown);
    }
    fn sentence(&self, text: &str) {
        self.0.sentence(text);
    }
}

impl TurnEngine for Runtime {
    fn respond(&self, input: &str, alternative: Option<&str>, answer: Option<Arc<dyn AnswerSink>>) -> (EngineReply, bool) {
        let answer = answer.map(|a| Arc::new(VoiceAnswerSink(a)) as Arc<dyn ari_engine::AnswerSink>);
        let (reply, used_alternative) = self.reply(input, alternative, answer);
        (reply.into(), used_alternative)
    }

    fn cancel_pending_reply(&self) {
        self.engine().clear_pending_turn();
    }

    fn set_conversation_active(&self, active: bool) {
        self.engine().set_conversation_active(active);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runtime() -> Runtime {
        Runtime::new(Providers::default())
    }

    fn text(reply: (Reply, bool)) -> String {
        match reply.0 {
            Reply::Text { body, .. } => body,
            other => panic!("expected a text reply, got {other:?}"),
        }
    }

    /// A skill whose settings action parks inside the call until released, so
    /// a test can ask what the rest of the engine can do meanwhile.
    struct BlockingSkill {
        entered: Arc<std::sync::Barrier>,
        release: Arc<std::sync::Barrier>,
    }

    impl Skill for BlockingSkill {
        fn id(&self) -> &str {
            "test.blocking"
        }
        fn specificity(&self) -> ari_core::Specificity {
            ari_core::Specificity::Low
        }
        fn score(&self, _input: &str, _ctx: &ari_core::SkillContext) -> f32 {
            0.0
        }
        fn execute(&self, _input: &str, _ctx: &ari_core::SkillContext) -> Response {
            Response::Text(String::new())
        }
        fn settings_action(&self, _action: &str, _values: &str) -> SettingsQueryResult {
            self.entered.wait();
            self.release.wait();
            SettingsQueryResult { ok: true, error: None, options: Vec::new(), message: None, refresh: false }
        }
    }

    /// A settings action can wait on a person — the Home Assistant sign-in
    /// blocks on an OAuth callback for up to five minutes. It used to do that
    /// holding the engine mutex, which every entry point here takes, so Ari
    /// went deaf to everything until it gave up.
    ///
    /// The wait is on another thread with a deadline rather than inline: a
    /// regression should fail this test, not hang it.
    #[test]
    fn a_blocked_settings_action_does_not_stop_the_engine_answering() {
        use std::sync::{Barrier, mpsc};
        use std::time::Duration;

        let runtime = Arc::new(runtime());
        let entered = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        runtime.engine().replace_skills(vec![Box::new(BlockingSkill {
            entered: Arc::clone(&entered),
            release: Arc::clone(&release),
        })]);

        let acting = {
            let runtime = Arc::clone(&runtime);
            std::thread::spawn(move || runtime.settings_action("test.blocking", "sign_in", HashMap::new()))
        };
        entered.wait();

        let (tx, rx) = mpsc::channel();
        {
            let runtime = Arc::clone(&runtime);
            std::thread::spawn(move || {
                let _ = tx.send(runtime.reply("hello", None, None));
            });
        }
        let answered = rx.recv_timeout(Duration::from_secs(5));

        release.wait();
        let action = acting.join().expect("settings action thread panicked");

        assert!(answered.is_ok(), "reply blocked behind a settings action that was waiting on the user");
        assert!(action.ok, "the settings action itself should still succeed");
    }

    #[test]
    fn a_settings_action_for_an_unloaded_skill_still_reports_it() {
        let r = runtime().settings_action("nobody.here", "sign_in", HashMap::new());
        assert!(!r.ok);
        assert_eq!(r.error.as_deref(), Some("skill not loaded: nobody.here"));
    }

    #[test]
    fn reload_community_skills_preserves_facts_and_keeps_builtins() {
        // An empty store dir → zero community skills. The reload must still
        // re-register the built-ins AND leave runtime state (here: remembered
        // facts) untouched. Rebuilding a fresh Engine — the old approach —
        // discarded the facts, which is the P0 this guards against.
        let base = std::env::temp_dir().join(format!("ari_runtime_reload_test_{}", std::process::id()));
        let store = base.join("skills");
        std::fs::remove_dir_all(&base).ok();
        std::fs::create_dir_all(&store).expect("store dir");

        let runtime = runtime();
        runtime.engine().set_remembered_facts(vec!["my name is Keith".to_string()]);

        let community = runtime.reload_community_skills(&store, LoadOptions::default());
        assert_eq!(community, 0, "empty store dir yields no community skills");

        assert_eq!(
            runtime.engine().remembered_facts(),
            vec!["my name is Keith".to_string()],
            "reload_community_skills discarded remembered facts",
        );

        // Built-ins must survive the reload: the calculator still answers.
        assert_eq!(text(runtime.reply("2 + 2", None, None)), "4", "calculator built-in lost after reload");

        std::fs::remove_dir_all(&base).ok();
    }

    #[test]
    fn a_greeting_is_answered() {
        let body = text(runtime().reply("hello", None, None));
        assert!(!body.is_empty());
        assert_ne!(body, FALLBACK_RESPONSE);
    }

    #[test]
    fn the_time_is_answered() {
        let body = text(runtime().reply("what time is it", None, None));
        assert!(body.starts_with("It's "), "response was: {body}");
    }

    #[test]
    fn a_calculation_is_answered() {
        assert_eq!(text(runtime().reply("calculate 5 + 3", None, None)), "8");
    }

    #[test]
    fn open_is_an_action_from_the_open_skill() {
        match runtime().reply("open spotify", None, None).0 {
            Reply::Action { json, skill_id, .. } => {
                let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
                assert_eq!(parsed["v"], 1);
                assert_eq!(parsed["launch_app"], "spotify");
                assert_eq!(skill_id, "open");
            }
            other => panic!("expected an action for open, got {other:?}"),
        }
    }

    #[test]
    fn gibberish_is_not_understood() {
        assert_eq!(
            runtime().reply("asdfghjkl", None, None),
            (Reply::NotUnderstood { body: "Sorry, I didn't understand that.".into() }, false)
        );
    }

    #[test]
    fn the_session_gets_the_reply_through_turn_engine() {
        let (reply, used_alternative) = TurnEngine::respond(&runtime(), "calculate 5 + 3", None, None);
        assert_eq!(reply, EngineReply::Text { body: "8".into(), markdown: None, flags: ReplyFlags::default() });
        assert!(!used_alternative);
    }
}
