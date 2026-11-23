# Audio In, Step 4: the Voice Session

Task 4.0 of `2026-11-22-audio-in-engine.md`, written from the code as it stands after step 3
(engine `dec09ea`, Android `3a08567`). Branch `audio-in-step-4` in both repos.

**Goal:** `voice/VoiceSession.kt` (1,556 lines), the capture stores and the false-wake decision
move into `ari-voice`. Kotlin keeps what only a phone can do: the overlay, the cue, speech
output, running action envelopes, the conversation log, alerts and cards, settings, and the
system speech service. Rules 1 to 9 of the parent plan apply unchanged.

## What the session is today

One coroutine per session on `Dispatchers.Main`, collecting `SpeechRecognizer.state`. Its
suspension points are where the Rust session waits:

| Kotlin | Waits for | Rust |
|---|---|---|
| `sttModelLoader.load` (cold start) | model load, up to its own timeout | host call on a worker |
| `speechOutput.speakAndAwait`, `Reply.speakAndAwait` | the utterance's done callback | `speak` event, `speech_finished` command |
| `checkNameEarly` | a cloud request | worker |
| `engineHolder.processSpokenInput`, `processInput` | the engine | engine worker |
| `transcribeOffline` | a re-decode | worker |
| `actionHandler.handle` | the envelope's side effects | host call on a worker |
| `cardActionDispatcher.dispatch` | a card button's effect | host call on a worker |
| `delay(...)` | 600 ms flash, 2.5 s error, 3 s dwell, 3 s name wait | deadline |
| silence watcher, a 1 s poll (500 ms for dictation) | idle time | the same poll, as deadlines |

Dismissal cancels the coroutine; the next suspension point throws. The Rust equivalent is a
session epoch: every worker result and timer carries the epoch it was started in, and one from
an older epoch is dropped. `handleFinalText`'s `finally` (the utterance capture) becomes "the
turn record is written when the turn ends, however it ends".

## Design

### Threads

- **Session thread** (`AriVoiceSession`): owns all state, reads one message queue with
  `recv_timeout` to the earliest deadline. Never blocks on anything else.
- **Engine worker**: one thread, FIFO, for every engine call (turns, `cancel_pending_reply`,
  `set_conversation_active`). FIFO keeps their order, which the Kotlin got from running them in
  sequence on Main.
- **Short workers**: a thread per blocking host call (`load_speech_model`, `handle_action`,
  `intercept`), the early name check and the offline re-decode. Each posts its result back.

Host callbacks are made from the session thread (or a worker, for the blocking ones) and must
not call back into the pipeline before returning. `start_turn` and `start_dictation` are the
exception the Kotlin needs: they post their message and wait for the session thread to have
handled it, so the first state is out before they return (the lock-screen race `start()`
documents).

### The core is synchronous

`session::Core` is a plain state machine: `handle(msg, now) -> ()`, with its side effects behind
a `SessionIo` trait (host, speech-to-text, engine, clock, spawn, schedule). The thread above is a
thin runner. The scenario harness (4.1) drives `Core` directly with a fake `SessionIo` and a
manual clock, and runs spawned work when the scenario says it finishes.

### Engine access

`ari-voice` does not depend on `ari-engine`. It defines `TurnEngine` (respond, with an answer
sink and an alternative transcript; cancel the pending reply; set conversation active) and an
`EngineReply` (text, action, binary, not understood, with the five flags). `ari-ffi` implements
it for `AriEngine` over the existing `respond_with`. Android attaches the engine with
`FfiVoicePipeline.attach_engine` once `EngineHolder` has built it; a turn that reaches the engine
before then waits, as `engineHolder.engine()` does. `EngineHolder`'s after-turn work (the lapse
watcher, the last cloud assistant) runs on the `engine_turn_finished` host callback, once per
engine call, as now.

### Speech-to-text inside the session

The session starts and stops listens itself. Its events no longer go to the host: `Stt` routes
them to the session, tagged with the listen's id, and the session drops any from a listen that
is not its current one. That replaces both the Kotlin `StateFlow` reset dance and
`stop_listen_owned`'s window between check and send.

The system speech service stays in Kotlin behind a **host transcriber hook**, as the parent plan
first intended (3.4): when the host says it transcribes, the session calls `host_listen(params)`
and `host_stop_listen()`, and the host reports through `FfiVoicePipeline.host_stt_event`.
`SystemSpeechSession` and `startServiceListening` stay; `SpeechRecognizer.state` goes.

### Strings

The engine never holds words. `phrase(Phrase) -> String` asks the host for "one moment" and "say
that again" (it picks at random from its own translations, as `WaitPhrases.kt` does), so the
overlay and the voice say the same one. Errors are typed: `NoSpeechModel`, `Stt(failure)`,
`NoSpeech`, `Failed(String)`.

### Cards stay in Kotlin

The session never sees an attachment. `handle_action` and `intercept` return whether there are
cards; the adapter keeps the attachments from the call that made them and pairs them with the
next `Responding` state. `Idle { keep_cards }` tells it to linger them. `CARD_DWELL_MS`,
`holdCards`, `dropCards` and `onCardAction` stay in Kotlin (4.4 of the parent plan says so; its
number list puts `CARD_DWELL_MS` with the session, which was wrong).

## Host interface (frozen)

Commands, on `FfiVoicePipeline`:

| Command | Replaces |
|---|---|
| `attach_engine(engine)` | `VoiceSession`'s `EngineHolder` |
| `configure_session(SessionConfig)` | settings reads at call time: barge-in toggle and echo canceller, the four capture switches, the captures directory, whether the host transcribes |
| `start_turn(verify_wake, quiet) -> bool` | `start()`; false when a session is already running |
| `start_dictation() -> bool`, `stop_dictation()` | the same |
| `dismiss()` | `dismiss()` from the overlay |
| `speech_finished(id)` | `speakAndAwait` returning |
| `host_stt_event(event)` | the system speech service's `_state` writes |
| `is_session_active() -> bool` | `isActive` |
| `submit_text(text, spoken)` | `ConversationViewModel.onTextSubmitted`'s engine part (4.5) |

Callbacks, on a new `FfiSessionHost` attached with `attach_session_host` (kept apart from
`FfiVoiceHost`, which the pipeline needs from its first moment):

| Callback | Kotlin side |
|---|---|
| `state(SessionState)` | the overlay's `StateFlow` |
| `reveal()` | `VoiceOverlayLauncher.launch` |
| `play_cue(volume)` | `playReadyCue` |
| `speak(id, text, flush)`, `stop_speaking()` | `SpeechOutput` (its safety timeout stays there) |
| `phrase(Phrase) -> String` | `WaitPhrases.kt` |
| `capture_mode(mode)` | `WakeWordService.applyCaptureMode` |
| `load_speech_model() -> bool` | `SttModelLoader` (cold start) |
| `host_listen(params)`, `host_stop_listen()` | the system speech service |
| `intercept(text, cards) -> Intercepted` | `AlertVoiceIntercept`, `CardActionVoiceIntercept`, `CardActionDispatcher`, and their log lines |
| `handle_action(json, skill_id) -> ActionOutcome` | `ActionHandler.handle` |
| `turn_logged(LoggedTurn)` | `ConversationLogRepository` |
| `answer_text(so_far, markdown)` | typed turns' streaming bubble (4.5) |
| `facts_changed()` | `setRememberedFacts(engine.rememberedFacts())` |
| `false_wake()` | `FalseWakeMonitor.onFalseWake` (notification and storage stay) |
| `dictated(text)` | `dictatedText` |
| `engine_turn_finished()` | `EngineHolder`'s lapse watcher and cloud assistant |

Interim exports deleted: `stt_event`, `start_listen`, `stop_listen`, `transcribe_again`,
`check_name_early`, `can_check_name_early`, `peek_recent` (the session takes its own pre-roll),
and the wake log's `outcome` (kept only for `NOT_SHOWN`, which the service writes before any
session), `listen_ended` and `early_name_check`.

## Tasks

- **4.1 Scenario harness** (`ari-voice/tests/session.rs`): fake `SessionIo`, scripted listen
  events, a manual clock. Scenarios written from the Kotlin before the port: accepted wake;
  rejected wake; no-speech wake; quiet wake shown on a partial; quiet wake turned away at the
  name wait; quiet wake shown by the early check; bare "Hey Ari"; cold start, loaded and failed;
  follow-up reply; entering and leaving "let's talk"; a barge-in; a stale final after dismiss;
  both silence timeouts; layer 2 and layer 3 retries; the correction flash; dismiss mid-turn;
  dictation; an STT error; card and alert intercepts.
- **4.2 Port the session** (`ari-voice/src/session/`): `Core`, the runner, the engine worker.
  `VoiceSessionTest` (195) and `BargeInTest` (76) become unit tests of the pure functions, which
  move as they are.
- **4.3 Capture stores and the false-wake count.** `AudioClipStore`'s write side (`wavBytes`,
  `clipStem`, eviction), `WakeCaptureStore`'s routing and `UtteranceCaptureStore`'s sidecar move
  with `AudioClipStoreTest`, `WakeCaptureStoreTest` and `UtteranceCaptureStoreTest`. Kotlin keeps
  the read side (stats, clear, files, share) and the directory names its settings, bug reports
  and contributions use; `WakeSampleStore` writes through the engine's `save_clip`.
  `FalseWakeMonitor.prune` and `burstDue` move with their test; the monitor keeps storage and
  the notification.
- **4.4 Android adapter.** `VoiceSession.kt` becomes the host: state and lingering flows for the
  overlay, cue, speech, actions, intercepts, log. `SpeechRecognizer.kt` keeps model control and
  the system service. Delete every interim export; regenerate the bindings.
- **4.5 Typed input** through `submit_text`: intercepts, the engine with the answer streamed to
  `answer_text`, facts, clearing, actions, the log, and speech when the words were dictated. The
  copy in `ConversationViewModel` goes.
- **4.6** Rewrite `docs/architecture.md` for the new boundary; changelogs.

4.1 to 4.4 land and get their phone check before 4.5 starts, on the same branch: the session
alone is the riskiest change of the migration, and typed input should not share its regression.

## Exit

`cargo test --workspace`, the scenarios, Android unit tests. Phone, after an install over the
top (tuning is untouched): wake and answer; a quiet wake turned away; a follow-up question;
"let's talk" with and without barge-in; dictation; a card answered by voice; an alarm stopped by
voice; a cold start; cloud and the system speech service; the lock screen. After 4.5: typed
turns, dictated turns, facts, "clear history".
