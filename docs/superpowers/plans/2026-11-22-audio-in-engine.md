# Audio In: Moving the Voice Pipeline into the Engine

Status: agreed with Keith, 2026-11-22. Expand each step's tasks into step-level detail
before executing it. Step 4 gets a full plan of its own, written when step 3 has merged.

**Goal:** Move the engine's boundary from "text in, response out" to "audio in, events out", so a
second frontend (Linux first, then macOS and Windows) reuses the wake word, speech-to-text and
voice session logic instead of rewriting it. Android behaves the same before and after every step.

**Why now:** `ari-android` is the only frontend, so there is exactly one consumer to keep working.
Every desktop frontend written before this lands would copy the logic below.

## Evidence (checked 2026-11-22)

- The voice code in `ari-android` is about 12,000 lines of Kotlin plus 1,201 lines of C++. Most of
  the Kotlin is Android plumbing (services, activities, WorkManager, geofences, notifications). The
  portable part is the wake engines (C++), the decode and endpoint loops in
  `stt/SpeechRecognizer.kt` (1,328 lines), the session in `voice/VoiceSession.kt` (1,547 lines), and
  about 1,500 lines of rules and tables with no Android imports.
- 23 of the last 93 commits on `ari-android` (30 days) touched code that moves. A long-lived branch
  would need each of those re-applied by hand in Rust.
- `sherpa-onnx-2.0.1.aar` ships `libsherpa-onnx-c-api.so` for every ABI, with a SONAME, needing only
  `libonnxruntime.so` from the same AAR. The engine can load it the way `ari-llm/src/litert.rs`
  loads LiteRT-LM.
- The sherpa C API has no per-window VAD probability. It exports the segment calls only
  (`AcceptWaveform`, `Detected`, `Front`, `Pop`). `stt/SpeechGate.kt` is built on `Vad.compute`,
  which exists in the JNI binding alone. The VAD therefore cannot move through sherpa.
- `OpenWakeWordEngine.cpp` already drives ONNX Runtime through its C API, `dlopen`ing
  `libonnxruntime.so`. `MicroWakeWordEngine.cpp` links tflite-micro statically. Only 223 of the
  1,201 C++ lines are JNI.
- The session has almost no tests. `VoiceSessionTest.kt` (195 lines) covers its top-level pure
  functions. The state machine itself is exercised only on the phone.
- `ui/conversation/ConversationViewModel.kt` repeats the session's handling of `factsChanged` and
  `clearConversation` for typed input.
- UniFFI is 0.31. Its Kotlin binding calls through JNA, maps `Vec<i16>` to `List<Short>` (one boxed
  object per sample) and `bytes` to `ByteArray` (copied through a `RustBuffer`).
- The mic loop reads 480 samples (30 ms), so it calls the detector 33 times a second for as long
  as Ari listens.

## What moves and what stays

| Moves into the engine | Today | Step |
|---|---|---|
| Wake phrase matching, mishear tables, name verdicts | `voice/WakePhrase.kt`, `earlyNameCheck` | 1 |
| Wake model registry, operating points, ladder, tuner fit | `wakeword/WakeWordModel.kt`, `TunedLadder.kt`, `WakeWordSensitivity.kt`, `FalseWakeRemedy.kt`, `LadderReview`, `SensitivityTuner.fit` | 1 |
| Cloud transcription client | `stt/CloudTranscriber.kt` | 1 |
| Wake engines, speech check, debounce, wake log | `cpp/*`, `WakeWordService.captureLoop`, `SpeechCheckPacing.kt`, `WakeLog.kt` | 2 |
| Voice activity detection | `stt/SpeechGate.kt` | 2 (wake side), 3 (listen side) |
| Ring buffer and listen feed | `audio/CaptureBus.kt`, `AudioRingBuffer.kt` | 3 |
| Streaming and buffered listening, endpointing, retry decode | `stt/SpeechRecognizer.kt`, `BufferedEndpoint.kt` | 3 |
| Tuner measurement (take collection, bisection) | `wakeword/SensitivityTuner.kt` | 3 |
| Voice session, barge-in rules, dictation | `voice/VoiceSession.kt`, `BargeIn.kt` | 4 |
| Capture decisions and stores | `WakeCaptureStore.kt`, `UtteranceCaptureStore.kt` | 4 |

Stays in each frontend, because each OS does it differently: opening the mic and choosing its
source, playing cues and speech, the foreground service and boot handling, the overlay and all UI,
running action envelopes, settings storage, model downloads, Android's system speech service, and
user-facing strings.

Stays in Android and is not shared (see Decisions): listening modes (`listening/*`), which are
rules about a phone's screen, charger, headset and geofences.

## Target design

### Crates and libraries

- **`crates/ari-voice`** (new): every portable piece above. No UniFFI in it. A Rust frontend links
  it directly; `ari-ffi` wraps it for Kotlin, and for Swift later.
- **`native/voice/`** (new, in `ari-engine`): the C++ wake engines moved from
  `ari-android/app/src/main/cpp`, a silero VAD class on ONNX Runtime, and one C header. Builds one
  shared library, `libari_voice_native.so`. The JNI wrappers are deleted. Each frontend's build
  compiles it: Android points `externalNativeBuild` at this `CMakeLists.txt`.
- **Loading:** `ari-voice` opens `libari_voice_native.so` and `libsherpa-onnx-c-api.so` at run
  time with `libloading`, by bare name, as `litert.rs` does. The `c_api!` loader macro there gains
  its second and third users, so it is extracted to one place and not copied.
- **`assets/voice/`** (new, in `ari-engine`): the wake models, openWakeWord's two feature models,
  `silero_vad.onnx`, and their licence files. Android adds the directory as an assets source. The
  engine is handed model bytes (wake, VAD) or a model directory (speech-to-text); it never reads an
  APK.

### Threads

| Thread | Owner | Does |
|---|---|---|
| Capture | frontend | Reads the mic, calls `push_audio`. Inside the call: ring buffer write, wake scoring, speech check. Same work, same thread as today. |
| Listen | engine | One per listen. Decodes, endpoints, emits speech-to-text events. Owns its sherpa streams alone and frees them itself. |
| Session | engine | One, long-lived. Owns all session state. One message queue; timers are `recv_timeout` deadlines. No async runtime. |
| Engine call | engine | `process_input` blocks, so it runs on a worker and posts its result back tagged with the turn generation. |

Host callbacks arrive on engine threads. A frontend hands them to its own UI thread and never calls
back into the pipeline from inside one. `FfiAnswerSink` already carries this rule.

### The interface

**Audio in** is a plain C function, not UniFFI:

```c
void ari_voice_push_audio(const int16_t *samples, size_t count);  /* 16 kHz mono */
```

It forwards to the one live pipeline (a static weak reference set when the pipeline is built: one
mic, one pipeline per process), so no raw pointer crosses the boundary. Android calls it from a JNI
shim of about 30 lines; a Rust frontend calls `VoicePipeline::push_audio(&[i16])` directly. The
24-hour path must cost no more than today's JNI call, and the UniFFI route (JNA, a copy, and boxing
for `Vec<i16>`) would not meet that.

**Everything else** is UniFFI: a `VoicePipeline` object for commands and a `FfiVoiceHost` callback
trait for events, both thin wrappers over `ari-voice`'s own types. The surface grows one step at a
time. Exports that exist only because part of the pipeline is still in Kotlin are marked interim
and deleted by the step that makes them internal.

| Step | Commands added | Events added | Interim, removed later |
|---|---|---|---|
| 1 | Plain functions: wake phrase match, model registry, operating point, ladder parse and fit; `CloudTranscriber` object | none | none |
| 2 | `configure_wake`, `start_wake`, `stop_wake`, `set_wake_paused`; `WakeDetector` object (tuner, parity test); wake log report | `wake_detected` | `set_listen_armed`, wake log writers called by the Kotlin session (gone in 3 and 4) |
| 3 | `load_stt`, `unload_stt`, `start_listen`, `stop_listen`, `tune_take`; host transcriber hook | `stt` (listening with partial, transcribing, done, no speech, error) | `peek_recent`, `last_listen_audio`, `retry_decode`, `check_name_early` (gone in 4) |
| 4 | `start_turn`, `start_dictation`, `submit_text`, `dismiss`, `speech_finished` | `state`, `reveal`, `play_cue`, `speak`, `stop_speaking`, `capture_mode`, `intercept`, `handle_action`, `turn_logged`, `facts_changed`, `dictated`, `false_wake` | none |

After step 4 the frontend's whole job in a voice turn is: push audio, show the state it is told,
play the sounds it is told, run the actions it is handed, and report when speech finished.

### Smaller decisions

- **Strings.** The engine emits typed states and phrase identifiers (`Error(NoSpeechModel)`,
  `Phrase::PleaseWait`). Wording and translation stay in each frontend's own pipeline.
- **Time.** Rule code takes the time as an argument in milliseconds, as `BufferedEndpoint` does
  now, so tests drive the clock. Threads read a monotonic clock. Kotlin reads the wall clock, but
  only ever subtracts, so the behaviour is the same.
- **Host transcription.** Android's system speech service stays in Kotlin behind a host
  transcriber hook: the engine feeds it listen audio and takes partials and finals back. The same
  hook later serves the speech services macOS and Windows provide.
- **Configuration.** Settings stay stored by the frontend, which pushes a config struct when one
  changes. The engine holds no settings file of its own for voice.

## Rules for every step

1. **Move, do not change.** Same constants, same order of checks, same log lines. A behaviour
   change found to be necessary is raised with Keith before it is made, and lands as its own commit.
2. **Comments travel.** The "why" comments in the Kotlin record phone findings that took weeks to
   earn. They move with the code they explain.
3. **Tests first.** Each Kotlin unit test is ported to Rust and passing before the Kotlin it
   covers is deleted. Anything numeric gets a parity fixture: old path and new path on the same
   audio, compared.
4. **One step in flight.** Each step is a branch of the same name in both repos, merged (engine
   first, then Android) once its phone check passes, before the next starts.
5. **Fixes during a step.** A fix on `main` to a file the step is moving is re-applied on the
   branch by hand and noted in the step's merge commit.
6. **Each step leaves the app working** and shippable to internal testers.
7. **Wipe installs.** Steps 1, 2 and 3 touch the tuning path: pull `files/wake-samples` first, then
   uninstall, install and `pm clear`.
8. **Phone checks** are appended to `Ari/to-test.md` when the step's build is ready.
9. The numbers below are copied exactly and are the first thing a reviewer diffs.

   Session: `SILENCE_TIMEOUT_MS` 30000, `WAKE_TURN_SILENCE_TIMEOUT_MS` 8000, `QUIET_GIVE_UP_MS`
   3000, `QUIET_LISTEN_LIMIT_MS` 6000, `QUIET_NAME_WAIT_MS` 3000, `MIN_RESPONSE_DWELL_MS` 3000,
   `CARD_DWELL_MS` 10000, `PREROLL_CAPTURE_SECONDS` 2.0, `CORRECTION_FLASH_MS` 600,
   `RE_ARM_CUE_VOLUME` 0.8.
   Streaming endpoint: `STABILITY_WINDOW_MS` 1500, `VETO_SPEECH_WINDOW_MS` 1000,
   `VETO_OVERRIDE_STABILITY_MS` 4000, `MAX_ONLINE_UTTERANCE_MS` 30000.
   Buffered endpoint: `MIN_SILENCE_AFTER_SPEECH_MS` 800, `MIN_UTTERANCE_MS` 500, `DETECTOR_HOLD_MS`
   4000, `MAX_SAMPLES` 480000.
   Audio: 16 kHz mono, 2 s ring, 100 ms (1,600 sample) pre-roll slices and decode batches.

## Working arrangement

A second checkout holds both repos on the step's branch, so the main checkout stays on `main` for
bug fixes and nothing is ever switched:

```bash
cd ~/LocalCode/keithvassallomt
mkdir Ari-migration
git -C Ari/ari-engine  worktree add ../../Ari-migration/ari-engine  -b audio-in-step-1
git -C Ari/ari-android worktree add ../../Ari-migration/ari-android -b audio-in-step-1
cp Ari/ari-android/local.properties Ari-migration/ari-android/
```

Android finds the engine at `../../ari-engine/crates/ari-ffi`, which resolves inside
`Ari-migration`. The first build there compiles everything from scratch. The Kotlin bindings in
`uniffi/ari_ffi/ari_ffi.kt` are checked-in source: regenerate them with `uniffi-bindgen` from the
built library after every FFI change and read the diff.

## Step 1: rules and tables

Branch `audio-in-step-1`. About 2 to 3 hours of work; half a day of normal use to check.

Only logic that Kotlin can call at low frequency moves here. Rules called per audio chunk
(`shouldEndpoint`, `BufferedEndpoint`, `SpeechCheckPacing`) move with their loops in steps 2 and 3,
and the session's four-line gate functions move with the session.

- **1.1** Create `crates/ari-voice`, add it to the workspace, and make `ari-ffi` depend on it.
- **1.2** Port `WakePhrase.kt` and `earlyNameCheck` to `wake_phrase.rs`, with `WakePhraseTest`
  (151 lines) and `EarlyNameCheckTest` (53). Java and Rust regex differ on Unicode word boundaries
  and case folding, so also run both implementations over every transcript in the capture sidecars
  under `ari-tools/wakeword/captures/` and require identical output.
- **1.3** Port the model registry, operating points, sensitivity levels, `TunedLadder`,
  `FalseWakeRemedy`, the `LadderReview` object and `SensitivityTuner.fit`, with `TunedLadderTest`
  (107), `WakeWordSensitivityTest` (120), `FalseWakeRemedyTest` (93) and `SensitivityTunerFitTest`
  (117). `LadderReviewWorker` and the tuner's UI flow stay in Kotlin and call the ported code.
- **1.4** Port `CloudTranscriber` on the workspace's blocking `reqwest`, with `CloudTranscriberTest`
  (100). Check the dependency tree before enabling `reqwest`'s multipart feature; build the body by
  hand if it drags much in.
- **1.5** Export through `ari-ffi`, regenerate the bindings, switch the Kotlin call sites, delete
  the Kotlin originals and their tests.
- **1.6** Update `docs/architecture.md` (boundary statement and component table) and the changelog.
  Find anything in `ari-tools` that reads the deleted Kotlin files (`eval_model.py` reads
  `WakeWordModel.kt`) and point it at the new source.

**Exit:** `cargo test --workspace` and Android unit tests pass. Phone: wake and command on-device;
a cloud transcription turn; a quiet wake that shows on the name; the sensitivity setting; one tuner
run that saves a ladder.

## Step 2: wake detection

Branch `audio-in-step-2`. About 3 to 5 hours of work; a day and a night to check.

- **2.0 Spike, before anything else:** a silero VAD class on ONNX Runtime in C++, compared with
  sherpa's `Vad.compute` window by window on a fixture clip (an instrumented test on the phone
  while both exist, and on the laptop). The speech check and both endpoint rules were measured on
  sherpa's numbers, so the port has to reproduce them. If it cannot, stop and bring the difference
  to Keith with the offline exam (`ari-tools/wakeword/trained/2026-10-07-vad-gate/`) rerun on the
  new numbers.
- **2.1** Move the C++ to `ari-engine/native/voice/`, add the C API, delete the JNI wrappers, build
  `libari_voice_native.so`. Point Android's `externalNativeBuild` at it. Add a host build for Linux
  so `cargo test` can load the library (the ONNX Runtime the CMake file already fetches for its
  headers includes the Linux library).
- **2.2** Move the model files and licences to `assets/voice/`.
- **2.3** In `ari-voice`: the native loader, `WakeDetector`, the ring buffer, the capture-loop
  logic (paused, armed, debounce, speech check with `SpeechCheckPacing` and its 71-line test), and
  `WakeLog` with `WakeLogTest`. The wake log's lines stay byte-identical: bug reports and
  `ari-tools` read them.
- **2.4** `ari_voice_push_audio`, the JNI shim, the `VoicePipeline` object, `wake_detected`.
- **2.5** Android: `captureLoop` shrinks to read, `CaptureBus.write`, push. `MicroWakeWord.kt`,
  `OpenWakeWord.kt` and `WakeDetector.kt` go. The tuner and `OpenWakeWordParityTest` use the FFI
  `WakeDetector`. For this one step the audio is held in two rings (Kotlin's for speech-to-text,
  Rust's for the speech check); step 3 removes Kotlin's.
- **2.6 Parity:** the openWakeWord fixture through the new path (worst difference today is
  0.0000011). For microWakeWord, record the old path's scores on a clip before deleting it and
  require the new path to match them exactly: it is the same compiled code.

**Exit:** as step 1, plus the parity tests. Phone, after a wipe install: all three wake words; the
speech check on and off; a tuner run; then a day and a night comparing the wake log summary (wakes,
drops, false wakes), the `AriMicCapture` thread's CPU ticks, and screen-off battery against the
187 mA baseline.

## Step 3: speech-to-text

Branch `audio-in-step-3`. About 3 to 5 hours of work; a day to check.

- **3.1** sherpa bindings. First check whether upstream publishes a maintained Rust crate that
  matches 2.0.1 and accepts a prebuilt shared library; vet it before use. Otherwise generate the
  bindings once from the 2.0.1 header and check them in. The config structs are large and nested,
  and a hand-written mismatch corrupts memory silently. Refuse to run against another version.
- **3.2** Port the streaming loop: 100 ms batching, the parallel stream that skips the pre-roll,
  finalising both streams at the endpoint, `shouldEndpoint` with `EndpointDecisionTest` (57). Port
  the buffered loop with `BufferedEndpoint` and its tests (84, and `QuietListenTest` 23), the
  loudness measure, the cloud path, the retry decode, the early name check, and model load and
  unload.
- **3.3** The Rust ring becomes the only one. The listen feed keeps today's guarantees: pre-roll
  delivered as 100 ms slices, an unbounded queue, no dropped audio, a producer that never blocks.
  `CaptureBus.kt`, `AudioRingBuffer.kt`, `SpeechRecognizer.kt`, `SpeechGate.kt` and
  `BufferedEndpoint.kt` are deleted.
- **3.4** The host transcriber hook, with `SystemSpeechService.kt` re-plumbed onto it.
- **3.5** Re-home every other reader of the bus: `SensitivityTuner` (take collection and bisection
  move to Rust behind `tune_take`), `WakeSampleRecorder`, `ConversationViewModel`. The Kotlin
  session consumes `stt` events where it read `SpeechRecognizer.state`.
- **3.6 Replay:** feed every clip in the utterance capture set through the Rust pipeline on the
  laptop, chunked as the phone chunked it, and compare with the transcripts in each sidecar. Read
  every difference; do not average them away.
- **3.7** Drop sherpa's JNI and C++ API libraries from the APK if nothing else uses them.

**Exit:** as before, plus the replay. Phone: English streaming, a buffered-model locale (the
Italian check goes to its usual tester), cloud, the system speech service, dictation, tap-to-talk,
a cold start, and whether Ari still knows when you have finished talking.

## Step 4: the voice session

Branch `audio-in-step-4`. About a day of work; a full regression to check.

- **4.0** Write this step's own plan from the code as it then stands, and freeze the host
  interface in it.
- **4.1** A scenario harness in `ari-voice`: a fake host, scripted wake and speech-to-text events,
  a manual clock. Write the scenarios from the Kotlin before porting it, so the port has something
  to fail: accepted wake, rejected wake, no-speech wake, quiet wake shown on a partial, bare "Hey
  Ari", cold start, follow-up reply, entering and leaving "let's talk", a barge-in, a stale final
  after dismiss, both silence timeouts, both retry layers, dismiss mid-turn, dictation.
- **4.2** Port the session with the `VoiceSessionTest` (195) and `BargeInTest` (76) cases.
- **4.3** Move the capture decisions and stores, and the false-wake count. The notification stays
  in Android.
- **4.4** Final host interface; delete every interim export. Kotlin's `VoiceSession` becomes an
  adapter: a state flow for the overlay, the cue player, speech output, the action handler, card
  lingering.
- **4.5** Route typed input through the same turn code and delete the copy in
  `ConversationViewModel`.
- **4.6** Rewrite `docs/architecture.md` for the new boundary.

**Exit:** as before, plus the scenarios. Phone: everything in `to-test.md` for this step,
including barge-in, "let's talk", dictation, follow-up questions, cards and the lock screen.

## Risks, most serious first

1. **VAD numbers change.** The C API cannot give what `SpeechGate` uses. Task 2.0 settles this
   before any other work depends on it.
2. **sherpa struct layout.** Task 3.1: generated bindings, a version check, and real-model tests
   on the laptop (behind `--ignored`, as the LiteRT ones are).
3. **Session behaviour nobody wrote down.** Task 4.1 writes it down first. Rule 2 keeps the
   comments.
4. **Battery on the always-on path.** The C entry point keeps the per-chunk cost where it is; step
   2's check measures it.
5. **Drift from `main` during a step.** Rules 4 and 5; short branches.
6. **Regex differences between Java and Rust.** Task 1.2's corpus comparison.
7. **Build.** Gradle compiling CMake from the engine's tree, and stale libraries: the Rust Gradle
   plugin packages every `.so` in Cargo's output, so check the APK's `lib/` after each step.
8. **Double frees under GrapheneOS's allocator.** The Kotlin avoids them by giving each sherpa
   stream one owning coroutine. In Rust each stream is owned by the listen thread and dropped
   there, which gives the same guarantee by construction. Do not share them.

## Decisions (Keith, 2026-11-22)

1. **Listening modes stay in Android.** They are rules about a phone, and a desktop has none of
   those signals. Revisit if desktop wants schedules.
2. **The non-commercial wake model moves into the engine repo** with its licence file, on the same
   footing as it sits in the Android repo today.
3. **Typed input shares the voice turn code (4.5),** which removes the last copy.

## Not in this plan

- Renaming `ari-linux` to `ari-desktop`, and any desktop code.
- A shared catalogue for speech-to-text model downloads. Downloads stay per frontend.
- Shared translations. Each frontend keeps its own strings.
- Any change to tuning, thresholds, models or the wake word itself.
