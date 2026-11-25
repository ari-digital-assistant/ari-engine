# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Removed

- llama.cpp, and with it GGUF models, the chat-template renderer, the per-ARM CPU builds
  and `on_device_runtime`. LiteRT-LM runs every on-device model; a GGUF path now fails
  to load with a message saying the model needs updating. `on_device_cpu_info` keeps
  only the CPU's features. Builds no longer need cmake's llama.cpp path, libclang or
  bindgen ([ari-android#52](https://github.com/ari-digital-assistant/ari-android/issues/52)).

### Added

- `ari_core::envelope`: a skill's presentation envelope parsed into types (cards, alerts,
  notifications and every action slot), with the defaults and leniency Android's
  `PresentationEnvelope.kt` had, which it replaces. `ari-ffi` exposes it as
  `parse_presentation_envelope`.
- `ari_core::envelope::card_press` (and `Card::press`): what pressing a card's button
  does, by the envelope's rules (`stop_alert`, `cancel` with the card's `on_cancel` first,
  then the button's words or a plain dismissal), moved from Android's
  `CardActionDispatcher.kt`. `ari-ffi` exposes it as `resolve_card_press`.
- `ari-runtime`, the engine as a frontend runs it, moved out of `ari-ffi`: the engine
  with its built-in and community skills, the platform providers (`Providers`),
  settings-time skill calls, the model catalog, the on-device model, and the replies
  turns get, as the voice session's `TurnEngine`; the skill store and registry
  (`skills::SkillRegistry`), the assistants on offer (`assistants::AssistantRegistry`)
  and skill settings (`settings::SettingsStore`), with the grants every install and
  reload loads with (`load_options`). A frontend written in Rust links it without
  `ari-ffi`. `AriEngine`, `SkillRegistry`, `AssistantRegistry` and
  `SkillSettingsStore` wrap them and their Kotlin API is unchanged;
  `build_engine_with_builtins` and `register_community_skills` are `ari_runtime`'s now.
- The voice session in `ari-voice`, moved from ari-android: a voice turn from the wake
  to the answer, with wake verification and quiet wakes, the early name check, the
  cold start, the retries, follow-up questions, "let's talk" and barge-in, dictation,
  the silence timeouts, the debug captures and the false-wake decision. Typed turns
  from a chat take the same steps (`submit_text`). A frontend implements
  `FfiSessionHost` to show, play and run what the session decides, starts turns with
  `start_turn`, and hands the engine over with `attach_engine`. The host's own
  transcriber (Android's system speech service) reports through `host_heard`, and the
  engine judges its words for the wake phrase.
- Speech-to-text in `ari-voice`, moved from ari-android: sherpa-onnx's streaming
  recogniser (Kroko, Nemotron) with its stability endpoint and VAD veto, the parallel
  stream and the retry decode; the buffered path for Whisper, Parakeet and the cloud; and
  the listen feed from the pipeline's ring. sherpa-onnx is reached through its C API,
  opened at run time, with struct bindings generated from 2.0.1's header; any other
  version is refused. `FfiVoicePipeline` gains `load_stt`, `unload_stt` and
  `set_cloud_mode`; the voice session starts and ends the listens.
- Wake word detection in `ari-voice`, moved from ari-android. The frontend pushes its
  microphone's audio through `ari_voice_push_audio`, and `VoicePipeline` runs the wake
  word, the 4-second debounce and the speech check, then tells the frontend through
  `FfiVoiceHost`. The engines are C++ in `native/voice/` (`libari_voice_native.so`):
  openWakeWord and microWakeWord as ari-android had them, and a Silero VAD that matches
  sherpa-onnx's scores. The wake log (`wake_log.rs`) writes the same lines as before.
  The models and their licences moved to `assets/voice/`.
- `ari-native`, which binds a C library opened at run time; LiteRT-LM's binding now
  uses it too.
- `ari-voice`, the voice pipeline every frontend shares, moving out of ari-android a step
  at a time (`docs/superpowers/plans/2026-11-22-audio-in-engine.md`). It starts with the
  rules and tables: finding the wake phrase in a transcript and the name checks on it,
  the bundled wake word models and their sensitivity ladders, ladders tuned on the phone,
  the false-wake remedy and the tuning fit. `ari-ffi` exports the ones the app still
  asks for (`wake_word_models`, `fit_tuned_ladder` and the rest), and ari-android no
  longer keeps its own copies.
- Cloud speech-to-text in `ari-voice::cloud_stt`, moved from ari-android: any
  OpenAI-compatible `/audio/transcriptions` endpoint, OpenAI's own or a self-hosted
  Whisper server, with each failure named by its cause. Plain `http://` endpoints work,
  and HTTPS is checked against the bundled Mozilla roots, as it is for skills and
  assistants.
- Assistant answers keep the markdown they were written in, for a host that renders it.
  `FfiResponse::Text` has a `markdown` field, set when an assistant answer formats
  anything, and `FfiAnswerSink::text` gets the markdown so far alongside the words.
  `body` is still the plain text, and is what gets spoken
  ([ari-android#57](https://github.com/ari-digital-assistant/ari-android/issues/57)).
- Cloud assistants can search the web for answers that need current information. An
  assistant skill's optional `web_search` block names the provider's search tool and
  whatever the search call needs that differs from the skill's own call (endpoint,
  request format, auth header, response path, instructions); older engines ignore it.
  `call_assistant_api` takes a `WebSearch`: answers allow a search, as does the one-shot
  call that may answer or hand off to a skill, while skill routing and
  `consult_assistant` round trips leave it off. A handoff requires one: the assistant is
  told to search before it answers, and OpenAI's Responses API gets a forced
  `tool_choice` (Anthropic's current models reject one and Gemini has none). Two request
  formats come with it:
  `openai_responses` (OpenAI's Responses API, sent without `temperature`, which its
  current models reject, and with `store: false`) and `gemini` (`generateContent`). An
  endpoint's `{model}` becomes the model ID. An answer that arrives in pieces is put
  back together before `response_path` runs: Claude's text after its last search or
  thinking block, a Responses reply's messages minus commentary, a Gemini candidate's
  parts minus thoughts.
- Handoff: with `set_handoff_enabled(true)`, the on-device assistant writes `QUERY:` and
  the question instead of answering when it doesn't know or the question needs live
  information, and Ari asks "Would you like me to ask Claude?". Yes sends the user's own
  words, with the conversation, to that cloud assistant, which must search the web for
  the answer; no drops it; anything else is a new request. It offers the cloud assistant
  that answered last, else the first with an API key, and none without a key.
  `handoff_assistants()` lists the ones it could offer, `last_cloud_assistant()` and
  `set_last_cloud_assistant()` let a frontend keep the last one across restarts. The
  marker and its question are never shown or spoken.
- Typed `/` commands for Ari's built-in phrases: `/clear`, `/remember …`, `/forget …`,
  `/forget-all` and `/recall` run exactly what saying "clear history", "remember …",
  "forget …", "forget everything about me" and "what do you remember about me" runs. The
  names are the same in every language. `slash_commands(locale)` lists them with a
  description in that language, for a frontend to offer as the user types.
- LiteRT-LM as a second runtime for the on-device assistant, chosen by the model file:
  `.litertlm` and `.task` files run on it, GGUF stays on llama.cpp. On a Pixel 10 Pro
  Fold it answered the same questions 1.7 times sooner on Small, 2.8 on Medium and 2.1 on
  Large, as accurately. It is opened at run time as `liblitert-lm.so`, which the host
  ships, and keeps a repacked copy of the weights beside the model for fast loads,
  removing older copies when the model file changes
  ([ari-android#52](https://github.com/ari-digital-assistant/ari-android/issues/52)).
- `on_device_runtime(model_path)` names the runtime a model file runs on, "LiteRT-LM" or
  "llama.cpp", for the host's diagnostics; `load_llm_model`'s description no longer says
  the model must be a GGUF
  ([ari-android#52](https://github.com/ari-digital-assistant/ari-android/issues/52)).
- `load_llm_model(model_path, use_gpu)` runs a LiteRT-LM model on the GPU when asked,
  falling back to the CPU when the GPU won't load it. The answer log line names what ran
  it ("LiteRT-LM GPU", or the CPU with the GPU's error). The GPU's weight caches (about
  as big as the CPU's) are removed when the model next loads on the CPU, and old ones
  when the model file changes
  ([ari-android#52](https://github.com/ari-digital-assistant/ari-android/issues/52)).
- A personality for assistant answers: the user's own description of how Ari should
  talk, added to the on-device and cloud assistants' instructions after Ari's own rules
  ([ari-android#31](https://github.com/ari-digital-assistant/ari-android/issues/31)).
- A conversation length setting: Short keeps five exchanges for 90 seconds of quiet (the
  old fixed limits), Medium ten for 5 minutes, Long twenty for 15 minutes
  ([ari-android#31](https://github.com/ari-digital-assistant/ari-android/issues/31)).
- `process_input_streaming(input, sink)` hands an on-device assistant answer to the sink
  while the model is still writing it: the words so far, for the host to show, and each
  finished sentence, so it can start speaking. Both are cleaned like the whole answer;
  an answer with no sentence to release early is spoken from the response as before
  ([ari-android#37](https://github.com/ari-digital-assistant/ari-android/issues/37)).
- "Clear history", "start over", "new conversation" and similar phrases forget the
  conversation so far (the follow-up memory and any question a skill is waiting on, but
  not remembered facts) and set `clear_conversation` on the response so the host clears
  its chat too. `clear_conversation()` does the same for a host that clears the chat
  another way. Before this, the phrases went to the assistant, which said it couldn't
  ([ari-android#40](https://github.com/ari-digital-assistant/ari-android/issues/40)).
- Store skill suggestions: when nothing installed handles what the user said and it
  matches one of a store skill's suggest phrases in their language, Ari offers that
  skill, before the assistant, as an action with `speak` and `suggest_skill: {id, name}`.
  Opt-in: a skill lists the phrases that clearly need it under `metadata.ari.suggest`
  (`{slot}` templates matched against the whole utterance, each needing a word of its
  own), and skills without a list, assistants and installed skills are never suggested.
  Matching patterns aren't used, because an uninstalled skill can't decline what its
  broad patterns catch. `ari-skill-validate --format=json` emits the lists per language
  as `suggest`, `IndexEntry` reads them, `SkillRegistry` keeps a copy of each index it
  fetches (`index_cache_path()`), and `set_skill_suggestions_path()` points the engine
  at it ([ari-android#44](https://github.com/ari-digital-assistant/ari-android/issues/44)).
- `release_llm_memory()` frees the on-device model's memory but keeps it set, so the next
  question loads it again. `unload_llm_model()` removes it, and the assistant stops
  answering until a model is set again
  ([ari-android#42](https://github.com/ari-digital-assistant/ari-android/issues/42)).

### Changed

- A store skill the frontend can't run is no longer suggested or offered: a frontend that
  can't send texts never suggests the message skill. `Engine::set_skill_suggestions_path`
  takes the host's capabilities, `IndexEntry::missing_capabilities` says what a registry
  skill needs that a host lacks, and `BrowseEntry::missing_capabilities` carries it to Browse.
- The voice session decides which words answer what is on screen: "stop" (or the
  alert's own Stop label) over a sounding alert, and a button's id or label on the
  newest card with buttons, with the endings speech recognition adds to it. The
  per-language stop phrases and endings moved here from ari-android.
  `FfiSessionHost.intercept` is replaced by `intercept_targets`, which lists the alerts
  and cards, and `stop_alert` and `press_card_button`, which act on the match.
- A spoken turn can carry a second transcript (the streaming recogniser's parallel
  stream, `Engine::set_alternative_transcript`). The engine acts on it instead when the
  first finds no skill and it does, before the assistant tier, and says which it acted
  on.
- Every on-device prompt has the model answer populations, sizes and other figures from
  what it knows, with an example, instead of calling them live information it can't
  reach ("I don't have live population data" for Australia). A model that copies the
  examples' `Response:` label no longer has it read out.
- The Medium and Large on-device models answer from what they know when asked about the
  best or most famous people and things, instead of calling it live information they
  can't reach, and answer a question about several things with a short list that covers
  every one of them. Small keeps its one-sentence prompt, which it answers better with
  ([ari-android#37](https://github.com/ari-digital-assistant/ari-android/issues/37),
  [ari-android#38](https://github.com/ari-digital-assistant/ari-android/issues/38)).
- On Android arm64, llama.cpp's CPU code ships in three builds, ARMv8.0, ARMv8.2 (dot
  product) and ARMv8.6 (int8 matrix multiply), and the on-device assistant loads the
  best one the phone supports. On a Pixel 10 Pro Fold the Medium model answers in
  3.1 s instead of 8.1 s ([ari-android#32](https://github.com/ari-digital-assistant/ari-android/issues/32)).
- `on_device_cpu_info()` reports which build loaded and the CPU's features.

### Fixed

- In a "let's talk" session with barge-in, a transcript made while Ari was speaking that
  the speech detector heard no speech in is ignored, and the session keeps listening. It
  was Ari's own voice through the echo canceller, taken as the user's reply. A listen's
  `SttEvent::Done` says whether the detector heard speech (`speech_heard`). While Ari
  answers, the overlay keeps the answer rather than showing partials, and the chime, the
  listening state and the reply's silence window wait until Ari has finished.
- A personality no longer talks the assistants out of formatting their answers. With
  "Be funny." set, the on-device model wrote lists as plain "•" lines with no bold. The
  personality block now says a personality changes how Ari sounds, not how a reply is
  laid out ([ari-android#58](https://github.com/ari-digital-assistant/ari-android/issues/58)).
- Remembering, forgetting or asking what Ari remembers now counts as conversation, so a
  run of them no longer ends in the conversation being forgotten mid-chat.
- The on-device and cloud assistants get what the user said, not the text Ari cleans up
  for its skills, which lowercases it and turns number words into digits: "who came
  second?" reached the model as "who came 2", and it asked what was meant. "Ask Claude
  …", a handoff and the conversation the assistants read back keep the user's words too.
- Web search only runs when what was said starts with a search command ("search for X",
  "look up X", "can you google X"), and searches for everything after the command. A
  question with "find" or "search" in the middle, such as "where do you find them in
  Europe?", goes to the assistant instead of the web, where it searched for "where do
  them in europe" with every "you", "find", "for" and "me" removed
  ([ari-android#43](https://github.com/ari-digital-assistant/ari-android/issues/43)).
- The on-device model's one-minute idle timer held a raw pointer to the model's state.
  Removing or switching the model within a minute of an answer left the timer touching
  freed memory. It now holds a weak reference and stops if the model is gone
  ([ari-android#42](https://github.com/ari-digital-assistant/ari-android/issues/42)).
- A conversation too long for the on-device model's context drops its oldest exchanges
  instead of failing the answer.
- Short on-device answers such as "Paris." or "1945." reach the user instead of being
  discarded as too short ([ari-android#32](https://github.com/ari-digital-assistant/ari-android/issues/32)).

- On-device answers keep every line, so a list is no longer cut down to its opening
  line ([ari-android#30](https://github.com/ari-digital-assistant/ari-android/issues/30)).
  Answers are capped at 256 tokens, and one that hits the cap drops its unfinished
  last line.
- Assistant replies, on-device and cloud, have their markdown reduced to plain text:
  emphasis, code, links, headings, quotes, rules and tables lose their syntax, and
  bullets become "•" ([ari-android#30](https://github.com/ari-digital-assistant/ari-android/issues/30)).
- "second" straight after a number is the unit of time, not the ordinal 2: "set a 10
  second timer" reached the timer skill as "set a 10 2 timer", and it asked for a
  duration. "twenty second" stays 22 after "the" or before "of", as in "the twenty
  second of may", and is twenty seconds anywhere else.
