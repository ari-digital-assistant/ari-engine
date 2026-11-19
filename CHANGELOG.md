# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- LiteRT-LM as a second runtime for the on-device assistant, chosen by the model file:
  `.litertlm` and `.task` files run on it, GGUF stays on llama.cpp. On a Pixel 10 Pro
  Fold it answered the same questions 1.7 times sooner on Small, 2.8 on Medium and 2.1 on
  Large, as accurately. It is opened at run time as `liblitert-lm.so`, which the host
  ships, and keeps a repacked copy of the weights beside the model for fast loads,
  removing older copies when the model file changes
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
