# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
