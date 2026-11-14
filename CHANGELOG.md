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

### Fixed

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
