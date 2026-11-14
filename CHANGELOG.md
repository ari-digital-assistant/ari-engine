# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- On-device answers keep every line, so a list is no longer cut down to its opening
  line ([ari-android#30](https://github.com/ari-digital-assistant/ari-android/issues/30)).
  Answers are capped at 256 tokens, and one that hits the cap drops its unfinished
  last line.
- Assistant replies, on-device and cloud, have their markdown reduced to plain text:
  emphasis, code, links, headings, quotes, rules and tables lose their syntax, and
  bullets become "•" ([ari-android#30](https://github.com/ari-digital-assistant/ari-android/issues/30)).
