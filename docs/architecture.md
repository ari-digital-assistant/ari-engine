# Ari Architecture — End to End

How a user utterance becomes a response, from wake word to output.

## The pipeline

```
User speaks
    │
    ▼
1. Wake word detection (microWakeWord, always listening)
    │
    ▼
2. Speech to text (on-device sherpa-onnx, or a cloud Whisper endpoint)
    │
    ▼
3. Input normalisation
    │
    ▼
4. Keyword/regex scoring — runs FIRST, for everyone (fast, free)
    │ matched
    ├──────────► Skill executes → response to user. Done.
    │
    │ no match
    ▼
5a. Example phrases — the skills' own phrasings, matched directly.
    │ Offline, instant, free. No model.
    │
    ├─ a phrase matches ─────────► Skill executes. Done.
    └─ nothing matches ──────────► step 5b
    ▼
5b. The assistant routes what's left
    │
    ├─ Cloud assistant, English → ONE-SHOT call to the cloud:
    │      routes to a skill ─────► Skill executes. Done.
    │      otherwise ─────────────► cloud answers directly. Done.
    │
    ├─ Non-English → the assistant (cloud or on-device) picks from the
    │  catalogue:
    │      picks a skill ─────────► Skill executes. Done.
    │      none ──────────────────► step 6
    │
    └─ English, no cloud assistant → nobody else routes; the on-device LLM
       is far too slow at it. Straight to step 6.
    ▼
6. Answer the leftover (a general question, not a skill request)
    │ on-device LLM or cloud assistant answers
    ├──────────► Response to user. Done.
    │
    │ no assistant → NotUnderstood
    ▼
7. STT retry (two additional passes with different audio slicing)
    │ retries re-enter at step 3
    │
    │ all retries exhausted
    ▼
8. "Sorry, I didn't understand that."
```

## Step by step

### 1. Wake word detection

microWakeWord runs continuously on a foreground service (`WakeWordService`),
processing audio from a single `AudioRecord` via `CaptureBus`. The same mic
feed is shared with STT — no handover, no gap.

Three bundled wake word models: `hey_ari` (default), `ok_ari`, `hey_jarvis`.
User picks one in Settings. The model runs TFLite inference every 30ms on
int8-quantised audio features from the `micro_speech` C preprocessor.

When the wake word fires, `CaptureBus` arms the STT channel and slices the
2-second ring buffer (pre-roll) into the STT stream. The voice overlay
activity launches over the lock screen via the SYSTEM_ALERT_WINDOW BAL
privilege.

### 2. Speech to text

The user picks on-device or cloud; which *model* serves on-device is decided
by their locale, not by them (`SttModelRegistry.onDeviceFor`). Three paths
result:

- **On-device streaming** (English → Kroko zipformer2): decodes in 100ms
  batches from the 2-second pre-roll, emitting partials as the user speaks.
- **On-device buffered** (everything else → Whisper-turbo): no partials; the
  whole utterance is decoded in one shot at end of speech. Kroko is
  English-only, so this is the only local option for other languages.
- **Cloud**: the utterance is buffered the same way, then POSTed to any
  OpenAI-compatible `/audio/transcriptions` endpoint — OpenAI's, or a
  self-hosted `faster-whisper` (Home Assistant's Whisper add-on among them).

Whichever path ran, the wake phrase is stripped from the transcript via regex
(`WakePhrase.kt`).

Endpoint detection is custom, because sherpa's built-in endpoint is disabled
(it freezes the stream on fire, and `reset()` destroys encoder context). The
streaming path ends an utterance when the partial text has been unchanged for
1500ms **and** the silero VAD has heard no speech for 1000ms — stability alone
measures the *decoder* going quiet, which in noise diverges from the user
going quiet, and amputated utterances. A 4s stability override stops
continuous speech-like noise (a television) from vetoing the endpoint forever,
and a 30s hard cap backstops both. The buffered paths use RMS silence
detection instead, having no partials to watch.

Model choice matters more than any of this. Nemotron 0.6B int8 was retired
after `ari-tools/scripts/stt_bench.py` replayed captured audio through it and
the 71 MB Kroko: identical input, "how's the weather" → "how's the weat"
versus the full phrase. Run the bench before changing pipeline code to chase a
transcription bug.

### 3. Input normalisation

The transcript goes to `AriEngine.processInput()` via FFI. The engine
normalises it:
- Lowercase
- Expand contractions ("what's" → "what is", "don't" → "do not", etc.)
- Strip punctuation (except math operators `+-*/.%^`)
- Convert number words to digits ("twenty five" → "25")

### 4. Keyword/regex scoring

Every registered skill (built-in + community) runs `score()` against the
normalised input. Skills declare keyword patterns and/or regex in their
manifest. The scorer computes a 0.0–1.0 confidence for each skill.

Three ranking rounds run in sequence. Each round has per-specificity
thresholds — High specificity skills get first crack, Low specificity
skills only enter in round 3:

| Round | High | Medium | Low |
|-------|------|--------|-----|
| 1     | ≥ 0.85 | excluded | excluded |
| 2     | ≥ 0.75 | ≥ 0.85 | excluded |
| 3     | ≥ 0.60 | ≥ 0.70 | ≥ 0.80 |

The first skill to clear its round's threshold wins. Its `execute()` runs
and the response is returned to the user.

This step is **fast, deterministic, and free** — no model inference, just
string matching. It handles the majority of everyday utterances.

### 5a. Example phrases

Anything the keyword scorer didn't claim is matched against the skills' own
example phrases. Every skill declares them — built-ins in
`example_utterances_for(locale)`, community skills in their manifest's
`examples:` block — and they are the oblique phrasings the keyword patterns
deliberately miss.

A phrase is a template: literal words plus `{slot}` placeholders, each of
which binds one or more words. `play {song}` matches "play hotel california"
but not "play", and not "shall i play something" — the match is anchored at
both ends, so a phrase claims the whole utterance or nothing.

Each phrase carries a **weight** on the same 0..=1 scale as
`matching.patterns`, reflecting how uniquely its wording points at one skill:
"play {song}" is 0.95, "can we have some {artist}" is 0.55, because the latter
could plausibly belong to several skills. Scoring then runs the *same* ranking
rounds the keyword tier uses, so weight and specificity arbitrate exactly as
they do there.

**It runs second, deliberately.** A `{slot}` phrase is a looser signal than an
explicit trigger, so it must never outrank one — an utterance a keyword
pattern claims never reaches this tier. A skill that wins here and then
declines (the `_ari_no_match` sentinel) falls through like any other tier.

Phrases are matched against normalised input, so they are stored normalised
too: `normalize_phrase` expands contractions in the literals while leaving
`{slot}` intact, since plain normalisation strips the braces. Manifest phrases
go through it at load; the built-in banks are static and stored pre-normalised,
with a test guarding that they stay that way. An un-normalised phrase is not an
error anywhere — it simply never matches.

> This tier replaced a fine-tuned on-device routing model in September 2026,
> and inherited its training corpus as the phrase banks. The post-mortem in
> `docs/postmortems/` covers why the model went.

### 5b. The assistant routes what's left

**Cloud assistant, English (one-shot).** A *single* call that either routes to
a skill (the model replies `SKILL: <id>`) or answers the question directly —
folding route+answer into one call avoids a second round-trip. (See
`Engine::route_or_answer`.)

**Non-English.** The engine asks the *active assistant* (cloud or on-device
LLM) to pick a skill id from the catalogue, or answer.

**No cloud assistant, any language.** Nothing else routes. The on-device LLM
takes ~22s to route because the catalogue prefill dominates, so the keyword and
phrase tiers' verdict stands and the query goes straight to step 6 to be
answered.

### 6. Answer the leftover

If routing produced no skill (no phrase matched, or the assistant said
"none"), the active assistant answers the question directly:

- **Builtin** — on-device GGUF model (Gemma 3 1B default). One-sentence
  general-knowledge answer. Lazy-loaded, 60s idle eviction.
- **API** — cloud provider (ChatGPT, Claude, Ollama, etc.) configured via a
  declarative `type: assistant` SKILL.md manifest. (For a cloud assistant the
  one-shot in step 5 already produced the answer in the same call.)
- **None** — no assistant configured. Returns `NotUnderstood` immediately.

Only one assistant can be active at a time. The user picks one in
Settings > Assistant.

### 7. STT retry

If the engine returns `NotUnderstood` (no skill matched and no assistant
answered), the Android host retries the speech-to-text pipeline with
different audio processing:

1. **Clean-start parallel stream** — skips the pre-roll, uses only live
   audio with a fresh encoder state. Different token commits may yield a
   different transcript.
2. **Offline full-buffer** — fresh stream, entire captured PCM in one
   `acceptWaveform` + `inputFinished`. Maximum decoder context.

Streaming (on-device English) only. The buffered paths have no second decoder
to disagree with, and a cloud retry would be the same request billed twice, so
both report `parallel = null` and skip the ladder entirely.

Worth knowing before relying on this: the ladder fires on `NotUnderstood`,
which a configured assistant makes rare — it answers the mangled transcript
rather than giving up, so the retries never run. A transcript arriving wrong
but *plausible* is not rescued here.

Each retry re-enters the pipeline at step 3 (normalisation) with the new
transcript. If a retry produces a transcript that matches a skill, the
user sees the corrected response with a brief flash of the corrected
transcript.

### 8. Final fallback

If all retries also return `NotUnderstood`, Ari says "Sorry, I didn't
understand that." and returns to listening for the wake word.

## Component locations

| Component | Location |
|-----------|----------|
| Wake word detection | `ari-android/.../wakeword/` (C++/JNI + Kotlin) |
| Audio pipeline | `ari-android/.../audio/CaptureBus.kt`, `AudioRingBuffer.kt` |
| STT | `ari-android/.../stt/SpeechRecognizer.kt` (sherpa-onnx) |
| Input normalisation | `ari-engine/crates/ari-core/src/lib.rs` |
| Keyword scoring | `ari-engine/crates/ari-skills/src/*.rs` (built-in), `ari-skill-loader` (community) |
| Phrase matching | `ari-engine/crates/ari-core/src/lib.rs` (`phrase_matches`), `ari-skill-loader/src/scoring.rs` |
| Assistant fallback | `ari-engine/crates/ari-llm/src/lib.rs` (builtin LLM), `ari-skill-loader/src/assistant.rs` (API adapter) |
| STT retry | `ari-android/.../voice/VoiceSession.kt` |
| Engine orchestration | `ari-engine/crates/ari-engine/src/lib.rs` (`process_input_traced`) |
| FFI boundary | `ari-engine/crates/ari-ffi/src/lib.rs` |

## What each layer catches

| Layer | Catches | Example |
|-------|---------|---------|
| Keyword scorer (always first) | Exact keyword/regex matches | "what time is it" → CurrentTime |
| Example phrases (on-device, second) | Oblique phrasings the keyword patterns deliberately miss, in every language a skill declares phrases for | "is it morning yet" → CurrentTime; "che ore sono ormai" → CurrentTime |
| Cloud one-shot | Routes *or* answers in a single call, for what the phrase tier declined | "remind me at 5" → Reminder; "capital of France" → answered |
| Assistant | Answers the general-knowledge questions routing left behind | "what's the capital of France" → "Paris." |
| STT retry | Misheard transcripts | "wheat time" (misheard) → retried → "what time" → CurrentTime |
