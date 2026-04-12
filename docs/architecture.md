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
2. Speech to text (sherpa-onnx streaming zipformer)
    │
    ▼
3. Input normalisation
    │
    ▼
4. Keyword/regex scoring (three ranking rounds)
    │ matched
    ├──────────► Skill executes → response to user. Done.
    │
    │ no match
    ▼
5. FunctionGemma router (optional, on-device LLM)
    │ matched skill
    ├──────────► Skill executes → response to user. Done.
    │ matched system action
    ├──────────► Android intent dispatched. Done.
    │
    │ declined
    ▼
6. Active assistant (on-device LLM or cloud API)
    │ answered
    ├──────────► Response to user. Done.
    │
    │ NotUnderstood
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

sherpa-onnx streaming zipformer decodes audio in 100ms batches, starting
from the 2-second pre-roll. The wake phrase is stripped from the transcript
via regex (`WakePhrase.kt`).

Endpoint detection is custom: 1500ms of unchanged partial text = done.
sherpa's built-in endpoint is disabled (it freezes the stream on fire and
`reset()` destroys encoder context).

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

### 5. FunctionGemma router (optional)

If enabled and a model is loaded, FunctionGemma (270M parameters, ~253MB
GGUF) gets a chance to route the query. It sees:
- The user's input
- A list of all registered skills with their descriptions

It either picks a skill (returns `RouteResult::Skill`), suggests a system
action like "set an alarm" (returns `RouteResult::Action`), or declines
(returns `RouteResult::NoMatch`).

This catches paraphrases the keyword matcher missed. For example, "launch
my music player" doesn't match the Open skill's keywords ("open", "launch"
+ target), but FunctionGemma understands the intent and routes it.

The model uses the same lazy lifecycle as the LLM fallback: loads on first
use, unloads after 60 seconds of idle. Sub-500ms inference on phone.

**If the router is disabled or no model is loaded, this step is skipped
entirely.** The flow goes straight from step 4 to step 6.

### 6. Active assistant

If no skill matched (keyword or router), the engine checks for an active
assistant provider. Three modes:

- **Builtin** — on-device GGUF model (Gemma 3 1B default). Answers general
  knowledge questions in one sentence. Lazy-loaded, 60s idle eviction.
- **API** — cloud provider (ChatGPT, Claude, Ollama, etc.) configured via
  a declarative `type: assistant` SKILL.md manifest. The engine builds an
  HTTP request from the manifest's API config, sends it, extracts the
  response.
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
| FunctionGemma router | `ari-engine/crates/ari-llm/src/lib.rs` (`FunctionGemmaRouter`) |
| Assistant fallback | `ari-engine/crates/ari-llm/src/lib.rs` (builtin LLM), `ari-skill-loader/src/assistant.rs` (API adapter) |
| STT retry | `ari-android/.../voice/VoiceSession.kt` |
| Engine orchestration | `ari-engine/crates/ari-engine/src/lib.rs` (`process_input_traced`) |
| FFI boundary | `ari-engine/crates/ari-ffi/src/lib.rs` |

## What each layer catches

| Layer | Catches | Example |
|-------|---------|---------|
| Keyword scorer | Exact keyword matches | "what time is it" → CurrentTime |
| FunctionGemma | Paraphrases, indirect language | "is it morning yet" → CurrentTime |
| FunctionGemma | System actions (future) | "set an alarm for 7am" → Android intent |
| Assistant | General knowledge | "what's the capital of France" → "Valletta... wait, Paris." |
| STT retry | Misheard transcripts | "wheat time" (misheard) → retried → "what time" → CurrentTime |
