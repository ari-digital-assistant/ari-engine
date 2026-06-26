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
4. Keyword/regex scoring — runs FIRST, for everyone (fast, free)
    │ matched
    ├──────────► Skill executes → response to user. Done.
    │
    │ no match — how the leftover is routed depends on the active assistant:
    ▼
5. Route the leftover
    │
    ├─ Cloud assistant configured → ONE-SHOT call to the cloud:
    │      routes to a skill ─────► Skill executes. Done.
    │      otherwise ─────────────► cloud answers directly. Done.
    │
    ├─ On-device assistant / no assistant, English → FunctionGemma:
    │      picks a skill ─────────► Skill executes. Done.
    │      abstains ──────────────► step 6
    │
    └─ Non-English → the assistant routes (FunctionGemma is English-only):
           picks a skill ─────────► Skill executes. Done.
           none ──────────────────► step 6
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

### 5. Route the leftover

Anything the keyword scorer didn't claim is routed here. **Which router runs
depends on the active assistant** — there are three branches:

**Cloud assistant configured (one-shot).** The query goes straight to the
cloud assistant in a *single* call that either routes to a skill (the model
replies `SKILL: <id>`) or answers the question directly. FunctionGemma is
**not** consulted — a capable cloud model both routes and abstains reliably,
and folding route+answer into one call avoids a second round-trip. (See
`Engine::route_or_answer`.)

**On-device assistant, or no assistant — English (FunctionGemma).**
FunctionGemma (270M parameters, ~253MB GGUF) routes the query. It sees the
input plus the catalogue of registered skills — declared by **short alias**
(the final id segment, e.g. `weather`, not `dev.heyari.weather`), because a
270M model can't reliably emit reverse-DNS ids; the engine resolves the alias
back. It either picks a skill or abstains (`NoMatch`). FunctionGemma is
trained on Ari's own skills + a balanced set of "answer nothing" negatives,
so it abstains on general-knowledge questions (~95% on the held-out eval)
rather than force-routing them. Lazy lifecycle: loads on first use, unloads
after 60s idle; sub-second inference on phone. If FunctionGemma abstains (or
isn't installed), the query falls to step 6.

**Non-English (assistant routes).** FunctionGemma is English-only, so for
other locales the engine asks the *active assistant* (cloud or on-device LLM)
to pick a skill id from the catalogue, or answer. This needs no per-language
router model to maintain — the trade is latency on the on-device path, since
the LLM must process the whole catalogue.

> The training pipeline (`ari-tools/functiongemma`) deliberately omits Google's
> mobile-actions demo dataset and scales negatives to the skill count; a
> **promotion gate** (`route-eval`) scores abstention on a held-out set and
> blocks any retrained model that regresses before it can ship.

### 6. Answer the leftover

If routing produced no skill (FunctionGemma abstained, or the assistant said
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
| Keyword scorer (always first) | Exact keyword/regex matches | "what time is it" → CurrentTime |
| Cloud one-shot | Routes *or* answers in a single call (cloud-assistant users) | "remind me at 5" → Reminder; "capital of France" → answered |
| FunctionGemma (on-device, English) | Paraphrases the keywords missed; abstains on general knowledge | "is it morning yet" → CurrentTime; "capital of France" → abstain |
| Assistant | Answers the general-knowledge questions routing left behind | "what's the capital of France" → "Paris." |
| STT retry | Misheard transcripts | "wheat time" (misheard) → retried → "what time" → CurrentTime |
