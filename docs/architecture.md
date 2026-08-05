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
5a. FunctionGemma — the on-device router. Runs whenever a model for the
    │ language being spoken is loaded. Offline, sub-second, free.
    │
    ├─ confident pick ───────────► Skill executes. Done.
    └─ abstains / below its floor ► step 5b
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

### 5a. FunctionGemma — the on-device router

Anything the keyword scorer didn't claim goes to the on-device router first,
**whenever a model exists for the language being spoken**. There is one
trained model per locale (`functiongemma-en-latest`, `functiongemma-it-latest`,
…), not one multilingual model; the host downloads the one matching the active
language and swaps it when the user switches.

FunctionGemma (270M parameters, ~253MB GGUF) sees the input plus the catalogue
of registered skills — declared by **short alias** (the final id segment, e.g.
`weather`, not `dev.heyari.weather`), because a 270M model can't reliably emit
reverse-DNS ids; the engine resolves the alias back. It either picks a skill or
abstains (`NoMatch`). It is trained on Ari's own skills plus a balanced set of
"answer nothing" negatives, so it abstains on general-knowledge questions
rather than force-routing them. Lazy lifecycle: loads on first use, unloads
after 60s idle; sub-second inference on phone.

**The confidence floor is what makes going first safe.** Every published model
ships a `min_confidence` in its manifest, derived from that specific model's
measured precision/abstention curve, and the device enforces it. A pick below
the floor is discarded and the query carries on to 5b. So the router only
speaks up when it is sure, and being wrong costs a fall-through rather than a
wrong answer.

**The locale must match.** The engine tracks which language the loaded model
was trained for (`Engine::set_router` takes both) and refuses to route with a
mismatched one. The host swaps models asynchronously on a language change, so
without this check there is a window where an English model would confidently
route Italian.

If no model is installed for the active language, this step is skipped
entirely.

> The training pipeline (`ari-tools/functiongemma`) deliberately omits Google's
> mobile-actions demo dataset and scales negatives to the skill count; a
> **promotion gate** (`route-eval`) scores precision and abstention on a
> generated eval bank at the model's own derived floor, and blocks any
> retrained model that regresses before it can ship.

### 5b. The assistant routes what's left

**Cloud assistant, English (one-shot).** A *single* call that either routes to
a skill (the model replies `SKILL: <id>`) or answers the question directly —
folding route+answer into one call avoids a second round-trip. (See
`Engine::route_or_answer`.)

**Non-English.** The engine asks the *active assistant* (cloud or on-device
LLM) to pick a skill id from the catalogue, or answer.

**English with no cloud assistant.** Nothing else routes. The on-device LLM
takes ~22s to route because the catalogue prefill dominates, so the router's
verdict stands and the query goes straight to step 6 to be answered.

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
| FunctionGemma router | `ari-engine/crates/ari-llm/src/lib.rs` (`FunctionGemmaRouter`) |
| Assistant fallback | `ari-engine/crates/ari-llm/src/lib.rs` (builtin LLM), `ari-skill-loader/src/assistant.rs` (API adapter) |
| STT retry | `ari-android/.../voice/VoiceSession.kt` |
| Engine orchestration | `ari-engine/crates/ari-engine/src/lib.rs` (`process_input_traced`) |
| FFI boundary | `ari-engine/crates/ari-ffi/src/lib.rs` |

## What each layer catches

| Layer | Catches | Example |
|-------|---------|---------|
| Keyword scorer (always first) | Exact keyword/regex matches | "what time is it" → CurrentTime |
| FunctionGemma (on-device, per-locale, second) | Paraphrases the keywords missed, in whatever language has a model; abstains on general knowledge and on anything below its floor | "is it morning yet" → CurrentTime; "che ore sono ormai" → CurrentTime; "capital of France" → abstain |
| Cloud one-shot | Routes *or* answers in a single call, for what the router declined | "remind me at 5" → Reminder; "capital of France" → answered |
| Assistant | Answers the general-knowledge questions routing left behind | "what's the capital of France" → "Paris." |
| STT retry | Misheard transcripts | "wheat time" (misheard) → retried → "what time" → CurrentTime |
