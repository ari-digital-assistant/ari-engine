# LiteRT-LM Backend for the On-device Assistant: Integration Plan

Status: draft, 2026-11-19, not yet signed off. Expand each task into step-level detail before
executing it.

**Goal:** Run the built-in assistant's models through LiteRT-LM instead of llama.cpp, so on-device
answers arrive about three times sooner (ari-android#52).

**Evidence (Pixel 10 Pro Fold, Medium tier, Ari's Medium system prompt, the 12 prompts from #32,
fresh conversation per question, two passes each, 2026-11-19):**

| Runtime | Median answer | Median first word | Model load |
|---|---|---|---|
| llama.cpp, shipped ARMv8.6 build | 4.27 s | 2.79 s | 2.4 s |
| LiteRT-LM 0.19.1, CPU | 1.51 s | 0.71 s | 0.6 s |
| LiteRT-LM 0.19.1, GPU (OpenCL) | 1.53 s | 0.52 s | 3.5 s |

Same facts in every answer. LiteRT sometimes writes longer bulleted lists, which the engine's
markdown stripping already handles. Small and Large are untested. Harness, results and the C test
program: `~/.cache/ari-litert-test/` (not in any repo).

**Architecture:** A second runtime inside `ari-llm`, behind the existing `Fallback` trait, so the
engine and the FFI surface stay as they are. `LazyLlmFallback` keeps its lifecycle (lazy load,
60 s idle eviction, `last_error`); only the loaded model gains a LiteRT-LM variant. The runtime is
chosen by the model file's extension (`.litertlm` or `.task` for LiteRT-LM, `.gguf` for llama.cpp). LiteRT-LM is reached through its C
API (`liblitert-lm.so`), loaded at runtime with `dlopen`.

**Tech stack:** Rust (`ari-llm`, `ari-ffi`, `ari-cli`), Kotlin/Gradle (`ari-android`), GitHub
Actions (`ari-tools` model mirror).

## Decisions

Each has a recommendation; Keith decides.

1. **Ship both runtimes during the switch, then retire llama.cpp.** A user's downloaded GGUF keeps
   answering until the `.litertlm` download for their tier is verified, and a bad LiteRT release
   can be rolled back by catalogue alone. Remove llama.cpp after one beta cycle on LiteRT with no
   regressions (Phase 5).
2. **CPU by default, GPU as an option.** The two are the same library, chosen per load by a string
   (`"cpu"` or `"gpu"`), so supporting both costs nothing in the build. On the Fold the GPU
   reaches the first word 0.2 s sooner but takes 3.5 s to load against 0.6 s, and the engine
   drops the model after 60 s idle, so most spoken questions pay the load. Cold, the CPU path
   reaches the first word in about 1.3 s and the GPU path in about 4 s. Other phones' GPUs
   (Adreno, Mali) are unmeasured; the option lets beta users report them.
3. **Runtime by file extension**, inside `ari-llm` (`LoadedModel::load`), so `load_llm_model`
   needs no change. No new FFI parameter for the runtime itself; the GPU option adds one (Phase 4).
4. **`dlopen` through `libloading`.** The C API's library has no SONAME, so Android's linker refuses
   any binary linked directly against its versioned symbols (seen on the phone: "cannot find
   liblitert-lm.so from verneed[0]"). Done with `libloading` 0.9.1 (ISC, current, its only
   Linux/Android dependency is `cfg-if`). Report the missing SONAME upstream.
5. **Small runs Google's 8-bit `.task` build** (`Gemma3-1B-IT_multi-prefill-seq_q8_ekv2048.task`,
   1.07 GB), decided by Keith on 2026-11-19 despite 1.7x against the 2x bar (Phase 0 result).
6. **Keep LiteRT-LM's weight cache.** With no cache folder set, LiteRT-LM writes
   `<model file>_<mtime>_<size>.xnnpack_cache` beside the model: Small 1.0 GB, Medium 0.79 GB,
   Large 2.21 GB. It is what makes a load fast (Large on the Fold: 7.2 s first, 0.8 s after), and
   the engine unloads after 60 s idle, so it is kept. On disk each tier then takes Small 2.1 GB
   (GGUF 0.8), Medium 3.4 GB (GGUF 3.1), Large 5.9 GB (GGUF 5.0). The engine removes a model's
   older caches when it loads; the app deletes a model's caches with the model (Task 7).
7. **Use Google's prebuilt library, pinned by version and SHA-256.** It is Apache-2.0, but
   building it means Bazel. The v0.19.1 release asset `litert_lm_c_api-1.0.0.zip` (123 MB) holds
   Android arm64 (22.8 MB) and x86_64 (26.9 MB), Linux x86_64 and arm64 (about 85 MB each),
   macOS and Windows builds, plus headers.

## Global constraints

- The engine's `Fallback` trait and `Engine` stay unchanged. The uniffi surface changes only
  with the Kotlin bindings regenerated in the same change (Phase 2), because uniffi checksums
  each method's docstring and the app asserts them at load.
- Same system prompts (`prompts/{en,it}/system*.md`), same 256-token answer cap, same 2048-token
  context, same history fitting, same thinking stripping, as today.
- Sampling must stay deterministic. LiteRT-LM 0.19.1 rejects the greedy and top-k sampler types
  on CPU ("not implemented yet"); top-p with `top_k = 1` gives the same single pick.
- Thinking off: pass a thinking config with `enable_thinking = false` on every conversation.
- New user-facing strings in English, plus Italian drafted and shipped per the standing rule.
- Engine, Android and ari-tools changes go direct to main (PRs only for `ari-skills/skills/`).
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Release builds need the SDK's cmake on PATH (`~/Android/Sdk/cmake/3.22.1/bin`).

## File structure

- `ari-engine/crates/ari-llm/src/litert.rs` (new): the C function table loaded with `libloading`,
  engine/conversation wrappers, stream-chunk parsing.
- `ari-engine/crates/ari-llm/src/lib.rs`: `LoadedModel` becomes an enum over the two runtimes;
  `LazyLlmFallback` and `LlmFallback` call through it.
- `ari-engine/crates/ari-llm/Cargo.toml`: `libloading`.
- `ari-engine/crates/ari-ffi/src/lib.rs`: runtime in diagnostics; GPU flag in Phase 4.
- `ari-engine/crates/ari-cli/src/main.rs`: `--llm-model` accepts `.litertlm` and `.task`.
- `ari-android/app/build.gradle.kts`: fetch and package `liblitert-lm.so`.
- `ari-android/app/src/main/java/dev/heyari/ari/llm/LlmModel.kt`: catalogue moves to LiteRT-LM files.
- `ari-android/.../DebugSettingsPage.kt`, `DeviceDiagnostics.kt`, bug report wire format: runtime.
- `ari-tools/.github/workflows/publish-llm-manifests.yml`: mirror the LiteRT-LM files.

---

### Phase 0: Decide with all three tiers measured

Pre-declared gate for building Phases 1 to 5.

- [ ] Run the #52 harness on the Fold for Small (`gemma3-1b-it-int4.litertlm`, 0.58 GB) and Large
  (`gemma-4-E4B-it.litertlm`, 3.66 GB) against today's Q4_K_M GGUFs, CPU only, ABBA order.
- [ ] Run the 38-question EN/IT accuracy set from #32 on all three tiers through LiteRT-LM.
- **Go if** every tier answers at least twice as fast (median) and no tier scores lower on the 38
  questions than its Q4_K_M GGUF did in #32 (Small 28, Medium 34, Large 36). Small's int4 file is
  a different quantisation from Q4_K_M, so its accuracy is the likeliest failure.

**Result, 2026-11-19 (Fold, CPU, answers graded blind; data in `~/.cache/ari-litert-test/p0/`):**

| Tier | Median answer, llama.cpp → LiteRT-LM | 38 questions, llama.cpp / LiteRT-LM | Gate |
|---|---|---|---|
| Small | 2.65 → 1.52 s (1.7x) | 31 / 20 | fails both |
| Medium | 4.27 → 1.51 s (2.8x) | 36 / 35 | passes |
| Large | 9.05 → 4.27 s (2.1x) | 38 / 38 | passes |

Small's `gemma3-1b-it-int4.litertlm` invents facts the GGUF gets right (Leonardo wrote the Divina
Commedia, 6 tablespoons in a cup, √144 ≈ 6.29). Not the test: answers are identical under three
seeds and the rendered prompt matches Gemma 3's format.

**Small, other builds of the same model (same day, `~/.cache/ari-litert-test/s2/`).** LiteRT-LM
loads the older MediaPipe `.task` files from the same repo:

| Small build | Size | Median answer (llama.cpp 2.25 s) | First word (llama.cpp 1.37 s) | 38 questions (llama.cpp 31) |
|---|---|---|---|---|
| `int4.litertlm` | 0.58 GB | 1.52 s (1.7x) | 0.74 s | 20 |
| `q4_block32_ekv4096.task` | 0.72 GB | 1.56 s (1.4x) | 0.81 s | 29 |
| `q8_ekv2048.task` | 1.07 GB | 1.33 s (1.7x) | 0.65 s (2.1x) | 31 |

The 8-bit build matches llama.cpp's accuracy; it misses the 2x bar on the whole answer (1.7x)
and meets it on the first word. Its context is 2048 tokens, the same as Ari's.

### Phase 1: LiteRT-LM runtime in the engine

Done 2026-11-19 except Task 5 (needs Keith's OK to post) and Task 4's FFI half (moved to Task 8).
Real-model tests pass on the laptop for LiteRT-LM Small, Medium and Large and llama.cpp Small;
`cargo test --workspace` passes; `ari-llm` cross-compiles for Android arm64.

- [x] **Task 1: C API bindings.** `litert.rs` loads `liblitert-lm.so` with `libloading` and resolves
  the functions the engine needs: engine settings and creation, session config, sampler params,
  thinking config, conversation config and creation, `send_message_stream`, stream-chunk
  accessors, tokenizer, token count, last error. Hand-written `extern "C"` signatures from the
  1.0.0 headers; no bindgen. Check the C API version with `litert_lm_get_api_version` at load and
  refuse a major version other than 1.
- [x] **Task 2: `LoadedModel` over two runtimes.** Today's code stays as the llama.cpp arm. The
  LiteRT arm: system prompt through `conversation_config_set_system_message`, history through
  `conversation_config_set_messages`, the question through `send_message_stream`, a fresh
  conversation per question. Stream chunks arrive as JSON messages; join their text parts
  (`serde_json`, already a dependency) and feed the running text to `on_text`. History fitting
  counts each message's tokens with `litert_lm_engine_tokenize` plus 8 per message for turn
  markers. A cut-off answer is one whose text takes 256 tokens or more, so
  `drop_unfinished_tail` still applies. `run_prompt` (Layer C) uses a single user turn with no
  system prompt. On load, older weight caches for the same model file are removed (Decision 6).
- [x] **Task 3: tests.** Unit tests with exact values for chunk parsing, message JSON, runtime
  choice by extension and stale-cache removal. Ignored real-model tests (`LLM_TEST_MODEL`,
  `LLM_TEST_TIER`, `LD_LIBRARY_PATH`) for answering and streaming, history, Layer C's bare
  prompt, and unload-then-answer, through either runtime. The sum question was dropped: Small
  on llama.cpp answers it wrong.
- [x] **Task 4: ari-cli.** `--llm-model` takes a `.litertlm` or `.task` file; help, README and
  `docs/architecture.md` say so. The FFI half (runtime in diagnostics, `load_llm_model`'s
  docstring, which still says GGUF) moves to Task 8, with the bindings regenerated.
- [ ] **Task 5: report the missing SONAME upstream** (google-ai-edge/LiteRT-LM), with the Android
  linker error.

### Phase 2: Android packaging and catalogue

Built 2026-11-19, awaiting the phone check. Notes from doing it:

- Manifests are new tags, `llm-<tier>-litert-latest` (Phase 3 publishes them). The worker saves
  whatever a manifest points at under the catalogue's file name, so repointing `llm-<tier>-latest`
  would make older apps save a `.task` file as `.gguf`.
- Each tier keeps its old GGUF as `legacyFileName`: `modelFile` falls back to it, so the
  assistant keeps answering on llama.cpp; its version reads `unknown`, so the update check
  offers the new file; the worker deletes it once the new file is verified. Deleting a model
  already removes the whole tier directory, LiteRT's weight cache included.
- `runtimeBytes` re-measured on the Fold with LiteRT-LM (peak RSS over three answers, larger of
  first and second load): Small 1.745 GB (was 1.425), Medium 1.531 GB (was 4.57), Large
  3.145 GB (was 6.1). Small now needs more than Medium, so `PhoneMemory.recommended` ranks by
  tier instead of memory. Medium now fits 6 GB phones and Large 8 GB.
- Bug reports get the runtime through the diagnostics text at the top of the report's log
  (`model: gemma4-e2b-q4 (LiteRT-LM)`). A `setup.modelRuntime` field would need ari-web's
  bug-report function, which keeps only known fields, changed and redeployed.

- [x] **Task 6: package the library.** A Gradle task downloads the pinned C API zip once (checked
  against its SHA-256), extracts `lib/android_arm64` and `lib/android_x86_64`, and adds them to
  jniLibs next to the llama.cpp libraries (same pattern as `copyLlamaLibs`). The library's
  minimum API level is 24; Ari's `minSdk` is 29.
- [x] **Task 7: catalogue.** Each tier in `LlmModel.kt` points at its LiteRT-LM file and manifest
  (Small the 8-bit `.task`, Decision 5). The old GGUF stays loaded until the new file is
  downloaded and verified, then is deleted. Deleting a model also deletes its
  `<file>_*.xnnpack_cache` files, and storage shown for a model includes them (Decision 6).
- [x] **Task 8: diagnostics.** `on_device_cpu_info` (or a sibling) reports the runtime: llama.cpp
  with its CPU build, or LiteRT-LM on CPU or GPU. `load_llm_model`'s docstring stops saying
  GGUF. Regenerate the Kotlin bindings; debug screen and bug reports show the runtime.
- [ ] **Phone check:** fresh install and upgrade-over-GGUF on the Fold, each tier answers, debug
  screen shows LiteRT-LM CPU.

### Phase 3: Model mirror

- [x] **Task 9 (2026-11-19, ari-tools `d939429`, run 44499427648; manifests and mirror
  verified):** add the three LiteRT-LM files to `publish-llm-manifests.yml`, mirrored to the
  heyari-models bucket like the GGUFs. `litert-community/Gemma3-1B-IT` is gated (Gemma licence,
  automatic approval), so the workflow needs a Hugging Face token from an account that accepted
  it. Medium and Large are Apache-2.0 and ungated.

### Phase 4: GPU option

- [x] **Task 10:** declare `<uses-native-library android:name="libOpenCL.so" android:required="false"/>`
  and the same for `libvndksupport.so` in the manifest; without them an app cannot open the
  vendor OpenCL driver (the shell test didn't need them).
- [x] **Task 11:** `load_llm_model` takes the accelerator (`use_gpu`; the engine's answer log line
  names what ran it, "LiteRT-LM GPU" or the CPU with the GPU's error); a setting in the on-device model screen,
  off by default. If the GPU engine fails to load, fall back to CPU and record why in `last_error`.
- [ ] **Phone check:** GPU on the Fold answers and the debug screen says GPU.

### Phase 5: Retire llama.cpp

After one beta cycle with LiteRT-LM as the default and no regressions in bug reports.

- Done 2026-11-19 at Keith's call, ahead of the beta cycle. No automatic download for a tier still
  on its GGUF (`ModelUpdateWorker` never installs by itself, by design): it stays counted as
  installed, so the update check offers its file, and until then the engine refuses the GGUF
  with a message to update.
- [x] **Task 12:** remove `llama-cpp-2`, `llama-cpp-sys-2`, `cpu_build.rs`, `copyLlamaLibs`, the
  minijinja template rendering and the GGUF catalogue entries. The APK loses the llama.cpp
  libraries.

## Risks and open questions

- **Model choice narrows.** LiteRT-LM runs only models converted to `.litertlm` (litert-community
  publishes Gemma, and some others); llama.cpp runs any GGUF. This is the one real reason to keep
  llama.cpp after Phase 5.
- **Young API.** C API 1.0.0 arrived in v0.19.0 with breaking changes; two sampler types are
  unimplemented on CPU. Pin the version and re-run Task 3's tests on every bump.
- **Binary we don't build.** Apache-2.0 source, Google-built binary. Acceptable for Play; would
  need a source build (Bazel) for any store that requires one.
- **APK size** grows by about 23 MB (arm64, uncompressed) while both runtimes ship (Phases 1 to 4).
- **Unmeasured:** battery and heat, GPUs other than the Fold's PowerVR, RAM use against llama.cpp.
- **Linux:** the same release has Linux x86_64 and arm64 libraries, so `ari-cli` and a future
  ari-linux frontend can use the same runtime; nothing here is Android-only beyond packaging.
