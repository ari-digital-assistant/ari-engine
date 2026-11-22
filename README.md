# ari-engine

The core of **[Ari](https://heyari.dev)** — an open-source, on-device digital
assistant. This repo takes an utterance and returns a response: text, a JSON
action envelope for the frontend to render, or nothing at all if no skill and
no assistant can help.

It is a Rust workspace with no UI of its own. The frontend is
[ari-android](https://github.com/ari-digital-assistant/ari-android) (a Linux
desktop one is planned and not yet started); the community skills live in
[ari-skills](https://github.com/ari-digital-assistant/ari-skills).

Everything here runs offline unless a skill or a cloud assistant is explicitly
configured to reach the network.

## How an utterance is answered

Four tiers, cheapest first, and most utterances never get past the second:

1. **Keyword and regex scoring** — every registered skill scores itself against
   the normalised input. Deterministic, instant, free.
2. **Example phrases** — the skills' own phrasings, matched as templates with
   `{slot}` placeholders. Still offline, still no model.
3. **The assistant routes** — a cloud assistant folds route-or-answer into one
   call; other configurations pick a skill id from the catalogue.
4. **The assistant answers** — whatever routing left behind, answered directly
   by the on-device model or a configured cloud provider.

The full pipeline, including wake word, speech-to-text and the retry ladder, is
in **[docs/architecture.md](docs/architecture.md)**.

## Crates

| Crate | What it is |
|---|---|
| `ari-core` | Shared types, input normalisation, phrase matching |
| `ari-engine` | Orchestration — registration, ranking rounds, conversation state |
| `ari-skills` | The built-in skills: time, date, calculator, greeting, open, search |
| `ari-skill-loader` | Community skills — manifests, the wasmtime sandbox, signature verification, registry updates |
| `ari-skill-validate` | The registry's validator, a thin binary over the loader |
| `ari-sign-bundle` | Ed25519 signing for published skill bundles |
| `ari-llm` | On-device LLM, lazy-loaded and idle-evicted |
| `ari-voice` | The voice pipeline every frontend shares, moving in from ari-android: wake word detection and its speech check, the wake log, wake phrase rules, wake word models and ladders, cloud speech-to-text |
| `ari-native` | Binds the C libraries Ari opens at run time rather than links |
| `ari-ffi` | The UniFFI boundary the frontends link against |
| `ari-cli` | A terminal harness for the whole thing |

`ari-skill-validate` being a thin wrapper over `ari-skill-loader` is deliberate:
it means the registry's validator cannot drift from the engine that will
actually run the skill.

## Build it

```bash
cargo build --workspace
cargo test --workspace
```

`ari-voice`'s tests run the wake engines, which are C++ in `native/voice/`.
Build them for your machine first (Linux x86_64, with cmake and clang; it
fetches TFLite Micro and ONNX Runtime the first time):

```bash
native/voice/build-host.sh
```

`ari-llm` opens LiteRT-LM's `liblitert-lm.so` at run time rather than linking
it, so building needs nothing of its own; running an on-device model needs the
library on the loader's path. Cross-compiling for Android needs only the NDK's
compilers, which Gradle's Rust plugin sets up.

If `cmake` is not installed system-wide, the Android SDK ships one — put it on
`PATH` rather than installing a second:

```bash
export PATH="$HOME/Android/Sdk/cmake/3.22.1/bin:$PATH"
```

Building without the model at all is supported and is what the skills registry
and the training pipeline use:

```bash
cargo build -p ari-ffi --no-default-features --bin keyword-hit
```

CI builds that combination separately, because a full-feature test run resolves
every `ari_llm` reference and so cannot catch a break in it.

## Try it

`ari-cli` runs the engine against a terminal, with the built-in skills
registered and no frontend in sight:

```bash
cargo run -p ari-cli -- "what time is it"
cargo run -p ari-cli -- --debug "what is 12 times 7"   # scoring trace on stderr
```

Give it no utterance and it reads one per line from stdin. `--extra-skill-dir`
sideloads a skill directory, `--llm-model <path>` wires up the on-device
fallback (a `.litertlm` or `.task` file, with `liblitert-lm.so` on
`LD_LIBRARY_PATH`), and `ari install` / `list` / `check-updates` / `update` drive the
skill store. `cargo run -p ari-cli -- --help` has the rest.

## How the frontends use it

Android compiles this workspace as part of its own Gradle build and expects it
as a **sibling checkout**:

```
your-workspace/
├── ari-android/
└── ari-engine/    ← you are here
```

`ari-ffi` exports the surface over UniFFI, which generates the Kotlin bindings.
Its `llm` and `not(llm)` twin `impl` blocks must stay identical down to argument
names and doc comments — UniFFI folds all three into per-method checksums, and
drift there breaks the bindings at runtime on the device rather than at compile
time. There is a test for it, and CI runs it.

## Docs

- **[docs/architecture.md](docs/architecture.md)** — the pipeline end to end
- **[docs/assistant-skills.md](docs/assistant-skills.md)** — the design of the
  `type: assistant` skill, for shipping an alternative brain
- Skill authoring lives in
  [ari-skills/docs](https://github.com/ari-digital-assistant/ari-skills/tree/main/docs)

## Contributing

Engine changes go straight to `main`. Only `ari-skills` gates things behind
pull requests. Run `cargo test --workspace` before you push; `ari-skills`
manifests are part of the keyword oracle's fixtures, so a change here can turn
that repo red and vice versa.

## Licence

[GPL-3.0](LICENSE).
