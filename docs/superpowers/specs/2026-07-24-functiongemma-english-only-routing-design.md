# FunctionGemma → English-only; cloud LLM routes everywhere else

**Date:** 2026-07-24
**Status:** Approved, ready for planning
**Repos touched:** `ari-engine`, `ari-android` (engine changes direct-to-main; Android direct-to-main)

## Problem

FunctionGemma — the on-device 270M skill router — does not work for non-English
languages. Trials on Italian confirmed it: the model routes confidently but
wrongly. Worse, since commit `4ad0191` ("run FunctionGemma first, for whatever
language has a model") the router runs *before* the cloud-LLM tier for any locale
whose model is loaded. So for Italian, FunctionGemma-IT misroutes above its
confidence floor and the cloud LLM — which *can* route Italian well — rarely gets
a turn.

The decision: **FunctionGemma is English-only. For every other language the
cloud LLM does the routing.** Offline (no cloud assistant), non-keyword requests
get a direct answer with no routing — the on-device Gemma is too slow to route
(~22s, catalogue prefill dominates) and, at this size, not good enough anyway.

## Key insight

The LLM routing this asks for is **already wired**:

- English + cloud → one-shot `route_or_answer` (routes *or* answers in one call).
- Non-English + cloud → two-step `try_assistant_route` (a routing call, localised
  per locale, then falls to a separate answer if the model picks NONE).

The reason it wasn't firing for Italian is FunctionGemma-IT preempting it. So the
work is mostly **getting FunctionGemma out of the way** for non-English and making
cloud-LLM routing the sole non-keyword router there — not building routing from
scratch.

## Target behaviour

After a keyword miss, the routing tiers resolve like this:

| Locale | Cloud assistant | Behaviour |
|---|---|---|
| `en` | yes | FunctionGemma → one-shot route-or-answer |
| `en` | no | FunctionGemma → on-device answer only |
| non-`en` | yes | **cloud LLM routes** (two-step), else cloud answers |
| non-`en` | no | on-device answer only (no routing) |

"Cloud routes, offline answers only" holds for every language. The on-device
Gemma is never asked to route.

## Changes

### Change 1 — Engine: FunctionGemma is English-only

Gate the single production consult chokepoint
[`router_for_active_locale`](../../../crates/ari-engine/src/lib.rs) so it returns
`None` unless the active locale is `en`. The existing
`router_locale == ctx.locale` check stays; we add the `ctx.locale == "en"`
requirement on top.

- Debug/eval entry points (`route_decision`, `route_raw`, `debug_route`) read
  `self.router` directly and are deliberately left unrestricted, so the eval
  harness can still score any model it's handed.
- `set_router` stays unrestricted — the host keeps telling the engine which model
  is loaded; the engine just declines to *consult* a non-English one in
  production.

### Change 2 — Engine: routing is cloud-only, offline answers only

Collapse `uses_assistant_routing(locale, has_cloud_assistant)` so it depends only
on `has_cloud_assistant` (the `locale != "en"` clause goes away). The locale
parameter becomes unused — drop it and update the one call site.

This is what makes non-English offline stop routing through the slow on-device
LLM: when there's no cloud assistant, no routing is attempted and the request
falls to the fallback tier and then the on-device answer path. When a cloud
assistant *is* present, the existing English one-shot / non-English two-step
paths run unchanged.

`has_cloud_assistant` is already derived as
`matches!(&self.active_assistant, Some(ActiveAssistant::Api { .. }))`, so when
`uses_assistant_routing` is true the active assistant is always the cloud API —
the on-device LLM can never be pulled into a routing call.

### Change 3 — Android: stop delivering the Italian router model

The delivery layer is locale-generic already. Gate
[`RouterPolicy.shouldHaveModel`](../../../../ari-android/app/src/main/java/dev/heyari/ari/router/RouterPolicy.kt)
to English:

```kotlin
suspend fun shouldHaveModel(locale: String): Boolean =
    locale == "en" && (downloadManager.isDownloaded(locale) || availability.isAvailable(locale))
```

Consequences fall out for free:

- For a non-English active locale, `reconcile` computes `required = false`, so it
  disables the router, unloads it from the engine, and `deleteLocalesExcept(null)`
  removes any stale IT model already on disk (upgrade cleanup, no migration code
  needed).
- Onboarding uses the same `shouldHaveModel(locale)` chokepoint, so picking
  Italian in the wizard won't fetch a router model either.
- No network probe fires for non-English, saving the nightly availability check
  for locales that will never have a model.

The training / gating / publishing workflows (Modal training, nightly GitHub
Actions, floating `functiongemma-<locale>-latest` releases) are **left dormant**,
not deleted — so the decision is reversible if a future model proves out.

## Tests

### Engine (flip existing, add new)

- `router_dispatches_for_a_non_english_locale_when_its_model_is_loaded` — rewrite
  to assert the IT router is **skipped** even with a matching-locale model loaded,
  and the request falls through (to fallback, since no assistant is configured in
  that test).
- `routing_backend_choice` — the `it` + no-cloud case flips from
  `assert!(uses_assistant_routing(...))` to `assert!(!...)`. Update assertions to
  the new signature.
- Add: with an `en` model loaded but active locale `it`, `process_input_traced`
  never consults the router (a CatalogCapturingRouter records no call).

### Android (add)

- `RouterPolicy` / `RouterPolicyTest`: `shouldHaveModel("it")` is `false` even
  when downloaded or available; `shouldHaveModel("en")` unchanged.
- Confirm a reconcile with active locale `it` disables the router and deletes
  on-disk locale dirs.

## Verification (device / emulator)

Per the "repeat the test phrase" convention, the exact utterances to retest:

**With a cloud assistant configured, locale `it`** — keyword-miss idioms route to
the time skill:

```
è ora di una birra?
```

```
is it beer o'clock yet?
```

**Offline (no cloud assistant), locale `it`** — same query returns a direct
answer, no routing:

```
è ora di una birra?
```

This is where we confirm the cloud routing prompt is actually good enough for
idiomatic queries. If it isn't, prompt-tuning `build_assistant_routing_prompt` is
the reactive follow-up — not part of this change.

## Out of scope (YAGNI)

- Deleting the IT router training / gating / publishing pipeline (dormant, per the
  scope decision).
- Upgrading non-English cloud routing from the two-step
  (`try_assistant_route`) to a one-shot localised route-or-answer. It's a nicer
  UX (single round-trip, answers when no skill fits) but not needed to unblock
  Italian routing; revisit only if the two-step proves weak in verification.
