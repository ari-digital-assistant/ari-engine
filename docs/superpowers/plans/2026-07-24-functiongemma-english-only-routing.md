# FunctionGemma English-only Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restrict the on-device FunctionGemma router to English, and make the cloud LLM the sole non-keyword router for every other language (offline = answer only, no routing).

**Architecture:** Two one-function gates in the engine (`ari-engine/crates/ari-engine/src/lib.rs`) plus one gate in the Android delivery policy (`RouterPolicy`). No new subsystems, no new dependencies. The LLM routing this enables is already wired — the change is removing FunctionGemma from the non-English path and stopping the Italian model from being delivered.

**Tech Stack:** Rust (engine, `cargo test`), Kotlin/Android (`./gradlew` unit tests, plain JUnit4).

## Global Constraints

- FunctionGemma is **English-only** — the core invariant this plan establishes. It routes non-English confidently but wrongly at 270M.
- Routing via the assistant happens **only when a cloud assistant is present**, for every locale. Offline (on-device LLM only) never routes — it answers directly.
- The on-device Gemma LLM is **never** asked to route (it's ~22s, catalogue prefill dominates).
- **Do not touch** the FunctionGemma training / gating / publishing workflows (Modal, GitHub Actions, floating `functiongemma-<locale>-latest` releases). Left dormant, not deleted.
- Do not introduce new translated user-facing strings (no translations for languages the dev doesn't know). This plan adds none.
- Engine and Android changes go **direct-to-main** (no PR required — PRs are only for `ari-skills/skills/`).
- Tests assert **exact** values and real behaviour — no weak thresholds.
- Engine crate name is `ari-engine`; run tests with `cargo test -p ari-engine <filter>`.
- Android gradle on this machine needs `JAVA_HOME=/usr/lib/jvm/java-25-openjdk`.
- Every commit message ends with:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  ```

## File Structure

- `ari-engine/crates/ari-engine/src/lib.rs` — both engine gates and their tests (existing large file; edits are localised to two functions + two tests).
- `ari-android/app/src/main/java/dev/heyari/ari/router/RouterPolicy.kt` — add a pure `routerSupportsLocale` predicate and short-circuit `shouldHaveModel`.
- `ari-android/app/src/test/java/dev/heyari/ari/router/RouterPolicyTest.kt` — **new** plain-JUnit test for the pure predicate (mirrors `RouterAvailabilityTest`).

---

### Task 1: Engine — FunctionGemma is English-only

**Files:**
- Modify: `ari-engine/crates/ari-engine/src/lib.rs` — `router_for_active_locale` (~L940-954)
- Test: `ari-engine/crates/ari-engine/src/lib.rs` — rewrite `router_dispatches_for_a_non_english_locale_when_its_model_is_loaded` (~L3044-3061)

**Interfaces:**
- Consumes: `Engine::set_router(Some((Box<dyn SkillRouter>, String)))`, `Engine::set_locale(String)`, `process_input_traced(&str) -> (Response, Option<DebugTrace>)`, test helpers `CatalogCapturingRouter { seen: Arc<Mutex<Vec<String>>> }` (records the catalogue and returns `RouteResult::NoMatch`), `unreachable_by_keyword(id, response)`, `fallback_response_for(locale)`.
- Produces: `router_for_active_locale` returns `None` for any non-`en` active locale. Behaviour for `en` is unchanged. Debug/eval paths (`route_decision`, `route_raw`, `debug_route`) are unaffected — they read `self.router` directly.

- [ ] **Step 1: Rewrite the test to assert the router is skipped for a non-English locale**

Replace the whole `router_dispatches_for_a_non_english_locale_when_its_model_is_loaded` test with this. It loads an Italian-locale model that *matches* the active locale — the strongest case — and asserts FunctionGemma is never consulted:

```rust
    #[test]
    fn router_is_english_only_even_with_a_matching_locale_model() {
        use std::sync::{Arc, Mutex};

        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut engine = Engine::new();
        engine.set_locale("it".to_string());
        engine.register_skill(Box::new(unreachable_by_keyword("meteo", "Sole.")));
        // An Italian-locale model is loaded and matches the active locale...
        engine.set_router(Some((
            Box::new(CatalogCapturingRouter { seen: seen.clone() }),
            "it".to_string(),
        )));

        let (response, _) = engine.process_input_traced("che tempo fa");

        // ...but FunctionGemma is English-only, so it is never consulted.
        assert!(
            seen.lock().unwrap().is_empty(),
            "FunctionGemma must not be consulted for a non-English locale, got {:?}",
            seen.lock().unwrap()
        );
        match response {
            Response::Text(t) => assert_eq!(t, fallback_response_for("it")),
            other => panic!("expected the Italian fallback, got {other:?}"),
        }
    }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p ari-engine router_is_english_only_even_with_a_matching_locale_model`
Expected: FAIL — the router still runs for `it`, so `seen` is non-empty (catalogue captured) and/or the response isn't the Italian fallback.

- [ ] **Step 3: Add the English-only gate to `router_for_active_locale`**

Replace the function (keep the existing doc comment, prepend the new rationale) so it reads:

```rust
    /// The on-device router, but only when the active language is English.
    ///
    /// FunctionGemma is English-only: at 270M it routes other languages
    /// confidently but wrongly, so the engine never consults it outside `en`.
    /// Non-English routing is the cloud LLM's job (see `uses_assistant_routing`).
    /// The debug/eval entry points read `self.router` directly and are
    /// deliberately exempt, so the eval harness can still score any model.
    ///
    /// The `router_locale == ctx.locale` check stays as a second guard: the
    /// host swaps models asynchronously on a language switch, and routing one
    /// language through another's model would be confident nonsense.
    fn router_for_active_locale(&self) -> Option<&dyn SkillRouter> {
        if self.ctx.locale != "en" {
            return None;
        }
        let router = self.router.as_ref()?;
        if self.router_locale.as_deref() != Some(self.ctx.locale.as_str()) {
            return None;
        }
        Some(router.as_ref())
    }
```

- [ ] **Step 4: Run the rewritten test plus the sibling router tests to verify they pass**

Run: `cargo test -p ari-engine router_`
Expected: PASS — including `router_is_english_only_even_with_a_matching_locale_model`, `router_is_skipped_when_its_model_is_for_another_locale`, `router_outranks_a_cloud_assistant_when_a_matching_model_is_loaded` (English — unchanged), and `unloading_the_router_clears_its_locale`.

- [ ] **Step 5: Commit**

```bash
cd ari-engine
git add crates/ari-engine/src/lib.rs
git commit -m "feat(router): FunctionGemma is English-only

The 270M on-device router routes non-English confidently but wrongly, so
the engine no longer consults it outside en. router_for_active_locale
returns None for any non-English active locale; the cloud LLM handles
routing for other languages. Debug/eval paths are unaffected.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Engine — routing is cloud-only, offline answers only

**Files:**
- Modify: `ari-engine/crates/ari-engine/src/lib.rs` — `uses_assistant_routing` (~L2005-2019) and its one call site (~L1564-1567)
- Test: `ari-engine/crates/ari-engine/src/lib.rs` — rewrite `routing_backend_choice` (~L3943-3952)

**Interfaces:**
- Consumes: `has_cloud_assistant: bool` (already computed at the call site as `matches!(&self.active_assistant, Some(ActiveAssistant::Api { .. }))`).
- Produces: `uses_assistant_routing(has_cloud_assistant: bool) -> bool` — single-argument now (locale dropped). Returns `has_cloud_assistant`. The English one-shot / non-English two-step branches below the call site are unchanged; they simply no longer run when there is no cloud assistant.

- [ ] **Step 1: Rewrite the `routing_backend_choice` test to the new one-argument contract**

Replace the whole test:

```rust
    #[test]
    fn routing_backend_choice() {
        // Routing via the assistant happens only when a cloud assistant is
        // present — for every locale. Offline (on-device LLM only), a
        // non-keyword request gets a direct answer, never a routing round-trip
        // through the slow on-device LLM.
        assert!(uses_assistant_routing(true));
        assert!(!uses_assistant_routing(false));
    }
```

- [ ] **Step 2: Run the test to verify it fails (compile error)**

Run: `cargo test -p ari-engine routing_backend_choice`
Expected: FAIL — compile error, `uses_assistant_routing` still takes two arguments (`locale`, `has_cloud_assistant`).

- [ ] **Step 3: Collapse `uses_assistant_routing` to depend only on the cloud assistant**

Replace the function and its doc comment:

```rust
/// Whether the assistant is asked to ROUTE what the router left behind, rather
/// than only to answer it. Consulted AFTER the router has had its turn (see
/// `process_input_traced`), so this decides the second tier, not the first.
///
/// Routing is cloud-only. A cloud assistant arbitrates well — it tells a skill
/// request from a general question reliably, in any language. The on-device
/// Gemma cannot (it's ~22s to route, catalogue prefill dominates, and at its
/// size the picks aren't reliable), so when there's no cloud assistant nothing
/// is asked to route: the request falls to the answer path and is answered
/// directly. English still gets FunctionGemma as its offline routing tier;
/// other languages get keyword matching and direct answers offline.
fn uses_assistant_routing(has_cloud_assistant: bool) -> bool {
    has_cloud_assistant
}
```

- [ ] **Step 4: Update the call site**

Find (~L1564-1567):

```rust
        let has_cloud_assistant =
            matches!(&self.active_assistant, Some(ActiveAssistant::Api { .. }));
        let use_assistant_routing =
            uses_assistant_routing(&self.ctx.locale, has_cloud_assistant);
```

Replace the last statement so it reads:

```rust
        let has_cloud_assistant =
            matches!(&self.active_assistant, Some(ActiveAssistant::Api { .. }));
        let use_assistant_routing = uses_assistant_routing(has_cloud_assistant);
```

- [ ] **Step 5: Run the test to verify it passes, then the whole engine suite**

Run: `cargo test -p ari-engine routing_backend_choice`
Expected: PASS.

Run: `cargo test -p ari-engine`
Expected: PASS — the full engine suite is green (catches any other caller or fallout).

- [ ] **Step 6: Commit**

```bash
cd ari-engine
git add crates/ari-engine/src/lib.rs
git commit -m "feat(router): assistant routing is cloud-only

uses_assistant_routing now depends solely on whether a cloud assistant is
present, dropping the old 'non-English always routes' clause. Offline, a
non-keyword request is answered directly instead of routed through the slow
on-device LLM. Non-English keeps its cloud two-step; English keeps its
cloud one-shot.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Android — stop delivering the non-English router model

**Files:**
- Modify: `ari-android/app/src/main/java/dev/heyari/ari/router/RouterPolicy.kt` — add a `companion object` with `routerSupportsLocale`, short-circuit `shouldHaveModel`
- Create: `ari-android/app/src/test/java/dev/heyari/ari/router/RouterPolicyTest.kt`

**Interfaces:**
- Consumes: nothing new. `shouldHaveModel(locale)` keeps its signature and existing callers (`requiredFromState`, onboarding wizard).
- Produces: `RouterPolicy.routerSupportsLocale(locale: String): Boolean` — pure, static, English-only. `shouldHaveModel` returns `false` for any non-English locale without touching `downloadManager` or `availability` (short-circuit), which makes `reconcile` compute `required = false` for non-English → it disables the router, unloads it, and deletes any stale on-disk model via the existing `else` branch.

- [ ] **Step 1: Write the failing test**

Create `ari-android/app/src/test/java/dev/heyari/ari/router/RouterPolicyTest.kt`:

```kotlin
package dev.heyari.ari.router

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The router is English-only. FunctionGemma routes non-English confidently
 * but wrongly at 270M, so no other language should ever want a model on disk
 * — [RouterPolicy.routerSupportsLocale] is the gate every model decision
 * passes through before any download/availability check.
 */
class RouterPolicyTest {

    @Test
    fun englishIsSupported() {
        assertTrue(RouterPolicy.routerSupportsLocale("en"))
    }

    @Test
    fun italianIsNotSupported() {
        assertFalse(RouterPolicy.routerSupportsLocale("it"))
    }

    @Test
    fun otherLanguagesAreNotSupported() {
        assertFalse(RouterPolicy.routerSupportsLocale("es"))
        assertFalse(RouterPolicy.routerSupportsLocale("fr"))
        assertFalse(RouterPolicy.routerSupportsLocale("de"))
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `JAVA_HOME=/usr/lib/jvm/java-25-openjdk ./gradlew :app:testDebugUnitTest --tests "dev.heyari.ari.router.RouterPolicyTest"`
Expected: FAIL — compile error / unresolved reference, `RouterPolicy.routerSupportsLocale` does not exist yet.

- [ ] **Step 3: Add the pure predicate and short-circuit `shouldHaveModel`**

In `RouterPolicy.kt`, change `shouldHaveModel` and add a `companion object` at the end of the class body (before the final closing brace of the class). First update the KDoc + signature of `shouldHaveModel`:

```kotlin
    /**
     * Whether [locale] should have a router model on disk — an install
     * already there, or one [RouterAvailability] says is published.
     *
     * The router is English-only ([routerSupportsLocale]): non-English
     * short-circuits to `false` here without a download or availability
     * check, which is what makes `reconcile` tear down any stale model left
     * from before this became English-only.
     *
     * Takes the locale rather than reading the active one so the onboarding
     * wizard can ask about the language being picked, which isn't active yet.
     *
     * On-disk outranks the probe outright. The probe answers "should I
     * download?", never "should I delete?": the floating release it reads
     * deletes and re-uploads its manifest on every republish, so that URL
     * genuinely 404s for a few seconds every night, forever. A device that
     * probes in that window caches "absent" for a day, and acting on it would
     * cost the user their routing tier plus a 253 MB re-download — on a
     * nightly schedule. Keeping a file that's already there costs nothing and
     * it is still this locale's own model, so no cross-locale rule is in play.
     */
    suspend fun shouldHaveModel(locale: String): Boolean =
        routerSupportsLocale(locale) &&
            (downloadManager.isDownloaded(locale) || availability.isAvailable(locale))
```

Then add the companion object as the last member of the class:

```kotlin
    companion object {
        /**
         * Whether the FunctionGemma router covers [locale] at all. It is
         * English-only — at 270M it routes other languages confidently but
         * wrongly — so this is the gate every model decision passes through
         * before any download or availability check. Non-English languages
         * route via the cloud LLM instead (handled in the engine), so they
         * never need a model on disk.
         */
        fun routerSupportsLocale(locale: String): Boolean = locale == "en"
    }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `JAVA_HOME=/usr/lib/jvm/java-25-openjdk ./gradlew :app:testDebugUnitTest --tests "dev.heyari.ari.router.RouterPolicyTest"`
Expected: PASS.

- [ ] **Step 5: Run the full router test package to check nothing else broke**

Run: `JAVA_HOME=/usr/lib/jvm/java-25-openjdk ./gradlew :app:testDebugUnitTest --tests "dev.heyari.ari.router.*"`
Expected: PASS — `RouterPolicyTest`, `RouterModelTest`, `RouterAvailabilityTest`, `RouterLegacyMigrationTest` all green.

- [ ] **Step 6: Commit**

```bash
cd ari-android
git add app/src/main/java/dev/heyari/ari/router/RouterPolicy.kt \
        app/src/test/java/dev/heyari/ari/router/RouterPolicyTest.kt
git commit -m "feat(router): deliver the router model for English only

The FunctionGemma router is English-only now, so RouterPolicy.shouldHaveModel
short-circuits to false for every other locale (via the new pure
routerSupportsLocale gate) without a download or availability probe. reconcile
then tears down any stale non-English model already on disk. The training and
publishing pipeline is left dormant.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Verify on the emulator + refresh memory/docs

This task has no code commit of its own — it confirms the behaviour end-to-end and cleans up stale notes.

**Files:**
- No repo files. Session-level memory under `~/.claude/.../memory/` (updated by the assistant, not a subagent).

- [ ] **Step 1: Build + install the app on the emulator**

Run: `JAVA_HOME=/usr/lib/jvm/java-25-openjdk ./gradlew :app:installDebug` (target `emulator-5554`; do not use Keith's physical Pixel).

- [ ] **Step 2: Verify Italian + cloud assistant routes to the time skill**

Configure a cloud assistant, set locale to Italian, and speak/type the exact phrase:

```
è ora di una birra?
```

Expected: routes to the time skill (a time answer), not a generic LLM ramble. Also confirm the English idiom still routes:

```
is it beer o'clock yet?
```

- [ ] **Step 3: Verify Italian offline answers, doesn't route**

Remove/disable the cloud assistant (on-device LLM only), locale Italian, same phrase:

```
è ora di una birra?
```

Expected: a direct answer, no skill routing. Confirm via logs that the router was not consulted and no assistant *routing* call was made.

- [ ] **Step 4: Confirm the Italian router model is gone**

With locale Italian, confirm the router directory holds no `it` model (it was deleted by `reconcile`) and no download was kicked. Check logs for the reconcile disabling the router.

- [ ] **Step 5: If Italian cloud routing is weak**

If Step 2 fails to route reliably, the fix is prompt-tuning `build_assistant_routing_prompt` (the non-English two-step prompt) — a reactive follow-up, out of scope for the tasks above. Note the failing phrases for a follow-up pass; do not silently expand scope.

- [ ] **Step 6: Refresh stale memory notes**

Update these memory files to reflect English-only FunctionGemma + cloud-only non-English routing (they currently claim "router-first, per locale" / "used at inference for IT"):
- `project_routing_arbitration.md`
- `project_functiongemma.md`
- `project_per_language_router.md`
- `project_plan4_router_delivery.md`

---

## Self-Review

**Spec coverage:**
- Change 1 (FunctionGemma English-only) → Task 1. ✅
- Change 2 (routing cloud-only, offline answers only) → Task 2. ✅
- Change 3 (Android stop IT delivery) → Task 3. ✅
- Spec test list: engine dispatch test flip → Task 1 Step 1; `routing_backend_choice` flip → Task 2 Step 1; "en model, it locale, router never consulted" → covered by the existing `router_is_skipped_when_its_model_is_for_another_locale` (unchanged) plus the stronger `router_is_english_only_even_with_a_matching_locale_model` in Task 1. Android `shouldHaveModel` non-English false → Task 3 `RouterPolicyTest`. ✅
- Spec verification (device, `è ora di una birra?` / `is it beer o'clock yet?`, offline = answer) → Task 4. ✅
- Out-of-scope (dormant pipeline, no one-shot upgrade) → honoured; Task 4 Step 5 keeps prompt-tuning explicitly reactive. ✅

**Placeholder scan:** No TBD/TODO/"handle edge cases". Every code step has literal code. Task 4 is intentionally manual (device verification) with concrete phrases and expected outcomes. ✅

**Type consistency:** `routerSupportsLocale(locale: String): Boolean` used identically in Task 3 impl and test. `uses_assistant_routing(has_cloud_assistant: bool) -> bool` used identically in Task 2 impl, call site, and test. `CatalogCapturingRouter { seen }` matches its definition (records catalogue, returns `NoMatch`) — asserting `seen.is_empty()` correctly proves "never called". ✅
