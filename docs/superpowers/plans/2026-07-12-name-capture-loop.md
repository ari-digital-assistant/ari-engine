# "Remember My Name" Name-Capture Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a bare "remember my name" elicit "What's your name?" and persist the reply as the user's name, so the personalised greeting works (EN + IT).

**Architecture:** Engine recognises a bare name request, sets the existing `pending_turn` slot with a reserved sentinel id (`__ari_name_capture`) and asks the name; the next utterance is caught by the existing pending-turn block, the name is extracted from the raw text and stored as a locale-natural fact (`my name is X` / `mi chiamo X`). Android's `detectUserName` gains IT patterns and the greeting refreshes after capture.

**Tech Stack:** Rust (ari-engine, `cargo test`), Kotlin/Compose (ari-android, JUnit4).

## Global Constraints

- **No skill touched → no PR.** `ari-engine` is direct-to-main; `ari-android` is direct-to-main. The two changes are independent (engine stores the fact string; Android reads it) — no cross-repo symbol dependency, no UniFFI regen (no new FFI surface; reuses `pending_turn`/`rearm`/`facts_changed`).
- **The stored name fact MUST match `detectUserName`:** EN `my name is <Name>`, IT `mi chiamo <Name>`.
- **Locale fallback:** every locale helper uses `match locale { "it" => …, _ => <English> }` — unknown locales fall back to English, never machine-translated. IT wording is DRAFT for Keith to confirm.
- **Casing:** extract the name from the **raw** utterance (`input` in `process_input_traced`), capitalising the first letter only if the token is all-lowercase.
- **Tests assert exact values** (project rule) — exact strings/enums.
- **Rust build env (Fedora):** if `cargo test` trips on `llama-cpp-sys` bindgen, apply the `BINDGEN_EXTRA_CLANG_ARGS` workaround (see the `reference_build_env_fedora` memory).
- **Android gradle** needs `JAVA_HOME=/usr/lib/jvm/java-25-openjdk`.

## File Structure

- `ari-engine/crates/ari-engine/src/lib.rs` — helpers (Task 1) + capture/resume wiring (Task 2) + tests (both).
- `ari-android/app/src/main/java/dev/heyari/ari/ui/conversation/EmptyStateLogic.kt` — IT name patterns (Task 3).
- `ari-android/app/src/test/java/dev/heyari/ari/ui/conversation/EmptyStateLogicTest.kt` — IT detect test (Task 3).
- `ari-android/app/src/main/java/dev/heyari/ari/ui/conversation/ConversationViewModel.kt` — refresh after capture (Task 3).

---

### Task 1: Engine helpers (pure, TDD)

**Files:**
- Modify: `ari-engine/crates/ari-engine/src/lib.rs`

**Interfaces:**
- Produces: `NAME_CAPTURE_SENTINEL`, `is_bare_name_request`, `extract_name`, `name_fact_for`, `name_prompt_for`, `name_captured_ack_for`, `name_not_caught_for` (+ private `name_fillers`, `name_refusals`, `capitalize_first`).

- [ ] **Step 1: Write the failing tests**

Add to the `mod tests` block (`#[cfg(test)]`, ~line 2410) in `lib.rs`:

```rust
    #[test]
    fn bare_name_request_detected_per_locale() {
        assert!(is_bare_name_request("my name", "en"));
        assert!(!is_bare_name_request("my name is john", "en"));
        assert!(!is_bare_name_request("i like pizza", "en"));
        assert!(is_bare_name_request("il mio nome", "it"));
        assert!(!is_bare_name_request("my name", "it"));
    }

    #[test]
    fn extract_name_handles_phrasings_and_casing() {
        assert_eq!(extract_name("John", "en").as_deref(), Some("John"));
        assert_eq!(extract_name("i'm sarah", "en").as_deref(), Some("Sarah"));
        assert_eq!(extract_name("my name is bob", "en").as_deref(), Some("Bob"));
        assert_eq!(extract_name("John Smith", "en").as_deref(), Some("John"));
        assert_eq!(extract_name("sono Giovanni", "it").as_deref(), Some("Giovanni"));
        assert_eq!(extract_name("il mio nome è Anna", "it").as_deref(), Some("Anna"));
        assert_eq!(extract_name("nope", "en"), None);
        assert_eq!(extract_name("", "en"), None);
    }

    #[test]
    fn name_fact_is_locale_natural() {
        assert_eq!(name_fact_for("en", "Keith"), "my name is Keith");
        assert_eq!(name_fact_for("it", "Giovanni"), "mi chiamo Giovanni");
        assert_eq!(name_fact_for("fr", "Marie"), "my name is Marie");
    }

    #[test]
    fn name_strings_are_localised() {
        assert_eq!(name_prompt_for("en"), "What's your name?");
        assert_eq!(name_prompt_for("it"), "Come ti chiami?");
        assert_eq!(name_prompt_for("fr"), "What's your name?");
        assert_eq!(name_captured_ack_for("en", "Keith"), "Nice to meet you, Keith!");
        assert_eq!(name_captured_ack_for("it", "Giovanni"), "Piacere di conoscerti, Giovanni!");
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ari-engine && cargo test -p ari-engine bare_name_request_detected_per_locale extract_name_handles_phrasings_and_casing name_fact_is_locale_natural name_strings_are_localised 2>&1 | tail -20`
Expected: compile error — the helpers don't exist yet.

- [ ] **Step 3: Add the const + helpers**

Add the const near the other engine consts (top of the file, e.g. next to `MAX_REMEMBERED_FACTS`):

```rust
/// Reserved `pending_turn` id for the engine-internal name-capture round-trip.
const NAME_CAPTURE_SENTINEL: &str = "__ari_name_capture";
```

Add the helpers near `fact_remembered_ack_for` (~line 307):

```rust
/// True when a stripped capture payload is a bare "my name" request (no name
/// given) — the trigger to elicit the name rather than store a fact.
fn is_bare_name_request(payload: &str, locale: &str) -> bool {
    match locale {
        "it" => payload == "il mio nome",
        _ => payload == "my name",
    }
}

/// Lead-in filler words skipped before the name token, plus obvious refusals.
fn name_fillers(locale: &str) -> &'static [&'static str] {
    match locale {
        "it" => &["mi", "chiamo", "sono", "il", "mio", "nome", "è", "chiamami",
                  "puoi", "chiamarmi", "ecco", "beh"],
        _ => &["my", "name", "name's", "is", "the", "i", "i'm", "im", "am", "it",
               "it's", "its", "call", "me", "you", "can", "just", "um", "uh", "er", "well"],
    }
}

fn name_refusals(locale: &str) -> &'static [&'static str] {
    match locale {
        "it" => &["no", "niente", "nulla", "boh"],
        _ => &["no", "nope", "nah", "nothing", "none", "nevermind"],
    }
}

/// Capitalise the first letter iff the token is all-lowercase (leave names
/// that already carry uppercase, e.g. "McDonald", intact).
fn capitalize_first(s: &str) -> String {
    if s.chars().any(|c| c.is_uppercase()) {
        return s.to_string();
    }
    let mut chars = s.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

/// Extract a display-ready first name from a raw reply, or None. Skips leading
/// filler words, rejects obvious refusals, takes the first name-like token.
fn extract_name(raw_input: &str, locale: &str) -> Option<String> {
    let fillers = name_fillers(locale);
    let refusals = name_refusals(locale);
    for word in raw_input.split_whitespace() {
        let token: String = word
            .chars()
            .filter(|c| c.is_alphabetic() || *c == '-' || *c == '\'')
            .collect();
        let token = token.trim_matches(|c| c == '-' || c == '\'').to_string();
        if token.is_empty() {
            continue;
        }
        let lower = token.to_lowercase();
        if fillers.contains(&lower.as_str()) {
            continue;
        }
        if refusals.contains(&lower.as_str()) {
            return None;
        }
        if token.chars().count() > 30 {
            return None;
        }
        return Some(capitalize_first(&token));
    }
    None
}

/// The locale-natural remembered fact for a captured name. Must stay readable
/// by the frontend's `detectUserName`.
fn name_fact_for(locale: &str, name: &str) -> String {
    match locale {
        "it" => format!("mi chiamo {name}"),
        _ => format!("my name is {name}"),
    }
}

/// Prompt asking the user for their name.
fn name_prompt_for(locale: &str) -> &'static str {
    match locale {
        "it" => "Come ti chiami?",
        _ => "What's your name?",
    }
}

/// Warm ack once the name is captured.
fn name_captured_ack_for(locale: &str, name: &str) -> String {
    match locale {
        "it" => format!("Piacere di conoscerti, {name}!"),
        _ => format!("Nice to meet you, {name}!"),
    }
}

/// Spoken when the reply held no usable name (single-shot; no retry).
fn name_not_caught_for(locale: &str) -> &'static str {
    match locale {
        "it" => "Scusa, non ho capito il nome — puoi dirmelo quando vuoi con \"mi chiamo …\".",
        _ => "Sorry, I didn't catch a name — you can tell me any time by saying \"my name is …\".",
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ari-engine && cargo test -p ari-engine bare_name_request_detected_per_locale extract_name_handles_phrasings_and_casing name_fact_is_locale_natural name_strings_are_localised 2>&1 | tail -20`
Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
cd ari-engine
git add crates/ari-engine/src/lib.rs
git commit -m "feat: name-capture helpers (bare-request detect, name extract, locale strings)"
```

---

### Task 2: Wire the elicit + resume branches (round-trip)

**Files:**
- Modify: `ari-engine/crates/ari-engine/src/lib.rs`

**Interfaces:**
- Consumes: Task 1 helpers, existing `set_pending_turn`, `take_pending_turn_if_fresh`, `capture_fact`, `is_cancel_phrase`.

- [ ] **Step 1: Write the failing round-trip tests**

Add to `mod tests`:

```rust
    #[test]
    fn remember_my_name_elicits_then_captures_en() {
        let mut engine = Engine::new();
        let (resp, _) = engine.process_input_traced("remember my name");
        assert!(matches!(resp, Response::Text(ref s) if s == name_prompt_for("en")),
            "bare request must ask for the name; got {resp:?}");
        assert!(engine.has_pending_turn(), "name capture must arm a pending turn");
        assert_eq!(engine.pending_turn.lock().unwrap().as_ref().unwrap().skill_id,
            NAME_CAPTURE_SENTINEL);

        let (resp, _) = engine.process_input_traced("i'm Keith");
        assert_eq!(resp, Response::Text("Nice to meet you, Keith!".to_string()));
        assert_eq!(engine.remembered_facts(), vec!["my name is Keith".to_string()]);
        assert!(!engine.has_pending_turn(), "slot must clear after capture");
    }

    #[test]
    fn remember_my_name_elicits_then_captures_it() {
        let mut engine = Engine::new();
        engine.set_locale("it".to_string());
        let (resp, _) = engine.process_input_traced("ricorda il mio nome");
        assert!(matches!(resp, Response::Text(ref s) if s == name_prompt_for("it")));
        let (resp, _) = engine.process_input_traced("sono Giovanni");
        assert_eq!(resp, Response::Text("Piacere di conoscerti, Giovanni!".to_string()));
        assert_eq!(engine.remembered_facts(), vec!["mi chiamo Giovanni".to_string()]);
    }

    #[test]
    fn name_reply_with_no_name_apologises_once() {
        let mut engine = Engine::new();
        let _ = engine.process_input_traced("remember my name");
        let (resp, _) = engine.process_input_traced("nope");
        assert!(matches!(resp, Response::Text(ref s) if s == name_not_caught_for("en")));
        assert!(engine.remembered_facts().is_empty(), "no name fact on unusable reply");
        assert!(!engine.has_pending_turn(), "single-shot: slot cleared, no retry");
    }

    #[test]
    fn cancel_during_name_capture_escapes() {
        let mut engine = Engine::new();
        let _ = engine.process_input_traced("remember my name");
        let (resp, _) = engine.process_input_traced("cancel");
        assert!(matches!(resp, Response::Text(ref s) if s == cancel_ack_for("en")));
        assert!(engine.remembered_facts().is_empty());
        assert!(!engine.has_pending_turn());
    }

    #[test]
    fn remember_my_name_with_name_stores_directly() {
        // Regression: the non-bare form must NOT elicit — it stores as-is.
        let mut engine = Engine::new();
        let (resp, _) = engine.process_input_traced("remember my name is Dave");
        assert!(matches!(resp, Response::Text(ref s) if s == fact_remembered_ack_for("en")));
        assert!(!engine.has_pending_turn());
        assert_eq!(engine.remembered_facts(), vec!["my name is dave".to_string()]);
    }
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd ari-engine && cargo test -p ari-engine remember_my_name name_reply_with_no_name cancel_during_name_capture 2>&1 | tail -20`
Expected: FAIL — the bare request currently stores `"my name"` and returns the generic ack (no prompt, no pending turn).

- [ ] **Step 3: Add the bare-request check to the capture branch**

In `process_input_traced`, change the capture branch (currently ~lib.rs:1097-1103):

```rust
            if let Some(fact) = remembered_fact_capture(&normalized, &self.ctx.locale) {
                self.capture_fact(fact);
                return (
                    Response::Text(fact_remembered_ack_for(&self.ctx.locale).to_string()),
                    None,
                );
            }
```

to:

```rust
            if let Some(fact) = remembered_fact_capture(&normalized, &self.ctx.locale) {
                if is_bare_name_request(fact, &self.ctx.locale) {
                    self.set_pending_turn(NAME_CAPTURE_SENTINEL, String::new());
                    return (
                        Response::Text(name_prompt_for(&self.ctx.locale).to_string()),
                        None,
                    );
                }
                self.capture_fact(fact);
                return (
                    Response::Text(fact_remembered_ack_for(&self.ctx.locale).to_string()),
                    None,
                );
            }
```

- [ ] **Step 4: Add the sentinel branch to the pending-turn resume block**

In the `if let Some(pending) = self.take_pending_turn_if_fresh() { … }` block, immediately **after** the `is_cancel_phrase` early-return and **before** the `if let Some(skill) = self.skills.iter()…` lookup (~lib.rs:1130), insert:

```rust
            // Engine-internal name capture: the reply is the user's name.
            if pending.skill_id == NAME_CAPTURE_SENTINEL {
                return match extract_name(input, &self.ctx.locale) {
                    Some(name) => {
                        self.capture_fact(&name_fact_for(&self.ctx.locale, &name));
                        (
                            Response::Text(name_captured_ack_for(&self.ctx.locale, &name)),
                            None,
                        )
                    }
                    None => (
                        Response::Text(name_not_caught_for(&self.ctx.locale).to_string()),
                        None,
                    ),
                };
            }
```

(`input` is the raw utterance param of `process_input_traced` — used so casing survives.)

- [ ] **Step 5: Run the new tests + the full engine suite**

Run: `cd ari-engine && cargo test -p ari-engine 2>&1 | tail -20`
Expected: all tests pass (the 5 new round-trip tests, Task 1's 4, and every pre-existing test — including the existing pending-turn/multi-turn tests, which must be unaffected).

- [ ] **Step 6: Commit**

```bash
cd ari-engine
git add crates/ari-engine/src/lib.rs
git commit -m "feat: elicit and capture the user's name on a bare 'remember my name'"
```

---

### Task 3: Android — IT name detection + greeting refresh

**Files:**
- Modify: `ari-android/app/src/main/java/dev/heyari/ari/ui/conversation/EmptyStateLogic.kt`
- Modify: `ari-android/app/src/test/java/dev/heyari/ari/ui/conversation/EmptyStateLogicTest.kt`
- Modify: `ari-android/app/src/main/java/dev/heyari/ari/ui/conversation/ConversationViewModel.kt`

- [ ] **Step 1: Write the failing IT detection test**

Add to `EmptyStateLogicTest.kt`:

```kotlin
    @Test
    fun `detectUserName reads Italian phrasings`() {
        assertEquals("Giovanni", detectUserName(listOf("mi chiamo Giovanni")))
        assertEquals("Anna", detectUserName(listOf("il mio nome è Anna")))
        assertEquals("Luca", detectUserName(listOf("chiamami Luca")))
        assertEquals("Keith", detectUserName(listOf("my name is Keith")))  // EN still works
        assertNull(detectUserName(listOf("mi piace la pizza")))
    }
```

(If `assertEquals`/`assertNull`/`Test` aren't imported in the file yet, add `import org.junit.Assert.assertEquals`, `import org.junit.Assert.assertNull`, `import org.junit.Test`.)

- [ ] **Step 2: Run to verify it fails**

Run: `cd ari-android && JAVA_HOME=/usr/lib/jvm/java-25-openjdk ./gradlew :app:testDebugUnitTest --tests "dev.heyari.ari.ui.conversation.EmptyStateLogicTest" 2>&1 | tail -15`
Expected: FAIL — the IT phrasings return null.

- [ ] **Step 3: Add the IT pattern to `detectUserName`**

In `EmptyStateLogic.kt`, extend `NAME_PATTERNS`:

```kotlin
private val NAME_PATTERNS = listOf(
    Regex("""(?:the user'?s name is|my name is|call me)\s+([\p{L}][\p{L}\-']{1,30})""", RegexOption.IGNORE_CASE),
    Regex("""(?:mi chiamo|il mio nome è|chiamami)\s+([\p{L}][\p{L}\-']{1,30})""", RegexOption.IGNORE_CASE),
)
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd ari-android && JAVA_HOME=/usr/lib/jvm/java-25-openjdk ./gradlew :app:testDebugUnitTest --tests "dev.heyari.ari.ui.conversation.EmptyStateLogicTest" 2>&1 | tail -8`
Expected: PASS.

- [ ] **Step 5: Refresh the greeting immediately after a name is captured**

In `ConversationViewModel.onTextSubmitted`, the facts-persist block reads:

```kotlin
                if (shouldPersistFacts(response)) {
                    settingsRepository.setRememberedFacts(
                        engineHolder.peek()?.rememberedFacts() ?: emptyList()
                    )
                }
```

Add `refreshEmptyState()` after the persist, so the greeting is ready the instant the user returns to an empty view:

```kotlin
                if (shouldPersistFacts(response)) {
                    settingsRepository.setRememberedFacts(
                        engineHolder.peek()?.rememberedFacts() ?: emptyList()
                    )
                    refreshEmptyState()
                }
```

- [ ] **Step 6: Compile + run the full unit suite**

Run: `cd ari-android && JAVA_HOME=/usr/lib/jvm/java-25-openjdk ./gradlew :app:compileDebugKotlin :app:testDebugUnitTest 2>&1 | tail -6`
Expected: BUILD SUCCESSFUL; all unit tests pass.

- [ ] **Step 7: Build + install + device sign-off**

```bash
cd ari-android && JAVA_HOME=/usr/lib/jvm/java-25-openjdk ./gradlew :app:assembleDebug
ANDROID_HOME=/home/keith/Android/Sdk /home/keith/Android/Sdk/platform-tools/adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Device check (Keith): on an empty conversation, tap **"Remember my name"** (or say it), answer with a name, confirm Ari acks warmly and — on the next empty view (background→foreground, or /reset if it refreshes) — the greeting reads "Good morning, <Name>". Repeat once in Italian if convenient (`Ricorda il mio nome` → `mi chiamo …`).

- [ ] **Step 8: Commit**

```bash
cd ari-android
git add app/src/main/java/dev/heyari/ari/ui/conversation/EmptyStateLogic.kt app/src/test/java/dev/heyari/ari/ui/conversation/EmptyStateLogicTest.kt app/src/main/java/dev/heyari/ari/ui/conversation/ConversationViewModel.kt
git commit -m "feat: read Italian name phrasings; refresh greeting after name capture"
```

---

## Self-Review

**Spec coverage:**
- Bare-name trigger, locale-aware → Task 1 `is_bare_name_request` + Task 2 capture branch. ✓
- Sentinel pending-turn elicit + resume, casing from raw → Task 2 Steps 3-4. ✓
- Single-shot on unusable reply; cancel escape → Task 2 tests + resume branch (cancel via existing code). ✓
- Locale-natural fact (`my name is`/`mi chiamo`) → Task 1 `name_fact_for`, asserted round-trip Task 2. ✓
- `detectUserName` IT extension → Task 3 Steps 1-3. ✓
- Greeting refresh after capture → Task 3 Step 5. ✓
- Localized prompt/ack/not-caught (EN + IT DRAFT) → Task 1 `name_*_for`. ✓
- No skill / no FFI regen → File Structure (engine internals + android only). ✓

**Placeholder scan:** No TBD/TODO; every step has complete code. IT strings are concrete (DRAFT for wording confirmation, not placeholders). ✓

**Type consistency:** `NAME_CAPTURE_SENTINEL` (Task 1) is set in Task 2 Step 3 and matched in Step 4. `extract_name`/`name_fact_for`/`name_captured_ack_for`/`name_not_caught_for`/`name_prompt_for` signatures (Task 1) match their Task 2 call sites. `set_pending_turn(&str, String)` matches `set_pending_turn(NAME_CAPTURE_SENTINEL, String::new())`. Stored facts (`"my name is Keith"` / `"mi chiamo Giovanni"`) match `detectUserName`'s regexes (Task 3). ✓
