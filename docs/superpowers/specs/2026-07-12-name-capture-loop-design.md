# "Remember My Name" Name-Capture Loop — Design

**Date:** 2026-07-12
**Components:** ari-engine (Rust core) + a small ari-android nudge. No skill → no PR.
**Status:** Approved, pending implementation plan

## Problem

Tapping the empty-state **"Remember my name"** chip (or saying "remember my name")
should make Ari ask *"What's your name?"* and store the answer as the user's name,
so the personalised greeting ("Good morning, Keith") kicks in. Today it does not.

The chip submits the literal string `"Remember my name"`. The engine's remembered-
facts intercept strips the `"remember "` prefix and stores the **useless fact
`"my name"`**, then returns the generic ack "Got it — I'll remember that." The
Android greeting derives the name by regex-scanning remembered facts
(`detectUserName`, matching `my name is X` / `call me X`), which `"my name"` never
matches — so the greeting stays anonymous. No name is ever elicited or stored.

(Note: `"remember my name is John"` — the *non-bare* form — already works today; the
bug is only the **bare** request with no name.)

## Approach

Make the engine recognise a **bare name request** and run a two-turn elicitation,
reusing the engine's existing multi-turn `pending_turn` slot with a **reserved
sentinel id** (`__ari_name_capture`).

- **Turn 1 (elicit):** in the remembered-facts capture branch, if the stripped
  payload is a bare name request (`"my name"` in EN, `"il mio nome"` in IT), set
  `pending_turn(__ari_name_capture, "")` and return the localized prompt
  *"What's your name?"*. The mic re-arms **for free** — the FFI `rearm` signal is
  already `engine.has_pending_turn()`.
- **Turn 2 (resume):** the existing pending-turn block already intercepts the next
  utterance. Before its skill lookup, add a sentinel branch: extract the name from
  the reply, `capture_fact(<locale-natural name fact>)`, and return a warm ack.
  Verbal "cancel" already escapes (existing code); an unusable reply apologises
  once and stops (the slot is already consumed → single-shot).

**Why the sentinel-in-`pending_turn` approach:** remembered-facts handling is
engine-internal inline code, *not* a `Skill`, and the multi-turn machinery is
skill-keyed (dispatches via `skills.iter().find(|s| s.id() == pending.skill_id)`).
Reusing the slot with a reserved id gives us the 60s TTL, verbal-cancel escape, and
mic re-arm with **no new parallel state and no FFI changes** — the resume branch
just checks the sentinel before the skill lookup.

### Rejected alternatives

- **A real name-capture built-in `Skill`** emitting `await_reply` / `execute_reply`.
  Cleanest *conceptually*, but built-in skills live in a separate crate and can't
  reach the engine-internal facts store (`capture_fact` is `pub(crate)`), and the
  bare request is already intercepted before routing — delegating to a skill means
  new engine↔skill plumbing for no gain. Rejected.
- **A separate `pending_name: AtomicBool` on `Engine`.** A second, parallel pending
  mechanism next to `pending_turn`, needing its own TTL/cancel/rearm wiring
  (including an FFI change to OR it into `rearm`). The sentinel reuses all of that.
  Rejected.
- **Store the name as EN-canonical `"my name is X"` for all locales.** Minimal, and
  the greeting would work — but an IT user asking "cosa sai di me" would see an
  English-structured line in their facts. Keith chose **proper localized** instead
  (below).

## Decisions (agreed)

1. **Sentinel `pending_turn`** (`__ari_name_capture`) reuses the existing multi-turn
   slot — no new state, no FFI change.
2. **Bare-name trigger is locale-aware:** payload `"my name"` (EN) / `"il mio nome"`
   (IT). Everything else (real facts, `"remember my name is John"`) is unchanged.
3. **Single-shot on an unusable reply** — apologise once and stop (no retry loop).
   Verbal "cancel" escapes via existing code.
4. **Preserve name casing** — extract the name from the *raw* utterance (confirmed
   available in `process_input_traced`), so the greeting reads "John" not "john";
   capitalise the first letter if the extracted token is all-lowercase.
5. **Proper localized name fact + `detectUserName` extension.** Store the locale-
   natural fact (`my name is X` in EN, `mi chiamo X` in IT) and extend
   `detectUserName` to also match IT phrasings (`mi chiamo` / `il mio nome è` /
   `chiamami`), so the greeting *and* the recall list read naturally in both.
6. **Scope:** `ari-engine` (direct-to-main) + a tiny `ari-android` change. No skill.

## Components

### Changed: `ari-engine/crates/ari-engine/src/lib.rs`

New helpers (near the other locale/memory helpers):

- `const NAME_CAPTURE_SENTINEL: &str = "__ari_name_capture";`
- `fn is_bare_name_request(payload: &str, locale: &str) -> bool` — `payload == "my
  name"` (default/EN) / `"il mio nome"` (it).
- `fn extract_name(raw_input: &str, locale: &str) -> Option<String>` — strip a
  leading filler phrase (locale-aware list: EN `my name is` / `i'm` / `i am` /
  `it's` / `call me` / `just` …; IT `mi chiamo` / `sono` / `il mio nome è` /
  `chiamami` …), case-insensitively; take the first token matching
  `[\p{L}][\p{L}\-']{1,30}`; preserve casing, capitalising the first letter if the
  token is all-lowercase. Returns `None` if no token found.
- `fn name_fact_for(locale: &str, name: &str) -> String` — `"my name is {name}"` /
  `"mi chiamo {name}"`.
- `fn name_prompt_for(locale) -> &'static str` — "What's your name?" / "Come ti
  chiami?".
- `fn name_captured_ack_for(locale, name) -> String` — "Nice to meet you, {name}!" /
  "Piacere di conoscerti, {name}!".
- `fn name_not_caught_for(locale) -> &'static str` — "Sorry, I didn't catch a name —
  you can tell me any time by saying \"my name is …\"." / IT equivalent with "mi
  chiamo …".

**Capture branch** (currently ~lib.rs:1097-1103) — check the bare request first:

```rust
if let Some(fact) = remembered_fact_capture(&normalized, &self.ctx.locale) {
    if is_bare_name_request(fact, &self.ctx.locale) {
        self.set_pending_turn(NAME_CAPTURE_SENTINEL, "");
        return (Response::Text(name_prompt_for(&self.ctx.locale).to_string()), None);
    }
    self.capture_fact(fact);
    return (Response::Text(fact_remembered_ack_for(&self.ctx.locale).to_string()), None);
}
```

**Resume branch** (inside the existing `take_pending_turn_if_fresh` block, after the
cancel check, before the skill lookup, ~lib.rs:1130) — `input` is the raw utterance:

```rust
if pending.skill_id == NAME_CAPTURE_SENTINEL {
    return match extract_name(input, &self.ctx.locale) {
        Some(name) => {
            self.capture_fact(&name_fact_for(&self.ctx.locale, &name));
            (Response::Text(name_captured_ack_for(&self.ctx.locale, &name)), None)
        }
        None => (Response::Text(name_not_caught_for(&self.ctx.locale).to_string()), None),
    };
}
```

`capture_fact` sets `facts_changed`, which the FFI already surfaces so Android
persists.

### Changed: `ari-android/.../ui/conversation/EmptyStateLogic.kt`

Extend `detectUserName`'s pattern list with IT phrasings:

```kotlin
private val NAME_PATTERNS = listOf(
    Regex("""(?:the user'?s name is|my name is|call me)\s+([\p{L}][\p{L}\-']{1,30})""", RegexOption.IGNORE_CASE),
    Regex("""(?:mi chiamo|il mio nome è|chiamami)\s+([\p{L}][\p{L}\-']{1,30})""", RegexOption.IGNORE_CASE),
)
```

### Changed: `ari-android/.../ui/conversation/ConversationViewModel.kt`

After a facts-changing turn persists (the `shouldPersistFacts` block in
`onTextSubmitted`, ~334-338), call `refreshEmptyState()` so the greeting is ready
the instant the user returns to an empty view. (Resume already refreshes at ~679;
this is for immediacy within the same session.)

### Locale strings

The IT chip string already exists (`empty_chip_remember_name` = "Ricorda il mio
nome" → payload "il mio nome" ✓). New engine strings carry EN + IT inline (matching
`fact_remembered_ack_for`); IT wording is DRAFT for Keith to confirm.

## Data flow

```
TURN 1 — chip taps or "remember my name" (bare)
  processInput("Remember my name")
    → normalize → "remember my name"
    → capture branch: remembered_fact_capture → "my name"
    → is_bare_name_request("my name","en") == true
    → set_pending_turn(__ari_name_capture, "")   → has_pending_turn() → rearm=true
    → Response::Text("What's your name?")         → mic re-arms (voice) / prompt shown (text)

TURN 2 — user replies "I'm Keith" (raw) / "Sono Giovanni"
  processInput("I'm Keith")
    → memory intercepts: no match
    → take_pending_turn_if_fresh() → sentinel
    → not a cancel phrase
    → extract_name("I'm Keith","en") → "Keith"
    → capture_fact("my name is Keith")            → facts_changed → Android persists
    → Response::Text("Nice to meet you, Keith!")
  → next empty view: detectUserName(facts) → "Keith" → "Good morning, Keith"
```

## Edge cases

- **Unusable reply** ("I'd rather not", STT junk) → `extract_name` returns `None` →
  apologise once; slot already consumed → no loop.
- **Verbal cancel** mid-loop → existing `is_cancel_phrase` escape returns the cancel
  ack; slot consumed.
- **`"remember my name is John"`** → payload `"my name is john"` → not bare → stored
  directly (unchanged behaviour).
- **Reply with filler/trailing words** ("I'm Sarah by the way") → lead-in stripped,
  first token `Sarah` taken.
- **Lowercase name** ("bob") → stored/greeted as "Bob" (first-letter capitalised).
- **TTL** — the reply must arrive within the existing 60s `PENDING_TURN_TTL`; after
  that the slot is stale and the utterance routes normally.
- **Changing an existing name** is out of scope: the chip only appears when no name
  is set; a voice re-trigger would append a second name fact and `detectUserName`
  returns the first. Not handled here.

## Testing

Rust unit tests (ari-engine) are the core — assert exact values:

- `is_bare_name_request`: `("my name","en")` true; `("my name is john","en")` false;
  `("il mio nome","it")` true; `("i like pizza","en")` false.
- `extract_name`: `"John"→"John"`, `"i'm sarah"→"Sarah"`, `"my name is bob"→"Bob"`,
  `"John Smith"→"John"`, `"sono Giovanni"("it")→"Giovanni"`, `"nope"`/empty→`None`.
- `name_fact_for`: `("en","Keith")→"my name is Keith"`; `("it","Giovanni")→"mi chiamo
  Giovanni"`.
- **Round-trip** on an `Engine`: `process_input("remember my name")` returns the
  prompt and sets a pending turn; a follow-up `process_input("I'm Keith")` returns
  the ack and leaves `remembered_facts()` containing exactly `"my name is Keith"`.
  IT variant: `"ricorda il mio nome"` → `"sono Giovanni"` → `"mi chiamo Giovanni"`.
- **Unusable reply**: prompt turn then `process_input("nope")` → not-caught ack, no
  name fact stored, slot cleared.
- **Cancel escape**: prompt turn then `process_input("cancel")` → cancel ack, no
  fact stored.

Android (JUnit4): extend the existing `detectUserName` test with IT cases
(`"mi chiamo Giovanni"→"Giovanni"`, `"il mio nome è Anna"→"Anna"`) and a negative.
Device sanity: tap the chip, answer, confirm the next empty view greets by name
(EN + IT).

## Out of scope

- Any skill change (engine-internal; no PR).
- Changing/replacing an already-set name (initial capture only).
- A dedicated name setting — the name stays derived from remembered facts, matching
  the existing contract.
- ari-linux (not yet implemented).
