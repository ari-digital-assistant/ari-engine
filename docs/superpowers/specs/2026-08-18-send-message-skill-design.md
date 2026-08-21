# Send a message to a contact via a service

**Date:** 2026-08-18
**Status:** Approved, ready for planning
**Repos touched:** `ari-engine`, `ari-android`, `ari-skills`

## Problem

"Tell Gail I'll be home soon on WhatsApp" has no path through Ari today. There is
no contacts capability, no message envelope slot, and no skill. The backlog
carried it as one line covering eight services, which hid the fact that those
services do not share a mechanism — some send outright, some can only hand off to
their own app, and one cannot identify a recipient at all.

Device testing on a Pixel 10 Pro Fold (GrapheneOS, 629 contacts) established what
is actually reachable. The findings that shape this design:

- **WhatsApp and Telegram both send-with-prefill** via documented URL schemes,
  reading an identifier straight out of their own `ContactsContract` rows —
  different column and different prefix per service, so both must be data.
- **The generic share intent carries the message and sends inline**, verified end
  to end on Messenger with no contact rows present at all. The user picks the
  recipient; the message survives intact.
- **Discord writes no contacts row for anybody.** Its identities have no
  phone-book relationship, so a spoken name cannot be resolved. Out of scope.
- **`ACTION_VIEW` on a contacts data row opens the right chat** in any app that
  syncs contacts, but carries no text — near-useless for messaging, and the
  reason it is the last rung rather than the elegant general answer.

## Key insight

**Two delivery modes, and only one of them needs a confirmation.**

A skill that truly sends (SMS, Matrix) takes an irreversible action on the
user's behalf with no human in the loop. That is the first such action Ari
would ever take, and a fuzzy contact match plus auto-send is a message to the
wrong person that cannot be recalled. It gets a spoken read-back by default.

A skill that hands off to another app's compose surface **already has a human in
the loop** — the user's own tap is the confirmation. Asking first would be a
double-confirm for no safety gain, and on a voice frontend it would cost a whole
extra turn. So hand-off gets a statement, not a question.

The setting therefore governs one mode only. There is nothing to disable in the
other.

## Target behaviour

| Delivery | Skill says | User does |
|---|---|---|
| `send`, confirm on *(default)* | "Your message to Mario says: I'll be home soon. Want me to send it?" | Answers, then it sends |
| `send`, confirm off | "Sent your message to Mario." | Nothing |
| `compose`, chat targeted | "That's ready for Mario — just tap send." | Taps send |
| `compose`, picker | "Your message is ready in Messenger — pick Mario to send it." | Finds Mario, taps send |

Confirmation is **on by default**. The setting is per-skill and does not affect
`compose` at all.

The two `compose` lines are not cosmetic variants. The targeted rung has the
recipient resolved and the chat open; the picker rung has handed over a message
with no recipient attached, and the user has to go and find them. Telling someone
their message is "prepared" when they still have a list to hunt is the kind of
small lie that makes an assistant feel unreliable.

**Both name the recipient, and that is a free safety check.** If the wrong Mario
was matched, the user finds out while looking at a compose box rather than after
the message has gone — the same protection the `send` read-back buys, at no extra
turn.

### The read-back must quote the user accurately

`process_input_traced` normalises before anything downstream sees the utterance —
lowercased, contractions expanded, punctuation stripped — and discards the raw
string. `SkillContext` carries only `locale` and `installed_apps`.

So "Tell Mario I'll be home soon" reaches the skill as
`tell mario i will be home soon` — lowercased, with the contraction expanded.
That is exactly what the keyword matcher wants, which is all normalisation was
ever built for. It is still unusable as a message body.

In `compose` mode that text sits in the target app's box where the user reads it
before tapping send, so it is not merely an internal wart — it is the thing the
user is asked to approve.

**`raw_input` on `SkillContext` is a hard prerequisite for this skill**, not a
nice-to-have. Nothing else in the design works without it.

## Changes

### Change 1 — Engine: expose the raw utterance

Add `raw_input: String` to
[`SkillContext`](../../../crates/ari-core/src/lib.rs) and populate it in
`process_input_traced` from the pre-normalise string that is currently dropped at
the top of the function. Additive on a plain struct; `Default` gets an empty
string, so no existing skill is affected.

The normalised text stays the input to `score()` and to matching — only the
skill's own body extraction reads `raw_input`.

`execute_reply` gets the same context. A skill that asks "what do you want to
say?" receives the answer as its message body, so a normalised reply would
reintroduce the exact problem this change exists to fix.

### Change 1b — Trigger phrases

Weight 0.95 unless noted. All patterns match post-normalise text.

```
send (a |an )?(message|text) to <name>
send <name> a (message|text)
message <name>
text <name>                              → forces sms
let <name> know
whatsapp|telegram|signal|slack <name>    → forces that service
```

Service-as-verb is worth having on its own: people say "WhatsApp me later", and
it removes the trailing `on <service>` parse for those cases.

**`tell <name>` cannot be a manifest pattern.** It collides with "tell me a
joke", "tell me the time", "tell me about the weather", "tell you what" — and
Rust's `regex` crate has no lookaround, by design, so `\btell (?!me\b)` isn't
available. It needs `matching.custom_score`, where the skill scores in Rust and
can inspect the token after `tell` directly. Blocklist the pronouns (`me`,
`us`, `you`, `him`, `her`, `them`) until Change 2 lands, after which the real
rule applies: **"tell X" is a message if and only if X is a contact.**

Two verbs are deliberately **not** claimed:

- `ask <name>` — collides with "ask what time it is" for little gain.
- `reply to <name>` — belongs to the notification-reply feature. Claiming it
  here would mean composing a *new* message when the user meant to reply to a
  live thread, which is worse than not matching at all.

### Change 2 — Engine: `contacts` capability

A new host import alongside `calendar` and `tasks`, which are the templates for
every touch point:

- `parse_capability` / `capability_name` in
  [`host_capabilities.rs`](../../../crates/ari-skill-loader/src/host_capabilities.rs)
- import table in [`wasm.rs`](../../../crates/ari-skill-loader/src/wasm.rs)
- provider trait + null impl in
  [`platform_capabilities.rs`](../../../crates/ari-skill-loader/src/platform_capabilities.rs)
- loader default, FFI builder method, uniffi regen
- Kotlin provider backed by `ContactsContract`, `READ_CONTACTS`

**Scope it to lookup, never enumeration.** The import is
`contacts_lookup(name) -> [Match]`, where a `Match` carries a display name and the
messaging channels that contact actually has. A community WASM skill must not be
able to walk the address book.

`contacts_permission_granted` is a separate import so `score()` can decline
early rather than failing deep in `execute`.

### Change 3 — Engine + frontends: `send_message` capability and envelope slot

A frontend capability gating one new single-value slot in the
[action envelope](../../../../ari-skills/docs/reference-actions.md):

```json
"message": {
  "service": "whatsapp",
  "recipient_id": "35699000000",
  "recipient_label": "Mario",
  "text": "I'll be home soon",
  "delivery": "compose"
}
```

`delivery` is `"send"` or `"compose"`. `recipient_id` is omitted for `compose`
when the target app will ask; `recipient_label` is display-only and never used to
address anything.

### Change 4 — Android: `MessageLauncher`

Modelled directly on
[`MusicLauncher`](../../../../ari-android/app/src/main/java/dev/heyari/ari/actions/MusicLauncher.kt) —
a service registry, a best-first strategy ladder, and a distinct result type per
outcome. The ladder, in runtime order:

1. **URL template** — `whatsapp://send?phone={id}&text={text}`,
   `tg://resolve?phone={id}&text={text}`. Skips the picker.
2. **Share intent** — `ACTION_SEND` `text/plain` scoped to the target package.
   The picker sends inline. Covers every share target on the device.
3. **Contacts data row** — `ACTION_VIEW` on the row URI **with the mimetype set
   via `setDataAndType`**; omitting the type sends it to the contacts app
   instead. Opens the chat, carries no text. Last resort only.

Prefer the custom scheme over the `https://` form: `wa.me` is an app link and
falls through to a browser install page when WhatsApp is absent.

The `<queries>` block in
[`AndroidManifest.xml`](../../../../ari-android/app/src/main/AndroidManifest.xml)
currently lists music packages only. Without the messaging packages added, the
launcher cannot see whether a target is installed, and the precondition check in
`score()` has nothing to check.

### Change 5 — Service registry as manifest data

Per-service knowledge is data, not Kotlin:

```yaml
- id: whatsapp
  packages: [com.whatsapp]
  mimetype: vnd.android.cursor.item/vnd.com.whatsapp.profile
  id_column: data1
  id_strip_suffix: "@s.whatsapp.net"
  url: "whatsapp://send?phone={id}&text={text}"
- id: telegram
  packages: [org.telegram.messenger]
  mimetype: vnd.android.cursor.item/vnd.org.telegram.messenger.android.profile
  id_column: data3
  id_strip_prefix: "Message +"
  url: "tg://resolve?phone={id}&text={text}"
```

The column differs, the prefix differs, and one strips a suffix while the other
strips a prefix. Hardcoding either shape breaks on the second service. Making it
data is also what lets someone add LINE, Viber, KakaoTalk or Zalo without an
engine release — the reason this feature cannot ship a Western-authored list.

### Change 6 — The skill

WASM Rust, `ari-skills/skills/message/`. Capabilities: `contacts`,
`send_message`, `storage_kv` (pending-send state across the confirm turn).

**Outcome types.** Each meaningfully different result gets its own variant — no
overloading a success type to mean "nothing matched":

| Variant | Response |
|---|---|
| `Sent` | "Sent your message to Mario." |
| `PreparedTargeted` | "That's ready for Mario — just tap send." |
| `PreparedPicker` | "Your message is ready in Messenger — pick Mario to send it." |
| `AwaitingConfirmation` | the read-back question |
| `ContactAmbiguous` | asks which one, via `await_reply` |
| `ContactNotFound` | "I couldn't find Mario in your contacts." |
| `ServiceUnavailable` | "Mario isn't on WhatsApp." |
| `NoChannel` | "I don't have a way to message Mario." |

**Recipient parsing.** Match the longest contact-name prefix against real lookup
results rather than splitting on the first word — "tell gail marie i will be
late" breaks any naive split, and only the address book can disambiguate.

**Confirmation flow** uses the existing `await_reply` mechanism; there is no new
multi-turn machinery. The pending send is round-tripped in the opaque context
blob. A reply that is not recognised as affirmative is treated as a decline —
fail-closed, because the cost of a false positive is an unrecallable message.

**Settings.** There is no boolean field type, so this is a two-option `select`,
which reads more clearly than a toggle for a safety control anyway:

```yaml
settings:
  - key: confirm_before_sending
    label: Before sending
    type: select
    default: always
    options:
      - value: always
        label: Read it back and ask
      - value: never
        label: Send straight away
  - key: default_service
    label: Send messages with
    type: select
    default: sms
    options: [sms, whatsapp, telegram, signal, matrix, slack]
```

**Why `default_service` is a setting and not a constant.** Defaulting to SMS
matches Siri and Google and is the predictable choice, so it stays the default.
But SMS is near-dead across much of Europe and costs money, while it is
completely ordinary in the US; someone in Brazil wants WhatsApp and someone in
Ukraine wants Telegram. Hardcoding it is the same mistake as hardcoding the
service list.

The setting only decides the bare `message <name>` case. Anything explicit wins:
`text gail` → sms, `whatsapp gail` → whatsapp, `… on telegram` → telegram.

**No body.** `message gail` with nothing to say asks "What do you want to say?"
via `await_reply` and takes the next utterance as the body — which is why
`execute_reply` needed the raw context too.

`setting_get` is ungated, so this needs no capability.

**Strings.** All user-facing text goes through `strings/en.json`. English only —
no translations are generated for this skill.

```json
"compose.targeted.speak": "That's ready for {recipient} — just tap send.",
"compose.picker.speak": "Your message is ready in {service} — pick {recipient} to send it."
```

## Tests

### Engine

- `raw_input` survives to the skill unmodified while `score()` still receives
  normalised text.
- A skill declaring `contacts` fails install against a host that does not grant
  it; succeeds against one that does.
- A `message` slot from a skill that did not declare `send_message` is stripped,
  matching how `critical_alert` is clamped today.

### Skill

- Recipient extraction: single-word name, two-word name, name that is a prefix of
  a longer contact, trailing `on <service>`, no service named.
- Body extraction preserves capitals, apostrophes and contractions from
  `raw_input`.
- `confirm_before_sending=always` yields `AwaitingConfirmation` and no message
  slot; `never` yields the slot directly.
- An affirmative reply sends; a negative reply and an unrecognised reply both
  abandon.
- `compose` delivery never asks for confirmation regardless of the setting.

### Android

- `MessageLauncher` falls from template to share intent when the package is
  absent; reports a distinct result for each rung.
- Contacts row route sets **both** data and type.

## Verification (device)

Per the repeat-the-test-phrase convention:

```
tell mario i'll be home soon on whatsapp
```

Expect the WhatsApp chat to open with **"I'll be home soon"** — capitalised,
apostrophe intact. That single check catches a missing `raw_input` immediately.

```
text mario i'll be home soon
```

With confirmation on, expect the read-back question and no send until answered.
Toggle the setting to "Send straight away" and repeat; expect no question.

```
tell mario i'll be there soon on messenger
```

Expect the picker with the text attached — the no-contacts-row path.

## Out of scope (YAGNI)

- **Discord and RCS.** Discord cannot resolve a recipient from a spoken name.
  RCS is a transport inside Google Messages, not an addressable target.
- **Matrix end-to-end encryption.** Olm/Megolm in a WASM skill is a project in
  itself; the SDK ships `sha2` and nothing more. Unencrypted rooms only.
- **Notification `RemoteInput` replies.** Genuinely hands-free and mostly plumbed
  already via `AriNotificationListenerService`, but it is a different feature
  with a different trigger. Separate spec.
- **Linux.** No frontend yet, and the capability docs already record that Linux
  grants nothing.
- **Sending email as the user.** Nobody lets another app post mail on your
  behalf without OAuth, and OAuth is not worth it for a case nobody has —
  people dictate "tell Gail I'm running late", not correspondence. Email goes
  out through `mailto:`, which fills in the address and body in the user's own
  client and leaves them one tap. Generic SMTP is out for a separate reason:
  no socket capability exists and none should be added.
- **Slack as a true send.** It reaches one workspace, needs a pasted token, and
  its own app is already a share target — the chooser path gets there for free.
  Matrix earned an HTTPS transport because it is federated and has no other
  route in; Slack has a perfectly good app.

## Resolved during review

**Compose phrasing split in two.** The original used one line for both rungs —
"I've prepared your message. You can now send it." Accurate for the targeted
rung, generous for the picker rung where the user still has to find the
recipient. Now two outcomes with distinct wording.

**Email dropped to `mailto:`.** Specced as OAuth against provider APIs, which
was carried over from the feasibility question "can we truly send" without
re-asking whether truly-sending email was wanted. It isn't.

**`speak` removed from the send path.** The skill was announcing "Sent your
message to Mario" on a request the frontend cannot always honour — no SMS
permission, no resolved number, and it composes instead. The frontend now
phrases every outcome from what actually happened, for compose and send alike.

## Found while building

Three defects in `normalize_input`, all from it having only ever been read by
the keyword matcher, where none of them mattered:

- `.replace("whats", "what is")` had no word boundary, so **"whatsapp" became
  "what isapp"** — every utterance naming WhatsApp reached every skill
  corrupted. Fixed here; the rules now apply per word.
- No `'ll` / `'ve` / `'d` rule, so "I'll" became "i ll" — a broken word under
  every reading. Fixed since, by a suffix pass alongside the whole-word table.
- The raw utterance was discarded entirely, which is what `raw_input` fixes.

Two capability-plumbing bugs the layered checks caught rather than users:

- `contacts_lookup` was registered inside the `MediaServices` gate, so the
  import only existed for skills declaring media services.
- `android_load_options` never granted `Contacts`, so the skill installed on
  the CLI and would have been refused on a real phone. There is now one
  `ALL_CAPABILITIES` list and a test asserting Android grants every entry.
