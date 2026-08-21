# Replying into a live conversation

**Date:** 2026-08-18
**Status:** Approved, ready for planning
**Repos touched:** `ari-android`, `ari-engine`, `ari-skills`
**Follows:** [send-message](2026-08-18-send-message-skill-design.md)

## Problem

Every transport the message skill has ends with the user looking at a screen.
SMS and Matrix send outright but need a spoken confirmation first; WhatsApp,
Telegram and email fill a compose box and wait for a tap. All fine at a desk.
None of it works while driving, which is the situation voice was supposed to be
for.

Android notifications carry `RemoteInput` reply actions, and any app holding
notification-listener access can fire one. This is the documented mechanism
Android Auto and Wear OS use, and WhatsApp, Telegram, Signal and Messenger all
support it. It **truly sends**, with no per-service code and no address book —
bounded to conversations that currently have a notification showing.

Ari already holds the permission. [`AriNotificationListenerService`] exists so
`MediaSessionManager.getActiveSessions` will return other apps' sessions, and
its own doc comment records that it "intentionally reads nothing". The gate,
the permission check and the settings deep-link are all built.

## Key insight

**Reply is a transport, not a skill.**

The intent is identical — get these words to Gail. What differs is the
mechanism, exactly as SMS differs from WhatsApp. Making it a separate skill
would mean two skills competing for "tell Gail I'm on my way", duplicated
recipient parsing, and a user who has to know which phrasing reaches which
mechanism.

**And it should be preferred, not merely available.** When Gail's notification
is live, replying into her thread beats composing on every axis: hands-free,
correctly threaded, no contacts lookup, works for services that have no other
route. The skill should reach for it first and fall back to the existing
transports when there's no live thread — which makes the genuinely hands-free
path the default rather than something the user has to phrase specially.

## The privacy constraint, which shapes everything

A `NotificationListenerService` sees **every notification from every app** —
banking, health, dating, two-factor codes. That is a far bigger surface than
`contacts`, which at least only exposes an address book the user chose to fill
in, and bigger than anything else Ari touches.

Nothing here is worth that if it's done carelessly. So:

- **Filter on arrival, not on use.** A notification from a package that isn't in
  the messaging catalogue is dropped in `onNotificationPosted` before anything
  reads it. No general store of notifications ever exists.
- **Never retain the message body.** What arrives is somebody else's words to
  the user. Ari needs the sender's name and the reply action; it does not need
  what was said, and must not keep it.
- **Keep the reply action, not the notification.** The retained shape is
  `(package, conversation title, PendingIntent + RemoteInput key)` — the minimum
  that can fire a reply.
- **Evict on dismissal and on a TTL.** `onNotificationRemoved` clears the entry;
  anything older than the TTL is dropped regardless, because a `PendingIntent`
  for a conversation the user finished with is a reply waiting to go to the
  wrong place.
- **Nothing crosses the FFI but a name.** The engine and the skill see a list of
  conversation display names. `PendingIntent`s stay in the frontend.

Worth stating plainly in review: **the skill never sees a notification.** It
asks "is there a live thread for Gail?" and gets yes or no.

## Target behaviour

| Situation | What happens |
|---|---|
| "Reply to Gail: on my way", her thread is live | Sends into the thread. Nothing to tap. |
| "Tell Gail I'm on my way", her thread is live | Same — the skill prefers the live thread |
| "Tell Gail I'm on my way", no live thread | Existing behaviour: send or compose per service |
| "Reply: on my way", exactly one live thread | Sends into it, naming who it went to |
| "Reply: on my way", several live threads | Asks which one |
| "Reply to Gail…", no live thread for Gail | Falls back to composing, and says so |

`reply` becomes a claimed verb — the send-message spec deliberately left it
unclaimed for exactly this.

**Confirmation.** A reply is a true send: nobody sees it before the recipient
does. It takes the same read-back as SMS and Matrix, governed by the same
`confirm_before_sending` setting. The bar for skipping it is *the user is
already looking at the message*, and here they are not.

**Naming who it went to is not optional.** "Sent" is a worse answer than "Sent
to Gail" when the skill picked the thread itself.

## Changes

### Change 1 — Android: read conversations off the listener

Override `onNotificationPosted` / `onNotificationRemoved` in
[`AriNotificationListenerService`], keeping a map of live conversations. Entry
criteria, all required:

- the posting package is in `messaging-services.json`
- the notification carries an action with a `RemoteInput`
- a usable conversation title exists

Ordered most-recent-first, so "reply" with no name means the newest thread.

### Change 2 — Android: fire the reply

Build the `RemoteInput` bundle, fill the results, send the `PendingIntent`.
Distinct outcomes, per the house rule that different results get different
types: `Sent`, `NoLiveThread`, `Ambiguous(names)`, `NoPermission`, `Failed`.

`NoPermission` is not a failure — it means the user hasn't granted notification
access, and the caller falls back to composing. The deep-link to grant it
already exists in [`MediaPermissions.kt`].

### Change 3 — Engine: `reply` slot and capability

A `reply` frontend capability gating a `reply` envelope slot:

```json
"reply": { "recipient_label": "Gail", "text": "On my way" }
```

Plus a host import so the skill can ask what's live before choosing a
transport — the same shape as `media_services`, returning display names only:

```rust
let live: Vec<String> = ari::live_conversations();
```

Ungated read of *names only* is the whole point: the skill can route
intelligently without ever touching a notification.

### Change 4 — Skill: prefer the live thread

In `resolve_and_act`, before the existing transport choice: if a live
conversation matches the recipient, emit a `reply` slot instead. Matching is
case-insensitive on the display name, with the same longest-match rule the
contacts resolution uses.

Add the `reply` verb family to `parse.rs`: `reply to <name>`, `reply <body>`,
`answer <name>`. A bare `reply` with no name is valid and means the newest
thread — which is the driving case.

## Tests

**Android** — catalogue filtering (a banking app's notification never enters the
map), eviction on removal and on TTL, most-recent-first ordering, a
notification with no `RemoteInput` is ignored, the retained record holds no
message body.

**Engine** — `reply` stripped from a skill that didn't declare the capability,
matching the `send_message` precedent.

**Skill** — a live thread beats composing; no live thread falls back; several
threads ask; `reply` with no name takes the newest; the confirmation still fires
for a reply.

## Verification (device)

With a real message received on WhatsApp, then:

```
reply on my way
```

```
tell mario i am five minutes away
```

The second is the one that matters: it must go into the existing thread rather
than opening a compose box, and Ari must say who it went to.

## Out of scope (YAGNI)

- **Reading messages out.** A different feature with a different consent
  question — this spec deliberately never retains a body.
- **Conversations without a live notification.** The whole mechanism is the
  notification; a dismissed one is gone. Composing already covers that.
- **Notification access in onboarding.** Asked for when it's first needed, like
  every other permission here.

[`AriNotificationListenerService`]: ../../../../ari-android/app/src/main/java/dev/heyari/ari/media/AriNotificationListenerService.kt
[`MediaPermissions.kt`]: ../../../../ari-android/app/src/main/java/dev/heyari/ari/media/MediaPermissions.kt
