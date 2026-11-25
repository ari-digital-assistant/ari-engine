//! The presentation envelope a skill's action reply carries, parsed into
//! types, so every frontend reads it the same way. Moved from ari-android's
//! `PresentationEnvelope.kt`. The wire format is documented in ari-skills'
//! `docs/reference-actions.md`.
//!
//! Parsing is as lenient as the Kotlin was: an absent, null or empty string
//! is no value, a number where a string was expected reads as its digits, a
//! malformed card, alert or notification is dropped rather than failing the
//! envelope, and a slot missing its one required field is no slot.

use serde_json::{Map, Value};

/// The only envelope version this engine reads.
pub const SUPPORTED_VERSION: i64 = 1;

/// An alert rings for this long, unless it says otherwise.
pub const DEFAULT_AUTO_STOP_MS: i64 = 120_000;

/// An alert repeats its speech this many times, unless it says otherwise.
pub const DEFAULT_MAX_CYCLES: i64 = 12;

#[derive(Debug, Clone, PartialEq)]
pub struct PresentationEnvelope {
    pub speak: Option<String>,
    /// Bubble text, when the skill wants it to differ from what is spoken.
    /// None means the bubble shows `speak`.
    pub display: Option<String>,
    pub cards: Vec<Card>,
    pub alerts: Vec<AlertSpec>,
    pub notifications: Vec<NotificationSpec>,
    pub launch_app: Option<String>,
    pub media: Option<MediaAction>,
    pub alarm: Option<AlarmAction>,
    pub navigate: Option<NavigateAction>,
    pub message: Option<MessageAction>,
    pub reply: Option<ReplyAction>,
    pub search: Option<String>,
    pub open_url: Option<String>,
    pub clipboard_text: Option<String>,
    pub dismiss_cards: Vec<String>,
    pub dismiss_notifications: Vec<String>,
    pub dismiss_alerts: Vec<String>,
    /// An utterance the frontend sends back through the engine once it has
    /// handled the rest, so a card's `on_cancel` can round-trip a cancel to
    /// its own skill.
    pub run_utterance: Option<String>,
    /// How sure the skill is of its own output: `High` when it didn't say.
    pub confidence: Confidence,
    /// What the skill noticed but couldn't use. None when `confidence` is
    /// `High`.
    pub unparsed: Option<String>,
    /// A store skill to offer, from the engine's own suggestion.
    pub suggest_skill: Option<SkillSuggestion>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Confidence {
    High,
    Partial,
    Low,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Card {
    pub id: String,
    /// The skill that sent it, for its `asset:` references.
    pub skill_id: String,
    pub title: String,
    pub subtitle: Option<String>,
    pub body: Option<String>,
    pub icon: Option<String>,
    pub countdown_to_ts_ms: Option<i64>,
    pub started_at_ts_ms: Option<i64>,
    pub progress: Option<f32>,
    pub accent: Accent,
    pub actions: Vec<CardAction>,
    pub on_complete: Option<OnComplete>,
    /// The `on_cancel` envelope as raw JSON, parsed when the card is
    /// cancelled: a skill can put any primitive in it.
    pub on_cancel: Option<String>,
    pub stat: Option<Stat>,
    pub list: Option<ListCard>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Accent {
    Default,
    Warning,
    Success,
    Critical,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IconText {
    pub icon: Option<String>,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stat {
    pub headline: String,
    pub caption: Option<String>,
    pub pill: Option<IconText>,
    pub metrics: Vec<IconText>,
    pub background: Option<String>,
    pub footer: Option<IconText>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ListRow {
    pub leading: String,
    pub icon: Option<String>,
    pub text: Option<String>,
    pub trailing: Option<String>,
    pub badge: Option<IconText>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ListCard {
    pub summary: Option<IconText>,
    pub rows: Vec<ListRow>,
    pub footer: Option<IconText>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CardAction {
    pub id: String,
    pub label: String,
    pub utterance: Option<String>,
    pub speak: Option<String>,
    pub style: ButtonStyle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ButtonStyle {
    Default,
    Primary,
    Destructive,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OnComplete {
    pub alert: Option<AlertSpec>,
    /// True unless the skill says otherwise.
    pub dismiss_card: bool,
    pub dismiss_notifications: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AlertSpec {
    pub id: String,
    pub skill_id: String,
    pub title: String,
    pub body: Option<String>,
    pub urgency: Urgency,
    /// `system.alarm`, `system.notification`, `system.silent` or
    /// `asset:<path>`.
    pub sound: String,
    pub speech_loop: Option<String>,
    pub auto_stop_ms: i64,
    pub max_cycles: i64,
    pub full_takeover: bool,
    pub icon: Option<String>,
    pub actions: Vec<AlertAction>,
}

/// The sound an alert makes when it doesn't name one.
pub const DEFAULT_ALERT_SOUND: &str = "system.notification";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Urgency {
    Normal,
    High,
    Critical,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlertAction {
    pub id: String,
    pub label: String,
    pub utterance: Option<String>,
    pub style: ButtonStyle,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NotificationSpec {
    pub id: String,
    pub skill_id: String,
    pub title: String,
    pub body: Option<String>,
    pub importance: Importance,
    pub sticky: bool,
    pub countdown_to_ts_ms: Option<i64>,
    pub actions: Vec<NotificationAction>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Importance {
    Min,
    Low,
    Default,
    High,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NotificationAction {
    pub id: String,
    pub label: String,
    pub utterance: Option<String>,
}

/// Play `query`, optionally on a named `service`, or a transport or volume
/// command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaAction {
    pub action: String,
    pub query: Option<String>,
    pub service: Option<String>,
    pub direction: Option<String>,
    pub level: Option<i64>,
    pub mute: Option<bool>,
}

/// `op` is `set` or `show`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlarmAction {
    pub op: String,
    pub hour: Option<i64>,
    pub minute: Option<i64>,
    pub message: Option<String>,
    pub days: Vec<String>,
    /// Set the alarm without showing the clock's own UI. True unless the
    /// skill says otherwise.
    pub skip_ui: bool,
}

/// `mode` is `default_app` (or none) or `turn_by_turn`; `place` is `home`
/// when the destination is the user's saved home.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NavigateAction {
    pub destination: String,
    pub mode: Option<String>,
    pub place: Option<String>,
}

/// A message to send. `delivery` is a request (`send` or `compose`), not a
/// promise: what actually happens is up to the app that sends it.
/// `recipient_label` is only for saying back; `recipient_id` addresses it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MessageAction {
    pub text: String,
    pub service: Option<String>,
    pub recipient_id: Option<String>,
    pub recipient_label: Option<String>,
    pub delivery: Option<String>,
}

/// A reply into a conversation with a live notification. No recipient
/// means the newest one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplyAction {
    pub text: String,
    pub recipient_label: Option<String>,
}

/// A store skill offered because it would handle what the user asked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillSuggestion {
    pub id: String,
    pub name: String,
}

impl PresentationEnvelope {
    /// The envelope in `json`, from skill `skill_id`, which is stamped on
    /// every card, alert and notification for its assets. None when it
    /// isn't JSON or isn't a version this engine reads.
    pub fn parse(json: &str, skill_id: &str) -> Option<Self> {
        Self::from_value(&serde_json::from_str(json).ok()?, skill_id)
    }

    pub fn from_value(json: &Value, skill_id: &str) -> Option<Self> {
        let o = json.as_object()?;
        if int(o, "v").unwrap_or(0) != SUPPORTED_VERSION {
            return None;
        }
        let dismiss = object(o, "dismiss");
        Some(Self {
            speak: string(o, "speak"),
            display: string(o, "display"),
            cards: objects(o, "cards").filter_map(|c| card(c, skill_id)).collect(),
            alerts: objects(o, "alerts").filter_map(|a| alert(a, skill_id)).collect(),
            notifications: objects(o, "notifications")
                .filter_map(|n| notification(n, skill_id))
                .collect(),
            launch_app: string(o, "launch_app"),
            media: object(o, "media").and_then(|m| {
                Some(MediaAction {
                    action: string(m, "action")?,
                    query: string(m, "query"),
                    service: string(m, "service"),
                    direction: string(m, "direction"),
                    level: int(m, "level"),
                    mute: boolean(m, "mute"),
                })
            }),
            alarm: object(o, "alarm").and_then(|a| {
                Some(AlarmAction {
                    op: string(a, "op")?,
                    hour: int(a, "hour"),
                    minute: int(a, "minute"),
                    message: string(a, "message"),
                    days: strings(a, "days"),
                    skip_ui: boolean(a, "skip_ui").unwrap_or(true),
                })
            }),
            navigate: object(o, "navigate").and_then(|n| {
                Some(NavigateAction {
                    destination: string(n, "destination")?,
                    mode: string(n, "mode"),
                    place: string(n, "place"),
                })
            }),
            message: object(o, "message").and_then(|m| {
                Some(MessageAction {
                    text: string(m, "text")?,
                    service: string(m, "service"),
                    recipient_id: string(m, "recipient_id"),
                    recipient_label: string(m, "recipient_label"),
                    delivery: string(m, "delivery"),
                })
            }),
            reply: object(o, "reply").and_then(|r| {
                Some(ReplyAction {
                    text: string(r, "text")?,
                    recipient_label: string(r, "recipient_label"),
                })
            }),
            search: string(o, "search"),
            open_url: string(o, "open_url"),
            clipboard_text: object(o, "clipboard").and_then(|c| string(c, "text")),
            dismiss_cards: dismiss.map(|d| strings(d, "cards")).unwrap_or_default(),
            dismiss_notifications: dismiss.map(|d| strings(d, "notifications")).unwrap_or_default(),
            dismiss_alerts: dismiss.map(|d| strings(d, "alerts")).unwrap_or_default(),
            run_utterance: string(o, "run_utterance"),
            confidence: match string(o, "confidence").map(|c| c.to_lowercase()).as_deref() {
                Some("partial") => Confidence::Partial,
                Some("low") => Confidence::Low,
                // Skills from before the signal say nothing: take them at
                // their word.
                _ => Confidence::High,
            },
            unparsed: string(o, "unparsed"),
            suggest_skill: object(o, "suggest_skill").and_then(|s| {
                let id = string(s, "id")?;
                Some(SkillSuggestion { name: string(s, "name").unwrap_or_else(|| id.clone()), id })
            }),
        })
    }

    /// Whether this puts anything on screen or takes anything off it, beyond
    /// words: what decides that the frontend's presentation runs at all.
    pub fn has_presentation(&self) -> bool {
        !self.cards.is_empty()
            || !self.alerts.is_empty()
            || !self.notifications.is_empty()
            || !self.dismiss_cards.is_empty()
            || !self.dismiss_notifications.is_empty()
            || !self.dismiss_alerts.is_empty()
            || self.suggest_skill.is_some()
    }
}

fn card(o: &Map<String, Value>, skill_id: &str) -> Option<Card> {
    Some(Card {
        id: string(o, "id")?,
        skill_id: skill_id.to_string(),
        title: string(o, "title")?,
        subtitle: string(o, "subtitle"),
        body: string(o, "body"),
        icon: string(o, "icon"),
        countdown_to_ts_ms: int(o, "countdown_to_ts_ms"),
        started_at_ts_ms: int(o, "started_at_ts_ms"),
        progress: object(o, "progress").and_then(|p| float(p, "value")).map(|v| v as f32),
        accent: match string(o, "accent").as_deref() {
            Some("warning") => Accent::Warning,
            Some("success") => Accent::Success,
            Some("critical") => Accent::Critical,
            _ => Accent::Default,
        },
        actions: objects(o, "actions")
            .filter_map(|a| {
                Some(CardAction {
                    id: string(a, "id")?,
                    label: string(a, "label")?,
                    utterance: string(a, "utterance"),
                    speak: string(a, "speak"),
                    style: style(a),
                })
            })
            .collect(),
        on_complete: object(o, "on_complete").map(|c| OnComplete {
            alert: object(c, "alert").and_then(|a| alert(a, skill_id)),
            dismiss_card: boolean(c, "dismiss_card").unwrap_or(true),
            dismiss_notifications: strings(c, "dismiss_notifications"),
        }),
        on_cancel: object(o, "on_cancel").map(|c| Value::Object(c.clone()).to_string()),
        stat: object(o, "stat").and_then(|s| {
            Some(Stat {
                headline: string(s, "headline")?,
                caption: string(s, "caption"),
                pill: object(s, "pill").and_then(icon_text),
                metrics: objects(s, "metrics").filter_map(icon_text).collect(),
                background: string(s, "background"),
                footer: object(s, "footer").and_then(icon_text),
            })
        }),
        list: object(o, "list").map(|l| ListCard {
            summary: object(l, "summary").and_then(icon_text),
            rows: objects(l, "rows")
                .filter_map(|r| {
                    Some(ListRow {
                        leading: string(r, "leading")?,
                        icon: string(r, "icon"),
                        text: string(r, "text"),
                        trailing: string(r, "trailing"),
                        badge: object(r, "badge").and_then(icon_text),
                    })
                })
                .collect(),
            footer: object(l, "footer").and_then(icon_text),
        }),
    })
}

fn alert(o: &Map<String, Value>, skill_id: &str) -> Option<AlertSpec> {
    Some(AlertSpec {
        id: string(o, "id")?,
        skill_id: skill_id.to_string(),
        title: string(o, "title")?,
        body: string(o, "body"),
        urgency: match string(o, "urgency").as_deref() {
            Some("critical") => Urgency::Critical,
            Some("high") => Urgency::High,
            _ => Urgency::Normal,
        },
        sound: string(o, "sound").unwrap_or_else(|| DEFAULT_ALERT_SOUND.to_string()),
        speech_loop: string(o, "speech_loop"),
        auto_stop_ms: int(o, "auto_stop_ms").unwrap_or(DEFAULT_AUTO_STOP_MS),
        max_cycles: int(o, "max_cycles").unwrap_or(DEFAULT_MAX_CYCLES),
        full_takeover: boolean(o, "full_takeover").unwrap_or(false),
        icon: string(o, "icon"),
        actions: objects(o, "actions")
            .filter_map(|a| {
                Some(AlertAction {
                    id: string(a, "id")?,
                    label: string(a, "label")?,
                    utterance: string(a, "utterance"),
                    style: style(a),
                })
            })
            .collect(),
    })
}

fn notification(o: &Map<String, Value>, skill_id: &str) -> Option<NotificationSpec> {
    Some(NotificationSpec {
        id: string(o, "id")?,
        skill_id: skill_id.to_string(),
        title: string(o, "title")?,
        body: string(o, "body"),
        importance: match string(o, "importance").as_deref() {
            Some("min") => Importance::Min,
            Some("low") => Importance::Low,
            Some("high") => Importance::High,
            _ => Importance::Default,
        },
        sticky: boolean(o, "sticky").unwrap_or(false),
        countdown_to_ts_ms: int(o, "countdown_to_ts_ms"),
        actions: objects(o, "actions")
            .filter_map(|a| {
                Some(NotificationAction {
                    id: string(a, "id")?,
                    label: string(a, "label")?,
                    utterance: string(a, "utterance"),
                })
            })
            .collect(),
    })
}

fn style(o: &Map<String, Value>) -> ButtonStyle {
    match string(o, "style").as_deref() {
        Some("primary") => ButtonStyle::Primary,
        Some("destructive") => ButtonStyle::Destructive,
        _ => ButtonStyle::Default,
    }
}

fn icon_text(o: &Map<String, Value>) -> Option<IconText> {
    Some(IconText { icon: string(o, "icon"), text: string(o, "text")? })
}

/// A value as text: a string as itself, anything else but null as its JSON.
/// Empty is no value.
fn as_text(value: &Value) -> Option<String> {
    let text = match value {
        Value::Null => return None,
        Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    (!text.is_empty()).then_some(text)
}

fn string(o: &Map<String, Value>, key: &str) -> Option<String> {
    o.get(key).and_then(as_text)
}

/// A whole number, from a number (a fraction is cut short) or a numeric
/// string.
fn int(o: &Map<String, Value>, key: &str) -> Option<i64> {
    match o.get(key)? {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)),
        Value::String(s) => s.trim().parse::<i64>().ok().or_else(|| s.trim().parse::<f64>().ok().map(|f| f as i64)),
        _ => None,
    }
}

fn float(o: &Map<String, Value>, key: &str) -> Option<f64> {
    match o.get(key)? {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

fn boolean(o: &Map<String, Value>, key: &str) -> Option<bool> {
    match o.get(key)? {
        Value::Bool(b) => Some(*b),
        Value::String(s) if s.eq_ignore_ascii_case("true") => Some(true),
        Value::String(s) if s.eq_ignore_ascii_case("false") => Some(false),
        _ => None,
    }
}

fn object<'a>(o: &'a Map<String, Value>, key: &str) -> Option<&'a Map<String, Value>> {
    o.get(key)?.as_object()
}

/// The objects in array `key`, skipping anything else in it.
fn objects<'a>(o: &'a Map<String, Value>, key: &str) -> impl Iterator<Item = &'a Map<String, Value>> {
    o.get(key).and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_object)
}

/// The non-empty texts in array `key`.
fn strings(o: &Map<String, Value>, key: &str) -> Vec<String> {
    o.get(key).and_then(Value::as_array).into_iter().flatten().filter_map(as_text).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SKILL_ID: &str = "dev.heyari.timer";

    fn parse(json: &str) -> PresentationEnvelope {
        PresentationEnvelope::parse(json, SKILL_ID).expect("an envelope")
    }

    #[test]
    fn a_missing_or_other_version_is_refused() {
        assert_eq!(PresentationEnvelope::parse("{}", SKILL_ID), None);
        assert_eq!(PresentationEnvelope::parse(r#"{"v":2}"#, SKILL_ID), None);
        assert_eq!(PresentationEnvelope::parse("not json", SKILL_ID), None);
        assert_eq!(PresentationEnvelope::parse("[1]", SKILL_ID), None);
    }

    #[test]
    fn a_minimal_envelope_is_empty() {
        let env = parse(r#"{"v":1}"#);
        assert_eq!(env.speak, None);
        assert!(env.cards.is_empty() && env.alerts.is_empty() && env.notifications.is_empty());
        assert_eq!((env.launch_app, env.search, env.media), (None, None, None));
        assert_eq!(env.confidence, Confidence::High);
        assert!(!parse(r#"{"v":1}"#).has_presentation());
    }

    #[test]
    fn the_version_may_come_as_a_string() {
        assert_eq!(parse(r#"{"v":"1","speak":"hi"}"#).speak.as_deref(), Some("hi"));
    }

    #[test]
    fn speak_display_launch_search_open_url_and_clipboard() {
        let env = parse(
            r#"{"v":1,"speak":"hello","display":"Hello!","launch_app":"Spotify","search":"capital of malta",
                "open_url":"https://heyari.dev","clipboard":{"text":"copied"},"run_utterance":"cancel it"}"#,
        );
        assert_eq!(env.speak.as_deref(), Some("hello"));
        assert_eq!(env.display.as_deref(), Some("Hello!"));
        assert_eq!(env.launch_app.as_deref(), Some("Spotify"));
        assert_eq!(env.search.as_deref(), Some("capital of malta"));
        assert_eq!(env.open_url.as_deref(), Some("https://heyari.dev"));
        assert_eq!(env.clipboard_text.as_deref(), Some("copied"));
        assert_eq!(env.run_utterance.as_deref(), Some("cancel it"));
    }

    #[test]
    fn media_play_with_a_service() {
        let media = parse(r#"{"v":1,"media":{"action":"play","query":"hotel california","service":"spotify"}}"#)
            .media
            .unwrap();
        assert_eq!(
            media,
            MediaAction {
                action: "play".into(),
                query: Some("hotel california".into()),
                service: Some("spotify".into()),
                direction: None,
                level: None,
                mute: None,
            }
        );
    }

    #[test]
    fn media_without_an_action_is_no_media() {
        assert_eq!(parse(r#"{"v":1,"media":{"query":"x"}}"#).media, None);
    }

    #[test]
    fn media_transport_and_volume() {
        let pause = parse(r#"{"v":1,"media":{"action":"pause"}}"#).media.unwrap();
        assert_eq!((pause.action.as_str(), pause.query), ("pause", None));
        let set = parse(r#"{"v":1,"media":{"action":"volume","level":50}}"#).media.unwrap();
        assert_eq!((set.level, set.direction), (Some(50), None));
        let mute = parse(r#"{"v":1,"media":{"action":"volume","mute":true}}"#).media.unwrap();
        assert_eq!(mute.mute, Some(true));
        let up = parse(r#"{"v":1,"media":{"action":"volume","direction":"up"}}"#).media.unwrap();
        assert_eq!((up.direction.as_deref(), up.level), (Some("up"), None));
    }

    #[test]
    fn a_card_with_a_countdown_and_what_happens_when_it_ends() {
        let env = parse(
            r#"{
              "v": 1,
              "speak": "Pasta timer set for 8 minutes.",
              "cards": [{
                "id": "card_pasta",
                "title": "Pasta timer",
                "subtitle": "Started just now",
                "icon": "asset:icons/timer.png",
                "countdown_to_ts_ms": 1000480000,
                "started_at_ts_ms": 1000000000,
                "accent": "default",
                "progress": {"value": 0.25},
                "actions": [{"id":"cancel","label":"Cancel","utterance":"cancel my pasta timer"}],
                "on_complete": {
                  "alert": {
                    "id":"alert_pasta",
                    "title":"Pasta timer done",
                    "urgency":"critical",
                    "sound":"asset:timer_ding.wav",
                    "speech_loop":"Pasta timer",
                    "auto_stop_ms":120000,
                    "max_cycles":12,
                    "full_takeover":true,
                    "icon":"asset:icons/timer.png",
                    "actions":[{"id":"stop_alert","label":"Stop","style":"primary"}]
                  },
                  "dismiss_card": true,
                  "dismiss_notifications": ["notif_pasta"]
                },
                "on_cancel": {"v":1,"speak":"Cancelled."}
              }]
            }"#,
        );
        assert_eq!(env.cards.len(), 1);
        let card = &env.cards[0];
        assert_eq!((card.id.as_str(), card.skill_id.as_str(), card.title.as_str()), ("card_pasta", SKILL_ID, "Pasta timer"));
        assert_eq!(card.icon.as_deref(), Some("asset:icons/timer.png"));
        assert_eq!((card.countdown_to_ts_ms, card.started_at_ts_ms), (Some(1000480000), Some(1000000000)));
        assert_eq!(card.progress, Some(0.25));
        assert_eq!(card.accent, Accent::Default);
        assert_eq!(
            card.actions,
            [CardAction {
                id: "cancel".into(),
                label: "Cancel".into(),
                utterance: Some("cancel my pasta timer".into()),
                speak: None,
                style: ButtonStyle::Default,
            }]
        );
        let done = card.on_complete.as_ref().unwrap();
        assert!(done.dismiss_card);
        assert_eq!(done.dismiss_notifications, ["notif_pasta"]);
        assert_eq!(
            done.alert,
            Some(AlertSpec {
                id: "alert_pasta".into(),
                skill_id: SKILL_ID.into(),
                title: "Pasta timer done".into(),
                body: None,
                urgency: Urgency::Critical,
                sound: "asset:timer_ding.wav".into(),
                speech_loop: Some("Pasta timer".into()),
                auto_stop_ms: 120_000,
                max_cycles: 12,
                full_takeover: true,
                icon: Some("asset:icons/timer.png".into()),
                actions: vec![AlertAction {
                    id: "stop_alert".into(),
                    label: "Stop".into(),
                    utterance: None,
                    style: ButtonStyle::Primary,
                }],
            })
        );
        let cancel = PresentationEnvelope::parse(card.on_cancel.as_deref().unwrap(), SKILL_ID).unwrap();
        assert_eq!(cancel.speak.as_deref(), Some("Cancelled."));
    }

    #[test]
    fn when_a_card_ends_it_goes_unless_it_says_otherwise() {
        let ends = |json: &str| parse(json).cards[0].on_complete.clone().unwrap();
        let plain = ends(r#"{"v":1,"cards":[{"id":"c","title":"t","on_complete":{}}]}"#);
        assert!(plain.dismiss_card);
        assert!(plain.dismiss_notifications.is_empty());
        assert_eq!(plain.alert, None);
        let kept = ends(r#"{"v":1,"cards":[{"id":"c","title":"t","on_complete":{"dismiss_card":false}}]}"#);
        assert!(!kept.dismiss_card);
    }

    #[test]
    fn an_alert_s_defaults() {
        let alert = parse(r#"{"v":1,"alerts":[{"id":"a","title":"A"}]}"#).alerts.remove(0);
        assert_eq!(alert.urgency, Urgency::Normal);
        assert_eq!(alert.sound, DEFAULT_ALERT_SOUND);
        assert_eq!((alert.auto_stop_ms, alert.max_cycles), (DEFAULT_AUTO_STOP_MS, DEFAULT_MAX_CYCLES));
        assert!(!alert.full_takeover);
    }

    #[test]
    fn accents_urgencies_and_styles() {
        let env = parse(
            r#"{"v":1,"cards":[{"id":"w","title":"t","accent":"warning"},{"id":"s","title":"t","accent":"success"},
                {"id":"c","title":"t","accent":"critical","actions":[{"id":"x","label":"X","style":"destructive"}]}],
                "alerts":[{"id":"h","title":"t","urgency":"high"}]}"#,
        );
        let accents: Vec<Accent> = env.cards.iter().map(|c| c.accent).collect();
        assert_eq!(accents, [Accent::Warning, Accent::Success, Accent::Critical]);
        assert_eq!(env.cards[2].actions[0].style, ButtonStyle::Destructive);
        assert_eq!(env.alerts[0].urgency, Urgency::High);
    }

    #[test]
    fn a_notification_with_a_countdown() {
        let env = parse(
            r#"{"v":1,"notifications":[{"id":"notif_pasta","title":"Pasta timer","body":"Running…",
                "importance":"default","sticky":true,"countdown_to_ts_ms":1000480000,
                "actions":[{"id":"cancel","label":"Cancel","utterance":"cancel my pasta timer"}]}]}"#,
        );
        assert_eq!(
            env.notifications,
            [NotificationSpec {
                id: "notif_pasta".into(),
                skill_id: SKILL_ID.into(),
                title: "Pasta timer".into(),
                body: Some("Running…".into()),
                importance: Importance::Default,
                sticky: true,
                countdown_to_ts_ms: Some(1000480000),
                actions: vec![NotificationAction {
                    id: "cancel".into(),
                    label: "Cancel".into(),
                    utterance: Some("cancel my pasta timer".into()),
                }],
            }]
        );
        let low = parse(r#"{"v":1,"notifications":[{"id":"n","title":"N","importance":"low"}]}"#);
        assert_eq!(low.notifications[0].importance, Importance::Low);
    }

    #[test]
    fn dismissals_count_as_presentation() {
        let env = parse(r#"{"v":1,"dismiss":{"cards":["card_a","card_b"],"notifications":["notif_a"],"alerts":["alert_a"]}}"#);
        assert_eq!(env.dismiss_cards, ["card_a", "card_b"]);
        assert_eq!(env.dismiss_notifications, ["notif_a"]);
        assert_eq!(env.dismiss_alerts, ["alert_a"]);
        assert!(env.has_presentation());
    }

    #[test]
    fn a_card_without_an_id_is_dropped_and_the_rest_kept() {
        let env = parse(r#"{"v":1,"cards":[{"title":"no id"},{"id":"good","title":"OK"},"junk"]}"#);
        let ids: Vec<&str> = env.cards.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(ids, ["good"]);
    }

    #[test]
    fn null_absent_and_empty_are_all_no_value() {
        for card in [r#"{"id":"c","title":"t","body":null}"#, r#"{"id":"c","title":"t"}"#, r#"{"id":"c","title":"t","body":""}"#] {
            assert_eq!(parse(&format!(r#"{{"v":1,"cards":[{card}]}}"#)).cards[0].body, None, "{card}");
        }
    }

    #[test]
    fn a_number_where_text_was_expected_reads_as_its_digits() {
        let env = parse(r#"{"v":1,"cards":[{"id":7,"title":"t"}]}"#);
        assert_eq!(env.cards[0].id, "7");
    }

    #[test]
    fn stat_and_list_cards() {
        let env = PresentationEnvelope::parse(
            r#"{"v":1,"cards":[
                {"id":"c1","title":"London","stat":{"headline":"21°","caption":"cloudy",
                  "pill":{"icon":"asset:ui/thermometer.png","text":"Feels like 20°"},
                  "metrics":[{"icon":"asset:ui/wind.png","text":"Wind 18 km/h"},{"icon":"no text"}],
                  "background":"asset:heroes/cloudy.png",
                  "footer":{"text":"Weather data by Open-Meteo.com"}}},
                {"id":"c2","title":"London","subtitle":"7-day forecast","list":{
                  "summary":{"text":"High 31°"},
                  "rows":[{"leading":"Wed","icon":"asset:icons/cloudy.png","text":"cloudy",
                           "trailing":"24° / 18°","badge":{"text":"43%"}},{"text":"no leading"}]}}
            ]}"#,
            "dev.heyari.weather",
        )
        .unwrap();
        assert_eq!(
            env.cards[0].stat,
            Some(Stat {
                headline: "21°".into(),
                caption: Some("cloudy".into()),
                pill: Some(IconText { icon: Some("asset:ui/thermometer.png".into()), text: "Feels like 20°".into() }),
                metrics: vec![IconText { icon: Some("asset:ui/wind.png".into()), text: "Wind 18 km/h".into() }],
                background: Some("asset:heroes/cloudy.png".into()),
                footer: Some(IconText { icon: None, text: "Weather data by Open-Meteo.com".into() }),
            })
        );
        assert_eq!(
            env.cards[1].list,
            Some(ListCard {
                summary: Some(IconText { icon: None, text: "High 31°".into() }),
                rows: vec![ListRow {
                    leading: "Wed".into(),
                    icon: Some("asset:icons/cloudy.png".into()),
                    text: Some("cloudy".into()),
                    trailing: Some("24° / 18°".into()),
                    badge: Some(IconText { icon: None, text: "43%".into() }),
                }],
                footer: None,
            })
        );
    }

    #[test]
    fn the_skill_id_is_stamped_on_every_card_alert_and_notification() {
        let env = PresentationEnvelope::parse(
            r#"{"v":1,
                "cards":[{"id":"c","title":"t","on_complete":{"alert":{"id":"a","title":"A","urgency":"high","sound":"system.alarm"}}}],
                "notifications":[{"id":"n","title":"N"}],
                "alerts":[{"id":"a2","title":"A2"}]}"#,
            "dev.heyari.demo",
        )
        .unwrap();
        assert_eq!(env.cards[0].skill_id, "dev.heyari.demo");
        assert_eq!(env.cards[0].on_complete.as_ref().unwrap().alert.as_ref().unwrap().skill_id, "dev.heyari.demo");
        assert_eq!(env.notifications[0].skill_id, "dev.heyari.demo");
        assert_eq!(env.alerts[0].skill_id, "dev.heyari.demo");
    }

    #[test]
    fn an_alarm_with_days_and_a_message() {
        let alarm = PresentationEnvelope::parse(
            r#"{"v":1,"alarm":{"op":"set","hour":7,"minute":0,"message":"Wake up","days":["mon","fri"],"skip_ui":true}}"#,
            "dev.heyari.alarm",
        )
        .unwrap()
        .alarm
        .unwrap();
        assert_eq!(
            alarm,
            AlarmAction {
                op: "set".into(),
                hour: Some(7),
                minute: Some(0),
                message: Some("Wake up".into()),
                days: vec!["mon".into(), "fri".into()],
                skip_ui: true,
            }
        );
    }

    #[test]
    fn an_alarm_to_show_has_no_time_and_skips_the_ui_unless_told() {
        let show = parse(r#"{"v":1,"alarm":{"op":"show"}}"#).alarm.unwrap();
        assert_eq!((show.op.as_str(), show.hour, show.days.len()), ("show", None, 0));
        assert!(parse(r#"{"v":1,"alarm":{"op":"set","hour":7}}"#).alarm.unwrap().skip_ui);
        assert!(!parse(r#"{"v":1,"alarm":{"op":"set","hour":7,"skip_ui":false}}"#).alarm.unwrap().skip_ui);
        assert_eq!(parse(r#"{"v":1,"alarm":{"hour":7}}"#).alarm, None);
    }

    #[test]
    fn navigation() {
        let n = parse(r#"{"v":1,"navigate":{"destination":"casa","mode":"turn_by_turn","place":"home"}}"#).navigate.unwrap();
        assert_eq!((n.destination.as_str(), n.mode.as_deref(), n.place.as_deref()), ("casa", Some("turn_by_turn"), Some("home")));
        let plain = parse(r#"{"v":1,"navigate":{"destination":"asda"}}"#).navigate.unwrap();
        assert_eq!((plain.mode, plain.place), (None, None));
        assert_eq!(parse(r#"{"v":1,"navigate":{"mode":"turn_by_turn"}}"#).navigate, None);
        assert_eq!(parse(r#"{"v":1,"speak":"hi"}"#).navigate, None);
    }

    #[test]
    fn a_message_needs_only_its_text_and_keeps_it_as_said() {
        let full = parse(
            r#"{"v":1,"message":{"service":"whatsapp","recipient_id":"35699000000","recipient_label":"Mario",
                "text":"I'll be home soon","delivery":"compose"}}"#,
        )
        .message
        .unwrap();
        assert_eq!(
            full,
            MessageAction {
                text: "I'll be home soon".into(),
                service: Some("whatsapp".into()),
                recipient_id: Some("35699000000".into()),
                recipient_label: Some("Mario".into()),
                delivery: Some("compose".into()),
            }
        );
        let bare = parse(r#"{"v":1,"message":{"text":"on my way"}}"#).message.unwrap();
        assert_eq!((bare.service, bare.recipient_id, bare.recipient_label, bare.delivery), (None, None, None, None));
        assert_eq!(parse(r#"{"v":1,"message":{"service":"whatsapp","recipient_label":"Mario"}}"#).message, None);
        assert_eq!(parse(r#"{"v":1,"speak":"hi"}"#).message, None);
    }

    #[test]
    fn a_reply_needs_its_text_and_no_recipient_means_the_newest() {
        let named = parse(r#"{"v":1,"reply":{"recipient_label":"Gail","text":"On my way"}}"#).reply.unwrap();
        assert_eq!((named.recipient_label.as_deref(), named.text.as_str()), (Some("Gail"), "On my way"));
        let newest = parse(r#"{"v":1,"reply":{"text":"On my way"}}"#).reply.unwrap();
        assert_eq!(newest.recipient_label, None);
        assert_eq!(parse(r#"{"v":1,"reply":{"recipient_label":"Gail"}}"#).reply, None);
        assert_eq!(parse(r#"{"v":1,"speak":"hi"}"#).reply, None);
    }

    #[test]
    fn a_skill_suggestion_counts_as_presentation_and_its_name_falls_back_to_its_id() {
        let env = PresentationEnvelope::parse(
            r#"{"v":1,"speak":"The Reminder skill can do that. Want to add it?",
                "suggest_skill":{"id":"dev.heyari.reminder","name":"Reminder"}}"#,
            "",
        )
        .unwrap();
        assert_eq!(env.suggest_skill, Some(SkillSuggestion { id: "dev.heyari.reminder".into(), name: "Reminder".into() }));
        assert!(env.has_presentation());
        let unnamed = parse(r#"{"v":1,"suggest_skill":{"id":"dev.heyari.reminder"}}"#).suggest_skill.unwrap();
        assert_eq!(unnamed.name, "dev.heyari.reminder");
        assert_eq!(parse(r#"{"v":1,"suggest_skill":{"name":"Reminder"}}"#).suggest_skill, None);
        assert!(!parse(r#"{"v":1,"speak":"hi"}"#).has_presentation());
    }

    #[test]
    fn confidence_is_read_in_any_case_and_high_by_default() {
        let partial = parse(r#"{"v":1,"confidence":"PARTIAL","unparsed":"tomorrow-ish"}"#);
        assert_eq!((partial.confidence, partial.unparsed.as_deref()), (Confidence::Partial, Some("tomorrow-ish")));
        assert_eq!(parse(r#"{"v":1,"confidence":"low"}"#).confidence, Confidence::Low);
        assert_eq!(parse(r#"{"v":1,"confidence":"unsure"}"#).confidence, Confidence::High);
    }
}
