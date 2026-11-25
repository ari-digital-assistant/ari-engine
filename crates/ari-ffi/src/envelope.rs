//! [`ari_core::envelope`] across the UniFFI boundary: the parsed
//! presentation envelope, for Android to read instead of parsing the JSON
//! itself.

use ari_core::envelope::{
    Accent, AlarmAction, AlertAction, AlertSpec, ButtonStyle, Card, CardAction, Confidence, IconText, Importance,
    ListCard, ListRow, MediaAction, MessageAction, NavigateAction, NotificationAction, NotificationSpec, OnComplete,
    PresentationEnvelope, ReplyAction, SkillSuggestion, Stat, Urgency,
};

/// The envelope in `json` from skill `skill_id`, or None when it isn't one
/// this engine reads.
#[uniffi::export]
pub fn parse_presentation_envelope(json: String, skill_id: String) -> Option<FfiPresentationEnvelope> {
    PresentationEnvelope::parse(&json, &skill_id).map(Into::into)
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiPresentationEnvelope {
    pub speak: Option<String>,
    pub display: Option<String>,
    pub cards: Vec<FfiCard>,
    pub alerts: Vec<FfiAlertSpec>,
    pub notifications: Vec<FfiNotificationSpec>,
    pub launch_app: Option<String>,
    pub media: Option<FfiMediaAction>,
    pub alarm: Option<FfiAlarmAction>,
    pub navigate: Option<FfiNavigateAction>,
    pub message: Option<FfiMessageAction>,
    pub reply: Option<FfiReplyAction>,
    pub search: Option<String>,
    pub open_url: Option<String>,
    pub clipboard_text: Option<String>,
    pub dismiss_cards: Vec<String>,
    pub dismiss_notifications: Vec<String>,
    pub dismiss_alerts: Vec<String>,
    pub run_utterance: Option<String>,
    pub confidence: FfiConfidence,
    pub unparsed: Option<String>,
    pub suggest_skill: Option<FfiSkillSuggestion>,
    /// [`PresentationEnvelope::has_presentation`].
    pub has_presentation: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum FfiConfidence {
    High,
    Partial,
    Low,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiCard {
    pub id: String,
    pub skill_id: String,
    pub title: String,
    pub subtitle: Option<String>,
    pub body: Option<String>,
    pub icon: Option<String>,
    pub countdown_to_ts_ms: Option<i64>,
    pub started_at_ts_ms: Option<i64>,
    pub progress: Option<f32>,
    pub accent: FfiAccent,
    pub actions: Vec<FfiCardAction>,
    pub on_complete: Option<FfiOnComplete>,
    pub on_cancel: Option<String>,
    pub stat: Option<FfiStat>,
    pub list: Option<FfiListCard>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum FfiAccent {
    Default,
    Warning,
    Success,
    Critical,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiIconText {
    pub icon: Option<String>,
    pub text: String,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiStat {
    pub headline: String,
    pub caption: Option<String>,
    pub pill: Option<FfiIconText>,
    pub metrics: Vec<FfiIconText>,
    pub background: Option<String>,
    pub footer: Option<FfiIconText>,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiListRow {
    pub leading: String,
    pub icon: Option<String>,
    pub text: Option<String>,
    pub trailing: Option<String>,
    pub badge: Option<FfiIconText>,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiListCard {
    pub summary: Option<FfiIconText>,
    pub rows: Vec<FfiListRow>,
    pub footer: Option<FfiIconText>,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiCardAction {
    pub id: String,
    pub label: String,
    pub utterance: Option<String>,
    pub speak: Option<String>,
    pub style: FfiButtonStyle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum FfiButtonStyle {
    Default,
    Primary,
    Destructive,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiOnComplete {
    pub alert: Option<FfiAlertSpec>,
    pub dismiss_card: bool,
    pub dismiss_notifications: Vec<String>,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiAlertSpec {
    pub id: String,
    pub skill_id: String,
    pub title: String,
    pub body: Option<String>,
    pub urgency: FfiUrgency,
    pub sound: String,
    pub speech_loop: Option<String>,
    pub auto_stop_ms: i64,
    pub max_cycles: i64,
    pub full_takeover: bool,
    pub icon: Option<String>,
    pub actions: Vec<FfiAlertAction>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum FfiUrgency {
    Normal,
    High,
    Critical,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiAlertAction {
    pub id: String,
    pub label: String,
    pub utterance: Option<String>,
    pub style: FfiButtonStyle,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiNotificationSpec {
    pub id: String,
    pub skill_id: String,
    pub title: String,
    pub body: Option<String>,
    pub importance: FfiImportance,
    pub sticky: bool,
    pub countdown_to_ts_ms: Option<i64>,
    pub actions: Vec<FfiNotificationAction>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum FfiImportance {
    Min,
    Low,
    Default,
    High,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiNotificationAction {
    pub id: String,
    pub label: String,
    pub utterance: Option<String>,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiMediaAction {
    pub action: String,
    pub query: Option<String>,
    pub service: Option<String>,
    pub direction: Option<String>,
    pub level: Option<i64>,
    pub mute: Option<bool>,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiAlarmAction {
    pub op: String,
    pub hour: Option<i64>,
    pub minute: Option<i64>,
    pub message: Option<String>,
    pub days: Vec<String>,
    pub skip_ui: bool,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiNavigateAction {
    pub destination: String,
    pub mode: Option<String>,
    pub place: Option<String>,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiMessageAction {
    pub text: String,
    pub service: Option<String>,
    pub recipient_id: Option<String>,
    pub recipient_label: Option<String>,
    pub delivery: Option<String>,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiReplyAction {
    pub text: String,
    pub recipient_label: Option<String>,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiSkillSuggestion {
    pub id: String,
    pub name: String,
}

impl From<PresentationEnvelope> for FfiPresentationEnvelope {
    fn from(e: PresentationEnvelope) -> Self {
        Self {
            has_presentation: e.has_presentation(),
            speak: e.speak,
            display: e.display,
            cards: e.cards.into_iter().map(Into::into).collect(),
            alerts: e.alerts.into_iter().map(Into::into).collect(),
            notifications: e.notifications.into_iter().map(Into::into).collect(),
            launch_app: e.launch_app,
            media: e.media.map(Into::into),
            alarm: e.alarm.map(Into::into),
            navigate: e.navigate.map(Into::into),
            message: e.message.map(Into::into),
            reply: e.reply.map(Into::into),
            search: e.search,
            open_url: e.open_url,
            clipboard_text: e.clipboard_text,
            dismiss_cards: e.dismiss_cards,
            dismiss_notifications: e.dismiss_notifications,
            dismiss_alerts: e.dismiss_alerts,
            run_utterance: e.run_utterance,
            confidence: match e.confidence {
                Confidence::High => FfiConfidence::High,
                Confidence::Partial => FfiConfidence::Partial,
                Confidence::Low => FfiConfidence::Low,
            },
            unparsed: e.unparsed,
            suggest_skill: e.suggest_skill.map(Into::into),
        }
    }
}

impl From<Card> for FfiCard {
    fn from(c: Card) -> Self {
        Self {
            id: c.id,
            skill_id: c.skill_id,
            title: c.title,
            subtitle: c.subtitle,
            body: c.body,
            icon: c.icon,
            countdown_to_ts_ms: c.countdown_to_ts_ms,
            started_at_ts_ms: c.started_at_ts_ms,
            progress: c.progress,
            accent: match c.accent {
                Accent::Default => FfiAccent::Default,
                Accent::Warning => FfiAccent::Warning,
                Accent::Success => FfiAccent::Success,
                Accent::Critical => FfiAccent::Critical,
            },
            actions: c.actions.into_iter().map(Into::into).collect(),
            on_complete: c.on_complete.map(Into::into),
            on_cancel: c.on_cancel,
            stat: c.stat.map(Into::into),
            list: c.list.map(Into::into),
        }
    }
}

impl From<IconText> for FfiIconText {
    fn from(i: IconText) -> Self {
        Self { icon: i.icon, text: i.text }
    }
}

impl From<Stat> for FfiStat {
    fn from(s: Stat) -> Self {
        Self {
            headline: s.headline,
            caption: s.caption,
            pill: s.pill.map(Into::into),
            metrics: s.metrics.into_iter().map(Into::into).collect(),
            background: s.background,
            footer: s.footer.map(Into::into),
        }
    }
}

impl From<ListRow> for FfiListRow {
    fn from(r: ListRow) -> Self {
        Self { leading: r.leading, icon: r.icon, text: r.text, trailing: r.trailing, badge: r.badge.map(Into::into) }
    }
}

impl From<ListCard> for FfiListCard {
    fn from(l: ListCard) -> Self {
        Self {
            summary: l.summary.map(Into::into),
            rows: l.rows.into_iter().map(Into::into).collect(),
            footer: l.footer.map(Into::into),
        }
    }
}

impl From<ButtonStyle> for FfiButtonStyle {
    fn from(s: ButtonStyle) -> Self {
        match s {
            ButtonStyle::Default => Self::Default,
            ButtonStyle::Primary => Self::Primary,
            ButtonStyle::Destructive => Self::Destructive,
        }
    }
}

impl From<CardAction> for FfiCardAction {
    fn from(a: CardAction) -> Self {
        Self { id: a.id, label: a.label, utterance: a.utterance, speak: a.speak, style: a.style.into() }
    }
}

impl From<OnComplete> for FfiOnComplete {
    fn from(o: OnComplete) -> Self {
        Self {
            alert: o.alert.map(Into::into),
            dismiss_card: o.dismiss_card,
            dismiss_notifications: o.dismiss_notifications,
        }
    }
}

impl From<AlertSpec> for FfiAlertSpec {
    fn from(a: AlertSpec) -> Self {
        Self {
            id: a.id,
            skill_id: a.skill_id,
            title: a.title,
            body: a.body,
            urgency: match a.urgency {
                Urgency::Normal => FfiUrgency::Normal,
                Urgency::High => FfiUrgency::High,
                Urgency::Critical => FfiUrgency::Critical,
            },
            sound: a.sound,
            speech_loop: a.speech_loop,
            auto_stop_ms: a.auto_stop_ms,
            max_cycles: a.max_cycles,
            full_takeover: a.full_takeover,
            icon: a.icon,
            actions: a.actions.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<AlertAction> for FfiAlertAction {
    fn from(a: AlertAction) -> Self {
        Self { id: a.id, label: a.label, utterance: a.utterance, style: a.style.into() }
    }
}

impl From<NotificationSpec> for FfiNotificationSpec {
    fn from(n: NotificationSpec) -> Self {
        Self {
            id: n.id,
            skill_id: n.skill_id,
            title: n.title,
            body: n.body,
            importance: match n.importance {
                Importance::Min => FfiImportance::Min,
                Importance::Low => FfiImportance::Low,
                Importance::Default => FfiImportance::Default,
                Importance::High => FfiImportance::High,
            },
            sticky: n.sticky,
            countdown_to_ts_ms: n.countdown_to_ts_ms,
            actions: n.actions.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<NotificationAction> for FfiNotificationAction {
    fn from(a: NotificationAction) -> Self {
        Self { id: a.id, label: a.label, utterance: a.utterance }
    }
}

impl From<MediaAction> for FfiMediaAction {
    fn from(m: MediaAction) -> Self {
        Self {
            action: m.action,
            query: m.query,
            service: m.service,
            direction: m.direction,
            level: m.level,
            mute: m.mute,
        }
    }
}

impl From<AlarmAction> for FfiAlarmAction {
    fn from(a: AlarmAction) -> Self {
        Self { op: a.op, hour: a.hour, minute: a.minute, message: a.message, days: a.days, skip_ui: a.skip_ui }
    }
}

impl From<NavigateAction> for FfiNavigateAction {
    fn from(n: NavigateAction) -> Self {
        Self { destination: n.destination, mode: n.mode, place: n.place }
    }
}

impl From<MessageAction> for FfiMessageAction {
    fn from(m: MessageAction) -> Self {
        Self {
            text: m.text,
            service: m.service,
            recipient_id: m.recipient_id,
            recipient_label: m.recipient_label,
            delivery: m.delivery,
        }
    }
}

impl From<ReplyAction> for FfiReplyAction {
    fn from(r: ReplyAction) -> Self {
        Self { text: r.text, recipient_label: r.recipient_label }
    }
}

impl From<SkillSuggestion> for FfiSkillSuggestion {
    fn from(s: SkillSuggestion) -> Self {
        Self { id: s.id, name: s.name }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_parsed_envelope_crosses_with_what_decides_its_presentation() {
        let env = parse_presentation_envelope(
            r#"{"v":1,"speak":"Pasta timer set.","cards":[{"id":"c","title":"Pasta","countdown_to_ts_ms":5,
                "actions":[{"id":"cancel","label":"Cancel","style":"destructive"}]}]}"#
                .into(),
            "dev.heyari.timer".into(),
        )
        .unwrap();
        assert!(env.has_presentation);
        assert_eq!(env.speak.as_deref(), Some("Pasta timer set."));
        assert_eq!(env.cards[0].skill_id, "dev.heyari.timer");
        assert_eq!(env.cards[0].countdown_to_ts_ms, Some(5));
        assert_eq!(env.cards[0].actions[0].style, FfiButtonStyle::Destructive);
        assert_eq!(env.confidence, FfiConfidence::High);
    }

    #[test]
    fn an_envelope_the_engine_cannot_read_crosses_as_none() {
        assert!(parse_presentation_envelope(r#"{"v":2}"#.into(), String::new()).is_none());
    }
}
