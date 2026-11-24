//! Words that answer what is on screen instead of going to the engine:
//! "stop" over a sounding alert, or a button's label with a card showing.
//! Moved from ari-android's `AlertVoiceIntercept.kt` and
//! `CardActionVoiceIntercept.kt`. The frontend lists what is on screen
//! ([`SessionHost::intercept_targets`]) and acts on what this picks.

use super::{Intercepted, SessionHost};

/// A sounding alert. `stop_label`: the label of its Stop button, which the
/// skill chose, in whatever language.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlertTarget {
    pub id: String,
    pub stop_label: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CardButton {
    pub id: String,
    pub label: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CardTarget {
    pub id: String,
    pub buttons: Vec<CardButton>,
}

/// What is on screen: the sounding alerts, and the cards, oldest first.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InterceptTargets {
    pub alerts: Vec<AlertTarget>,
    pub cards: Vec<CardTarget>,
}

/// What pressing a card's button did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CardPressed {
    /// The card has gone since [`SessionHost::intercept_targets`]: the words
    /// go to the engine.
    Gone,
    /// The button had nothing to say.
    Silent,
    /// The button answered. `attachments`: the answer has any, which the
    /// frontend holds, `cards` among them.
    Spoken { text: String, cards: bool, attachments: bool },
}

#[derive(Debug, PartialEq, Eq)]
enum InterceptMatch {
    Alert { alert_id: String },
    Card { card_id: String, button_id: String },
}

/// Words that stop a sounding alert when said on their own. English carries
/// the recogniser's habit of hearing "stop" as "stopped" or "stops". Any
/// language without a list of its own gets English's, as it did on Android.
fn stop_phrases(locale: &str) -> &'static [&'static str] {
    match locale {
        "it" => &["ferma", "stop", "basta"],
        _ => &["stop", "stopped", "stops"],
    }
}

/// Endings speech recognition adds to a button's label said on its own, so
/// saying the label still presses it: "cancel" heard as "cancelled" (the
/// doubled l), "canceled" or "cancels". Italian has none: there the
/// recogniser changes an ending rather than adding one, and no case of it is
/// known yet.
fn spoken_endings(locale: &str) -> &'static [&'static str] {
    match locale {
        "it" => &[],
        _ => &["s", "d", "ed", "led"],
    }
}

/// Give `text` to a sounding alert or, when `cards`, the card on screen, if
/// it answers one. The alert comes first: while one rings, a bare "stop" can
/// only mean it.
pub(crate) fn intercept(host: &dyn SessionHost, text: &str, cards: bool, locale: &str) -> Intercepted {
    match resolve(text, cards, &host.intercept_targets(), locale) {
        None => Intercepted::No,
        Some(InterceptMatch::Alert { alert_id }) => {
            host.stop_alert(&alert_id);
            Intercepted::Alert
        }
        Some(InterceptMatch::Card { card_id, button_id }) => match host.press_card_button(&card_id, &button_id) {
            CardPressed::Gone => Intercepted::No,
            CardPressed::Silent => Intercepted::CardSilent,
            CardPressed::Spoken { text, cards, attachments } => Intercepted::CardSpoken { text, cards, attachments },
        },
    }
}

/// The words must be, as a whole utterance, a stop phrase or the alert's own
/// Stop label, or a button's id or label (with one of the language's endings
/// on it, if no button matches exactly). Case and the trailing punctuation
/// typing and the recogniser both add don't count.
fn resolve(text: &str, cards: bool, targets: &InterceptTargets, locale: &str) -> Option<InterceptMatch> {
    let lower = text.trim().to_lowercase();
    let words = lower.trim_end_matches(['.', '!', '?', ',']);
    if words.is_empty() {
        return None;
    }
    let said = |name: &str| name.to_lowercase() == words;

    let stop = stop_phrases(locale).contains(&words);
    if let Some(alert) = targets.alerts.iter().find(|a| stop || a.stop_label.as_deref().is_some_and(said)) {
        return Some(InterceptMatch::Alert { alert_id: alert.id.clone() });
    }
    if !cards {
        return None;
    }

    // Only the newest card with buttons: the older ones are taken as handled,
    // and mustn't take a "yes" meant for a new question.
    let card = targets.cards.iter().rfind(|c| !c.buttons.is_empty())?;
    fn names(b: &CardButton) -> [&str; 2] {
        [&b.id, &b.label]
    }
    let button = card.buttons.iter().find(|b| names(b).into_iter().any(said)).or_else(|| {
        let endings = spoken_endings(locale);
        card.buttons.iter().find(|b| names(b).into_iter().any(|n| endings.iter().any(|e| said(&format!("{n}{e}")))))
    })?;
    Some(InterceptMatch::Card { card_id: card.id.clone(), button_id: button.id.clone() })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn alert(id: &str, stop_label: Option<&str>) -> AlertTarget {
        AlertTarget { id: id.into(), stop_label: stop_label.map(str::to_string) }
    }

    fn card(id: &str, buttons: &[(&str, &str)]) -> CardTarget {
        CardTarget {
            id: id.into(),
            buttons: buttons.iter().map(|(id, label)| CardButton { id: (*id).into(), label: (*label).into() }).collect(),
        }
    }

    fn ringing(alerts: Vec<AlertTarget>) -> InterceptTargets {
        InterceptTargets { alerts, cards: vec![] }
    }

    fn showing(cards: Vec<CardTarget>) -> InterceptTargets {
        InterceptTargets { alerts: vec![], cards }
    }

    fn stopped(id: &str) -> Option<InterceptMatch> {
        Some(InterceptMatch::Alert { alert_id: id.into() })
    }

    fn pressed(card_id: &str, button_id: &str) -> Option<InterceptMatch> {
        Some(InterceptMatch::Card { card_id: card_id.into(), button_id: button_id.into() })
    }

    #[test]
    fn stop_over_a_ringing_alert_stops_it() {
        let t = ringing(vec![alert("alert_t1", Some("Stop"))]);
        assert_eq!(resolve("stop", true, &t, "en"), stopped("alert_t1"));
    }

    #[test]
    fn the_recogniser_s_mishearings_of_stop_count_too() {
        let t = ringing(vec![alert("alert_t1", Some("Stop"))]);
        assert_eq!(resolve("stopped", true, &t, "en"), stopped("alert_t1"));
        assert_eq!(resolve("Stops.", true, &t, "en"), stopped("alert_t1"));
    }

    #[test]
    fn case_and_trailing_punctuation_don_t_matter() {
        let t = ringing(vec![alert("alert_t1", Some("Stop"))]);
        assert_eq!(resolve("  Stop!  ", true, &t, "en"), stopped("alert_t1"));
        assert_eq!(resolve("STOP.", true, &t, "en"), stopped("alert_t1"));
    }

    #[test]
    fn the_skill_s_own_stop_label_works_in_any_language() {
        // The label alone is enough: "alt" is in no language's phrase list.
        let t = ringing(vec![alert("alert_t1", Some("Alt"))]);
        assert_eq!(resolve("alt", true, &t, "de"), stopped("alert_t1"));
        assert_eq!(resolve("halt", true, &t, "de"), None);
    }

    #[test]
    fn italian_has_its_own_stop_phrases() {
        let t = ringing(vec![alert("alert_t1", None)]);
        assert_eq!(resolve("basta", true, &t, "it"), stopped("alert_t1"));
        assert_eq!(resolve("Ferma.", true, &t, "it"), stopped("alert_t1"));
        assert_eq!(resolve("stopped", true, &t, "it"), None);
        assert_eq!(resolve("basta", true, &t, "en"), None);
    }

    #[test]
    fn a_language_without_a_list_gets_english_s() {
        let t = ringing(vec![alert("alert_t1", None)]);
        assert_eq!(resolve("stopped", true, &t, "fr"), stopped("alert_t1"));
    }

    #[test]
    fn the_stop_phrases_work_on_an_alert_with_no_buttons() {
        let t = ringing(vec![alert("alert_t1", None)]);
        assert_eq!(resolve("stop", true, &t, "en"), stopped("alert_t1"));
    }

    #[test]
    fn a_stop_phrase_stops_the_first_alert_ringing() {
        let t = ringing(vec![alert("alert_t1", None), alert("alert_t2", None)]);
        assert_eq!(resolve("stop", true, &t, "en"), stopped("alert_t1"));
    }

    #[test]
    fn nothing_ringing_means_no_intercept() {
        assert_eq!(resolve("stop", true, &InterceptTargets::default(), "en"), None);
    }

    #[test]
    fn longer_requests_go_to_the_engine() {
        // "stop the timer" is the timer skill's to answer, not a bare stop.
        let t = ringing(vec![alert("alert_t1", Some("Stop"))]);
        for text in ["stop the timer", "what's the weather", "", "   ", "?"] {
            assert_eq!(resolve(text, true, &t, "en"), None, "{text:?}");
        }
    }

    #[test]
    fn an_alert_is_stopped_even_when_cards_may_not_answer() {
        let t = InterceptTargets { alerts: vec![alert("alert_t1", None)], cards: vec![card("c1", &[("stop", "Stop")])] };
        assert_eq!(resolve("stop", false, &t, "en"), stopped("alert_t1"));
    }

    #[test]
    fn the_alert_comes_before_the_card() {
        let t = InterceptTargets { alerts: vec![alert("alert_t1", None)], cards: vec![card("c1", &[("stop", "Stop")])] };
        assert_eq!(resolve("stop", true, &t, "en"), stopped("alert_t1"));
    }

    #[test]
    fn a_multi_word_label_presses_its_button() {
        let t = showing(vec![card("c1", &[("apple_music", "Apple Music")])]);
        assert_eq!(resolve("apple music", true, &t, "en"), pressed("c1", "apple_music"));
    }

    #[test]
    fn a_button_s_id_presses_it_too() {
        let t = showing(vec![card("c1", &[("spotify", "Open Spotify")])]);
        assert_eq!(resolve("spotify", true, &t, "en"), pressed("c1", "spotify"));
    }

    #[test]
    fn a_button_match_ignores_case_and_trailing_punctuation() {
        let t = showing(vec![card("c1", &[("tidal", "Tidal")])]);
        assert_eq!(resolve("TIDAL", true, &t, "en"), pressed("c1", "tidal"));
        assert_eq!(resolve("tidal!", true, &t, "en"), pressed("c1", "tidal"));
    }

    #[test]
    fn words_that_name_no_button_go_to_the_engine() {
        let t = showing(vec![card("c1", &[("spotify", "Spotify")])]);
        for text in ["deezer", "play spotify", "", "   "] {
            assert_eq!(resolve(text, true, &t, "en"), None, "{text:?}");
        }
    }

    #[test]
    fn no_card_is_pressed_when_cards_may_not_answer() {
        // The engine waits on a reply to its own question.
        let t = showing(vec![card("c1", &[("spotify", "Spotify")])]);
        assert_eq!(resolve("spotify", false, &t, "en"), None);
    }

    #[test]
    fn only_the_newest_card_with_buttons_answers() {
        let t = showing(vec![
            card("old", &[("yes", "Yes"), ("no", "No")]),
            card("new", &[("cancel", "Cancel")]),
            card("plain", &[]),
        ]);
        assert_eq!(resolve("cancel", true, &t, "en"), pressed("new", "cancel"));
        assert_eq!(resolve("yes", true, &t, "en"), None);
    }

    #[test]
    fn a_label_heard_with_an_ending_still_presses_the_button() {
        // Kroko heard "cancel" as "cancelled" on 2026-11-22, and the timer's
        // Cancel button wasn't pressed.
        let t = showing(vec![card("c1", &[("cancel", "Cancel")])]);
        for heard in ["cancelled", "Canceled.", "cancels"] {
            assert_eq!(resolve(heard, true, &t, "en"), pressed("c1", "cancel"), "{heard:?}");
        }
    }

    #[test]
    fn an_exact_match_wins_over_an_ending() {
        let t = showing(vec![card("c1", &[("stop", "Stop"), ("stops", "Stops")])]);
        assert_eq!(resolve("stops", true, &t, "en"), pressed("c1", "stops"));
    }

    #[test]
    fn only_the_language_s_endings_count() {
        let t = showing(vec![card("c1", &[("yes", "Yes")])]);
        assert_eq!(resolve("yesterday", true, &t, "en"), None);
        assert_eq!(resolve("yesss", true, &t, "en"), None);
        let cancel = showing(vec![card("c1", &[("cancel", "Cancel")])]);
        assert_eq!(resolve("cancelled", true, &cancel, "it"), None);
    }
}
