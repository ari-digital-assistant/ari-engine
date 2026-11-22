//! The session's decisions that need no state, from the top of
//! `VoiceSession.kt` and from `BargeIn.kt`.

use super::captures::{UtteranceOutcome, UtteranceTurn};
use super::{EngineReply, ReplyFlags};

fn flags(reply: &EngineReply) -> Option<ReplyFlags> {
    match reply {
        EngineReply::Text { flags, .. } | EngineReply::Action { flags, .. } => Some(*flags),
        _ => None,
    }
}

/// Does this reply ask for a spoken answer, and so re-arm the mic without a
/// second wake word? Only text and action replies carry the flag; a binary
/// or not-understood reply never re-arms.
pub fn should_rearm(reply: &EngineReply) -> bool {
    flags(reply).is_some_and(|f| f.rearm)
}

pub fn should_enter_conversation(reply: &EngineReply) -> bool {
    flags(reply).is_some_and(|f| f.enter_conversation)
}

pub fn should_exit_conversation(reply: &EngineReply) -> bool {
    flags(reply).is_some_and(|f| f.exit_conversation)
}

/// Did this turn change the engine's remembered facts (a remember or forget
/// command)?
pub fn should_persist_facts(reply: &EngineReply) -> bool {
    flags(reply).is_some_and(|f| f.facts_changed)
}

/// Did the user clear the conversation this turn ("clear history")? The
/// engine has already forgotten it; the log is cleared to match.
pub fn should_clear_conversation(reply: &EngineReply) -> bool {
    flags(reply).is_some_and(|f| f.clear_conversation)
}

/// The name verdict a wake turn is judged on. A name the early check heard
/// stands: the user has already seen Ari answer to it, and a full transcript
/// can miss a name the short clip caught ("Hayori navigate to McDonald's",
/// 2026-09-10), so the full one never turns the wake away after that.
pub fn wake_name_verdict(name_heard_early: bool, name_matched: Option<bool>) -> Option<bool> {
    if name_heard_early { Some(true) } else { name_matched }
}

/// Whether a partial shows a quiet wake: it holds the name, or the locale
/// forms no verdict, where nothing will ever say no and hiding would only
/// lose a real call. Showing is all it decides; the whole transcript is still
/// judged. The buffered paths' empty partial never shows one, which keeps
/// their quiet wakes hidden until the transcript is in.
pub fn partial_reveals_wake(name_matched: Option<bool>) -> bool {
    name_matched != Some(false)
}

/// Post-hoc wake verification. The 64 KB detector fires on phonetic shape
/// and saturates its confidence even when it's wrong, so the threshold has no
/// headroom left; the recogniser, which is already transcribing the same
/// audio from the listen feed's rewind, is the second opinion. A wake turn
/// whose transcript contains speech but no "ari"-ish name token was not
/// addressed to Ari.
///
/// Fails open on every ambiguity: a wrongly rejected genuine command reads as
/// "Ari ignored me", which is a worse experience than a spurious chime.
pub fn should_accept_wake(verify_wake: bool, raw: Option<&str>, name_matched: Option<bool>) -> bool {
    match (verify_wake, name_matched, raw) {
        (false, _, _) => true,
        (_, None, _) => true,
        (_, _, None) => true,
        (_, _, Some(raw)) if raw.trim().is_empty() => true,
        (_, Some(matched), _) => matched,
    }
}

/// A wake turn whose transcript came back blank: no speech in the wake
/// phrase's own pre-roll or after it, so nobody said "Hey Ari".
/// [`should_accept_wake`] fails open on a blank transcript, which is right for
/// the question it answers, but the accept path would then file this wake's
/// pre-roll as a confirmed true positive, so this is asked first.
///
/// A missing `raw` is unknown rather than blank.
pub fn is_no_speech_wake(verify_wake: bool, raw: Option<&str>) -> bool {
    verify_wake && raw.is_some_and(|r| r.trim().is_empty())
}

/// Which kind of turn produced this utterance, for the capture sidecar.
/// Sampled at the top of a turn: `awaiting_reply` is consumed part-way
/// through dispatch, and `talk_mode` is set by the reply that enters
/// continuous mode, so the turn that opens a "let's talk" session is still an
/// opening one.
pub fn utterance_turn(talk_mode: bool, awaiting_reply: bool) -> UtteranceTurn {
    if talk_mode {
        UtteranceTurn::Talk
    } else if awaiting_reply {
        UtteranceTurn::Reply
    } else {
        UtteranceTurn::Opening
    }
}

/// What became of the transcript. `corrected` is true when a retry layer
/// supplied the text the engine finally acted on: the signal that Ari needed
/// more than one go at hearing this, which is exactly what the capture exists
/// to surface.
pub fn utterance_outcome(reply: &EngineReply, corrected: bool) -> UtteranceOutcome {
    if matches!(reply, EngineReply::NotUnderstood { .. }) {
        UtteranceOutcome::NotUnderstood
    } else if corrected {
        UtteranceOutcome::Rescued
    } else {
        UtteranceOutcome::Answered
    }
}

/// How the engine replied, for the capture sidecar.
pub fn response_label(reply: &EngineReply) -> String {
    match reply {
        EngineReply::Text { .. } => "text".into(),
        EngineReply::Action { skill_id, .. } => format!("action({skill_id})"),
        EngineReply::Binary { mime, .. } => format!("binary({mime})"),
        EngineReply::NotUnderstood { .. } => "not-understood".into(),
    }
}

/// Barge-in only runs when the user enabled it and the device actually
/// offers a hardware echo canceller. Without one, a live mic during speech
/// would transcribe Ari, so talk mode stays turn-based.
pub fn barge_in_effective(toggle_enabled: bool, echo_canceller: bool) -> bool {
    toggle_enabled && echo_canceller
}

/// Cut Ari off only when he is actually speaking and barge-in is effective.
pub fn should_cut_tts(speaking: bool, barge_in_effective: bool) -> bool {
    speaking && barge_in_effective
}

/// A final tagged with a turn id other than the current one is a straggler
/// from a superseded turn (a decode that landed after the speech was cut)
/// and must be dropped so it can't fire a phantom turn.
pub fn is_stale_turn(final_turn_id: u64, current_turn_id: u64) -> bool {
    final_turn_id != current_turn_id
}

/// A barge-in turn's speech, finishing, may clear the shared `speaking` flag
/// only if it is still the current one. On a barge the next turn sets
/// `speaking` and starts speech of its own before the cut one reports
/// finished; that late report must not clear its successor's flag. With no
/// current speech (a dismiss cleared it, and the flag with it) nobody owns
/// the flag.
pub fn owns_speaking_flag(current: Option<u64>, finished: u64) -> bool {
    current == Some(finished)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(flags: ReplyFlags) -> EngineReply {
        EngineReply::Text { body: "Which one?".into(), markdown: None, flags }
    }

    fn action(flags: ReplyFlags) -> EngineReply {
        EngineReply::Action { json: "{\"v\":1}".into(), skill_id: "dev.heyari.music".into(), flags }
    }

    fn not_understood() -> EngineReply {
        EngineReply::NotUnderstood { body: "Sorry, I didn't catch that.".into() }
    }

    fn binary() -> EngineReply {
        EngineReply::Binary { mime: "image/png".into(), len: 4 }
    }

    const NONE: ReplyFlags = ReplyFlags {
        rearm: false,
        enter_conversation: false,
        exit_conversation: false,
        facts_changed: false,
        clear_conversation: false,
    };

    #[test]
    fn text_and_action_rearm_only_when_flagged() {
        let rearm = ReplyFlags { rearm: true, ..NONE };
        assert!(should_rearm(&text(rearm)));
        assert!(should_rearm(&action(rearm)));
        assert!(!should_rearm(&text(NONE)));
        assert!(!should_rearm(&action(NONE)));
    }

    #[test]
    fn not_understood_and_binary_never_rearm() {
        assert!(!should_rearm(&not_understood()));
        assert!(!should_rearm(&binary()));
    }

    #[test]
    fn conversation_flags_are_read_only_where_set() {
        let enter = ReplyFlags { enter_conversation: true, ..NONE };
        assert!(should_enter_conversation(&text(enter)));
        assert!(should_enter_conversation(&action(enter)));
        assert!(!should_enter_conversation(&text(NONE)));
        assert!(!should_enter_conversation(&not_understood()));

        let exit = ReplyFlags { exit_conversation: true, ..NONE };
        assert!(should_exit_conversation(&text(exit)));
        assert!(should_exit_conversation(&action(exit)));
        assert!(!should_exit_conversation(&text(NONE)));
        assert!(!should_exit_conversation(&not_understood()));

        let clear = ReplyFlags { clear_conversation: true, ..NONE };
        assert!(should_clear_conversation(&text(clear)));
        assert!(should_clear_conversation(&action(clear)));
        assert!(!should_clear_conversation(&text(NONE)));
        assert!(!should_clear_conversation(&not_understood()));

        let facts = ReplyFlags { facts_changed: true, ..NONE };
        assert!(should_persist_facts(&text(facts)));
        assert!(!should_persist_facts(&text(NONE)));
        assert!(!should_persist_facts(&binary()));
    }

    #[test]
    fn non_wake_turns_are_never_verified() {
        assert!(should_accept_wake(false, Some("hey there mate"), Some(false)));
    }

    #[test]
    fn a_missing_verdict_fails_open() {
        assert!(should_accept_wake(true, Some("hey there mate"), None));
    }

    #[test]
    fn a_blank_transcript_fails_open() {
        assert!(should_accept_wake(true, Some(""), Some(false)));
        assert!(should_accept_wake(true, Some("   "), Some(false)));
        assert!(should_accept_wake(true, None, Some(false)));
    }

    #[test]
    fn a_wake_turn_is_judged_on_its_name_token() {
        assert!(should_accept_wake(true, Some("hey ari whats the weather"), Some(true)));
        assert!(!should_accept_wake(true, Some("hey there mate"), Some(false)));
    }

    #[test]
    fn a_name_heard_early_is_not_taken_back_by_the_whole_transcript() {
        let raw = Some("Hayori navigate to McDonald's");
        assert!(should_accept_wake(true, raw, wake_name_verdict(true, Some(false))));
        assert!(!should_accept_wake(true, raw, wake_name_verdict(false, Some(false))));
        assert_eq!(wake_name_verdict(false, None), None);
    }

    #[test]
    fn a_quiet_wake_shows_on_a_partial_with_the_name_or_where_no_verdict_is_possible() {
        assert!(partial_reveals_wake(Some(true)));
        assert!(partial_reveals_wake(None));
        assert!(!partial_reveals_wake(Some(false)));
    }

    #[test]
    fn talk_mode_outranks_awaiting_reply_when_labelling_a_turn() {
        assert_eq!(utterance_turn(true, true), UtteranceTurn::Talk);
        assert_eq!(utterance_turn(true, false), UtteranceTurn::Talk);
        assert_eq!(utterance_turn(false, true), UtteranceTurn::Reply);
        assert_eq!(utterance_turn(false, false), UtteranceTurn::Opening);
    }

    #[test]
    fn a_retry_corrected_transcript_is_a_rescue_unless_nothing_understood_it() {
        let answered = EngineReply::Text { body: "Sunny.".into(), markdown: None, flags: NONE };
        assert_eq!(utterance_outcome(&answered, true), UtteranceOutcome::Rescued);
        assert_eq!(utterance_outcome(&answered, false), UtteranceOutcome::Answered);
        assert_eq!(utterance_outcome(&not_understood(), true), UtteranceOutcome::NotUnderstood);
        assert_eq!(utterance_outcome(&not_understood(), false), UtteranceOutcome::NotUnderstood);
    }

    #[test]
    fn the_response_label_names_the_skill_that_answered() {
        assert_eq!(response_label(&action(NONE)), "action(dev.heyari.music)");
        assert_eq!(response_label(&text(NONE)), "text");
        assert_eq!(response_label(&binary()), "binary(image/png)");
        assert_eq!(response_label(&not_understood()), "not-understood");
    }

    #[test]
    fn a_wake_whose_transcript_came_back_blank_heard_no_speech() {
        assert!(is_no_speech_wake(true, Some("")));
        assert!(is_no_speech_wake(true, Some("  ")));
        assert!(!is_no_speech_wake(true, Some("turn on the lights")));
        assert!(!is_no_speech_wake(true, None));
        assert!(!is_no_speech_wake(false, Some("")));
    }

    #[test]
    fn barge_in_needs_the_toggle_and_an_echo_canceller() {
        for (toggle, aec, expected) in [(true, true, true), (true, false, false), (false, true, false), (false, false, false)] {
            assert_eq!(barge_in_effective(toggle, aec), expected, "{toggle},{aec}");
        }
    }

    #[test]
    fn speech_is_cut_only_while_speaking_with_barge_in() {
        assert!(should_cut_tts(true, true));
        assert!(!should_cut_tts(false, true));
        assert!(!should_cut_tts(true, false));
    }

    #[test]
    fn a_turn_is_stale_when_its_id_differs() {
        assert!(is_stale_turn(3, 4));
        assert!(!is_stale_turn(4, 4));
    }

    #[test]
    fn only_the_current_speech_owns_the_speaking_flag() {
        assert!(owns_speaking_flag(Some(7), 7));
        assert!(!owns_speaking_flag(Some(8), 7));
        assert!(!owns_speaking_flag(None, 7));
    }
}
