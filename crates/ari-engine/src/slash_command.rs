//! Typed `/command` shortcuts for Ari's built-in commands, the ones a user
//! otherwise says as a phrase ("clear history", "remember that …"). The names
//! are the same in every language; each expands to its phrase in the active
//! locale, so a command does exactly what saying it would.

use crate::{
    capture_prefixes, clear_conversation_phrases, forget_all_phrases, forget_prefixes,
    recall_query_phrases,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlashCommand {
    Clear,
    Remember,
    Forget,
    ForgetAll,
    Recall,
}

impl SlashCommand {
    /// Every command, in the order a frontend lists them.
    pub const ALL: [SlashCommand; 5] = [
        SlashCommand::Clear,
        SlashCommand::Remember,
        SlashCommand::Forget,
        SlashCommand::ForgetAll,
        SlashCommand::Recall,
    ];

    /// What follows the `/`.
    pub fn name(self) -> &'static str {
        match self {
            SlashCommand::Clear => "clear",
            SlashCommand::Remember => "remember",
            SlashCommand::Forget => "forget",
            SlashCommand::ForgetAll => "forget-all",
            SlashCommand::Recall => "recall",
        }
    }

    /// Whether the command needs words after it ("/remember I like tea").
    pub fn takes_argument(self) -> bool {
        matches!(self, SlashCommand::Remember | SlashCommand::Forget)
    }

    // it: DRAFT, needs native review (see plan Global Constraints).
    pub fn description_for(self, locale: &str) -> &'static str {
        match (self, locale) {
            (SlashCommand::Clear, "it") => "Cancella la conversazione e ricomincia",
            (SlashCommand::Clear, _) => "Clear the conversation and start over",
            (SlashCommand::Remember, "it") => "Ricorda qualcosa su di te",
            (SlashCommand::Remember, _) => "Remember something about you",
            (SlashCommand::Forget, "it") => "Dimentica una cosa che mi hai detto",
            (SlashCommand::Forget, _) => "Forget something you told me",
            (SlashCommand::ForgetAll, "it") => "Dimentica tutto su di te",
            (SlashCommand::ForgetAll, _) => "Forget everything about you",
            (SlashCommand::Recall, "it") => "Cosa ricordo di te",
            (SlashCommand::Recall, _) => "What I remember about you",
        }
    }

    /// The phrase this command stands for in `locale`. The prefix lists run
    /// longest-first, so their last entry is the bare verb ("remember ").
    fn phrase(self, locale: &str, argument: &str) -> String {
        let prefixed = |prefixes: &[&str]| {
            let verb = prefixes.last().expect("every locale has a prefix");
            format!("{verb}{argument}").trim_end().to_string()
        };
        match self {
            SlashCommand::Clear => clear_conversation_phrases(locale)[0].to_string(),
            SlashCommand::Remember => prefixed(capture_prefixes(locale)),
            SlashCommand::Forget => prefixed(forget_prefixes(locale)),
            SlashCommand::ForgetAll => forget_all_phrases(locale)[0].to_string(),
            SlashCommand::Recall => recall_query_phrases(locale)[0].to_string(),
        }
    }
}

/// The phrase a typed `/command` stands for in `locale`, or `None` when the
/// post-`normalize_input` utterance `normalized` isn't one of the commands.
pub(crate) fn expand(normalized: &str, locale: &str) -> Option<String> {
    let rest = normalized.strip_prefix('/')?;
    let (name, argument) = rest.split_once(' ').unwrap_or((rest, ""));
    let command = SlashCommand::ALL.into_iter().find(|c| c.name() == name)?;
    Some(command.phrase(locale, argument.trim()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Engine;
    use ari_core::Response;

    #[test]
    fn expands_each_command_to_its_english_phrase() {
        assert_eq!(expand("/clear", "en").as_deref(), Some("clear history"));
        assert_eq!(expand("/remember i like tea", "en").as_deref(), Some("remember i like tea"));
        assert_eq!(expand("/forget i like tea", "en").as_deref(), Some("forget i like tea"));
        assert_eq!(expand("/forget-all", "en").as_deref(), Some("forget everything about me"));
        assert_eq!(expand("/recall", "en").as_deref(), Some("what do you remember about me"));
    }

    #[test]
    fn expands_each_command_to_its_italian_phrase() {
        assert_eq!(expand("/clear", "it").as_deref(), Some("cancella la cronologia"));
        assert_eq!(expand("/remember vivo a roma", "it").as_deref(), Some("ricorda vivo a roma"));
        assert_eq!(expand("/forget vivo a roma", "it").as_deref(), Some("dimentica vivo a roma"));
        assert_eq!(expand("/forget-all", "it").as_deref(), Some("dimentica tutto su di me"));
        assert_eq!(expand("/recall", "it").as_deref(), Some("cosa ti ricordi di me"));
    }

    #[test]
    fn a_locale_without_phrases_expands_to_english() {
        assert_eq!(expand("/recall", "fr").as_deref(), Some("what do you remember about me"));
    }

    #[test]
    fn a_command_missing_its_argument_expands_to_the_bare_verb() {
        assert_eq!(expand("/remember", "en").as_deref(), Some("remember"));
        assert_eq!(expand("/forget", "it").as_deref(), Some("dimentica"));
    }

    #[test]
    fn anything_else_is_not_a_command() {
        assert_eq!(expand("clear", "en"), None);
        assert_eq!(expand("/", "en"), None);
        assert_eq!(expand("/rec", "en"), None);
        assert_eq!(expand("/clearance", "en"), None);
        assert_eq!(expand("8 / 2", "en"), None);
        assert_eq!(expand("/card-demo 30 pasta", "en"), None);
    }

    #[test]
    fn only_remember_and_forget_take_an_argument() {
        let taking: Vec<&str> = SlashCommand::ALL
            .into_iter()
            .filter(|c| c.takes_argument())
            .map(|c| c.name())
            .collect();
        assert_eq!(taking, vec!["remember", "forget"]);
    }

    #[test]
    fn typed_commands_run_what_their_phrases_run() {
        let engine = Engine::new();
        let (resp, _) = engine.process_input_traced("/remember I like tea");
        assert!(
            matches!(resp, Response::Text(ref s) if s == "Got it — I'll remember that."),
            "got {resp:?}"
        );
        assert_eq!(engine.remembered_facts(), vec!["i like tea".to_string()]);

        let (resp, _) = engine.process_input_traced("/recall");
        assert!(
            matches!(resp, Response::Text(ref s) if s == "Here's what I remember about you:\n- i like tea"),
            "got {resp:?}"
        );

        let (resp, _) = engine.process_input_traced("/forget i like tea");
        assert!(matches!(resp, Response::Text(ref s) if s == "Okay, I've forgotten that."), "got {resp:?}");
        assert!(engine.remembered_facts().is_empty());

        engine.process_input_traced("/remember I live in Carlisle");
        let (resp, _) = engine.process_input_traced("/forget-all");
        assert!(
            matches!(resp, Response::Text(ref s) if s == "Okay, I've forgotten everything I knew about you."),
            "got {resp:?}"
        );
        assert!(engine.remembered_facts().is_empty());

        engine.record_assistant_turn("q1", "a1");
        let (resp, _) = engine.process_input_traced("/clear");
        assert!(matches!(resp, Response::Text(ref s) if s == "Done, fresh start."), "got {resp:?}");
        assert!(engine.conversation_context().is_empty());
        assert!(engine.take_clear_signal());
    }

    #[test]
    fn typed_commands_work_the_same_in_italian() {
        let mut engine = Engine::new();
        engine.set_locale("it".to_string());
        let (resp, _) = engine.process_input_traced("/remember vivo a Carlisle");
        assert!(matches!(resp, Response::Text(ref s) if s == "Fatto, me ne ricorderò."), "got {resp:?}");
        assert_eq!(engine.remembered_facts(), vec!["vivo a carlisle".to_string()]);
    }
}
