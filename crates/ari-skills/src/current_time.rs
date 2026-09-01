use ari_core::{ExampleUtterance, Response, Skill, SkillContext, Specificity};
use chrono::Local;

// English + Italian. The scorer is locale-agnostic; the words don't
// collide across languages, so a single union table keeps Stage 1
// keyword routing fast (no cloud round-trip needed for "che ora è").
//
// Italian trigger shapes after `normalize_input` (lowercases, strips
// elisions like `l'ora` → `l ora`):
//   - "che ora è" / "a che ora" → ["che", "ora"]
//   - "che ore sono" / "che ore" → ["che", "ore"]
//   - "dimmi l'ora" → "dimmi l ora" → ["dimmi", "ora"]
//   - "ora attuale" → ["ora", "attuale"]
const TRIGGER_PHRASES: &[&[&str]] = &[
    // English
    &["what", "time"],
    &["current", "time"],
    &["tell", "time"],
    &["what is", "time"],
    // Italian
    &["che", "ora"],
    &["che", "ore"],
    &["dimmi", "ora"],
    &["ora", "attuale"],
];

// Router training examples. Natural raw text as a user would actually say it.
// (Whether the generator should normalise these to match inference is the
// parity spike's question — not this file's.)
const CURRENT_TIME_EXAMPLES_EN: &[ExampleUtterance] = &[
    ExampleUtterance { text: "what time is it now", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "what is the time right now", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "can you check the time", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "tell me what time it is", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "give me the time", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "what time is it at the moment", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "do you have the current time", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "what is the time please", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "tell me the time please", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "i need the time", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "what is the exact time", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "what is the precise time", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "show me the time", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "read me the time", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "what is the time on the clock", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what does the clock say", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what time is it exactly", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "hey what is the time", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "so what time is it", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "just tell me the time", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "what time is it currently", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what is the local time", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what is the time here", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "could you check the time for me", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "mind giving me the time", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "would you tell me the time", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "what is it saying on the clock", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what hour are we at", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what hour is it now", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it morning yet", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "is it afternoon already", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it evening now", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it past noon", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "has it gone noon", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it past midnight", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "are we into the afternoon yet", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "is it still morning", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "is it nearly evening", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it getting late", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "how late is it now", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it too late to ring someone", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it late in the evening", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it early still", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "how early is it right now", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what is the time of day", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "is it am or pm", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it am or pm right now", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "are we in the am or the pm", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "my watch stopped what time is it", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "my watch died what is the time", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "my phone is off what time is it", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "the clock on the wall stopped can you tell me the time", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "my watch says 12 is that right", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is my clock correct what time is it really", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "i think my watch is slow what is the actual time", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "the clock stopped ages ago what is the real time", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "my watchs battery went what is the time", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "not sure my clock is right what time is it", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "my watch just died", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "my watch battery has gone flat", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "my watch isnt ticking anymore", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "my watch has frozen", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "my watch stopped in the night", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "my watch has been stuck for hours", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "the wall clock has stopped", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "the kitchen clock is wrong", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "the clock in the hall has stopped", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "the clock seems to have stopped", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "i totally lost track of the time", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "i have no idea what time it is", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "no clue what time it is right now", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "i lost all sense of time what is it now", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "cant tell what time it is anymore", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "i wasnt watching the clock what is the time", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "i zoned out what is the time", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "time got away from me what is it now", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "is it time to go yet", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "is it home time yet", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "is it lunchtime yet", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it nearly lunchtime", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it dinner time yet", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it time for bed", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "is it knocking off time yet", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it beer o clock yet", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it that late already", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "surely its not that late what time is it", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "wait what time is it", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "hang on what is the time", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "i need to know the time", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "i have to catch a train what time is it", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what is the time i cant be late", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "how much time before noon", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "tell me the hour", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "what o clock is it", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what is the time now please", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "quick tell me the time", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "gimme the time", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "the time please", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "time check", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "give me a time check", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "can i get a time check", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "what is the time reading", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "read out the time", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "announce the time", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "speak the time", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "say the time", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "what is the clock at", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "current time", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "how long until midday", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "how far off is noon", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "roughly what time is it", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "approximately what time is it", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "any idea of the time", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "got any idea what time it is", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "you know what time it is", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "do you know the time offhand", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "would you happen to have the time", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "sorry do you have the time", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "excuse me what is the time", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "pardon me what time is it", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what is the time mate", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "oi what is the time", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "tell us the time", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "what time you got", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "what you got for the time", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "what is the time then", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "so how late are we", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "are we late what is the time", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "is it lunch time already", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "has the morning gone", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is the morning over", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "is it still the afternoon", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what is the time getting on to", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "is it getting on for evening", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "how far into the day are we", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "whereabouts are we in the day", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "is the day nearly done", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "is it close to midnight", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "are we near noon", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is noon close", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it around lunchtime", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it roughly dinner time", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what is the time about now", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "about what time is it", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "what would the time be", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "what is the time meant to be", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "what should the clock say by now", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "did i oversleep what is the time", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "i overslept didnt i what is the time", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it time to wake up", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "is it time to get up yet", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "what is the time i think i slept in", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "i must have dozed off what is the time", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "how long was i out what is the time", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "i nodded off what time is it now", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "crikey what time is it", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "has it turned 3 yet", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "has it gone 5 yet", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it past 8 yet", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it gone 10 already", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "has it hit noon yet", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it 9 oclock yet", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it after 6 now", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "are we past 7", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it before 9 still", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what is the time roughly now", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "hey ari what is the time", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "ari what time is it", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "ari tell me the time", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "ari do you have the time", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "what is the time love", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "tell me the time real quick", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "1 sec what is the time", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "before i forget what is the time", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "what is the time currently showing", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "how many hours till midnight", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "how long left in the day", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "how much of the day is left", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "is it still office hours", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "are we still in working hours", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it too early to head out", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it late enough to leave yet", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what is the time getting on for", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "is it past teatime yet", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "how far along is the day now", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "do you know what time it is by any chance", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "would you kindly tell me the time", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "could you possibly tell me the time", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "any chance you could tell me the time", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "if you dont mind what is the time", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "sorry to bother you what is the time", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "might i trouble you for the time", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "would you be able to tell me the time", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "do you reckon you know the time", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "if its no trouble what is the time", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "would you say its evening yet", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "has it turned to evening yet", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it evening already or not", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "reckon its evening by now", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "would you call this evening", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "has evening come around yet", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it nearly evening time", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "are we into the evening yet", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "my watch has packed in what is the time", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "my watch has conked out what time is it", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "my watch quit on me what is the time", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "my clock has died what time is it", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "the battery in my watch is dead what is the time", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "my watch wont tick what is the time", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "my wristwatch has stopped so what is the time", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "roughly what hour is it now", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "about what hour are we on", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what time is it", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "what is the time", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "tell me the time", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "what time do you have", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "do you know what time it is", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "what hour is it", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "can you tell me the time", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "what is the current time", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "is it morning or afternoon", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "how late is it", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what time is it right now", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "got the time", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "what is the time now", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "could you tell me the time please", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "i need to know what time it is", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "time please", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "what time have you got", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "is it late", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "am or pm right now", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "check the time for me", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "i wonder what time it is", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "any idea what time it is", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "do you have the time", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "quick what time is it", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "is it still early", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "how early is it", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "tell me the current time", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "what is the clock say", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "current time please", args: "{}", weight: 0.75 },
];

const CURRENT_TIME_EXAMPLES_IT: &[ExampleUtterance] = &[
    ExampleUtterance { text: "che ora fa", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "che orario è", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che orario abbiamo", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che ore abbiamo", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che ora del giorno è", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "mi sai dire l ora", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "me lo dici che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "dimmi un po che ore sono", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "senti che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "allora che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ma che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "quindi che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "oh che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ehi che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "scusami che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "perdonami che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "di grazia che ora è", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "per cortesia l ora", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "gentilmente mi dici l ora", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che ore saranno", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "chissà che ore saranno", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "mi domando che ore siano", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "vorrei sapere che ore sono", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "avrei bisogno di sapere l ora esatta", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "ho bisogno dell ora esatta", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "mi indichi l ora", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "indicami l ora", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "controlla che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "guarda un po che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "dai dimmi l ora", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "forza dimmi l ora", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "su dimmi che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "l ora adesso per favore", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "l ora attuale grazie", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che ore sono di preciso", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che ore sono esattamente", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "l ora precisa per favore", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "mi diresti l ora per cortesia", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "mi diresti che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "sai per caso l ora", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "sai mica che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "sapresti per caso l ora", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che ora è di preciso", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ma tu ce l hai l ora", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "riesci a dirmi che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "puoi guardare che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "puoi controllare l ora", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "mi controlli l ora", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "mi guardi che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "mi faresti sapere l ora", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "mi fai sapere che ore sono", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "che ore sono per favore", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ti spiace dirmi l ora", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ti dispiacerebbe dirmi che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "avresti l ora da darmi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "spara l ora", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "vai con l ora", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "dimmi l orario", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che orario è adesso", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che orario segna l orologio", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "l orario esatto per favore", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "quanto è l ora", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "a che ora siamo", args: "{}", weight: 0.55 },
    ExampleUtterance { text: "in che ora siamo", args: "{}", weight: 0.55 },
    ExampleUtterance { text: "che ora ci troviamo", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "dimmi l ora esatta", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "mi serve sapere l ora", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "mi servirebbe sapere che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che ora è ora", args: "{}", weight: 0.55 },
    ExampleUtterance { text: "sai dirmi l ora attuale", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "l ora corrente per favore", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "qual è l ora esatta", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "qual è l ora corrente", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "qual è l ora precisa", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "mi dai l ora", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "dammi l ora", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "passami l ora", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "dimmi che ore sono adesso", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è già mezzogiorno", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "siamo già a mezzogiorno", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è già mezzanotte", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è ora di pranzo", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è già ora di pranzo", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è ora di cena", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è già ora di cena", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è ancora presto per pranzo", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è già pomeriggio", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "siamo già di pomeriggio", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "è sera ormai", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "si è fatta sera", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è già sera", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è già notte fonda", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è già ora di andare", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "è già ora di uscire", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è già così tardi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "si è fatto tardi davvero", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "quanto è tardi ormai", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è molto tardi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è tardissimo vero", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è ancora presto vero", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "è troppo presto per uscire", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è ancora mattina", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "è già mattina inoltrata", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "siamo già a fine giornata", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "è già ora di dormire", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "è ora di andare a letto", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è già ora di svegliarsi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è ora di alzarsi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è già l una passata", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è già passata l ora di pranzo", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è mattina o già pomeriggio", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "è pomeriggio o già sera", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "siamo di mattina o di pomeriggio", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è ancora giorno o è già sera", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "è passata mezzanotte", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è già ora di merenda", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è ora del caffè", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è già l ora di chiusura", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è già ora di rientrare", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è già ora di cena o è presto", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "si è fermato l orologio che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "l orologio è fermo mi dici l ora", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "non funziona l orologio che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ho l orologio scarico che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "si è scaricato l orologio dimmi l ora", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "non mi fido dell orologio che ore sono davvero", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "il mio orologio è indietro che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "l orologio del muro è fermo che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ho scordato l orologio a casa che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "senza orologio non so l ora me la dici", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "l orologio si è bloccato che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "credo che l orologio sia fermo che ore sono davvero", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "l orologio segna l ora sbagliata che ore sono giuste", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ho il telefono scarico e non so l ora che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "non ho l orologio addosso che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "mi si è fermato l orologio dimmi tu l ora", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "non ho idea di che ore siano", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "non so più che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "boh non so che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ho perso di vista l ora che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "non riesco a capire che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "sono qui da ore che ora è ormai", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "non mi sono accorto dell ora che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ho lavorato senza guardare l ora che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "non capisco più che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "sono distratto da ore che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "devo uscire che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ho un appuntamento che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "sto per fare tardi che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "manca molto a mezzogiorno", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "arrivo tardi che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "il treno è tra poco che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "quanto ci vuole a sera che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "controllami l ora che devo andare", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "dimmi l ora che ho un impegno", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "guarda l ora per me", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "dimmi tu che ore sono", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "verifica l ora per favore", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che ore sono che sto perdendo la nozione", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "mi ricordi che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che ora abbiamo adesso", args: "{}", weight: 0.55 },
    ExampleUtterance { text: "quant è l ora adesso", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "orario per cortesia", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "l ora grazie", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "solo l ora grazie", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "mi basta sapere l ora", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "voglio solo sapere l ora", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "dimmi solo che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che ora è arrivata", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "a che ora siamo arrivati", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "sai che ore si sono fatte", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che ore si sono fatte", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "guarda che ore si sono fatte", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che ore fanno adesso", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che ore fa adesso", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "in questo preciso momento che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "proprio adesso che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "adesso di preciso che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ora come ora che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "in questo istante che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "puoi dirmi giusto l ora", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "mi sai giusto dire l ora", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "l ora del momento per favore", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che ora fa il tuo orologio", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che ora tieni tu", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che ora hai tu", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "sai tu che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "tu che ora hai", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "controlla tu l ora", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "che ore sono su di te", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che ora batte adesso", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è già l ora", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è arrivata l ora", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è quasi ora", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è già ora vero", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "è tardi o presto adesso", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "fammi sapere l ora precisa", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "dimmi con precisione l ora", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "avrei giusto bisogno dell ora", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "mi occorre sapere l ora", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "mi occorre l ora esatta", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è già buio", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ormai è buio", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "s è fatto buio", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "si è fatto buio presto", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "cala la sera", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "sta calando la sera", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è calata la sera", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "il sole è già tramontato", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "il sole sta tramontando", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ormai è quasi sera", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "sarà già sera", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è già scesa la sera", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "si sta facendo sera", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "fa già scuro", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "è già l ora di cena vero", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è già mattino", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è già mattinata inoltrata", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è ancora presto per la cena", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è ancora presto o si è fatto tardi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ormai è quasi mezzogiorno", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "siamo già a sera vero", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è già l ora della merenda", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è già l imbrunire", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è già l ora del tè", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è ancora giorno oppure è già sera", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "è già calato il sole vero", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "sai che orario è", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "mi dici che orario è", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "che orario abbiamo ora", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che orario segna il tuo orologio", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "dimmi tu che orario è", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "che orario facciamo adesso", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che ora è", args: "{}", weight: 0.55 },
    ExampleUtterance { text: "che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "dimmi l ora", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "mi dici che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "sai che ora è", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che ora è adesso", args: "{}", weight: 0.55 },
    ExampleUtterance { text: "mi puoi dire l ora", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "qual è l ora attuale", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è mattina o pomeriggio", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "si è fatto tardi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che ore sono adesso", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ce l hai l ora", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che ore sono in questo momento", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "mi potrebbe dire l ora per favore", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "ho bisogno di sapere che ore sono", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "l ora per favore", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hai l ora", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è tardi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "scusa sai l ora", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "guarda che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "chissà che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "per caso sai che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hai idea di che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "presto che ore sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è ancora presto", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "è presto o tardi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "dimmi l ora attuale", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che ore segna l orologio", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "l ora esatta per favore", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "sapresti dirmi l ora", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "mi sapresti dire l ora", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "potresti dirmi l ora", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "avrei bisogno di sapere l ora", args: "{}", weight: 0.85 },
];

pub struct CurrentTimeSkill;

impl CurrentTimeSkill {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CurrentTimeSkill {
    fn default() -> Self {
        Self::new()
    }
}

impl Skill for CurrentTimeSkill {
    fn id(&self) -> &str {
        "current_time"
    }

    fn description(&self) -> &str {
        "Tells the current time. Use when the user asks what time it is, what hour it is, whether it is morning or afternoon, or anything about the current time of day."
    }

    fn specificity(&self) -> Specificity {
        Specificity::High
    }

    fn example_utterances(&self) -> &[ExampleUtterance] {
        CURRENT_TIME_EXAMPLES_EN
    }

    fn example_utterances_for(&self, locale: &str) -> &[ExampleUtterance] {
        match locale {
            "it" => CURRENT_TIME_EXAMPLES_IT,
            _ => CURRENT_TIME_EXAMPLES_EN,
        }
    }

    fn score(&self, input: &str, _ctx: &SkillContext) -> f32 {
        let words: Vec<&str> = input.split_whitespace().collect();

        let mut best_score: f32 = 0.0;

        for phrase in TRIGGER_PHRASES {
            let matched = phrase
                .iter()
                .filter(|keyword| words.iter().any(|w| w == *keyword))
                .count();

            if matched == phrase.len() {
                let coverage = matched as f32 / words.len().max(1) as f32;
                let phrase_score = 0.5 + (coverage * 0.5);
                best_score = best_score.max(phrase_score);
            }
        }

        best_score
    }

    fn execute(&self, _input: &str, ctx: &SkillContext) -> Response {
        let now = Local::now();
        // Locale-aware time format. English keeps 12-hour with AM/PM
        // ("It's 3:25 PM."). Other shipped locales use 24-hour ("alle
        // 15:25") — that's the conventional written form in IT/ES/FR/DE
        // and avoids translating the AM/PM tokens.
        let response = match ctx.locale.as_str() {
            "it" => format!("Sono le {}.", now.format("%H:%M")),
            _ => format!("It's {}.", now.format("%-I:%M %p")),
        };
        Response::Text(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> SkillContext {
        SkillContext::default()
    }

    // Score formula: 0.5 + (matched_keywords / total_words * 0.5)
    // Triggers: ["what","time"], ["current","time"], ["tell","time"], ["what is","time"]

    #[test]
    fn score_what_time_is_it() {
        let skill = CurrentTimeSkill::new();
        // "what time is it" = 4 words, ["what","time"] matches 2 keywords
        // coverage = 2/4 = 0.5, score = 0.5 + 0.5*0.5 = 0.75
        assert_eq!(skill.score("what time is it", &ctx()), 0.75);
    }

    #[test]
    fn score_current_time() {
        let skill = CurrentTimeSkill::new();
        // "current time" = 2 words, 2 keywords match, coverage = 1.0
        // score = 0.5 + 1.0*0.5 = 1.0
        assert_eq!(skill.score("current time", &ctx()), 1.0);
    }

    #[test]
    fn score_tell_me_the_time() {
        let skill = CurrentTimeSkill::new();
        // "tell me the time" = 4 words, ["tell","time"] = 2 match
        // coverage = 2/4 = 0.5, score = 0.75
        assert_eq!(skill.score("tell me the time", &ctx()), 0.75);
    }

    #[test]
    fn score_diluted_by_extra_words() {
        let skill = CurrentTimeSkill::new();
        // "can you please tell me the time right now" = 9 words, 2 match
        // coverage = 2/9 ≈ 0.222, score = 0.5 + 0.222*0.5 ≈ 0.611
        let score = skill.score("can you please tell me the time right now", &ctx());
        assert!((score - 0.611).abs() < 0.01, "score was {score}");
    }

    #[test]
    fn score_zero_on_no_keyword_match() {
        let skill = CurrentTimeSkill::new();
        assert_eq!(skill.score("hello there", &ctx()), 0.0);
        assert_eq!(skill.score("what is the weather", &ctx()), 0.0);
    }

    #[test]
    fn score_zero_on_partial_keyword() {
        let skill = CurrentTimeSkill::new();
        // "what" alone doesn't trigger — needs "what" AND "time"
        assert_eq!(skill.score("what is up", &ctx()), 0.0);
    }

    #[test]
    fn score_zero_when_keyword_is_substring_of_other_word() {
        // Regression: scorer used `w.contains(**keyword)` which
        // false-positived on words containing the keyword as a
        // substring — "runtimes" tripped "time", "lifetime" likewise.
        // Word-equality is the right test.
        let skill = CurrentTimeSkill::new();
        assert_eq!(
            skill.score("what does the internet say about async runtimes in rust", &ctx()),
            0.0,
        );
        assert_eq!(skill.score("what is my lifetime achievement", &ctx()), 0.0);
        assert_eq!(skill.score("what about overtime pay", &ctx()), 0.0);
    }

    #[test]
    fn execute_format_matches_12hr_with_am_pm() {
        let skill = CurrentTimeSkill::new();
        let resp = skill.execute("what time is it", &ctx());
        match resp {
            Response::Text(s) => {
                // Format: "It's H:MM AM." or "It's HH:MM PM."
                let inner = s.strip_prefix("It's ").expect("should start with 'It's '");
                let inner = inner.strip_suffix('.').expect("should end with '.'");
                // Must contain a colon and end with AM or PM
                assert!(inner.contains(':'), "no colon in time: {inner}");
                assert!(
                    inner.ends_with("AM") || inner.ends_with("PM"),
                    "no AM/PM in time: {inner}"
                );
                // Hour part should be 1-12
                let hour: u32 = inner.split(':').next().unwrap().parse().unwrap();
                assert!((1..=12).contains(&hour), "hour out of range: {hour}");
                // Minute part should be 00-59
                let min_str = &inner.split(':').nth(1).unwrap()[..2];
                let min: u32 = min_str.parse().unwrap();
                assert!(min <= 59, "minute out of range: {min}");
            }
            _ => panic!("expected Text response"),
        }
    }

    #[test]
    fn specificity_is_high() {
        assert_eq!(CurrentTimeSkill::new().specificity(), Specificity::High);
    }

    #[test]
    fn execute_italian_uses_24h_and_italian_text() {
        let skill = CurrentTimeSkill::new();
        let mut italian = SkillContext::default();
        italian.locale = "it".to_string();
        let resp = skill.execute("che ora e", &italian);
        match resp {
            Response::Text(s) => {
                // Italian: "Sono le HH:MM." — 24-hour, no AM/PM, leading
                // "Sono le". Don't pin the exact time; just shape.
                assert!(
                    s.starts_with("Sono le "),
                    "Italian response should start with 'Sono le ': {s}"
                );
                assert!(s.ends_with('.'));
                assert!(!s.contains("AM"));
                assert!(!s.contains("PM"));
                // Pull the HH:MM out of "Sono le HH:MM."
                let inner = s
                    .strip_prefix("Sono le ")
                    .and_then(|s| s.strip_suffix('.'))
                    .expect("expected 'Sono le HH:MM.' shape");
                assert!(inner.contains(':'), "expected HH:MM, got {inner}");
            }
            _ => panic!("expected Text response"),
        }
    }

    #[test]
    fn score_italian_che_ora() {
        let skill = CurrentTimeSkill::new();
        // "che ora è" — the canonical Italian "what time is it"
        // After normalize_input("che ora è", "it") the input stays as
        // "che ora è" (lowercase, è preserved as alphanumeric). Words
        // = ["che", "ora", "è"]. Phrase ["che", "ora"] matches:
        // coverage = 2/3, score = 0.5 + 0.667*0.5 ≈ 0.833.
        let score = skill.score("che ora è", &ctx());
        assert!(score > 0.5, "expected score > 0.5, got {score}");
    }

    #[test]
    fn score_italian_che_ore_sono() {
        let skill = CurrentTimeSkill::new();
        // "che ore sono" — the other common Italian time query
        let score = skill.score("che ore sono", &ctx());
        assert!(score > 0.5, "expected score > 0.5, got {score}");
    }

    #[test]
    fn score_italian_dimmi_lora_after_normalisation() {
        let skill = CurrentTimeSkill::new();
        // "dimmi l'ora" → after `strip_italian_elisions` becomes
        // "dimmi l ora" (3 tokens). Phrase ["dimmi", "ora"] matches.
        let score = skill.score("dimmi l ora", &ctx());
        assert!(score > 0.5, "expected score > 0.5, got {score}");
    }

    #[test]
    fn execute_unknown_locale_falls_back_to_english() {
        let skill = CurrentTimeSkill::new();
        let mut other = SkillContext::default();
        other.locale = "ja".to_string();
        let resp = skill.execute("what time is it", &other);
        match resp {
            Response::Text(s) => assert!(s.starts_with("It's "), "fallback to English: {s}"),
            _ => panic!("expected Text response"),
        }
    }

    #[test]
    fn execute_spanish_falls_back_to_english_format() {
        let skill = CurrentTimeSkill::new();
        let mut es = SkillContext::default();
        es.locale = "es".to_string();
        let resp = skill.execute("what time is it", &es);
        match resp {
            Response::Text(s) => assert!(s.starts_with("It's "), "expected English fallback: {s}"),
            _ => panic!("expected Text response"),
        }
    }

    #[test]
    fn example_utterances_for_defaults_to_base_examples() {
        let skill = CurrentTimeSkill::new();
        // Unlocalised locales return the base (English) set. "it" now has
        // its own arm — see `italian_router_examples`.
        assert_eq!(skill.example_utterances_for("en"), skill.example_utterances());
        assert_eq!(skill.example_utterances_for("fr"), skill.example_utterances());
        assert!(!skill.example_utterances_for("en").is_empty());
    }

    #[test]
    fn italian_router_examples() {
        let skill = CurrentTimeSkill::new();
        let it = skill.example_utterances_for("it");
        let en = skill.example_utterances_for("en");
        assert!(
            !it.iter().any(|e| en.iter().any(|x| x.text == e.text)),
            "an English phrase leaked into the Italian arm"
        );
        assert_ne!(it, en, "Italian examples are distinct from English");
        assert!(it.iter().any(|e| e.text == "che ora è"), "canonical Italian phrasing present");
        assert!(it.iter().all(|e| e.args == "{}"), "current_time is parameterless");
        assert!(en.iter().any(|e| e.text == "what time is it"), "English arm unchanged");
        assert_eq!(skill.example_utterances_for("fr"), en, "unknown locale falls back to English");
    }

    #[test]
    fn italian_polite_conditional_paraphrases_are_router_examples() {
        let skill = CurrentTimeSkill::new();
        let it = skill.example_utterances_for("it");
        // These are the polite forms the keyword scorer misses — exactly what
        // example phrases are for. Stored normalised, so the elided "l'ora"
        // reads as "l ora" here, matching what `normalize_input` produces.
        assert!(it.iter().any(|e| e.text == "sapresti dirmi l ora"),
            "polite conditional paraphrase must be an example phrase");
        assert!(it.iter().any(|e| e.text.contains("sapresti") || e.text.contains("potresti")),
            "at least one conditional-politeness form present");
    }
}
