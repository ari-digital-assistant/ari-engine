use ari_core::{ExampleUtterance, Response, Skill, SkillContext, Specificity};
use chrono::{Datelike, Local};

// English + Italian. Same union-dictionary pattern as `current_time` —
// keeps Stage 1 keyword routing fast for both languages, no cloud
// round-trip needed. Italian shapes after normalize_input:
//   - "che giorno è" / "che giorno è oggi" → ["che", "giorno"]
//   - "che data è" / "data di oggi" → ["che", "data"] / ["oggi", "data"]
//   - "che data abbiamo" → ["che", "data"]
//   - "in che giorno siamo" → ["che", "giorno"]
const TRIGGER_PHRASES: &[&[&str]] = &[
    // English
    &["what", "date"],
    &["today", "date"],
    &["current", "date"],
    &["what", "day"],
    &["which", "day"],
    // Italian
    &["che", "giorno"],
    &["che", "data"],
    &["data", "oggi"],
    &["data", "attuale"],
];

// Router training examples. Natural raw text as a user would actually say it.
// (Whether the generator should normalise these to match inference is the
// parity spike's question — not this file's.)
const DATE_EXAMPLES_EN: &[ExampleUtterance] = &[
    ExampleUtterance { text: "what is the date for today", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what is todays date then", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what day is it today", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "what date is it today", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "tell me todays date", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "tell me the date", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what day of the week is it", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "which day of the week are we on", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "what day are we", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "what is the date", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "give me todays date", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "i need todays date", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "do you know todays date", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "can you give me the date", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what is the date right now", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what is the date this morning", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "show me the date", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what does the calendar say", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what is the date on the calendar", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what day does the calendar say", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "what day is today again", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "what is the day were on", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "tell me what the date is", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what is the full date today", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what is the date exactly", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what is todays exact date", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "could you tell me todays date", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what is the day of the week today", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "does today fall on a monday", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is today tuesday", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it friday yet", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it saturday already", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is today the weekend", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it the weekend yet", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it still a weekday", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what weekday is it", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "which weekday are we on", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it sunday", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it midweek", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "are we at the weekend already", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it nearly the weekend", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "how many days till the weekend", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "i forget what day it is", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "i cant remember what day it is", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "i lost track of what day it is", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "what day of the year is it", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "how many days into the year are we", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "i have no idea what day it is", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "no clue what the date is", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what day even is it", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "honestly what day is it", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it really wednesday already", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "wait what day is it", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "hang on what is the date", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what day did we land on", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "the days are blurring what date is it", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "every day feels the same what is the date", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ive lost all track of the days what is today", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "i genuinely dont know the date", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "cant keep track of dates what is today", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "what is the date im so lost", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what day of the month is it", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what date of the month are we on", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what is the number today", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what number day is it today", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what month are we in", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "which month is it now", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it still june", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "are we in july yet", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "has the month changed", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "how far through the month are we", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "whereabouts in the month are we", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "are we near the end of the month", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it the 1 of the month", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it month end yet", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "how many days left in the month", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "how much of the month is left", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what week of the month is it", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "are we halfway through the month", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is the month almost done", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it the middle of the month", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what part of the month are we in", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it late in the month", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "are we past the middle of the month", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what should i put as the date", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what date do i write on this", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what is the date for this form", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what do i date this letter", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what is todays date for the record", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "put a date on this what is today", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "i need to date a cheque what is today", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what is the date so i can sign this", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what date shall i note down", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what is the date im filling something in", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what day should i write down", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what is the current date", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "current date please", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "just the date today please", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "the date thanks", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "the date if you would", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "gimme the date", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "quick what is the date", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "just tell me the date", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it the 1 yet", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it the 15 today", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it payday yet", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it the end of the month yet", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it new years yet", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "has the 1 of the month come", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it christmas eve yet", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is today the last day of the month", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it a bank holiday today", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is today a holiday", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "so what day is it", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "and what day is today", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "sorry what is the date", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "excuse me what day is it", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "any idea what the date is", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "you know what day it is", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "do you happen to know the date", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what is the date mate", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what is the date love", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "tell us the date", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what day you got", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "what date is it then", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what is the day and date", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "give me the day and date", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "what is the date and day please", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what calendar day is it", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what day falls today", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what weekday are we", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what day of the week does today land on", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what day of the week are we having", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "what date does today fall on", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what is the numerical date today", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what is the date in numbers", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what is todays date in full", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "spell out todays date", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "read me the date", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "announce the date", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "say todays date", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what is the date reading", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what is showing on the calendar today", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it still the same day", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what day are we into now", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "what day has it become", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what day is it now then", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "what is the date at this point", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "where are we date wise today", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what is the date this fine day", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what is the date good sir", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ari what is the date", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ari what day is it", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "ari tell me todays date", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hey ari what is the date today", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what is todays date please ari", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what day is it supposed to be", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what day should it be today", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "what is the date meant to be today", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "give me the exact date", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what date are we sitting on today", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "which date is it today", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what day of the week does it fall on", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it the last friday of the month", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "have we hit the new month yet", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what day is it turning into", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "what is the date give or take", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "roughly what date is it", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what day were we on again", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "is it still the 1 week of the month", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it monday already", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it thursday already", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it the weekend already", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "cant believe its wednesday already", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it really friday now", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "surely its not friday already", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "has friday come around already", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it tuesday already", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "give me the date off the calendar", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "read me the calendar date", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what is the calendar date today", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "tell me the calendar date", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what date does the calendar show", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what is the date showing on the calendar", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "give me todays calendar date", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what day is the calendar on", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is the month almost over", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "are we near month end now", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "how far through the year are we", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "are we in the back half of the month", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is the month coming to an end", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "how deep into the month are we now", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "are we over halfway through the year", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is the year nearly done", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what is the date today", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what day is it", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "what is today s date", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "which day of the week is it", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "what date is it", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "tell me today s date", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what day are we on", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "is it monday today", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what is the date", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "do you know today s date", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "can you tell me the date", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what day of the week is it today", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "i need to know the date", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "the date please", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is today a weekday", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what is today", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "which day is today", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "tell me what day it is", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "date please", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "current date", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what is today s date", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "is it the weekend", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what day is today", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "do you know what day it is", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "i forgot what day it is", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "is it still tuesday", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what is the day today", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "today s date please", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "check the date for me", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "could you tell me the date", args: "{}", weight: 0.95 },
];

const DATE_EXAMPLES_IT: &[ExampleUtterance] = &[
    ExampleUtterance { text: "oggi che giorno della settimana è", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "in che data siamo", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "in che giorno siamo oggi", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "a che giorno siamo", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "a quanti siamo oggi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "a quanti ne abbiamo oggi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "a quanti ne abbiamo", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "quanti ne abbiamo oggi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "quanti del mese siamo", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che numero è oggi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che data abbiamo oggi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che data abbiamo", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "mi ricordi che giorno è", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "sai dirmi la data di oggi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "mi sai dire che giorno è", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "dimmi per favore la data", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "oggi è che giorno", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "fammi sapere la data", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "dammi la data di oggi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "dammi la data", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "passami la data di oggi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "qual è il giorno di oggi", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "che giorno abbiamo", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "che giorno abbiamo oggi", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "che giorno siamo", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "oggi a quanti siamo", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che dì è oggi", args: "{}", weight: 0.55 },
    ExampleUtterance { text: "mi dici tu che giorno è", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "dimmi tu la data", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "senti che giorno è oggi", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "allora che giorno è oggi", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "ma che giorno è oggi", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "scusa che giorno è oggi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "di preciso che data è", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che data è di preciso", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "qual è la data esatta di oggi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "qual è la data precisa", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "mi indichi la data", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "indicami il giorno di oggi", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "controlla che giorno è oggi", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "verifica la data per favore", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "mi verifichi la data di oggi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "puoi guardare che giorno è", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "puoi controllare la data", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "mi controlli la data", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "sapresti dirmi la data", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "sapresti dirmi che giorno è oggi", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "riesci a dirmi la data", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "mi faresti sapere che giorno è", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "ti spiace dirmi la data", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ti dispiacerebbe dirmi che giorno è", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "vorrei sapere che giorno è oggi", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "avrei bisogno di sapere la data", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "ho bisogno di sapere che giorno è", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "mi serve sapere la data", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "mi servirebbe sapere che giorno è oggi", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "sai per caso che giorno è", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "sai mica che data è oggi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "per caso sai la data di oggi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hai idea di che giorno sia oggi", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "chissà che giorno è oggi", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "che data corre oggi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "in che data ci troviamo", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che giorno corre", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che giorno è oggi di grazia", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "la data odierna per favore", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "la data di oggi grazie", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "solo la data grazie", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "mi basta la data di oggi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "voglio solo sapere la data", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "dimmi solo che giorno è", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "qual è la giornata di oggi", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "che giorno viene oggi", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "oggi corrisponde a che data", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "a quale giorno siamo arrivati", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "sai che giorno si è fatto", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "che giorno ci troviamo oggi", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "oggi è già venerdì", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "ma è già venerdì", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "siamo già a venerdì", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "è ancora mercoledì", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "siamo ancora a mercoledì", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è già il weekend", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è già fine settimana", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "manca tanto al weekend che giorno è", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "oggi è lunedì vero", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "oggi è lunedì o martedì", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "siamo già a giovedì", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è già giovedì", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "oggi è sabato o domenica", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è già sabato", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "è ancora giovedì o è venerdì", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che giorno della settimana abbiamo", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "in che giorno della settimana siamo", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "oggi si lavora vero", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "oggi è un giorno lavorativo", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "oggi è festivo", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è un giorno festivo oggi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "oggi è feriale o festivo", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "oggi si lavora o è festa", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "domani è già sabato che giorno è oggi", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "quanti giorni mancano al weekend oggi che giorno è", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è già lunedì di nuovo", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "è di nuovo lunedì vero", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "ancora lunedì oggi", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "che giorno lavorativo è oggi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "oggi è mercoledì o giovedì", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "siamo a inizio o fine settimana", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che giorno della settimana è oggi di preciso", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "il mese sta volando a che giorno siamo", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "come vola il mese che data è", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "il mese è già a metà vero", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "siamo già a metà mese", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "siamo già a fine mese", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "quanti giorni sono passati questo mese", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "a che giorno del mese siamo", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che data del mese è oggi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "il mese è quasi finito che data è", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "non mi accorgo di come passa il mese che giorno è", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "in che mese siamo", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che mese è adesso", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "siamo già a luglio", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ma siamo già in luglio", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che mese stiamo vivendo", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "a che punto siamo del mese", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che settimana del mese è", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "in che settimana del mese siamo", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "il mese scorre veloce che data è oggi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "siamo a inizio mese vero", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è già passata metà del mese che data è", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "quanto è avanzato il mese che giorno è", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "il mese è quasi agli sgoccioli che data è", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "siamo agli inizi del mese vero", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ho perso completamente il conto dei giorni che giorno è", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "ho perso il filo dei giorni oggi che giorno è", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "non tengo più il conto dei giorni che giorno è", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "con questi giorni tutti uguali non so più la data", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "questi giorni si assomigliano tutti che giorno è", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "tra un giorno e l altro ho perso la data", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ho perso di vista il calendario che giorno è", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "non ho più il senso dei giorni che data è", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "i giorni si confondono che giorno è oggi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "non so nemmeno più che giorno sia", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "boh non so che giorno è oggi", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "non mi ricordo più la data di oggi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "non mi ricordo che giorno sia oggi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "non ho idea del giorno che giorno è", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "mi sono perso i giorni che data è", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "in ferie ho perso il conto dei giorni che giorno è", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "dopo le vacanze non so più che giorno è", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "questi giorni volano che data è", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "non riesco a stare dietro ai giorni che giorno è", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "sono giorni che non guardo il calendario che data è", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ho perso la nozione dei giorni che giorno è", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "devo segnare la data che giorno è oggi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "sto compilando un modulo che data è oggi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "devo mettere la data che giorno è", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "per la firma che data metto oggi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che data va scritta oggi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che data ci mettiamo oggi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "oggi in che data siamo per l assegno", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "mi serve la data per il documento che giorno è", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "quale data segno oggi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che giorno è che devo prendere nota", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "dimmi la data che devo appuntarla", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "controllami la data di oggi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "guardami che giorno è", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "dimmi tu che giorno è oggi", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "verificami il giorno di oggi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che data è che ho un promemoria", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che giorno è che ho una scadenza", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "oggi scade qualcosa che giorno è", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "che giorno è oggi che ho perso il conto", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "aiutami con la data di oggi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "solo per curiosità che giorno è", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "così per sapere che data è oggi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "tanto per sapere che giorno siamo", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "giusto per curiosità in che data siamo", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "oggi quanti ne abbiamo", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ma quanti ne abbiamo", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "senti quanti ne abbiamo", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "scusa quanti ne abbiamo oggi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "quanti ne abbiamo di preciso", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "quanti ne abbiamo questo mese", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "a quanti ne siamo oggi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "dimmi quanti ne abbiamo", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "a che punto siamo dell anno", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "a che punto dell anno siamo arrivati", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "a che punto dell anno ci troviamo", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "quanto è avanzato l anno", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "l anno è già a metà vero", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "siamo già a metà anno", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "quanti mesi dell anno sono passati", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "a che punto del mese siamo arrivati", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "l anno sta volando a che mese siamo", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "siamo a inizio o fine anno", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "sai quanti ne abbiamo oggi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ma oggi quanti ne abbiamo", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "allora quanti ne abbiamo", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "quanti ne abbiamo adesso", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "quanti giorni ne abbiamo oggi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "chissà quanti ne abbiamo oggi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che giorno è oggi", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "che giorno è", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "che data è oggi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "qual è la data di oggi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che data è", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "dimmi la data di oggi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "in che giorno siamo", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "oggi è lunedì", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "qual è la data", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "sai che data è oggi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "mi puoi dire la data", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che giorno della settimana è oggi", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "ho bisogno di sapere la data", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "la data per favore", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "oggi è un giorno feriale", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "oggi che giorno è", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "che giorno della settimana è", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "dimmi che giorno è", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "data per favore", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "data di oggi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "qual è la data odierna", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è il weekend", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "oggi che data è", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "sai che giorno è oggi", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "non mi ricordo che giorno è", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "siamo ancora a martedì", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "dimmi la data", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "la data di oggi per favore", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "controlla la data", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "mi potrebbe dire la data", args: "{}", weight: 0.95 },
];

pub struct DateSkill;

impl DateSkill {
    pub fn new() -> Self {
        Self
    }
}

impl Default for DateSkill {
    fn default() -> Self {
        Self::new()
    }
}

impl Skill for DateSkill {
    fn id(&self) -> &str {
        "current_date"
    }

    fn description(&self) -> &str {
        "Tells today's date. Use when the user asks what day it is, what date it is, which day of the week it is, or anything about today's date."
    }

    fn specificity(&self) -> Specificity {
        Specificity::High
    }

    fn example_utterances(&self) -> &[ExampleUtterance] {
        DATE_EXAMPLES_EN
    }

    fn example_utterances_for(&self, locale: &str) -> &[ExampleUtterance] {
        match locale {
            "it" => DATE_EXAMPLES_IT,
            _ => DATE_EXAMPLES_EN,
        }
    }

    fn score(&self, input: &str, _ctx: &SkillContext) -> f32 {
        let words: Vec<&str> = input.split_whitespace().collect();

        // English "time" / Italian "ora"/"ore" — any of these in the
        // input means the user wants the time skill, not date. Mirrors
        // the English-only guard that was here before; same intent.
        const TIME_WORDS: &[&str] = &["time", "ora", "ore"];
        if words.iter().any(|w| TIME_WORDS.contains(w)) {
            return 0.0;
        }

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
        // chrono's `%A` (weekday) and `%B` (month) format using the
        // system's C-locale by default — that's English-only on most
        // builds. Hand-roll the locale-specific tables so the response
        // doesn't depend on what locales the host happens to have
        // installed at the OS level.
        let weekday_idx = now.weekday().num_days_from_monday() as usize; // 0..=6
        let month_idx = now.month() as usize; // 1..=12
        let day = now.day();
        let year = now.year();
        let response = match ctx.locale.as_str() {
            "it" => {
                let weekday = ITALIAN_WEEKDAYS[weekday_idx];
                let month = ITALIAN_MONTHS[month_idx];
                format!("Oggi è {} {} {} {}.", weekday, day, month, year)
            }
            _ => {
                let formatted = now.format("%A, %B %-d, %Y").to_string();
                format!("Today is {}.", formatted)
            }
        };
        Response::Text(response)
    }
}

// Index 0 = Monday (chrono's `num_days_from_monday`).
const ITALIAN_WEEKDAYS: [&str; 7] = [
    "lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica",
];

// Index 0 unused — months are 1..=12.
const ITALIAN_MONTHS: [&str; 13] = [
    "", "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio",
    "agosto", "settembre", "ottobre", "novembre", "dicembre",
];

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> SkillContext {
        SkillContext::default()
    }

    // Score formula: same as CurrentTimeSkill — 0.5 + (matched/total * 0.5)
    // But returns 0.0 if "time" is in the input

    #[test]
    fn score_what_date() {
        let skill = DateSkill::new();
        // "what is the date" = 4 words, ["what","date"] match 2
        // coverage = 2/4 = 0.5, score = 0.75
        assert_eq!(skill.score("what is the date", &ctx()), 0.75);
    }

    #[test]
    fn score_what_day() {
        let skill = DateSkill::new();
        // "what day is it" = 4 words, ["what","day"] match 2
        assert_eq!(skill.score("what day is it", &ctx()), 0.75);
    }

    #[test]
    fn score_current_date() {
        let skill = DateSkill::new();
        // "current date" = 2 words, 2 match, coverage = 1.0
        assert_eq!(skill.score("current date", &ctx()), 1.0);
    }

    #[test]
    fn score_zero_when_time_present() {
        let skill = DateSkill::new();
        // Disambiguation: "time" in input → 0.0
        assert_eq!(skill.score("what time is it", &ctx()), 0.0);
        assert_eq!(skill.score("date and time", &ctx()), 0.0);
    }

    #[test]
    fn score_zero_on_unrelated() {
        let skill = DateSkill::new();
        assert_eq!(skill.score("hello there", &ctx()), 0.0);
        assert_eq!(skill.score("open spotify", &ctx()), 0.0);
    }

    #[test]
    fn score_zero_when_keyword_is_substring_of_other_word() {
        // Regression: scorer used `w.contains(**keyword)` which
        // false-positived on words containing the keyword as a
        // substring — "today" tripped "today" inside "todays" etc.
        let skill = DateSkill::new();
        assert_eq!(
            skill.score("what is sundays special at the deli", &ctx()),
            0.0,
        );
        assert_eq!(skill.score("what is the holiday discount", &ctx()), 0.0);
        // "what" and "today" both as standalone words still trigger.
        assert!(skill.score("what is the date today", &ctx()) > 0.0);
    }

    #[test]
    fn score_italian_che_giorno() {
        let skill = DateSkill::new();
        // "che giorno è" — the canonical Italian "what day is it"
        let score = skill.score("che giorno è", &ctx());
        assert!(score > 0.5, "expected score > 0.5, got {score}");
    }

    #[test]
    fn score_italian_che_data() {
        let skill = DateSkill::new();
        // "che data è oggi" — Italian "what date is it today"
        let score = skill.score("che data è oggi", &ctx());
        assert!(score > 0.5, "expected score > 0.5, got {score}");
    }

    #[test]
    fn score_italian_zero_when_ora_present() {
        // "ora" (Italian for "hour") in the input means the user wants
        // the time skill, not date — same logic as the existing
        // English "time" guard. Without this, "che ora è" would match
        // both date (no — actually no Italian date phrase, fine) and
        // current_time. Just a sanity check that date doesn't
        // false-positive on Italian time queries.
        let skill = DateSkill::new();
        assert_eq!(skill.score("che ora è", &ctx()), 0.0);
    }

    #[test]
    fn execute_italian_uses_italian_weekday_and_month() {
        let skill = DateSkill::new();
        let mut italian = SkillContext::default();
        italian.locale = "it".to_string();
        let resp = skill.execute("che giorno è oggi", &italian);
        match resp {
            Response::Text(s) => {
                // Shape: "Oggi è <weekday> <day> <month> <year>."
                assert!(s.starts_with("Oggi è "));
                assert!(s.ends_with('.'));
                // At least one Italian weekday or month must be present —
                // the exact ones depend on test-run date, so we check
                // membership rather than equality.
                let italian_weekday = ITALIAN_WEEKDAYS.iter().any(|w| s.contains(*w));
                let italian_month = ITALIAN_MONTHS.iter().any(|m| !m.is_empty() && s.contains(*m));
                assert!(italian_weekday, "no Italian weekday in: {s}");
                assert!(italian_month, "no Italian month in: {s}");
                // No English months should appear
                assert!(!s.contains("January"));
                assert!(!s.contains("Today is"));
            }
            _ => panic!("expected Text response"),
        }
    }

    #[test]
    fn execute_format_weekday_month_day_year() {
        let skill = DateSkill::new();
        let resp = skill.execute("what date is it", &ctx());
        match resp {
            Response::Text(s) => {
                // Format: "Today is Wednesday, April 6, 2026."
                let inner = s
                    .strip_prefix("Today is ")
                    .expect("should start with 'Today is '");
                let inner = inner.strip_suffix('.').expect("should end with '.'");
                let parts: Vec<&str> = inner.splitn(2, ", ").collect();
                assert_eq!(parts.len(), 2, "expected 'Weekday, Month Day, Year' got: {inner}");
                let weekdays = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"];
                assert!(weekdays.contains(&parts[0]), "bad weekday: {}", parts[0]);
                // Rest should be "Month Day, Year"
                assert!(parts[1].contains(", "), "missing year separator in: {}", parts[1]);
            }
            _ => panic!("expected Text"),
        }
    }

    #[test]
    fn specificity_is_high() {
        assert_eq!(DateSkill::new().specificity(), Specificity::High);
    }

    #[test]
    fn italian_router_examples() {
        let skill = DateSkill::new();
        let it = skill.example_utterances_for("it");
        let en = skill.example_utterances_for("en");
        assert!(
            !it.iter().any(|e| en.iter().any(|x| x.text == e.text)),
            "an English phrase leaked into the Italian arm"
        );
        assert_ne!(it, en, "Italian examples are distinct from English");
        assert!(it.iter().any(|e| e.text == "che giorno è oggi"), "canonical Italian phrasing present");
        assert!(it.iter().all(|e| e.args == "{}"), "date is parameterless");
        assert!(en.iter().any(|e| e.text == "what day is it"), "English arm unchanged");
        assert_eq!(skill.example_utterances_for("fr"), en, "unknown locale falls back to English");
    }

    #[test]
    fn execute_spanish_falls_back_to_english() {
        let skill = DateSkill::new();
        let mut es = SkillContext::default();
        es.locale = "es".to_string();
        let resp = skill.execute("what date is it", &es);
        match resp {
            Response::Text(s) => assert!(s.starts_with("Today is "), "expected English fallback: {s}"),
            _ => panic!("expected Text"),
        }
    }
}
