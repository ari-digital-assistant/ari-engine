use ari_core::{ExampleUtterance, Response, Skill, SkillContext, Specificity};

// English + Italian trigger words. Same union-dictionary approach as
// the reminder skill's parser — words don't collide across these
// languages, so a single contains-check disambiguates.
const GREETINGS: &[&str] = &[
    // English
    "hello", "hi", "hey", "heya", "howdy", "greetings", "good morning",
    "good afternoon", "good evening", "yo", "sup", "hiya", "ello",
    "hey ari", "hi ari", "hello ari",
    // Italian. No "buonanotte" — it's a farewell, and English doesn't
    // list "good night" either. Answering it with "Ciao! Cosa posso fare
    // per te?" is the wrong end of the conversation; let it fall through
    // to the assistant, which is what the Italian router examples assume.
    "ciao", "salve", "buongiorno", "buonasera",
    "ciao ari", "salve ari",
];

const HOW_ARE_YOU: &[&[&str]] = &[
    // English
    &["how", "are", "you"],
    &["how", "you", "doing"],
    &["how", "is", "it", "going"],
    &["what", "is", "up"],
    &["what", "up"],
    // Italian
    &["come", "stai"],
    &["come", "va"],
];

const RESPONSES_EN: &[&str] = &[
    "Hey there! What can I do for you?",
    "Hello! How can I help?",
    "Hi! What's on your mind?",
    "Hey! Ready when you are.",
];

const RESPONSES_IT: &[&str] = &[
    "Ciao! Cosa posso fare per te?",
    "Ciao! Come posso aiutarti?",
    "Ciao! A cosa stai pensando?",
    "Ciao! Sono qui quando vuoi.",
];

fn responses_for_locale(locale: &str) -> &'static [&'static str] {
    match locale {
        "it" => RESPONSES_IT,
        _ => RESPONSES_EN,
    }
}

fn how_are_you_response(locale: &str) -> &'static str {
    match locale {
        "it" => "Sto benissimo, grazie! Come posso aiutarti?",
        _ => "I'm doing great, thanks for asking! How can I help you?",
    }
}

// Router training examples. Natural raw text as a user would actually say it.
// (Whether the generator should normalise these to match inference is the
// parity spike's question — not this file's.)
const GREETING_EXAMPLES_EN: &[ExampleUtterance] = &[
    ExampleUtterance { text: "hi there", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hello there", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hiya ari", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "yo ari", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "mornin", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "morning ari", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "good afternoon ari", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "afternoon", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "good evening ari", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "evening ari", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "good day", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "g day", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "morning morning", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "a very good morning", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "good morning to you ari", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "top of the day to you", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hello good sir", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hey lovely", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hi there ari", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hey up ari", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "how goes it", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "how goes it ari", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hows it all going", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "hows your day been so far", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "how are you feeling today", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "how are you getting on", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "you keeping well", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "keeping busy", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hope you are well ari", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hey hows things with you", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what is up", args: "{}", weight: 0.55 },
    ExampleUtterance { text: "wassup", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what is up ari", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what is going on", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "how are you today", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hows it going", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "hows it going ari", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "hows things", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hows everything", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "so how have you been", args: "{}", weight: 0.55 },
    ExampleUtterance { text: "how you been", args: "{}", weight: 0.55 },
    ExampleUtterance { text: "how are you keeping", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "morning hope youre good", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hello lovely to catch you", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "well hello there ari", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hey there hows life", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "good to speak to you again", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "hiya hows it going", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "you alright", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "you alright ari", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "alright", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "alright ari", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "alright there", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "you good", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "you doing ok", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "how are we doing", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "how is it going today", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "hows your day", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "hows life", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "top of the morning to you", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "good morning to you", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "morning ari how are you", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "and a good morning to you", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "evening ari hows things", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "good evening to you", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "afternoon ari", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "lovely morning isnt it", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "beautiful day today", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "bit of a grey morning hey", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "long time no chat", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "its been ages hasnt it", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "its been a while", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "been a while hasnt it", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ages since we spoke", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "we havent talked in forever", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "havent heard from you in a bit", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "its been too long", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "its been such a long time", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hey its been a long time", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "good to see you again", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "great to see you", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "lovely to see you", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "good to have you back", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "there you are", args: "{}", weight: 0.55 },
    ExampleUtterance { text: "so we meet again", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "back again", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "did you miss me", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "long time no chat ari", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "miss me", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "look who it is again", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "well look who it is", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "look whos here", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "look whos back", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "well if it isnt ari", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "if it isnt my old friend", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "theres the legend", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ah there she is", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ah there you are", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hey you", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hey you there", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "you there", args: "{}", weight: 0.55 },
    ExampleUtterance { text: "you around", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "anybody home", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "you awake", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "you listening", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "can you hear me", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hows tricks", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hows it hanging", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what are you up to", args: "{}", weight: 0.55 },
    ExampleUtterance { text: "whatcha doing", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what is new", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "anything new", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what is the story", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hows your morning", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "hows your afternoon", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hows your evening", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "having a good day", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "hope youre well", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ey up", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "now then", args: "{}", weight: 0.55 },
    ExampleUtterance { text: "alright me old mucker", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hows it going pal", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hey buddy", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hey mate", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hi mate", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hello mate", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hiya love", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hello love", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hey pal", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hello friend", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hey friend", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hey champ", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hello you legend", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "wotcha", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "oi oi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ahoy", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ahoy there", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "how goes things", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hi how are you", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hey how are you doing", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hello how are things", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hi there hows it going", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hey ari hows it going", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "morning hows things", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "good morning how are you", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "good evening how are you", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "hello hope youre good", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hi hope youre doing well", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hey good to see you", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "hi again", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hello again", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hey its me", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hi its me again", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hello you", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hey stranger", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hello stranger", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "well hello", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "why hello there", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hope you had a good weekend", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hope your weekend was good", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "how was your weekend", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "how was your day", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "hope youre having a good 1", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hows my favourite assistant", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hows my assistant doing", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hey how have things been", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hi how has your day been", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "good to chat again", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "nice to chat", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "lovely to hear from you", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "always good to see you", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "glad to catch you", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "glad youre here", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "pleased to see you", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what a nice surprise", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "fancy running into you", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "so nice to see you", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "well well well", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hey hey", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hi hi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hello hello", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "heyo", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "helloo", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hihi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "yoo", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "oi ari", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "psst ari", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hey there you", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hello my friend", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hey there friend", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "morning sunshine", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hello sunshine", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hey sunshine", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "rise and shine", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "good to see your face", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hello again ari", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hey there ari how are you", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hey now", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hows my favourite ai", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "nice to see you again", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "so nice to see you again", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "lovely seeing you", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "great to see you again", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "fancy meeting you here", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "fancy bumping into you", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "well look whos here again", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "guess whos here", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "well if it isnt you", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "good to have you back again", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "hello", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hey", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hey there", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "howdy", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "good morning", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "good afternoon", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "good evening", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "yo", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "sup", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "what is up", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "hiya", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "heya", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "hello ari", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hi ari", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hey ari", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "good morning ari", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "greetings", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "how are you", args: "{}", weight: 0.55 },
    ExampleUtterance { text: "how are you doing", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "how s it going", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "what is going on", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "how do you do", args: "{}", weight: 0.55 },
    ExampleUtterance { text: "nice to meet you", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "hey there ari", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "morning", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "evening", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "how are things", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "how you doing", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "what is happening", args: "{}", weight: 0.6 },
];

const GREETING_EXAMPLES_IT: &[ExampleUtterance] = &[
    ExampleUtterance { text: "ciao ari come va", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ehilà ari", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ciao ari ci sei", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ci sei ari", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ehi ci sei", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "sei sveglia ari", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "sei lì ari", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ciao a te", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "salve a te", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "salve ari come va", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "buondì ari", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "buon pomeriggio ari come va", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "buongiorno a te", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "buongiorno bello", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "buongiorno caro", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "buonasera cara", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "buon pomeriggio a te", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "buondì a te", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "buongiorno ari come stai", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "buonasera ari come stai", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "buonasera ari tutto bene", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ciao ari tutto a posto", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ehi ari come butta", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ohi ari", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ohilà ari", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ueh ari", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ueilà ari", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "weilà ari", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ehi tu ari", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ohilà bella", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ohi bella", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "oi bella", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ehi bella", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ciao bello", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ciao caro", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ciao carissima", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ciao amico", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ciao amica", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ciao amico mio", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ehi amico", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ehi socio", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ehi capo", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ciao capo", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ehi ciao", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ciao ciao come stai", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ehilà come va", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ehilà come butta", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ohilà come va", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ciao bella come stai", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ciao caro come stai", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ehi ari eccomi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ari ci sei ancora", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ma ci sei ari", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "sei ancora sveglia ari", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ari sei lì", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "pronto ari", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ari mi senti", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ehi ari mi senti", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ci sentiamo ari come va", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "eccomi ari", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "sono di nuovo qui ari", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "chi si rivede", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "chi si vede", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ma chi si rivede", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "guarda chi si rivede", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "oh chi si rivede", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ma guarda chi si vede", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "chi non muore si rivede", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "eccoti qua", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "eccoti finalmente", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "rieccoti", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "rieccoci", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "eccola qua", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ma ci sei ancora", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "guarda un po chi si rivede", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ma va chi si rivede", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "toh chi si rivede", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "eccoti di nuovo", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "sei tornata eh", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "sei di nuovo qui eh", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "ancora qua tu", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "di nuovo insieme eh", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "da quanto non ci sentiamo", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è un bel po che non ci sentiamo", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "quant è che non ti sento", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "quant è che non ci parliamo", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è da un po che non ci sentiamo", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "quant è che non ci vediamo", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "quanto è che non ti facevi sentire", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "da quanto manchi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "sparita da un po eh", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "non ti facevi sentire da un po", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è una vita che non ci sentiamo", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è secoli che non ci parliamo", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che si dice di bello", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che mi dici di bello", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che mi racconti di bello", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "come va la vita ultimamente", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "allora come butta", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "come ti va", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "come va lì da te", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "come vanno le cose dalle tue parti", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che si fa di bello", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che combini", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che combini di bello", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che fai di bello", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "cosa mi racconti", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "novità in giro", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ci sono novità", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che c è di nuovo", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "come butta lì", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "come vanno le cose lì", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che si racconta", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che si racconta di bello", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "come andiamo", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "tutto bene dalle tue parti", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "e allora come va", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "beh come va", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "dimmi come va", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "raccontami come va", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "su come va", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "allora che si dice", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "e quindi che si dice", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "dai che si dice", args: "{}", weight: 0.75 },
    ExampleUtterance { text: "era ora di risentirci", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "guarda un po che bel momento per risentirci", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ci voleva un saluto ogni tanto", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "un saluto ci sta sempre", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "piacere mio", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "molto piacere", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "è un piacere", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che piacere sentirti", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che piacere", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che bello sentirti", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che bello risentirti", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che bello ritrovarti", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "felice di sentirti", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "contento di sentirti", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "sono felice di risentirti", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che bella sorpresa", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che bella sorpresa sentirti", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che sorpresa sei tu", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ma che bella sorpresa", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "bello averti qui", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "bello riaverti qui", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ehi come butta", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "come butta amico", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "come butta bella", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "tutto bene amico", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "tutto a posto amico", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "come va amico mio", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "come va socio", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "come va capo", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "salve salve", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "buon dì a tutti", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "salutami", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "un saluto", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "un salutone", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "salutoni", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ti saluto ari", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "saluti ari", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ehi salve", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "salve come stai", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "buongiorno come butta", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "buonasera come butta", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "buondì come va", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ciao come butta", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ciao come te la passi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ehi come te la passi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ehi come vanno le cose", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ciao come vanno le cose", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "buondì come stai", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "buon pomeriggio come stai", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "buonasera come va la vita", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ciao bella come va", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ciao bello come va", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "come stai di bello", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "come va di bello", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che si dice in giro", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che novità ci sono", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che mi combini", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "come butta di bello", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "come te la passi in questi giorni", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "come stai in questi giorni", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "come vanno le cose in questi giorni", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "bentrovato ari", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "bentrovata", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ben ritrovato", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ciao ciao ari", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "salve salve ari", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "olà ari", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ciao bellissima", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ehilà bella", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ciao a te ari", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "un salutone ari", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ti saluto", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ma guarda chi c è", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "guarda un po chi c è", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "ma chi si vede mai", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "oh guarda chi c è", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ma tu guarda chi si rivede", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "chi si fa vedere", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "guarda un po chi si fa vedere", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "ti si rivede eh", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ma va chi si vede", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "e guarda chi si rivede", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ma chi si rivede mai", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "chi si rivede finalmente", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "rieccoti qua", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "eccoti di ritorno", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "finalmente ti rivedo", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "guarda un po chi ricompare", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "una vita che non ci si vede", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ce n è voluto per rivederti", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "guarda tu chi si fa rivedere", args: "{}", weight: 0.85 },
    ExampleUtterance { text: "ciao", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "salve", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "buongiorno", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "buonasera", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "buon pomeriggio", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ehi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ehilà", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "buondì", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ciao ari", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "salve ari", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "buongiorno ari", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "buonasera ari", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ehi ari", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "come stai", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "come va", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "come sta", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "tutto bene", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "tutto a posto", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "come butta", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "come te la passi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "come vanno le cose", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "come procede", args: "{}", weight: 0.6 },
    ExampleUtterance { text: "che si dice", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "che mi racconti", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "novità", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "piacere di conoscerti", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "come stai oggi", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "ciao come stai", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "salve come sta", args: "{}", weight: 0.95 },
    ExampleUtterance { text: "come va la vita", args: "{}", weight: 0.95 },
];

pub struct GreetingSkill;

impl GreetingSkill {
    pub fn new() -> Self {
        Self
    }
}

impl Default for GreetingSkill {
    fn default() -> Self {
        Self::new()
    }
}

impl Skill for GreetingSkill {
    fn id(&self) -> &str {
        "greeting"
    }

    fn description(&self) -> &str {
        "Responds to greetings. Use when the user says hello, hi, hey, good morning, good evening, howdy, what's up, or asks how Ari is doing."
    }

    fn specificity(&self) -> Specificity {
        Specificity::Low
    }

    fn example_utterances(&self) -> &[ExampleUtterance] {
        GREETING_EXAMPLES_EN
    }

    fn example_utterances_for(&self, locale: &str) -> &[ExampleUtterance] {
        match locale {
            "it" => GREETING_EXAMPLES_IT,
            _ => GREETING_EXAMPLES_EN,
        }
    }

    fn score(&self, input: &str, _ctx: &SkillContext) -> f32 {
        let words: Vec<&str> = input.split_whitespace().collect();

        for phrase in HOW_ARE_YOU {
            let matched = phrase
                .iter()
                .filter(|kw| words.contains(kw))
                .count();
            if matched == phrase.len() {
                return 0.9;
            }
        }

        for greeting in GREETINGS {
            let greeting_words: Vec<&str> = greeting.split_whitespace().collect();
            let matched = greeting_words
                .iter()
                .filter(|kw| words.contains(kw))
                .count();
            if matched == greeting_words.len() {
                let coverage = matched as f32 / words.len().max(1) as f32;
                return 0.6 + (coverage * 0.4);
            }
        }

        0.0
    }

    fn execute(&self, input: &str, ctx: &SkillContext) -> Response {
        let words: Vec<&str> = input.split_whitespace().collect();
        let is_how_are_you = HOW_ARE_YOU.iter().any(|phrase| {
            phrase.iter().all(|kw| words.contains(kw))
        });

        if is_how_are_you {
            return Response::Text(how_are_you_response(ctx.locale.as_str()).to_string());
        }

        let responses = responses_for_locale(ctx.locale.as_str());
        let idx = input.len() % responses.len();
        Response::Text(responses[idx].to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> SkillContext {
        SkillContext::default()
    }

    // Score for HOW_ARE_YOU phrases: always 0.9
    // Score for GREETINGS: 0.6 + (matched/total_words * 0.4)

    #[test]
    fn score_single_word_greeting() {
        let skill = GreetingSkill::new();
        // "hello" = 1 word, 1 match, coverage = 1.0
        // score = 0.6 + 1.0*0.4 = 1.0
        assert_eq!(skill.score("hello", &ctx()), 1.0);
        assert_eq!(skill.score("hi", &ctx()), 1.0);
        assert_eq!(skill.score("hey", &ctx()), 1.0);
        assert_eq!(skill.score("heya", &ctx()), 1.0);
        assert_eq!(skill.score("yo", &ctx()), 1.0);
    }

    #[test]
    fn score_greeting_diluted_by_extra_words() {
        let skill = GreetingSkill::new();
        // "hello there" = 2 words, "hello" matches, coverage = 1/2
        // score = 0.6 + 0.5*0.4 = 0.8
        assert_eq!(skill.score("hello there", &ctx()), 0.8);
    }

    #[test]
    fn score_multi_word_greeting() {
        let skill = GreetingSkill::new();
        // "good morning" = 2 words, both match the GREETINGS entry, coverage = 2/2 = 1.0
        // score = 0.6 + 1.0*0.4 = 1.0
        assert_eq!(skill.score("good morning", &ctx()), 1.0);
    }

    #[test]
    fn score_how_are_you_always_09() {
        let skill = GreetingSkill::new();
        assert_eq!(skill.score("how are you", &ctx()), 0.9);
        assert_eq!(skill.score("how are you doing today", &ctx()), 0.9);
    }

    #[test]
    fn score_what_is_up() {
        let skill = GreetingSkill::new();
        assert_eq!(skill.score("what is up", &ctx()), 0.9);
    }

    #[test]
    fn score_zero_on_unrelated() {
        let skill = GreetingSkill::new();
        assert_eq!(skill.score("what time is it", &ctx()), 0.0);
        assert_eq!(skill.score("calculate 2 plus 2", &ctx()), 0.0);
    }

    #[test]
    fn execute_how_are_you_returns_specific_response() {
        let skill = GreetingSkill::new();
        let resp = skill.execute("how are you", &ctx());
        assert_eq!(
            matches!(resp, Response::Text(ref s) if s == "I'm doing great, thanks for asking! How can I help you?"),
            true
        );
    }

    #[test]
    fn execute_what_is_up_returns_specific_response() {
        let skill = GreetingSkill::new();
        let resp = skill.execute("what is up", &ctx());
        match resp {
            Response::Text(s) => assert_eq!(s, "I'm doing great, thanks for asking! How can I help you?"),
            _ => panic!("expected Text"),
        }
    }

    #[test]
    fn execute_regular_greeting_picks_from_responses() {
        let skill = GreetingSkill::new();
        // Response selection: input.len() % RESPONSES_EN.len()
        // "hello" = 5 chars, 5 % 4 = 1 → RESPONSES_EN[1]
        let resp = skill.execute("hello", &ctx());
        match resp {
            Response::Text(s) => assert_eq!(s, "Hello! How can I help?"),
            _ => panic!("expected Text"),
        }
    }

    #[test]
    fn execute_italian_how_are_you() {
        let skill = GreetingSkill::new();
        let mut italian = SkillContext::default();
        italian.locale = "it".to_string();
        let resp = skill.execute("come stai", &italian);
        match resp {
            Response::Text(s) => assert_eq!(
                s,
                "Sto benissimo, grazie! Come posso aiutarti?"
            ),
            _ => panic!("expected Text"),
        }
    }

    #[test]
    fn execute_italian_regular_greeting_picks_from_italian_responses() {
        let skill = GreetingSkill::new();
        let mut italian = SkillContext::default();
        italian.locale = "it".to_string();
        // "ciao" = 4 chars, 4 % 4 = 0 → RESPONSES_IT[0]
        let resp = skill.execute("ciao", &italian);
        match resp {
            Response::Text(s) => assert_eq!(s, "Ciao! Cosa posso fare per te?"),
            _ => panic!("expected Text"),
        }
    }

    #[test]
    fn score_italian_greeting_triggers() {
        let skill = GreetingSkill::new();
        // Italian greeting "ciao" must score above 0 — the union
        // dictionary lets the same scorer recognise both languages.
        assert_eq!(skill.score("ciao", &ctx()), 1.0);
        assert_eq!(skill.score("buongiorno", &ctx()), 1.0);
    }

    #[test]
    fn farewells_are_not_greetings() {
        let skill = GreetingSkill::new();
        // Both languages agree: saying good night is leaving, not arriving.
        assert_eq!(skill.score("buonanotte", &ctx()), 0.0);
        assert_eq!(skill.score("good night", &ctx()), 0.0);
        // The evening greeting it rhymes with is still a greeting.
        assert_eq!(skill.score("buonasera", &ctx()), 1.0);
        assert_eq!(skill.score("good evening", &ctx()), 1.0);
    }

    #[test]
    fn execute_different_input_different_response() {
        let skill = GreetingSkill::new();
        // "hi" = 2 chars, 2 % 4 = 2 → RESPONSES[2]
        let resp = skill.execute("hi", &ctx());
        match resp {
            Response::Text(s) => assert_eq!(s, "Hi! What's on your mind?"),
            _ => panic!("expected Text"),
        }
    }

    #[test]
    fn specificity_is_low() {
        assert_eq!(GreetingSkill::new().specificity(), Specificity::Low);
    }

    #[test]
    fn italian_router_examples() {
        let skill = GreetingSkill::new();
        let it = skill.example_utterances_for("it");
        let en = skill.example_utterances_for("en");
        assert!(
            !it.iter().any(|e| en.iter().any(|x| x.text == e.text)),
            "an English phrase leaked into the Italian arm"
        );
        assert_ne!(it, en, "Italian examples are distinct from English");
        assert!(it.iter().any(|e| e.text == "ciao"), "canonical Italian greeting present");
        assert!(it.iter().all(|e| e.args == "{}"), "greeting is parameterless");
        assert!(en.iter().any(|e| e.text == "hello"), "English arm unchanged");
        assert_eq!(skill.example_utterances_for("fr"), en, "unknown locale falls back to English");
    }

    #[test]
    fn execute_spanish_locale_falls_back_to_english() {
        let skill = GreetingSkill::new();
        let mut es = SkillContext::default();
        es.locale = "es".to_string();
        // After the strip, "es" is no longer special-cased -> English responses.
        // "hello" = 5 chars, 5 % 4 = 1 -> RESPONSES_EN[1].
        let resp = skill.execute("hello", &es);
        match resp {
            Response::Text(s) => assert_eq!(s, "Hello! How can I help?"),
            _ => panic!("expected Text"),
        }
    }
}
