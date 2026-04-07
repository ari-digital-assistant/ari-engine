use ari_engine::Engine;
use ari_skills::{CalculatorSkill, CurrentTimeSkill, DateSkill, GreetingSkill, OpenSkill, SearchSkill};
use std::io::{self, BufRead};

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();

    let debug = args.iter().position(|a| a == "--debug");
    if let Some(idx) = debug {
        args.remove(idx);
    }

    let mut engine = Engine::new();
    engine.set_debug(debug.is_some());
    engine.register_skill(Box::new(CurrentTimeSkill::new()));
    engine.register_skill(Box::new(DateSkill::new()));
    engine.register_skill(Box::new(CalculatorSkill::new()));
    engine.register_skill(Box::new(GreetingSkill::new()));
    engine.register_skill(Box::new(OpenSkill::new()));
    engine.register_skill(Box::new(SearchSkill::new()));

    if !args.is_empty() {
        let input = args.join(" ");
        let response = engine.process_input(&input);
        print_response(&response);
        return;
    }

    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.trim().is_empty() {
            continue;
        }
        let response = engine.process_input(&line);
        print_response(&response);
    }
}

fn print_response(response: &ari_core::Response) {
    match response {
        ari_core::Response::Text(s) => println!("{s}"),
        ari_core::Response::Action(v) => println!("{}", serde_json::to_string_pretty(v).unwrap_or_else(|_| v.to_string())),
        ari_core::Response::Binary { mime, data } => {
            println!("[binary: {mime}, {} bytes]", data.len())
        }
    }
}
