use std::collections::HashMap;
use std::env;
use week4_fluid::flow::{random_field, taylor_green};

fn usage() -> ! {
    eprintln!(
        "usage: field taylor-green --n N [--nu NU --t T] | field random --n N --seed S --k-min A --k-max B"
    );
    std::process::exit(2)
}

fn parse_values(arguments: &[String]) -> HashMap<String, String> {
    let mut values = HashMap::new();
    let mut index = 0;
    while index < arguments.len() {
        if !arguments[index].starts_with("--") || index + 1 == arguments.len() {
            usage();
        }
        values.insert(
            arguments[index].trim_start_matches("--").to_string(),
            arguments[index + 1].clone(),
        );
        index += 2;
    }
    values
}

fn required<T: std::str::FromStr>(values: &HashMap<String, String>, name: &str) -> T {
    values
        .get(name)
        .unwrap_or_else(|| usage())
        .parse()
        .unwrap_or_else(|_| usage())
}

fn main() {
    let arguments: Vec<String> = env::args().skip(1).collect();
    if arguments.is_empty() {
        usage();
    }
    let case = &arguments[0];
    let values = parse_values(&arguments[1..]);
    let n: usize = required(&values, "n");
    if n < 4 || n % 2 != 0 {
        usage();
    }
    let field = match case.as_str() {
        "taylor-green" => {
            let time: f64 = values
                .get("t")
                .map(|value| value.parse().unwrap_or_else(|_| usage()))
                .unwrap_or(0.0);
            let viscosity: f64 = if time > 0.0 {
                required(&values, "nu")
            } else {
                values
                    .get("nu")
                    .map(|value| value.parse().unwrap_or_else(|_| usage()))
                    .unwrap_or(0.0)
            };
            taylor_green(n, viscosity, time)
        }
        "random" => random_field(
            n,
            required(&values, "seed"),
            required(&values, "k-min"),
            required(&values, "k-max"),
        ),
        _ => usage(),
    };
    println!("{}", serde_json::to_string(&field).unwrap());
}
