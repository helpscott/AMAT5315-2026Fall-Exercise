use serde_json::json;
use std::collections::HashMap;
use std::env;
use std::fs::{self, File};
use std::io::{self, BufWriter, Read, Write};
use std::path::Path;
use week4_fluid::integrator::{Euler, Integrator, Midpoint, Rk4};
use week4_fluid::{FieldData, FlowSolver};

struct Config {
    method: String,
    nu: f64,
    dt: f64,
    t_end: f64,
    every: f64,
    out: String,
}

fn usage() -> ! {
    eprintln!("usage: fluid --method euler|rk2|rk4 --nu NU --dt DT --t-end T --every DT --out DIR");
    std::process::exit(2)
}

fn parse() -> Config {
    let arguments: Vec<String> = env::args().skip(1).collect();
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
    let get = |name: &str| values.get(name).unwrap_or_else(|| usage());
    let method = get("method").clone();
    if !matches!(method.as_str(), "euler" | "rk2" | "rk4") {
        usage();
    }
    let config = Config {
        method,
        nu: get("nu").parse().unwrap_or_else(|_| usage()),
        dt: get("dt").parse().unwrap_or_else(|_| usage()),
        t_end: get("t-end").parse().unwrap_or_else(|_| usage()),
        every: get("every").parse().unwrap_or_else(|_| usage()),
        out: get("out").clone(),
    };
    if config.nu < 0.0 || config.dt <= 0.0 || config.t_end < 0.0 || config.every <= 0.0 {
        usage();
    }
    config
}

fn write_array(writer: &mut BufWriter<File>, name: &str, values: &[f64]) -> io::Result<()> {
    write!(writer, ",\"{name}\":[")?;
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            write!(writer, ",")?;
        }
        write!(writer, "{value:.6}")?;
    }
    write!(writer, "]")
}

fn write_frame(
    writer: &mut BufWriter<File>,
    solver: &FlowSolver,
    state: &[week4_fluid::Complex64],
    time: f64,
    step: usize,
) -> io::Result<()> {
    let (u, v, omega) = solver.fields_from_vorticity(state);
    write!(writer, "{{\"t\":{time:.6},\"step\":{step}")?;
    write_array(writer, "u", &u)?;
    write_array(writer, "v", &v)?;
    write_array(writer, "omega", &omega)?;
    writeln!(writer, "}}")
}

fn main() -> io::Result<()> {
    let config = parse();
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    let field: FieldData = serde_json::from_str(&input).map_err(io::Error::other)?;
    if field.u.len() != field.n * field.n || field.v.len() != field.n * field.n {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "field array length",
        ));
    }
    fs::create_dir_all(&config.out)?;
    let run = json!({
        "case": field.case,
        "n": field.n,
        "seed": field.seed,
        "k_band": field.k_band,
        "method": config.method,
        "nu": config.nu,
        "dt": config.dt,
        "t_end": config.t_end,
        "snapshot_every": config.every,
    });
    serde_json::to_writer_pretty(File::create(Path::new(&config.out).join("run.json"))?, &run)?;
    let mut fields = BufWriter::new(File::create(Path::new(&config.out).join("fields.jsonl"))?);
    let solver = FlowSolver::new(field.n, config.nu);
    let mut state = solver.vorticity_from_velocity(&field.u, &field.v);
    let method: Box<dyn Integrator> = match config.method.as_str() {
        "euler" => Box::new(Euler),
        "rk2" => Box::new(Midpoint),
        "rk4" => Box::new(Rk4),
        _ => unreachable!(),
    };
    let snapshot_steps = (config.every / config.dt).round().max(1.0) as usize;
    let total_steps = (config.t_end / config.dt).ceil() as usize;
    println!("t\tE\tZ");
    let (initial_energy, initial_enstrophy) = solver.diagnostics(&state);
    println!("{:.6}\t{initial_energy:.6}\t{initial_enstrophy:.6}", 0.0);
    write_frame(&mut fields, &solver, &state, 0.0, 0)?;
    for step in 1..=total_steps {
        let time = (step - 1) as f64 * config.dt;
        let mut rate = |_stage_time: f64,
                        values: &[week4_fluid::Complex64],
                        output: &mut [week4_fluid::Complex64]| {
            solver.rate(values, output);
        };
        state = method.step(time, &state, config.dt, &mut rate);
        solver.mask(&mut state);
        let next_time = step as f64 * config.dt;
        let (energy, enstrophy) = solver.diagnostics(&state);
        if !energy.is_finite() || !enstrophy.is_finite() {
            println!("{next_time:.6}\t{energy}\t{enstrophy}");
            fields.flush()?;
            std::process::exit(1);
        }
        if step % snapshot_steps == 0 {
            println!("{next_time:.6}\t{energy:.6}\t{enstrophy:.6}");
            write_frame(&mut fields, &solver, &state, next_time, step)?;
        }
    }
    fields.flush()?;
    Ok(())
}
