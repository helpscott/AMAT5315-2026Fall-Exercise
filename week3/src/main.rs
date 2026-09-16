use ising::Ising;
use std::{
    env,
    fs::{self, File},
    io::{self, Write},
    path::Path,
};

#[derive(Clone, Copy, Debug, PartialEq)]
enum Update {
    Metropolis,
    Wolff,
}
struct Config {
    update: Update,
    l: usize,
    t_from: f64,
    t_to: f64,
    t_step: f64,
    discard: usize,
    measure: usize,
    every: usize,
    seed: u64,
    out: String,
}
fn usage() -> ! {
    eprintln!(
        "usage: ising --update metropolis|wolff --l L --t-from A --t-to B --t-step D --discard N --measure N --every K --seed S --out DIR"
    );
    std::process::exit(2)
}
fn parse() -> Config {
    let a: Vec<String> = env::args().collect();
    let mut vals = std::collections::HashMap::new();
    let mut i = 1;
    while i < a.len() {
        if !a[i].starts_with("--") || i + 1 >= a.len() {
            usage()
        }
        vals.insert(a[i][2..].to_string(), a[i + 1].clone());
        i += 2;
    }
    let update = match vals.get("update").map(String::as_str) {
        Some("metropolis") => Update::Metropolis,
        Some("wolff") => Update::Wolff,
        _ => usage(),
    };
    let get = |k: &str| {
        vals.get(k).unwrap_or_else(|| {
            eprintln!("missing --{k}");
            usage()
        })
    };
    let l = get("l").parse().unwrap_or_else(|_| usage());
    let t_from = get("t-from").parse().unwrap_or_else(|_| usage());
    let t_to = get("t-to").parse().unwrap_or_else(|_| usage());
    let t_step = get("t-step").parse().unwrap_or_else(|_| usage());
    let discard = get("discard").parse().unwrap_or_else(|_| usage());
    let measure = get("measure").parse().unwrap_or_else(|_| usage());
    let seed = get("seed").parse().unwrap_or_else(|_| usage());
    let every = vals
        .get("every")
        .map(|x| x.parse().unwrap_or(0))
        .unwrap_or(0);
    let out = get("out").clone();
    if l < 2 || t_step <= 0.0 || t_to < t_from {
        usage()
    };
    Config {
        update,
        l,
        t_from,
        t_to,
        t_step,
        discard,
        measure,
        every,
        seed,
        out,
    }
}
fn grid(c: &Config) -> Vec<f64> {
    let n = ((c.t_to - c.t_from) / c.t_step + 1e-9).floor() as usize;
    (0..=n).map(|i| c.t_from + i as f64 * c.t_step).collect()
}
fn jnum(x: f64) -> String {
    format!("{x:.6}")
}
fn main() -> io::Result<()> {
    let c = parse();
    let ts = grid(&c);
    fs::create_dir_all(&c.out)?;
    let mut run = File::create(Path::new(&c.out).join("run.json"))?;
    let update = match c.update {
        Update::Metropolis => "metropolis",
        Update::Wolff => "wolff",
    };
    writeln!(
        run,
        "{{\"L\":{},\"update\":\"{}\",\"t_grid\":[{}],\"discard\":{},\"measure\":{},\"seed\":{},\"sample_every\":1,\"time_unit\":\"{}\"}}",
        c.l,
        update,
        ts.iter().map(|t| jnum(*t)).collect::<Vec<_>>().join(","),
        c.discard,
        c.measure,
        c.seed,
        if c.update == Update::Metropolis {
            "sweep"
        } else {
            "cluster_flip"
        }
    )?;
    let mut series = File::create(Path::new(&c.out).join("series.jsonl"))?;
    let mut spins_file = if c.every > 0 {
        Some(File::create(Path::new(&c.out).join("spins.jsonl"))?)
    } else {
        None
    };
    println!(
        "T\tmean_abs_M\t{}",
        if c.update == Update::Metropolis {
            "acceptance"
        } else {
            "mean_cluster_size"
        }
    );
    let mut model = Ising::new(c.l, c.seed);
    let mut global_step = 0usize;
    for &t in &ts {
        let mut accepted = 0usize;
        let mut proposals = 0usize;
        let mut clusters = 0usize;
        let mut abs_sum = 0.0;
        for _ in 0..c.discard {
            global_step += 1;
            match c.update {
                Update::Metropolis => {
                    accepted += model.metropolis_sweep(t);
                    proposals += c.l * c.l
                }
                Update::Wolff => {
                    clusters += model.wolff_move(t);
                }
            }
        }
        for sweep in 1..=c.measure {
            global_step += 1;
            let cl = match c.update {
                Update::Metropolis => {
                    accepted += model.metropolis_sweep(t);
                    proposals += c.l * c.l;
                    0
                }
                Update::Wolff => {
                    let n = model.wolff_move(t);
                    clusters += n;
                    n
                }
            };
            let m = model.magnetization();
            let e = model.energy();
            abs_sum += m.abs();
            if c.update == Update::Metropolis {
                writeln!(
                    series,
                    "{{\"L\":{},\"T\":{},\"sweep\":{},\"M\":{},\"E\":{}}}",
                    c.l,
                    jnum(t),
                    sweep,
                    jnum(m),
                    jnum(e)
                )?;
            } else {
                writeln!(
                    series,
                    "{{\"L\":{},\"T\":{},\"sweep\":{},\"M\":{},\"E\":{},\"cluster_size\":{}}}",
                    c.l,
                    jnum(t),
                    sweep,
                    jnum(m),
                    jnum(e),
                    cl
                )?;
            }
            if let Some(f) = spins_file.as_mut() {
                if sweep % c.every == 0 {
                    let body = model
                        .state()
                        .iter()
                        .map(|s| s.to_string())
                        .collect::<Vec<_>>()
                        .join(",");
                    writeln!(
                        f,
                        "{{\"L\":{},\"T\":{},\"sweep\":{},\"m\":{},\"spins\":[{}]}}",
                        c.l,
                        jnum(t),
                        global_step,
                        jnum(m),
                        body
                    )?;
                }
            }
        }
        let rate = if c.update == Update::Metropolis {
            accepted as f64 / proposals.max(1) as f64
        } else {
            clusters as f64 / (c.discard + c.measure) as f64
        };
        println!(
            "{t:.6}\t{:.6}\t{rate:.6}",
            abs_sum / c.measure.max(1) as f64
        );
    }
    Ok(())
}
