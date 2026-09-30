use anyhow::{Context, Result, ensure};
use clap::{Parser, ValueEnum};
use ndarray::{Array2, Array3};
use ndarray_npy::{read_npy, write_npy};
use seismic::{
    model::{Experiment, l2},
    solver,
};
use serde_json::{Map, Value, json};
use std::{
    fs::{self, File},
    io::BufWriter,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, ValueEnum)]
enum Mode {
    Forward,
    Born,
    Adjoint,
}

#[derive(Clone, Debug, ValueEnum)]
enum Storage {
    Full,
    Treeverse,
}

#[derive(Debug, Parser)]
#[command(about = "Acoustic wave simulation and Enzyme derivatives")]
struct Cli {
    #[arg(long)]
    experiment: PathBuf,
    #[arg(long, value_enum)]
    mode: Mode,
    #[arg(long)]
    out: PathBuf,
    #[arg(long)]
    every: Option<usize>,
    #[arg(long)]
    data: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "full")]
    storage: Storage,
    #[arg(long)]
    checkpoints: Option<usize>,
}

fn write_json(path: &Path, value: &impl serde::Serialize) -> Result<()> {
    let file = File::create(path).with_context(|| format!("create {}", path.display()))?;
    serde_json::to_writer_pretty(BufWriter::new(file), value)
        .with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    ensure!(cli.every.is_none_or(|x| x > 0), "--every must be positive");
    fs::create_dir_all(&cli.out)?;
    let experiment = Experiment::read(&cli.experiment)?;
    let compact = experiment.compact_json();
    let prepared = experiment.prepare()?;
    let e = &prepared.experiment;
    let mode_name = match cli.mode {
        Mode::Forward => "forward",
        Mode::Born => "born",
        Mode::Adjoint => "adjoint",
    };
    let mut run = Map::new();
    run.insert("experiment_file".into(), json!(cli.experiment));
    run.insert("experiment".into(), compact);
    let mut result = json!({
        "mode": mode_name,
        "nx": e.nx,
        "nz": e.nz,
        "dx": e.dx,
        "dt": e.dt,
        "steps": e.steps,
        "shots": prepared.shots.len(),
        "receivers": prepared.receivers.len(),
    });
    println!("shot\tmode\tdata_l2_norm");

    match cli.mode {
        Mode::Forward => {
            let output = solver::forward(&prepared, cli.every)?;
            let traces = Array3::from_shape_vec(
                (prepared.shots.len(), e.steps, prepared.receivers.len()),
                output.traces.clone(),
            )?;
            write_npy(cli.out.join("traces.npy"), &traces)?;
            for shot in 0..prepared.shots.len() {
                let start = shot * e.steps * prepared.receivers.len();
                let end = start + e.steps * prepared.receivers.len();
                println!("{shot}\tforward\t{:.12e}", l2(&output.traces[start..end]));
            }
            if cli.every.is_some() {
                let wavefield = Array3::from_shape_vec(
                    (output.frame_steps.len(), e.nz, e.nx),
                    output.wavefield,
                )?;
                let echo =
                    Array3::from_shape_vec((output.frame_steps.len(), e.nz, e.nx), output.echo)?;
                write_npy(cli.out.join("wavefield.npy"), &wavefield)?;
                write_npy(cli.out.join("echo.npy"), &echo)?;
                run.insert("recording".into(), json!({
                    "every": cli.every,
                    "steps": output.frame_steps,
                    "times": output.frame_steps.iter().map(|&s| s as f64 * e.dt).collect::<Vec<_>>(),
                }));
            }
        }
        Mode::Born => {
            let output = solver::born(&prepared)?;
            let data = Array3::from_shape_vec(
                (prepared.shots.len(), e.steps, prepared.receivers.len()),
                output.data.clone(),
            )?;
            write_npy(cli.out.join("born_data.npy"), &data)?;
            for shot in 0..prepared.shots.len() {
                let start = shot * e.steps * prepared.receivers.len();
                let end = start + e.steps * prepared.receivers.len();
                println!("{shot}\tborn\t{:.12e}", l2(&output.data[start..end]));
            }
        }
        Mode::Adjoint => {
            let data_path = cli.data.as_ref().context("adjoint mode requires --data")?;
            let data: Array3<f64> =
                read_npy(data_path).with_context(|| format!("read {}", data_path.display()))?;
            ensure!(
                data.shape() == [prepared.shots.len(), e.steps, prepared.receivers.len()],
                "data shape {:?} does not match experiment",
                data.shape()
            );
            let flat = data.into_raw_vec_and_offset().0;
            let output = match cli.storage {
                Storage::Full => {
                    ensure!(
                        cli.checkpoints.is_none(),
                        "--checkpoints is only valid with treeverse storage"
                    );
                    solver::adjoint_full(&prepared, &flat, cli.every)?
                }
                Storage::Treeverse => {
                    ensure!(cli.every.is_none(), "treeverse recording is not supported");
                    solver::adjoint_treeverse(
                        &prepared,
                        &flat,
                        cli.checkpoints
                            .context("treeverse storage requires --checkpoints")?,
                    )?
                }
            };
            let image = Array2::from_shape_vec((e.nz, e.nx), output.image.clone())?;
            write_npy(cli.out.join("image.npy"), &image)?;
            println!("all\tadjoint\t{:.12e}", l2(&output.image));
            if !output.wavefield.is_empty() {
                let wavefield = Array3::from_shape_vec(
                    (output.frame_steps.len(), e.nz, e.nx),
                    output.wavefield,
                )?;
                write_npy(cli.out.join("wavefield.npy"), &wavefield)?;
                run.insert("recording".into(), json!({
                    "every": cli.every,
                    "steps": output.frame_steps,
                    "times": output.frame_steps.iter().map(|&s| s as f64 * e.dt).collect::<Vec<_>>(),
                }));
            }
            for (shot, log) in output.action_logs.iter().enumerate() {
                write_json(&cli.out.join(format!("actions-{shot}.json")), log)?;
            }
            result.as_object_mut().unwrap().insert(
                "statistics".into(),
                serde_json::to_value(&output.statistics)?,
            );
        }
    }
    write_json(&cli.out.join("run.json"), &Value::Object(run))?;
    write_json(&cli.out.join("result.json"), &result)?;
    Ok(())
}
