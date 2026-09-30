use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{fs::File, io::BufReader, path::Path};

use crate::enzyme;

#[derive(Clone, Debug, Deserialize)]
pub struct Experiment {
    pub schema: String,
    pub nx: usize,
    pub nz: usize,
    pub dx: f64,
    pub dt: f64,
    pub steps: usize,
    pub source_frequency: f64,
    pub source_peak_time: f64,
    pub source_amplitude: f64,
    pub sponge_width: usize,
    pub sponge_strength: f64,
    pub shots: Vec<[f64; 2]>,
    pub receivers: Vec<[f64; 2]>,
    pub length_unit_m: f64,
    pub time_unit_s: f64,
    pub name: String,
    pub background: Vec<Vec<f64>>,
    pub perturbation: Vec<Vec<f64>>,
    pub provenance: Value,
}

#[derive(Clone, Debug)]
pub struct Prepared {
    pub experiment: Experiment,
    pub background: Vec<f64>,
    pub perturbation: Vec<f64>,
    pub damping: Vec<f64>,
    pub shots: Vec<[usize; 2]>,
    pub receivers: Vec<[usize; 2]>,
    pub source_footprints: Vec<Vec<f64>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct State {
    pub previous: Vec<f64>,
    pub current: Vec<f64>,
}

impl State {
    pub fn zeros(n: usize) -> Self {
        Self { previous: vec![0.0; n], current: vec![0.0; n] }
    }
}

impl Experiment {
    pub fn read(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let value: Self = serde_json::from_reader(BufReader::new(file))
            .with_context(|| format!("parse {}", path.display()))?;
        value.validate()?;
        Ok(value)
    }

    fn validate(&self) -> Result<()> {
        ensure!(self.schema == "week5-seismic-experiment-v1", "unsupported schema {}", self.schema);
        ensure!(self.background.len() == self.nz && self.perturbation.len() == self.nz, "field row count mismatch");
        ensure!(self.background.iter().all(|row| row.len() == self.nx), "background shape mismatch");
        ensure!(self.perturbation.iter().all(|row| row.len() == self.nx), "perturbation shape mismatch");
        for &[x, z] in self.shots.iter().chain(self.receivers.iter()) {
            ensure!(x.fract() == 0.0 && z.fract() == 0.0 && x >= 0.0 && z >= 0.0, "survey point [{x},{z}] is not a grid index");
            ensure!((x as usize) < self.nx && (z as usize) < self.nz, "survey point [{x},{z}] outside grid");
        }
        ensure!(self.nx >= 3 && self.nz >= 3 && self.dx > 0.0 && self.dt > 0.0, "invalid grid");
        Ok(())
    }

    pub fn compact_json(&self) -> Value {
        json!({
            "schema": self.schema,
            "name": self.name,
            "nx": self.nx,
            "nz": self.nz,
            "dx": self.dx,
            "dt": self.dt,
            "steps": self.steps,
            "source_frequency": self.source_frequency,
            "source_peak_time": self.source_peak_time,
            "source_amplitude": self.source_amplitude,
            "sponge_width": self.sponge_width,
            "sponge_strength": self.sponge_strength,
            "shots": self.shots,
            "receivers": self.receivers,
            "length_unit_m": self.length_unit_m,
            "time_unit_s": self.time_unit_s,
            "provenance": self.provenance,
        })
    }

    pub fn prepare(self) -> Result<Prepared> {
        let background = flatten(&self.background, self.nx, self.nz)?;
        let perturbation = flatten(&self.perturbation, self.nx, self.nz)?;
        let mut damping = vec![0.0; self.nx * self.nz];
        for z in 0..self.nz {
            for x in 0..self.nx {
                let edge = x.min(self.nx - 1 - x).min(z).min(self.nz - 1 - z);
                let taper = if edge < self.sponge_width {
                    1.0 - edge as f64 / self.sponge_width as f64
                } else {
                    0.0
                };
                damping[z * self.nx + x] = self.sponge_strength * taper * taper;
            }
        }
        let shots: Vec<[usize; 2]> = self.shots.iter().map(|&[x, z]| [x as usize, z as usize]).collect();
        let receivers: Vec<[usize; 2]> = self.receivers.iter().map(|&[x, z]| [x as usize, z as usize]).collect();
        let source_footprints = shots.iter().map(|&shot| gaussian(self.nx, self.nz, shot)).collect();
        Ok(Prepared { experiment: self, background, perturbation, damping, shots, receivers, source_footprints })
    }
}

fn flatten(rows: &[Vec<f64>], nx: usize, nz: usize) -> Result<Vec<f64>> {
    if rows.len() != nz || rows.iter().any(|row| row.len() != nx) {
        bail!("field does not have shape [{nz}, {nx}]");
    }
    Ok(rows.iter().flat_map(|row| row.iter().copied()).collect())
}

fn gaussian(nx: usize, nz: usize, [sx, sz]: [usize; 2]) -> Vec<f64> {
    let mut result = vec![0.0; nx * nz];
    for z in 0..nz {
        for x in 0..nx {
            let dx = x as f64 - sx as f64;
            let dz = z as f64 - sz as f64;
            result[z * nx + x] = (-0.5 * (dx * dx + dz * dz)).exp();
        }
    }
    result
}

pub fn source_scale(experiment: &Experiment, step: usize) -> f64 {
    let time = step as f64 * experiment.dt;
    let theta = std::f64::consts::PI * experiment.source_frequency * (time - experiment.source_peak_time);
    experiment.source_amplitude * (1.0 - 2.0 * theta * theta) * (-theta * theta).exp()
}

pub fn source_for(prepared: &Prepared, shot: usize, step: usize) -> Vec<f64> {
    let scale = source_scale(&prepared.experiment, step);
    prepared.source_footprints[shot].iter().map(|x| scale * x).collect()
}

pub fn advance(prepared: &Prepared, velocity: &[f64], shot: usize, step: usize, state: &State) -> Result<State> {
    let e = &prepared.experiment;
    let source = source_for(prepared, shot, step);
    let next = enzyme::primal(
        &state.previous, &state.current, velocity, &prepared.damping, &source,
        e.nx, e.nz, e.dx, e.dt,
    )?;
    Ok(State { previous: state.current.clone(), current: next })
}

pub fn l2(values: &[f64]) -> f64 {
    values.iter().map(|x| x * x).sum::<f64>().sqrt()
}
