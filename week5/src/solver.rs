use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::{
    enzyme,
    model::{Prepared, State, advance, source_for},
};

#[derive(Debug)]
pub struct ForwardResult {
    pub traces: Vec<f64>,
    pub wavefield: Vec<f32>,
    pub echo: Vec<f32>,
    pub frame_steps: Vec<usize>,
}

#[derive(Debug)]
pub struct BornResult {
    pub data: Vec<f64>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ReverseStatistics {
    pub storage: String,
    pub checkpoints: Option<usize>,
    pub reverse_calls: usize,
    pub scheduler_forward_calls: usize,
    pub peak_saved_states: usize,
    pub peak_saved_bytes: usize,
    pub per_shot: Vec<ShotStatistics>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ShotStatistics {
    pub reverse_calls: usize,
    pub scheduler_forward_calls: usize,
    pub peak_saved_states: usize,
}

#[derive(Debug)]
pub struct AdjointResult {
    pub image: Vec<f64>,
    pub wavefield: Vec<f32>,
    pub frame_steps: Vec<usize>,
    pub statistics: ReverseStatistics,
    pub action_logs: Vec<ActionLog>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Action {
    pub action: String,
    pub tau: usize,
    pub delta: usize,
    pub step: usize,
    pub depth: usize,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ActionLog {
    pub actions: Vec<Action>,
    pub peak_mem: usize,
}

#[derive(Clone, Debug)]
struct AdjointState {
    previous: Vec<f64>,
    current: Vec<f64>,
}

impl AdjointState {
    fn zeros(n: usize) -> Self {
        Self {
            previous: vec![0.0; n],
            current: vec![0.0; n],
        }
    }
}

fn velocity_with_perturbation(prepared: &Prepared) -> Vec<f64> {
    prepared
        .background
        .iter()
        .zip(&prepared.perturbation)
        .map(|(a, b)| a + b)
        .collect()
}

pub fn forward(prepared: &Prepared, every: Option<usize>) -> Result<ForwardResult> {
    let e = &prepared.experiment;
    let n = e.nx * e.nz;
    let nr = prepared.receivers.len();
    let mut traces = vec![0.0; prepared.shots.len() * e.steps * nr];
    let mut wavefield = Vec::new();
    let mut echo = Vec::new();
    let mut frame_steps = Vec::new();
    let perturbed = velocity_with_perturbation(prepared);

    for shot in 0..prepared.shots.len() {
        let mut state = State::zeros(n);
        let mut perturbed_state = State::zeros(n);
        if shot == 0 && every.is_some() {
            wavefield.extend(state.current.iter().map(|&x| x as f32));
            echo.extend(std::iter::repeat_n(0.0_f32, n));
            frame_steps.push(0);
        }
        for step in 0..e.steps {
            state = advance(prepared, &prepared.background, shot, step, &state)?;
            if shot == 0 && every.is_some() {
                perturbed_state = advance(prepared, &perturbed, shot, step, &perturbed_state)?;
            }
            for (receiver, [x, z]) in prepared.receivers.iter().copied().enumerate() {
                traces[(shot * e.steps + step) * nr + receiver] = state.current[z * e.nx + x];
            }
            let state_step = step + 1;
            if shot == 0 && every.is_some_and(|stride| state_step % stride == 0) {
                wavefield.extend(state.current.iter().map(|&x| x as f32));
                echo.extend(
                    perturbed_state
                        .current
                        .iter()
                        .zip(&state.current)
                        .map(|(a, b)| (a - b) as f32),
                );
                frame_steps.push(state_step);
            }
        }
    }
    Ok(ForwardResult {
        traces,
        wavefield,
        echo,
        frame_steps,
    })
}

pub fn born(prepared: &Prepared) -> Result<BornResult> {
    let e = &prepared.experiment;
    let n = e.nx * e.nz;
    let nr = prepared.receivers.len();
    let mut data = vec![0.0; prepared.shots.len() * e.steps * nr];
    for shot in 0..prepared.shots.len() {
        let mut state = State::zeros(n);
        let mut tangent = State::zeros(n);
        for step in 0..e.steps {
            let source = source_for(prepared, shot, step);
            let (next, d_next) = enzyme::jvp(
                &state.previous,
                &tangent.previous,
                &state.current,
                &tangent.current,
                &prepared.background,
                &prepared.perturbation,
                &prepared.damping,
                &source,
                e.nx,
                e.nz,
                e.dx,
                e.dt,
            )?;
            state = State {
                previous: state.current,
                current: next,
            };
            tangent = State {
                previous: tangent.current,
                current: d_next,
            };
            for (receiver, [x, z]) in prepared.receivers.iter().copied().enumerate() {
                data[(shot * e.steps + step) * nr + receiver] = tangent.current[z * e.nx + x];
            }
        }
    }
    Ok(BornResult { data })
}

fn reverse_one(
    prepared: &Prepared,
    shot: usize,
    step: usize,
    primal: &State,
    mut output: AdjointState,
    data: &[f64],
    image: &mut [f64],
) -> Result<AdjointState> {
    let e = &prepared.experiment;
    let n = e.nx * e.nz;
    let nr = prepared.receivers.len();
    for (receiver, [x, z]) in prepared.receivers.iter().copied().enumerate() {
        output.current[z * e.nx + x] += data[(shot * e.steps + step) * nr + receiver];
    }
    let source = source_for(prepared, shot, step);
    let (d_previous, mut d_current, d_velocity) = enzyme::vjp(
        &primal.previous,
        &primal.current,
        &prepared.background,
        &prepared.damping,
        &source,
        &output.current,
        e.nx,
        e.nz,
        e.dx,
        e.dt,
    )?;
    ensure!(d_velocity.len() == n, "bad velocity adjoint length");
    for i in 0..n {
        d_current[i] += output.previous[i];
        image[i] += d_velocity[i];
    }
    Ok(AdjointState {
        previous: d_previous,
        current: d_current,
    })
}

pub fn adjoint_full(
    prepared: &Prepared,
    data: &[f64],
    every: Option<usize>,
) -> Result<AdjointResult> {
    let e = &prepared.experiment;
    let n = e.nx * e.nz;
    let expected = prepared.shots.len() * e.steps * prepared.receivers.len();
    ensure!(
        data.len() == expected,
        "data has {} values, expected {expected}",
        data.len()
    );
    let mut image = vec![0.0; n];
    let mut wavefield = Vec::new();
    let mut frame_steps = Vec::new();
    let mut per_shot = Vec::new();

    for shot in 0..prepared.shots.len() {
        let mut states = Vec::with_capacity(e.steps + 1);
        states.push(State::zeros(n));
        for step in 0..e.steps {
            let next = advance(
                prepared,
                &prepared.background,
                shot,
                step,
                states.last().unwrap(),
            )?;
            states.push(next);
        }
        let mut adjoint = AdjointState::zeros(n);
        if shot == 0 && every.is_some() {
            wavefield.extend(adjoint.current.iter().map(|&x| x as f32));
            frame_steps.push(e.steps);
        }
        for step in (0..e.steps).rev() {
            adjoint = reverse_one(
                prepared,
                shot,
                step,
                &states[step],
                adjoint,
                data,
                &mut image,
            )?;
            if shot == 0 && every.is_some_and(|stride| step % stride == 0) {
                wavefield.extend(adjoint.current.iter().map(|&x| x as f32));
                frame_steps.push(step);
            }
        }
        per_shot.push(ShotStatistics {
            reverse_calls: e.steps,
            scheduler_forward_calls: e.steps,
            peak_saved_states: e.steps + 1,
        });
    }
    let peak_saved_states = e.steps + 1;
    Ok(AdjointResult {
        image,
        wavefield,
        frame_steps,
        statistics: ReverseStatistics {
            storage: "full".into(),
            checkpoints: None,
            reverse_calls: prepared.shots.len() * e.steps,
            scheduler_forward_calls: prepared.shots.len() * e.steps,
            peak_saved_states,
            peak_saved_bytes: peak_saved_states * 2 * n * 8,
            per_shot,
        },
        action_logs: Vec::new(),
    })
}

fn binomial(n: usize, mut k: usize) -> u128 {
    if k > n {
        return 0;
    }
    k = k.min(n - k);
    let mut value = 1_u128;
    for i in 1..=k {
        value = value * (n - k + i) as u128 / i as u128;
    }
    value
}

fn binomial_fit(steps: usize, delta: usize) -> usize {
    let mut tau = 1;
    while steps as u128 > binomial(tau + delta, tau) {
        tau += 1;
    }
    tau
}

fn midpoint(delta: usize, tau: usize, sigma: usize, phi: usize) -> usize {
    let denominator = tau + delta;
    let mut kappa = (delta * sigma + tau * phi).div_ceil(denominator);
    if kappa >= phi && delta > 0 {
        kappa = (sigma + 1).max(phi - 1);
    }
    kappa
}

struct TreeverseContext<'a> {
    prepared: &'a Prepared,
    shot: usize,
    data: &'a [f64],
    image: &'a mut [f64],
    states: BTreeMap<usize, State>,
    log: ActionLog,
}

impl TreeverseContext<'_> {
    fn push(&mut self, action: &str, tau: usize, delta: usize, step: usize, depth: usize) {
        self.log.actions.push(Action {
            action: action.into(),
            tau,
            delta,
            step,
            depth,
        });
    }

    fn recurse(
        &mut self,
        gradient: Option<AdjointState>,
        mut delta: usize,
        mut tau: usize,
        beta: usize,
        sigma: usize,
        mut phi: usize,
        depth: usize,
    ) -> Result<AdjointState> {
        if sigma > beta {
            delta -= 1;
            let mut state = self.states.get(&beta).expect("valid restore").clone();
            self.push("restore", tau, delta, beta, depth);
            for step in beta..sigma {
                state = advance(
                    self.prepared,
                    &self.prepared.background,
                    self.shot,
                    step,
                    &state,
                )?;
                self.push("call", tau, delta, step, depth);
            }
            self.states.insert(sigma, state);
            self.push("store", tau, delta, sigma, depth);
            self.log.peak_mem = self.log.peak_mem.max(self.states.len());
        }
        let mut gradient = gradient;
        let mut kappa = midpoint(delta, tau, sigma, phi);
        while tau > 0 && kappa < phi {
            gradient = Some(self.recurse(gradient, delta, tau, sigma, kappa, phi, depth + 1)?);
            tau -= 1;
            phi = kappa;
            kappa = midpoint(delta, tau, sigma, phi);
        }
        let output = gradient.unwrap_or_else(|| {
            AdjointState::zeros(self.prepared.experiment.nx * self.prepared.experiment.nz)
        });
        let primal = self.states.get(&sigma).expect("scheduled state").clone();
        let result = reverse_one(
            self.prepared,
            self.shot,
            sigma,
            &primal,
            output,
            self.data,
            self.image,
        )?;
        self.push("grad", tau, delta, sigma, depth);
        if sigma > beta {
            self.states.remove(&sigma);
            self.push("fetch", tau, delta, sigma, depth);
        }
        Ok(result)
    }
}

pub fn adjoint_treeverse(
    prepared: &Prepared,
    data: &[f64],
    checkpoints: usize,
) -> Result<AdjointResult> {
    let e = &prepared.experiment;
    ensure!(
        checkpoints >= 1,
        "treeverse needs at least one checkpoint slot"
    );
    let n = e.nx * e.nz;
    let expected = prepared.shots.len() * e.steps * prepared.receivers.len();
    ensure!(
        data.len() == expected,
        "data has {} values, expected {expected}",
        data.len()
    );
    let mut image = vec![0.0; n];
    let mut logs = Vec::new();
    let mut per_shot = Vec::new();
    for shot in 0..prepared.shots.len() {
        let tau = binomial_fit(e.steps, checkpoints);
        let mut states = BTreeMap::new();
        states.insert(0, State::zeros(n));
        let mut context = TreeverseContext {
            prepared,
            shot,
            data,
            image: &mut image,
            states,
            log: ActionLog {
                actions: Vec::new(),
                peak_mem: 1,
            },
        };
        context.recurse(None, checkpoints, tau, 0, 0, e.steps, 0)?;
        let calls = context
            .log
            .actions
            .iter()
            .filter(|a| a.action == "call")
            .count();
        let grads = context
            .log
            .actions
            .iter()
            .filter(|a| a.action == "grad")
            .count();
        per_shot.push(ShotStatistics {
            reverse_calls: grads,
            scheduler_forward_calls: calls,
            peak_saved_states: context.log.peak_mem,
        });
        logs.push(context.log);
    }
    let peak_saved_states = logs.iter().map(|log| log.peak_mem).max().unwrap_or(1);
    let reverse_calls = per_shot.iter().map(|x| x.reverse_calls).sum();
    let scheduler_forward_calls = per_shot.iter().map(|x| x.scheduler_forward_calls).sum();
    Ok(AdjointResult {
        image,
        wavefield: Vec::new(),
        frame_steps: Vec::new(),
        statistics: ReverseStatistics {
            storage: "treeverse".into(),
            checkpoints: Some(checkpoints),
            reverse_calls,
            scheduler_forward_calls,
            peak_saved_states,
            peak_saved_bytes: peak_saved_states * 2 * n * 8,
            per_shot,
        },
        action_logs: logs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_treeverse_call_counts() {
        let expected = [(1, 28_680), (3, 1_695), (5, 990), (10, 642)];
        for (delta, calls) in expected {
            let tau = binomial_fit(240, delta);
            fn count(
                delta: usize,
                mut tau: usize,
                beta: usize,
                sigma: usize,
                mut phi: usize,
            ) -> usize {
                let mut total = sigma - beta;
                let delta = if sigma > beta { delta - 1 } else { delta };
                let mut kappa = midpoint(delta, tau, sigma, phi);
                while tau > 0 && kappa < phi {
                    total += count(delta, tau, sigma, kappa, phi);
                    tau -= 1;
                    phi = kappa;
                    kappa = midpoint(delta, tau, sigma, phi);
                }
                total
            }
            assert_eq!(count(delta, tau, 0, 0, 240), calls);
        }
    }
}
