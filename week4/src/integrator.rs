use num_complex::Complex64;

pub type RateFunction<'a> = dyn FnMut(f64, &[Complex64], &mut [Complex64]) + 'a;

/// One interface for explicit time integrators over a complex state vector.
pub trait Integrator {
    fn name(&self) -> &'static str;
    fn step(
        &self,
        time: f64,
        state: &[Complex64],
        dt: f64,
        rate: &mut RateFunction<'_>,
    ) -> Vec<Complex64>;
}

fn shifted(state: &[Complex64], rate: &[Complex64], scale: f64) -> Vec<Complex64> {
    state
        .iter()
        .zip(rate)
        .map(|(&value, &slope)| value + slope * scale)
        .collect()
}

#[derive(Clone, Copy, Debug)]
pub struct Euler;

impl Integrator for Euler {
    fn name(&self) -> &'static str {
        "euler"
    }

    fn step(
        &self,
        time: f64,
        state: &[Complex64],
        dt: f64,
        rate: &mut RateFunction<'_>,
    ) -> Vec<Complex64> {
        let mut k1 = vec![Complex64::default(); state.len()];
        rate(time, state, &mut k1);
        shifted(state, &k1, dt)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Midpoint;

impl Integrator for Midpoint {
    fn name(&self) -> &'static str {
        "rk2"
    }

    fn step(
        &self,
        time: f64,
        state: &[Complex64],
        dt: f64,
        rate: &mut RateFunction<'_>,
    ) -> Vec<Complex64> {
        let mut k1 = vec![Complex64::default(); state.len()];
        let mut k2 = vec![Complex64::default(); state.len()];
        rate(time, state, &mut k1);
        let trial = shifted(state, &k1, 0.5 * dt);
        rate(time + 0.5 * dt, &trial, &mut k2);
        shifted(state, &k2, dt)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Rk4;

impl Integrator for Rk4 {
    fn name(&self) -> &'static str {
        "rk4"
    }

    fn step(
        &self,
        time: f64,
        state: &[Complex64],
        dt: f64,
        rate: &mut RateFunction<'_>,
    ) -> Vec<Complex64> {
        let mut k1 = vec![Complex64::default(); state.len()];
        let mut k2 = vec![Complex64::default(); state.len()];
        let mut k3 = vec![Complex64::default(); state.len()];
        let mut k4 = vec![Complex64::default(); state.len()];
        rate(time, state, &mut k1);
        rate(time + 0.5 * dt, &shifted(state, &k1, 0.5 * dt), &mut k2);
        rate(time + 0.5 * dt, &shifted(state, &k2, 0.5 * dt), &mut k3);
        rate(time + dt, &shifted(state, &k3, dt), &mut k4);
        state
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                value + (k1[index] + 2.0 * k2[index] + 2.0 * k3[index] + k4[index]) * (dt / 6.0)
            })
            .collect()
    }
}

/// Deliberately wrong RK4 weights used by the order-diagnostic plot.
pub fn equal_weight_rk4_step(
    time: f64,
    state: &[Complex64],
    dt: f64,
    rate: &mut RateFunction<'_>,
) -> Vec<Complex64> {
    let mut k1 = vec![Complex64::default(); state.len()];
    let mut k2 = vec![Complex64::default(); state.len()];
    let mut k3 = vec![Complex64::default(); state.len()];
    let mut k4 = vec![Complex64::default(); state.len()];
    rate(time, state, &mut k1);
    rate(time + 0.5 * dt, &shifted(state, &k1, 0.5 * dt), &mut k2);
    rate(time + 0.5 * dt, &shifted(state, &k2, 0.5 * dt), &mut k3);
    rate(time + dt, &shifted(state, &k3, dt), &mut k4);
    state
        .iter()
        .enumerate()
        .map(|(index, &value)| value + (k1[index] + k2[index] + k3[index] + k4[index]) * (dt / 4.0))
        .collect()
}
