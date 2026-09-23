use crate::integrator::{equal_weight_rk4_step, Integrator};
use crate::spectral::{fft_1d, wave_number};
use num_complex::Complex64;
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug)]
pub enum LineDerivative {
    Fourier,
    Centered,
}

pub fn periodic_grid(n: usize) -> Vec<f64> {
    (0..n)
        .map(|index| 2.0 * PI * index as f64 / n as f64)
        .collect()
}

pub fn periodic_gaussian(n: usize, center: f64, sigma: f64) -> Vec<Complex64> {
    periodic_grid(n)
        .into_iter()
        .map(|x| {
            let value = (-3..=3)
                .map(|image| {
                    let distance = x - center + image as f64 * 2.0 * PI;
                    (-0.5 * (distance / sigma).powi(2)).exp()
                })
                .sum();
            Complex64::new(value, 0.0)
        })
        .collect()
}

pub fn fourier_rate(state: &[Complex64], c: f64, nu: f64, output: &mut [Complex64]) {
    let n = state.len();
    let mut coefficients = fft_1d(state, false);
    for (index, value) in coefficients.iter_mut().enumerate() {
        let k = wave_number(index, n) as f64;
        let advective_k = if index == n / 2 { 0.0 } else { k };
        *value *= Complex64::new(-nu * k * k, -c * advective_k);
    }
    let physical = fft_1d(&coefficients, true);
    for (target, value) in output.iter_mut().zip(physical) {
        *target = value / n as f64;
    }
}

pub fn centered_rate(state: &[Complex64], c: f64, nu: f64, output: &mut [Complex64]) {
    let n = state.len();
    let dx = 2.0 * PI / n as f64;
    for index in 0..n {
        let left = state[(index + n - 1) % n];
        let right = state[(index + 1) % n];
        let first = (right - left) / (2.0 * dx);
        let second = (right - 2.0 * state[index] + left) / (dx * dx);
        output[index] = -c * first + nu * second;
    }
}

pub fn exact_fourier(initial: &[Complex64], c: f64, nu: f64, time: f64) -> Vec<Complex64> {
    let n = initial.len();
    let mut coefficients = fft_1d(initial, false);
    for (index, value) in coefficients.iter_mut().enumerate() {
        let k = wave_number(index, n) as f64;
        let advective_k = if index == n / 2 { 0.0 } else { k };
        *value *= (Complex64::new(-nu * k * k, -c * advective_k) * time).exp();
    }
    fft_1d(&coefficients, true)
        .into_iter()
        .map(|value| value / n as f64)
        .collect()
}

pub fn integrate(
    initial: &[Complex64],
    c: f64,
    nu: f64,
    dt: f64,
    end_time: f64,
    derivative: LineDerivative,
    integrator: &dyn Integrator,
) -> Vec<Complex64> {
    integrate_impl(
        initial,
        c,
        nu,
        dt,
        end_time,
        derivative,
        Some(integrator),
        false,
    )
}

pub fn integrate_equal_weight_rk4(
    initial: &[Complex64],
    c: f64,
    nu: f64,
    dt: f64,
    end_time: f64,
    derivative: LineDerivative,
) -> Vec<Complex64> {
    integrate_impl(initial, c, nu, dt, end_time, derivative, None, true)
}

fn integrate_impl(
    initial: &[Complex64],
    c: f64,
    nu: f64,
    dt: f64,
    end_time: f64,
    derivative: LineDerivative,
    integrator: Option<&dyn Integrator>,
    equal_weights: bool,
) -> Vec<Complex64> {
    let mut state = initial.to_vec();
    let mut time = 0.0;
    while time < end_time - 1e-14 {
        let step = dt.min(end_time - time);
        let mut rate =
            |_stage_time: f64, values: &[Complex64], output: &mut [Complex64]| match derivative {
                LineDerivative::Fourier => fourier_rate(values, c, nu, output),
                LineDerivative::Centered => centered_rate(values, c, nu, output),
            };
        state = if equal_weights {
            equal_weight_rk4_step(time, &state, step, &mut rate)
        } else {
            integrator.unwrap().step(time, &state, step, &mut rate)
        };
        time += step;
    }
    state
}

pub fn max_error(left: &[Complex64], right: &[Complex64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(&a, &b)| (a - b).norm())
        .fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrator::{Euler, Midpoint, Rk4};

    #[test]
    fn nyquist_mode_does_not_advect() {
        let n = 16;
        let state: Vec<_> = (0..n)
            .map(|index| Complex64::new(if index % 2 == 0 { 1.0 } else { -1.0 }, 0.0))
            .collect();
        let mut output = vec![Complex64::default(); n];
        fourier_rate(&state, 3.0, 0.0, &mut output);
        assert!(output.iter().all(|value| value.norm() < 1e-11));
    }

    #[test]
    fn every_integrator_advances_one_exact_wave() {
        let n = 32;
        let k = 3.0;
        let initial: Vec<_> = periodic_grid(n)
            .into_iter()
            .map(|x| Complex64::new((k * x).sin(), 0.0))
            .collect();
        let exact = exact_fourier(&initial, 1.0, 0.05, 0.01);
        let methods: [(&dyn Integrator, f64); 3] =
            [(&Euler, 2e-3), (&Midpoint, 2e-5), (&Rk4, 2e-9)];
        for (method, tolerance) in methods {
            let numerical = integrate(
                &initial,
                1.0,
                0.05,
                0.01,
                0.01,
                LineDerivative::Fourier,
                method,
            );
            assert!(
                max_error(&numerical, &exact) < tolerance,
                "{}",
                method.name()
            );
        }
    }
}
