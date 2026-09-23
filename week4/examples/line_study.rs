use num_complex::Complex64;
use serde_json::json;
use std::f64::consts::PI;
use week4_fluid::integrator::{Euler, Integrator, Midpoint, Rk4};
use week4_fluid::line::{
    exact_fourier, integrate, integrate_equal_weight_rk4, max_error, periodic_gaussian,
    LineDerivative,
};

fn real(values: &[Complex64]) -> Vec<f64> {
    values.iter().map(|value| value.re).collect()
}

fn pulse_history(dt: f64) -> (Vec<f64>, Vec<Vec<f64>>) {
    let initial = periodic_gaussian(64, PI / 2.0, 0.35);
    let method = Rk4;
    let mut state = initial;
    let mut times = vec![0.0];
    let mut frames = vec![real(&state)];
    let total_steps = (6.0 / dt).ceil() as usize;
    for step in 1..=total_steps {
        let time = (step - 1) as f64 * dt;
        let mut rate = |_time: f64, values: &[Complex64], output: &mut [Complex64]| {
            week4_fluid::line::fourier_rate(values, 1.0, 0.05, output)
        };
        state = method.step(time, &state, dt, &mut rate);
        times.push(step as f64 * dt);
        frames.push(real(&state));
    }
    (times, frames)
}

fn main() {
    let real_axis: Vec<f64> = (0..=240).map(|i| -4.0 + 6.0 * i as f64 / 240.0).collect();
    let imag_axis: Vec<f64> = (0..=240).map(|i| -4.0 + 8.0 * i as f64 / 240.0).collect();
    let method = Rk4;
    let mut growth = Vec::with_capacity(real_axis.len() * imag_axis.len());
    for &imaginary in &imag_axis {
        for &real in &real_axis {
            let lambda = Complex64::new(real, imaginary);
            let mut rate = |_time: f64, values: &[Complex64], output: &mut [Complex64]| {
                output[0] = lambda * values[0];
            };
            let advanced = method.step(0.0, &[Complex64::new(1.0, 0.0)], 1.0, &mut rate);
            growth.push(advanced[0].norm());
        }
    }

    let line_modes = |dt: f64| -> Vec<[f64; 2]> {
        (-32..=31)
            .map(|k| {
                let advective_k = if k == -32 { 0.0 } else { k as f64 };
                [-0.05 * (k * k) as f64 * dt, -advective_k * dt]
            })
            .collect()
    };
    let (stable_times, stable_frames) = pulse_history(0.045);
    let (unstable_times, unstable_frames) = pulse_history(0.056);

    let profile_initial = periodic_gaussian(64, PI / 2.0, 0.25);
    let profile_exact = exact_fourier(&profile_initial, 1.0, 0.002, 2.0 * PI);
    let profile_fourier = integrate(
        &profile_initial,
        1.0,
        0.002,
        0.02,
        2.0 * PI,
        LineDerivative::Fourier,
        &Rk4,
    );
    let profile_centered = integrate(
        &profile_initial,
        1.0,
        0.002,
        0.02,
        2.0 * PI,
        LineDerivative::Centered,
        &Rk4,
    );
    let profile_euler = integrate(
        &profile_initial,
        1.0,
        0.002,
        0.005,
        2.0 * PI,
        LineDerivative::Fourier,
        &Euler,
    );

    let order_initial = periodic_gaussian(64, PI / 2.0, 0.35);
    let order_exact = exact_fourier(&order_initial, 1.0, 0.05, 1.0);
    let steps = [0.02, 0.01, 0.005, 0.0025];
    let method_errors = |integrator: &dyn Integrator| -> Vec<f64> {
        steps
            .iter()
            .map(|&dt| {
                let numerical = integrate(
                    &order_initial,
                    1.0,
                    0.05,
                    dt,
                    1.0,
                    LineDerivative::Fourier,
                    integrator,
                );
                max_error(&numerical, &order_exact)
            })
            .collect()
    };
    let equal_errors: Vec<f64> = steps
        .iter()
        .map(|&dt| {
            max_error(
                &integrate_equal_weight_rk4(
                    &order_initial,
                    1.0,
                    0.05,
                    dt,
                    1.0,
                    LineDerivative::Fourier,
                ),
                &order_exact,
            )
        })
        .collect();

    let output = json!({
        "stability": {
            "real": real_axis,
            "imag": imag_axis,
            "growth": growth,
            "modes_0045": line_modes(0.045),
            "modes_0056": line_modes(0.056),
            "stable_times": stable_times,
            "stable_frames": stable_frames,
            "unstable_times": unstable_times,
            "unstable_frames": unstable_frames,
        },
        "accuracy": {
            "x": (0..64).map(|i| 2.0 * PI * i as f64 / 64.0).collect::<Vec<_>>(),
            "exact": real(&profile_exact),
            "rk4_fourier": real(&profile_fourier),
            "rk4_centered": real(&profile_centered),
            "euler_fourier": real(&profile_euler),
            "profile_errors": {
                "rk4_fourier": max_error(&profile_fourier, &profile_exact),
                "rk4_centered": max_error(&profile_centered, &profile_exact),
                "euler_fourier": max_error(&profile_euler, &profile_exact),
            },
            "steps": steps,
            "euler_errors": method_errors(&Euler),
            "midpoint_errors": method_errors(&Midpoint),
            "rk4_errors": method_errors(&Rk4),
            "equal_weight_errors": equal_errors,
        }
    });
    println!("{}", serde_json::to_string(&output).unwrap());
}
