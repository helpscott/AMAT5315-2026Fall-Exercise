use crate::spectral::{wave_number, Spectral2D};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FieldData {
    pub case: String,
    pub n: usize,
    pub seed: Option<u64>,
    pub k_band: Option<[i32; 2]>,
    pub u: Vec<f64>,
    pub v: Vec<f64>,
}

pub struct FlowSolver {
    pub n: usize,
    pub nu: f64,
    pub cutoff: i32,
    spectral: Spectral2D,
}

impl FlowSolver {
    pub fn new(n: usize, nu: f64) -> Self {
        assert!(n >= 4 && n % 2 == 0);
        Self {
            n,
            nu,
            cutoff: (n / 3) as i32,
            spectral: Spectral2D::new(n),
        }
    }

    pub fn mask(&self, coefficients: &mut [Complex64]) {
        for y in 0..self.n {
            let ky = wave_number(y, self.n);
            for x in 0..self.n {
                let kx = wave_number(x, self.n);
                if kx.abs() > self.cutoff || ky.abs() > self.cutoff {
                    coefficients[y * self.n + x] = Complex64::default();
                }
            }
        }
    }

    pub fn vorticity_from_velocity(&self, u: &[f64], v: &[f64]) -> Vec<Complex64> {
        let u_hat = self.spectral.forward_real(u);
        let v_hat = self.spectral.forward_real(v);
        let mut omega_hat = vec![Complex64::default(); self.n * self.n];
        for y in 0..self.n {
            let ky = wave_number(y, self.n) as f64;
            for x in 0..self.n {
                let kx = wave_number(x, self.n) as f64;
                let index = y * self.n + x;
                omega_hat[index] =
                    Complex64::new(0.0, kx) * v_hat[index] - Complex64::new(0.0, ky) * u_hat[index];
            }
        }
        self.mask(&mut omega_hat);
        omega_hat
    }

    pub fn fields_from_vorticity(&self, omega_hat: &[Complex64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut u_hat = vec![Complex64::default(); omega_hat.len()];
        let mut v_hat = vec![Complex64::default(); omega_hat.len()];
        for y in 0..self.n {
            let ky = wave_number(y, self.n) as f64;
            for x in 0..self.n {
                let kx = wave_number(x, self.n) as f64;
                let index = y * self.n + x;
                let k_squared = kx * kx + ky * ky;
                if k_squared > 0.0 {
                    let psi_hat = omega_hat[index] / k_squared;
                    u_hat[index] = Complex64::new(0.0, ky) * psi_hat;
                    v_hat[index] = -Complex64::new(0.0, kx) * psi_hat;
                }
            }
        }
        (
            self.spectral.inverse_real(&u_hat),
            self.spectral.inverse_real(&v_hat),
            self.spectral.inverse_real(omega_hat),
        )
    }

    pub fn rate(&self, omega_hat: &[Complex64], output: &mut [Complex64]) {
        let mut psi_hat = vec![Complex64::default(); omega_hat.len()];
        let mut u_hat = vec![Complex64::default(); omega_hat.len()];
        let mut v_hat = vec![Complex64::default(); omega_hat.len()];
        let mut dx_hat = vec![Complex64::default(); omega_hat.len()];
        let mut dy_hat = vec![Complex64::default(); omega_hat.len()];
        for y in 0..self.n {
            let ky = wave_number(y, self.n) as f64;
            for x in 0..self.n {
                let kx = wave_number(x, self.n) as f64;
                let index = y * self.n + x;
                let k_squared = kx * kx + ky * ky;
                if k_squared > 0.0 {
                    psi_hat[index] = omega_hat[index] / k_squared;
                    u_hat[index] = Complex64::new(0.0, ky) * psi_hat[index];
                    v_hat[index] = -Complex64::new(0.0, kx) * psi_hat[index];
                }
                dx_hat[index] = Complex64::new(0.0, kx) * omega_hat[index];
                dy_hat[index] = Complex64::new(0.0, ky) * omega_hat[index];
            }
        }
        let u = self.spectral.inverse_real(&u_hat);
        let v = self.spectral.inverse_real(&v_hat);
        let dx = self.spectral.inverse_real(&dx_hat);
        let dy = self.spectral.inverse_real(&dy_hat);
        let product: Vec<_> = u
            .iter()
            .zip(v.iter())
            .zip(dx.iter().zip(dy.iter()))
            .map(|((&u_value, &v_value), (&dx_value, &dy_value))| {
                u_value * dx_value + v_value * dy_value
            })
            .collect();
        let mut product_hat = self.spectral.forward_real(&product);
        self.mask(&mut product_hat);
        for y in 0..self.n {
            let ky = wave_number(y, self.n) as f64;
            for x in 0..self.n {
                let kx = wave_number(x, self.n) as f64;
                let index = y * self.n + x;
                output[index] =
                    -product_hat[index] - self.nu * (kx * kx + ky * ky) * omega_hat[index];
            }
        }
        self.mask(output);
    }

    pub fn diagnostics(&self, omega_hat: &[Complex64]) -> (f64, f64) {
        let mut velocity_norm = 0.0;
        for y in 0..self.n {
            let ky = wave_number(y, self.n) as f64;
            for x in 0..self.n {
                let kx = wave_number(x, self.n) as f64;
                let index = y * self.n + x;
                let k_squared = kx * kx + ky * ky;
                if k_squared > 0.0 {
                    velocity_norm += omega_hat[index].norm_sqr() / k_squared;
                }
            }
        }
        let energy = 0.5 * velocity_norm;
        let enstrophy = 0.5 * omega_hat.iter().map(|value| value.norm_sqr()).sum::<f64>();
        (energy, enstrophy)
    }

    pub fn spectral(&self) -> &Spectral2D {
        &self.spectral
    }
}

pub fn taylor_green(n: usize, nu: f64, time: f64) -> FieldData {
    let decay = (-2.0 * nu * time).exp();
    let mut u = Vec::with_capacity(n * n);
    let mut v = Vec::with_capacity(n * n);
    for y_index in 0..n {
        let y = 2.0 * PI * y_index as f64 / n as f64;
        for x_index in 0..n {
            let x = 2.0 * PI * x_index as f64 / n as f64;
            u.push(x.cos() * y.sin() * decay);
            v.push(-x.sin() * y.cos() * decay);
        }
    }
    FieldData {
        case: "taylor-green".to_string(),
        n,
        seed: None,
        k_band: None,
        u,
        v,
    }
}

#[derive(Clone, Debug)]
struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9e3779b97f4a7c15 } else { seed },
        }
    }

    fn unit(&mut self) -> f64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        let value = x.wrapping_mul(0x2545f4914f6cdd1d);
        value as f64 / (u64::MAX as f64 + 1.0)
    }
}

fn mode_index(k: i32, n: usize) -> usize {
    if k >= 0 {
        k as usize
    } else {
        (n as i32 + k) as usize
    }
}

pub fn random_field(n: usize, seed: u64, k_min: i32, k_max: i32) -> FieldData {
    assert!(k_min > 0 && k_max >= k_min && k_max < n as i32 / 3);
    let solver = FlowSolver::new(n, 0.0);
    let mut omega_hat = vec![Complex64::default(); n * n];
    let mut rng = DeterministicRng::new(seed);
    for ky in -k_max..=k_max {
        for kx in -k_max..=k_max {
            let radius_squared = kx * kx + ky * ky;
            if radius_squared < k_min * k_min || radius_squared > k_max * k_max {
                continue;
            }
            if ky < 0 || (ky == 0 && kx <= 0) {
                continue;
            }
            let phase = 2.0 * PI * rng.unit();
            let value = Complex64::from_polar(1.0, phase);
            let positive = mode_index(ky, n) * n + mode_index(kx, n);
            let negative = mode_index(-ky, n) * n + mode_index(-kx, n);
            omega_hat[positive] = value;
            omega_hat[negative] = value.conj();
        }
    }
    let (u_unscaled, v_unscaled, _) = solver.fields_from_vorticity(&omega_hat);
    let energy = 0.5
        * u_unscaled
            .iter()
            .zip(v_unscaled.iter())
            .map(|(&u, &v)| u * u + v * v)
            .sum::<f64>()
        / (n * n) as f64;
    let scale = (0.5 / energy).sqrt();
    for value in &mut omega_hat {
        *value *= scale;
    }
    let (u, v, _) = solver.fields_from_vorticity(&omega_hat);
    FieldData {
        case: "random".to_string(),
        n,
        seed: Some(seed),
        k_band: Some([k_min, k_max]),
        u,
        v,
    }
}

pub fn relative_l2(left: &[f64], right: &[f64]) -> f64 {
    let numerator = left
        .iter()
        .zip(right)
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    let denominator = right.iter().map(|value| value * value).sum::<f64>().sqrt();
    numerator / denominator
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn taylor_green_has_exact_initial_diagnostics() {
        let field = taylor_green(32, 0.1, 0.0);
        let solver = FlowSolver::new(32, 0.1);
        let omega = solver.vorticity_from_velocity(&field.u, &field.v);
        let (energy, enstrophy) = solver.diagnostics(&omega);
        assert!((energy - 0.25).abs() < 1e-12);
        assert!((enstrophy - 0.5).abs() < 1e-12);
    }

    #[test]
    fn random_field_has_half_unit_energy() {
        let field = random_field(64, 2026, 2, 6);
        let solver = FlowSolver::new(64, 0.0);
        let omega = solver.vorticity_from_velocity(&field.u, &field.v);
        let (energy, _) = solver.diagnostics(&omega);
        assert!((energy - 0.5).abs() < 1e-12);
    }
}
