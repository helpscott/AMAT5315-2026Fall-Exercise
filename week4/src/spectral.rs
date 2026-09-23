use num_complex::Complex64;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

pub fn wave_number(index: usize, n: usize) -> i32 {
    if index < (n + 1) / 2 {
        index as i32
    } else {
        index as i32 - n as i32
    }
}

pub fn fft_1d(values: &[Complex64], inverse: bool) -> Vec<Complex64> {
    let mut planner = FftPlanner::<f64>::new();
    let fft = if inverse {
        planner.plan_fft_inverse(values.len())
    } else {
        planner.plan_fft_forward(values.len())
    };
    let mut result = values.to_vec();
    fft.process(&mut result);
    result
}

pub struct Spectral2D {
    pub n: usize,
    forward: Arc<dyn Fft<f64>>,
    inverse: Arc<dyn Fft<f64>>,
}

impl Spectral2D {
    pub fn new(n: usize) -> Self {
        let mut planner = FftPlanner::<f64>::new();
        Self {
            n,
            forward: planner.plan_fft_forward(n),
            inverse: planner.plan_fft_inverse(n),
        }
    }

    fn transform(&self, values: &mut [Complex64], inverse: bool) {
        let plan = if inverse {
            &self.inverse
        } else {
            &self.forward
        };
        for row in values.chunks_exact_mut(self.n) {
            plan.process(row);
        }
        let mut column = vec![Complex64::default(); self.n];
        for x in 0..self.n {
            for y in 0..self.n {
                column[y] = values[y * self.n + x];
            }
            plan.process(&mut column);
            for y in 0..self.n {
                values[y * self.n + x] = column[y];
            }
        }
    }

    /// Normalized Fourier coefficients: inverse(forward(field)) == field.
    pub fn forward_real(&self, field: &[f64]) -> Vec<Complex64> {
        assert_eq!(field.len(), self.n * self.n);
        let mut values: Vec<_> = field
            .iter()
            .map(|&value| Complex64::new(value, 0.0))
            .collect();
        self.transform(&mut values, false);
        let scale = (self.n * self.n) as f64;
        for value in &mut values {
            *value /= scale;
        }
        values
    }

    pub fn inverse_real(&self, coefficients: &[Complex64]) -> Vec<f64> {
        assert_eq!(coefficients.len(), self.n * self.n);
        let mut values = coefficients.to_vec();
        self.transform(&mut values, true);
        values.into_iter().map(|value| value.re).collect()
    }

    pub fn derivative(&self, coefficients: &[Complex64], x_order: u32, y_order: u32) -> Vec<f64> {
        let mut differentiated = coefficients.to_vec();
        for y in 0..self.n {
            let ky = wave_number(y, self.n) as f64;
            for x in 0..self.n {
                let kx = wave_number(x, self.n) as f64;
                let factor =
                    Complex64::new(0.0, kx).powu(x_order) * Complex64::new(0.0, ky).powu(y_order);
                differentiated[y * self.n + x] *= factor;
            }
        }
        self.inverse_real(&differentiated)
    }

    pub fn laplacian(&self, coefficients: &[Complex64]) -> Vec<f64> {
        let mut result = coefficients.to_vec();
        for y in 0..self.n {
            let ky = wave_number(y, self.n) as f64;
            for x in 0..self.n {
                let kx = wave_number(x, self.n) as f64;
                result[y * self.n + x] *= -(kx * kx + ky * ky);
            }
        }
        self.inverse_real(&result)
    }
}
