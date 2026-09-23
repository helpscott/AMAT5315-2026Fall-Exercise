use std::f64::consts::PI;
use week4_fluid::spectral::Spectral2D;

fn max_error(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0, f64::max)
}

fn centered(n: usize, field: &[f64], operation: &str) -> Vec<f64> {
    let dx = 2.0 * PI / n as f64;
    let at = |x: usize, y: usize| field[(y % n) * n + (x % n)];
    let mut output = vec![0.0; n * n];
    for y in 0..n {
        for x in 0..n {
            let left = (x + n - 1) % n;
            let right = (x + 1) % n;
            let down = (y + n - 1) % n;
            let up = (y + 1) % n;
            output[y * n + x] = match operation {
                "dx" => (at(right, y) - at(left, y)) / (2.0 * dx),
                "dxx" => (at(right, y) - 2.0 * at(x, y) + at(left, y)) / (dx * dx),
                "dxdy" => {
                    (at(right, up) - at(right, down) - at(left, up) + at(left, down))
                        / (4.0 * dx * dx)
                }
                "lap" => {
                    (at(right, y) + at(left, y) + at(x, up) + at(x, down) - 4.0 * at(x, y))
                        / (dx * dx)
                }
                _ => unreachable!(),
            };
        }
    }
    output
}

fn fields(n: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mut g = Vec::with_capacity(n * n);
    let mut exact = vec![Vec::with_capacity(n * n); 4];
    for y_index in 0..n {
        let y = 2.0 * PI * y_index as f64 / n as f64;
        for x_index in 0..n {
            let x = 2.0 * PI * x_index as f64 / n as f64;
            let value = (3.0 * x).sin() * (2.0 * y).cos();
            g.push(value);
            exact[0].push(3.0 * (3.0 * x).cos() * (2.0 * y).cos());
            exact[1].push(-9.0 * value);
            exact[2].push(-6.0 * (3.0 * x).cos() * (2.0 * y).sin());
            exact[3].push(-13.0 * value);
        }
    }
    (g, exact)
}

fn main() {
    let operations = ["dx", "dxx", "dxdy", "lap"];
    let labels = ["dx", "dxx", "dxdy", "laplacian"];
    let (g32, exact32) = fields(32);
    let (g64, exact64) = fields(64);
    let spectral = Spectral2D::new(32);
    let coefficients = spectral.forward_real(&g32);
    let fourier = [
        spectral.derivative(&coefficients, 1, 0),
        spectral.derivative(&coefficients, 2, 0),
        spectral.derivative(&coefficients, 1, 1),
        spectral.laplacian(&coefficients),
    ];
    println!("derivative\tfd_n32\tfd_n64\tratio\tfourier_n32");
    for index in 0..4 {
        let error32 = max_error(&centered(32, &g32, operations[index]), &exact32[index]);
        let error64 = max_error(&centered(64, &g64, operations[index]), &exact64[index]);
        let spectral_error = max_error(&fourier[index], &exact32[index]);
        println!(
            "{}\t{error32:.12e}\t{error64:.12e}\t{:.6}\t{spectral_error:.12e}",
            labels[index],
            error32 / error64
        );
    }
}
