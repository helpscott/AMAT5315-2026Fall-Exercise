pub mod enzyme;
pub mod model;
pub mod solver;

#[cfg(test)]
mod tests {
    use super::enzyme;

    #[test]
    fn enzyme_cube_smoke_test() {
        let values = enzyme::cube_check(2.0);
        assert_eq!(values, [8.0, 12.0, 12.0]);
    }

    #[test]
    fn local_step_jvp_and_vjp_are_transposes() {
        let (nx, nz) = (5, 5);
        let n = nx * nz;
        let previous: Vec<f64> = (0..n).map(|i| (i as f64 * 0.13).sin()).collect();
        let current: Vec<f64> = (0..n).map(|i| (i as f64 * 0.07).cos()).collect();
        let velocity = vec![1.8; n];
        let damping = vec![0.1; n];
        let source = vec![0.0; n];
        let d_previous: Vec<f64> = (0..n).map(|i| (i as f64 * 0.17).cos()).collect();
        let d_current: Vec<f64> = (0..n).map(|i| (i as f64 * 0.11).sin()).collect();
        let d_velocity: Vec<f64> = (0..n).map(|i| 0.01 * (i as f64 * 0.19).cos()).collect();
        let (_, tangent) = enzyme::jvp(
            &previous, &d_previous, &current, &d_current, &velocity, &d_velocity,
            &damping, &source, nx, nz, 1.0, 0.2,
        ).unwrap();
        let weight: Vec<f64> = (0..n).map(|i| (i as f64 * 0.23).sin()).collect();
        let (a, b, c) = enzyme::vjp(
            &previous, &current, &velocity, &damping, &source, &weight,
            nx, nz, 1.0, 0.2,
        ).unwrap();
        let left: f64 = tangent.iter().zip(&weight).map(|(x, y)| x * y).sum();
        let right: f64 = a.iter().zip(&d_previous).map(|(x, y)| x * y).sum::<f64>()
            + b.iter().zip(&d_current).map(|(x, y)| x * y).sum::<f64>()
            + c.iter().zip(&d_velocity).map(|(x, y)| x * y).sum::<f64>();
        assert!((left - right).abs() < 1e-11, "left={left}, right={right}");
    }
}

