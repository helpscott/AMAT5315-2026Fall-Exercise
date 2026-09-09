use super::{energy, force};

#[test]
fn well_depth() {
    let r0 = 2.0_f64.powf(1.0 / 6.0);
    assert!((energy(r0) + 1.0).abs() < 1e-12);
}

#[test]
fn force_matches_negative_energy_derivative() {
    let r0 = 2.0_f64.powf(1.0 / 6.0);
    let h = 1e-5;
    let separations = [0.9, 1.0, 1.1, r0, 1.2, 1.5, 2.0, 2.5, 3.0];

    for r in separations {
        let analytical = force(r);
        let numerical = -(energy(r + h) - energy(r - h)) / (2.0 * h);
        let tolerance = 1e-6 * analytical.abs().max(1.0);
        assert!(
            (analytical - numerical).abs() < tolerance,
            "r={r}: force={analytical}, negative derivative={numerical}, tolerance={tolerance}"
        );
    }
}
