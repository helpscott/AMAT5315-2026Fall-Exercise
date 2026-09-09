/// Lennard-Jones pair energy in reduced units, for separation r > 0.
pub fn energy(r: f64) -> f64 {
    let inv_r2 = 1.0 / (r * r);
    let inv_r6 = inv_r2 * inv_r2 * inv_r2;
    4.0 * (inv_r6 * inv_r6 - inv_r6)
}

/// Radial pair force: positive means repulsion, negative means attraction.
/// The separation r must be positive.
pub fn force(_r: f64) -> f64 {
    todo!("implement Lennard-Jones force")
}

#[cfg(test)]
mod tests;
