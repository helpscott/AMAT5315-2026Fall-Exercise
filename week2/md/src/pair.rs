/// Lennard-Jones pair energy in reduced units, for separation r > 0.
pub fn energy(r: f64) -> f64 {
    let inv_r2 = 1.0 / (r * r);
    let inv_r6 = inv_r2 * inv_r2 * inv_r2;
    4.0 * (inv_r6 * inv_r6 - inv_r6)
}

/// Radial pair force: positive means repulsion, negative means attraction.
/// The separation r must be positive.
pub fn force(r: f64) -> f64 {
    let inv_r = 1.0 / r;
    let inv_r2 = inv_r * inv_r;
    let inv_r6 = inv_r2 * inv_r2 * inv_r2;
    24.0 * inv_r * (2.0 * inv_r6 * inv_r6 - inv_r6)
}

#[cfg(test)]
mod tests;
