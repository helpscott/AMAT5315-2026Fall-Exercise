/// Lennard-Jones pair energy in reduced units, for separation r > 0.
pub fn energy(_r: f64) -> f64 {
    todo!("implement Lennard-Jones energy")
}

/// Radial pair force: positive means repulsion, negative means attraction.
/// The separation r must be positive.
pub fn force(_r: f64) -> f64 {
    todo!("implement Lennard-Jones force")
}

#[cfg(test)]
mod tests;
