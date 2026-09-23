//! Fourier method-of-lines tools for the Week 4 continuum-fluid exercise.

pub mod flow;
pub mod integrator;
pub mod line;
pub mod spectral;

pub use flow::{FieldData, FlowSolver};
pub use integrator::{Euler, Integrator, Midpoint, Rk4};
pub use num_complex::Complex64;
