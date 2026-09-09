# Two-atom molecular dynamics design

Status: proposed for review. This document specifies Part 3; implementation follows
approval of this design and the subsequent implementation plan.

## Objective and existing code

Extend the existing `week2/md` Rust crate so that the same two-atom experiment can
run with forward Euler or velocity-Verlet through one trait. Measure total-energy
error at every step and generate the two-panel `week2/dimer.png` figure.

The existing greeting, pair-energy function, analytical pair-force function,
three unit tests, field example, and field plot remain part of the project.
Reuse `md::pair::{energy, force}` for all interactions. The Python plotting
environment is already available at `week2/.venv`.

## Approaches considered

| Approach | Benefit | Trade-off |
| --- | --- | --- |
| A. Vector-backed System and a generic Integrator driver (recommended) | Matches the teaching interface, keeps state ownership explicit, and accommodates later atom counts | Requires small, separate state and integrator modules |
| B. Fixed-size two-atom state with the same trait | Smallest representation for this experiment | The state representation must change for the many-atom stage |
| C. Vector-backed System with runtime trait objects | Allows selecting an integrator as a runtime object | Adds dynamic dispatch where the current experiment can select concrete types directly |

Approach A uses `&impl Integrator`. Each integrator is a stateless rule; the
particle state owns the working arrays. The comparison changes only the
integrator value supplied to the shared experiment driver.

## Physical contract

- Two atoms in two dimensions, with masses equal to 1.
- Initial positions: `[[0.0, 0.0], [1.2, 0.0]]`.
- Initial velocities: `[[0.0, 0.0], [0.0, 0.0]]`.
- Plain Lennard-Jones potential in reduced units, with open boundaries.
- Time step: `dt = 0.01`.
- Short comparison: 500 steps for each integrator, ending at time 5.
- Long comparison: 5000 steps for velocity-Verlet, ending at time 50.
- Total energy: `E = 0.5 * sum(vx*vx + vy*vy) + sum(i<j, U(r_ij))`.
- Reference energy: the initial state before the first step, `E0 = U(1.2)`.
- Relative error: `(E - E0) / abs(E0)`.

For a pair, define `d = x_i - x_j`, `r = |d|`, and `F_i = F(r) * d/r`.
Add the same pair vector with opposite signs to the two accelerations, since
mass is 1. Visit every pair once. At the specified initial separation the
forces point inward.

## State, interfaces, and borrowing

`System` owns `Vec<[f64; 2]>` arrays for positions, velocities, and
accelerations, plus the potential energy at the current positions. The arrays
have crate-internal visibility for the integrators and immutable public accessors. Its
constructor accepts positions and velocities, validates their shape and
values, and computes the initial accelerations and potential energy.

Read-only accessors return borrowed slices. Measurements take `&System` and
read the stored state without cloning the arrays. The update driver takes
`&mut System` so the integrator has exclusive permission to change the state.
The cached accelerations and potential energy always correspond to the
current positions after a successful construction or step. External callers
cannot directly mutate the positions and invalidate this cache.

Proposed public interface:

```rust
pub struct System {
    pub(crate) positions: Vec<[f64; 2]>,
    pub(crate) velocities: Vec<[f64; 2]>,
    pub(crate) accelerations: Vec<[f64; 2]>,
    potential_energy: f64,
}
pub struct Euler;
pub struct VelocityVerlet;
pub enum SimulationError {
    InvalidState(&'static str),
    InvalidTimeStep(f64),
    OverlappingAtoms { i: usize, j: usize },
    NonFiniteCalculation,
}

pub trait Integrator {
    fn step(&self, system: &mut System, dt: f64) -> Result<(), SimulationError>;
}

pub fn advance(
    method: &impl Integrator,
    system: &mut System,
    dt: f64,
) -> Result<(), SimulationError>;
```

The declarations above describe the interface, not implementation scaffolding.
`System::new(positions, velocities)` returns `Result<System, SimulationError>`.
`positions()`, `velocities()`, and `accelerations()` expose immutable slices;
`kinetic_energy()`, `potential_energy()`, and `total_energy()` return `f64`.
`System::refresh_forces()` is crate-internal and refreshes accelerations and
potential energy together, returning `Result<(), SimulationError>`.
`SimulationError` reports invalid input, overlapping atoms, or a non-finite
calculation through `Display` and the standard error trait.

## Numerical updates and the force cache

Forward Euler updates every position using the old velocity, and every
velocity using the old acceleration. It then refreshes the accelerations
and potential energy at the new positions for the next step and measurement.
Using an already-updated velocity for its position update would be a
different method and must be caught by a one-step test.

Velocity-Verlet performs a half kick with the old acceleration, a drift with
the half-step velocity, one force-and-potential refresh at the new positions,
and a second half kick with the new acceleration. The resulting acceleration
cache is reused by the next step. The constructor supplies the force
calculation before the first step, so later Verlet steps require exactly
one new pair-force pass.

Accelerations are reset and reused in place when forces are refreshed.
Both algorithms use the same force-and-potential calculation. Energy
sampling adds the instantaneous kinetic energy to the cached potential
energy, avoiding an extra pair search solely to record energy.

## Experiment output

A shared `run_dimer(method: &impl Integrator, dt: f64, steps: usize)` creates
the exact initial state above, calls `advance` for each step, and returns
`Result<Vec<EnergySample>, SimulationError>`.

Each `EnergySample` contains `step: usize`, `time: f64`, `total_energy: f64`,
and `relative_error: f64`. Include the initial sample at step 0, followed by
one sample per completed step. The short runs therefore have 501 samples,
and the long run has 5001. A zero-step call returns only the initial sample.

The `dimer` example emits CSV with a run label and these sample fields for
Euler-500, Verlet-500, and Verlet-5000. The Python plotter launches this
example and renders its output using the existing plotting environment.
It computes no independent trajectory or integration steps.

The left plot shows both short-run errors against time. The right plot
shows the long Verlet run with `1000 * relative_error` on a clearly labeled
y-axis. The plot, tests, and reported measurements use the same Rust driver.

## Errors and invariants

Reject empty particle arrays, unequal position/velocity counts, non-finite
input coordinates or velocities, coincident atoms, and non-positive or
non-finite time steps. Detect non-finite calculations during integration
and energy sampling. Propagate an error and stop the run before emitting a
failed frame. A failed step does not promise to restore the old state;
callers stop and discard that run.

The dimer driver uses the fixed nonzero reference energy `U(1.2)`. Its
relative-error calculation therefore has a well-defined denominator.

## Tests and acceptance criteria

The first new test is the shared-driver dimer comparison, written and run
before implementation. Its failure (unimplemented API or missing symbols)
is committed as the red checkpoint. Earlier tests retain their original
assertions and remain available.

Required checks:

1. Both 500-step runs use `run_dimer` with identical `dt`, step count, and
   initial state. The only changed argument is the integrator value.
2. Velocity-Verlet's maximum absolute relative error over the 500-step run
   is less than `1e-3`; Euler's final signed relative error exceeds `0.5`.
3. The 5000-step Verlet run remains finite and has maximum absolute relative
   error below `1e-3`. This makes the required long-run boundedness check
   concrete for the specified interval; it is not a claim about infinite time.
4. Pair accelerations point inward from the specified initial state, sum
   to zero with each Cartesian component smaller than `1e-12` in absolute
   value, and remain consistent with
   the force calculation used by both integrators.
5. A first Euler step from rest leaves positions unchanged while changing
   velocities, catching accidental use of the new velocity in the drift.
6. The system rejects the invalid inputs named above. Samples have the
   expected count, time values, finite energies, and an initial zero error.
7. Existing greeting and pair-function tests still pass. Source inspection
   confirms one new force refresh per Verlet step and a shared trait driver.
8. `week2/dimer.png` is regenerated from the example, with both short-run
   curves, the separate long-run panel, and an explicit factor of 1000.

The time-reversal experiment in the sheet is an optional extension and is
not part of these baseline acceptance criteria.

## File responsibilities

| File | Responsibility |
| --- | --- |
| `week2/md/src/system.rs` | Owned state, input validation, shared open-boundary pair accumulation, energy measurements, simulation errors |
| `week2/md/src/integrator.rs` | Integrator trait, shared advance function, Euler and velocity-Verlet |
| `week2/md/src/experiment.rs` | Exact dimer initialization and per-step energy samples |
| `week2/md/src/lib.rs` | Expose the new library modules alongside the existing pair module and greeting |
| `week2/md/tests/dimer.rs` | Shared-driver conservation comparison and long-run check |
| Unit tests beside the new modules | One-step update behavior and state/input invariants |
| `week2/md/examples/dimer.rs` | Export the three experiment runs as CSV |
| `week2/plot_dimer.py` | Render CSV into the two-panel plot |
| `week2/dimer.png` and `week2/README.md` | Inspectable evidence, measured errors, and reproduction commands |

## Review and execution sequence

Review and approve this design, then write a separate implementation plan
under `docs/superpowers/plans/` with the dimer test before the implementation.
Review that plan before executing it. Preserve a committed red checkpoint
and subsequent green commits. Review the completed code against both
documents, rerun relevant checks after fixes, and give the learner the
independent verification commands.

## Sources

- `week2-learning-sheet.pdf`, Part 3, pages 6-8: force convention, shared
  trait, initial state, update rules, energy bounds, plot, and verification.
- Existing repository source and the independently verified Parts 1-2.
- Superpowers skills at revision
  `b36e0829c6d0140e93cfef2ca599b1b07d4a7797`, recorded in
  `.agents/skills/SUPERPOWERS.md`.
