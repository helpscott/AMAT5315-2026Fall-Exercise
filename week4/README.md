# Week 4 - Continuum fluid dynamics

This crate advances one-dimensional advection-diffusion and the two-dimensional
incompressible vorticity equation on periodic domains. Forward Euler, explicit
midpoint (`rk2`), and classical fourth-order Runge-Kutta (`rk4`) share one
`Integrator` trait. Spatial derivatives use Fourier multipliers; the line study
also implements second-order centred differences. The fluid solver uses the
two-thirds dealiasing rule on both the carried vorticity and every nonlinear
product.

`field.design.toml` and `fluid.design.toml` are the command contracts.

## Install and test

Run from a clean clone with Rust and Python 3.10 or newer:

```bash
cd week4
cargo test
cargo install --path . --quiet
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-analysis.txt
```

The tests cover the Nyquist convention, every integrator on an exact Fourier
wave, exact Taylor-Green diagnostics, random-field normalization, and both CLI
contracts.

## Contract pipelines

The exact Taylor-Green verification pipeline is:

```bash
mkdir -p evidence artifacts
field taylor-green --n 64 | fluid --method rk4 --nu 0.1 --dt 0.01 --t-end 1 --every 0.1 --out artifacts/taylor-green > artifacts/taylor-green.tsv
field taylor-green --n 64 --nu 0.1 --t 1 > artifacts/taylor-green/exact-t1.json
```

The seeded nonlinear flow pipeline is:

```bash
field random --n 128 --seed 2026 --k-min 2 --k-max 6 | fluid --method rk4 --nu 0.004 --dt 0.01 --t-end 10 --every 0.1 --out artifacts/random > artifacts/random.tsv
```

`fields.jsonl` stores six-decimal `u`, `v`, and `omega` arrays. `run.json`
records the initial-field metadata and every integration setting. The large
`artifacts/` tree remains local and ignored by Git.

## Regenerate every committed evidence file

Run these commands from `week4` in order. Each command is followed by every
committed evidence file it produces.

```bash
python3 scripts/derivatives.py
# Prints the full-precision Fourier and finite-difference derivative table.

python3 scripts/line.py
# evidence/line-stability.png
# evidence/line-accuracy.png

python3 scripts/run_core.py
# Generates artifacts/taylor-green, artifacts/random, and artifacts/unstable.

python3 scripts/taylor_green.py
# evidence/taylor-green.png

python3 scripts/random_flow.py
# evidence/random.png

python3 scripts/run_stability.py
# Generates artifacts/scan and artifacts/sensitivity.

python3 scripts/blowup.py
# evidence/blowup.png

python3 scripts/sensitivity.py
# evidence/sensitivity.png

python3 scripts/order.py
# evidence/order.png

python3 scripts/convergence.py
# evidence/convergence.json
# evidence/convergence.png
```

The line script calls the Rust library example, so its measured stability map,
pulse integrations, and time-order data use the same integrators as `fluid`.
The stability runner begins with the prescribed random-flow steps `0.038` and
`0.040`, then lowers or raises them by `0.002` until it has one run that reaches
`t=10` and one that becomes non-finite. This is required because the seeded
phase generator may give a different largest speed from the reference.

## Measured acceptance results

The derivative comparison is:

| derivative | centred N=32 | centred N=64 | ratio | Fourier N=32 |
|---|---:|---:|---:|---:|
| `dx` | 0.170504 | 0.043185 | 3.948 | `1.51e-14` |
| `dxx` | 0.257242 | 0.064871 | 3.965 | `1.52e-13` |
| `dxdy` | 0.485339 | 0.124294 | 3.905 | `5.15e-14` |
| Laplacian | 0.308383 | 0.077705 | 3.969 | `1.59e-13` |

Grid doubling therefore gives the required factor of about four for centred
differences, while Fourier differentiation reaches roundoff.

For the line equation, RK4's spectral boundary is crossed between the plotted
steps `0.045` and `0.056`, around the analytic limit `0.0494`. The larger step
develops the required two-grid-point instability. The measured convergence
slopes are Euler `1.033`, midpoint `2.005`, RK4 `4.004`, and equal-weight RK4
`2.003`; every slope is within 15% of its required value. The one-lap maximum
errors are `1.80e-5` for Fourier RK4, `0.310` for centred-difference RK4, and
`0.210` for Fourier Euler, matching the overlap, dispersive wake, and artificial
growth shown in `line-accuracy.png`.

Taylor-Green starts at energy `0.250000` and enstrophy `0.500000`; at `t=1`
they are `0.167580` and `0.335160`, agreeing to six decimals. Its final velocity
relative error is `7.04e-7`, below the required `1e-5`.

The random flow starts at energy `0.500000`, enstrophy `6.634685`, and ends at
`0.287376`, `0.992354` at `t=10`. Energy falls by less than half while
enstrophy falls by a factor of `6.69`, approximately seven, because viscosity
removes short filaments before the large vortices.

For Taylor-Green at `N=64, nu=0.1`, the predicted diffusive limit is `0.031576`:
`dt=0.032` reaches `t=8`, while `dt=0.033` becomes non-finite at `t=7.82`.
For the random field, the measured initial maximum speed is `2.5908`, giving
the advective bound `0.01839`. RK4 reaches `t=10` at `dt=0.030` and becomes
non-finite at `dt=0.032`, so the measured boundary is between `1.63` and `1.74`
times the bound, within the required factor of one to three. Euler at `dt=0.01`
becomes non-finite at `t=1.11`.

At stable `dt=0.01`, the seeded random-flow perturbation grows by a factor of
`30.36` by `t=20`, above the required tenfold growth, while the Taylor-Green
distance decays into the six-decimal recording floor. Both unperturbed and
perturbed stable runs continue to lose energy; this distinguishes physical
sensitivity from numerical blow-up.

The Taylor-Green fluid order fit gives `4.104`, within 15% of four. Random-flow
self-convergence gives slope `4.035`, within the required interval `[3.7, 4.3]`.
Fourth-order Richardson estimation selects `dt=0.0125`: its predicted error is
`4.02e-6` and its measured error is `3.93e-6`, both below `5e-6`; `dt=0.02`
fails the target.

## Limits and final acceptance

The Richardson estimate isolates temporal error on the `N=128` grid at `t=2`.
It is not an error bound for the continuous PDE, and it does not include spatial
truncation. The stability estimates are modal limits or bounds; the exact time
at which roundoff becomes visible depends on how far a step lies outside the
boundary. The random-flow sensitivity comparison demonstrates divergence of
two valid solutions, not time-step error.

Final acceptance from a clean clone consists of:

1. `cargo test` passes all library and CLI tests.
2. The derivative table has finite-difference ratios near four and Fourier
   errors below `1e-10`.
3. All nine committed evidence files listed above are recreated by the stated
   scripts; each committed file is below 5 MB.
4. Taylor-Green agrees to six decimals and has velocity error below `1e-5`.
5. Both stability boundaries are bracketed as described, Euler fails before
   `t=2`, and the physical random-flow perturbation grows more than tenfold.
6. Both measured RK4 slopes meet their stated tolerances and the selected step
   has predicted and measured errors below `5e-6`.
7. `git status --short` contains no files from `artifacts/`.
