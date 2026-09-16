# Week 3 — 2D Ising Monte Carlo

This directory implements the square-lattice, zero-field Ising model with periodic boundaries. The command-line program starts from an all-up lattice, warms through an ascending temperature ramp, and writes machine-readable run metadata and time series.

## Build and smoke tests

```bash
cd week3
cargo test
cargo install --path . --quiet
ising --update metropolis --l 16 --t-from 2.0 --t-to 2.0 --t-step 0.1 --discard 100 --measure 100 --every 20 --seed 2026 --out runs/smoke
```

The contract is in `ising.design.toml`. `metropolis` uses L² independent site proposals per sweep and reports acceptance; `wolff` uses one cluster flip per move and reports mean cluster size. `series.jsonl` stores signed magnetization and energy per site; `spins.jsonl` is written only when `--every > 0`.

## Required evidence runs

```bash
ising --update metropolis --l 64 --t-from 1.8 --t-to 1.8 --t-step 0.1 --discard 2000 --measure 2000 --seed 2026 --out artifacts/T1.8
ising --update metropolis --l 64 --t-from 3.0 --t-to 3.0 --t-step 0.1 --discard 2000 --measure 2000 --seed 2026 --out artifacts/T3.0
ising --update metropolis --l 64 --t-from 3.1 --t-to 3.1 --t-step 0.1 --discard 2000 --measure 2000 --seed 2026 --out artifacts/T3.1
ising --update metropolis --l 64 --t-from 1.5 --t-to 3.5 --t-step 0.05 --discard 2000 --measure 200 --every 20 --seed 2026 --out runs/ramp
cp runs/ramp/spins.jsonl spins.jsonl
ising --update metropolis --l 32 --t-from 1.5 --t-to 3.5 --t-step 0.1 --discard 2000 --measure 5000 --seed 1042 --out artifacts/coarse-l32
ising --update metropolis --l 64 --t-from 1.5 --t-to 3.5 --t-step 0.1 --discard 2000 --measure 5000 --seed 42 --out artifacts/coarse-l64
ising --update metropolis --l 32 --t-from 2.0 --t-to 2.6 --t-step 0.05 --discard 2000 --measure 100000 --seed 1042 --out artifacts/window-l32
ising --update metropolis --l 64 --t-from 2.0 --t-to 2.6 --t-step 0.05 --discard 2000 --measure 100000 --seed 42 --out artifacts/window-l64
ising --update wolff --l 64 --t-from 2.0 --t-to 2.6 --t-step 0.05 --discard 20000 --measure 100000 --seed 42 --out artifacts/wolff-l64
ising --update wolff --l 32 --t-from 2.0 --t-to 2.6 --t-step 0.05 --discard 20000 --measure 100000 --seed 1042 --out artifacts/wolff-l32
```

The long runs are intentionally kept under `artifacts/` and are ignored by Git. Evidence scripts can be run from this directory:

```bash
python3 scripts/peaks.py
python3 scripts/errors.py > evidence/errors.stdout
python3 scripts/boltzmann.py
python3 scripts/trace.py
python3 scripts/acf_binning.py
python3 scripts/tau.py
python3 scripts/bootstrap.py > evidence/bootstrap.txt
python3 scripts/magnetization_compare.py
python3 scripts/compare.py
```

These create the plots and `peaks.txt`/`errors.txt`. Statistical uncertainty is reported using both a naive standard error and autocorrelation-aware block estimates. The exact critical temperature is `2/ln(1+sqrt(2)) = 2.26919`; finite-size peak shifts are extrapolated linearly in `1/L`. Wolff and Metropolis uncertainties are provisional whenever the chosen block length has not reached a visible plateau.

## Viewer capture and clean-clone check

Open the course spin viewer with the raw committed URL for `week3/spins.jsonl`, load the file, select `T=1.8`, `2.3`, and `3.0`, and save the three PNG captures as `evidence/viewer-T1.8.png`, `evidence/viewer-T2.3.png`, and `evidence/viewer-T3.0.png`. In a second terminal, follow this README from a clean clone and verify every command above creates the named output files. Commit only source, scripts, evidence, README, and `spins.jsonl`; never commit `artifacts/`.
