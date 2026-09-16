# Week 3 — 2D Ising Monte Carlo

This crate simulates the zero-field square-lattice Ising model with periodic
boundaries. It implements single-spin Metropolis sweeps and Wolff cluster
moves, uses a deterministic seeded RNG, and records signed magnetization and
energy per spin as JSON Lines. The exact command contract is
`ising.design.toml`.

## Install and test

Run these commands from a clean clone. Python 3.10 or newer is recommended.

```bash
cd week3
cargo test
cargo install --path . --quiet
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-analysis.txt
```

A Metropolis sweep is `L^2` independent site proposals. A Wolff step is one
cluster flip. `run.json` records this distinction in `time_unit`, together with
the seed and every CLI parameter. `series.jsonl` contains one row per measured
step. `spins.jsonl` is produced only when `--every` is positive.

## Required runs

The three single-temperature checks establish ordered, disordered, and nearby
high-temperature behavior:

```bash
ising --update metropolis --l 64 --t-from 1.8 --t-to 1.8 --t-step 0.1 --discard 2000 --measure 2000 --seed 2026 --out runs/T1.8
ising --update metropolis --l 64 --t-from 3.0 --t-to 3.0 --t-step 0.1 --discard 2000 --measure 2000 --seed 2026 --out runs/T3.0
ising --update metropolis --l 64 --t-from 3.1 --t-to 3.1 --t-step 0.1 --discard 2000 --measure 2000 --seed 2026 --out runs/T3.1
```

Create the public viewer recording and copy it to the required committed path:

```bash
ising --update metropolis --l 64 --t-from 1.5 --t-to 3.5 --t-step 0.05 --discard 2000 --measure 200 --every 20 --seed 2026 --out runs/ramp
cp runs/ramp/spins.jsonl spins.jsonl
```

The statistical analysis merges each coarse grid with its longer critical
window; the critical-window values replace overlapping coarse-grid values.

```bash
ising --update metropolis --l 32 --t-from 1.5 --t-to 3.5 --t-step 0.1 --discard 2000 --measure 5000 --seed 1042 --out artifacts/coarse-l32
ising --update metropolis --l 64 --t-from 1.5 --t-to 3.5 --t-step 0.1 --discard 2000 --measure 5000 --seed 42 --out artifacts/coarse-l64
ising --update metropolis --l 32 --t-from 2.0 --t-to 2.6 --t-step 0.05 --discard 2000 --measure 100000 --seed 1042 --out artifacts/window-l32
ising --update metropolis --l 64 --t-from 2.0 --t-to 2.6 --t-step 0.05 --discard 2000 --measure 100000 --seed 42 --out artifacts/window-l64
ising --update wolff --l 32 --t-from 2.0 --t-to 2.6 --t-step 0.05 --discard 20000 --measure 100000 --seed 1042 --out artifacts/wolff-l32
ising --update wolff --l 64 --t-from 2.0 --t-to 2.6 --t-step 0.05 --discard 20000 --measure 100000 --seed 42 --out artifacts/wolff-l64
```

The large run directories stay under ignored `artifacts/`; they must not be
committed. Every invocation starts from an all-up lattice and carries the final
state forward through the ascending temperature ramp.

## Produce every evidence file

Run the scripts from `week3` in this order:

```bash
python3 scripts/peaks.py
# evidence/magnetization.png, evidence/susceptibility.png, evidence/peaks.txt

python3 scripts/errors.py
# evidence/errors.txt

python3 scripts/boltzmann.py
# evidence/boltzmann.png

python3 scripts/trace.py
# evidence/trace.png

python3 scripts/acf_binning.py
# evidence/acf-binning.png

python3 scripts/tau.py
# evidence/tau.png

python3 scripts/bootstrap.py
# evidence/chi-bootstrap.png, evidence/bootstrap.txt

python3 scripts/magnetization_compare.py
# evidence/magnetization-compare.png, evidence/sampler-comparison.txt

python3 scripts/compare.py
# evidence/tau-compare.png, evidence/work-comparison.txt
```

The analysis uses

```text
chi = L^2 ( <M^2> - <|M|>^2 ) / T
```

and fits the five points surrounding each susceptibility maximum with a
quadratic. With two sizes, the infinite-size estimate is
`Tc = 2*Tpeak(L=64) - Tpeak(L=32)`. Autocorrelation times use an FFT
autocorrelation function and the running window `lag <= 6*tau`. Bootstrap
uncertainties use 500 moving-block resamples. An uncertainty is called resolved
only when the estimates from 2000-, 4000-, and 8000-step blocks agree within
10% of their mean.

## Viewer evidence

The committed recording is available at:

```text
https://raw.githubusercontent.com/helpscott/AMAT5315-2026Fall-Exercise/main/week3/spins.jsonl
```

Open `https://giggleliu.github.io/AMAT5315-2026Fall/week3-viewer.html`, paste
the raw URL, and load it. For each of `T=1.8`, `T=2.3`, and `T=3.0`, select the
temperature, use **Copy link**, open that link in a fresh private window, and
use **Save PNG**. The resulting official-viewer exports are:

- `evidence/viewer-T1.8.png`
- `evidence/viewer-T2.3.png`
- `evidence/viewer-T3.0.png`

The copied links are:

- `https://giggleliu.github.io/AMAT5315-2026Fall/week3-viewer.html?src=https%3A%2F%2Fraw.githubusercontent.com%2Fhelpscott%2FAMAT5315-2026Fall-Exercise%2Fmain%2Fweek3%2Fspins.jsonl&T=1.8`
- `https://giggleliu.github.io/AMAT5315-2026Fall/week3-viewer.html?src=https%3A%2F%2Fraw.githubusercontent.com%2Fhelpscott%2FAMAT5315-2026Fall-Exercise%2Fmain%2Fweek3%2Fspins.jsonl&T=2.3`
- `https://giggleliu.github.io/AMAT5315-2026Fall/week3-viewer.html?src=https%3A%2F%2Fraw.githubusercontent.com%2Fhelpscott%2FAMAT5315-2026Fall-Exercise%2Fmain%2Fweek3%2Fspins.jsonl&T=3`

## Results and limits

The measured Metropolis peaks are `2.35100` for `L=32` and `2.31605` for
`L=64`, giving `Tc=2.28109`. This differs from the exact
`2/ln(1+sqrt(2)) = 2.26919` by `+0.525%`, within the 2% target. The three
Metropolis bootstrap standard errors are `0.00868`, `0.00840`, and `0.01094`.
They do not meet the 10% stability rule, so the sampling error is reported as
unresolved. The five-point fit and two-size extrapolation also introduce
systematic error that the bootstrap does not measure.

At `L=64, T=2.3`, Metropolis gives `<|M|>=0.46305` with a 4000-step block
error of `0.01992`; Wolff gives `0.43647 +/- 0.00156`. Their normalized
difference is `d=1.33`, but agreement remains provisional because the
Metropolis block error has not stabilized. The Wolff extrapolation gives
`Tc=2.27508`, `+0.260%` from the exact value. Its three bootstrap errors are
`0.000676`, `0.000611`, and `0.000578`; these also narrowly miss the strict 10%
stability rule.

The Metropolis integrated autocorrelation time at `T=2.3` is `580.38` sweeps,
so 100000 measurements contain only about `86.15` effective independent
samples. Matching today's naive error after accounting for this correlation
would require about `116075805` sweeps. A measured rate of about `7519`
sweeps/s gives roughly `4.29` hours. On the work-normalized scale used in
`evidence/work-comparison.txt`, Wolff reduces the critical autocorrelation time
by a factor of about `546`.

## Repeatability and final acceptance

The seeded output is byte reproducible. A compact check is:

```bash
rm -rf /tmp/ising-repeat-a /tmp/ising-repeat-b
ising --update metropolis --l 16 --t-from 2.0 --t-to 2.1 --t-step 0.1 --discard 100 --measure 200 --every 20 --seed 2026 --out /tmp/ising-repeat-a
ising --update metropolis --l 16 --t-from 2.0 --t-to 2.1 --t-step 0.1 --discard 100 --measure 200 --every 20 --seed 2026 --out /tmp/ising-repeat-b
diff -ru /tmp/ising-repeat-a /tmp/ising-repeat-b
```

Final acceptance from a clean clone consists of:

1. `cargo test` passes, including the CLI contract tests.
2. The repeatability `diff` is empty.
3. All commands above recreate their named files without editing source.
4. `spins.jsonl` is below 5 MB and its raw GitHub URL returns HTTP 200.
5. Each viewer PNG opens and visibly shows the official viewer stamp, the raw
   source URL, and the requested temperature.
6. `git status --short` shows no generated `artifacts/` or `runs/` files staged
   for commit.
