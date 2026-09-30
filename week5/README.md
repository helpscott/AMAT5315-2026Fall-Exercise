# Week 5 - Automatic differentiation and checkpointing

This exercise differentiates a damped acoustic time-stepping simulation. It starts with hand-written JAX forward and reverse passes for a Lennard-Jones pair energy, then uses Rust/Enzyme JVPs and VJPs to form Born data and a reverse-time migration (RTM) image. A Treeverse schedule reproduces the full-history image while retaining only a small number of restart states.

## Environment and fixed inputs

The Rust crate pins `nightly-2026-09-05` with the `enzyme` component. `build.rs` applies `-Zautodiff=Enable` only to the allocation-free `no_std` kernel in `src/kernel.rs`; the CLI and file handling compile as ordinary Rust. The course setup instructions are preserved in `.agents/skills/enzyme-setup/SKILL.md`.

Download the ignored experiment files before reproducing the runs:

```bash
curl -fLO https://giggleliu.github.io/AMAT5315-2026Fall/downloads/week5-inputs.zip
unzip -o week5-inputs.zip
rm week5-inputs.zip
cd week5
```

The archive supplies `inputs/reflector.json` and `inputs/marmousi.json`. `inputs/` and all generated `.npy` arrays remain outside Git. `MARMOUSI-LICENSE` records the model's license and provenance.

Set up the plotting environment and build the solver:

```bash
uv sync
rustup toolchain install nightly-2026-09-05 --profile minimal --component enzyme
cargo test --release
cargo build --release
```

After recreating all arrays and plots, run the consolidated acceptance check:

```bash
uv run scripts/verify_all.py
```

The local derivative smoke test requires `cube(2) = 8` and both Enzyme derivatives to equal 12. A second unit test checks the dot-product identity for one differentiated wave-equation step. The Treeverse unit test checks the four reference recomputation counts.

## Part 1 - forward and reverse AD

Run:

```bash
uv run scripts/ad_analysis.py
```

Evidence:

| File | Quantity and claim |
| --- | --- |
| `artifacts/ad/derivatives.json` | Values, tangents and adjoints at `r = 1.3`; the adjoint of the shared node `a` contains both reverse paths. |
| `artifacts/ad/modes.png` | Hand-written forward and reverse derivatives against the analytic derivative and centered finite differences over 601 samples. |
| `artifacts/ad/graph.png` | Four operations in the JAX primal graph. |
| `artifacts/ad/grad-graph.png` | JAX gradient graph; `add_any` joins the two contributions to `a`. |
| `artifacts/ad/scaling.png` | Forward cost grows with the `3N` input directions; one reverse VJP remains within a few energy evaluations. |

The maximum errors are `2.13e-14` for forward AD, `1.42e-14` for reverse AD, and `4.95e-9` for the finite difference. The largest relative cluster-force errors are `1.55e-15` and `1.98e-15` for forward and reverse mode.

## Part 2 - acoustic forward model

The restart state is exactly two fields, `State { previous, current } = (u^(n-1), u^n)`. The next field uses centered time differences, the five-point Laplacian, the specified quadratic sponge, a Ricker pulse, and a unit-width Gaussian source footprint. Receivers sample `u^(n+1)` after each update.

Run and plot:

```bash
target/release/seismic --experiment inputs/reflector.json --mode forward --every 3 --out artifacts/forward
uv run scripts/plot_forward.py
```

Evidence:

| File | Quantity and claim |
| --- | --- |
| `artifacts/inputs.png` | Three shots, fourteen receivers, the 2.1 km reflector and the 0.8 Hz Ricker pulse. |
| `artifacts/forward/run.json` | Compact experiment reference and the recorded frame steps/times. |
| `artifacts/forward/result.json` | Forward output dimensions. |
| `artifacts/forward/gathers.png` | Three shot gathers with the direct event and later negative lobe. |
| `artifacts/forward/wavefield.png` | Full wavefield for shot 0 at step 150, 3.00 s. |
| `artifacts/forward/echo.png` | Perturbed-minus-background field at the same step; the weak reflector echo is isolated. |

The all-trace L2 norm is `11.574769503614`. The outer-shot maxima are `0.608095143427` and the central-shot maximum is `0.592713973467`, all at trace index 83. At step 150 the echo maximum is `0.006224704`, 2.44% of the full-field maximum `0.255243421`.

## Part 3 - Enzyme Born and adjoint passes

`src/kernel.rs` contains one local acoustic update. Enzyme differentiates that update to produce a JVP and VJP; `src/solver.rs` composes those calls through time. Forward mode propagates the input velocity perturbation to the Born data. Reverse mode injects receiver weights, sends the state adjoint backward, and sums the velocity adjoint into the image.

Run and plot:

```bash
target/release/seismic --experiment inputs/reflector.json --mode born --out artifacts/born
target/release/seismic --experiment inputs/reflector.json --mode adjoint \
  --data artifacts/born/born_data.npy --every 3 --out artifacts/adjoint
uv run scripts/plot_adjoint.py
```

Evidence:

| File | Quantity and claim |
| --- | --- |
| `artifacts/born/run.json`, `artifacts/born/result.json` | Born experiment reference and output dimensions. |
| `artifacts/adjoint/run.json`, `artifacts/adjoint/result.json` | Full-history reverse run and its 241-state storage statistics. |
| `artifacts/adjoint/image.png` | Known perturbation, raw RTM image and depth profile; the peak is at 2.1 km. |
| `artifacts/adjoint/wavefield.png` | Step 132 of the reverse recording; the adjoint field refocuses on the reflector. |

For `d = Jm`, the two transpose-identity values are `sum(d^2) = 0.034847890215163782` and `sum(m * J^T d) = 0.034847890215163761`. Their relative difference is `5.97e-16`. The image-depth error is 0.0 km.

## Part 4 - Treeverse checkpointing

The port in `src/solver.rs` stores complete two-field states. A `restore` selects a saved restart state, `call` replays one forward step, `grad` performs one Enzyme VJP, and `fetch` frees a slot. The image accumulator is never restored, so replay supplies primal states without undoing already accumulated image contributions.

Run the reflector schedules and audit them:

```bash
for b in 1 3 5 10; do
  target/release/seismic --experiment inputs/reflector.json --mode adjoint \
    --data artifacts/born/born_data.npy --storage treeverse --checkpoints "$b" \
    --out "artifacts/checkpoint-$b"
done
uv run scripts/plot_checkpoints.py
```

Evidence:

| File | Quantity and claim |
| --- | --- |
| `artifacts/checkpoint-{1,3,5,10}/run.json` | Input reference for each checkpoint budget. |
| `artifacts/checkpoint-{1,3,5,10}/result.json` | Image-storage and recomputation statistics. |
| `artifacts/checkpoint-{1,3,5,10}/actions-{0,1,2}.json` | Complete schedule for every reflector shot; audit checks reverse order, restores and budget. |
| `artifacts/checkpoint-actions.png` | Budget-5 sawtooth of store, restore, replay, gradient and fetch actions. |
| `artifacts/checkpoint-work.png` | Measured storage-work trade-off against full history. |

Every checkpointed image is bitwise equal to the full-history image. Budgets 1, 3, 5 and 10 require 28,680, 1,695, 990 and 642 forward calls per shot and peak at 2, 4, 6 and 11 saved states. Every action audit reports zero missing/out-of-order gradient steps, zero invalid restores and zero budget overruns. Full history stores 241 states or 6,481,936 bytes.

Run Marmousi with checkpointing; do not run its adjoint with full history:

```bash
target/release/seismic --experiment inputs/marmousi.json --mode born --out artifacts/marmousi-born
target/release/seismic --experiment inputs/marmousi.json --mode adjoint \
  --data artifacts/marmousi-born/born_data.npy --storage treeverse --checkpoints 5 \
  --out artifacts/marmousi-image
uv run scripts/plot_marmousi.py
```

Evidence:

| File | Quantity and claim |
| --- | --- |
| `artifacts/marmousi-born/run.json`, `artifacts/marmousi-born/result.json` | Nine-shot Born run metadata. |
| `artifacts/marmousi-image/run.json`, `artifacts/marmousi-image/result.json` | Six-state checkpointed migration statistics. |
| `artifacts/marmousi-image/actions-{0..8}.json` | Complete Treeverse schedule for every Marmousi shot. |
| `artifacts/marmousi.png` | Smooth background, dipping perturbations, the 10 km shot gather and raw checkpointed RTM image on one depth-independent amplitude scale. |

The Marmousi image L2 norm is `6.703774060378e-4`. Six saved states occupy `20,788,320` bytes, matching the answer-key requirements.

The final image is `J^T J m`, not `m`. Even with an accurate derivative, finite source bandwidth, limited receiver aperture and uneven illumination make `J^T J` a blurred, spatially varying imaging operator. It locates illuminated interfaces but does not reproduce every input amplitude or fault, and deeper layers remain weak.

## Instructor demonstration

Show `artifacts/ad/modes.png`, `artifacts/ad/grad-graph.png`, `artifacts/ad/scaling.png`, `artifacts/adjoint/image.png`, `artifacts/checkpoint-actions.png`, `artifacts/checkpoint-work.png`, and `artifacts/marmousi.png`. Explain the two fields in `State`, point to one generated Enzyme JVP/VJP wrapper in `src/kernel.rs`, and use the zero checkpoint-image difference to show that replay does not reset the accumulated image.
