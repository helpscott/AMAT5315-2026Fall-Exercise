#!/usr/bin/env python3
"""Run random-flow refinement and select a step with Richardson estimation."""

import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import ARTIFACTS, EVIDENCE, build, field, final_record, relative_l2, run_fluid


build()
EVIDENCE.mkdir(exist_ok=True)
directory = ARTIFACTS / "convergence"
initial = field(["random", "--n", "128", "--seed", "2026", "--k-min", "2", "--k-max", "6"])
steps = np.asarray([0.02, 0.0125, 0.01])
reference_step = 0.0025
all_steps = [*steps, reference_step]
fields = {}
for dt in all_steps:
    output = directory / f"rk4-dt{dt:g}"
    result = run_fluid(
        initial,
        [
            "--method",
            "rk4",
            "--nu",
            "0.004",
            "--dt",
            str(dt),
            "--t-end",
            "2",
            "--every",
            "2",
        ],
        output,
        directory / f"rk4-dt{dt:g}.tsv",
    )
    result.check_returncode()
    fields[dt] = np.asarray(final_record(output)["omega"])

reference = fields[reference_step]
errors = np.asarray([relative_l2(fields[dt], reference) for dt in steps])
slope, intercept = np.polyfit(np.log(steps), np.log(errors), 1)
richardson_001 = float(
    np.linalg.norm(fields[0.02] - fields[0.01]) / (15.0 * np.linalg.norm(fields[0.01]))
)
predicted = {float(dt): richardson_001 * (dt / 0.01) ** 4 for dt in steps}
eligible = [dt for dt in steps if predicted[float(dt)] < 5e-6]
if not eligible:
    raise SystemExit("no candidate step meets the 5e-6 Richardson target")
chosen = float(max(eligible))
measured = {float(dt): float(error) for dt, error in zip(steps, errors)}

report = {
    "reference_dt": reference_step,
    "runs": [
        {"dt": float(dt), "relative_omega_error": measured[float(dt)]} for dt in steps
    ],
    "fitted_slope": float(slope),
    "richardson_error_at_dt_0.01": richardson_001,
    "predicted_errors": {str(dt): value for dt, value in predicted.items()},
    "target": 5e-6,
    "chosen_dt": chosen,
    "chosen_predicted_error": predicted[chosen],
    "chosen_measured_error": measured[chosen],
}
(EVIDENCE / "convergence.json").write_text(json.dumps(report, indent=2))

fig, axis = plt.subplots(figsize=(6.9, 4.7))
axis.loglog(steps, errors, "o-", label=f"measured, slope={slope:.3f}")
grid = np.linspace(steps.min(), steps.max(), 150)
axis.loglog(grid, np.exp(intercept) * grid**slope, "--", alpha=0.8, label="log-log fit")
axis.axhline(5e-6, color="C3", ls=":", label="5e-6 target")
axis.scatter(
    [chosen],
    [measured[chosen]],
    s=100,
    facecolors="none",
    edgecolors="black",
    linewidths=1.5,
    label=f"chosen dt={chosen:g}",
)
axis.set(
    xlabel="time step dt",
    ylabel="relative omega error at t=2",
    title="Random-flow temporal refinement, N=128",
)
axis.grid(True, which="both", alpha=0.25)
axis.legend(fontsize=8)
fig.tight_layout()
fig.savefig(EVIDENCE / "convergence.png", dpi=180)
plt.close(fig)

print(json.dumps(report, indent=2))
if not 3.7 <= slope <= 4.3:
    raise SystemExit(f"random-flow order {slope:.4f} is outside [3.7, 4.3]")
