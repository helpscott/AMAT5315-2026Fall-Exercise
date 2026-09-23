#!/usr/bin/env python3
"""Measure RK4's temporal order on Taylor-Green flow."""

import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import ARTIFACTS, EVIDENCE, build, field, final_record, relative_l2, run_fluid


build()
EVIDENCE.mkdir(exist_ok=True)
directory = ARTIFACTS / "order"
initial = field(["taylor-green", "--n", "8"])
exact = field(["taylor-green", "--n", "8", "--nu", "0.5", "--t", "2"])
exact_velocity = np.concatenate([exact["u"], exact["v"]])
steps = np.asarray([0.4, 0.25, 0.2])
errors = []
for dt in steps:
    output = directory / f"rk4-dt{dt:g}"
    result = run_fluid(
        initial,
        [
            "--method",
            "rk4",
            "--nu",
            "0.5",
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
    final = final_record(output)
    numerical = np.concatenate([final["u"], final["v"]])
    errors.append(relative_l2(numerical, exact_velocity))
errors = np.asarray(errors)
slope, intercept = np.polyfit(np.log(steps), np.log(errors), 1)

fig, axis = plt.subplots(figsize=(6.8, 4.6))
axis.loglog(steps, errors, "o", ms=7, label=f"RK4 measured, slope={slope:.3f}")
grid = np.linspace(steps.min(), steps.max(), 120)
axis.loglog(grid, np.exp(intercept) * grid**slope, "-", alpha=0.8)
axis.axhline(1e-7, color="0.5", ls=":", label="six-decimal storage floor")
axis.set(
    xlabel="time step dt",
    ylabel="relative velocity error at t=2",
    title="Taylor-Green temporal order, N=8, nu=0.5",
)
axis.grid(True, which="both", alpha=0.25)
axis.legend()
fig.tight_layout()
fig.savefig(EVIDENCE / "order.png", dpi=180)
plt.close(fig)

report = {"steps": steps.tolist(), "errors": errors.tolist(), "slope": float(slope)}
(directory / "order.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
if not 3.4 <= slope <= 4.6:
    raise SystemExit(f"RK4 order {slope:.4f} is outside the required 15% interval")
