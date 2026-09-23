#!/usr/bin/env python3
"""Plot physical perturbation growth in the two stable flows."""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import ARTIFACTS, EVIDENCE, records


EVIDENCE.mkdir(exist_ok=True)
directory = ARTIFACTS / "sensitivity"
fig, axis = plt.subplots(figsize=(7.4, 4.7))
for name, color in (("random", "C3"), ("taylor-green", "C0")):
    original = records(directory / f"{name}-original/fields.jsonl")
    changed = records(directory / f"{name}-perturbed/fields.jsonl")
    times = []
    distances = []
    for first, second in zip(original, changed):
        omega = np.asarray(first["omega"])
        difference = np.asarray(second["omega"]) - omega
        times.append(first["t"])
        distances.append(np.linalg.norm(difference) / np.linalg.norm(omega))
    distances = np.maximum(distances, 1e-9)
    axis.semilogy(times, distances, "o-", ms=3, color=color, label=name)
    print(f"{name}_initial_distance {distances[0]:.12e}")
    print(f"{name}_final_distance {distances[-1]:.12e}")
    print(f"{name}_growth {distances[-1] / distances[0]:.6f}")
axis.axhline(1e-6, color="0.5", ls=":", label="six-decimal field floor")
axis.set(
    xlabel="time t",
    ylabel="relative vorticity distance",
    title="Stable RK4 runs from a small initial perturbation",
)
axis.grid(True, which="both", alpha=0.25)
axis.legend()
fig.tight_layout()
fig.savefig(EVIDENCE / "sensitivity.png", dpi=180)
plt.close(fig)
