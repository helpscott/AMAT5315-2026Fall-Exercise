#!/usr/bin/env python3
"""Check the Boltzmann energy-histogram ratio at T=3.0 and T=3.1."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import EVIDENCE, ROOT, rows


def total_energies(temperature: float) -> np.ndarray:
    path = ROOT / "runs" / f"T{temperature:.1f}" / "series.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"required Part 1 run is missing: {path}")
    return np.asarray([record["E"] * 4096 for record in rows(path)], dtype=float)


EVIDENCE.mkdir(exist_ok=True)
cold = total_energies(3.0)
hot = total_energies(3.1)
left = 40.0 * np.floor(min(cold.min(), hot.min()) / 40.0)
right = 40.0 * np.ceil(max(cold.max(), hot.max()) / 40.0)
edges = np.arange(left, right + 40.0, 40.0)
cold_counts, _ = np.histogram(cold, bins=edges)
hot_counts, _ = np.histogram(hot, bins=edges)
centers = (edges[:-1] + edges[1:]) / 2.0
valid = (cold_counts >= 5) & (hot_counts >= 5)
ratio = np.log(hot_counts[valid] / cold_counts[valid])
prediction_slope = 1.0 / 3.0 - 1.0 / 3.1
intercept = float(np.mean(ratio - prediction_slope * centers[valid]))

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
axes[0].stairs(cold_counts, edges, label="T=3.0", color="C0")
axes[0].stairs(hot_counts, edges, label="T=3.1", color="C3")
axes[0].set(xlabel="total energy E", ylabel="sweeps in 40-unit bin")
axes[0].legend()
axes[1].scatter(centers[valid], ratio, s=28, label="measured ratio")
axes[1].plot(
    centers[valid],
    prediction_slope * centers[valid] + intercept,
    "k--",
    label=f"Boltzmann slope {prediction_slope:.7f}",
)
axes[1].set(xlabel="total energy E", ylabel="ln(P3.1 / P3.0)")
axes[1].legend()
fig.tight_layout()
fig.savefig(EVIDENCE / "boltzmann.png", dpi=180)
plt.close(fig)

fit_slope = float(np.polyfit(centers[valid], ratio, 1)[0])
print(f"predicted slope={prediction_slope:.7f}; fitted slope={fit_slope:.7f}")
