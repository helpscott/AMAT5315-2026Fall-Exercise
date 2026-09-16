#!/usr/bin/env python3
"""Draw the equilibrium curves and locate the two finite-size peaks."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (
    EVIDENCE,
    EXACT_TC,
    five_point_peak,
    merged_metropolis,
    observable_metrics,
    onsager_magnetization,
)


EVIDENCE.mkdir(exist_ok=True)
metrics = {
    lattice_size: observable_metrics(merged_metropolis(lattice_size), lattice_size)
    for lattice_size in (32, 64)
}

temperatures, mean_abs, _ = metrics[64]
fine_grid = np.linspace(1.5, 3.5, 800)
fig, axis = plt.subplots(figsize=(7.2, 4.6))
axis.plot(fine_grid, onsager_magnetization(fine_grid), "k--", label="Onsager, infinite lattice")
axis.plot(temperatures, mean_abs, "o-", ms=4, label="measured, L=64")
axis.axvline(EXACT_TC, color="0.45", ls=":", label=f"exact Tc={EXACT_TC:.5f}")
axis.set(xlabel="temperature T", ylabel="mean |M|", xlim=(1.48, 3.52), ylim=(-0.02, 1.02))
axis.legend()
fig.tight_layout()
fig.savefig(EVIDENCE / "magnetization.png", dpi=180)
plt.close(fig)

peaks = {}
fig, axis = plt.subplots(figsize=(7.2, 4.6))
colors = {32: "C0", 64: "C1"}
for lattice_size in (32, 64):
    temperatures, _, susceptibility = metrics[lattice_size]
    peak, coefficients, fit_temperatures = five_point_peak(temperatures, susceptibility)
    peaks[lattice_size] = peak
    axis.plot(temperatures, susceptibility, "o-", ms=4, color=colors[lattice_size], label=f"L={lattice_size}")
    fit_grid = np.linspace(fit_temperatures[0], fit_temperatures[-1], 160)
    axis.plot(fit_grid, np.polyval(coefficients, fit_grid), "--", color=colors[lattice_size])
    axis.axvline(peak, color=colors[lattice_size], ls=":", alpha=0.8)
axis.axvline(EXACT_TC, color="k", ls="--", label=f"exact Tc={EXACT_TC:.5f}")
axis.set(xlabel="temperature T", ylabel="susceptibility chi")
axis.legend()
fig.tight_layout()
fig.savefig(EVIDENCE / "susceptibility.png", dpi=180)
plt.close(fig)

estimated_tc = 2.0 * peaks[64] - peaks[32]
relative_error = 100.0 * (estimated_tc - EXACT_TC) / EXACT_TC
with (EVIDENCE / "peaks.txt").open("w") as output:
    output.write("L cold_mean_abs_M T_peak\n")
    for lattice_size in (32, 64):
        _, means, _ = metrics[lattice_size]
        output.write(f"{lattice_size} {means[0]:.6f} {peaks[lattice_size]:.6f}\n")
    output.write(f"Tc_extrapolated {estimated_tc:.6f}\n")
    output.write(f"Tc_exact {EXACT_TC:.6f}\n")
    output.write(f"relative_deviation_percent {relative_error:+.4f}\n")

print(f"Tc={estimated_tc:.6f}; deviation={relative_error:+.4f}%")
