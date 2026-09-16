#!/usr/bin/env python3
"""Block-bootstrap the two finite-size susceptibility peaks."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (
    BLOCK_LENGTHS,
    EVIDENCE,
    EXACT_TC,
    bootstrap_susceptibility,
    five_point_peak,
    load_run,
    observable_metrics,
)


REPLICATES = 500
EVIDENCE.mkdir(exist_ok=True)
window = {lattice_size: load_run(f"window-l{lattice_size}") for lattice_size in (32, 64)}
central = {
    lattice_size: observable_metrics(window[lattice_size], lattice_size)
    for lattice_size in (32, 64)
}
central_peaks = {
    lattice_size: five_point_peak(central[lattice_size][0], central[lattice_size][2])[0]
    for lattice_size in (32, 64)
}
central_tc = 2.0 * central_peaks[64] - central_peaks[32]

rng = np.random.default_rng(2026)
results = {}
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), sharey=False)

for lattice_size, axis in zip((32, 64), axes):
    temperatures, _, susceptibility = central[lattice_size]
    peak, coefficients, fit_temperatures = five_point_peak(temperatures, susceptibility)
    curve_grid = np.linspace(fit_temperatures[0], fit_temperatures[-1], 161)
    axis.plot(temperatures, susceptibility, "ko", ms=4, label="measured chi")
    fit_grid = np.linspace(fit_temperatures[0], fit_temperatures[-1], 160)
    axis.plot(fit_grid, np.polyval(coefficients, fit_grid), "k-", lw=1.5, label="five-point fit")
    axis.axvline(peak, color="k", ls=":", lw=1)
    for block_length, color in zip(BLOCK_LENGTHS, ("C0", "C1", "C2")):
        bootstrap_temperatures, simulated = bootstrap_susceptibility(
            window[lattice_size], lattice_size, block_length, REPLICATES, rng
        )
        fits = []
        peaks = np.full(REPLICATES, np.nan)
        failures = 0
        for replicate_index, replicate in enumerate(simulated):
            try:
                fitted_peak, fitted_coefficients, _ = five_point_peak(
                    bootstrap_temperatures, replicate
                )
            except ValueError:
                failures += 1
                continue
            peaks[replicate_index] = fitted_peak
            fits.append(np.polyval(fitted_coefficients, curve_grid))
        results[(lattice_size, block_length)] = (
            peaks,
            failures,
            np.asarray(fits),
        )
        if fits:
            lower, upper = np.percentile(np.asarray(fits), [5, 95], axis=0)
            axis.fill_between(
                curve_grid,
                lower,
                upper,
                color=color,
                alpha=0.13,
                label=f"{block_length}-sweep 90% envelope",
            )
    axis.axvline(EXACT_TC, color="C3", ls="--", label=f"exact Tc={EXACT_TC:.5f}")
    axis.set(
        title=f"L={lattice_size}",
        xlabel="temperature T",
        ylabel="susceptibility chi",
        xlim=(fit_temperatures[0] - 0.05, fit_temperatures[-1] + 0.05),
        ylim=(0, None),
    )
    axis.legend(fontsize=8)

tc_results = {}
for block_length in BLOCK_LENGTHS:
    peaks_32, failures_32, _ = results[(32, block_length)]
    peaks_64, failures_64, _ = results[(64, block_length)]
    valid_mask = np.isfinite(peaks_32) & np.isfinite(peaks_64)
    extrapolated = 2.0 * peaks_64[valid_mask] - peaks_32[valid_mask]
    tc_results[block_length] = (
        float(extrapolated.mean()),
        float(extrapolated.std(ddof=1)),
        failures_32,
        failures_64,
        int(valid_mask.sum()),
    )

errors = np.asarray([tc_results[length][1] for length in BLOCK_LENGTHS])
stable = float(errors.max() - errors.min()) <= 0.1 * float(errors.mean())
fig.suptitle(
    f"Metropolis block bootstrap: Tc={central_tc:.5f}; sampling error "
    f"{'resolved' if stable else 'unresolved'}"
)
fig.tight_layout()
fig.savefig(EVIDENCE / "chi-bootstrap.png", dpi=180)
plt.close(fig)

with (EVIDENCE / "bootstrap.txt").open("w") as output:
    output.write(f"central_Tc {central_tc:.6f}\n")
    output.write("block_length mean_Tc bootstrap_se valid failed_L32 failed_L64\n")
    for block_length in BLOCK_LENGTHS:
        mean_tc, error, failed_32, failed_64, valid = tc_results[block_length]
        output.write(
            f"{block_length} {mean_tc:.6f} {error:.6f} {valid} "
            f"{failed_32} {failed_64}\n"
        )
    output.write(f"sampling_error {'resolved' if stable else 'unresolved'}\n")

print((EVIDENCE / "bootstrap.txt").read_text(), end="")
