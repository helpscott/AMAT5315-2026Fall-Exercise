#!/usr/bin/env python3
"""Compare Metropolis and Wolff equilibrium measurements."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (
    BLOCK_LENGTHS,
    EVIDENCE,
    EXACT_TC,
    bootstrap_mean_abs_error,
    bootstrap_susceptibility,
    five_point_peak,
    load_run,
    observable_metrics,
)


REPLICATES = 500
EVIDENCE.mkdir(exist_ok=True)
rng = np.random.default_rng(5315)
metropolis = load_run("window-l64")
wolff_64 = load_run("wolff-l64")
wolff = {32: load_run("wolff-l32"), 64: wolff_64}

mean_bootstrap = {}
for name, grouped in (("Metropolis", metropolis), ("Wolff", wolff_64)):
    for block_length in BLOCK_LENGTHS:
        mean_bootstrap[(name, block_length)] = bootstrap_mean_abs_error(
            grouped, block_length, REPLICATES, rng
        )

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7))
for name, color in (("Metropolis", "C0"), ("Wolff", "C3")):
    temperatures, means, errors = mean_bootstrap[(name, 4000)]
    axes[0].errorbar(
        temperatures,
        means,
        yerr=errors,
        fmt="o-",
        ms=4,
        capsize=2,
        color=color,
        label=f"{name}, 4000-step blocks",
    )
axes[0].axvline(EXACT_TC, color="k", ls="--", label=f"exact Tc={EXACT_TC:.5f}")
axes[0].set(xlabel="temperature T", ylabel="mean |M|", title="Sampler agreement, L=64")
axes[0].legend(fontsize=8)

wolff_metrics = {lattice_size: observable_metrics(wolff[lattice_size], lattice_size) for lattice_size in (32, 64)}
wolff_peaks = {}
for lattice_size, color in ((32, "C0"), (64, "C1")):
    temperatures, _, susceptibility = wolff_metrics[lattice_size]
    peak, coefficients, fit_temperatures = five_point_peak(temperatures, susceptibility)
    wolff_peaks[lattice_size] = peak
    axes[1].plot(temperatures, susceptibility, "o-", ms=4, color=color, label=f"Wolff L={lattice_size}")
    fit_grid = np.linspace(fit_temperatures[0], fit_temperatures[-1], 160)
    axes[1].plot(fit_grid, np.polyval(coefficients, fit_grid), "--", color=color)
    axes[1].axvline(peak, color=color, ls=":", alpha=0.8)
wolff_tc = 2.0 * wolff_peaks[64] - wolff_peaks[32]
axes[1].axvline(EXACT_TC, color="k", ls="--", label=f"exact Tc={EXACT_TC:.5f}")
axes[1].set(xlabel="temperature T", ylabel="susceptibility chi", title=f"Wolff peaks; extrapolated Tc={wolff_tc:.5f}")
axes[1].legend(fontsize=8)

temperature_index = int(np.where(np.isclose(mean_bootstrap[("Metropolis", 4000)][0], 2.3))[0][0])
agreement = {}
stable_by_sampler = {}
for name in ("Metropolis", "Wolff"):
    errors = np.asarray(
        [mean_bootstrap[(name, length)][2][temperature_index] for length in BLOCK_LENGTHS]
    )
    stable_by_sampler[name] = float(errors.max() - errors.min()) <= 0.1 * float(errors.mean())
    agreement[name] = (
        mean_bootstrap[(name, 4000)][1][temperature_index],
        mean_bootstrap[(name, 4000)][2][temperature_index],
        errors,
    )
distance = abs(agreement["Metropolis"][0] - agreement["Wolff"][0]) / np.hypot(
    agreement["Metropolis"][1], agreement["Wolff"][1]
)
status = (
    "agreement"
    if distance <= 3 and all(stable_by_sampler.values())
    else "agreement provisional"
    if distance <= 3
    else "discrepancy"
)
axes[0].text(
    0.03,
    0.05,
    f"T=2.3: d={distance:.2f}; {status}",
    transform=axes[0].transAxes,
    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
)

tc_bootstrap = {}
for block_length in BLOCK_LENGTHS:
    peaks = {}
    failures = {}
    for lattice_size in (32, 64):
        temperatures, simulated = bootstrap_susceptibility(
            wolff[lattice_size], lattice_size, block_length, REPLICATES, rng
        )
        peak_samples = np.full(REPLICATES, np.nan)
        for replicate_index, replicate in enumerate(simulated):
            try:
                peak_samples[replicate_index] = five_point_peak(temperatures, replicate)[0]
            except ValueError:
                pass
        peaks[lattice_size] = peak_samples
        failures[lattice_size] = int(np.isnan(peak_samples).sum())
    valid = np.isfinite(peaks[32]) & np.isfinite(peaks[64])
    samples = 2.0 * peaks[64][valid] - peaks[32][valid]
    tc_bootstrap[block_length] = (
        float(samples.mean()),
        float(samples.std(ddof=1)),
        int(valid.sum()),
        failures[32],
        failures[64],
    )

wolff_tc_errors = np.asarray(
    [tc_bootstrap[block_length][1] for block_length in BLOCK_LENGTHS]
)
wolff_tc_stable = float(wolff_tc_errors.max() - wolff_tc_errors.min()) <= (
    0.1 * float(wolff_tc_errors.mean())
)

fig.tight_layout()
fig.savefig(EVIDENCE / "magnetization-compare.png", dpi=180)
plt.close(fig)

with (EVIDENCE / "sampler-comparison.txt").open("w") as output:
    output.write("T=2.3 sampler mean_abs_M block4000_se block2000_4000_8000_se stable\n")
    for name in ("Metropolis", "Wolff"):
        mean, error, errors = agreement[name]
        error_text = ",".join(f"{value:.6f}" for value in errors)
        output.write(
            f"{name} {mean:.6f} {error:.6f} {error_text} "
            f"{str(stable_by_sampler[name]).lower()}\n"
        )
    output.write(f"d {distance:.3f}\nstatus {status}\n")
    output.write(f"Wolff_Tpeak_L32 {wolff_peaks[32]:.6f}\n")
    output.write(f"Wolff_Tpeak_L64 {wolff_peaks[64]:.6f}\n")
    output.write(f"Wolff_Tc_extrapolated {wolff_tc:.6f}\n")
    output.write(
        f"Wolff_relative_deviation_percent "
        f"{100.0 * (wolff_tc - EXACT_TC) / EXACT_TC:.4f}\n"
    )
    output.write("block_length mean_Tc bootstrap_se valid failed_L32 failed_L64\n")
    for block_length in BLOCK_LENGTHS:
        output.write(
            f"{block_length} {tc_bootstrap[block_length][0]:.6f} "
            f"{tc_bootstrap[block_length][1]:.6f} {tc_bootstrap[block_length][2]} "
            f"{tc_bootstrap[block_length][3]} {tc_bootstrap[block_length][4]}\n"
        )
    output.write(
        f"Wolff_sampling_error "
        f"{'resolved' if wolff_tc_stable else 'unresolved'}\n"
    )

print((EVIDENCE / "sampler-comparison.txt").read_text(), end="")
