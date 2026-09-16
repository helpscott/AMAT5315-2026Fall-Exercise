#!/usr/bin/env python3
"""Compare autocorrelation times in spin-update work units."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import EVIDENCE, EXACT_TC, load_run, magnetizations, tau_int


EVIDENCE.mkdir(exist_ok=True)
metropolis = load_run("window-l64")
wolff = load_run("wolff-l64")

curves = {}
details = {}
for name, grouped in (("Metropolis", metropolis), ("Wolff", wolff)):
    temperatures = np.asarray(list(grouped), dtype=float)
    own_step_tau = []
    work_tau = []
    mean_clusters = []
    for records_at_temperature in grouped.values():
        correlation_time = tau_int(np.abs(magnetizations(records_at_temperature)))
        own_step_tau.append(correlation_time)
        if name == "Wolff":
            mean_cluster = float(np.mean([record["cluster_size"] for record in records_at_temperature]))
            mean_clusters.append(mean_cluster)
            work_tau.append(correlation_time * mean_cluster / (64**2))
        else:
            mean_clusters.append(float("nan"))
            work_tau.append(correlation_time)
    curves[name] = (temperatures, np.asarray(work_tau))
    details[name] = (np.asarray(own_step_tau), np.asarray(mean_clusters))

fig, axis = plt.subplots(figsize=(7.2, 4.6))
for name, color in (("Metropolis", "C0"), ("Wolff", "C3")):
    axis.plot(*curves[name], "o-", ms=4, color=color, label=name)
axis.axvline(EXACT_TC, color="k", ls="--", label=f"exact Tc={EXACT_TC:.5f}")
axis.set_yscale("log")
axis.set(xlabel="temperature T", ylabel="work-normalized tau, L=64")
axis.legend()
fig.tight_layout()
fig.savefig(EVIDENCE / "tau-compare.png", dpi=180)
plt.close(fig)

index = int(np.where(np.isclose(curves["Metropolis"][0], 2.3))[0][0])
metropolis_work = float(curves["Metropolis"][1][index])
wolff_work = float(curves["Wolff"][1][index])
ratio = metropolis_work / wolff_work
with (EVIDENCE / "work-comparison.txt").open("w") as output:
    output.write("T 2.300000\n")
    output.write(f"metropolis_tau_sweeps {details['Metropolis'][0][index]:.6f}\n")
    output.write(f"wolff_tau_moves {details['Wolff'][0][index]:.6f}\n")
    output.write(f"wolff_mean_cluster_size {details['Wolff'][1][index]:.6f}\n")
    output.write(f"wolff_tau_work {wolff_work:.6f}\n")
    output.write(f"metropolis_to_wolff_work_ratio {ratio:.2f}\n")

print((EVIDENCE / "work-comparison.txt").read_text(), end="")
