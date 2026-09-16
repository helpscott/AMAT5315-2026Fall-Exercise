#!/usr/bin/env python3
"""Plot Metropolis autocorrelation time across the full temperature range."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import EVIDENCE, EXACT_TC, magnetizations, merged_metropolis, tau_int


EVIDENCE.mkdir(exist_ok=True)
fig, axis = plt.subplots(figsize=(7.2, 4.6))
for lattice_size, color in ((32, "C0"), (64, "C1")):
    grouped = merged_metropolis(lattice_size)
    temperatures = np.asarray(list(grouped))
    times = np.asarray(
        [tau_int(np.abs(magnetizations(records))) for records in grouped.values()]
    )
    axis.plot(temperatures, times, "o-", ms=4, color=color, label=f"L={lattice_size}")
axis.axvline(EXACT_TC, color="k", ls="--", label=f"exact Tc={EXACT_TC:.5f}")
axis.set_yscale("log")
axis.set(xlabel="temperature T", ylabel="integrated autocorrelation time (sweeps)")
axis.legend()
fig.tight_layout()
fig.savefig(EVIDENCE / "tau.png", dpi=180)
plt.close(fig)
