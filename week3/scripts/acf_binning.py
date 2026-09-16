#!/usr/bin/env python3
"""Show the autocorrelation and block-error growth at L=64, T=2.3."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (
    EVIDENCE,
    autocorrelation,
    block_standard_error,
    magnetizations,
    merged_metropolis,
    tau_int,
)


EVIDENCE.mkdir(exist_ok=True)
values = np.abs(magnetizations(merged_metropolis(64)[2.3]))
rho = autocorrelation(values)
correlation_time = tau_int(values)
block_lengths = np.asarray([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 4000, 5000])
errors = np.asarray([block_standard_error(values, int(length))[0] for length in block_lengths])

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
maximum_lag = min(4000, len(rho) - 1)
axes[0].plot(np.arange(maximum_lag + 1), rho[: maximum_lag + 1], color="C3")
axes[0].axhline(0, color="0.6", lw=0.8)
axes[0].axvline(correlation_time, color="k", ls="--", label=f"tau={correlation_time:.1f}")
axes[0].set(xlabel="lag t (sweeps)", ylabel="autocorrelation of |M|")
axes[0].legend()
axes[1].plot(block_lengths, errors, "o-", color="C0")
axes[1].set_xscale("log")
axes[1].set(xlabel="block length (sweeps)", ylabel="standard error of mean |M|")
axes[1].annotate(
    f"{len(values) // 5000} blocks remain",
    xy=(5000, errors[-1]),
    xytext=(-110, 24),
    textcoords="offset points",
    arrowprops={"arrowstyle": "->"},
)
fig.tight_layout()
fig.savefig(EVIDENCE / "acf-binning.png", dpi=180)
plt.close(fig)

trend = "unresolved" if errors[-1] > 1.05 * errors[-3] else "plateau candidate"
print(f"tau_int={correlation_time:.3f}; 5000-sweep block error={errors[-1]:.6g}; {trend}")
