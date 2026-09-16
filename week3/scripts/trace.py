#!/usr/bin/env python3
"""Plot the first 2000 measured sweeps at critical and hot temperatures."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import EVIDENCE, magnetizations, merged_metropolis


EVIDENCE.mkdir(exist_ok=True)
series = merged_metropolis(64)
fig, axes = plt.subplots(2, 1, figsize=(8, 5.8), sharex=True, sharey=True)
for axis, temperature, color in zip(axes, (2.3, 3.0), ("C3", "C0")):
    values = np.abs(magnetizations(series[temperature])[:2000])
    axis.plot(np.arange(1, len(values) + 1), values, lw=0.9, color=color)
    axis.axhline(values.mean(), color="k", ls="--", lw=1, label=f"mean={values.mean():.3f}")
    axis.set_ylabel(f"|M|, T={temperature:.1f}")
    axis.legend(loc="upper right")
axes[-1].set_xlabel("measurement sweep")
fig.tight_layout()
fig.savefig(EVIDENCE / "trace.png", dpi=180)
plt.close(fig)
