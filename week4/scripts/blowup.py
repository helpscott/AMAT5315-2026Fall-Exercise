#!/usr/bin/env python3
"""Plot energy on both sides of the measured stability limits."""

import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import ARTIFACTS, EVIDENCE, table


EVIDENCE.mkdir(exist_ok=True)
scan = ARTIFACTS / "scan"
report = json.loads((scan / "scan.json").read_text())


def curve(name: str) -> np.ndarray:
    return table(scan / f"{name}.tsv")


fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
for dt, style in ((0.032, "-"), (0.033, "--")):
    values = curve(f"taylor-green-rk4-dt{dt:g}")
    finite = np.isfinite(values[:, 1])
    axes[0].semilogy(values[finite, 0], values[finite, 1], style, label=f"RK4 dt={dt:g}")
    if not np.all(finite):
        stop = values[np.where(~finite)[0][0], 0]
        axes[0].axvline(stop, color="C3", ls=":", alpha=0.7)
        axes[0].text(
            stop - 0.06,
            0.014,
            f"non-finite t={stop:.2f}",
            rotation=90,
            va="bottom",
            ha="right",
            color="C3",
        )
time = np.linspace(0, 8, 300)
axes[0].semilogy(time, 0.25 * np.exp(-0.4 * time), "k:", label="exact energy")
axes[0].set(
    xlabel="time t",
    ylabel="energy E(t)",
    title=f"Taylor-Green; predicted dtcrit={report['predicted_diffusive_limit']:.4f}",
)
axes[0].legend(fontsize=8)

stable_dt = report["random_stable_dt"]
unstable_dt = report["random_unstable_dt"]
for dt, style, label in (
    (stable_dt, "-", f"RK4 dt={stable_dt:g}, reaches t=10"),
    (unstable_dt, "--", f"RK4 dt={unstable_dt:g}, unstable"),
):
    values = curve(f"random-rk4-dt{dt:g}")
    finite = np.isfinite(values[:, 1])
    axes[1].semilogy(values[finite, 0], values[finite, 1], style, label=label)
    if not np.all(finite):
        stop = values[np.where(~finite)[0][0], 0]
        axes[1].axvline(stop, color="C3", ls=":", alpha=0.7)
        axes[1].text(
            stop + 0.04,
            0.30,
            f"RK4 non-finite t={stop:.2f}",
            rotation=90,
            va="bottom",
            color="C3",
        )
euler = curve("random-euler-dt0.01")
finite = np.isfinite(euler[:, 1])
axes[1].semilogy(euler[finite, 0], euler[finite, 1], ":", lw=2, label="Euler dt=0.01")
if not np.all(finite):
    stop = euler[np.where(~finite)[0][0], 0]
    axes[1].axvline(stop, color="C2", ls=":", alpha=0.7)
    axes[1].text(
        stop + 0.04,
        0.30,
        f"Euler non-finite t={stop:.2f}",
        rotation=90,
        va="bottom",
        color="C2",
    )
axes[1].set(
    xlabel="time t",
    ylabel="energy E(t)",
    title=f"Random flow; advective bound={report['predicted_advective_bound']:.4f}",
)
axes[1].legend(fontsize=8)
for axis in axes:
    axis.grid(True, which="both", alpha=0.25)
fig.tight_layout()
fig.savefig(EVIDENCE / "blowup.png", dpi=180)
plt.close(fig)

print(json.dumps(report, indent=2))
