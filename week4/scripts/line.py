#!/usr/bin/env python3
"""Generate both line-equation evidence figures from the Rust library."""

from __future__ import annotations

import json
import subprocess

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

from common import ARTIFACTS, EVIDENCE, ROOT


ARTIFACTS.mkdir(exist_ok=True)
EVIDENCE.mkdir(exist_ok=True)
result = subprocess.run(
    ["cargo", "run", "--release", "--quiet", "--example", "line_study"],
    cwd=ROOT,
    check=True,
    capture_output=True,
    text=True,
)
data = json.loads(result.stdout)
(ARTIFACTS / "line-study.json").write_text(json.dumps(data))

stability = data["stability"]
real_axis = np.asarray(stability["real"])
imag_axis = np.asarray(stability["imag"])
growth = np.asarray(stability["growth"]).reshape(len(imag_axis), len(real_axis))
real_grid, imag_grid = np.meshgrid(real_axis, imag_axis)
z = real_grid + 1j * imag_grid
functions = {
    "Euler": 1 + z,
    "midpoint": 1 + z + z**2 / 2,
    "RK4": 1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24,
}

fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.7), gridspec_kw={"width_ratios": [1.08, 1, 1]})
image = axes[0].pcolormesh(
    real_axis,
    imag_axis,
    np.maximum(growth, 1e-3),
    shading="auto",
    cmap="viridis",
    norm=LogNorm(vmin=1e-2, vmax=1e2),
)
styles = {"Euler": "--", "midpoint": ":", "RK4": "-"}
for name, values in functions.items():
    axes[0].contour(
        real_axis,
        imag_axis,
        np.abs(values),
        levels=[1],
        colors="black",
        linestyles=styles[name],
        linewidths=1.5,
    )
for key, marker, label in (
    ("modes_0045", "o", "line modes, dt=0.045"),
    ("modes_0056", "x", "line modes, dt=0.056"),
):
    modes = np.asarray(stability[key])
    axes[0].scatter(modes[:, 0], modes[:, 1], s=13, marker=marker, label=label)
axes[0].set(
    xlabel="Re(z)",
    ylabel="Im(z)",
    title="Measured RK4 growth and stability boundaries",
    xlim=(-4, 1),
    ylim=(-4, 4),
)
handles, labels = axes[0].get_legend_handles_labels()
handles = [
    Line2D([0], [0], color="black", ls=styles[name], label=f"{name}: |R|=1")
    for name in functions
] + handles
labels = [f"{name}: |R|=1" for name in functions] + labels
axes[0].legend(handles, labels, fontsize=7, loc="upper left")
fig.colorbar(image, ax=axes[0], label="measured growth per step")

for axis, prefix, title in (
    (axes[1], "stable", "RK4 dt=0.045: stable"),
    (axes[2], "unstable", "RK4 dt=0.056: unstable"),
):
    frames = np.asarray(stability[f"{prefix}_frames"])
    times = np.asarray(stability[f"{prefix}_times"])
    axis.imshow(
        frames,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        extent=(0, 2 * np.pi, times[-1], times[0]),
    )
    axis.set(xlabel="x", ylabel="time t", title=title)
    axis.set_xticks([0, np.pi, 2 * np.pi], ["0", "pi", "2pi"])
fig.tight_layout()
fig.savefig(EVIDENCE / "line-stability.png", dpi=180)
plt.close(fig)

accuracy = data["accuracy"]
x = np.asarray(accuracy["x"])
fig, axes = plt.subplots(1, 2, figsize=(11.7, 4.6))
axes[0].plot(x, accuracy["exact"], "k-", lw=2, label="exact")
axes[0].plot(x, accuracy["rk4_fourier"], "--", label="RK4, Fourier, dt=0.02")
axes[0].plot(x, accuracy["rk4_centered"], "-.", label="RK4, centered, dt=0.02")
axes[0].plot(x, accuracy["euler_fourier"], ":", lw=2, label="Euler, Fourier, dt=0.005")
axes[0].set(xlabel="x", ylabel="u(x, 2pi)", title="One-lap pulse propagation")
axes[0].set_xticks([0, np.pi, 2 * np.pi], ["0", "pi", "2pi"])
axes[0].legend(fontsize=8)

steps = np.asarray(accuracy["steps"])
series = {
    "Euler": np.asarray(accuracy["euler_errors"]),
    "midpoint": np.asarray(accuracy["midpoint_errors"]),
    "RK4": np.asarray(accuracy["rk4_errors"]),
    "equal-weight RK4": np.asarray(accuracy["equal_weight_errors"]),
}
slopes = {}
for name, errors in series.items():
    slope, intercept = np.polyfit(np.log(steps), np.log(errors), 1)
    slopes[name] = float(slope)
    axes[1].loglog(steps, errors, "o-", label=f"{name}, slope={slope:.2f}")
axes[1].set(xlabel="time step dt", ylabel="maximum error at t=1", title="Temporal convergence")
axes[1].legend(fontsize=8)
axes[1].grid(True, which="both", alpha=0.25)
fig.tight_layout()
fig.savefig(EVIDENCE / "line-accuracy.png", dpi=180)
plt.close(fig)

print("profile maximum errors")
for name, error in accuracy["profile_errors"].items():
    print(f"{name} {error:.12e}")
print("fitted slopes")
for name, slope in slopes.items():
    print(f"{name} {slope:.6f}")
