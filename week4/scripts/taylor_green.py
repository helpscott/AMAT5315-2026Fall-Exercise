#!/usr/bin/env python3
"""Compare the Taylor-Green recording with its exact final field."""

import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import ARTIFACTS, EVIDENCE, records, relative_l2


EVIDENCE.mkdir(exist_ok=True)
frames = records(ARTIFACTS / "taylor-green/fields.jsonl")
exact = json.loads((ARTIFACTS / "taylor-green/exact-t1.json").read_text())
final = frames[-1]
computed_velocity = np.concatenate([np.asarray(final["u"]), np.asarray(final["v"])])
exact_velocity = np.concatenate([np.asarray(exact["u"]), np.asarray(exact["v"])])
error = relative_l2(computed_velocity, exact_velocity)
print(f"relative_velocity_error {error:.12e}")

n = 64
coordinates = np.arange(n) * 2 * np.pi / n
maximum = max(np.max(np.abs(np.asarray(frame["omega"]))) for frame in (frames[0], frames[-1]))
fig, axes = plt.subplots(1, 2, figsize=(10.7, 4.6), sharex=True, sharey=True)
for axis, frame in zip(axes, (frames[0], frames[-1])):
    omega = np.asarray(frame["omega"]).reshape(n, n)
    u = np.asarray(frame["u"]).reshape(n, n)
    v = np.asarray(frame["v"]).reshape(n, n)
    image = axis.imshow(
        omega,
        origin="lower",
        extent=(0, 2 * np.pi, 0, 2 * np.pi),
        cmap="RdBu_r",
        vmin=-maximum,
        vmax=maximum,
    )
    stride = 5
    axis.quiver(
        coordinates[::stride],
        coordinates[::stride],
        u[::stride, ::stride],
        v[::stride, ::stride],
        color="black",
        alpha=0.7,
        scale=12,
    )
    axis.set(title=f"t={frame['t']:.1f}, max |omega|={np.max(np.abs(omega)):.3f}", xlabel="x", ylabel="y")
    axis.set_xticks([0, np.pi, 2 * np.pi], ["0", "pi", "2pi"])
    axis.set_yticks([0, np.pi, 2 * np.pi], ["0", "pi", "2pi"])
color_axis = fig.add_axes([0.91, 0.16, 0.018, 0.67])
fig.colorbar(image, cax=color_axis, label="vorticity omega")
fig.suptitle(f"Taylor-Green decay, nu=0.1; relative velocity error={error:.2e}")
fig.subplots_adjust(left=0.07, right=0.88, bottom=0.12, top=0.84, wspace=0.22)
fig.savefig(EVIDENCE / "taylor-green.png", dpi=180)
plt.close(fig)
