#!/usr/bin/env python3
"""Plot the seeded random flow at four requested times."""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import ARTIFACTS, EVIDENCE, records, table


EVIDENCE.mkdir(exist_ok=True)
frames = records(ARTIFACTS / "random/fields.jsonl")
diagnostics = table(ARTIFACTS / "random.tsv")
selected = []
for target in (0.0, 2.0, 5.0, 10.0):
    selected.append(min(frames, key=lambda frame: abs(frame["t"] - target)))
maximum = max(float(np.max(np.abs(frame["omega"]))) for frame in selected)

fig, axes = plt.subplots(1, 4, figsize=(15.3, 4.0), sharex=True, sharey=True)
for axis, frame in zip(axes, selected):
    omega = np.asarray(frame["omega"]).reshape(128, 128)
    row = diagnostics[np.argmin(np.abs(diagnostics[:, 0] - frame["t"]))]
    image = axis.imshow(
        omega,
        origin="lower",
        extent=(0, 2 * np.pi, 0, 2 * np.pi),
        cmap="RdBu_r",
        vmin=-maximum,
        vmax=maximum,
    )
    axis.set_title(f"t={frame['t']:.0f}\nE={row[1]:.3f}, Z={row[2]:.3f}")
    axis.set_xlabel("x")
    axis.set_xticks([0, np.pi, 2 * np.pi], ["0", "pi", "2pi"])
axes[0].set_ylabel("y")
axes[0].set_yticks([0, np.pi, 2 * np.pi], ["0", "pi", "2pi"])
color_axis = fig.add_axes([0.92, 0.17, 0.012, 0.62])
fig.colorbar(image, cax=color_axis, label="vorticity omega")
fig.suptitle("Random flow: filaments fade before large vortices")
fig.subplots_adjust(left=0.05, right=0.90, bottom=0.13, top=0.82, wspace=0.08)
fig.savefig(EVIDENCE / "random.png", dpi=180)
plt.close(fig)

print(f"initial_energy {diagnostics[0, 1]:.6f}")
print(f"final_energy {diagnostics[-1, 1]:.6f}")
print(f"initial_enstrophy {diagnostics[0, 2]:.6f}")
print(f"final_enstrophy {diagnostics[-1, 2]:.6f}")
print(f"enstrophy_decay_factor {diagnostics[0, 2] / diagnostics[-1, 2]:.6f}")
