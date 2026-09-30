#!/usr/bin/env python3
"""Draw the required four-panel Marmousi migration evidence."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
EXPERIMENT = json.loads((ROOT / "inputs" / "marmousi.json").read_text())


def main() -> None:
    background = np.asarray(EXPERIMENT["background"])
    perturbation = np.asarray(EXPERIMENT["perturbation"])
    born = np.load(ARTIFACTS / "marmousi-born" / "born_data.npy")
    image = np.load(ARTIFACTS / "marmousi-image" / "image.npy")
    result = json.loads((ARTIFACTS / "marmousi-image" / "result.json").read_text())
    statistics = result["statistics"]

    length_scale = EXPERIMENT["length_unit_m"] / 1000
    dx_km = EXPERIMENT["dx"] * length_scale
    time_scale = EXPERIMENT["time_unit_s"]
    extent = [0, (EXPERIMENT["nx"] - 1) * dx_km, (EXPERIMENT["nz"] - 1) * dx_km, 0]
    receivers = np.asarray(EXPERIMENT["receivers"])
    receiver_km = receivers[:, 0] * dx_km
    times = (np.arange(EXPERIMENT["steps"]) + 1) * EXPERIMENT["dt"] * time_scale
    source_x = np.asarray(EXPERIMENT["shots"])[:, 0] * dx_km
    shot = int(np.argmin(np.abs(source_x - 10.0)))

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.2))
    p0 = axes[0, 0].imshow(background, extent=extent, aspect="auto", cmap="viridis")
    axes[0, 0].set(title="Smoothed Marmousi background", xlabel="Horizontal position (km)", ylabel="Depth (km)")
    fig.colorbar(p0, ax=axes[0, 0], label="Speed (km/s)", pad=0.02)

    p1 = axes[1, 0].imshow(perturbation, extent=extent, aspect="auto", cmap="seismic",
                           vmin=-np.max(np.abs(perturbation)), vmax=np.max(np.abs(perturbation)))
    axes[1, 0].set(title="Short-wavelength perturbation", xlabel="Horizontal position (km)", ylabel="Depth (km)")
    fig.colorbar(p1, ax=axes[1, 0], label="Velocity perturbation (km/s)", pad=0.02)

    gather = born[shot]
    gather_limit = np.max(np.abs(gather))
    p2 = axes[0, 1].imshow(gather, aspect="auto", cmap="seismic", vmin=-gather_limit, vmax=gather_limit,
                           extent=[receiver_km[0], receiver_km[-1], times[-1], times[0]])
    axes[0, 1].set(title=f"Born gather; source x = {source_x[shot]:.1f} km",
                   xlabel="Receiver position (km)", ylabel="Time (s)")
    fig.colorbar(p2, ax=axes[0, 1], label="Scattered pressure (arbitrary units)", pad=0.02)

    image_limit = np.max(np.abs(image))
    p3 = axes[1, 1].imshow(image, extent=extent, aspect="auto", cmap="seismic", vmin=-image_limit, vmax=image_limit)
    axes[1, 1].set(title="Checkpointed migration image", xlabel="Horizontal position (km)", ylabel="Depth (km)")
    fig.colorbar(p3, ax=axes[1, 1], label="Adjoint image (arbitrary units)", pad=0.02)

    for ax in [axes[0, 0], axes[1, 0], axes[1, 1]]:
        ax.add_patch(Rectangle((8, 0), 6, 3, fill=False, color="black", ls="--", lw=1.1))
        ax.text(8.15, 0.35, "Comparison region", fontsize=8,
                bbox=dict(facecolor="white", alpha=0.65, edgecolor="none"))
    fig.suptitle("Marmousi Born modeling and six-state reverse-time migration")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(ARTIFACTS / "marmousi.png", dpi=180)
    plt.close(fig)

    image_norm = float(np.linalg.norm(image))
    print(f"Marmousi image L2 norm: {image_norm:.12e}")
    print(f"peak saved states: {statistics['peak_saved_states']}")
    print(f"peak saved bytes: {statistics['peak_saved_bytes']}")


if __name__ == "__main__":
    main()
