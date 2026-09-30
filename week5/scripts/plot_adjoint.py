#!/usr/bin/env python3
"""Validate the Born/adjoint pair and draw reflector-migration evidence."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = json.loads((ROOT / "inputs" / "reflector.json").read_text())
BORN = ROOT / "artifacts" / "born"
ADJOINT = ROOT / "artifacts" / "adjoint"


def main() -> None:
    perturbation = np.asarray(EXPERIMENT["perturbation"])
    born = np.load(BORN / "born_data.npy")
    image = np.load(ADJOINT / "image.npy")
    left = float(np.sum(born * born))
    right = float(np.sum(perturbation * image))
    relative = abs(left - right) / max(abs(left), abs(right))

    x_slice = slice(7, 34)
    z_slice = slice(10, 34)
    cropped_m = perturbation[z_slice, x_slice]
    cropped_image = image[z_slice, x_slice]
    length_scale = EXPERIMENT["length_unit_m"] / 1000
    dx_km = EXPERIMENT["dx"] * length_scale
    x = np.arange(7, 34) * dx_km
    z = np.arange(10, 34) * dx_km
    profile = np.linalg.norm(cropped_image, axis=1)
    peak_index = int(np.argmax(profile)) + 10
    peak_depth = peak_index * dx_km
    true_depth = 2.1

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 1, 0.75]})
    extent = [x[0], x[-1], z[-1], z[0]]
    p0 = axes[0].imshow(cropped_m, extent=extent, aspect="auto", cmap="seismic",
                        vmin=-np.max(np.abs(cropped_m)), vmax=np.max(np.abs(cropped_m)))
    axes[0].axhline(true_depth, color="black", ls="--", lw=1)
    axes[0].set(title="1. Known reflector", xlabel="Horizontal position (km)", ylabel="Depth (km)")
    fig.colorbar(p0, ax=axes[0], orientation="horizontal", pad=0.14, label="Velocity change (km/s)")
    p1 = axes[1].imshow(cropped_image, extent=extent, aspect="auto", cmap="seismic",
                        vmin=-np.max(np.abs(cropped_image)), vmax=np.max(np.abs(cropped_image)))
    axes[1].axhline(true_depth, color="black", ls="--", lw=1)
    axes[1].set(title="2. Raw signed RTM image", xlabel="Horizontal position (km)")
    fig.colorbar(p1, ax=axes[1], orientation="horizontal", pad=0.14, label="Image (arbitrary units)")
    axes[2].plot(profile, z, lw=2)
    axes[2].axhline(true_depth, color="black", ls="--", lw=1, label="Known: 2.1 km")
    axes[2].axhline(peak_depth, color="#d62728", ls=":", lw=1.4, label=f"Peak: {peak_depth:.1f} km")
    axes[2].invert_yaxis()
    axes[2].set(title="3. Depth profile", xlabel="Row L2 norm")
    axes[2].legend(fontsize=8)
    fig.suptitle(f"RTM locates the reflector; depth error = {abs(peak_depth-true_depth):.1f} km")
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.22, top=0.84, wspace=0.28)
    fig.savefig(ADJOINT / "image.png", dpi=180)
    plt.close(fig)

    run = json.loads((ADJOINT / "run.json").read_text())
    frame_steps = run["recording"]["steps"]
    frame = np.load(ADJOINT / "wavefield.npy")[frame_steps.index(132)]
    vmax = np.max(np.abs(frame))
    full_extent = [0, (EXPERIMENT["nx"] - 1) * dx_km, (EXPERIMENT["nz"] - 1) * dx_km, 0]
    fig, ax = plt.subplots(figsize=(6.2, 5.1))
    panel = ax.imshow(frame, extent=full_extent, cmap="seismic", vmin=-vmax, vmax=vmax, aspect="equal")
    ax.axhline(true_depth, color="black", ls="--", lw=1, label="Reflector")
    ax.set(xlabel="Horizontal position (km)", ylabel="Depth (km)", title="Adjoint field at step 132; 2.64 s")
    ax.legend(loc="lower left")
    fig.colorbar(panel, ax=ax, label="Adjoint pressure (arbitrary units)")
    fig.tight_layout()
    fig.savefig(ADJOINT / "wavefield.png", dpi=180)
    plt.close(fig)

    print(f"transpose left  = {left:.16e}")
    print(f"transpose right = {right:.16e}")
    print(f"relative difference = {relative:.3e}")
    print(f"true depth = {true_depth:.1f} km; image peak = {peak_depth:.1f} km; difference = {abs(peak_depth-true_depth):.1f} km")


if __name__ == "__main__":
    main()
