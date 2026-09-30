#!/usr/bin/env python3
"""Plot the fixed reflector experiment and forward-model evidence."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "inputs" / "reflector.json"
OUT = ROOT / "artifacts"
FORWARD = OUT / "forward"


def extent_km(experiment):
    scale = experiment["length_unit_m"] / 1000
    return [0, (experiment["nx"] - 1) * experiment["dx"] * scale,
            (experiment["nz"] - 1) * experiment["dx"] * scale, 0]


def main() -> None:
    experiment = json.loads(INPUT.read_text())
    background = np.asarray(experiment["background"])
    perturbation = np.asarray(experiment["perturbation"])
    shots = np.asarray(experiment["shots"])
    receivers = np.asarray(experiment["receivers"])
    length_scale = experiment["length_unit_m"] / 1000
    time_scale = experiment["time_unit_s"]

    steps = np.arange(experiment["steps"])
    reduced_time = steps * experiment["dt"]
    theta = np.pi * experiment["source_frequency"] * (reduced_time - experiment["source_peak_time"])
    pulse = (1 - 2 * theta**2) * np.exp(-theta**2)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    image = axes[0].imshow(background + perturbation, extent=extent_km(experiment), cmap="viridis", aspect="equal")
    axes[0].scatter(receivers[:, 0] * length_scale, receivers[:, 1] * length_scale,
                    marker="v", s=28, color="#00d9d9", edgecolor="black", linewidth=0.4, label="Receivers")
    axes[0].scatter(shots[:, 0] * length_scale, shots[:, 1] * length_scale,
                    marker="*", s=80, color="#ff3030", edgecolor="black", linewidth=0.4, label="Sources")
    axes[0].axhline(2.1, color="white", lw=1.3, ls="--", label="Thin reflector")
    axes[0].set(xlabel="Horizontal position (km)", ylabel="Depth (km)", title="Seismic acquisition")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.colorbar(image, ax=axes[0], label="Speed (km/s)")
    axes[1].plot(reduced_time * time_scale, pulse, lw=2)
    axes[1].axvline(experiment["source_peak_time"] * time_scale, color="black", ls="--", lw=1)
    axes[1].set(xlabel="Time (s)", ylabel="Source pulse g(t)",
                title="Ricker pulse; peak frequency 0.8 Hz, peak time 1.5 s")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "inputs.png", dpi=180)
    plt.close(fig)

    traces = np.load(FORWARD / "traces.npy")
    receiver_km = receivers[:, 0] * length_scale
    trace_times = (np.arange(experiment["steps"]) + 1) * experiment["dt"] * time_scale
    limit = np.max(np.abs(traces))
    fig, axes = plt.subplots(1, len(shots), figsize=(12, 4.7), sharey=True)
    for shot, ax in enumerate(axes):
        panel = ax.imshow(
            traces[shot], aspect="auto", cmap="seismic", vmin=-limit, vmax=limit,
            extent=[receiver_km[0], receiver_km[-1], trace_times[-1], trace_times[0]],
        )
        ax.set_title(f"Shot {shot}; source x = {shots[shot,0] * length_scale:.1f} km")
        ax.set_xlabel("Receiver position (km)")
    axes[0].set_ylabel("Time (s)")
    fig.colorbar(panel, ax=axes, label="Pressure (arbitrary units)", shrink=0.85, pad=0.02)
    fig.suptitle("Forward receiver gathers", y=0.99)
    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.12, top=0.88, wspace=0.12)
    fig.savefig(FORWARD / "gathers.png", dpi=180)
    plt.close(fig)

    run = json.loads((FORWARD / "run.json").read_text())
    frame_steps = run["recording"]["steps"]
    index = frame_steps.index(150)
    wavefield = np.load(FORWARD / "wavefield.npy")[index]
    echo = np.load(FORWARD / "echo.npy")[index]
    for field, name, title in [
        (wavefield, "wavefield.png", "Forward wavefield; shot 0, t = 3.00 s"),
        (echo, "echo.png", "Reflector echo; shot 0, t = 3.00 s"),
    ]:
        vmax = np.max(np.abs(field))
        fig, ax = plt.subplots(figsize=(6.2, 5.1))
        panel = ax.imshow(field, extent=extent_km(experiment), cmap="seismic", vmin=-vmax, vmax=vmax, aspect="equal")
        ax.scatter(receivers[:, 0] * length_scale, receivers[:, 1] * length_scale,
                   marker="v", s=18, color="#00d9d9", edgecolor="black", linewidth=0.3)
        ax.scatter(shots[0, 0] * length_scale, shots[0, 1] * length_scale,
                   marker="*", s=70, color="#ff3030", edgecolor="black", linewidth=0.4)
        ax.axhline(2.1, color="black", lw=1, ls="--")
        ax.set(xlabel="Horizontal position (km)", ylabel="Depth (km)", title=title)
        fig.colorbar(panel, ax=ax, label="Pressure (arbitrary units)")
        fig.tight_layout()
        fig.savefig(FORWARD / name, dpi=180)
        plt.close(fig)

    print(f"all-trace L2 norm: {np.linalg.norm(traces):.12f}")
    for shot in range(len(shots)):
        flat_index = np.argmax(np.abs(traces[shot]))
        step, receiver = np.unravel_index(flat_index, traces[shot].shape)
        print(f"shot {shot}: max={traces[shot, step, receiver]:.12f}, trace_index={step}, receiver={receiver}")
    print(f"step 150: direct max={np.max(np.abs(wavefield)):.9f}, echo max={np.max(np.abs(echo)):.9f}, "
          f"ratio={np.max(np.abs(echo))/np.max(np.abs(wavefield)):.4%}")


if __name__ == "__main__":
    main()
