"""Render the pair field sampled by the Rust library, without reimplementing it."""

import argparse
import io
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=root / "field.png")
    args = parser.parse_args()
    sampled = subprocess.run(
        ["cargo", "run", "--quiet", "--release", "--manifest-path",
         str(root / "md" / "Cargo.toml"), "--example", "field"],
        check=True, text=True, stdout=subprocess.PIPE,
    )
    data = np.genfromtxt(io.StringIO(sampled.stdout), delimiter=",", names=True)
    n = len(np.unique(data["x"]))
    x, y, potential, fx, fy = (data[key].reshape(n, n)
                              for key in ["x", "y", "energy", "fx", "fy"])

    fig, ax = plt.subplots(figsize=(8.0, 7.2))
    fig.subplots_adjust(left=0.10, right=0.85, bottom=0.13, top=0.89)
    palette = LinearSegmentedColormap.from_list("pair_energy", ["#3585b7", "#f7f7f7", "#e97961"])
    colors = ax.pcolormesh(x, y, np.clip(potential, -1, 1),
                          cmap=palette, vmin=-1, vmax=1, shading="auto")
    selection = np.s_[6:-6:12, 6:-6:12]
    qx, qy, qfx, qfy = (array[selection] for array in [x, y, fx, fy])
    magnitude = np.hypot(qfx, qfy)
    visible = (np.hypot(qx, qy) > 0.35) & (magnitude > 0)
    # Compress lengths monotonically, preserving every force direction.
    factor = 0.19 * np.cbrt(magnitude[visible] / (1 + magnitude[visible])) / magnitude[visible]
    ax.quiver(qx[visible], qy[visible], qfx[visible] * factor, qfy[visible] * factor,
              angles="xy", scale_units="xy", scale=1, pivot="mid", color="#202020",
              width=0.003, headwidth=3.5, headlength=4.5)

    r0 = 2 ** (1 / 6)
    theta = np.linspace(0, 2 * np.pi, 500)
    ax.plot(r0 * np.cos(theta), r0 * np.sin(theta), "--", color="#111111",
            linewidth=1.4, label=r"$F=0$ at $r_0=2^{1/6}$")
    ax.scatter([0], [0], s=35, color="black", zorder=5, label="Fixed atom")
    ax.set(xlim=(-2.4, 2.4), ylim=(-2.4, 2.4), xlabel=r"$x / \sigma$",
           ylabel=r"$y / \sigma$", aspect="equal")
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    fig.suptitle("Lennard-Jones pair energy and force", fontsize=16, y=0.965)
    ax.set_title("Repulsive core and attractive energy well", fontsize=11, pad=10)
    colorbar = fig.colorbar(colors, ax=ax, fraction=0.047, pad=0.035)
    colorbar.set_label(r"Pair energy $U(r) / \varepsilon$", labelpad=10)
    colorbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    fig.text(0.5, 0.04, "Colors clipped to [-1, 1]; arrow lengths compressed. Values from md::pair.",
             ha="center", fontsize=9, color="#444444")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    plt.close(fig)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
