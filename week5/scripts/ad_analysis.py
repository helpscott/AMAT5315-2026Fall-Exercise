#!/usr/bin/env python3
"""Reproduce the Week 5 pair-energy AD checks and scaling evidence."""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "ad"
OUT.mkdir(parents=True, exist_ok=True)


def energy(r):
    a = r**-6
    b = a**2
    c = b - a
    return 4.0 * c


def analytic(r):
    return 24.0 * (r**-7 - 2.0 * r**-13)


def hand_forward(r):
    dr = jnp.ones_like(r)
    a, da = jax.jvp(lambda x: x**-6, (r,), (dr,))
    b, db = jax.jvp(lambda x: x**2, (a,), (da,))
    c, dc = jax.jvp(lambda x, y: x - y, (b, a), (db, da))
    value, dvalue = jax.jvp(lambda x: 4.0 * x, (c,), (dc,))
    return value, dvalue, {"r": dr, "a": da, "b": db, "c": dc, "U": dvalue}


def hand_reverse(r):
    a = r**-6
    b = a**2
    c = b - a
    value, pull_u = jax.vjp(lambda x: 4.0 * x, c)
    (bar_c,) = pull_u(jnp.ones_like(value))
    _, pull_c = jax.vjp(lambda x, y: x - y, b, a)
    bar_b, bar_a_direct = pull_c(bar_c)
    _, pull_b = jax.vjp(lambda x: x**2, a)
    (bar_a_square,) = pull_b(bar_b)
    bar_a = bar_a_direct + bar_a_square
    _, pull_a = jax.vjp(lambda x: x**-6, r)
    (bar_r,) = pull_a(bar_a)
    return value, bar_r, {
        "U": jnp.ones_like(value), "c": bar_c, "b": bar_b, "a": bar_a, "r": bar_r
    }


def graph_figure(fun, argument, path: Path, title: str) -> None:
    jaxpr = jax.make_jaxpr(fun)(argument).jaxpr
    equations = list(jaxpr.eqns)
    producers = {}
    labels = []
    for i, equation in enumerate(equations):
        labels.append(equation.primitive.name)
        for var in equation.outvars:
            producers[str(var)] = i
    edges = []
    for j, equation in enumerate(equations):
        for var in equation.invars:
            if str(var) in producers:
                edges.append((producers[str(var)], j))

    fig_width = max(6.4, 0.72 * len(equations))
    fig, ax = plt.subplots(figsize=(fig_width, 3.2))
    positions = {}
    levels = [0] * len(equations)
    for a, b in edges:
        levels[b] = max(levels[b], levels[a] + 1)
    counts = {}
    for i, level in enumerate(levels):
        row = counts.get(level, 0)
        counts[level] = row + 1
        positions[i] = (level, -row)
    max_rows = max(counts.values(), default=1)
    for i, (x, y) in positions.items():
        ax.text(
            x, y, labels[i], ha="center", va="center", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="#e8f2ff", ec="#356aa0"),
        )
    for a, b in edges:
        xa, ya = positions[a]
        xb, yb = positions[b]
        ax.annotate("", (xb - 0.15, yb), (xa + 0.15, ya), arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlim(-0.6, max(levels, default=0) + 0.6)
    ax.set_ylim(-max_rows + 0.25, 0.75)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def lattice(n: int) -> np.ndarray:
    side = int(np.ceil(n ** (1 / 3)))
    spacing = 2 ** (1 / 6)
    grid = np.stack(np.meshgrid(np.arange(side), np.arange(side), np.arange(side), indexing="ij"), -1)
    points = grid.reshape(-1, 3)[:n].astype(np.float64) * spacing
    return points


def cluster_energy(x):
    differences = x[:, None, :] - x[None, :, :]
    r2 = jnp.sum(differences * differences, axis=-1)
    mask = jnp.triu(jnp.ones(r2.shape, dtype=bool), 1)
    safe = jnp.where(mask, r2, 1.0)
    inv6 = safe**-3
    return jnp.sum(jnp.where(mask, 4.0 * (inv6 * inv6 - inv6), 0.0))


def analytic_cluster_gradient(x: np.ndarray) -> np.ndarray:
    d = x[:, None, :] - x[None, :, :]
    r2 = np.sum(d * d, axis=-1)
    mask = ~np.eye(len(x), dtype=bool)
    safe = np.where(mask, r2, 1.0)
    r = np.sqrt(safe)
    du = 24.0 * (r**-7 - 2.0 * r**-13)
    terms = np.where(mask[:, :, None], du[:, :, None] * d / r[:, :, None], 0.0)
    return np.sum(terms, axis=1)


def timed(callable_, repeats=5):
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = callable_()
        jax.block_until_ready(result)
        values.append(time.perf_counter() - start)
    return min(values)


def scaling() -> tuple[list[int], list[float], list[float], float, float]:
    sizes = [64, 128, 256, 512, 1024]
    reverse_ratios, forward_ratios = [], []
    reverse_error = 0.0
    forward_error = 0.0
    for n in sizes:
        rng = np.random.default_rng(20260930 + n)
        x_np = lattice(n) + rng.normal(0.0, 0.05, (n, 3))
        x = jnp.asarray(x_np)
        e_fun = jax.jit(cluster_energy)
        g_fun = jax.jit(jax.grad(cluster_energy))
        direction = jnp.zeros_like(x).at[0, 0].set(1.0)
        jvp_fun = jax.jit(lambda y, v: jax.jvp(cluster_energy, (y,), (v,))[1])
        jax.block_until_ready(e_fun(x))
        gradient = g_fun(x)
        jax.block_until_ready(gradient)
        jax.block_until_ready(jvp_fun(x, direction))
        energy_time = timed(lambda: e_fun(x))
        reverse_time = timed(lambda: g_fun(x))
        one_direction = timed(lambda: jvp_fun(x, direction), repeats=7)
        reverse_ratios.append(reverse_time / energy_time)
        forward_ratios.append((3 * n) * one_direction / energy_time)
        analytic_g = analytic_cluster_gradient(x_np)
        scale = max(np.max(np.abs(analytic_g)), 1e-300)
        reverse_error = max(reverse_error, float(np.max(np.abs(np.asarray(gradient) - analytic_g)) / scale))
        if n == sizes[0]:
            flat = x.reshape(-1)
            forward_g = []
            for k in range(flat.size):
                basis = jnp.zeros_like(flat).at[k].set(1.0).reshape(x.shape)
                forward_g.append(jvp_fun(x, basis))
            forward_g = np.asarray(forward_g).reshape(x.shape)
            forward_error = float(np.max(np.abs(forward_g - analytic_g)) / scale)
        print(f"N={n:4d} inputs={3*n:4d} reverse/energy={reverse_ratios[-1]:.3f} "
              f"forward/energy={forward_ratios[-1]:.3f}")
    return sizes, forward_ratios, reverse_ratios, forward_error, reverse_error


def main() -> None:
    point = jnp.asarray(1.3)
    value, tangent, tangents = hand_forward(point)
    _, adjoint, adjoints = hand_reverse(point)
    jax_grad = jax.grad(energy)(point)
    payload = {
        "r": float(point),
        "energy": float(value),
        "tangents": {key: float(item) for key, item in tangents.items()},
        "adjoints": {key: float(item) for key, item in adjoints.items()},
        "jax_grad": float(jax_grad),
    }
    (OUT / "derivatives.json").write_text(json.dumps(payload, indent=2) + "\n")

    r = jnp.linspace(0.95, 2.5, 601)
    _, forward_values, _ = hand_forward(r)
    _, reverse_values, _ = hand_reverse(r)
    exact = analytic(r)
    finite = (energy(r + 1e-6) - energy(r - 1e-6)) / (2e-6)
    errors = {
        "forward": float(jnp.max(jnp.abs(forward_values - exact))),
        "reverse": float(jnp.max(jnp.abs(reverse_values - exact))),
        "finite_difference": float(jnp.max(jnp.abs(finite - exact))),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(r, exact, color="black", lw=2, label="analytic")
    axes[0].plot(r, forward_values, "--", label="forward AD")
    axes[0].plot(r, reverse_values, ":", label="reverse AD")
    axes[0].set(xlabel="Separation r", ylabel="dU/dr", title="Lennard-Jones derivative")
    axes[0].legend()
    axes[1].semilogy(r, jnp.maximum(jnp.abs(finite - exact), 1e-18), label="finite difference")
    axes[1].semilogy(r, jnp.maximum(jnp.abs(forward_values - exact), 1e-18), label="forward AD")
    axes[1].semilogy(r, jnp.maximum(jnp.abs(reverse_values - exact), 1e-18), label="reverse AD")
    axes[1].set(xlabel="Separation r", ylabel="Absolute error", title="Error against analytic derivative")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "modes.png", dpi=180)
    plt.close(fig)
    print("maximum derivative errors:", errors)

    graph_figure(energy, point, OUT / "graph.png", "JAX primal graph: U(r)")
    graph_figure(jax.grad(energy), point, OUT / "grad-graph.png", "JAX reverse graph: dU/dr")

    sizes, forward_ratios, reverse_ratios, forward_error, reverse_error = scaling()
    inputs = np.asarray(sizes) * 3
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.loglog(inputs, forward_ratios, "o-", label="forward mode, one JVP per input")
    ax.loglog(inputs, reverse_ratios, "s-", label="reverse mode, one VJP")
    guide = forward_ratios[0] * inputs / inputs[0]
    ax.loglog(inputs, guide, "k--", alpha=0.6, label="proportional to input count")
    ax.set(xlabel="Inputs P = 3N", ylabel="Gradient time / energy time", title="Gradient of a Lennard-Jones cluster energy")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "scaling.png", dpi=180)
    plt.close(fig)
    print(f"largest relative force error: forward={forward_error:.3e}, reverse={reverse_error:.3e}")


if __name__ == "__main__":
    main()
