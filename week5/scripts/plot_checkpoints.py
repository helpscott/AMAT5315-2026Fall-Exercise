#!/usr/bin/env python3
"""Audit Treeverse schedules and plot their storage/work trade-off."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
BUDGETS = [1, 3, 5, 10]


def audit(actions, budget, steps=240):
    stored = {0}
    invalid_restores = 0
    budget_overruns = 0
    grads = []
    for item in actions:
        action = item["action"]
        step = item["step"]
        if action == "restore" and step not in stored:
            invalid_restores += 1
        elif action == "store":
            stored.add(step)
            if len(stored) > budget + 1:
                budget_overruns += 1
        elif action == "fetch":
            stored.discard(step)
        elif action == "grad":
            grads.append(step)
    expected = list(range(steps - 1, -1, -1))
    grad_mismatches = sum(a != b for a, b in zip(grads, expected)) + abs(len(grads) - len(expected))
    return grad_mismatches, invalid_restores, budget_overruns


def main() -> None:
    full_image = np.load(ARTIFACTS / "adjoint" / "image.npy")
    calls, peaks, bytes_ = [], [], []
    for budget in BUDGETS:
        folder = ARTIFACTS / f"checkpoint-{budget}"
        result = json.loads((folder / "result.json").read_text())
        statistics = result["statistics"]
        image = np.load(folder / "image.npy")
        relative = np.linalg.norm(image - full_image) / np.linalg.norm(full_image)
        calls.append(statistics["per_shot"][0]["scheduler_forward_calls"])
        peaks.append(statistics["peak_saved_states"])
        bytes_.append(statistics["peak_saved_bytes"])
        totals = np.zeros(3, dtype=int)
        for shot in range(3):
            log = json.loads((folder / f"actions-{shot}.json").read_text())
            totals += np.asarray(audit(log["actions"], budget))
        print(f"budget {budget:2d}: relative image error={relative:.3e}, peak states={peaks[-1]}, "
              f"forward calls/shot={calls[-1]}, audits={tuple(totals)}")

    log = json.loads((ARTIFACTS / "checkpoint-5" / "actions-0.json").read_text())
    colors = {"store": "#1f77b4", "restore": "#ff7f0e", "call": "#2ca02c", "grad": "#d62728", "fetch": "#9467bd"}
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    for action, color in colors.items():
        indices = [i for i, item in enumerate(log["actions"]) if item["action"] == action]
        timesteps = [log["actions"][i]["step"] for i in indices]
        ax.scatter(indices, timesteps, s=5 if action in {"call", "grad"} else 13,
                   color=color, label=action.capitalize(), alpha=0.8, rasterized=True)
    ax.set(xlabel="Operation index", ylabel="Timestep", title="Treeverse schedule; reflector shot 0, budget 5")
    ax.legend(ncol=5, loc="upper right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "checkpoint-actions.png", dpi=180)
    plt.close(fig)

    full_states = 241
    state_bytes = 2 * 41 * 41 * 8
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    axes[0].semilogy(BUDGETS, calls, "o-", label="Treeverse")
    axes[0].axhline(240, color="black", ls="--", label="Full history")
    axes[0].set(xlabel="Additional checkpoint slots", ylabel="Forward steps per shot",
                title="Recomputation cost", xticks=BUDGETS)
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(BUDGETS, bytes_, "o-", label="Treeverse")
    axes[1].axhline(full_states * state_bytes, color="black", ls="--",
                    label=f"Full history: {full_states * state_bytes:,} bytes")
    for budget, peak, value in zip(BUDGETS, peaks, bytes_):
        axes[1].annotate(f"{peak} states", (budget, value), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
    axes[1].set(xlabel="Additional checkpoint slots", ylabel="Peak saved-state storage (bytes)",
                title="Two wavefields per saved state", xticks=BUDGETS)
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "checkpoint-work.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
