#!/usr/bin/env python3
"""Fail fast if any Week 5 numerical acceptance criterion is missing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"


def close(value, reference, relative):
    return abs(value - reference) <= relative * abs(reference)


def audit(path: Path, steps: int, budget: int):
    actions = json.loads(path.read_text())["actions"]
    stored = {0}
    grads = []
    invalid = overruns = 0
    for item in actions:
        action, step = item["action"], item["step"]
        if action == "restore" and step not in stored:
            invalid += 1
        elif action == "store":
            stored.add(step)
            overruns += len(stored) > budget + 1
        elif action == "fetch":
            stored.discard(step)
        elif action == "grad":
            grads.append(step)
    assert grads == list(range(steps - 1, -1, -1))
    assert invalid == 0 and overruns == 0


def main() -> None:
    derivative = json.loads((ART / "ad" / "derivatives.json").read_text())
    assert abs(derivative["energy"] - (-0.6570169144600471)) < 1e-12
    assert abs(derivative["tangents"]["U"] - 2.239979929791143) < 1e-12
    assert abs(derivative["adjoints"]["r"] - 2.239979929791143) < 1e-12
    assert abs(derivative["adjoints"]["a"] - (-2.3425903117359734)) < 1e-12

    experiment = json.loads((ROOT / "inputs" / "reflector.json").read_text())
    traces = np.load(ART / "forward" / "traces.npy")
    assert close(float(np.linalg.norm(traces)), 11.574770, 1e-4)
    expected_max = [0.60809514, 0.59271397, 0.60809514]
    for shot, expected in enumerate(expected_max):
        index = np.unravel_index(np.argmax(np.abs(traces[shot])), traces[shot].shape)
        assert index[0] == 83 and close(float(traces[shot][index]), expected, 1e-4)

    born = np.load(ART / "born" / "born_data.npy")
    image = np.load(ART / "adjoint" / "image.npy")
    perturbation = np.asarray(experiment["perturbation"])
    left = float(np.sum(born * born))
    right = float(np.sum(perturbation * image))
    assert abs(left - right) / max(abs(left), abs(right)) < 1e-9
    profile = np.linalg.norm(image[10:34, 7:34], axis=1)
    assert abs((int(np.argmax(profile)) + 10) - 21) <= 1

    expected = {1: (2, 28_680), 3: (4, 1_695), 5: (6, 990), 10: (11, 642)}
    for budget, (states, calls) in expected.items():
        folder = ART / f"checkpoint-{budget}"
        checkpointed = np.load(folder / "image.npy")
        assert np.linalg.norm(checkpointed - image) / np.linalg.norm(image) < 1e-9
        stats = json.loads((folder / "result.json").read_text())["statistics"]
        assert stats["peak_saved_states"] == states
        assert stats["per_shot"][0]["scheduler_forward_calls"] == calls
        for shot in range(3):
            audit(folder / f"actions-{shot}.json", 240, budget)

    marmousi_image = np.load(ART / "marmousi-image" / "image.npy")
    marmousi_stats = json.loads((ART / "marmousi-image" / "result.json").read_text())["statistics"]
    assert close(float(np.linalg.norm(marmousi_image)), 6.7037741e-4, 1e-4)
    assert marmousi_stats["peak_saved_states"] == 6
    assert marmousi_stats["peak_saved_bytes"] == 20_788_320
    for shot in range(9):
        audit(ART / "marmousi-image" / f"actions-{shot}.json", 1200, 5)

    too_large = [path for path in ART.rglob("*") if path.is_file() and path.suffix != ".npy" and path.stat().st_size >= 5_000_000]
    assert not too_large, f"committed evidence at least 5 MB: {too_large}"
    print("PASS: all Week 5 numerical, schedule, storage and evidence-size checks")


if __name__ == "__main__":
    main()
