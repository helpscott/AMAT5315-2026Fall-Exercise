"""Shared runners and readers for the Week 4 evidence scripts."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
EVIDENCE = ROOT / "evidence"


def build() -> None:
    subprocess.run(["cargo", "build", "--release"], cwd=ROOT, check=True)


def field(arguments: list[str]) -> dict:
    result = subprocess.run(
        [str(ROOT / "target/release/field"), *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def run_fluid(
    initial: dict,
    arguments: list[str],
    output_directory: Path,
    tsv_path: Path,
) -> subprocess.CompletedProcess[str]:
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [str(ROOT / "target/release/fluid"), *arguments, "--out", str(output_directory)],
        cwd=ROOT,
        input=json.dumps(initial),
        capture_output=True,
        text=True,
    )
    tsv_path.write_text(result.stdout)
    if result.stderr:
        print(result.stderr, end="")
    return result


def records(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def table(path: Path) -> np.ndarray:
    rows = []
    with path.open() as stream:
        next(stream)
        for line in stream:
            fields = line.split()
            if len(fields) == 3:
                rows.append([float(value) for value in fields])
    return np.asarray(rows)


def relative_l2(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right) / np.linalg.norm(right))


def final_record(directory: Path) -> dict:
    return records(directory / "fields.jsonl")[-1]
