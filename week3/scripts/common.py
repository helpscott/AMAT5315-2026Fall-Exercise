"""Shared analysis helpers for the Week 3 Ising evidence."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
EXACT_TC = 2.26919
BLOCK_LENGTHS = (2000, 4000, 8000)


def rows(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def by_temp(records: list[dict]) -> dict[float, list[dict]]:
    grouped: dict[float, list[dict]] = defaultdict(list)
    for record in records:
        grouped[round(float(record["T"]), 8)].append(record)
    return dict(sorted(grouped.items()))


def load_run(name: str) -> dict[float, list[dict]]:
    path = ROOT / "artifacts" / name / "series.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"required run is missing: {path}")
    return by_temp(rows(path))


def merged_metropolis(lattice_size: int) -> dict[float, list[dict]]:
    """Return the 27-temperature grid, preferring the long window rows."""
    merged = load_run(f"coarse-l{lattice_size}")
    merged.update(load_run(f"window-l{lattice_size}"))
    return dict(sorted(merged.items()))


def magnetizations(records: list[dict]) -> np.ndarray:
    return np.asarray([record["M"] for record in records], dtype=float)


def observable_metrics(
    grouped: dict[float, list[dict]], lattice_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    temperatures = np.asarray(list(grouped), dtype=float)
    mean_abs = []
    susceptibility = []
    for temperature, records_at_temperature in grouped.items():
        values = magnetizations(records_at_temperature)
        abs_mean = float(np.mean(np.abs(values)))
        mean_abs.append(abs_mean)
        susceptibility.append(
            lattice_size**2 * (float(np.mean(values**2)) - abs_mean**2) / temperature
        )
    return temperatures, np.asarray(mean_abs), np.asarray(susceptibility)


def five_point_peak(
    temperatures: np.ndarray, susceptibility: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    maximum = int(np.argmax(susceptibility))
    if maximum < 2 or maximum > len(temperatures) - 3:
        raise ValueError("susceptibility maximum does not have two points on each side")
    fitted_temperatures = temperatures[maximum - 2 : maximum + 3]
    coefficients = np.polyfit(
        fitted_temperatures, susceptibility[maximum - 2 : maximum + 3], 2
    )
    if coefficients[0] >= 0:
        raise ValueError("fitted parabola does not bend downward")
    peak = float(-coefficients[1] / (2 * coefficients[0]))
    if not fitted_temperatures[0] <= peak <= fitted_temperatures[-1]:
        raise ValueError("fitted peak lies outside its five temperatures")
    return peak, coefficients, fitted_temperatures


def autocorrelation(values: np.ndarray) -> np.ndarray:
    """Unbiased normalized autocorrelation computed with an FFT."""
    centered = np.asarray(values, dtype=float) - float(np.mean(values))
    count = len(centered)
    if count < 2:
        return np.ones(count)
    variance = float(np.dot(centered, centered) / count)
    if variance == 0:
        return np.ones(count)
    fft_length = 1 << (2 * count - 1).bit_length()
    transform = np.fft.rfft(centered, fft_length)
    covariance = np.fft.irfft(transform * np.conjugate(transform), fft_length)[:count]
    covariance /= np.arange(count, 0, -1)
    return covariance / variance


def tau_int(values: np.ndarray) -> float:
    """Integrated autocorrelation time with the six-tau running window."""
    rho = autocorrelation(values)
    total = 0.5
    for lag in range(1, len(rho)):
        total += float(rho[lag])
        if total <= 0.5:
            return 0.5
        if lag > 6.0 * total:
            break
    return max(0.5, total)


def block_standard_error(values: np.ndarray, block_length: int) -> tuple[float, int]:
    values = np.asarray(values, dtype=float)
    block_count = len(values) // block_length
    if block_count < 2:
        return math.nan, block_count
    means = values[: block_count * block_length].reshape(block_count, block_length).mean(1)
    return float(means.std(ddof=1) / math.sqrt(block_count)), block_count


def block_moment_tables(values: np.ndarray, block_length: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    block_count = len(values) // block_length
    if block_count < 2:
        raise ValueError(f"only {block_count} complete blocks of length {block_length}")
    blocks = values[: block_count * block_length].reshape(block_count, block_length)
    return np.abs(blocks).mean(1), (blocks**2).mean(1)


def bootstrap_susceptibility(
    grouped: dict[float, list[dict]],
    lattice_size: int,
    block_length: int,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    temperatures = np.asarray(list(grouped), dtype=float)
    simulated = np.empty((replicates, len(temperatures)), dtype=float)
    for column, (temperature, records_at_temperature) in enumerate(grouped.items()):
        values = magnetizations(records_at_temperature)
        block_abs, block_sq = block_moment_tables(values, block_length)
        picks = rng.integers(0, len(block_abs), size=(replicates, len(block_abs)))
        abs_mean = block_abs[picks].mean(axis=1)
        sq_mean = block_sq[picks].mean(axis=1)
        simulated[:, column] = lattice_size**2 * (sq_mean - abs_mean**2) / temperature
    return temperatures, simulated


def bootstrap_mean_abs_error(
    grouped: dict[float, list[dict]],
    block_length: int,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    temperatures = np.asarray(list(grouped), dtype=float)
    means = np.empty(len(temperatures), dtype=float)
    errors = np.empty(len(temperatures), dtype=float)
    for column, records_at_temperature in enumerate(grouped.values()):
        values = np.abs(magnetizations(records_at_temperature))
        block_count = len(values) // block_length
        blocks = values[: block_count * block_length].reshape(block_count, block_length).mean(1)
        picks = rng.integers(0, block_count, size=(replicates, block_count))
        samples = blocks[picks].mean(axis=1)
        means[column] = values.mean()
        errors[column] = samples.std(ddof=1)
    return temperatures, means, errors


def onsager_magnetization(temperatures: np.ndarray) -> np.ndarray:
    result = np.zeros_like(temperatures, dtype=float)
    below = temperatures < EXACT_TC
    result[below] = (1.0 - np.sinh(2.0 / temperatures[below]) ** -4) ** 0.125
    return result
