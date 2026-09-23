#!/usr/bin/env python3
"""Run the stability brackets and the paired perturbation experiments."""

from __future__ import annotations

import json
import math

import numpy as np

from common import ARTIFACTS, build, field, run_fluid


build()
scan = ARTIFACTS / "scan"
scan.mkdir(parents=True, exist_ok=True)
taylor = field(["taylor-green", "--n", "64"])
random = field(["random", "--n", "128", "--seed", "2026", "--k-min", "2", "--k-max", "6"])


def scan_run(initial: dict, name: str, method: str, dt: float, end: float, nu: float) -> bool:
    label = f"{name}-{method}-dt{dt:g}"
    result = run_fluid(
        initial,
        [
            "--method",
            method,
            "--nu",
            str(nu),
            "--dt",
            str(dt),
            "--t-end",
            str(end),
            "--every",
            "0.5",
        ],
        scan / label,
        scan / f"{label}.tsv",
    )
    return result.returncode == 0


taylor_status = {
    0.032: scan_run(taylor, "taylor-green", "rk4", 0.032, 8.0, 0.1),
    0.033: scan_run(taylor, "taylor-green", "rk4", 0.033, 8.0, 0.1),
}

random_status = {}
for dt in (0.038, 0.040):
    random_status[dt] = scan_run(random, "random", "rk4", dt, 10.0, 0.004)
if all(random_status.values()):
    for dt in np.arange(0.042, 0.062, 0.002):
        value = round(float(dt), 3)
        random_status[value] = scan_run(random, "random", "rk4", value, 10.0, 0.004)
        if not random_status[value]:
            break
elif not any(random_status.values()):
    for dt in np.arange(0.036, 0.018, -0.002):
        value = round(float(dt), 3)
        random_status[value] = scan_run(random, "random", "rk4", value, 10.0, 0.004)
        if random_status[value]:
            break

stable_steps = sorted(dt for dt, stable in random_status.items() if stable)
unstable_steps = sorted(dt for dt, stable in random_status.items() if not stable)
if not stable_steps or not unstable_steps:
    raise SystemExit(f"failed to bracket random-flow stability: {random_status}")
stable_dt = max(dt for dt in stable_steps if dt < min(unstable_steps))
unstable_dt = min(dt for dt in unstable_steps if dt > stable_dt)
euler_stable = scan_run(random, "random", "euler", 0.01, 10.0, 0.004)
if euler_stable:
    raise SystemExit("Euler dt=0.01 unexpectedly reached t=10")

u = np.asarray(random["u"])
v = np.asarray(random["v"])
maximum_speed = float(np.max(np.sqrt(u * u + v * v)))
maximum_component = float(max(np.max(np.abs(u)), np.max(np.abs(v))))
cutoff = 128 // 3
advective_bound = 2.83 / (maximum_speed * math.sqrt(2.0) * cutoff)
diffusive_bound = 2.785 / (0.1 * 2.0 * (64 // 3) ** 2)

report = {
    "taylor_status": {str(key): value for key, value in taylor_status.items()},
    "random_status": {str(key): value for key, value in random_status.items()},
    "random_stable_dt": stable_dt,
    "random_unstable_dt": unstable_dt,
    "euler_reaches_t10": euler_stable,
    "random_initial_maximum_speed": maximum_speed,
    "random_initial_maximum_component": maximum_component,
    "predicted_advective_bound": advective_bound,
    "predicted_diffusive_limit": diffusive_bound,
}
(scan / "scan.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))


def perturbed(initial: dict) -> tuple[dict, float]:
    altered = json.loads(json.dumps(initial))
    n = initial["n"]
    u_values = np.asarray(initial["u"], dtype=float).reshape(n, n)
    v_values = np.asarray(initial["v"], dtype=float).reshape(n, n)
    scale = float(max(np.max(np.abs(u_values)), np.max(np.abs(v_values))))
    coordinates = np.arange(n) * 2.0 * np.pi / n
    x, y = np.meshgrid(coordinates, coordinates)
    amplitude = 7e-5 * scale
    # delta omega = -A cos(3x) cos(4y), with -laplacian(psi)=delta omega.
    u_values += (4.0 * amplitude / 25.0) * np.cos(3.0 * x) * np.sin(4.0 * y)
    v_values += (-3.0 * amplitude / 25.0) * np.sin(3.0 * x) * np.cos(4.0 * y)
    altered["u"] = u_values.ravel().tolist()
    altered["v"] = v_values.ravel().tolist()
    return altered, scale


sensitivity = ARTIFACTS / "sensitivity"
sensitivity.mkdir(parents=True, exist_ok=True)
sensitivity_report = {}
for name, initial, nu in (("taylor-green", taylor, 0.1), ("random", random, 0.004)):
    changed, component_scale = perturbed(initial)
    sensitivity_report[name] = {"maximum_component": component_scale}
    for suffix, starting_field in (("original", initial), ("perturbed", changed)):
        result = run_fluid(
            starting_field,
            [
                "--method",
                "rk4",
                "--nu",
                str(nu),
                "--dt",
                "0.01",
                "--t-end",
                "20",
                "--every",
                "0.5",
            ],
            sensitivity / f"{name}-{suffix}",
            sensitivity / f"{name}-{suffix}.tsv",
        )
        result.check_returncode()
(sensitivity / "sensitivity.json").write_text(json.dumps(sensitivity_report, indent=2))
