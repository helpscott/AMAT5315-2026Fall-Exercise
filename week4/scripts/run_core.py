#!/usr/bin/env python3
"""Run the contract Taylor-Green, random, and clearly unstable cases."""

import json

from common import ARTIFACTS, build, field, run_fluid


build()
ARTIFACTS.mkdir(exist_ok=True)

taylor = field(["taylor-green", "--n", "64"])
result = run_fluid(
    taylor,
    ["--method", "rk4", "--nu", "0.1", "--dt", "0.01", "--t-end", "1", "--every", "0.1"],
    ARTIFACTS / "taylor-green",
    ARTIFACTS / "taylor-green.tsv",
)
result.check_returncode()
exact = field(["taylor-green", "--n", "64", "--nu", "0.1", "--t", "1"])
(ARTIFACTS / "taylor-green" / "exact-t1.json").write_text(json.dumps(exact))

random = field(["random", "--n", "128", "--seed", "2026", "--k-min", "2", "--k-max", "6"])
result = run_fluid(
    random,
    ["--method", "rk4", "--nu", "0.004", "--dt", "0.01", "--t-end", "10", "--every", "0.1"],
    ARTIFACTS / "random",
    ARTIFACTS / "random.tsv",
)
result.check_returncode()

unstable = run_fluid(
    taylor,
    ["--method", "rk4", "--nu", "0.1", "--dt", "0.04", "--t-end", "4", "--every", "0.1"],
    ARTIFACTS / "unstable/taylor-green",
    ARTIFACTS / "unstable/taylor-green.tsv",
)
if unstable.returncode == 0:
    raise SystemExit("the dt=0.04 Taylor-Green diagnostic did not become non-finite")

print((ARTIFACTS / "taylor-green.tsv").read_text().splitlines()[0])
print((ARTIFACTS / "taylor-green.tsv").read_text().splitlines()[1])
print((ARTIFACTS / "taylor-green.tsv").read_text().splitlines()[-1])
print((ARTIFACTS / "random.tsv").read_text().splitlines()[1])
print((ARTIFACTS / "random.tsv").read_text().splitlines()[-1])
