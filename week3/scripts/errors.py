#!/usr/bin/env python3
"""Report naive, blocked, and autocorrelation-corrected errors."""

import math

import numpy as np

from common import EVIDENCE, magnetizations, merged_metropolis, tau_int


EVIDENCE.mkdir(exist_ok=True)
with (EVIDENCE / "errors.txt").open("w") as output:
    output.write("L T mean_abs_M naive_se block50_se ratio tau_int honest_se\n")
    for lattice_size in (32, 64):
        for temperature, records_at_temperature in merged_metropolis(lattice_size).items():
            values = np.abs(magnetizations(records_at_temperature))
            count = len(values)
            naive = float(values.std(ddof=1) / math.sqrt(count))
            correlation_time = tau_int(values)
            blocks = np.array_split(values, 50)
            blocked = float(np.std([block.mean() for block in blocks], ddof=1) / math.sqrt(50))
            honest = naive * math.sqrt(2.0 * correlation_time)
            output.write(
                f"{lattice_size} {temperature:.2f} {values.mean():.6f} "
                f"{naive:.8g} {blocked:.8g} {blocked / naive:.3f} "
                f"{correlation_time:.3f} {honest:.8g}\n"
            )
