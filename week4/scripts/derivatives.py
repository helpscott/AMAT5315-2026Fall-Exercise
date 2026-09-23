#!/usr/bin/env python3
"""Print the full-precision derivative comparison from the Rust routines."""

import subprocess

from common import ROOT


subprocess.run(
    ["cargo", "run", "--release", "--quiet", "--example", "derivative_check"],
    cwd=ROOT,
    check=True,
)
