"""Runs the GNN's tests (gnn_cases.py) in their own pytest process.

torch and LightGBM each bring an OpenMP runtime (both libomp on macOS), and one process
can't use both: LightGBM then torch hangs, torch then LightGBM segfaults (checked on the
M1, 2026-09-28). Just importing torch is enough. The CV runs never load both (--model gnn
doesn't import LightGBM), but this suite would, so torch stays out of its process.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


def test_gnn_cases_in_their_own_process():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch isn't installed (prediction extra)")
    tests = Path(__file__).resolve().parent
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "gnn_cases.py"],
        cwd=tests,
        capture_output=True,
        check=False,
        text=True,
    )
    assert r.returncode == 0, r.stdout[-6000:] + r.stderr[-3000:]
