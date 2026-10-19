"""Prevent accidental replacement of immutable historical results."""
from pathlib import Path

HISTORICAL = Path(__file__).resolve().parents[2] / 'results/metrics/historical'

def protect_historical(path):
    target = Path(path).resolve()
    if target == HISTORICAL or HISTORICAL in target.parents:
        raise ValueError('Historical results are read-only; choose results/metrics/new or another output path.')
