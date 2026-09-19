"""Lightweight smoke tests for evaluation metrics and Fourier exog shape.

Run from project root:
  python -m pytest tests/test_smoke.py -q
or:
  python tests/test_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate import mae, rmse, mape, all_metrics  # noqa: E402
from data_prep import build_fourier_exog  # noqa: E402


def test_metrics_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert mae(y, y) == 0.0
    assert rmse(y, y) == 0.0
    assert mape(y, y) == 0.0
    m = all_metrics(y, y)
    assert m["R2"] == 1.0


def test_metrics_known_error():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])
    assert abs(mae(y_true, y_pred) - (2 + 2 + 3) / 3) < 1e-9


def test_fourier_exog_shape():
    idx = pd.date_range("2013-11-01", periods=144, freq="10min", tz="Europe/Rome")
    exog = build_fourier_exog(idx, idx[0], k_daily=4, k_weekly=3)
    # 4 daily + 3 weekly harmonics × (sin, cos) = 14 columns
    assert exog.shape == (144, 14)
    assert np.isfinite(exog.values).all()


if __name__ == "__main__":
    test_metrics_perfect_prediction()
    test_metrics_known_error()
    test_fourier_exog_shape()
    print("All smoke tests passed.")
