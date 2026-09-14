"""Shared data-loading and windowing utilities for the forecasting experiments
(Assignment Section 4). Loads one square's series from the processed parquet
dataset, splits it into a training period and the Dec 16-22 evaluation week,
and builds supervised sliding windows for the deep learning models.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

DATA_DIR = Path("data/processed")
ROME = "Europe/Rome"

TEST_START = pd.Timestamp("2013-12-16 00:00:00", tz=ROME)
TEST_END = pd.Timestamp("2013-12-22 23:50:00", tz=ROME)  # inclusive, last 10-min slot of Dec 22


def load_square_series(square_id: int) -> pd.Series:
    dataset = ds.dataset(DATA_DIR / "internet_traffic.parquet", format="parquet")
    table = dataset.to_table(filter=(ds.field("square_id") == square_id))
    df = table.to_pandas()
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(ROME)
    df = df.sort_values("datetime").set_index("datetime")["internet"]
    df = df[~df.index.duplicated(keep="first")]
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="10min")
    df = df.reindex(full_index)
    n_missing = int(df.isna().sum())
    df = df.interpolate(limit_direction="both")
    df.attrs["n_missing_interpolated"] = n_missing
    return df


def train_test_split(series: pd.Series):
    train = series[series.index < TEST_START]
    test = series[(series.index >= TEST_START) & (series.index <= TEST_END)]
    return train, test


def fourier_terms(index: pd.DatetimeIndex, period: int, k: int, t0):
    """Sin/cos harmonic regressors for a given seasonal period (in # of 10-min steps)."""
    t = ((index - t0) / pd.Timedelta(minutes=10)).astype(float).values
    cols = {}
    for i in range(1, k + 1):
        cols[f"sin_p{period}_{i}"] = np.sin(2 * np.pi * i * t / period)
        cols[f"cos_p{period}_{i}"] = np.cos(2 * np.pi * i * t / period)
    return pd.DataFrame(cols, index=index)


def build_fourier_exog(index: pd.DatetimeIndex, t0, k_daily=4, k_weekly=3):
    daily = fourier_terms(index, period=144, k=k_daily, t0=t0)
    weekly = fourier_terms(index, period=1008, k=k_weekly, t0=t0)
    return pd.concat([daily, weekly], axis=1)


class WindowScaler:
    """Min-max scaler fit only on the training portion of a series."""

    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, values: np.ndarray):
        self.min_ = float(np.min(values))
        self.max_ = float(np.max(values))
        return self

    def transform(self, values: np.ndarray):
        return (values - self.min_) / (self.max_ - self.min_ + 1e-9)

    def inverse_transform(self, values: np.ndarray):
        return values * (self.max_ - self.min_ + 1e-9) + self.min_


def make_windows(values: np.ndarray, index: pd.DatetimeIndex, window: int):
    """Sliding windows: X[i] = values[i:i+window], y[i] = values[i+window],
    target_time[i] = index[i+window]. One-step-ahead, non-overlapping targets not required."""
    X, y, target_time = [], [], []
    for i in range(len(values) - window):
        X.append(values[i:i + window])
        y.append(values[i + window])
        target_time.append(index[i + window])
    return np.array(X), np.array(y), pd.DatetimeIndex(target_time)


def prepare_dl_dataset(series: pd.Series, window: int, val_days: int = 7):
    """Builds scaled train/val/test windowed datasets for the DL models.
    Scaler is fit only on the training portion (everything before TEST_START).
    Validation = last `val_days` of the training period (time-based split).
    """
    train, test = train_test_split(series)
    scaler = WindowScaler().fit(train.values)

    val_start = TEST_START - pd.Timedelta(days=val_days)
    tr_only = train[train.index < val_start]
    val_only = train[train.index >= val_start - pd.Timedelta(minutes=10 * window)]  # keep warm-up context

    full_scaled = pd.Series(scaler.transform(series.values), index=series.index)

    def windows_for_range(lo, hi):
        ctx_lo = lo - pd.Timedelta(minutes=10 * window)
        seg = full_scaled[(full_scaled.index >= ctx_lo) & (full_scaled.index <= hi)]
        X, y, t = make_windows(seg.values, seg.index, window)
        keep = (t >= lo) & (t <= hi)
        return X[keep], y[keep], t[keep]

    X_train, y_train, t_train = windows_for_range(tr_only.index.min() + pd.Timedelta(minutes=10 * window), val_start - pd.Timedelta(minutes=10))
    X_val, y_val, t_val = windows_for_range(val_start, TEST_START - pd.Timedelta(minutes=10))
    X_test, y_test, t_test = windows_for_range(TEST_START, TEST_END)

    return {
        "scaler": scaler,
        "train": (X_train, y_train, t_train),
        "val": (X_val, y_val, t_val),
        "test": (X_test, y_test, t_test),
    }
