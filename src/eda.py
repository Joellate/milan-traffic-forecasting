"""Exploratory analysis (Assignment Section 2).

Produces:
  reports/figures/total_traffic_distribution.png
  reports/figures/timeseries_first_two_weeks.png
  reports/figures/top_square_stl_decomposition.png
  reports/figures/top_square_acf_pacf.png
  reports/tables/top_squares.csv
  data/interim/eda_stats.json

Reads only what is needed for each figure (predicate pushdown on square_id
for the parquet dataset, which is sorted by square_id) so that no more than
one square's series is held in memory at a time, keeping this stage
lightweight even though the parquet dataset itself covers 10,000 squares x
62 days.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller

DATA_DIR = Path("data/processed")
FIG_DIR = Path("reports/figures")
TAB_DIR = Path("reports/tables")
INTERIM_DIR = Path("data/interim")
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

ROME = "Europe/Rome"


def load_square_series(square_id: int) -> pd.Series:
    dataset = ds.dataset(DATA_DIR / "internet_traffic.parquet", format="parquet")
    table = dataset.to_table(filter=(ds.field("square_id") == square_id))
    df = table.to_pandas()
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(ROME)
    df = df.sort_values("datetime").set_index("datetime")["internet"]
    df = df[~df.index.duplicated(keep="first")]
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="10min")
    df = df.reindex(full_index)
    return df


def fig1_distribution(totals: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(totals["total_internet"], bins=80, color="#3b6fa0", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Total internet traffic over 2-month period (per square, a.u.)")
    ax.set_ylabel("Number of squares")
    ax.set_title("Distribution of total Internet traffic across the 10,000 Milan grid squares")
    ax.axvline(totals["total_internet"].median(), color="#c0392b", linestyle="--", linewidth=1, label="median")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "total_traffic_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(np.log10(totals["total_internet"] + 1), bins=80, color="#3b6fa0", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("log10(total internet traffic + 1)")
    ax.set_ylabel("Number of squares")
    ax.set_title("Distribution of total Internet traffic (log scale)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "total_traffic_distribution_log.png", dpi=150)
    plt.close(fig)


def fig2_timeseries(square_ids, labels):
    fig, axes = plt.subplots(len(square_ids), 1, figsize=(11, 2.2 * len(square_ids)), sharex=True)
    stats = {}
    for ax, sq, label in zip(axes, square_ids, labels):
        s = load_square_series(sq)
        two_weeks = s[(s.index >= s.index.min()) & (s.index < s.index.min() + pd.Timedelta(days=14))]
        ax.plot(two_weeks.index, two_weeks.values, linewidth=0.7, color="#2c6e91")
        ax.set_ylabel("Internet\ntraffic", fontsize=8)
        ax.set_title(f"Square {sq} ({label})", fontsize=9, loc="left")
        stats[str(sq)] = {
            "label": label,
            "mean": float(two_weeks.mean()), "std": float(two_weeks.std()),
            "max": float(two_weeks.max()), "n_missing": int(two_weeks.isna().sum()),
        }
    axes[-1].set_xlabel("Date")
    fig.suptitle("Internet traffic time series — first two weeks (Nov 1–14, 2013)", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "timeseries_first_two_weeks.png", dpi=150)
    plt.close(fig)
    return stats


def fig3_stl(square_id: int):
    s = load_square_series(square_id).interpolate(limit_direction="both")
    stl = STL(s, period=144, robust=True).fit()
    fig = stl.plot()
    fig.set_size_inches(10, 7)
    fig.suptitle(f"STL decomposition — Square {square_id} (daily period = 144 intervals)", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "top_square_stl_decomposition.png", dpi=150)
    plt.close(fig)

    adf_raw = adfuller(s.values, autolag="AIC")
    adf_resid = adfuller(stl.resid.dropna().values, autolag="AIC")
    var_trend = float(np.var(stl.trend))
    var_seasonal = float(np.var(stl.seasonal))
    var_resid = float(np.var(stl.resid))
    var_total = var_trend + var_seasonal + var_resid
    return {
        "adf_raw_series": {"statistic": float(adf_raw[0]), "p_value": float(adf_raw[1])},
        "adf_stl_residual": {"statistic": float(adf_resid[0]), "p_value": float(adf_resid[1])},
        "variance_share": {
            "trend": var_trend / var_total, "seasonal": var_seasonal / var_total, "residual": var_resid / var_total,
        },
    }


def fig4_acf_pacf(square_id: int):
    s = load_square_series(square_id).interpolate(limit_direction="both")
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    plot_acf(s, lags=300, ax=axes[0])
    axes[0].set_title(f"ACF — Square {square_id} (up to 300 lags = ~2 days)")
    plot_pacf(s, lags=300, ax=axes[1], method="ywm")
    axes[1].set_title(f"PACF — Square {square_id} (up to 300 lags = ~2 days)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "top_square_acf_pacf.png", dpi=150)
    plt.close(fig)


def main():
    totals = pd.read_parquet(DATA_DIR / "total_traffic_per_square.parquet")
    fig1_distribution(totals)

    top3 = totals.head(3)
    top3.to_csv(TAB_DIR / "top_squares.csv", index=False)
    print("Top 3 squares by total traffic:")
    print(top3)

    square_ids = list(top3["square_id"]) + [4159, 4556]
    labels = ["rank1", "rank2", "rank3", "sq4159", "sq4556"]
    ts_stats = fig2_timeseries(square_ids, labels)

    top1 = int(top3.iloc[0]["square_id"])
    stl_stats = fig3_stl(top1)
    fig4_acf_pacf(top1)

    summary = {
        "top3_squares": top3.to_dict(orient="records"),
        "timeseries_stats_first_two_weeks": ts_stats,
        "stl_and_stationarity_top1": stl_stats,
        "top1_square": top1,
    }
    with open(INTERIM_DIR / "eda_stats.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
