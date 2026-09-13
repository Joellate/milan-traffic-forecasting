"""Demonstrates the memory-management problem and the optimisation applied.

Compares:
  (A) naive load: pandas.read_csv with default dtypes, all 8 columns, one
      full day in memory at once.
  (B) optimised load: chunked streaming directly from the zip archive,
      restricted to the 3 needed columns, compact dtypes, and immediate
      per-chunk aggregation (square_id, interval) -> summed internet
      traffic, so the raw rows never fully materialise in RAM.

Run on an 8-core / 8.4GB RAM machine. Results are written to
data/interim/memory_demo.json for use in the report.
"""
import gc
import json
import time
import zipfile
from pathlib import Path

import pandas as pd
import psutil

from manifest import build_manifest

COLUMNS = ["square_id", "timestamp", "country_code", "sms_in", "sms_out", "call_in", "call_out", "internet"]
OUT_PATH = Path("data/interim/memory_demo.json")
SAMPLE_DATE = "2013-11-01"


def rss_mb():
    return psutil.Process().memory_info().rss / 1e6


def naive_load(zip_path, entry):
    gc.collect()
    before = rss_mb()
    t0 = time.time()
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(entry) as f:
            df = pd.read_csv(f, sep="\t", header=None, names=COLUMNS)
    elapsed = time.time() - t0
    after = rss_mb()
    stats = {
        "rows": len(df),
        "pandas_memory_usage_deep_mb": df.memory_usage(deep=True).sum() / 1e6,
        "process_rss_before_mb": before,
        "process_rss_after_mb": after,
        "process_rss_delta_mb": after - before,
        "elapsed_sec": elapsed,
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    }
    del df
    gc.collect()
    return stats


def optimised_load(zip_path, entry, chunksize=500_000):
    gc.collect()
    before = rss_mb()
    t0 = time.time()
    dtype = {"square_id": "int16", "timestamp": "int64", "internet": "float32"}
    agg = None
    peak_rss = before
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(entry) as f:
            reader = pd.read_csv(
                f, sep="\t", header=None, names=COLUMNS,
                usecols=["square_id", "timestamp", "internet"],
                dtype=dtype, chunksize=chunksize,
            )
            for chunk in reader:
                chunk = chunk.dropna(subset=["internet"])
                grouped = chunk.groupby(["square_id", "timestamp"], as_index=False)["internet"].sum()
                if agg is None:
                    agg = grouped
                else:
                    agg = pd.concat([agg, grouped], ignore_index=True)
                    agg = agg.groupby(["square_id", "timestamp"], as_index=False)["internet"].sum()
                peak_rss = max(peak_rss, rss_mb())
    elapsed = time.time() - t0
    after = rss_mb()
    stats = {
        "rows_after_aggregation": len(agg),
        "pandas_memory_usage_deep_mb": agg.memory_usage(deep=True).sum() / 1e6,
        "process_rss_before_mb": before,
        "process_rss_after_mb": after,
        "process_rss_peak_mb": peak_rss,
        "process_rss_delta_mb": after - before,
        "elapsed_sec": elapsed,
        "dtypes": {c: str(t) for c, t in agg.dtypes.items()},
    }
    del agg
    gc.collect()
    return stats


def main():
    manifest = build_manifest()
    zip_path, entry = manifest[SAMPLE_DATE]
    print(f"Sample file: {entry} from {zip_path}")

    print("Running naive load...")
    naive = naive_load(zip_path, entry)
    print(json.dumps(naive, indent=2))

    print("Running optimised load...")
    optimised = optimised_load(zip_path, entry)
    print(json.dumps(optimised, indent=2))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({"sample_date": SAMPLE_DATE, "naive": naive, "optimised": optimised}, f, indent=2)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
