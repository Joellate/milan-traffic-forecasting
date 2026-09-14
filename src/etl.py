"""Main ETL: streams all 62 daily files directly from the zip archives,
keeps only (square_id, timestamp, internet), aggregates internet traffic
per (square_id, timestamp) summing across country codes, and writes:

  data/processed/internet_traffic.parquet   -- full long-format series
  data/processed/total_traffic_per_square.parquet -- total traffic per square
  data/interim/etl_run_stats.json           -- timing / memory evidence

Never holds a raw day file fully in memory (see memory_demo.py for the
naive-vs-chunked comparison); each day is streamed in 500k-row chunks and
aggregated before the next chunk is read.
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
DTYPE = {"square_id": "int16", "timestamp": "int64", "internet": "float32"}
CHUNKSIZE = 500_000

OUT_DIR = Path("data/processed")
INTERIM_DIR = Path("data/interim")


def rss_mb():
    return psutil.Process().memory_info().rss / 1e6


def aggregate_day(zip_path, entry):
    agg = None
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(entry) as f:
            reader = pd.read_csv(
                f, sep="\t", header=None, names=COLUMNS,
                usecols=["square_id", "timestamp", "internet"],
                dtype=DTYPE, chunksize=CHUNKSIZE,
            )
            for chunk in reader:
                chunk = chunk.dropna(subset=["internet"])
                grouped = chunk.groupby(["square_id", "timestamp"], as_index=False)["internet"].sum()
                if agg is None:
                    agg = grouped
                else:
                    agg = pd.concat([agg, grouped], ignore_index=True)
                    agg = agg.groupby(["square_id", "timestamp"], as_index=False)["internet"].sum()
    agg["internet"] = agg["internet"].astype("float32")
    return agg


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest()
    dates = list(manifest.keys())
    print(f"Processing {len(dates)} days: {dates[0]} .. {dates[-1]}")

    t_start = time.time()
    peak_rss = rss_mb()
    day_frames = []
    running_total = None
    per_day_stats = []

    for i, date in enumerate(dates):
        zip_path, entry = manifest[date]
        t0 = time.time()
        day_df = aggregate_day(zip_path, entry)
        elapsed = time.time() - t0

        day_total = day_df.groupby("square_id", as_index=False)["internet"].sum()
        if running_total is None:
            running_total = day_total
        else:
            running_total = pd.concat([running_total, day_total], ignore_index=True)
            running_total = running_total.groupby("square_id", as_index=False)["internet"].sum()

        day_frames.append(day_df)
        cur_rss = rss_mb()
        peak_rss = max(peak_rss, cur_rss)
        per_day_stats.append({"date": date, "rows": len(day_df), "elapsed_sec": elapsed, "rss_mb": cur_rss})
        print(f"[{i+1}/{len(dates)}] {date}: {len(day_df):,} rows, {elapsed:.1f}s, RSS {cur_rss:.0f}MB")
        gc.collect()

    print("Concatenating all days...")
    t_concat0 = time.time()
    full = pd.concat(day_frames, ignore_index=True)
    del day_frames
    gc.collect()
    concat_elapsed = time.time() - t_concat0
    peak_rss = max(peak_rss, rss_mb())

    full = full.sort_values(["square_id", "timestamp"]).reset_index(drop=True)

    print(f"Full dataset: {len(full):,} rows, in-memory size {full.memory_usage(deep=True).sum()/1e6:.1f} MB")

    t_write0 = time.time()
    full.to_parquet(OUT_DIR / "internet_traffic.parquet", index=False, compression="snappy")
    running_total.rename(columns={"internet": "total_internet"}).sort_values(
        "total_internet", ascending=False
    ).reset_index(drop=True).to_parquet(OUT_DIR / "total_traffic_per_square.parquet", index=False)
    write_elapsed = time.time() - t_write0

    total_elapsed = time.time() - t_start
    parquet_size_mb = (OUT_DIR / "internet_traffic.parquet").stat().st_size / 1e6

    stats = {
        "n_days": len(dates),
        "date_range": [dates[0], dates[-1]],
        "total_rows": int(len(full)),
        "in_memory_size_mb": float(full.memory_usage(deep=True).sum() / 1e6),
        "peak_process_rss_mb": float(peak_rss),
        "concat_elapsed_sec": concat_elapsed,
        "parquet_write_elapsed_sec": write_elapsed,
        "total_elapsed_sec": total_elapsed,
        "parquet_file_size_mb": parquet_size_mb,
        "per_day_stats": per_day_stats,
    }
    with open(INTERIM_DIR / "etl_run_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nDone in {total_elapsed/60:.1f} min. Peak RSS: {peak_rss:.0f} MB. "
          f"Parquet size: {parquet_size_mb:.1f} MB (vs ~20GB raw text).")


if __name__ == "__main__":
    main()
