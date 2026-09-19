"""Systematic LSTM hyperparameter search on square 5161 validation MAE.

Re-runs the reduced grid documented in data/interim/lstm_tuning.json when
processed Parquet is available. Selection criterion: minimum validation MAE
on the Dec 9–15 hold-out (see data_prep.prepare_dl_dataset).

Usage (from project root, after ETL):
  python src/tune_lstm.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from data_prep import load_square_series, prepare_dl_dataset
from model_lstm import build_model

INTERIM = Path("data/interim")
INTERIM.mkdir(parents=True, exist_ok=True)
OUT = INTERIM / "lstm_tuning.json"

# Reduced grid — full Cartesian product is too expensive on CPU; this matches
# the candidates reported in the research report.
CANDIDATES = [
    {"window": 72, "units": (64, 32), "dropout": 0.1, "batch_size": 256},
    {"window": 288, "units": (64, 32), "dropout": 0.1, "batch_size": 256},
    {"window": 144, "units": (32, 16), "dropout": 0.1, "batch_size": 256},
    {"window": 144, "units": (128, 64), "dropout": 0.1, "batch_size": 256},
    {"window": 144, "units": (64, 32), "dropout": 0.2, "batch_size": 256},
    {"window": 144, "units": (64, 32), "dropout": 0.1, "batch_size": 128},
    {"window": 144, "units": (64, 32), "dropout": 0.1, "batch_size": 256},
]


def eval_candidate(series, cfg, seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    window = cfg["window"]
    data = prepare_dl_dataset(series, window=window)
    X_train, y_train, _ = data["train"]
    X_val, y_val, _ = data["val"]
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]

    model = build_model(window, cfg["units"], cfg["dropout"])
    es = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=40,
        batch_size=cfg["batch_size"],
        callbacks=[es],
        verbose=0,
    )
    val_mae = float(min(history.history["val_mae"]))
    return {
        "window": window,
        "units": list(cfg["units"]),
        "dropout": cfg["dropout"],
        "batch_size": cfg["batch_size"],
        "val_mae": val_mae,
        "epochs_used": len(history.history["loss"]),
    }


def main():
    series = load_square_series(5161)
    results = []
    for cfg in CANDIDATES:
        print(f"Evaluating {cfg} ...")
        row = eval_candidate(series, cfg)
        results.append(row)
        print(f"  val_mae={row['val_mae']:.5f} epochs={row['epochs_used']}")

    best = min(results, key=lambda r: r["val_mae"])
    for r in results:
        r["selected"] = (
            r["window"] == best["window"]
            and r["units"] == best["units"]
            and r["dropout"] == best["dropout"]
            and r["batch_size"] == best["batch_size"]
        )

    payload = {
        "area": 5161,
        "selection_criterion": "minimum mean validation MAE on Dec 9–15 hold-out",
        "results": results,
        "selected": {
            "window": best["window"],
            "units": best["units"],
            "dropout": best["dropout"],
            "batch_size": best["batch_size"],
            "epochs_max": 40,
            "patience": 5,
        },
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")
    print("Selected:", payload["selected"])


if __name__ == "__main__":
    main()
