"""Orchestrates the full Section 4 forecasting experiment: 3 models x 3
geographical areas. Produces, per (model, area):
  - a superposed actual-vs-predicted plot for Dec 16-22
  - MAE/RMSE/MAPE/R²
  - fit/predict timing
Writes:
  reports/figures/forecast_<model>_<area_label>.png   (9 plots)
  reports/tables/metrics_<area_label>.csv             (3 tables)
  reports/tables/timing.csv
  data/interim/experiment_results.json
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data_prep import load_square_series
from evaluate import all_metrics
import model_sarima
import model_lstm
import model_transformer

FIG_DIR = Path("reports/figures")
TAB_DIR = Path("reports/tables")
INTERIM_DIR = Path("data/interim")
for d in (FIG_DIR, TAB_DIR, INTERIM_DIR):
    d.mkdir(parents=True, exist_ok=True)

AREAS = [
    {"square_id": 5161, "label": "top1_5161", "display": "Square 5161 (highest total traffic)"},
    {"square_id": 4159, "label": "sq4159", "display": "Square 4159"},
    {"square_id": 4556, "label": "sq4556", "display": "Square 4556"},
]

SARIMA_ORDER = (3, 0, 1)  # selected via grid search on area 5161 (AIC), see sarima_order_selection.json
                           # (3,0,2) had marginally lower AIC (88018.7 vs 88019.0) but (3,0,1) is
                           # preferred for parsimony given the negligible difference
LSTM_CFG = dict(window=144, units=(64, 32), dropout=0.1, epochs=40, batch_size=256, patience=5)
TRANSFORMER_CFG = dict(window=144, d_model=32, num_heads=4, ff_dim=64, num_layers=2, dropout=0.1,
                        epochs=25, batch_size=512, patience=4)


def plot_forecast(index, y_true, y_pred, title, path):
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(index, y_true, label="Actual", color="#2c3e50", linewidth=1.0)
    ax.plot(index, y_pred, label="Predicted", color="#e67e22", linewidth=1.0, alpha=0.85)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Date")
    ax.set_ylabel("Internet traffic")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_area(area):
    square_id = area["square_id"]
    label = area["label"]
    print(f"\n=== Area {label} (square {square_id}) ===")
    series = load_square_series(square_id)

    results = {}

    print("Fitting SARIMA/DHR...")
    r = model_sarima.fit_and_forecast(series, order=SARIMA_ORDER)
    metrics = all_metrics(r["y_true"], r["y_pred"])
    results["dhr_arma"] = {**metrics, "fit_time_sec": r["fit_time_sec"], "predict_time_sec": r["predict_time_sec"],
                            "order": r["order"], "aic": r["aic"], "n_params": r["n_params"]}
    plot_forecast(r["index"], r["y_true"], r["y_pred"],
                  f"DHR-ARMA — {area['display']} — Dec 16-22, 2013",
                  FIG_DIR / f"forecast_dhr_arma_{label}.png")
    print(metrics)

    print("Training LSTM...")
    r = model_lstm.train_and_forecast(series, **LSTM_CFG)
    metrics = all_metrics(r["y_true"], r["y_pred"])
    results["lstm"] = {**metrics, "fit_time_sec": r["fit_time_sec"], "predict_time_sec": r["predict_time_sec"],
                        "n_epochs_trained": r["n_epochs_trained"], "n_params": r["n_params"]}
    plot_forecast(r["index"], r["y_true"], r["y_pred"],
                  f"LSTM — {area['display']} — Dec 16-22, 2013",
                  FIG_DIR / f"forecast_lstm_{label}.png")
    print(metrics)

    print("Training Transformer...")
    r = model_transformer.train_and_forecast(series, **TRANSFORMER_CFG)
    metrics = all_metrics(r["y_true"], r["y_pred"])
    results["transformer"] = {**metrics, "fit_time_sec": r["fit_time_sec"], "predict_time_sec": r["predict_time_sec"],
                               "n_epochs_trained": r["n_epochs_trained"], "n_params": r["n_params"]}
    plot_forecast(r["index"], r["y_true"], r["y_pred"],
                  f"Transformer — {area['display']} — Dec 16-22, 2013",
                  FIG_DIR / f"forecast_transformer_{label}.png")
    print(metrics)

    return results


def main():
    all_results = {}
    for area in AREAS:
        t0 = time.time()
        all_results[area["label"]] = run_area(area)
        all_results[area["label"]]["_area_total_time_sec"] = time.time() - t0

    with open(INTERIM_DIR / "experiment_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    for area in AREAS:
        label = area["label"]
        rows = []
        for model_name, m in all_results[label].items():
            if model_name.startswith("_"):
                continue
            rows.append({"model": model_name, "MAE": m["MAE"], "RMSE": m["RMSE"],
                           "MAPE": m["MAPE"], "R2": m["R2"]})
        pd.DataFrame(rows).to_csv(TAB_DIR / f"metrics_{label}.csv", index=False)

    timing_rows = []
    for area in AREAS:
        label = area["label"]
        for model_name, m in all_results[label].items():
            if model_name.startswith("_"):
                continue
            timing_rows.append({
                "area": label, "model": model_name,
                "fit_time_sec": m["fit_time_sec"], "predict_time_sec": m["predict_time_sec"],
            })
    pd.DataFrame(timing_rows).to_csv(TAB_DIR / "timing.csv", index=False)

    print("\nAll experiments complete.")
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
