"""Model 1: Dynamic Harmonic Regression (Fourier terms for daily + weekly
seasonality) with ARMA errors, fit via statsmodels SARIMAX.

Rationale (see report Section "Methodology"): a full seasonal ARIMA with
seasonal order at lag 144 (10-min data -> daily period) is computationally
prohibitive to fit by exact MLE over multi-week training windows, because
the state-space dimension used by the Kalman filter scales with the
seasonal period. Following Hyndman & Athanasopoulos (Forecasting: Principles
and Practice, ch. 12) and De Livera et al. (2011, JASA) on complex/multiple
seasonality, daily and weekly periodicity are instead captured with Fourier
regressors, and a low-order ARMA model captures the remaining short-range
autocorrelation in the residuals. This is our classical statistical
baseline, contrasted with the two deep sequential models.
"""
import time

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from data_prep import build_fourier_exog, train_test_split


def grid_search_order(train: pd.Series, exog: pd.DataFrame, candidates, d=0):
    """Small AIC-based grid search over (p, q) with fixed d, documented as the
    'iterative experimentation' hyperparameter tuning step for this model."""
    results = []
    for p, q in candidates:
        try:
            mod = SARIMAX(train.values, exog=exog.values, order=(p, d, q),
                           enforce_stationarity=False, enforce_invertibility=False)
            res = mod.fit(disp=False, maxiter=50)
            results.append({"p": p, "d": d, "q": q, "aic": float(res.aic)})
        except Exception as e:
            results.append({"p": p, "d": d, "q": q, "aic": None, "error": str(e)})
    return sorted([r for r in results if r["aic"] is not None], key=lambda r: r["aic"]), results


def fit_and_forecast(series: pd.Series, order=(2, 0, 1), k_daily=4, k_weekly=3):
    train, test = train_test_split(series)
    t0 = train.index.min()
    exog_train = build_fourier_exog(train.index, t0, k_daily, k_weekly)
    exog_test = build_fourier_exog(test.index, t0, k_daily, k_weekly)

    t_fit0 = time.time()
    mod_train = SARIMAX(train.values, exog=exog_train.values, order=order,
                         enforce_stationarity=False, enforce_invertibility=False)
    res_train = mod_train.fit(disp=False, maxiter=100)
    fit_time = time.time() - t_fit0

    full = pd.concat([train, test])
    exog_full = build_fourier_exog(full.index, t0, k_daily, k_weekly)
    t_pred0 = time.time()
    mod_full = SARIMAX(full.values, exog=exog_full.values, order=order,
                        enforce_stationarity=False, enforce_invertibility=False)
    res_full = mod_full.filter(res_train.params)
    start = len(train)
    end = len(full) - 1
    pred = res_full.get_prediction(start=start, end=end, dynamic=False)
    y_pred = pred.predicted_mean
    predict_time = time.time() - t_pred0

    return {
        "y_true": test.values,
        "y_pred": np.asarray(y_pred),
        "index": test.index,
        "fit_time_sec": fit_time,
        "predict_time_sec": predict_time,
        "order": order,
        "aic": float(res_train.aic),
        "n_params": len(res_train.params),
    }
