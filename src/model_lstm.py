"""Model 2: stacked LSTM for one-step-ahead forecasting.

Input representation: a sliding window of the last `window` scaled 10-min
internet-traffic readings (univariate). Output: a single Dense unit
predicting the next 10-min value. Chosen because recurrent gating lets the
network learn nonlinear short- and medium-range temporal dependencies
directly from data, without the linear/additive-seasonality assumptions of
the harmonic-regression baseline (see EDA: traffic is non-Gaussian and
strongly right-skewed with recurring but not perfectly regular daily peaks).
"""
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from data_prep import prepare_dl_dataset


def build_model(window, units=(64, 32), dropout=0.1):
    inputs = keras.Input(shape=(window, 1))
    x = inputs
    for i, u in enumerate(units):
        x = layers.LSTM(u, return_sequences=(i < len(units) - 1))(x)
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model


def train_and_forecast(series, window=144, units=(64, 32), dropout=0.1,
                        epochs=40, batch_size=256, patience=5, seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    data = prepare_dl_dataset(series, window=window)
    X_train, y_train, _ = data["train"]
    X_val, y_val, _ = data["val"]
    X_test, y_test, t_test = data["test"]
    scaler = data["scaler"]

    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    model = build_model(window, units, dropout)
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)

    t_fit0 = time.time()
    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=0,
    )
    fit_time = time.time() - t_fit0

    t_pred0 = time.time()
    y_pred_scaled = model.predict(X_test, verbose=0).ravel()
    predict_time = time.time() - t_pred0

    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_true = scaler.inverse_transform(y_test)

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "index": t_test,
        "fit_time_sec": fit_time,
        "predict_time_sec": predict_time,
        "n_epochs_trained": len(history.history["loss"]),
        "val_loss_history": history.history["val_loss"],
        "n_params": model.count_params(),
        "window": window,
        "units": units,
    }
