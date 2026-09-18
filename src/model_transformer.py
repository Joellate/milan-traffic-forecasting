"""Model 3: small Transformer encoder for one-step-ahead forecasting.

Same windowed univariate input representation as the LSTM (last `window`
scaled readings), but temporal dependencies are captured with multi-head
self-attention instead of recurrence. Attention lets the model weigh any
past time step directly (e.g. the same slot 144 steps ago for daily
periodicity) without propagating information sequentially through a hidden
state, which recurrent models can struggle to preserve over long windows
(Vaswani et al., 2017). A fixed sinusoidal positional encoding is added
since self-attention has no inherent notion of order.
"""
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from data_prep import prepare_dl_dataset


def positional_encoding(length, depth):
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :] / depth
    angle_rates = 1 / (10000 ** depths)
    angle_rads = positions * angle_rates
    pos_encoding = np.concatenate([np.sin(angle_rads[:, 0::2] if depth % 2 == 0 else angle_rads),
                                    np.cos(angle_rads[:, 0::2] if depth % 2 == 0 else angle_rads)], axis=-1)
    return tf.cast(pos_encoding[np.newaxis, :, :depth], dtype=tf.float32)


def transformer_encoder_block(x, d_model, num_heads, ff_dim, dropout):
    attn_out = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout)(x, x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn_out)
    ff = layers.Dense(ff_dim, activation="relu")(x)
    ff = layers.Dense(d_model)(ff)
    ff = layers.Dropout(dropout)(ff)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ff)
    return x


def build_model(window, d_model=32, num_heads=4, ff_dim=64, num_layers=2, dropout=0.1):
    inputs = keras.Input(shape=(window, 1))
    x = layers.Dense(d_model)(inputs)
    x = x + positional_encoding(window, d_model)
    for _ in range(num_layers):
        x = transformer_encoder_block(x, d_model, num_heads, ff_dim, dropout)
    # Use the representation at the most recent time step rather than average-
    # pooling across the whole window: the EDA PACF (Section 2) showed the
    # partial autocorrelation of this series cuts off sharply after 2-3 lags,
    # so most of the one-step-ahead signal sits at the end of the window and
    # averaging over all 144 steps was found to dilute it (see model iteration
    # log: MAPE dropped substantially after this change).
    x = x[:, -1, :]
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model


def train_and_forecast(series, window=144, d_model=32, num_heads=4, ff_dim=64,
                        num_layers=2, dropout=0.1, epochs=40, batch_size=256,
                        patience=5, seed=42):
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

    model = build_model(window, d_model, num_heads, ff_dim, num_layers, dropout)
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
    }
