"""ANN behavioral PA model (Keras)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ampbench.models.preprocessing import ANNBundle, denormalize_iq_prediction


def nmse_db_loss(y_true, y_pred):
    """Keras/TF NMSE loss in dB (matches thesis training objective)."""
    import tensorflow as tf

    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    power_true = tf.reduce_mean(tf.square(y_true))
    nmse = mse / (power_true + 1e-12)
    return 10.0 * tf.math.log(nmse) / tf.math.log(10.0)


def _require_keras():
    try:
        from keras import Input
        from keras.layers import Dense
        from keras.models import Sequential
        from keras.optimizers import Adam
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "ANN training requires TensorFlow/Keras. Install with: pip install -e \".[ann]\""
        ) from exc
    return Input, Dense, Sequential, Adam


def build_ann(memory: int, hidden=(64, 32), learning_rate: float = 1e-4):
    Input, Dense, Sequential, Adam = _require_keras()

    layers = [Input(shape=(2 * memory,)), Dense(hidden[0], activation="relu")]
    for units in hidden[1:]:
        layers.append(Dense(units, activation="relu"))
    layers.append(Dense(2, activation="linear"))
    model = Sequential(layers)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=nmse_db_loss, metrics=["mae"])
    return model


@dataclass
class ANNTrainResult:
    model: object
    history: dict
    bundle: ANNBundle


def train_ann(
    bundle: ANNBundle,
    epochs: int = 300,
    batch_size: int = 320,
    learning_rate: float = 1e-4,
    hidden: tuple[int, ...] = (64, 32),
    verbose: int = 1,
) -> ANNTrainResult:
    model = build_ann(bundle.memory, hidden=hidden, learning_rate=learning_rate)
    hist = model.fit(
        bundle.X_train,
        bundle.Y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(bundle.X_valid, bundle.Y_valid),
        verbose=verbose,
    )
    return ANNTrainResult(model=model, history=dict(hist.history), bundle=bundle)


def predict_iq(model, X: np.ndarray, bundle: ANNBundle) -> tuple[np.ndarray, np.ndarray]:
    y_norm = model.predict(X, verbose=0)
    return denormalize_iq_prediction(y_norm, bundle.stats)


def save_ann(model, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)


def load_ann(path: Path):
    from keras.models import load_model

    return load_model(path, custom_objects={"nmse_db_loss": nmse_db_loss})
