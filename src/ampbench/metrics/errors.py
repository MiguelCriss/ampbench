"""Error metrics for complex and real signals."""
from __future__ import annotations

import numpy as np

SUPPORTED = ("mse", "rmse", "mae", "mape", "nmse", "nmse_db", "medae")


def calculate_error(y_true, y_pred, metric: str = "nmse_db") -> float:
    """Compute an error metric between predictions and ground truth.

    Accepts real or complex arrays. Complex errors use magnitude of the residual.
    ``nmse_db`` matches the thesis reporting convention (10*log10(mse/var(|y|))).
    """
    metric_key = "nmse_db" if metric == "NMSE_dB" else metric.lower()

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    magnitude_error = np.abs(y_true - y_pred)

    if metric_key == "mse":
        return float(np.mean(magnitude_error**2))
    if metric_key == "rmse":
        return float(np.sqrt(np.mean(magnitude_error**2)))
    if metric_key == "mae":
        return float(np.mean(magnitude_error))
    if metric_key == "mape":
        return float(np.mean(magnitude_error / np.maximum(np.abs(y_true), 1e-12)) * 100)
    if metric_key == "nmse":
        variance = float(np.var(np.abs(y_true)))
        return float(np.mean(magnitude_error**2) / max(variance, 1e-30))
    if metric_key == "nmse_db":
        variance = float(np.var(np.abs(y_true)))
        nmse = float(np.mean(magnitude_error**2) / max(variance, 1e-30))
        return float(10.0 * np.log10(max(nmse, 1e-30)))
    if metric_key == "medae":
        return float(np.median(magnitude_error))

    raise ValueError(f"Unknown metric {metric!r}. Supported: {SUPPORTED}")
