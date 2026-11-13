"""Classical behavioral PA models: MPM, Hammerstein, Wiener."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from tqdm import tqdm

from ampbench.metrics.errors import calculate_error
from ampbench.models.features import fir_design_matrix, mpm_design_matrix

ModelName = Literal["MPM", "Hammerstein", "Wiener"]


@dataclass
class TraditionalResult:
    model: ModelName
    order: int  # N
    memory: int  # M
    coefficients: dict
    y_pred: np.ndarray
    train_error: float | None = None
    valid_error: float | None = None
    test_error: float | None = None


def fit_mpm(x: np.ndarray, y: np.ndarray, order: int, memory: int) -> np.ndarray:
    X = mpm_design_matrix(x, order, memory)
    coef, *_ = np.linalg.lstsq(X, np.asarray(y, dtype=np.complex128), rcond=None)
    return coef


def predict_mpm(x: np.ndarray, coef: np.ndarray, order: int, memory: int) -> np.ndarray:
    return mpm_design_matrix(x, order, memory) @ coef


def _poly_fit(x: np.ndarray, y: np.ndarray, order: int) -> np.ndarray:
    X = np.vstack([np.asarray(x, dtype=np.complex128) ** k for k in range(order + 1)]).T
    b, *_ = np.linalg.lstsq(X, np.asarray(y, dtype=np.complex128), rcond=None)
    return b


def _poly_apply(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    # np.polyval expects highest degree first
    return np.polyval(b[::-1], np.asarray(x, dtype=np.complex128))


def _fir_fit(x: np.ndarray, y: np.ndarray, memory: int) -> np.ndarray:
    X = fir_design_matrix(x, memory)
    h, *_ = np.linalg.lstsq(X, np.asarray(y, dtype=np.complex128), rcond=None)
    return h


def _fir_apply(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    memory = len(h) - 1
    return fir_design_matrix(x, memory) @ h


def fit_hammerstein(x: np.ndarray, y: np.ndarray, order: int, memory: int) -> tuple[np.ndarray, np.ndarray]:
    b = _poly_fit(x, y, order)
    ynl = _poly_apply(x, b)
    h = _fir_fit(ynl, y, memory)
    return b, h


def predict_hammerstein(x: np.ndarray, b: np.ndarray, h: np.ndarray) -> np.ndarray:
    return _fir_apply(_poly_apply(x, b), h)


def fit_wiener(x: np.ndarray, y: np.ndarray, order: int, memory: int) -> tuple[np.ndarray, np.ndarray]:
    h = _fir_fit(x, y, memory)
    ylti = _fir_apply(x, h)
    b = _poly_fit(ylti, y, order)
    return h, b


def predict_wiener(x: np.ndarray, h: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _poly_apply(_fir_apply(x, h), b)


def fit_traditional(
    model: ModelName,
    x_train: np.ndarray,
    y_train: np.ndarray,
    order: int,
    memory: int,
    x_valid: np.ndarray | None = None,
    y_valid: np.ndarray | None = None,
    x_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    metric: str = "nmse_db",
) -> TraditionalResult:
    if model == "MPM":
        coef = fit_mpm(x_train, y_train, order, memory)
        coefficients = {"a": coef}
        predict = lambda x: predict_mpm(x, coef, order, memory)
    elif model == "Hammerstein":
        b, h = fit_hammerstein(x_train, y_train, order, memory)
        coefficients = {"b": b, "h": h}
        predict = lambda x: predict_hammerstein(x, b, h)
    elif model == "Wiener":
        h, b = fit_wiener(x_train, y_train, order, memory)
        coefficients = {"h": h, "b": b}
        predict = lambda x: predict_wiener(x, h, b)
    else:
        raise ValueError(f"Unknown model {model!r}")

    y_hat_train = predict(x_train)
    result = TraditionalResult(
        model=model,
        order=order,
        memory=memory,
        coefficients=coefficients,
        y_pred=y_hat_train,
        train_error=calculate_error(y_train, y_hat_train, metric),
    )
    if x_valid is not None and y_valid is not None:
        result.valid_error = calculate_error(y_valid, predict(x_valid), metric)
    if x_test is not None and y_test is not None:
        y_test_hat = predict(x_test)
        result.test_error = calculate_error(y_test, y_test_hat, metric)
        result.y_pred = y_test_hat
    return result


def grid_search_traditional(
    model: ModelName,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    orders: list[int],
    memories: list[int],
    metric: str = "nmse_db",
) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int], dict]]:
    """Sweep N/M without forcing plots. Returns train/valid error grids and coef map."""
    err_train = np.full((len(orders), len(memories)), np.nan)
    err_valid = np.full((len(orders), len(memories)), np.nan)
    coef_map: dict[tuple[int, int], dict] = {}

    total = len(orders) * len(memories)
    with tqdm(total=total, desc=f"{model} grid search") as pbar:
        for i, n in enumerate(orders):
            for j, m in enumerate(memories):
                try:
                    res = fit_traditional(
                        model,
                        x_train,
                        y_train,
                        order=n,
                        memory=m,
                        x_valid=x_valid,
                        y_valid=y_valid,
                        metric=metric,
                    )
                    err_train[i, j] = res.train_error
                    err_valid[i, j] = res.valid_error
                    coef_map[(n, m)] = res.coefficients
                except Exception as exc:  # noqa: BLE001 - collect and continue grid
                    err_train[i, j] = np.nan
                    err_valid[i, j] = np.nan
                    coef_map[(n, m)] = {"error": str(exc)}
                pbar.update(1)
    return err_train, err_valid, coef_map


def predict_traditional(
    model: ModelName,
    x: np.ndarray,
    coefficients: dict,
    order: int,
    memory: int,
) -> np.ndarray:
    if model == "MPM":
        return predict_mpm(x, coefficients["a"], order, memory)
    if model == "Hammerstein":
        return predict_hammerstein(x, coefficients["b"], coefficients["h"])
    if model == "Wiener":
        return predict_wiener(x, coefficients["h"], coefficients["b"])
    raise ValueError(f"Unknown model {model!r}")
