"""Normalization and ANN feature preparation from I/Q datasets."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ampbench.data.iq import IQDataset
from ampbench.models.features import memory_matrix


@dataclass(frozen=True)
class MinMaxScaler1D:
    lo: float
    hi: float

    def transform(self, x: np.ndarray) -> np.ndarray:
        span = self.hi - self.lo
        if span == 0:
            return np.zeros_like(x, dtype=float)
        return (np.asarray(x, dtype=float) - self.lo) / span

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float) * (self.hi - self.lo) + self.lo


@dataclass
class IQNormStats:
    i_in: MinMaxScaler1D
    q_in: MinMaxScaler1D
    i_out: MinMaxScaler1D
    q_out: MinMaxScaler1D


@dataclass
class ANNBundle:
    """Normalized memory features for ANN training/eval."""

    X_train: np.ndarray
    Y_train: np.ndarray
    X_valid: np.ndarray
    Y_valid: np.ndarray
    X_test: np.ndarray
    Y_test: np.ndarray
    X_full: np.ndarray
    stats: IQNormStats
    memory: int


def _fit_scaler(x: np.ndarray) -> MinMaxScaler1D:
    return MinMaxScaler1D(float(np.min(x)), float(np.max(x)))


def _scale_keep_zero(x: np.ndarray, scaler: MinMaxScaler1D) -> np.ndarray:
    out = scaler.transform(x)
    return np.where(np.asarray(x) == 0, 0.0, out)


def build_ann_bundle(
    train: IQDataset,
    valid: IQDataset,
    test: IQDataset,
    full: IQDataset,
    memory: int = 7,
) -> ANNBundle:
    """Create ANN inputs/targets from I/Q splits (thesis convention)."""
    stats = IQNormStats(
        i_in=_fit_scaler(train.i_in),
        q_in=_fit_scaler(train.q_in),
        i_out=_fit_scaler(train.i_out),
        q_out=_fit_scaler(train.q_out),
    )

    def pack_X(ds: IQDataset) -> np.ndarray:
        i_n = stats.i_in.transform(ds.i_in)
        q_n = _scale_keep_zero(ds.q_in, stats.q_in)
        return np.hstack((memory_matrix(i_n, memory), memory_matrix(q_n, memory)))

    def pack_Y(ds: IQDataset) -> np.ndarray:
        return np.column_stack(
            (
                stats.i_out.transform(ds.i_out),
                stats.q_out.transform(ds.q_out),
            )
        )

    return ANNBundle(
        X_train=pack_X(train),
        Y_train=pack_Y(train),
        X_valid=pack_X(valid),
        Y_valid=pack_Y(valid),
        X_test=pack_X(test),
        Y_test=pack_Y(test),
        X_full=pack_X(full),
        stats=stats,
        memory=memory,
    )


def denormalize_iq_prediction(
    y_norm: np.ndarray,
    stats: IQNormStats,
) -> tuple[np.ndarray, np.ndarray]:
    """Map ANN normalized outputs back to I/Q."""
    i = stats.i_out.inverse(y_norm[:, 0])
    q = stats.q_out.inverse(y_norm[:, 1])
    return i, q
