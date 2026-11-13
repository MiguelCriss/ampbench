"""Feature builders shared by ANN and classical PA models."""
from __future__ import annotations

import numpy as np


def memory_matrix(signal: np.ndarray, memory: int) -> np.ndarray:
    """Build a delay-line matrix of shape (L, memory).

    Column ``i`` contains ``signal`` delayed by ``i`` samples (zero-padded).
    """
    if memory < 1:
        raise ValueError("memory must be >= 1")
    x = np.asarray(signal, dtype=float)
    L = len(x)
    X = np.zeros((L, memory), dtype=float)
    for i in range(memory):
        X[i:, i] = x[: L - i]
    return X


def mpm_design_matrix(x: np.ndarray, order: int, memory: int) -> np.ndarray:
    """Memory polynomial design matrix (complex).

    For each delay ``m`` in ``0..memory`` and nonlinearity ``k`` in ``0..order``:
    ``phi = x[n-m] * |x[n-m]|**k``.

    Column index is ``m * (order + 1) + k`` (fixed vs legacy ``m * order + k``,
    which collapses columns when order=0).
    """
    if order < 0 or memory < 0:
        raise ValueError("order and memory must be >= 0")
    x = np.asarray(x, dtype=np.complex128)
    L = len(x)
    n_cols = (order + 1) * (memory + 1)
    X = np.zeros((L, n_cols), dtype=np.complex128)
    for n in range(L):
        for m in range(memory + 1):
            if n - m < 0:
                continue
            xm = x[n - m]
            ax = np.abs(xm)
            for k in range(order + 1):
                X[n, m * (order + 1) + k] = xm * (ax**k)
    return X


def fir_design_matrix(signal: np.ndarray, memory: int) -> np.ndarray:
    """Toeplitz / FIR delay matrix with ``memory + 1`` taps."""
    x = np.asarray(signal)
    L = len(x)
    X = np.zeros((L, memory + 1), dtype=np.complex128 if np.iscomplexobj(x) else float)
    for i in range(memory + 1):
        X[i:, i] = x[: L - i]
    return X
