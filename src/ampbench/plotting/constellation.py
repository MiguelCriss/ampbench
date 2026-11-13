"""Honest I/Q constellation plots (no synthetic jitter)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ampbench.plotting.utils import finalize_figure


def plot_constellation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: str = "I/Q constellation",
    model_label: str = "Model",
    max_points: int = 5000,
    show: bool = False,
    save_path: Path | str | None = None,
) -> Figure:
    """Plot raw complex baseband samples — measured vs predicted only."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = min(len(y_true), len(y_pred))
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points).astype(int)
    else:
        idx = np.arange(n)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(np.real(y_true[idx]), np.imag(y_true[idx]), ".", ms=2, alpha=0.4, label="PA measured")
    ax.plot(np.real(y_pred[idx]), np.imag(y_pred[idx]), ".", ms=2, alpha=0.4, label=model_label)
    ax.set_xlabel("In-phase (I)")
    ax.set_ylabel("Quadrature (Q)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(markerscale=4)
    return finalize_figure(fig, show=show, save_path=save_path)
