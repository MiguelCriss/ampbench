"""Time-domain comparison plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ampbench.plotting.utils import finalize_figure


def plot_vout_timeseries(
    time: np.ndarray,
    vout_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    *,
    title: str = "Vout time series",
    t_max: float | None = None,
    scale: float = 1.0,
    ylabel: str = "Vout",
    show: bool = False,
    save_path: Path | str | None = None,
) -> Figure:
    """Overlay measured Vout and one or more model predictions."""
    time = np.asarray(time)
    mask = slice(None) if t_max is None else time <= t_max

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(time[mask], np.asarray(vout_true)[mask] * scale, "-", lw=1.2, label="PA measured")
    for label, series in predictions.items():
        ax.plot(time[mask], np.asarray(series)[mask] * scale, ".", ms=3, alpha=0.6, label=label)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return finalize_figure(fig, show=show, save_path=save_path)
