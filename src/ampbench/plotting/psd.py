"""Power spectral density plots without cosmetic residual scaling."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ampbench.metrics.signals import baseband_psd_db
from ampbench.plotting.utils import finalize_figure


def plot_psd_comparison(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    sample_rate_hz: float = 100e6,
    smooth_window: int = 100,
    title: str = "Output PSD",
    model_label: str = "Model",
    show_error: bool = True,
    show: bool = False,
    save_path: Path | str | None = None,
) -> Figure:
    """Compare measured vs predicted PSD.

    The residual spectrum is the PSD of ``(y_true - y_pred)`` with **no**
    multiplicative/additive cosmetic offsets.
    """
    f_t, psd_t = baseband_psd_db(y_true, sample_rate_hz, smooth_window)
    f_p, psd_p = baseband_psd_db(y_pred, sample_rate_hz, smooth_window)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(f_t, psd_t, label="PA measured", lw=1.2)
    ax.plot(f_p, psd_p, label=model_label, lw=1.2)
    if show_error:
        f_e, psd_e = baseband_psd_db(np.asarray(y_true) - np.asarray(y_pred), sample_rate_hz, smooth_window)
        ax.plot(f_e, psd_e, label="Residual (true − pred)", lw=1.0, alpha=0.9)
    ax.set_xlabel("Frequency offset [MHz]")
    ax.set_ylabel("PSD [dB]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return finalize_figure(fig, show=show, save_path=save_path)
