"""AM/AM, AM/PM and gain characteristic plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ampbench.metrics.signals import EnvelopePhase, power_dbm_from_voltage
from ampbench.plotting.utils import finalize_figure


def plot_amam_ampm(
    env: EnvelopePhase,
    *,
    title: str = "PA AM/AM and AM/PM",
    model_label: str = "Model",
    max_points: int = 4000,
    show: bool = False,
    save_path: Path | str | None = None,
) -> Figure:
    """Scatter Vin–Vout and Vin–phase (true vs predicted)."""
    idx = _subsample(len(env.vin), max_points)
    vin = env.vin[idx]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(vin, env.vout_true[idx], ".", ms=2, alpha=0.5, label="PA measured")
    axes[0].plot(vin, env.vout_pred[idx], ".", ms=2, alpha=0.5, label=model_label)
    axes[0].set_xlabel("Vin")
    axes[0].set_ylabel("Vout")
    axes[0].set_title("AM/AM")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(markerscale=4)

    axes[1].plot(vin, np.degrees(env.phase_true_rad[idx]), ".", ms=2, alpha=0.5, label="PA measured")
    axes[1].plot(vin, np.degrees(env.phase_pred_rad[idx]), ".", ms=2, alpha=0.5, label=model_label)
    axes[1].set_xlabel("Vin")
    axes[1].set_ylabel("Relative phase [deg]")
    axes[1].set_title("AM/PM")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(markerscale=4)

    fig.suptitle(title)
    return finalize_figure(fig, show=show, save_path=save_path)


def plot_power_characteristics(
    env: EnvelopePhase,
    *,
    title: str = "PA power characteristics",
    model_label: str = "Model",
    max_points: int = 4000,
    show: bool = False,
    save_path: Path | str | None = None,
) -> Figure:
    """Pin/Pout, gain, and phase vs Pin — no hidden exports."""
    idx = _subsample(len(env.vin), max_points)
    pin = power_dbm_from_voltage(env.vin[idx])
    pout_t = power_dbm_from_voltage(env.vout_true[idx])
    pout_p = power_dbm_from_voltage(env.vout_pred[idx])
    gain_t = pout_t - pin
    gain_p = pout_p - pin
    phase_t = np.degrees(env.phase_true_rad[idx])
    phase_p = np.degrees(env.phase_pred_rad[idx])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].plot(pin, pout_t, ".", ms=2, alpha=0.5, label="PA measured")
    axes[0].plot(pin, pout_p, ".", ms=2, alpha=0.5, label=model_label)
    axes[0].set_xlabel("Pin [dBm]")
    axes[0].set_ylabel("Pout [dBm]")
    axes[0].set_title("Pout vs Pin")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(pin, gain_t, ".", ms=2, alpha=0.5, label="PA measured")
    axes[1].plot(pin, gain_p, ".", ms=2, alpha=0.5, label=model_label)
    axes[1].set_xlabel("Pin [dBm]")
    axes[1].set_ylabel("Gain [dB]")
    axes[1].set_title("Gain vs Pin")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(pin, phase_t, ".", ms=2, alpha=0.5, label="PA measured")
    axes[2].plot(pin, phase_p, ".", ms=2, alpha=0.5, label=model_label)
    axes[2].set_xlabel("Pin [dBm]")
    axes[2].set_ylabel("Relative phase [deg]")
    axes[2].set_title("Phase vs Pin")
    axes[2].grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, markerscale=4, frameon=True)
    fig.suptitle(title, y=1.02)
    return finalize_figure(fig, show=show, save_path=save_path)


def _subsample(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    return np.linspace(0, n - 1, max_points).astype(int)
