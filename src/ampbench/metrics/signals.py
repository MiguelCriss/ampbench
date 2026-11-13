"""Signal transforms used for honest evaluation plots."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EnvelopePhase:
    vin: np.ndarray
    vout_true: np.ndarray
    vout_pred: np.ndarray
    phase_true_rad: np.ndarray
    phase_pred_rad: np.ndarray


def complex_envelopes(x: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> EnvelopePhase:
    """Derive Vin/Vout and relative phase from complex baseband I/Q.

    Relative phase uses ``angle(y * conj(x))`` (no cosmetic clamping or jitter).
    """
    x = np.asarray(x)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return EnvelopePhase(
        vin=np.abs(x),
        vout_true=np.abs(y_true),
        vout_pred=np.abs(y_pred),
        phase_true_rad=np.angle(y_true * np.conj(x)),
        phase_pred_rad=np.angle(y_pred * np.conj(x)),
    )


def power_dbm_from_voltage(v: np.ndarray) -> np.ndarray:
    """Legacy thesis mapping: 10*log10(V^2 / 0.001)."""
    v = np.asarray(v, dtype=float)
    return 10.0 * np.log10(np.maximum(np.square(v), 1e-30) / 0.001)


def moving_average(x: np.ndarray, window: int = 100) -> np.ndarray:
    if window <= 1:
        return np.asarray(x, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(np.asarray(x, dtype=float), kernel, mode="same")


def baseband_psd_db(
    signal: np.ndarray,
    sample_rate_hz: float,
    smooth_window: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (freq_mhz_shifted, psd_db) without cosmetic offsets."""
    sig = np.asarray(signal)
    spectrum = np.fft.fft(sig)
    psd = np.abs(spectrum) ** 2 / max(len(sig), 1)
    freq_hz = np.fft.fftfreq(len(sig), d=1.0 / sample_rate_hz)
    freq_mhz = np.fft.fftshift(freq_hz) / 1e6
    psd_db = moving_average(10.0 * np.log10(np.maximum(np.fft.fftshift(psd), 1e-30)), smooth_window)
    return freq_mhz, psd_db
