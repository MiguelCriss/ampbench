"""Optional synthesis helpers for static PA curves.

These replace the old `gen_new.py` / `gen_dat_PAcomercial.py` scripts:
interpolation is explicit, noise is seeded, and nothing pops interactive plots.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d

from ampbench.data.curves import PACurves


@dataclass(frozen=True)
class SynthesisConfig:
    n_points: int = 10_000
    noise_scale: float = 0.03
    seed: int = 42
    kind: str = "linear"


def densify_curves(
    curves: PACurves,
    config: SynthesisConfig | None = None,
) -> PACurves:
    """Interpolate Vin/Vout/phase onto a denser Vin grid (optional light noise)."""
    config = config or SynthesisConfig()
    order = np.argsort(curves.vin)
    vin = curves.vin[order]
    vout = curves.vout[order]
    phase = curves.phase_rad[order]

    vin_new = np.linspace(float(vin.min()), float(vin.max()), config.n_points)
    vout_new = interp1d(vin, vout, kind=config.kind, fill_value="extrapolate")(vin_new)
    phase_new = interp1d(vin, phase, kind=config.kind, fill_value="extrapolate")(vin_new)

    if config.noise_scale > 0:
        rng = np.random.default_rng(config.seed)
        vout_std = float(np.std(vout))
        phase_std = float(np.std(phase))
        vout_new = vout_new + rng.normal(scale=config.noise_scale * vout_std, size=config.n_points)
        phase_new = phase_new + rng.normal(scale=config.noise_scale * phase_std, size=config.n_points)

    return PACurves(
        amp_id=curves.amp_id,
        vin=vin_new,
        vout=vout_new,
        phase_rad=phase_new,
        source_path=curves.source_path,
    )


def curves_to_iq_columns(curves: PACurves) -> np.ndarray:
    """Map static (vin, vout, phase) to a 5-column I/Q table.

    Convention (legacy `gen_new.py`):
    - I_in = vin, Q_in = 0
    - I_out = vout * cos(phase), Q_out = vout * sin(phase)
    - time = 0..N-1
    """
    time = np.arange(curves.n_points, dtype=float)
    i_in = curves.vin
    q_in = np.zeros_like(i_in)
    i_out = curves.vout * np.cos(curves.phase_rad)
    q_out = curves.vout * np.sin(curves.phase_rad)
    return np.column_stack((time, i_in, q_in, i_out, q_out))
