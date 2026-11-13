"""Load digitized static PA curves (Vin / Vout / phase)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ampbench.data.catalog import AmplifierSpec, get_amplifier

CANONICAL_CURVE_FILE = "vin_vout_fase.txt"


@dataclass(frozen=True)
class PACurves:
    """Static PA characteristic curves."""

    amp_id: str
    vin: np.ndarray
    vout: np.ndarray
    phase_rad: np.ndarray
    source_path: Path

    def __post_init__(self) -> None:
        n = len(self.vin)
        if not (len(self.vout) == n and len(self.phase_rad) == n):
            raise ValueError("vin/vout/phase length mismatch")

    @property
    def n_points(self) -> int:
        return int(len(self.vin))

    def as_array(self) -> np.ndarray:
        """Return shape (N, 3): vin, vout, phase_rad."""
        return np.column_stack((self.vin, self.vout, self.phase_rad))

    def pin_dbm(self, r_ohm: float = 50.0) -> np.ndarray:
        """Approximate Pin [dBm] from Vin peak assuming P = V^2 / R (thesis convention used R-free V^2/0.001 in places)."""
        # Keep the same convention as legacy plotting: 10*log10(V^2 / 0.001)
        return 10.0 * np.log10(np.square(self.vin) / 0.001)

    def pout_dbm(self) -> np.ndarray:
        return 10.0 * np.log10(np.square(self.vout) / 0.001)

    def gain_db(self) -> np.ndarray:
        return self.pout_dbm() - self.pin_dbm()


def _load_three_column_txt(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"Expected >=3 columns in {path}, got shape {data.shape}")
    return data[:, 0], data[:, 1], data[:, 2]


def _load_two_column_txt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected >=2 columns in {path}, got shape {data.shape}")
    return data[:, 0], data[:, 1]


def load_pa_curves(amp_id: str) -> PACurves:
    """Load canonical Vin/Vout/phase curves for an amplifier id."""
    amp = get_amplifier(amp_id)
    return load_pa_curves_from_spec(amp)


def load_pa_curves_from_spec(amp: AmplifierSpec) -> PACurves:
    canonical = amp.raw_dir / CANONICAL_CURVE_FILE
    if canonical.is_file():
        vin, vout, phase = _load_three_column_txt(canonical)
        return PACurves(amp.id, vin, vout, phase, canonical)

    # Fallback: combine separate files if present
    vout_path = amp.raw_dir / "vout_vs_vin.txt"
    fase_path = amp.raw_dir / "fase_vs_vin.txt"
    if vout_path.is_file() and fase_path.is_file():
        vin_a, vout = _load_two_column_txt(vout_path)
        vin_b, phase = _load_two_column_txt(fase_path)
        if len(vin_a) != len(vin_b) or not np.allclose(vin_a, vin_b, rtol=0, atol=1e-9):
            raise ValueError(f"Vin mismatch between {vout_path.name} and {fase_path.name}")
        return PACurves(amp.id, vin_a, vout, phase, vout_path)

    raise FileNotFoundError(
        f"No curve files for {amp.id} in {amp.raw_dir}. "
        f"Expected {CANONICAL_CURVE_FILE} or vout_vs_vin.txt + fase_vs_vin.txt"
    )


def ensure_canonical_curve_file(amp_id: str, overwrite: bool = False) -> Path:
    """Write vin_vout_fase.txt if missing (from split files)."""
    amp = get_amplifier(amp_id)
    out = amp.raw_dir / CANONICAL_CURVE_FILE
    if out.is_file() and not overwrite:
        return out
    curves = load_pa_curves(amp_id)
    np.savetxt(
        out,
        curves.as_array(),
        delimiter="\t",
        fmt="%.9g",
        header="vin\tvout\tphase_rad",
        comments="",
    )
    return out
