"""Amplifier and dataset catalog."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ampbench.paths import data_dir, raw_dir


@dataclass(frozen=True)
class AmplifierSpec:
    id: str
    name: str
    manufacturer: str | None
    raw_subdir: str
    notes: str | None = None

    @property
    def raw_dir(self) -> Path:
        return raw_dir() / self.raw_subdir


@dataclass(frozen=True)
class IQDatasetSpec:
    id: str
    rel_path: str
    columns: tuple[str, ...]
    split_ratios: tuple[float, float, float]
    split_seed: int
    description: str

    @property
    def path(self) -> Path:
        return data_dir() / self.rel_path


# Source of truth (code). Mirrored in data/catalog.yaml for humans.
AMPLIFIERS: tuple[AmplifierSpec, ...] = (
    AmplifierSpec(
        id="tga2595",
        name="TGA2595",
        manufacturer="Qorvo",
        raw_subdir="tga2595",
        notes="Primary 5G PA used in the degree project; digitized Gain / AM-PM / Vin-Vout.",
    ),
    AmplifierSpec(
        id="k_hpa10w_1721",
        name="K-HPA10W-1721-SSPA",
        manufacturer="ReliaSat",
        raw_subdir="k_hpa10w_1721",
        notes="Workbook note: ~19.5 GHz without DPD.",
    ),
    AmplifierSpec(
        id="q_hpa2w_3743",
        name="Q-HPA2W-3743",
        manufacturer=None,
        raw_subdir="q_hpa2w_3743",
        notes="Workbook note: ~37 GHz.",
    ),
    AmplifierSpec(
        id="hpa11201721qf",
        name="HPA11201721QF",
        manufacturer=None,
        raw_subdir="hpa11201721qf",
        notes="Workbook note: ~20 GHz.",
    ),
)

IQ_QAM256 = IQDatasetSpec(
    id="iq_qam256",
    rel_path="processed/iq_qam256/dataset.txt",
    columns=("time", "I_in", "Q_in", "I_out", "Q_out"),
    split_ratios=(0.70, 0.15, 0.15),
    split_seed=42,
    description=(
        "Complex baseband I/Q capture used for ANN / MPM / Hammerstein / Wiener "
        "training in the thesis (QAM-256 drive via Simulink)."
    ),
)


def get_amplifier(amp_id: str) -> AmplifierSpec:
    for amp in AMPLIFIERS:
        if amp.id == amp_id:
            return amp
    known = ", ".join(a.id for a in AMPLIFIERS)
    raise KeyError(f"Unknown amplifier id {amp_id!r}. Known: {known}")


def list_amplifiers() -> tuple[AmplifierSpec, ...]:
    return AMPLIFIERS
