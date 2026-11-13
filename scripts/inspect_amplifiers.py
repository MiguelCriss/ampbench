"""Print a short inventory of all amplifiers and the I/Q dataset."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ampbench.data import IQ_QAM256, list_amplifiers, load_iq_dataset, load_pa_curves


def main() -> None:
    print("== Amplifiers (raw curves) ==")
    for amp in list_amplifiers():
        curves = load_pa_curves(amp.id)
        print(
            f"- {amp.id:16} {amp.name:22} points={curves.n_points:4d} "
            f"vin=[{curves.vin.min():.4g}, {curves.vin.max():.4g}] "
            f"file={curves.source_path.name}"
        )
        if amp.notes:
            print(f"  note: {amp.notes}")

    print("\n== Processed I/Q ==")
    iq = load_iq_dataset(IQ_QAM256)
    print(f"- {IQ_QAM256.id}: shape=({iq.n_samples}, 5) path={IQ_QAM256.path}")
    print(f"  {IQ_QAM256.description}")


if __name__ == "__main__":
    main()
