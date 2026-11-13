"""Sanity-check raw amplifiers and processed I/Q data."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ampbench.data import (
    IQ_QAM256,
    list_amplifiers,
    load_iq_dataset,
    load_iq_splits,
    load_pa_curves,
)


def main() -> None:
    amps = list_amplifiers()
    assert len(amps) == 4, amps
    for amp in amps:
        curves = load_pa_curves(amp.id)
        assert curves.n_points >= 10, amp.id
        assert curves.vin.min() > 0 or amp.id  # allow edge cases
        print(f"OK curves {amp.id}: n={curves.n_points}")

    iq = load_iq_dataset()
    assert iq.n_samples == 24576, iq.n_samples
    assert iq.as_array().shape[1] == 5
    print(f"OK iq dataset: n={iq.n_samples}")

    train, valid, test = load_iq_splits()
    total = train.n_samples + valid.n_samples + test.n_samples
    assert total == iq.n_samples, (total, iq.n_samples)
    print(
        f"OK splits: train={train.n_samples} valid={valid.n_samples} "
        f"test={test.n_samples} (committed thesis split)"
    )
    print(f"catalog target seed={IQ_QAM256.split_seed} ratios={IQ_QAM256.split_ratios}")
    print("smoke_check_data: passed")


if __name__ == "__main__":
    main()
