"""Fast smoke test for classical models (no TensorFlow required)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ampbench.data import load_iq_splits
from ampbench.models import fit_traditional


def main() -> None:
    train, valid, test = load_iq_splits()
    # Small slice keeps CI/local smoke fast
    n = 2000
    x_tr, y_tr = train.x[:n], train.y[:n]
    x_va, y_va = valid.x[:n], valid.y[:n]
    x_te, y_te = test.x[:n], test.y[:n]

    for model in ("MPM", "Hammerstein", "Wiener"):
        res = fit_traditional(
            model,
            x_tr,
            y_tr,
            order=1,
            memory=3,
            x_valid=x_va,
            y_valid=y_va,
            x_test=x_te,
            y_test=y_te,
            metric="nmse_db",
        )
        assert res.train_error is not None and np_isfinite(res.train_error), model
        print(
            f"OK {model}: train={res.train_error:.3f} dB  "
            f"valid={res.valid_error:.3f} dB  test={res.test_error:.3f} dB"
        )
    print("smoke_check_models: passed")


def np_isfinite(x: float) -> bool:
    import math

    return math.isfinite(x)


if __name__ == "__main__":
    main()
