"""Train a classical PA behavioral model (MPM / Hammerstein / Wiener)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ampbench.data import load_iq_dataset, load_iq_splits
from ampbench.metrics import calculate_error
from ampbench.models import fit_traditional, predict_traditional
from ampbench.paths import repo_root


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["MPM", "Hammerstein", "Wiener"], default="MPM")
    p.add_argument("--order", type=int, default=1, help="Nonlinearity order N")
    p.add_argument("--memory", type=int, default=7, help="Memory depth M")
    p.add_argument("--metric", default="nmse_db")
    p.add_argument(
        "--max-train",
        type=int,
        default=0,
        help="Optional subsample of training rows (0 = all). Useful for quick smoke tests.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Directory for coefficients + metrics (default: artifacts/traditional/<model>).",
    )
    args = p.parse_args()

    train, valid, test = load_iq_splits()
    full = load_iq_dataset()

    x_tr, y_tr = train.x, train.y
    if args.max_train and args.max_train < len(x_tr):
        x_tr, y_tr = x_tr[: args.max_train], y_tr[: args.max_train]

    result = fit_traditional(
        args.model,
        x_tr,
        y_tr,
        order=args.order,
        memory=args.memory,
        x_valid=valid.x,
        y_valid=valid.y,
        x_test=test.x,
        y_test=test.y,
        metric=args.metric,
    )

    y_full = predict_traditional(
        args.model, full.x, result.coefficients, args.order, args.memory
    )
    full_err = calculate_error(full.y, y_full, args.metric)

    out_dir = args.out or (repo_root() / "artifacts" / "traditional" / args.model.lower())
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "coefficients.npz",
        **{k: np.asarray(v) for k, v in result.coefficients.items()},
        order=args.order,
        memory=args.memory,
    )
    metrics = {
        "model": args.model,
        "order": args.order,
        "memory": args.memory,
        "metric": args.metric,
        "train": result.train_error,
        "valid": result.valid_error,
        "test": result.test_error,
        "full": full_err,
        "n_train_used": int(len(x_tr)),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
