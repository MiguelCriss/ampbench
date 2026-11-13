"""Train the ANN behavioral PA model."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ampbench.data import load_iq_dataset, load_iq_splits
from ampbench.metrics import calculate_error
from ampbench.models import build_ann_bundle, predict_iq, save_ann, train_ann
from ampbench.paths import repo_root


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--memory", type=int, default=7)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=320)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--verbose", type=int, default=1)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: artifacts/ann).",
    )
    args = p.parse_args()

    train, valid, test = load_iq_splits()
    full = load_iq_dataset()
    bundle = build_ann_bundle(train, valid, test, full, memory=args.memory)

    result = train_ann(
        bundle,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        verbose=args.verbose,
    )

    i_test, q_test = predict_iq(result.model, bundle.X_test, bundle)
    y_test_hat = i_test + 1j * q_test
    test_err = calculate_error(test.y, y_test_hat, "nmse_db")

    out_dir = args.out or (repo_root() / "artifacts" / "ann")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.keras"
    save_ann(result.model, model_path)
    metrics = {
        "memory": args.memory,
        "epochs": args.epochs,
        "test_nmse_db": test_err,
        "final_train_loss": float(result.history["loss"][-1]),
        "final_val_loss": float(result.history["val_loss"][-1]),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"wrote {model_path}")


if __name__ == "__main__":
    main()
