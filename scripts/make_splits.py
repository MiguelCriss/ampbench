"""Create train/valid/test splits from a processed dataset file."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ampbench.data import IQ_QAM256, write_splits


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=IQ_QAM256.path)
    p.add_argument("--seed", type=int, default=IQ_QAM256.split_seed)
    p.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Sequential split (legacy thesis behavior).",
    )
    p.add_argument(
        "--write",
        action="store_true",
        help="Actually overwrite train/valid/test. Without this flag, dry-run only.",
    )
    args = p.parse_args()

    if not args.write:
        print(
            "Dry-run only (refusing to overwrite committed splits). "
            "Re-run with --write if you intend to regenerate."
        )
        print(f"dataset={args.dataset} seed={args.seed} shuffle={not args.no_shuffle}")
        return

    paths = write_splits(
        args.dataset,
        seed=args.seed,
        shuffle=not args.no_shuffle,
        ratios=IQ_QAM256.split_ratios,
    )
    print("Wrote:", ", ".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
