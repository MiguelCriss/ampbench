"""Export a non-interactive PNG preview of raw PA curves (optional)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt

from ampbench.data import get_amplifier, list_amplifiers, load_pa_curves
from ampbench.paths import repo_root


def plot_one(amp_id: str, out_path: Path) -> None:
    curves = load_pa_curves(amp_id)
    amp = get_amplifier(amp_id)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(curves.vin, curves.vout, ".", markersize=4)
    axes[0].set_xlabel("Vin")
    axes[0].set_ylabel("Vout")
    axes[0].set_title("AM/AM (Vin–Vout)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(curves.vin, curves.phase_rad, ".", markersize=4)
    axes[1].set_xlabel("Vin")
    axes[1].set_ylabel("Phase [rad]")
    axes[1].set_title("AM/PM (Vin–phase)")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"{amp.name} ({amp.id})")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--amp", default="all", help="Amplifier id or 'all'")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root() / "artifacts" / "curve_previews",
    )
    args = p.parse_args()

    ids = [a.id for a in list_amplifiers()] if args.amp == "all" else [args.amp]
    for amp_id in ids:
        plot_one(amp_id, args.out_dir / f"{amp_id}.png")


if __name__ == "__main__":
    main()
