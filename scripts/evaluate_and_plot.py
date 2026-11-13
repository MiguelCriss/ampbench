"""Fit a classical model and export honest evaluation figures (no interactive popups)."""
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
from ampbench.metrics.signals import complex_envelopes
from ampbench.models import fit_traditional, predict_traditional
from ampbench.paths import repo_root
from ampbench.plotting import (
    plot_amam_ampm,
    plot_constellation,
    plot_power_characteristics,
    plot_psd_comparison,
    plot_vout_timeseries,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["MPM", "Hammerstein", "Wiener"], default="MPM")
    p.add_argument("--order", type=int, default=1)
    p.add_argument("--memory", type=int, default=7)
    p.add_argument("--max-train", type=int, default=8000)
    p.add_argument("--sample-rate", type=float, default=100e6)
    p.add_argument("--show", action="store_true", help="Also open interactive windows")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: artifacts/eval/<model>).",
    )
    args = p.parse_args()

    train, valid, test = load_iq_splits()
    full = load_iq_dataset()
    n = args.max_train if args.max_train > 0 else len(train.x)

    result = fit_traditional(
        args.model,
        train.x[:n],
        train.y[:n],
        order=args.order,
        memory=args.memory,
        x_valid=valid.x,
        y_valid=valid.y,
        x_test=test.x,
        y_test=test.y,
        metric="nmse_db",
    )
    y_hat = predict_traditional(args.model, full.x, result.coefficients, args.order, args.memory)
    env = complex_envelopes(full.x, full.y, y_hat)

    out = args.out or (repo_root() / "artifacts" / "eval" / args.model.lower())
    out.mkdir(parents=True, exist_ok=True)

    # Circular phase residual (avoids wrap-around artifacts in raw NMSE on angle).
    phase_resid = np.angle(np.exp(1j * (env.phase_true_rad - env.phase_pred_rad)))
    metrics = {
        "model": args.model,
        "order": args.order,
        "memory": args.memory,
        "train_nmse_db": result.train_error,
        "valid_nmse_db": result.valid_error,
        "test_nmse_db": result.test_error,
        "full_nmse_db": calculate_error(full.y, y_hat, "nmse_db"),
        "full_vout_nmse_db": calculate_error(env.vout_true, env.vout_pred, "nmse_db"),
        "full_phase_mae_deg": float(np.mean(np.abs(phase_resid)) * 180.0 / np.pi),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    label = f"{args.model} (N={args.order}, M={args.memory})"
    common = {"show": args.show, "model_label": label}
    plot_amam_ampm(env, title=f"{label} — AM/AM & AM/PM", save_path=out / "amam_ampm.png", **common)
    plot_power_characteristics(env, title=f"{label} — power curves", save_path=out / "power_curves.png", **common)
    plot_constellation(full.y, y_hat, title=f"{label} — constellation", save_path=out / "constellation.png", **common)
    plot_psd_comparison(
        full.y,
        y_hat,
        sample_rate_hz=args.sample_rate,
        title=f"{label} — PSD",
        save_path=out / "psd.png",
        **common,
    )
    # Short time window for readability
    t_max = float(full.time[min(2000, len(full.time) - 1)])
    plot_vout_timeseries(
        full.time,
        env.vout_true,
        {label: env.vout_pred},
        title=f"{label} — Vout time series",
        t_max=t_max,
        save_path=out / "timeseries.png",
        show=args.show,
    )

    # Copy key figures into docs for the public README (optional publish set)
    docs_fig = repo_root() / "docs" / "figures" / "eval"
    docs_fig.mkdir(parents=True, exist_ok=True)
    for name in ("amam_ampm.png", "power_curves.png", "constellation.png", "psd.png"):
        src = out / name
        if src.exists():
            (docs_fig / f"{args.model.lower()}_{name}").write_bytes(src.read_bytes())

    print(json.dumps(metrics, indent=2))
    print(f"figures -> {out}")


if __name__ == "__main__":
    main()
