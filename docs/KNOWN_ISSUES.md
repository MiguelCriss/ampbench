# Known issues / design notes

## Fixed in Phase 3–4 (public API)

- Training/grid-search do not call plots.
- MPM column indexing uses `m * (N + 1) + k`.
- Classical + ANN entrypoints: `scripts/train_*.py`.
- Plotting is opt-in (`show=False` by default) via `ampbench.plotting`.
- Constellation plots use raw I/Q only (no random jitter / affine cosmetics).
- PSD residual is `PSD(y_true - y_pred)` with no `0.9*x - 13` offsets.
- Plot functions do not write CSV/pickle as hidden side effects.
- Grid search visualization uses heatmaps (`plot_nm_error_heatmap`), not blocking 3D windows.

## Legacy only

`legacy/traditional/funciones.py` and `legacy/plots/` still contain the old forced/cosmetic behavior for historical reference. Do not use them in new work.

## Metric note

Prefer complex-baseband `nmse_db` for model quality. For phase, report circular MAE in degrees (`angle(exp(j*(φ_true-φ_pred)))`), not raw NMSE on wrapped angles.
