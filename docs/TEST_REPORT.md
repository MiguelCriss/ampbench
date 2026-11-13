# Pre-commit test report

Date: 2026-11-13  
Host: Windows / Python 3.13

## Results

| # | Check | Result |
|---|--------|--------|
| 1 | `publish_preflight.py` | PASS |
| 2 | `smoke_check_data.py` | PASS (4 PAs + I/Q 24576) |
| 3 | `smoke_check_models.py` | PASS (MPM/Hammerstein/Wiener) |
| 4 | `inspect_amplifiers.py` | PASS |
| 5 | `make_splits.py` dry-run | PASS |
| 6 | Package imports | PASS (`ampbench==0.1.0`) |
| 7 | `train_traditional.py` MPM | PASS (~−22.9 dB test) |
| 8 | `train_traditional.py` Hammerstein | PASS (~−21.5 dB test) |
| 9 | `train_traditional.py` Wiener | PASS (~−21.6 dB test) |
| 10 | `evaluate_and_plot.py` | PASS (5 PNGs + metrics.json) |
| 11 | `densify_curves` / synthesize | PASS |
| 12 | `export_curve_preview.py` | PASS |
| 13 | `pip install -e .` | PASS |
| 14 | ANN without TF (error message) | PASS (clear install hint) |
| 15 | `pip install -e ".[ann]"` + `train_ann.py --epochs 2` | PASS (smoke; not converged) |

## Notes

- `_local_only/` and `artifacts/` remain gitignored.
- ANN 2-epoch smoke is only a pipeline check; use more epochs for real metrics.
- Classical path does not require TensorFlow.
