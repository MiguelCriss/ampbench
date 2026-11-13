# Reproducibility

## Environment

- Python **3.10+**
- OS: Windows / Linux / macOS
- Optional GPU: only for ANN (TensorFlow); classical models are CPU-only

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -U pip
pip install -e ".[ann]"         # or: pip install -e .  (classical only)
```

Lock-free policy: minimum versions are pinned in `pyproject.toml` / `requirements*.txt`. For a frozen environment, export after install:

```bash
pip freeze > artifacts/pip-freeze.txt
```

## Canonical data

| Artifact | Path | Notes |
|----------|------|-------|
| Raw PA curves | `data/raw/<amp_id>/vin_vout_fase.txt` | Digitized Vin/Vout/phase |
| Full I/Q set | `data/processed/iq_qam256/dataset.txt` | 24576 × 5 |
| Splits | `.../train.txt`, `valid.txt`, `test.txt` | **Thesis sequential 70/15/15** (committed) |

Do **not** overwrite committed splits unless intentional:

```bash
python scripts/make_splits.py              # dry-run
python scripts/make_splits.py --write      # shuffle + seed=42 (changes files)
```

## Determinism

| Step | Deterministic? | Controls |
|------|----------------|----------|
| Classical fit (lstsq) | Yes (same NumPy/OpenBLAS build) | `--order`, `--memory`, train slice |
| Split regeneration | Yes if seeded | `--seed` (default 42) |
| ANN training | Partially | TF/CUDA nondeterminism; set seeds best-effort, expect small float drift |
| Plots | Yes given same predictions | `evaluate_and_plot.py` |

Recommended classical command for README figures:

```bash
python scripts/evaluate_and_plot.py --model MPM --order 1 --memory 3 --max-train 4000
```

Full-data classical train:

```bash
python scripts/train_traditional.py --model MPM --order 1 --memory 7
```

ANN (slower):

```bash
python scripts/train_ann.py --memory 7 --epochs 300 --batch-size 320 --learning-rate 1e-4
```

Outputs land in `artifacts/` (gitignored). Example PNGs for docs are copied under `docs/figures/`.

## Verification checklist

```bash
python scripts/smoke_check_data.py
python scripts/smoke_check_models.py
python scripts/inspect_amplifiers.py
```

Pass criteria:

1. Four amplifiers load with `n_points >= 10`
2. I/Q dataset shape `(24576, 5)`
3. Committed splits sum to 24576
4. MPM / Hammerstein / Wiener return finite `nmse_db` on a 2000-sample smoke slice

## What is *not* bit-identical to the thesis GUI scripts

- MPM design-matrix indexing uses `m*(N+1)+k` (correct); legacy used `m*N+k`
- Phase reporting uses circular MAE; plots use `angle(y*conj(x))`
- No forced interactive windows or cosmetic PSD/constellation transforms

Legacy scripts remain under `legacy/` for archaeology only.
