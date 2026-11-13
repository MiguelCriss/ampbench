# Data catalog

## Pipeline

```text
Manufacturer datasheet curves (digitized)
        │
        ▼
data/raw/<amp_id>/vin_vout_fase.txt      ← 4 amplifiers
        │
        ├── (optional) densify / map to I/Q   [ampbench.data.synthesize]
        │
        └── Thesis path: Simulink QAM-256 through PA behavioral path
                │
                ▼
        data/processed/iq_qam256/dataset.txt
                │
                ▼
        train.txt / valid.txt / test.txt  → models (ANN, MPM, …)
```

## Power amplifiers

| ID | Device | Manufacturer | Raw folder |
|----|--------|--------------|------------|
| `tga2595` | TGA2595 | Qorvo | `data/raw/tga2595/` |
| `k_hpa10w_1721` | K-HPA10W-1721-SSPA | ReliaSat | `data/raw/k_hpa10w_1721/` |
| `q_hpa2w_3743` | Q-HPA2W-3743 | — | `data/raw/q_hpa2w_3743/` |
| `hpa11201721qf` | HPA11201721QF | — | `data/raw/hpa11201721qf/` |

Canonical file per amp: **`vin_vout_fase.txt`** (Vin, Vout, phase in radians).

Datasheet PDFs are **not** redistributed. Local copies live under `_local_only/datasheets/`. Cite the manufacturer product page in publications.

## Processed I/Q dataset

- `data/processed/iq_qam256/dataset.txt` — columns: `time, I_in, Q_in, I_out, Q_out`
- **Committed splits** — original thesis split (70/15/15, **sequential**, bit-stable)
- **Regeneration** — `python scripts/make_splits.py --write` uses shuffle + `seed=42` (overwrites committed files; dry-run by default)

## Legacy scripts

`legacy/dataset_scripts/` (`gen_new.py`, `gen_dat_PAcomercial.py`, `gen_dat_MATLAB.py`) are **deprecated**. Replacements:

| Old | New |
|-----|-----|
| `gen_dat_MATLAB.py` | `ampbench.data.splits` / `scripts/make_splits.py` |
| `gen_new.py` / `gen_dat_PAcomercial.py` | `ampbench.data.synthesize` + `ampbench.data.curves` |

## Senior policy (PDFs / Excel)

1. Publish **your** numerical extracts (`.txt`).
2. Keep manufacturer PDFs and bulky Excel/Simulink exports in `_local_only/` (gitignored).
3. Document device identifiers so others can fetch datasheets themselves.
