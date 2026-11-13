# Data layout

```text
data/
  raw/<amp_id>/
    vin_vout_fase.txt     # canonical: Vin, Vout, phase [rad]
    vout_vs_vin.txt       # optional convenience extract
    fase_vs_vin.txt       # optional convenience extract
  processed/iq_qam256/
    dataset.txt           # time, I_in, Q_in, I_out, Q_out
    train.txt             # committed thesis split (70%)
    valid.txt             # 15%
    test.txt              # 15%
  catalog.yaml
```

## Canonical curve format

Tab-separated, no header required (loaders accept headerless numeric tables):

```text
vin   vout   phase_rad
```

## Commands

```bash
python scripts/inspect_amplifiers.py
python scripts/smoke_check_data.py
python scripts/make_splits.py          # dry-run
python scripts/export_curve_preview.py # writes PNGs under artifacts/
```
