# Contributing

This repository is primarily a research/portfolio toolkit. Small fixes and clarity improvements are welcome.

## Local checks

```bash
pip install -e ".[dev,ann]"
python scripts/smoke_check_data.py
python scripts/smoke_check_models.py
```

## Guidelines

- Keep evaluation plots honest (no cosmetic transforms).
- Do not commit manufacturer datasheet PDFs or `_local_only/` content.
- Prefer extending `src/ampbench/` over editing `legacy/`.
- Update `docs/REPRODUCIBILITY.md` if you change default train/eval commands.
