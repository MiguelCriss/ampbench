# GitHub publish checklist

Repo name suggestion: **`ampbench`**  
Expected URL: `https://github.com/MiguelCriss/ampbench`

## Pre-flight (done in Phase 6 tooling)

- [x] Public tree has no `_local_only/` datasheets/Excel (gitignored)
- [x] `artifacts/` runtime outputs gitignored
- [x] README + figures + citation + reproducibility docs present
- [x] Smoke checks pass (`smoke_check_data.py`, `smoke_check_models.py`)
- [x] License MIT on code; datasheet copyright called out

## Commands to publish

From `ampbench/`:

```bash
# 1) Final local checks
python scripts/smoke_check_data.py
python scripts/smoke_check_models.py

# 2) Init (if needed) and first commit
git init -b main
git add .
git status   # confirm: no _local_only, no artifacts/*.png dumps, no .h5/.pkl

git commit -m "$(cat <<'EOF'
Initial public release of ampbench.

Behavioral PA modeling toolkit with data, classical/ANN models, and honest eval plots.
EOF
)"

# 3) Create GitHub repo + push (GitHub CLI)
gh repo create MiguelCriss/ampbench --public --source=. --remote=origin --push

# Or manual:
# git remote add origin https://github.com/MiguelCriss/ampbench.git
# git push -u origin main
```

## Alignerr link

After push, paste this into the Alignerr GitHub field:

```text
https://github.com/MiguelCriss/ampbench
```

## Post-publish sanity

- [ ] README renders images on GitHub
- [ ] Clone in a clean folder and run smoke checks
- [ ] Confirm datasheet PDFs are **not** visible in the GitHub file browser
