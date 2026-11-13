"""Fail if the working tree looks unsafe to publish publicly."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_SUFFIXES = {".h5", ".pkl", ".mat", ".pdf"}
FORBIDDEN_NAMES = {"predicciones_mpm.txt", "predicciones_hammerstein.txt", "predicciones_wiener.txt"}


def main() -> int:
    errors: list[str] = []

    local_only = ROOT / "_local_only"
    if local_only.exists():
        # Directory may exist locally; it must be gitignored, not absent.
        gi = (ROOT / ".gitignore").read_text(encoding="utf-8")
        if "_local_only/" not in gi:
            errors.append("_local_only/ exists but is not listed in .gitignore")

    tracked_candidates = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith("_local_only/") or rel.startswith("artifacts/"):
            continue
        if rel.startswith(".git/"):
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden suffix in public tree: {rel}")
        if path.name.lower() in FORBIDDEN_NAMES:
            errors.append(f"forbidden dump in public tree: {rel}")
        if path.stat().st_size > 5 * 1024 * 1024:
            errors.append(f"file >5MB in public tree: {rel} ({path.stat().st_size} bytes)")
        tracked_candidates.append(rel)

    required = [
        "README.md",
        "LICENSE",
        "pyproject.toml",
        "src/ampbench/__init__.py",
        "data/processed/iq_qam256/dataset.txt",
        "docs/REPRODUCIBILITY.md",
        "docs/DATA.md",
        "CITATION.cff",
    ]
    for req in required:
        if not (ROOT / req).exists():
            errors.append(f"missing required file: {req}")

    if errors:
        print("publish_preflight: FAILED")
        for e in errors:
            print(" -", e)
        return 1

    print(f"publish_preflight: OK ({len(tracked_candidates)} public files scanned)")
    print("Reminder: keep _local_only/ and artifacts/ untracked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
