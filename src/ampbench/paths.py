"""Repository path helpers."""
from __future__ import annotations

from pathlib import Path

_REPO_ROOT: Path | None = None


def repo_root() -> Path:
    """Return the ampbench repository root (folder that contains `data/` and `src/`)."""
    global _REPO_ROOT
    if _REPO_ROOT is not None:
        return _REPO_ROOT

    here = Path(__file__).resolve()
    for candidate in (here.parents[2], here.parents[1], Path.cwd()):
        if (candidate / "data").is_dir() and (candidate / "src").is_dir():
            _REPO_ROOT = candidate
            return candidate
    raise FileNotFoundError("Could not locate ampbench repo root from " + str(here))


def data_dir() -> Path:
    return repo_root() / "data"


def raw_dir() -> Path:
    return data_dir() / "raw"


def processed_dir() -> Path:
    return data_dir() / "processed"
