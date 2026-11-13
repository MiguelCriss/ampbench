"""Deterministic dataset splitting."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ampbench.data.catalog import IQ_QAM256


def split_array(
    data: np.ndarray,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int | None = 42,
    shuffle: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split rows into train/valid/test.

    If ``shuffle`` is False, uses sequential slicing (thesis legacy behavior).
    If ``seed`` is set with shuffle=True, splitting is reproducible.
    """
    if len(ratios) != 3:
        raise ValueError("ratios must be (train, valid, test)")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1, got {sum(ratios)}")

    data = np.asarray(data)
    n = len(data)
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
        data = data[idx]

    n_train = int(ratios[0] * n)
    n_valid = int(ratios[1] * n)
    train = data[:n_train]
    valid = data[n_train : n_train + n_valid]
    test = data[n_train + n_valid :]
    return train, valid, test


def write_splits(
    dataset_path: Path,
    out_dir: Path | None = None,
    ratios: tuple[float, float, float] = IQ_QAM256.split_ratios,
    seed: int = IQ_QAM256.split_seed,
    shuffle: bool = True,
    delimiter: str = "\t",
    fmt: str = "%.9f",
) -> tuple[Path, Path, Path]:
    """Read a dataset file and write train/valid/test beside it (or in out_dir)."""
    out_dir = out_dir or dataset_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Accept tab or whitespace
    try:
        data = np.loadtxt(dataset_path, delimiter="\t")
    except ValueError:
        data = np.loadtxt(dataset_path)

    train, valid, test = split_array(data, ratios=ratios, seed=seed, shuffle=shuffle)
    paths = (
        out_dir / "train.txt",
        out_dir / "valid.txt",
        out_dir / "test.txt",
    )
    np.savetxt(paths[0], train, delimiter=delimiter, fmt=fmt)
    np.savetxt(paths[1], valid, delimiter=delimiter, fmt=fmt)
    np.savetxt(paths[2], test, delimiter=delimiter, fmt=fmt)
    return paths
