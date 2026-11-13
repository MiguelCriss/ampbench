"""Load processed complex baseband I/Q datasets."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ampbench.data.catalog import IQ_QAM256, IQDatasetSpec


@dataclass(frozen=True)
class IQDataset:
    """Complex baseband input/output capture."""

    time: np.ndarray
    i_in: np.ndarray
    q_in: np.ndarray
    i_out: np.ndarray
    q_out: np.ndarray
    source_path: Path

    @property
    def n_samples(self) -> int:
        return int(len(self.time))

    @property
    def x(self) -> np.ndarray:
        """Complex input."""
        return self.i_in + 1j * self.q_in

    @property
    def y(self) -> np.ndarray:
        """Complex output."""
        return self.i_out + 1j * self.q_out

    def as_array(self) -> np.ndarray:
        """Return shape (N, 5): time, I_in, Q_in, I_out, Q_out."""
        return np.column_stack((self.time, self.i_in, self.q_in, self.i_out, self.q_out))


def _detect_delimiter(path: Path) -> str | None:
    sample = path.read_text(encoding="utf-8", errors="replace").splitlines()[:5]
    if any("\t" in line for line in sample):
        return "\t"
    return None  # numpy default whitespace


def load_iq_txt(path: Path) -> IQDataset:
    delim = _detect_delimiter(path)
    data = np.loadtxt(path, delimiter=delim)
    if data.ndim != 2 or data.shape[1] < 5:
        raise ValueError(f"Expected 5 columns in {path}, got shape {data.shape}")
    return IQDataset(
        time=data[:, 0],
        i_in=data[:, 1],
        q_in=data[:, 2],
        i_out=data[:, 3],
        q_out=data[:, 4],
        source_path=path,
    )


def load_iq_dataset(spec: IQDatasetSpec | None = None) -> IQDataset:
    spec = spec or IQ_QAM256
    if not spec.path.is_file():
        raise FileNotFoundError(spec.path)
    return load_iq_txt(spec.path)


def load_iq_splits(
    directory: Path | None = None,
) -> tuple[IQDataset, IQDataset, IQDataset]:
    """Load committed train/valid/test next to the main dataset."""
    directory = directory or IQ_QAM256.path.parent
    train = load_iq_txt(directory / "train.txt")
    valid = load_iq_txt(directory / "valid.txt")
    test = load_iq_txt(directory / "test.txt")
    return train, valid, test
