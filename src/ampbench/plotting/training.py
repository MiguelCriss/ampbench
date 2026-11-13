"""Training-curve plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ampbench.plotting.utils import finalize_figure


def plot_training_history(
    history: dict,
    *,
    loss_key: str = "loss",
    val_key: str = "val_loss",
    title: str = "Training history",
    ylabel: str = "Loss",
    show: bool = False,
    save_path: Path | str | None = None,
) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if loss_key in history:
        ax.plot(history[loss_key], label="train")
    if val_key in history:
        ax.plot(history[val_key], label="validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return finalize_figure(fig, show=show, save_path=save_path)


def plot_nm_error_heatmap(
    err_train,
    err_valid,
    orders: list[int],
    memories: list[int],
    *,
    metric: str = "nmse_db",
    title: str = "N/M grid search",
    show: bool = False,
    save_path: Path | str | None = None,
) -> Figure:
    """2D heatmaps instead of blocking 3D surfaces."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, data, name in (
        (axes[0], err_train, "train"),
        (axes[1], err_valid, "validation"),
    ):
        im = ax.imshow(data, origin="lower", aspect="auto")
        ax.set_xticks(range(len(memories)), memories)
        ax.set_yticks(range(len(orders)), orders)
        ax.set_xlabel("M (memory)")
        ax.set_ylabel("N (order)")
        ax.set_title(f"{metric} — {name}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    return finalize_figure(fig, show=show, save_path=save_path)
