"""Figure I/O helpers — never force interactive display."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def finalize_figure(
    fig: Figure,
    *,
    show: bool = False,
    save_path: Path | str | None = None,
    dpi: int = 120,
    close: bool = True,
) -> Figure:
    """Optionally save and/or show a figure.

    Defaults are non-interactive: no show, optional save, close after save
    unless the caller keeps the figure (close=False).
    """
    fig.tight_layout()
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    elif close and save_path is not None:
        plt.close(fig)
    return fig
