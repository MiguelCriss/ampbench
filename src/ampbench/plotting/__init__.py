"""Honest, opt-in plotting (no cosmetic hacks, no forced plt.show)."""

from ampbench.plotting.constellation import plot_constellation
from ampbench.plotting.pa_curves import plot_amam_ampm, plot_power_characteristics
from ampbench.plotting.psd import plot_psd_comparison
from ampbench.plotting.timeseries import plot_vout_timeseries
from ampbench.plotting.training import plot_nm_error_heatmap, plot_training_history
from ampbench.plotting.utils import finalize_figure

__all__ = [
    "finalize_figure",
    "plot_amam_ampm",
    "plot_constellation",
    "plot_nm_error_heatmap",
    "plot_power_characteristics",
    "plot_psd_comparison",
    "plot_training_history",
    "plot_vout_timeseries",
]
