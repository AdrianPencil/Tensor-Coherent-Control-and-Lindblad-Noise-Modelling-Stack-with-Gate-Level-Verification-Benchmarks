"""
viz.styles

Central place for plotting styles, kept minimal and optional.

We avoid hard-coding colors and instead expose a light rcParams preset that
improves readability without enforcing a strong theme.
"""

from dataclasses import dataclass

import matplotlib as mpl

__all__ = [
    "PlotStyle",
    "DEFAULT_STYLE",
    "apply_style",
]


@dataclass(frozen=True, slots=True)
class PlotStyle:
    """
    A small set of matplotlib rcParams keys.

    Using a dedicated object makes it easy to keep styles consistent in reports.
    """

    rcparams: dict[str, object]


DEFAULT_STYLE = PlotStyle(
    rcparams={
        "figure.dpi": 120,
        "savefig.dpi": 120,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
    }
)


def apply_style(style: PlotStyle = DEFAULT_STYLE) -> None:
    """
    Apply the given PlotStyle globally via matplotlib rcParams.
    """
    mpl.rcParams.update(style.rcparams)
