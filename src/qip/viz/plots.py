"""
viz.plots

Minimal plotting helpers used by reports/case studies.

We keep plotting functions thin and return the figure object so callers can:
- save it
- embed it
- further modify it without editing this module
"""

from typing import Iterable

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from qip.core.linalg import as_c128

__all__ = [
    "plot_state_probabilities",
    "plot_psd",
]


def plot_state_probabilities(
    psi: npt.ArrayLike,
    *,
    title: str | None = None,
) -> plt.Figure:
    """
    Plot computational basis probabilities |psi_i|^2 as a bar chart.
    """
    v = as_c128(psi).reshape(-1)
    p = (v.real * v.real) + (v.imag * v.imag)
    p = np.ascontiguousarray(p.astype(np.float64, copy=False))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(np.arange(p.size), p)
    ax.set_xlabel("basis index")
    ax.set_ylabel("probability")
    if title is not None:
        ax.set_title(title)
    return fig


def plot_psd(
    omega: npt.ArrayLike,
    S: npt.ArrayLike,
    *,
    title: str | None = None,
    loglog: bool = True,
) -> plt.Figure:
    """
    Plot a PSD curve S(omega).

    If loglog=True, uses log-log axes which is common for noise spectra.
    """
    w = np.ascontiguousarray(np.asarray(omega, dtype=np.float64).reshape(-1))
    y = np.ascontiguousarray(np.asarray(S, dtype=np.float64).reshape(-1))
    if w.size != y.size:
        raise ValueError("omega and S must have the same length")

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    if loglog:
        ax.loglog(w, y)
    else:
        ax.plot(w, y)
    ax.set_xlabel("omega (rad/s)")
    ax.set_ylabel("S(omega)")
    if title is not None:
        ax.set_title(title)
    return fig
