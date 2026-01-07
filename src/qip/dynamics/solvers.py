"""
dynamics.solvers

A tiny ODE solver layer (no SciPy dependency).

We expose one stepping method:
- RK4 (Runge-Kutta 4) for vectors and matrices

Higher layers can build convenience wrappers (propagators/workflows).
"""

from typing import Callable

import numpy as np
import numpy.typing as npt

from qip.core.linalg import as_c128

__all__ = [
    "rk4_step_vector",
    "rk4_step_matrix",
    "integrate_rk4_matrix",
]


def rk4_step_vector(
    f: Callable[[float, npt.NDArray[np.complex128]], npt.NDArray[np.complex128]],
    t: float,
    y: npt.ArrayLike,
    dt: float,
) -> npt.NDArray[np.complex128]:
    """
    One RK4 step for a complex128 vector y.
    """
    y0 = as_c128(y).reshape(-1)
    h = np.float64(dt)

    k1 = f(t, y0)
    k2 = f(t + 0.5 * float(h), y0 + 0.5 * h * k1)
    k3 = f(t + 0.5 * float(h), y0 + 0.5 * h * k2)
    k4 = f(t + float(h), y0 + h * k3)

    y1 = y0 + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return np.ascontiguousarray(y1)


def rk4_step_matrix(
    f: Callable[[float, npt.NDArray[np.complex128]], npt.NDArray[np.complex128]],
    t: float,
    y: npt.NDArray[np.complex128],
    dt: float,
) -> npt.NDArray[np.complex128]:
    """
    One RK4 step for a complex128 matrix y.
    """
    y0 = as_c128(y)
    h = np.float64(dt)

    k1 = f(t, y0)
    k2 = f(t + 0.5 * float(h), y0 + 0.5 * h * k1)
    k3 = f(t + 0.5 * float(h), y0 + 0.5 * h * k2)
    k4 = f(t + float(h), y0 + h * k3)

    y1 = y0 + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return np.ascontiguousarray(y1)


def integrate_rk4_matrix(
    f: Callable[[float, npt.NDArray[np.complex128]], npt.NDArray[np.complex128]],
    y0: npt.ArrayLike,
    t_grid: npt.ArrayLike,
) -> npt.NDArray[np.complex128]:
    """
    Integrate matrix ODE over a monotone time grid.

    Returns an array of shape (T, d, d) where T=len(t_grid).
    """
    t = np.asarray(t_grid, dtype=np.float64).reshape(-1)
    if t.size < 2:
        raise ValueError("t_grid must have at least two points")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("t_grid must be strictly increasing")

    y = as_c128(y0)
    if y.ndim != 2 or y.shape[0] != y.shape[1]:
        raise ValueError("y0 must be a square matrix")

    out = np.zeros((t.size, y.shape[0], y.shape[1]), dtype=np.complex128)
    out[0] = y
    for i in range(t.size - 1):
        dt = float(t[i + 1] - t[i])
        y = rk4_step_matrix(f, float(t[i]), y, dt)
        out[i + 1] = y
    return np.ascontiguousarray(out)
