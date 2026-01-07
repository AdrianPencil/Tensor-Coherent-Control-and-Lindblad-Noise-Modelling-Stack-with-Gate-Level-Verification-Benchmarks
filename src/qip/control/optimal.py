"""
control.optimal

A minimal, non-bloated "good enough" optimizer for small problems.

We provide one routine:
- random_search: for low-dimensional parameter vectors

This is used in early prototyping and case studies before introducing
heavier optimal-control machinery.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt

__all__ = [
    "RandomSearchResult",
    "random_search",
]


@dataclass(frozen=True, slots=True)
class RandomSearchResult:
    x_best: npt.NDArray[np.float64]
    f_best: np.float64
    n_evals: int


def random_search(
    objective: Callable[[npt.NDArray[np.float64]], float],
    bounds: npt.ArrayLike,
    n_samples: int,
    *,
    seed: int | None = None,
) -> RandomSearchResult:
    """
    Random search over a box defined by bounds.

    bounds: array shape (d, 2) with [low, high] per coordinate
    """
    b = np.ascontiguousarray(np.asarray(bounds, dtype=np.float64))
    if b.ndim != 2 or b.shape[1] != 2:
        raise ValueError("bounds must have shape (d, 2)")
    low = b[:, 0]
    high = b[:, 1]
    if np.any(high <= low):
        raise ValueError("each bound must satisfy high > low")

    n = int(n_samples)
    if n <= 0:
        raise ValueError("n_samples must be positive")

    rng = np.random.default_rng(seed)
    d = b.shape[0]

    x_best = np.zeros((d,), dtype=np.float64)
    f_best = np.float64(np.inf)

    X = low[None, :] + (high - low)[None, :] * rng.random((n, d), dtype=np.float64)
    for i in range(n):
        x = np.ascontiguousarray(X[i])
        f = np.float64(objective(x))
        if f < f_best:
            f_best = f
            x_best = x

    return RandomSearchResult(x_best=np.ascontiguousarray(x_best), f_best=f_best, n_evals=n)
