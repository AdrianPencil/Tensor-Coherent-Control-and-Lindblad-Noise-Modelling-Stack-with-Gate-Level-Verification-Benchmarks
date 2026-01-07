"""
dynamics.adiabatic

A minimal adiabatic schedule utility.

We represent a scalar schedule s(t) in [0, 1] and provide:
- value(t)
- derivative(t) (for diagnostics / adiabaticity metrics later)
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = [
    "AdiabaticSchedule",
]


@dataclass(frozen=True, slots=True)
class AdiabaticSchedule:
    """
    Smooth-ish schedule based on a cosine ramp.

    For t in [0, T]:
      s(t) = 0.5 * (1 - cos(pi t / T))
    """

    T: float

    def value(self, t: float) -> np.float64:
        T = float(self.T)
        if T <= 0.0:
            raise ValueError("T must be positive")
        x = float(t) / T
        x = float(np.clip(x, 0.0, 1.0))
        return np.float64(0.5 * (1.0 - np.cos(np.pi * x)))

    def derivative(self, t: float) -> np.float64:
        T = float(self.T)
        if T <= 0.0:
            raise ValueError("T must be positive")
        x = float(t) / T
        if x <= 0.0 or x >= 1.0:
            return np.float64(0.0)
        return np.float64(0.5 * (np.pi / T) * np.sin(np.pi * x))

    def values(self, t: npt.ArrayLike) -> npt.NDArray[np.float64]:
        tt = np.asarray(t, dtype=np.float64).reshape(-1)
        if float(self.T) <= 0.0:
            raise ValueError("T must be positive")
        x = np.clip(tt / float(self.T), 0.0, 1.0)
        s = 0.5 * (1.0 - np.cos(np.pi * x))
        return np.ascontiguousarray(s.astype(np.float64, copy=False))
