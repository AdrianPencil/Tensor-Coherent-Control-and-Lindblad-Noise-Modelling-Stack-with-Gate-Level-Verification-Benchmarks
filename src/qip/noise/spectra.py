"""
noise.spectra

Noise power spectral density (PSD) models.

We represent PSD as S(ω) with ω in rad/s.

This module provides:
- PSDModel: a callable wrapper with parameters
- basic models: white, ohmic, one_over_f
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt

__all__ = [
    "PSDModel",
    "white_psd",
    "ohmic_psd",
    "one_over_f_psd",
]


@dataclass(frozen=True, slots=True)
class PSDModel:
    """
    PSD model S(ω) returning float64 values, vectorized over ω.

    The callable must accept a float64 ndarray and return float64 ndarray.
    """

    eval: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]

    def __call__(self, omega: npt.ArrayLike) -> npt.NDArray[np.float64]:
        w = np.asarray(omega, dtype=np.float64)
        w = np.ascontiguousarray(w)
        out = self.eval(w)
        return np.ascontiguousarray(np.asarray(out, dtype=np.float64))


def white_psd(S0: float) -> PSDModel:
    """
    S(ω) = S0
    """
    s0 = np.float64(S0)

    def _eval(w: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.full_like(w, s0, dtype=np.float64)

    return PSDModel(eval=_eval)


def ohmic_psd(eta: float, omega_c: float) -> PSDModel:
    """
    Ohmic with exponential cutoff:
      S(ω) = eta * ω * exp(-|ω|/ω_c)
    """
    e = np.float64(eta)
    wc = np.float64(omega_c)
    if wc <= 0.0:
        raise ValueError("omega_c must be positive")

    def _eval(w: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        a = np.abs(w)
        return e * a * np.exp(-a / wc)

    return PSDModel(eval=_eval)


def one_over_f_psd(A: float, alpha: float, omega_ref: float) -> PSDModel:
    """
    1/f^alpha style:
      S(ω) = A * (|ω|/ω_ref)^(-alpha)
    """
    a0 = np.float64(A)
    al = np.float64(alpha)
    w0 = np.float64(omega_ref)
    if w0 <= 0.0:
        raise ValueError("omega_ref must be positive")

    def _eval(w: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        a = np.abs(w) / w0
        a = np.maximum(a, np.float64(1.0e-30))
        return a0 * np.power(a, -al)

    return PSDModel(eval=_eval)
