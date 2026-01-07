"""
core.linalg

Small, numerically-stable linear algebra building blocks.
We keep this tight because higher layers (operators/dynamics/circuits) should
compose from a small foundation.

Main idea:
- enforce contiguous arrays and canonical dtypes
- provide a couple of "physics-native" primitives (dagger, expm for Hermitian)
"""

from __future__ import annotations

from typing import overload

import numpy as np
import numpy.typing as npt

from qip.core.config import DEFAULT, QIPConfig
from qip.core.types import ComplexArray, RealArray

__all__ = [
    "as_f64",
    "as_c128",
    "dag",
    "trace",
    "kron",
    "is_hermitian",
    "expm_hermitian",
]


@overload
def as_f64(x: npt.ArrayLike, *, cfg: QIPConfig = DEFAULT) -> RealArray: ...


def as_f64(x: npt.ArrayLike, *, cfg: QIPConfig = DEFAULT) -> RealArray:
    """
    Convert input into float64 ndarray with desired contiguity.
    """
    a = np.asarray(x, dtype=cfg.dtype_real, order=cfg.array_order)
    if not a.flags.c_contiguous and cfg.array_order == "C":
        a = np.ascontiguousarray(a)
    return a


@overload
def as_c128(x: npt.ArrayLike, *, cfg: QIPConfig = DEFAULT) -> ComplexArray: ...


def as_c128(x: npt.ArrayLike, *, cfg: QIPConfig = DEFAULT) -> ComplexArray:
    """
    Convert input into complex128 ndarray with desired contiguity.
    """
    a = np.asarray(x, dtype=cfg.dtype_complex, order=cfg.array_order)
    if not a.flags.c_contiguous and cfg.array_order == "C":
        a = np.ascontiguousarray(a)
    return a


def dag(a: npt.ArrayLike, *, cfg: QIPConfig = DEFAULT) -> ComplexArray:
    """
    Conjugate transpose (dagger).
    """
    m = as_c128(a, cfg=cfg)
    return np.conjugate(m).T


def trace(a: npt.ArrayLike, *, cfg: QIPConfig = DEFAULT) -> np.complex128:
    """
    Trace with complex128 accumulation.
    """
    m = as_c128(a, cfg=cfg)
    return np.trace(m, dtype=np.complex128)


def kron(a: npt.ArrayLike, b: npt.ArrayLike, *, cfg: QIPConfig = DEFAULT) -> ComplexArray:
    """
    Kronecker product as complex128, contiguous.
    """
    ka = as_c128(a, cfg=cfg)
    kb = as_c128(b, cfg=cfg)
    out = np.kron(ka, kb)
    return np.ascontiguousarray(out)


def is_hermitian(a: npt.ArrayLike, *, cfg: QIPConfig = DEFAULT) -> bool:
    """
    Check Hermiticity within cfg tolerances.
    """
    m = as_c128(a, cfg=cfg)
    return np.allclose(m, dag(m, cfg=cfg), atol=cfg.atol, rtol=cfg.rtol)


def expm_hermitian(h: npt.ArrayLike, t: float, *, cfg: QIPConfig = DEFAULT) -> ComplexArray:
    """
    U = exp(-i H t) for Hermitian H using eigen-decomposition.

    This avoids SciPy dependency and is stable for Hermitian matrices:
      H = V diag(w) V†
      exp(-i H t) = V diag(exp(-i w t)) V†
    """
    H = as_c128(h, cfg=cfg)
    w, v = np.linalg.eigh(H)
    phases = np.exp(-1j * w * np.float64(t)).astype(np.complex128, copy=False)
    U = (v * phases) @ np.conjugate(v).T
    return np.ascontiguousarray(U)
