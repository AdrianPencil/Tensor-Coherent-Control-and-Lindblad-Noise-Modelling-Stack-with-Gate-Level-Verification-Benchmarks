"""
core.types

Centralized type aliases used across the project.

We keep the aliases focused:
- RealArray/ComplexArray: canonical ndarray types
- ArrayLikeReal/ArrayLikeComplex: inputs accepted by conversion helpers
"""

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np
import numpy.typing as npt

__all__ = [
    "F64",
    "C128",
    "RealArray",
    "ComplexArray",
    "ArrayLikeReal",
    "ArrayLikeComplex",
    "DTypeName",
    "ArraySpec",
]

F64: TypeAlias = np.float64
C128: TypeAlias = np.complex128

RealArray: TypeAlias = npt.NDArray[np.float64]
ComplexArray: TypeAlias = npt.NDArray[np.complex128]

ArrayLikeReal: TypeAlias = npt.ArrayLike
ArrayLikeComplex: TypeAlias = npt.ArrayLike

DTypeName: TypeAlias = Literal["float64", "complex128"]


@dataclass(frozen=True, slots=True)
class ArraySpec:
    """
    A small descriptor used for diagnostics and cache keys.

    This is not meant to be a full validation system -just enough structure
    for consistent logging/caching decisions.
    """

    shape: tuple[int, ...]
    dtype: str
    order: Literal["C", "F", "A"]

    @staticmethod
    def of(x: np.ndarray) -> "ArraySpec":
        order = "C" if x.flags.c_contiguous else ("F" if x.flags.f_contiguous else "A")
        return ArraySpec(shape=tuple(int(s) for s in x.shape), dtype=str(x.dtype), order=order)
