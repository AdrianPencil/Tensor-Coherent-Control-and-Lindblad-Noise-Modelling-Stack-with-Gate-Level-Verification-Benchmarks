"""
core.config

A single configuration object to keep numerics consistent across the stack.
This avoids threading dozens of parameters through every call.
"""

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np

__all__ = [
    "QIPConfig",
    "DEFAULT",
]


@dataclass(frozen=True, slots=True)
class QIPConfig:
    """
    Global-ish numerical defaults.

    dtype_real/dtype_complex enforce double precision.
    """

    dtype_real: type[np.floating] = np.float64
    dtype_complex: type[np.complexfloating] = np.complex128
    atol: float = 1.0e-12
    rtol: float = 1.0e-12
    seed: int | None = None
    cache_max_items: int = 256
    array_order: Literal["C", "F"] = "C"


DEFAULT: Final[QIPConfig] = QIPConfig()
