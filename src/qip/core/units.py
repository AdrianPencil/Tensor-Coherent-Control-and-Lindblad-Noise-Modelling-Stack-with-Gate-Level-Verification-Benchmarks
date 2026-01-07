"""
core.units

We keep "units" minimal and explicit: define a small set of physical constants
and a couple of helpers to standardize conventions across modules.

Everything here is SI unless otherwise stated.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np

__all__ = [
    "PhysicalConstants",
    "CONST",
    "GHz",
    "MHz",
    "kHz",
    "ns",
    "us",
    "ms",
    "to_angular_frequency",
]

GHz: Final[float] = 1.0e9
MHz: Final[float] = 1.0e6
kHz: Final[float] = 1.0e3

ns: Final[float] = 1.0e-9
us: Final[float] = 1.0e-6
ms: Final[float] = 1.0e-3


@dataclass(frozen=True, slots=True)
class PhysicalConstants:
    """
    Small subset of constants commonly needed for qubit/device modeling.
    """

    h: float
    hbar: float
    k_B: float
    e: float


CONST: Final[PhysicalConstants] = PhysicalConstants(
    h=6.62607015e-34,
    hbar=1.054571817e-34,
    k_B=1.380649e-23,
    e=1.602176634e-19,
)


def to_angular_frequency(f_hz: float | np.ndarray) -> float | np.ndarray:
    """
    Convert frequency f (Hz) to angular frequency ω (rad/s): ω = 2π f.
    """
    return 2.0 * np.pi * f_hz
