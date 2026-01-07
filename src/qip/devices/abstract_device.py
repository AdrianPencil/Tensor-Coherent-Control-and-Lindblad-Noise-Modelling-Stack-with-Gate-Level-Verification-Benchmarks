"""
devices.abstract_device

Device adapter protocol.

A "device" provides:
- dimension (Hilbert space size)
- a drift Hamiltonian
- a mapping from named controls -> control Hamiltonian terms
- optional noise rates (effective) for building Lindblad models
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from qip.control.calibration import Calibration
from qip.control.controls import ControlHamiltonian

__all__ = [
    "EffectiveNoise",
    "Device",
]


@dataclass(frozen=True, slots=True)
class EffectiveNoise:
    """
    Minimal effective noise rates (rad/s).
    """

    gamma_1: np.float64
    gamma_phi: np.float64


class Device(Protocol):
    @property
    def name(self) -> str:
        ...

    @property
    def dim(self) -> int:
        ...

    def drift_hamiltonian(self) -> npt.NDArray[np.complex128]:
        ...

    def control_hamiltonian(self, calibration: Calibration) -> ControlHamiltonian:
        ...

    def effective_noise(self) -> EffectiveNoise | None:
        ...
