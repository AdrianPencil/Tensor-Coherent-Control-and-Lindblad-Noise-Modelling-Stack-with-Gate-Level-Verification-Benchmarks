"""
devices.photonic

Minimal photonic qubit model using a dual-rail encoding abstraction.

We treat the logical qubit as a 2D Hilbert space:
  |0> = |10>  (one photon in rail A)
  |1> = |01>  (one photon in rail B)

In this reduced model:
- drift is typically 0
- controls correspond to beamsplitter-like rotations (X/Y) and phase shifts (Z)
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from qip.control.calibration import Calibration
from qip.control.controls import ControlHamiltonian, ControlTerm
from qip.devices.abstract_device import EffectiveNoise
from qip.operators.pauli import pauli

__all__ = [
    "PhotonicDualRailQubit",
]


@dataclass(frozen=True, slots=True)
class PhotonicDualRailQubit:
    drive_phase: float = 0.0
    drive_scale: float = 1.0
    gamma_1: float | None = None
    gamma_phi: float | None = None

    @property
    def name(self) -> str:
        return "photonic_dual_rail_qubit"

    @property
    def dim(self) -> int:
        return 2

    def drift_hamiltonian(self) -> npt.NDArray[np.complex128]:
        return np.ascontiguousarray(np.zeros((2, 2), dtype=np.complex128))

    def control_hamiltonian(self, calibration: Calibration) -> ControlHamiltonian:
        X = pauli("X")
        Y = pauli("Y")

        phi = float(self.drive_phase) + float(calibration.phase_offset)
        Hc = 0.5 * (np.cos(phi) * X + np.sin(phi) * Y)
        Hc = np.ascontiguousarray(Hc)

        scale = float(self.drive_scale) * float(calibration.drive_scale)

        def u_of_t(t: float) -> float:
            return scale

        term = ControlTerm(Hc=Hc, u_of_t=u_of_t)
        return ControlHamiltonian.from_terms(self.drift_hamiltonian(), [term])

    def effective_noise(self) -> EffectiveNoise | None:
        if self.gamma_1 is None and self.gamma_phi is None:
            return None
        g1 = np.float64(0.0 if self.gamma_1 is None else float(self.gamma_1))
        gp = np.float64(0.0 if self.gamma_phi is None else float(self.gamma_phi))
        return EffectiveNoise(gamma_1=g1, gamma_phi=gp)
