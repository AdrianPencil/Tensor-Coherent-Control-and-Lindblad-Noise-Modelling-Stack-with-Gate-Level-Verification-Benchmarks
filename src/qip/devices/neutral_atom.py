"""
devices.neutral_atom

Minimal neutral-atom qubit (2-level), consistent with other devices.

Model:
- drift: H0 = (ω0 / 2) Z
- drive: Hd(t) = (Ω(t) / 2) (cos φ X + sin φ Y)

Noise rates are optional and can be attached for early Lindblad modeling.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from qip.control.calibration import Calibration
from qip.control.controls import ControlHamiltonian, ControlTerm
from qip.devices.abstract_device import EffectiveNoise
from qip.operators.pauli import pauli

__all__ = [
    "NeutralAtomQubit2Level",
]


@dataclass(frozen=True, slots=True)
class NeutralAtomQubit2Level:
    omega_0: float
    drive_phase: float = 0.0
    gamma_1: float | None = None
    gamma_phi: float | None = None

    @property
    def name(self) -> str:
        return "neutral_atom_qubit_2level"

    @property
    def dim(self) -> int:
        return 2

    def drift_hamiltonian(self) -> npt.NDArray[np.complex128]:
        Z = pauli("Z")
        H0 = 0.5 * np.complex128(float(self.omega_0)) * Z
        return np.ascontiguousarray(H0)

    def control_hamiltonian(self, calibration: Calibration) -> ControlHamiltonian:
        X = pauli("X")
        Y = pauli("Y")

        phi = float(self.drive_phase) + float(calibration.phase_offset)
        Hc = 0.5 * (np.cos(phi) * X + np.sin(phi) * Y)
        Hc = np.ascontiguousarray(Hc)

        scale = float(calibration.drive_scale)

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
