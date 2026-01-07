"""
circuits.models

Ideal + noisy gate models.

We keep two models:
- IdealGateModel: uses the provided gate matrix
- DepolarizingGateModel: applies a depolarizing channel after each 1-qubit gate

Noise here is intentionally simple; richer noise models live under noise/.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from qip.core.config import DEFAULT, QIPConfig
from qip.core.linalg import as_c128, dag
from qip.circuits.gates import Gate
from qip.noise.channels import depolarizing

__all__ = [
    "IdealGateModel",
    "DepolarizingGateModel",
]


@dataclass(frozen=True, slots=True)
class IdealGateModel:
    cfg: QIPConfig = DEFAULT

    def unitary(self, gate: Gate) -> npt.NDArray[np.complex128]:
        U = as_c128(gate.U, cfg=self.cfg)
        return np.ascontiguousarray(U)


@dataclass(frozen=True, slots=True)
class DepolarizingGateModel:
    p_1q: float
    cfg: QIPConfig = DEFAULT

    def apply_to_density(self, gate: Gate, rho: npt.ArrayLike) -> npt.NDArray[np.complex128]:
        ρ = as_c128(rho, cfg=self.cfg)
        U = as_c128(gate.U, cfg=self.cfg)
        ρp = U @ ρ @ np.conjugate(U).T
        ρp = np.ascontiguousarray(ρp)

        if gate.arity == 1 and float(self.p_1q) > 0.0:
            ch = depolarizing(float(self.p_1q), cfg=self.cfg)
            return ch.apply(ρp)

        return ρp
