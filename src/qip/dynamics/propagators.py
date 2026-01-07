"""
dynamics.propagators

Minimal propagation helpers built on:
- expm_hermitian for unitary evolution
- a solver stepper for Lindblad density evolution

This stays small on purpose: detailed integrators belong in solvers.py.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from qip.core.config import DEFAULT, QIPConfig
from qip.core.linalg import as_c128, expm_hermitian
from qip.dynamics.hamiltonian import Hamiltonian
from qip.dynamics.lindblad import LindbladModel
from qip.dynamics.solvers import rk4_step_matrix

__all__ = [
    "UnitaryPropagator",
    "propagate_density_lindblad_rk4",
]


@dataclass(frozen=True, slots=True)
class UnitaryPropagator:
    """
    Unitary propagator for a (static) Hermitian Hamiltonian.

    U(dt) = exp(-i H dt)
    """

    H: npt.NDArray[np.complex128]
    cfg: QIPConfig = DEFAULT

    @staticmethod
    def from_hamiltonian(H: Hamiltonian, *, cfg: QIPConfig = DEFAULT) -> "UnitaryPropagator":
        H0 = H.matrix(0.0)
        return UnitaryPropagator(H=np.ascontiguousarray(as_c128(H0, cfg=cfg)), cfg=cfg)

    def U(self, dt: float) -> npt.NDArray[np.complex128]:
        return expm_hermitian(self.H, float(dt), cfg=self.cfg)

    def apply_to_state(self, psi: npt.ArrayLike, dt: float) -> npt.NDArray[np.complex128]:
        v = as_c128(psi, cfg=self.cfg).reshape(-1)
        if v.size != self.H.shape[0]:
            raise ValueError("state dimension mismatch")
        return np.ascontiguousarray(self.U(dt) @ v)

    def apply_to_density(self, rho: npt.ArrayLike, dt: float) -> npt.NDArray[np.complex128]:
        ρ = as_c128(rho, cfg=self.cfg)
        d = self.H.shape[0]
        if ρ.shape != (d, d):
            raise ValueError("density dimension mismatch")
        U = self.U(dt)
        ρp = U @ ρ @ np.conjugate(U).T
        return np.ascontiguousarray(ρp)


def propagate_density_lindblad_rk4(
    model: LindbladModel,
    rho0: npt.ArrayLike,
    t0: float,
    t1: float,
    n_steps: int,
) -> npt.NDArray[np.complex128]:
    """
    Propagate a density matrix using fixed-step RK4 over [t0, t1].

    Returns the final density matrix at t1.
    """
    ρ = as_c128(rho0, cfg=model.cfg)
    if ρ.shape != (model.dim, model.dim):
        raise ValueError("rho0 has wrong shape")
    n = int(n_steps)
    if n <= 0:
        raise ValueError("n_steps must be positive")

    t = float(t0)
    dt = (float(t1) - float(t0)) / float(n)
    for _ in range(n):
        ρ = rk4_step_matrix(model.drho_dt, t, ρ, dt)
        t += dt
    return np.ascontiguousarray(ρ)
