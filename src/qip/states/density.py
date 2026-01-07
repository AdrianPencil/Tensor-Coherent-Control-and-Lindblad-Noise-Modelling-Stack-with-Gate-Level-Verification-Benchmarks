"""
states.density

DensityMatrix is the default mixed-state representation.

Core invariants:
- ρ is square, complex128, contiguous
- trace(ρ) is typically 1 (not enforced on construction)
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from qip.core.config import DEFAULT, QIPConfig
from qip.core.linalg import as_c128, dag, trace
from qip.states.state import StateVector

__all__ = [
    "DensityMatrix",
    "density_from_state",
]


@dataclass(frozen=True, slots=True)
class DensityMatrix:
    """
    Density matrix ρ in C^{dxd}.
    """

    rho: npt.NDArray[np.complex128]
    cfg: QIPConfig = DEFAULT

    @staticmethod
    def from_matrix(rho: npt.ArrayLike, *, cfg: QIPConfig = DEFAULT) -> "DensityMatrix":
        m = as_c128(rho, cfg=cfg)
        if m.ndim != 2 or m.shape[0] != m.shape[1]:
            raise ValueError("density matrix must be a square 2D matrix")
        return DensityMatrix(rho=np.ascontiguousarray(m), cfg=cfg)

    @property
    def dim(self) -> int:
        return int(self.rho.shape[0])

    def tr(self) -> np.complex128:
        return trace(self.rho, cfg=self.cfg)

    def is_hermitian(self) -> bool:
        return np.allclose(self.rho, dag(self.rho, cfg=self.cfg), atol=self.cfg.atol, rtol=self.cfg.rtol)

    def is_trace_one(self) -> bool:
        t = self.tr()
        return np.allclose(t, np.complex128(1.0 + 0.0j), atol=self.cfg.atol, rtol=self.cfg.rtol)

    def expectation(self, op: npt.ArrayLike) -> np.complex128:
        a = as_c128(op, cfg=self.cfg)
        if a.ndim != 2 or a.shape != self.rho.shape:
            raise ValueError("expectation requires an operator with matching shape")
        val = np.trace(self.rho @ a, dtype=np.complex128)
        return val.astype(np.complex128, copy=False)

    def evolved_by_unitary(self, u: npt.ArrayLike) -> "DensityMatrix":
        U = as_c128(u, cfg=self.cfg)
        if U.ndim != 2 or U.shape != self.rho.shape:
            raise ValueError("unitary evolution requires matching shapes")
        rho_p = U @ self.rho @ np.conjugate(U).T
        return DensityMatrix(rho=np.ascontiguousarray(rho_p), cfg=self.cfg)


def density_from_state(state: StateVector) -> DensityMatrix:
    v = state.vector.reshape(-1, 1)
    rho = v @ np.conjugate(v).T
    return DensityMatrix(rho=np.ascontiguousarray(rho), cfg=state.cfg)
