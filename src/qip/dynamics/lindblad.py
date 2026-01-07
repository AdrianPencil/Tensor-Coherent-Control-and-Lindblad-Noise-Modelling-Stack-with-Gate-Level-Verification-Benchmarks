"""
dynamics.lindblad

Lindblad master equation for density matrices:

dρ/dt = -i [H, ρ] + Σ_k γ_k (L_k ρ L_k† - 1/2 {L_k† L_k, ρ})

This module keeps a minimal representation:
- H(t): Hamiltonian
- collapse operators L_k (dense complex128)
- rates γ_k (float64)
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt

from qip.core.config import DEFAULT, QIPConfig
from qip.core.linalg import as_c128, dag
from qip.dynamics.hamiltonian import Hamiltonian

__all__ = [
    "LindbladModel",
]


@dataclass(frozen=True, slots=True)
class LindbladModel:
    """
    Lindblad model for a d-dimensional density matrix.
    """

    H: Hamiltonian
    collapse_ops: tuple[npt.NDArray[np.complex128], ...]
    rates: npt.NDArray[np.float64]
    cfg: QIPConfig = DEFAULT

    @staticmethod
    def from_operators(
        H: Hamiltonian,
        collapse_ops: Sequence[npt.ArrayLike],
        rates: Sequence[float],
        *,
        cfg: QIPConfig = DEFAULT,
    ) -> "LindbladModel":
        ops = tuple(np.ascontiguousarray(as_c128(L, cfg=cfg)) for L in collapse_ops)
        g = np.ascontiguousarray(np.asarray(rates, dtype=np.float64).reshape(-1))
        if len(ops) != int(g.size):
            raise ValueError("collapse_ops and rates must have the same length")
        d = H.dim
        for L in ops:
            if L.shape != (d, d):
                raise ValueError("collapse operator has wrong shape")
        if np.any(g < 0.0):
            raise ValueError("rates must be non-negative")
        return LindbladModel(H=H, collapse_ops=ops, rates=g, cfg=cfg)

    @property
    def dim(self) -> int:
        return self.H.dim

    def drho_dt(self, t: float, rho: npt.ArrayLike) -> npt.NDArray[np.complex128]:
        """
        Vectorized RHS for the master equation, returning a (d, d) complex128 array.
        """
        cfg = self.cfg
        ρ = as_c128(rho, cfg=cfg)
        if ρ.shape != (self.dim, self.dim):
            raise ValueError("rho has wrong shape")

        Ht = self.H.matrix(t)
        comm = Ht @ ρ - ρ @ Ht
        out = (-1j) * comm

        for L, gamma in zip(self.collapse_ops, self.rates, strict=True):
            if gamma == 0.0:
                continue
            Lρ = L @ ρ
            Ld = dag(L, cfg=cfg)
            jump = Lρ @ Ld
            LdL = Ld @ L
            anti = LdL @ ρ + ρ @ LdL
            out += np.complex128(gamma) * (jump - 0.5 * anti)

        return np.ascontiguousarray(out)
