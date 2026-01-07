"""
dynamics.hamiltonian

A minimal Hamiltonian abstraction.

Two modes:
- static: H is a fixed dense Hermitian matrix
- time-dependent: H(t) provided by a callable returning a dense matrix

This keeps higher layers clean: solvers/propagators only depend on `matrix(t)`.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt

from qip.core.config import DEFAULT, QIPConfig
from qip.core.linalg import as_c128, is_hermitian
from qip.operators.pauli import pauli_string

__all__ = [
    "Hamiltonian",
]


@dataclass(frozen=True, slots=True)
class Hamiltonian:
    """
    Hamiltonian H(t) acting on C^d.

    If `H_of_t` is None, the Hamiltonian is static and stored in `H0`.
    """

    H0: npt.NDArray[np.complex128]
    H_of_t: Callable[[float], npt.NDArray[np.complex128]] | None = None
    cfg: QIPConfig = DEFAULT

    @staticmethod
    def from_matrix(H: npt.ArrayLike, *, cfg: QIPConfig = DEFAULT, check: bool = True) -> "Hamiltonian":
        m = as_c128(H, cfg=cfg)
        if m.ndim != 2 or m.shape[0] != m.shape[1]:
            raise ValueError("Hamiltonian must be a square 2D matrix")
        m = np.ascontiguousarray(m)
        if check and not is_hermitian(m, cfg=cfg):
            raise ValueError("Hamiltonian must be Hermitian within tolerances")
        return Hamiltonian(H0=m, H_of_t=None, cfg=cfg)

    @staticmethod
    def from_pauli_terms(
        terms: dict[str, float | complex],
        n_qubits: int,
        *,
        cfg: QIPConfig = DEFAULT,
        check: bool = True,
    ) -> "Hamiltonian":
        n = int(n_qubits)
        if n < 0:
            raise ValueError("n_qubits must be >= 0")
        dim = 1 << n
        H = np.zeros((dim, dim), dtype=cfg.dtype_complex)
        for label, coeff in terms.items():
            H += np.complex128(coeff) * pauli_string(label, cfg=cfg)
        H = np.ascontiguousarray(H)
        if check and not is_hermitian(H, cfg=cfg):
            raise ValueError("constructed Hamiltonian is not Hermitian within tolerances")
        return Hamiltonian(H0=H, H_of_t=None, cfg=cfg)

    @staticmethod
    def time_dependent(
        H_of_t: Callable[[float], npt.ArrayLike],
        dim: int,
        *,
        cfg: QIPConfig = DEFAULT,
        check: bool = True,
    ) -> "Hamiltonian":
        d = int(dim)
        if d <= 0:
            raise ValueError("dim must be positive")

        def _wrapped(t: float) -> npt.NDArray[np.complex128]:
            m = as_c128(H_of_t(float(t)), cfg=cfg)
            if m.shape != (d, d):
                raise ValueError("H(t) returned wrong shape")
            m = np.ascontiguousarray(m)
            if check and not is_hermitian(m, cfg=cfg):
                raise ValueError("H(t) must be Hermitian within tolerances")
            return m

        H0 = np.zeros((d, d), dtype=cfg.dtype_complex)
        return Hamiltonian(H0=np.ascontiguousarray(H0), H_of_t=_wrapped, cfg=cfg)

    @property
    def dim(self) -> int:
        return int(self.H0.shape[0])

    def matrix(self, t: float = 0.0) -> npt.NDArray[np.complex128]:
        if self.H_of_t is None:
            return self.H0
        return self.H_of_t(float(t))
