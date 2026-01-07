"""
control.controls

Control stack that maps pulses to time-dependent Hamiltonians.

We keep one concrete representation:
  H(t) = H0 + Σ_k u_k(t) H_k

where H0 and H_k are dense complex128 Hermitian matrices,
and u_k(t) are real-valued envelopes.
"""

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import numpy.typing as npt

from qip.core.config import DEFAULT, QIPConfig
from qip.core.linalg import as_c128, is_hermitian

__all__ = [
    "ControlTerm",
    "ControlHamiltonian",
]


@dataclass(frozen=True, slots=True)
class ControlTerm:
    """
    A single control term u(t) * Hc.
    """

    Hc: npt.NDArray[np.complex128]
    u_of_t: Callable[[float], float]

    def value(self, t: float) -> npt.NDArray[np.complex128]:
        return np.complex128(float(self.u_of_t(float(t)))) * self.Hc


@dataclass(frozen=True, slots=True)
class ControlHamiltonian:
    """
    Time-dependent Hamiltonian defined by a static drift plus control terms.
    """

    H0: npt.NDArray[np.complex128]
    terms: tuple[ControlTerm, ...]
    cfg: QIPConfig = DEFAULT

    @staticmethod
    def from_terms(
        H0: npt.ArrayLike,
        terms: Sequence[ControlTerm],
        *,
        cfg: QIPConfig = DEFAULT,
        check: bool = True,
    ) -> "ControlHamiltonian":
        H0m = np.ascontiguousarray(as_c128(H0, cfg=cfg))
        if H0m.ndim != 2 or H0m.shape[0] != H0m.shape[1]:
            raise ValueError("H0 must be a square matrix")
        if check and not is_hermitian(H0m, cfg=cfg):
            raise ValueError("H0 must be Hermitian within tolerances")

        tt = tuple(terms)
        d = H0m.shape[0]
        for term in tt:
            Hc = np.ascontiguousarray(as_c128(term.Hc, cfg=cfg))
            if Hc.shape != (d, d):
                raise ValueError("control term matrix has wrong shape")
            if check and not is_hermitian(Hc, cfg=cfg):
                raise ValueError("control term matrix must be Hermitian within tolerances")

        return ControlHamiltonian(H0=H0m, terms=tt, cfg=cfg)

    @property
    def dim(self) -> int:
        return int(self.H0.shape[0])

    def matrix(self, t: float) -> npt.NDArray[np.complex128]:
        Ht = self.H0.copy()
        for term in self.terms:
            Ht += term.value(t)
        return np.ascontiguousarray(Ht)

    def matrix_on_grid(self, t: npt.ArrayLike) -> npt.NDArray[np.complex128]:
        """
        Return H(t_i) for each t_i.

        Output shape: (T, d, d)
        """
        tt = np.ascontiguousarray(np.asarray(t, dtype=np.float64).reshape(-1))
        d = self.dim
        out = np.zeros((tt.size, d, d), dtype=self.cfg.dtype_complex)
        for i, ti in enumerate(tt.tolist()):
            out[i] = self.matrix(float(ti))
        return np.ascontiguousarray(out)
