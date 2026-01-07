"""
operators.basis

A minimal basis layer used by states/operators/circuits.

Design goals:
- explicit dimension bookkeeping
- fast vector construction (contiguous complex128)
- simple projector/outer-product utilities
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from qip.core.config import DEFAULT, QIPConfig
from qip.core.linalg import as_c128

__all__ = [
    "ComputationalBasis",
    "ket",
    "bra",
    "projector",
    "outer",
]

_I1: Final[np.complex128] = np.complex128(1.0 + 0.0j)


@dataclass(frozen=True, slots=True)
class ComputationalBasis:
    """
    Computational basis for n qubits: |0...0>, |0...1>, ..., |1...1>.

    Index convention:
      index in [0, 2^n - 1] corresponds to bitstring in standard binary.
    """

    n_qubits: int
    cfg: QIPConfig = DEFAULT

    @property
    def dim(self) -> int:
        return 1 << int(self.n_qubits)

    def ket(self, index: int) -> npt.NDArray[np.complex128]:
        return ket(index, self.dim, cfg=self.cfg)

    def projector(self, index: int) -> npt.NDArray[np.complex128]:
        v = self.ket(index)
        return projector(v, cfg=self.cfg)


def ket(index: int, dim: int, *, cfg: QIPConfig = DEFAULT) -> npt.NDArray[np.complex128]:
    """
    |index> in C^dim as a complex128, C-contiguous vector.
    """
    i = int(index)
    d = int(dim)
    if i < 0 or i >= d:
        raise ValueError(f"ket index out of range: index={i}, dim={d}")
    v = np.zeros((d,), dtype=cfg.dtype_complex)
    v[i] = _I1
    return np.ascontiguousarray(v)


def bra(v: npt.ArrayLike, *, cfg: QIPConfig = DEFAULT) -> npt.NDArray[np.complex128]:
    """
    <v| as a (1, d) row vector.
    """
    a = as_c128(v, cfg=cfg).reshape(-1)
    return np.ascontiguousarray(np.conjugate(a)[None, :])


def outer(a: npt.ArrayLike, b: npt.ArrayLike, *, cfg: QIPConfig = DEFAULT) -> npt.NDArray[np.complex128]:
    """
    |a><b| with complex128 accumulation.
    """
    va = as_c128(a, cfg=cfg).reshape(-1)
    vb = as_c128(b, cfg=cfg).reshape(-1)
    m = va[:, None] @ np.conjugate(vb)[None, :]
    return np.ascontiguousarray(m)


def projector(v: npt.ArrayLike, *, cfg: QIPConfig = DEFAULT) -> npt.NDArray[np.complex128]:
    """
    |v><v| (not normalized unless v is normalized).
    """
    return outer(v, v, cfg=cfg)
