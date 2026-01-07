"""
operators.operator

Dense operator wrapper with a clean, explicit API.

This is the main "operator" type used across the stack.
Sparse operators live in operators.sparse and can be converted to dense.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from qip.core.config import DEFAULT, QIPConfig
from qip.core.linalg import as_c128, dag, is_hermitian, kron

__all__ = [
    "Operator",
]


@dataclass(frozen=True, slots=True)
class Operator:
    """
    Dense operator with complex128 matrix storage.

    The matrix is assumed to act on column vectors: |ψ'> = A |ψ|.
    """

    matrix: npt.NDArray[np.complex128]
    cfg: QIPConfig = DEFAULT

    @staticmethod
    def from_matrix(a: npt.ArrayLike, *, cfg: QIPConfig = DEFAULT) -> "Operator":
        m = as_c128(a, cfg=cfg)
        if m.ndim != 2 or m.shape[0] != m.shape[1]:
            raise ValueError("Operator must be a square 2D matrix")
        return Operator(matrix=np.ascontiguousarray(m), cfg=cfg)

    @staticmethod
    def identity(dim: int, *, cfg: QIPConfig = DEFAULT) -> "Operator":
        d = int(dim)
        if d <= 0:
            raise ValueError("dim must be positive")
        m = np.eye(d, dtype=cfg.dtype_complex)
        return Operator(matrix=np.ascontiguousarray(m), cfg=cfg)

    @property
    def dim(self) -> int:
        return int(self.matrix.shape[0])

    def dagger(self) -> "Operator":
        return Operator(matrix=np.ascontiguousarray(dag(self.matrix, cfg=self.cfg)), cfg=self.cfg)

    def is_hermitian(self) -> bool:
        return is_hermitian(self.matrix, cfg=self.cfg)

    def apply(self, state: npt.ArrayLike) -> npt.NDArray[np.complex128]:
        v = as_c128(state, cfg=self.cfg).reshape(-1)
        if v.size != self.dim:
            raise ValueError(f"apply dimension mismatch: got {v.size}, expected {self.dim}")
        out = self.matrix @ v
        return np.ascontiguousarray(out)

    def __matmul__(self, other: "Operator") -> "Operator":
        if self.dim != other.dim:
            raise ValueError("operator composition requires matching dimensions")
        m = self.matrix @ other.matrix
        return Operator(matrix=np.ascontiguousarray(m), cfg=self.cfg)

    def tensor(self, other: "Operator") -> "Operator":
        m = kron(self.matrix, other.matrix, cfg=self.cfg)
        return Operator(matrix=np.ascontiguousarray(m), cfg=self.cfg)
