"""
operators.sparse

A minimal sparse operator representation without SciPy.

This is intentionally small and pragmatic:
- COO storage (rows, cols, data) with shape
- fast matvec for vectors
- conversion to dense for interop
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from qip.core.config import DEFAULT, QIPConfig
from qip.core.linalg import as_c128

__all__ = [
    "SparseOperatorCOO",
]

_ZERO: Final[np.complex128] = np.complex128(0.0 + 0.0j)


@dataclass(frozen=True, slots=True)
class SparseOperatorCOO:
    """
    COO sparse matrix (complex128).

    rows, cols, data must have the same length nnz.
    """

    rows: npt.NDArray[np.int64]
    cols: npt.NDArray[np.int64]
    data: npt.NDArray[np.complex128]
    shape: tuple[int, int]
    cfg: QIPConfig = DEFAULT

    @staticmethod
    def from_coo(
        rows: npt.ArrayLike,
        cols: npt.ArrayLike,
        data: npt.ArrayLike,
        shape: tuple[int, int],
        *,
        cfg: QIPConfig = DEFAULT,
    ) -> "SparseOperatorCOO":
        r = np.ascontiguousarray(np.asarray(rows, dtype=np.int64).reshape(-1))
        c = np.ascontiguousarray(np.asarray(cols, dtype=np.int64).reshape(-1))
        d = np.ascontiguousarray(as_c128(data, cfg=cfg).reshape(-1))
        if r.shape != c.shape or r.shape != d.shape:
            raise ValueError("rows, cols, data must have identical shapes")
        m, n = int(shape[0]), int(shape[1])
        if m <= 0 or n <= 0:
            raise ValueError("shape must be positive in both dimensions")
        return SparseOperatorCOO(rows=r, cols=c, data=d, shape=(m, n), cfg=cfg)

    @property
    def nnz(self) -> int:
        return int(self.data.size)

    def to_dense(self) -> npt.NDArray[np.complex128]:
        m, n = self.shape
        out = np.zeros((m, n), dtype=self.cfg.dtype_complex)
        if self.nnz == 0:
            return np.ascontiguousarray(out)
        np.add.at(out, (self.rows, self.cols), self.data)
        return np.ascontiguousarray(out)

    def matvec(self, x: npt.ArrayLike) -> npt.NDArray[np.complex128]:
        xv = as_c128(x, cfg=self.cfg).reshape(-1)
        m, n = self.shape
        if xv.size != n:
            raise ValueError(f"matvec dimension mismatch: got {xv.size}, expected {n}")
        y = np.zeros((m,), dtype=self.cfg.dtype_complex)
        if self.nnz == 0:
            return np.ascontiguousarray(y)
        y_part = self.data * xv[self.cols]
        np.add.at(y, self.rows, y_part)
        return np.ascontiguousarray(y)
