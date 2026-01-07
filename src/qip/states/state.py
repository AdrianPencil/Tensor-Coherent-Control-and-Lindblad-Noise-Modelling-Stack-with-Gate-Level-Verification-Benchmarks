"""
states.state

StateVector is the default pure-state representation.

We keep this file small and explicit:
- store |ψ> as a complex128 contiguous vector
- provide normalization and basic inner-product operations
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from qip.core.config import DEFAULT, QIPConfig
from qip.core.linalg import as_c128

__all__ = [
    "StateVector",
]


@dataclass(frozen=True, slots=True)
class StateVector:
    """
    Pure state |ψ> in C^d stored as a complex128 1D vector.
    """

    vector: npt.NDArray[np.complex128]
    cfg: QIPConfig = DEFAULT

    @staticmethod
    def from_array(v: npt.ArrayLike, *, cfg: QIPConfig = DEFAULT) -> "StateVector":
        a = as_c128(v, cfg=cfg).reshape(-1)
        if a.size == 0:
            raise ValueError("state vector must be non-empty")
        return StateVector(vector=np.ascontiguousarray(a), cfg=cfg)

    @property
    def dim(self) -> int:
        return int(self.vector.size)

    def norm(self) -> np.float64:
        return np.float64(np.linalg.norm(self.vector))

    def normalized(self) -> "StateVector":
        n = self.norm()
        if n == 0.0:
            raise ValueError("cannot normalize the zero vector")
        v = self.vector / n
        return StateVector(vector=np.ascontiguousarray(v), cfg=self.cfg)

    def inner(self, other: "StateVector") -> np.complex128:
        if self.dim != other.dim:
            raise ValueError("inner product requires matching dimensions")
        return np.vdot(self.vector, other.vector).astype(np.complex128, copy=False)
