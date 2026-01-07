"""
noise.channels

Quantum channels as Kraus operators (dense).

Core:
- Channel(Kraus): apply to density matrices
- Constructors for common 1-qubit noise channels
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt

from qip.core.config import DEFAULT, QIPConfig
from qip.core.linalg import as_c128, dag

__all__ = [
    "Channel",
    "phase_flip",
    "depolarizing",
    "amplitude_damping",
]


@dataclass(frozen=True, slots=True)
class Channel:
    """
    Kraus channel: ρ -> Σ_k K_k ρ K_k†
    """

    kraus: tuple[npt.NDArray[np.complex128], ...]
    cfg: QIPConfig = DEFAULT

    @staticmethod
    def from_kraus(kraus: Sequence[npt.ArrayLike], *, cfg: QIPConfig = DEFAULT) -> "Channel":
        Ks = tuple(np.ascontiguousarray(as_c128(K, cfg=cfg)) for K in kraus)
        if len(Ks) == 0:
            raise ValueError("kraus list must be non-empty")
        d = Ks[0].shape[0]
        for K in Ks:
            if K.ndim != 2 or K.shape[0] != K.shape[1]:
                raise ValueError("each Kraus operator must be square")
            if K.shape != (d, d):
                raise ValueError("all Kraus operators must share the same shape")
        return Channel(kraus=Ks, cfg=cfg)

    @property
    def dim(self) -> int:
        return int(self.kraus[0].shape[0])

    def apply(self, rho: npt.ArrayLike) -> npt.NDArray[np.complex128]:
        ρ = as_c128(rho, cfg=self.cfg)
        if ρ.shape != (self.dim, self.dim):
            raise ValueError("rho has wrong shape")
        out = np.zeros_like(ρ)
        for K in self.kraus:
            out += K @ ρ @ dag(K, cfg=self.cfg)
        return np.ascontiguousarray(out)

    def compose(self, other: "Channel") -> "Channel":
        """
        Composition: self ∘ other (apply other first, then self).
        """
        if self.dim != other.dim:
            raise ValueError("channel composition requires matching dimensions")
        Ks = []
        for A in self.kraus:
            for B in other.kraus:
                Ks.append(np.ascontiguousarray(A @ B))
        return Channel.from_kraus(Ks, cfg=self.cfg)


def phase_flip(p: float, *, cfg: QIPConfig = DEFAULT) -> Channel:
    """
    Phase-flip channel: with prob p apply Z.
    """
    pp = float(p)
    if pp < 0.0 or pp > 1.0:
        raise ValueError("p must be in [0, 1]")
    I = np.eye(2, dtype=cfg.dtype_complex)
    Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=cfg.dtype_complex)
    K0 = np.sqrt(1.0 - pp) * I
    K1 = np.sqrt(pp) * Z
    return Channel.from_kraus([K0, K1], cfg=cfg)


def depolarizing(p: float, *, cfg: QIPConfig = DEFAULT) -> Channel:
    """
    1-qubit depolarizing channel:
      ρ -> (1 - p) ρ + (p/3)(XρX + YρY + ZρZ)
    """
    pp = float(p)
    if pp < 0.0 or pp > 1.0:
        raise ValueError("p must be in [0, 1]")
    I = np.eye(2, dtype=cfg.dtype_complex)
    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=cfg.dtype_complex)
    Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=cfg.dtype_complex)
    Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=cfg.dtype_complex)
    K0 = np.sqrt(1.0 - pp) * I
    a = np.sqrt(pp / 3.0)
    return Channel.from_kraus([K0, a * X, a * Y, a * Z], cfg=cfg)


def amplitude_damping(gamma: float, *, cfg: QIPConfig = DEFAULT) -> Channel:
    """
    Amplitude damping (T1) channel for a qubit with parameter gamma in [0, 1].
    """
    g = float(gamma)
    if g < 0.0 or g > 1.0:
        raise ValueError("gamma must be in [0, 1]")
    K0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - g)]], dtype=cfg.dtype_complex)
    K1 = np.array([[0.0, np.sqrt(g)], [0.0, 0.0]], dtype=cfg.dtype_complex)
    return Channel.from_kraus([K0, K1], cfg=cfg)
