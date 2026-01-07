"""
metrics.leakage

Leakage metrics for models that embed a computational subspace inside a
larger Hilbert space.

We keep a single function:
- leakage_probability: 1 - Tr(P ρ) where P projects onto computational subspace
"""

import numpy as np
import numpy.typing as npt

from qip.core.config import DEFAULT, QIPConfig
from qip.core.linalg import as_c128, trace

__all__ = [
    "leakage_probability",
]


def leakage_probability(
    rho: npt.ArrayLike,
    proj_comp: npt.ArrayLike,
    *,
    cfg: QIPConfig = DEFAULT,
) -> np.float64:
    ρ = as_c128(rho, cfg=cfg)
    P = as_c128(proj_comp, cfg=cfg)
    if ρ.ndim != 2 or ρ.shape[0] != ρ.shape[1]:
        raise ValueError("rho must be a square matrix")
    if P.shape != ρ.shape:
        raise ValueError("proj_comp must match rho shape")

    pop = trace(P @ ρ, cfg=cfg).real
    pop = float(np.clip(pop, 0.0, 1.0))
    return np.float64(1.0 - pop)
