"""
metrics.fidelity

Core fidelity measures used across tests and workflows.

We keep two main functions:
- state_fidelity(|ψ>, |φ>) = |<ψ|φ>|^2
- density_fidelity(ρ, σ) using Uhlmann fidelity via eigen-decomposition

For density fidelity:
  F(ρ, σ) = (Tr sqrt( sqrt(ρ) σ sqrt(ρ) ))^2
"""

import numpy as np
import numpy.typing as npt

from qip.core.config import DEFAULT, QIPConfig
from qip.core.linalg import as_c128, dag, trace

__all__ = [
    "state_fidelity",
    "density_fidelity",
]


def state_fidelity(
    psi: npt.ArrayLike,
    phi: npt.ArrayLike,
    *,
    cfg: QIPConfig = DEFAULT,
) -> np.float64:
    a = as_c128(psi, cfg=cfg).reshape(-1)
    b = as_c128(phi, cfg=cfg).reshape(-1)
    if a.size != b.size:
        raise ValueError("state_fidelity requires matching dimensions")
    inner = np.vdot(a, b).astype(np.complex128, copy=False)
    return np.float64((inner.real * inner.real) + (inner.imag * inner.imag))


def density_fidelity(
    rho: npt.ArrayLike,
    sigma: npt.ArrayLike,
    *,
    cfg: QIPConfig = DEFAULT,
) -> np.float64:
    ρ = as_c128(rho, cfg=cfg)
    σ = as_c128(sigma, cfg=cfg)
    if ρ.ndim != 2 or ρ.shape[0] != ρ.shape[1]:
        raise ValueError("rho must be a square matrix")
    if σ.shape != ρ.shape:
        raise ValueError("sigma must match rho shape")

    ρh = 0.5 * (ρ + dag(ρ, cfg=cfg))
    w, v = np.linalg.eigh(ρh)
    w = np.maximum(w, np.float64(0.0))
    sqrt_w = np.sqrt(w, dtype=np.float64)
    sqrt_rho = (v * sqrt_w.astype(np.complex128, copy=False)) @ np.conjugate(v).T

    A = sqrt_rho @ σ @ sqrt_rho
    Ah = 0.5 * (A + dag(A, cfg=cfg))
    wa, _ = np.linalg.eigvalsh(Ah), None
    wa = np.maximum(wa, np.float64(0.0))
    tr_sqrt = np.sum(np.sqrt(wa, dtype=np.float64), dtype=np.float64)
    return np.float64(tr_sqrt * tr_sqrt)
