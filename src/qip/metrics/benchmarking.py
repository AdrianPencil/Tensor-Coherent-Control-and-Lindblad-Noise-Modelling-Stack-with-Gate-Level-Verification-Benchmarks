"""
metrics.benchmarking

Small benchmarking metrics used in workflows.

We include:
- average_gate_infidelity from process fidelity proxy (unitary vs unitary)
- depolarizing_parameter_from_infidelity for 1-qubit quick checks
"""

import numpy as np
import numpy.typing as npt

from qip.core.config import DEFAULT, QIPConfig
from qip.core.linalg import as_c128, dag, trace

__all__ = [
    "unitary_process_fidelity",
    "average_gate_infidelity",
    "depolarizing_parameter_from_infidelity",
]


def unitary_process_fidelity(
    U_target: npt.ArrayLike,
    U_actual: npt.ArrayLike,
    *,
    cfg: QIPConfig = DEFAULT,
) -> np.float64:
    U = as_c128(U_target, cfg=cfg)
    V = as_c128(U_actual, cfg=cfg)
    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise ValueError("U_target must be a square matrix")
    if V.shape != U.shape:
        raise ValueError("U_actual must match U_target shape")

    d = U.shape[0]
    M = dag(U, cfg=cfg) @ V
    tr = trace(M, cfg=cfg)
    val = (tr.real * tr.real) + (tr.imag * tr.imag)
    return np.float64(val / (float(d) * float(d)))


def average_gate_infidelity(
    U_target: npt.ArrayLike,
    U_actual: npt.ArrayLike,
    *,
    cfg: QIPConfig = DEFAULT,
) -> np.float64:
    d = int(as_c128(U_target, cfg=cfg).shape[0])
    Fp = unitary_process_fidelity(U_target, U_actual, cfg=cfg)
    Favg = (float(d) * float(Fp) + 1.0) / (float(d) + 1.0)
    return np.float64(1.0 - Favg)


def depolarizing_parameter_from_infidelity(r: float, dim: int) -> np.float64:
    """
    For a depolarizing channel in dimension d:
      r = 1 - F_avg
      p ≈ r * (d + 1) / d
    """
    d = float(int(dim))
    if d <= 0.0:
        raise ValueError("dim must be positive")
    rr = float(r)
    if rr < 0.0:
        raise ValueError("r must be >= 0")
    return np.float64(rr * (d + 1.0) / d)
