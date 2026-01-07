"""
states.tomography

Minimal state reconstruction helpers.

Primary entry point:
- pauli_expectations_to_density: reconstruct ρ from Pauli expectation values

For n qubits:
  ρ = (1 / 2^n) * Σ_P <P> P
where P runs over the n-qubit Pauli basis including I...I.
"""

from dataclasses import dataclass

import numpy as np

from qip.core.config import DEFAULT, QIPConfig
from qip.operators.pauli import pauli_basis_labels, pauli_string
from qip.states.density import DensityMatrix

__all__ = [
    "PauliTomographyResult",
    "pauli_expectations_to_density",
]


@dataclass(frozen=True, slots=True)
class PauliTomographyResult:
    """
    Container for reconstruction outputs and basic diagnostics.
    """

    density: DensityMatrix
    n_qubits: int
    used_labels: int


def pauli_expectations_to_density(
    expectations: dict[str, float | complex],
    n_qubits: int,
    *,
    cfg: QIPConfig = DEFAULT,
) -> PauliTomographyResult:
    n = int(n_qubits)
    if n < 0:
        raise ValueError("n_qubits must be >= 0")

    labels = pauli_basis_labels(n)
    dim = 1 << n
    rho = np.zeros((dim, dim), dtype=cfg.dtype_complex)

    used = 0
    for lab in labels:
        if lab in expectations:
            e = np.complex128(expectations[lab])
            rho += e * pauli_string(lab, cfg=cfg)
            used += 1

    rho *= np.complex128(1.0 / float(dim))
    dm = DensityMatrix.from_matrix(rho, cfg=cfg)
    return PauliTomographyResult(density=dm, n_qubits=n, used_labels=used)
