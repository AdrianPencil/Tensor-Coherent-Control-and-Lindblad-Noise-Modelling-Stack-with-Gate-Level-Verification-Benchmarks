"""
operators.pauli

Pauli operator library + tensor products for Pauli strings.

Conventions:
- All matrices are complex128, C-contiguous.
- Labels: "I", "X", "Y", "Z"
- Pauli string: e.g. "IXZ" meaning I ⊗ X ⊗ Z (left-to-right)
"""

from typing import Final, Mapping

import numpy as np
import numpy.typing as npt

from qip.core.config import DEFAULT, QIPConfig
from qip.core.linalg import as_c128, kron

__all__ = [
    "PAULI",
    "pauli",
    "pauli_string",
    "pauli_basis_labels",
]

_I: Final[npt.NDArray[np.complex128]] = np.ascontiguousarray(
    np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
)
_X: Final[npt.NDArray[np.complex128]] = np.ascontiguousarray(
    np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
)
_Y: Final[npt.NDArray[np.complex128]] = np.ascontiguousarray(
    np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
)
_Z: Final[npt.NDArray[np.complex128]] = np.ascontiguousarray(
    np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
)

PAULI: Final[Mapping[str, npt.NDArray[np.complex128]]] = {
    "I": _I,
    "X": _X,
    "Y": _Y,
    "Z": _Z,
}


def pauli(label: str, *, cfg: QIPConfig = DEFAULT) -> npt.NDArray[np.complex128]:
    """
    Return the 2x2 Pauli matrix for label in {"I","X","Y","Z"}.
    """
    s = str(label).upper()
    if s not in PAULI:
        raise ValueError(f"unknown Pauli label: {label!r}")
    return as_c128(PAULI[s], cfg=cfg)


def pauli_string(labels: str, *, cfg: QIPConfig = DEFAULT) -> npt.NDArray[np.complex128]:
    """
    Tensor-product Pauli for a string like "IXZ".

    For empty string, returns 1x1 identity.
    """
    s = str(labels).upper().strip()
    if s == "":
        return np.ascontiguousarray(np.array([[1.0 + 0.0j]], dtype=cfg.dtype_complex))
    out = pauli(s[0], cfg=cfg)
    for ch in s[1:]:
        out = kron(out, pauli(ch, cfg=cfg), cfg=cfg)
    return out


def pauli_basis_labels(n_qubits: int) -> list[str]:
    """
    Return labels for the full n-qubit Pauli basis, size 4^n.

    Ordering is lexicographic over ("I","X","Y","Z") for each qubit.
    """
    n = int(n_qubits)
    if n < 0:
        raise ValueError("n_qubits must be >= 0")
    if n == 0:
        return [""]
    alphabet = ["I", "X", "Y", "Z"]
    labels = [""]
    for _ in range(n):
        labels = [p + a for p in labels for a in alphabet]
    return labels
