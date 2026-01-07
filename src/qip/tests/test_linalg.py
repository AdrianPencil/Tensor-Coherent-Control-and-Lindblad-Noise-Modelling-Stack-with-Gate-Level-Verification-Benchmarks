"""
Tests for core.linalg.

These are designed to be small and deterministic:
- dtype/contiguity conversions
- dagger and kron sanity checks
- expm_hermitian produces a unitary for Hermitian input
"""

import numpy as np

from qip.core.linalg import as_c128, dag, expm_hermitian, kron


def test_as_c128_is_contiguous() -> None:
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64, order="F")
    b = as_c128(a)
    assert b.dtype == np.complex128
    assert b.flags.c_contiguous
    assert b.shape == (2, 2)


def test_dag_involution() -> None:
    a = np.array([[1.0 + 2.0j, 3.0 - 1.0j], [0.0 + 0.5j, -2.0 + 0.0j]], dtype=np.complex128)
    d = dag(a)
    dd = dag(d)
    assert np.allclose(dd, a)


def test_kron_shape() -> None:
    a = np.eye(2, dtype=np.complex128)
    b = np.eye(3, dtype=np.complex128)
    k = kron(a, b)
    assert k.shape == (6, 6)
    assert k.flags.c_contiguous


def test_expm_hermitian_is_unitary() -> None:
    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    H = 0.25 * X
    U = expm_hermitian(H, t=0.7)
    I = np.eye(2, dtype=np.complex128)
    assert np.allclose(np.conjugate(U).T @ U, I, atol=1.0e-12, rtol=1.0e-12)
