"""
Tests for noise.channels.

Core property checks:
- trace preservation for common channels
- positivity is not exhaustively tested here, but these basic constructions
  should behave as expected.
"""

import numpy as np

from qip.noise.channels import amplitude_damping, depolarizing, phase_flip


def test_phase_flip_trace_preserving() -> None:
    rho = np.array([[0.7, 0.2 + 0.1j], [0.2 - 0.1j, 0.3]], dtype=np.complex128)
    ch = phase_flip(0.25)
    out = ch.apply(rho)
    assert np.allclose(np.trace(out), np.trace(rho), atol=1.0e-12, rtol=1.0e-12)


def test_depolarizing_trace_preserving() -> None:
    rho = np.array([[0.55, 0.1j], [-0.1j, 0.45]], dtype=np.complex128)
    ch = depolarizing(0.9)
    out = ch.apply(rho)
    assert np.allclose(np.trace(out), np.trace(rho), atol=1.0e-12, rtol=1.0e-12)


def test_amplitude_damping_trace_preserving() -> None:
    rho = np.array([[0.2, 0.0], [0.0, 0.8]], dtype=np.complex128)
    ch = amplitude_damping(0.4)
    out = ch.apply(rho)
    assert np.allclose(np.trace(out), np.trace(rho), atol=1.0e-12, rtol=1.0e-12)
