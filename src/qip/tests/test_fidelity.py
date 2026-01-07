"""
Tests for metrics.fidelity.

We test:
- state fidelity is 1 for identical states
- density fidelity is 1 for identical pure states
"""

import numpy as np

from qip.metrics.fidelity import density_fidelity, state_fidelity
from qip.states.density import density_from_state
from qip.states.state import StateVector


def test_state_fidelity_identity() -> None:
    psi = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    assert np.allclose(state_fidelity(psi, psi), 1.0, atol=1.0e-12, rtol=1.0e-12)


def test_density_fidelity_for_pure_states() -> None:
    psi = StateVector.from_array([1.0 + 0.0j, 0.0 + 0.0j]).normalized()
    rho = density_from_state(psi).rho
    fid = density_fidelity(rho, rho)
    assert np.allclose(fid, 1.0, atol=1.0e-10, rtol=1.0e-10)
