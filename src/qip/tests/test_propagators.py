"""
Tests for dynamics.propagators.

We test:
- a simple unitary rotation implements an X-like action up to global phase
- Lindblad propagation keeps trace approximately 1 for small time steps
"""

import numpy as np

from qip.core.linalg import as_c128
from qip.dynamics.hamiltonian import Hamiltonian
from qip.dynamics.lindblad import LindbladModel
from qip.dynamics.propagators import UnitaryPropagator, propagate_density_lindblad_rk4
from qip.metrics.fidelity import state_fidelity
from qip.operators.pauli import pauli


def test_unitary_propagator_x_gate_up_to_phase() -> None:
    X = pauli("X")
    Omega = 2.0e8
    H = Hamiltonian.from_matrix(0.5 * np.complex128(Omega) * X)

    prop = UnitaryPropagator.from_hamiltonian(H)
    dt = float(np.pi / Omega)

    ket0 = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    ket1 = np.array([0.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex128)

    psi = prop.apply_to_state(ket0, dt)
    fid = state_fidelity(psi, ket1)
    assert float(fid) > 1.0 - 1.0e-12


def test_lindblad_trace_is_preserved() -> None:
    X = pauli("X")
    Z = pauli("Z")

    Omega = 5.0e7
    H = Hamiltonian.from_matrix(0.5 * np.complex128(Omega) * X)

    L_relax = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
    L_deph = Z

    model = LindbladModel.from_operators(H=H, collapse_ops=[L_relax, L_deph], rates=[1.0e5, 2.0e5])

    rho0 = np.zeros((2, 2), dtype=np.complex128)
    rho0[0, 0] = np.complex128(1.0 + 0.0j)

    rho1 = propagate_density_lindblad_rk4(model=model, rho0=rho0, t0=0.0, t1=2.0e-8, n_steps=200)
    tr = np.trace(as_c128(rho1))
    assert np.allclose(tr, np.complex128(1.0 + 0.0j), atol=1.0e-9, rtol=1.0e-9)
