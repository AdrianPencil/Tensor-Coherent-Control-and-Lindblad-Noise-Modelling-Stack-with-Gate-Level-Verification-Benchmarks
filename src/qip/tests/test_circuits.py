"""
Tests for circuits layer.

We test:
- single-qubit X maps |0> -> |1>
- a small 2-qubit Bell-state circuit produces the expected statevector
"""

import numpy as np

from qip.circuits.circuits import Circuit
from qip.circuits.gates import CNOT, H, X
from qip.circuits.simulation import CircuitSimulator
from qip.metrics.fidelity import state_fidelity


def test_single_qubit_x() -> None:
    c = Circuit.empty(1).append(X(0))
    sim = CircuitSimulator()
    psi = sim.simulate_statevector(c).vector
    ket1 = np.array([0.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex128)
    assert float(state_fidelity(psi, ket1)) > 1.0 - 1.0e-12


def test_bell_state() -> None:
    c = Circuit.empty(2).append(H(0)).append(CNOT(0, 1))
    sim = CircuitSimulator()
    psi = sim.simulate_statevector(c).vector

    bell = np.zeros((4,), dtype=np.complex128)
    bell[0] = 1.0 / np.sqrt(2.0)
    bell[3] = 1.0 / np.sqrt(2.0)

    assert float(state_fidelity(psi, bell)) > 1.0 - 1.0e-12
