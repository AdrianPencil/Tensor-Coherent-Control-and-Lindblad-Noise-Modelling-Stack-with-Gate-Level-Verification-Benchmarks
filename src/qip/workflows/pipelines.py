"""
workflows.pipelines

Pipelines orchestrate multiple steps into one reusable routine.

This file intentionally keeps one concrete pipeline:
- run_circuit_and_score: simulate and compute a fidelity score
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from qip.circuits.circuits import Circuit
from qip.circuits.simulation import CircuitSimulator
from qip.metrics.fidelity import state_fidelity

__all__ = [
    "PipelineResult",
    "run_circuit_and_score",
]


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """
    Standard small return object for early pipelines.
    """

    score: np.float64
    metadata: dict[str, float]


def run_circuit_and_score(
    circuit: Circuit,
    psi_target: npt.ArrayLike,
    *,
    psi0: npt.ArrayLike | None = None,
    simulator: CircuitSimulator | None = None,
) -> PipelineResult:
    """
    Simulate a circuit and score against a target state.

    score = state_fidelity(psi_final, psi_target)
    """
    sim = CircuitSimulator() if simulator is None else simulator
    psi_final = sim.simulate_statevector(circuit, psi0=psi0).vector
    score = state_fidelity(psi_final, psi_target)
    return PipelineResult(score=score, metadata={"n_qubits": float(circuit.n_qubits), "n_gates": float(len(circuit.gates))})
