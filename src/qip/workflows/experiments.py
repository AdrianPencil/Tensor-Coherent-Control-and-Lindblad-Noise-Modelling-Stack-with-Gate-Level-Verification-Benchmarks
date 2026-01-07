"""
workflows.experiments

Experiment definitions are thin "callable bundles" that run a simulation and
return structured outputs suitable for reports.

This keeps workflows reusable and testable.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from qip.circuits.circuits import Circuit
from qip.circuits.simulation import CircuitSimulator
from qip.metrics.fidelity import state_fidelity
from qip.states.state import StateVector

__all__ = [
    "CircuitExperimentResult",
    "RunCircuitExperiment",
]


@dataclass(frozen=True, slots=True)
class CircuitExperimentResult:
    """
    Output of running a circuit experiment.

    fidelity is optional and only filled if a target state is provided.
    """

    final_state: StateVector
    fidelity: np.float64 | None


@dataclass(frozen=True, slots=True)
class RunCircuitExperiment:
    """
    A simple experiment: run a circuit and optionally compare to a target state.
    """

    simulator: CircuitSimulator

    def run(
        self,
        circuit: Circuit,
        psi0: npt.ArrayLike | None = None,
        psi_target: npt.ArrayLike | None = None,
    ) -> CircuitExperimentResult:
        """
        Run the circuit simulation.

        If psi_target is provided, computes |<psi_target|psi_final>|^2.
        """
        final_state = self.simulator.simulate_statevector(circuit, psi0=psi0)
        if psi_target is None:
            return CircuitExperimentResult(final_state=final_state, fidelity=None)
        fid = state_fidelity(final_state.vector, psi_target)
        return CircuitExperimentResult(final_state=final_state, fidelity=fid)
