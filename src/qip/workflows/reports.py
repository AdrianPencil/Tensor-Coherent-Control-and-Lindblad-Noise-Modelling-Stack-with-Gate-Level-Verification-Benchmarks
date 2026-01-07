"""
workflows.reports

Report builders produce plain-python data structures that can be rendered to:
- console
- markdown
- latex later (docs layer)

We keep one entry point:
- circuit_report_dict
"""

from dataclasses import dataclass

import numpy as np

from qip.workflows.experiments import CircuitExperimentResult

__all__ = [
    "CircuitReport",
    "circuit_report_dict",
]


@dataclass(frozen=True, slots=True)
class CircuitReport:
    """
    Minimal report container for circuit experiments.
    """

    summary: dict[str, float]

    def as_dict(self) -> dict[str, float]:
        """
        Return a JSON-friendly dict of scalar summary values.
        """
        return dict(self.summary)


def circuit_report_dict(result: CircuitExperimentResult) -> CircuitReport:
    """
    Build a minimal scalar-only report from a circuit experiment result.
    """
    fid = result.fidelity
    summary: dict[str, float] = {
        "dim": float(result.final_state.dim),
        "norm": float(np.linalg.norm(result.final_state.vector)),
    }
    if fid is not None:
        summary["fidelity"] = float(fid)
    return CircuitReport(summary=summary)
