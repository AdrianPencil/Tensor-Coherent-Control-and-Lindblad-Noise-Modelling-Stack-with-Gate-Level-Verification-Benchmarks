"""
case_studies

Concrete, end-to-end pipelines that stitch together:
Hamiltonian -> noise -> control -> verification.

These are intentionally "brutally concrete" so they can serve as:
- integration tests
- examples
- templates for new devices
"""

__all__ = [
    "ScQubitCaseStudyParams",
    "run_sc_qubit_end_to_end",
]

from qip.case_studies.parameters import ScQubitCaseStudyParams
from qip.case_studies.sc_qubit_end_to_end import run_sc_qubit_end_to_end
