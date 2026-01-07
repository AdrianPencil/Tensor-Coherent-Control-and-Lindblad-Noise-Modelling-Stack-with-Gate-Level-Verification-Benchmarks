"""
case_studies.parameters

One place for case study parameters.

These parameters intentionally represent "the pipeline choices":
- where data lives
- how we interpret it (omega_ref, omega_ir)
- how we run a short dynamical simulation to validate control + noise
"""

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ScQubitCaseStudyParams",
]


@dataclass(frozen=True, slots=True)
class ScQubitCaseStudyParams:
    """
    End-to-end superconducting qubit case study configuration.

    data_dir
      Directory containing:
      - calibration.json
      - noise_psd.csv

    omega_ref
      Reference ω for the 1/f^alpha fit model.

    omega_ir
      Small frequency used to probe low-frequency PSD for a dephasing proxy.

    coupling
      Scalar coupling used in gamma_1 ≈ coupling^2 * S(ω_01).

    dt, n_steps
      Fixed-step integration settings for a short Lindblad evolution.
    """

    data_dir: str

    omega_ref: np.float64
    omega_ir: np.float64
    coupling: np.float64

    dt: np.float64
    n_steps: int

    target_gate: str

    @staticmethod
    def default(data_dir: str, n_steps: int = 2000, dt: float = 1.0e-10) -> "ScQubitCaseStudyParams":
        """
        Provide conservative defaults that run quickly.

        The physical meaning is intentionally approximate; the case study is a
        scaffold that is easy to refine as you attach real device models.
        """
        return ScQubitCaseStudyParams(
            data_dir=str(data_dir),
            omega_ref=np.float64(2.0 * np.pi * 1.0e6),
            omega_ir=np.float64(2.0 * np.pi * 1.0),
            coupling=np.float64(1.0),
            dt=np.float64(dt),
            n_steps=int(n_steps),
            target_gate="X",
        )
