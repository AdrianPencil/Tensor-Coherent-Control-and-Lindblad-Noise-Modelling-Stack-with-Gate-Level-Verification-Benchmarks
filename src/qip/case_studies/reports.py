"""
case_studies.reports

Reporting helpers specific to case studies.

We keep one report builder:
- sc_qubit_report_dict: returns scalar summaries and optionally a PSD fit curve
  that can be plotted by higher layers.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from qip.case_studies.sc_qubit_end_to_end import ScQubitEndToEndResult
from qip.noise.spectra import one_over_f_psd, white_psd

__all__ = [
    "ScQubitReport",
    "sc_qubit_report",
]


@dataclass(frozen=True, slots=True)
class ScQubitReport:
    """
    Case study report object.

    summary
      Scalar values appropriate for tables and regression tests.
    psd_curve
      Optional dense curve for visualization: (omega_grid, psd_fit)
    """

    summary: dict[str, float]
    psd_curve: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None

    def as_dict(self) -> dict[str, float]:
        """
        Return JSON-friendly scalar report.
        """
        return dict(self.summary)


def sc_qubit_report(
    result: ScQubitEndToEndResult,
    omega_grid: npt.ArrayLike | None = None,
) -> ScQubitReport:
    """
    Build a report from a ScQubitEndToEndResult.

    If omega_grid is provided, also returns a fitted PSD curve on that grid.
    """
    summary = dict(result.summary)

    if omega_grid is None:
        return ScQubitReport(summary=summary, psd_curve=None)

    w = np.ascontiguousarray(np.asarray(omega_grid, dtype=np.float64).reshape(-1))
    A = float(summary["fit_A"])
    alpha = float(summary["fit_alpha"])
    omega_ref = float(summary.get("omega_ref", summary.get("fit_omega_ref", 2.0 * np.pi * 1.0e6)))
    S0 = float(summary["fit_S0"])

    psd1 = one_over_f_psd(A, alpha, omega_ref)
    psd2 = white_psd(S0)
    y = np.ascontiguousarray(psd1(w) + psd2(w))

    return ScQubitReport(summary=summary, psd_curve=(w, y))
