"""
case_studies.sc_qubit_end_to_end

Superconducting qubit end-to-end pipeline:
- load calibration + measured PSD
- fit a simple PSD model
- derive effective rates (gamma_phi, gamma_1)
- build a Lindblad model
- simulate a short driven evolution intended to realize an X gate
- verify via fidelity against the ideal target

This is a scaffold:
- it is deterministic, typed, and testable
- it is easy to replace the physics proxies with richer models later
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from qip.case_studies.datasets import load_sc_qubit_dataset
from qip.case_studies.parameters import ScQubitCaseStudyParams
from qip.control.calibration import Calibration
from qip.dynamics.hamiltonian import Hamiltonian
from qip.dynamics.lindblad import LindbladModel
from qip.dynamics.propagators import propagate_density_lindblad_rk4
from qip.metrics.fidelity import density_fidelity
from qip.noise.identification import fit_one_over_f_plus_white
from qip.noise.spectra import one_over_f_psd, white_psd
from qip.noise.models import dephasing_rate_from_psd, relaxation_rate_from_psd
from qip.operators.pauli import pauli
from qip.states.density import DensityMatrix

__all__ = [
    "ScQubitEndToEndResult",
    "run_sc_qubit_end_to_end",
]


@dataclass(frozen=True, slots=True)
class ScQubitEndToEndResult:
    """
    Outputs of the end-to-end superconducting qubit case study.

    summary
      Scalar results suitable for console / JSON / tables.
    rho_final
      Final density matrix after the simulated sequence.
    rho_target
      Ideal density matrix for the intended gate applied to |0>.
    """

    summary: dict[str, float]
    rho_final: DensityMatrix
    rho_target: DensityMatrix


def _target_density_for_gate(gate: str) -> DensityMatrix:
    """
    Construct the ideal target state for a named 1-qubit gate applied to |0>.

    Currently supported: "X", "I"
    """
    g = str(gate).upper().strip()
    if g not in {"X", "I"}:
        raise ValueError("target_gate must be one of: X, I")

    ket0 = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    if g == "I":
        psi = ket0
    else:
        X = pauli("X")
        psi = X @ ket0

    rho = psi.reshape(2, 1) @ np.conjugate(psi).reshape(1, 2)
    return DensityMatrix.from_matrix(np.ascontiguousarray(rho))


def run_sc_qubit_end_to_end(params: ScQubitCaseStudyParams) -> ScQubitEndToEndResult:
    """
    Execute the full superconducting-qubit pipeline.

    The simulation is performed in an effective rotating frame:
    - drift is set to 0 (we focus on the driven rotation)
    - drive Hamiltonian uses X with an effective Rabi rate Ω
    """
    ds = load_sc_qubit_dataset(params.data_dir)

    cal = Calibration(
        omega_01=np.float64(ds.calibration.get("omega_01", 0.0)),
        drive_scale=np.float64(ds.calibration.get("drive_scale", 1.0)),
        phase_offset=np.float64(ds.calibration.get("phase_offset", 0.0)),
    )

    fit = fit_one_over_f_plus_white(ds.omega, ds.psd, float(params.omega_ref))
    psd_model = one_over_f_psd(float(fit.A), float(fit.alpha), float(fit.omega_ref))
    psd_white = white_psd(float(fit.S0))

    def _combined_psd(omega: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.ascontiguousarray(psd_model(omega) + psd_white(omega))

    from qip.noise.spectra import PSDModel

    psd = PSDModel(eval=_combined_psd)

    gamma_phi = dephasing_rate_from_psd(psd, float(params.omega_ir))
    gamma_1 = relaxation_rate_from_psd(psd, float(cal.omega_01), float(params.coupling))

    X = pauli("X")
    Z = pauli("Z")

    L_relax = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
    L_deph = Z

    H_drift = np.zeros((2, 2), dtype=np.complex128)

    Omega = float(cal.drive_scale)
    if Omega <= 0.0:
        raise ValueError("drive_scale must be > 0")

    H_drive = 0.5 * np.complex128(Omega) * X
    H = Hamiltonian.from_matrix(H_drift + H_drive, check=True)

    model = LindbladModel.from_operators(
        H=H,
        collapse_ops=[L_relax, L_deph],
        rates=[float(gamma_1), float(gamma_phi)],
    )

    rho0 = np.zeros((2, 2), dtype=np.complex128)
    rho0[0, 0] = np.complex128(1.0 + 0.0j)

    T_gate = float(np.pi / Omega)
    n_steps = int(params.n_steps)
    dt = float(params.dt)
    T_sim = dt * float(n_steps)

    if T_sim <= 0.0:
        raise ValueError("dt*n_steps must be positive")

    rho_final = propagate_density_lindblad_rk4(
        model=model,
        rho0=rho0,
        t0=0.0,
        t1=min(T_gate, T_sim),
        n_steps=n_steps,
    )

    rho_target = _target_density_for_gate(params.target_gate).rho
    fid = density_fidelity(rho_final, rho_target)

    summary = {
        "omega_01": float(cal.omega_01),
        "drive_scale": float(cal.drive_scale),
        "fit_A": float(fit.A),
        "fit_alpha": float(fit.alpha),
        "fit_S0": float(fit.S0),
        "fit_rms": float(fit.rms),
        "gamma_phi": float(gamma_phi),
        "gamma_1": float(gamma_1),
        "T_gate": float(T_gate),
        "T_sim": float(T_sim),
        "density_fidelity": float(fid),
    }

    return ScQubitEndToEndResult(
        summary=summary,
        rho_final=DensityMatrix.from_matrix(rho_final),
        rho_target=DensityMatrix.from_matrix(rho_target),
    )
