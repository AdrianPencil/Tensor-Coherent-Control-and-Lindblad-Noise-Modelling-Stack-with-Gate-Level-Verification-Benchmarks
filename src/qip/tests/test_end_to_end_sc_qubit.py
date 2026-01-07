"""
Integration test for the superconducting-qubit end-to-end case study.

We generate a temporary dataset and run the full pipeline.
The goal is a stable regression check, not a high-accuracy physics validation.
"""

import json
import numpy as np

from qip.case_studies.parameters import ScQubitCaseStudyParams
from qip.case_studies.sc_qubit_end_to_end import run_sc_qubit_end_to_end


def test_end_to_end_sc_qubit(tmp_path) -> None:
    data_dir = tmp_path / "superconducting_qubit"
    data_dir.mkdir(parents=True, exist_ok=True)

    cal = {
        "omega_01": 2.0 * np.pi * 5.0e9,
        "drive_scale": 2.0 * np.pi * 1.0e8,
        "phase_offset": 0.0,
    }
    (data_dir / "calibration.json").write_text(json.dumps(cal, indent=2), encoding="utf-8")

    omega_ref = 2.0 * np.pi * 1.0e6
    A = 1.0e-12
    alpha = 1.0
    S0 = 1.0e-15
    omega = 2.0 * np.pi * np.array([1.0, 10.0, 100.0, 1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e7], dtype=np.float64)
    psd = A * (np.abs(omega) / omega_ref) ** (-alpha) + S0

    lines = ["omega_rad_s,psd_value"]
    for w, s in zip(omega.tolist(), psd.tolist(), strict=True):
        lines.append(f"{w:.16e},{s:.16e}")
    (data_dir / "noise_psd.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    params = ScQubitCaseStudyParams.default(data_dir=str(data_dir), n_steps=400, dt=1.0e-10)
    out = run_sc_qubit_end_to_end(params)

    assert "density_fidelity" in out.summary
    fid = float(out.summary["density_fidelity"])
    assert 0.0 <= fid <= 1.0
