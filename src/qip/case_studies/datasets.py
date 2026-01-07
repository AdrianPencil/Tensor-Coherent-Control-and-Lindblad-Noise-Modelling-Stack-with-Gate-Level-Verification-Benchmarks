"""
case_studies.datasets

Loaders for measured-ish inputs used by the case study.

We keep it simple:
- calibration.json -> dict[str, float]
- noise_psd.csv -> (omega_rad_s, S_omega)

The CSV is expected to have 2 columns:
  omega_rad_s, psd_value
with a header allowed (commented or first line text).
"""

from dataclasses import dataclass
import json
import os

import numpy as np
import numpy.typing as npt

__all__ = [
    "ScQubitDataset",
    "load_sc_qubit_dataset",
]


@dataclass(frozen=True, slots=True)
class ScQubitDataset:
    """
    Dataset bundle for the superconducting qubit case study.

    calibration
      Raw dict containing at least omega_01 and drive_scale.
    omega
      Angular frequency grid (rad/s).
    psd
      PSD values on that grid.
    """

    calibration: dict[str, float]
    omega: npt.NDArray[np.float64]
    psd: npt.NDArray[np.float64]


def load_sc_qubit_dataset(data_dir: str) -> ScQubitDataset:
    """
    Load calibration and PSD from the case-study data directory.
    """
    d = str(data_dir)
    cal_path = os.path.join(d, "calibration.json")
    psd_path = os.path.join(d, "noise_psd.csv")

    with open(cal_path, "r", encoding="utf-8") as f:
        cal_obj = json.load(f)
    if not isinstance(cal_obj, dict):
        raise ValueError("calibration.json must contain an object")

    calibration = {str(k): float(v) for k, v in cal_obj.items()}

    try:
        raw = np.loadtxt(psd_path, delimiter=",", dtype=np.float64, ndmin=2)
    except ValueError:
        raw = np.loadtxt(psd_path, delimiter=",", dtype=np.float64, ndmin=2, skiprows=1)

    if raw.shape[1] < 2:
        raise ValueError("noise_psd.csv must have at least 2 columns: omega, psd")

    omega = np.ascontiguousarray(raw[:, 0].reshape(-1))
    psd = np.ascontiguousarray(raw[:, 1].reshape(-1))

    if omega.size < 3:
        raise ValueError("noise_psd.csv must contain at least 3 samples")
    if np.any(omega < 0.0):
        raise ValueError("omega must be >= 0")
    if np.any(psd < 0.0):
        raise ValueError("psd must be >= 0")

    return ScQubitDataset(calibration=calibration, omega=omega, psd=psd)
