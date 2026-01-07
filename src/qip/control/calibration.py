"""
control.calibration

Calibration parameters for mapping abstract controls to device-native units.

This is intentionally minimal:
- store a few common scalars
- provide JSON load/save helpers to support case studies
"""

from dataclasses import dataclass
from typing import Any

import json
import numpy as np

__all__ = [
    "Calibration",
    "calibration_from_json",
    "calibration_to_json",
]


@dataclass(frozen=True, slots=True)
class Calibration:
    """
    Typical qubit control calibration parameters.

    omega_01: qubit frequency (rad/s)
    drive_scale: maps envelope amplitude to rad/s Rabi rate
    phase_offset: global phase offset (rad)
    """

    omega_01: np.float64
    drive_scale: np.float64
    phase_offset: np.float64 = np.float64(0.0)

    def as_dict(self) -> dict[str, float]:
        return {
            "omega_01": float(self.omega_01),
            "drive_scale": float(self.drive_scale),
            "phase_offset": float(self.phase_offset),
        }


def calibration_from_json(path: str) -> Calibration:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise ValueError("calibration JSON must be an object")

    omega_01 = np.float64(obj["omega_01"])
    drive_scale = np.float64(obj["drive_scale"])
    phase_offset = np.float64(obj.get("phase_offset", 0.0))

    if omega_01 < 0.0:
        raise ValueError("omega_01 must be >= 0")
    if drive_scale <= 0.0:
        raise ValueError("drive_scale must be > 0")

    return Calibration(omega_01=omega_01, drive_scale=drive_scale, phase_offset=phase_offset)


def calibration_to_json(cal: Calibration, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cal.as_dict(), f, indent=2, sort_keys=True)
