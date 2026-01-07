"""
noise.models

Minimal links from spectra -> effective rates.

These are intentionally "thin" because real devices require careful derivations.
Still, having stable, typed utilities keeps case studies consistent.
"""

from dataclasses import dataclass

import numpy as np

from qip.noise.spectra import PSDModel

__all__ = [
    "NoiseRates",
    "dephasing_rate_from_psd",
    "relaxation_rate_from_psd",
]


@dataclass(frozen=True, slots=True)
class NoiseRates:
    """
    Common effective rates for a two-level system.
    """

    gamma_phi: np.float64
    gamma_1: np.float64


def dephasing_rate_from_psd(psd: PSDModel, omega_ir: float) -> np.float64:
    """
    A pragmatic dephasing proxy using low-frequency PSD:

    gamma_phi ≈ 0.5 * S(ω_ir)

    omega_ir is a small infrared frequency (rad/s) used to avoid ω=0 ambiguity.
    """
    w = np.ascontiguousarray(np.asarray([float(omega_ir)], dtype=np.float64))
    if w[0] < 0.0:
        raise ValueError("omega_ir must be >= 0")
    S = psd(w)[0]
    return np.float64(0.5 * S)


def relaxation_rate_from_psd(psd: PSDModel, omega_01: float, coupling: float) -> np.float64:
    """
    A pragmatic T1 proxy:

    gamma_1 ≈ coupling^2 * S(ω_01)

    coupling absorbs matrix elements / device coupling constants.
    """
    w01 = float(omega_01)
    if w01 < 0.0:
        raise ValueError("omega_01 must be >= 0")
    c = np.float64(coupling)
    w = np.ascontiguousarray(np.asarray([w01], dtype=np.float64))
    S = psd(w)[0]
    return np.float64((c * c) * S)
