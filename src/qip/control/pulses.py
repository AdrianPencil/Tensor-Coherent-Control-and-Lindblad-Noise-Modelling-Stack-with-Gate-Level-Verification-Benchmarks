"""
control.pulses

Small pulse library with vectorized evaluation.

A pulse is a real-valued envelope a(t) sampled at times t.
Higher layers map envelopes to Hamiltonian control terms.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

__all__ = [
    "Pulse",
    "GaussianPulse",
    "SquarePulse",
    "CosineRiseFallPulse",
    "DRAGPulse",
]


class Pulse(Protocol):
    def amplitude(self, t: npt.ArrayLike) -> npt.NDArray[np.float64]:
        ...


@dataclass(frozen=True, slots=True)
class GaussianPulse:
    """
    a(t) = amp * exp(-0.5 * ((t - t0)/sigma)^2)
    """

    amp: float
    t0: float
    sigma: float

    def amplitude(self, t: npt.ArrayLike) -> npt.NDArray[np.float64]:
        tt = np.ascontiguousarray(np.asarray(t, dtype=np.float64))
        s = float(self.sigma)
        if s <= 0.0:
            raise ValueError("sigma must be positive")
        x = (tt - float(self.t0)) / s
        a = float(self.amp) * np.exp(-0.5 * x * x)
        return np.ascontiguousarray(a.astype(np.float64, copy=False))


@dataclass(frozen=True, slots=True)
class SquarePulse:
    """
    a(t) = amp for t in [t_start, t_end], else 0
    """

    amp: float
    t_start: float
    t_end: float

    def amplitude(self, t: npt.ArrayLike) -> npt.NDArray[np.float64]:
        tt = np.ascontiguousarray(np.asarray(t, dtype=np.float64))
        if float(self.t_end) < float(self.t_start):
            raise ValueError("t_end must be >= t_start")
        mask = (tt >= float(self.t_start)) & (tt <= float(self.t_end))
        a = np.where(mask, float(self.amp), 0.0)
        return np.ascontiguousarray(a.astype(np.float64, copy=False))


@dataclass(frozen=True, slots=True)
class CosineRiseFallPulse:
    """
    Smooth square-ish pulse with cosine ramps.

    a(t) = amp * r(t)
    r(t) ramps 0->1 over [t_start, t_start + tau_rise]
    holds 1 over [t_start + tau_rise, t_end - tau_fall]
    ramps 1->0 over [t_end - tau_fall, t_end]
    """

    amp: float
    t_start: float
    t_end: float
    tau_rise: float
    tau_fall: float

    def amplitude(self, t: npt.ArrayLike) -> npt.NDArray[np.float64]:
        tt = np.ascontiguousarray(np.asarray(t, dtype=np.float64))
        t0 = float(self.t_start)
        t1 = float(self.t_end)
        tr = float(self.tau_rise)
        tf = float(self.tau_fall)

        if t1 < t0:
            raise ValueError("t_end must be >= t_start")
        if tr < 0.0 or tf < 0.0:
            raise ValueError("tau_rise/tau_fall must be >= 0")
        if tr + tf > (t1 - t0) + 1.0e-30:
            raise ValueError("tau_rise + tau_fall must be <= pulse duration")

        r = np.zeros_like(tt, dtype=np.float64)

        if tr > 0.0:
            x = (tt - t0) / tr
            m = (tt >= t0) & (tt < t0 + tr)
            r[m] = 0.5 * (1.0 - np.cos(np.pi * x[m]))

        hold_start = t0 + tr
        hold_end = t1 - tf
        m_hold = (tt >= hold_start) & (tt <= hold_end)
        r[m_hold] = 1.0

        if tf > 0.0:
            x = (t1 - tt) / tf
            m = (tt > hold_end) & (tt <= t1)
            r[m] = 0.5 * (1.0 - np.cos(np.pi * x[m]))

        a = float(self.amp) * r
        return np.ascontiguousarray(a.astype(np.float64, copy=False))


@dataclass(frozen=True, slots=True)
class DRAGPulse:
    """
    DRAG-like envelope pair (I, Q) derived from a Gaussian.

    I(t) = A * g(t)
    Q(t) = beta * d/dt g(t)

    This does not apply any device-specific scaling; it only produces envelopes.
    """

    amp: float
    t0: float
    sigma: float
    beta: float

    def iq(self, t: npt.ArrayLike) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        tt = np.ascontiguousarray(np.asarray(t, dtype=np.float64))
        s = float(self.sigma)
        if s <= 0.0:
            raise ValueError("sigma must be positive")

        x = (tt - float(self.t0)) / s
        g = np.exp(-0.5 * x * x).astype(np.float64, copy=False)

        I = float(self.amp) * g

        dg_dt = (-(x / s) * g).astype(np.float64, copy=False)
        Q = float(self.beta) * dg_dt

        return np.ascontiguousarray(I), np.ascontiguousarray(Q)

    def amplitude(self, t: npt.ArrayLike) -> npt.NDArray[np.float64]:
        I, _ = self.iq(t)
        return I
