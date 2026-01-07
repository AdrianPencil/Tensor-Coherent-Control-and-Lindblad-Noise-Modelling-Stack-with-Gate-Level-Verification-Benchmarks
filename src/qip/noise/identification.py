"""
noise.identification

Simple parameter identification utilities for PSD curves.

We keep one concrete fitter:
  S(ω) = A * (|ω|/ω_ref)^(-alpha) + S0

Fit strategy:
- grid search alpha over a small range
- for each alpha, solve least squares for (A, S0) linearly
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = [
    "OneOverFPlusWhiteFit",
    "fit_one_over_f_plus_white",
]


@dataclass(frozen=True, slots=True)
class OneOverFPlusWhiteFit:
    """
    Fit result for S(ω) = A*(|ω|/ω_ref)^(-alpha) + S0.
    """

    A: np.float64
    alpha: np.float64
    S0: np.float64
    omega_ref: np.float64
    rms: np.float64


def fit_one_over_f_plus_white(
    omega: npt.ArrayLike,
    S: npt.ArrayLike,
    omega_ref: float,
    *,
    alpha_grid: npt.ArrayLike | None = None,
) -> OneOverFPlusWhiteFit:
    w = np.asarray(omega, dtype=np.float64).reshape(-1)
    y = np.asarray(S, dtype=np.float64).reshape(-1)
    if w.size != y.size:
        raise ValueError("omega and S must have the same length")
    if w.size < 3:
        raise ValueError("need at least 3 points to fit")

    w0 = np.float64(omega_ref)
    if w0 <= 0.0:
        raise ValueError("omega_ref must be positive")

    a = np.abs(w) / float(w0)
    a = np.maximum(a, np.float64(1.0e-30))

    if alpha_grid is None:
        alpha_grid = np.linspace(0.0, 2.0, 161, dtype=np.float64)

    alphas = np.asarray(alpha_grid, dtype=np.float64).reshape(-1)
    if alphas.size == 0:
        raise ValueError("alpha_grid must be non-empty")

    best_rms = np.float64(np.inf)
    best_A = np.float64(0.0)
    best_alpha = np.float64(0.0)
    best_S0 = np.float64(0.0)

    ones = np.ones_like(a, dtype=np.float64)
    for alpha in alphas:
        x1 = np.power(a, -alpha)
        X = np.stack([x1, ones], axis=1)
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        A_hat = np.float64(beta[0])
        S0_hat = np.float64(beta[1])
        y_hat = A_hat * x1 + S0_hat
        err = y - y_hat
        rms = np.float64(np.sqrt(np.mean(err * err)))
        if rms < best_rms:
            best_rms = rms
            best_A = A_hat
            best_alpha = np.float64(alpha)
            best_S0 = S0_hat

    return OneOverFPlusWhiteFit(
        A=best_A,
        alpha=best_alpha,
        S0=best_S0,
        omega_ref=w0,
        rms=best_rms,
    )
