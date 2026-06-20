"""Fried parameter (r0) estimation from reconstructed wavefronts / slopes.

Two independent estimators, reported together as a cross-check:

  * Zernike-variance fit (primary). For Kolmogorov turbulence the variance of
    the j-th Noll Zernike coefficient is c_j (D/r0)^(5/3), where c_j is the
    per-mode coefficient derived from Noll's (1976) residual table
    (c_j = Delta_{j-1} - Delta_j). Fitting a band of mid-order modes (skipping
    tip/tilt, which the finite outer scale suppresses, and the high orders that
    measurement noise dominates) gives (D/r0)^(5/3), hence r0.

  * Slope-variance (cross-check). The angle-of-arrival variance over a
    sub-aperture of diameter d is sigma^2 = 0.162 lambda^2 r0^(-5/3) d^(-1/3)
    (Saint-Jacques 1998); inverting gives r0 directly from raw slopes, with no
    reconstruction.
"""

from __future__ import annotations

import numpy as np

from ..config import SimConfig

# Noll (1976) residual wavefront variance after J terms, in units of (D/r0)^(5/3).
_NOLL_RESIDUAL = [
    None, 1.0299, 0.582, 0.134, 0.111, 0.0880, 0.0648, 0.0587, 0.0525, 0.0463,
    0.0401, 0.0377, 0.0352, 0.0328, 0.0304, 0.0279, 0.0267, 0.0255, 0.0243,
    0.0232, 0.0220, 0.0208,
]


def noll_mode_variance_coeff(j: int) -> float:
    """Per-mode Kolmogorov coefficient c_j so that <a_j^2> = c_j (D/r0)^(5/3)."""
    if j < 2 or j >= len(_NOLL_RESIDUAL):
        raise ValueError(f"mode coefficient tabulated only for 2 <= j <= {len(_NOLL_RESIDUAL)-1}")
    return _NOLL_RESIDUAL[j - 1] - _NOLL_RESIDUAL[j]


def r0_from_zernike_variance(
    coeffs: np.ndarray, cfg: SimConfig, band: tuple[int, int] = (4, 15)
) -> float:
    """r0 [m] from the variance of reconstructed Zernike coefficients.

    ``coeffs`` is (n_frames, n_modes) for Noll modes j = 2, 3, ... (piston
    excluded, as produced by :class:`~aowfs.recon.ZernikeReconstructor`). The
    band is an inclusive Noll-index range of mid-order modes.
    """
    var_j = coeffs.var(axis=0)
    j_lo, j_hi = band
    measured = sum(var_j[j - 2] for j in range(j_lo, j_hi + 1))
    theory = sum(noll_mode_variance_coeff(j) for j in range(j_lo, j_hi + 1))
    dr0_53 = measured / theory  # (D/r0)^(5/3)
    return cfg.pupil.diameter_m / dr0_53 ** (3.0 / 5.0)


def r0_from_slope_variance(gradients: np.ndarray, cfg: SimConfig) -> float:
    """r0 [m] from the angle-of-arrival variance of raw slopes (cross-check).

    ``gradients`` is (n_frames, n_valid_subaps, 2) wavefront gradients [rad/m].
    """
    lam = cfg.pupil.wavelength_m
    d = cfg.subap_pitch_m
    alpha = gradients * (lam / (2.0 * np.pi))  # angle of arrival [rad]
    # Variance over time, per sub-aperture and axis, then averaged.
    slope_var = alpha.var(axis=0).mean()
    return ((0.162 * lam ** 2 * d ** (-1.0 / 3.0)) / slope_var) ** (3.0 / 5.0)
