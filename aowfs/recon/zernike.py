"""Self-contained Zernike polynomials (Noll-ordered, Noll-normalised).

We do not use aotools' Zernike helpers: their radial function calls the removed
``numpy.math.factorial`` and crashes on numpy >= 2. Implementing the basis here
also lets us evaluate the polynomials *unmasked* (i.e. continued beyond the unit
disk), which is required so that the mean Zernike gradient over a pupil-edge
sub-aperture is exact and consistent with the SH forward model.

Conventions:
  * Noll (1976) single index j (j=1 piston, 2 tip, 3 tilt, 4 focus, ...).
  * Normalisation such that <Z_j^2> over the unit disk = 1 (Noll), so a Zernike
    coefficient is directly the RMS wavefront [rad] contributed by that mode and
    the residual after J modes follows Noll's Delta_J tabulation.
"""

from __future__ import annotations

import math

import numpy as np


def noll_to_nm(j: int) -> tuple[int, int]:
    """Map a Noll index j (>=1) to radial/azimuthal orders (n, m)."""
    if j < 1:
        raise ValueError("Noll index j must be >= 1")
    n = 0
    j1 = j - 1
    while j1 > n:
        n += 1
        j1 -= n
    m = (-1) ** j * ((n % 2) + 2 * int((j1 + ((n + 1) % 2)) / 2.0))
    return n, m


def _radial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    """Zernike radial polynomial R_n^m(rho) (valid for rho > 1 too)."""
    m = abs(m)
    R = np.zeros_like(rho)
    for k in range((n - m) // 2 + 1):
        c = (
            (-1) ** k
            * math.factorial(n - k)
            / (
                math.factorial(k)
                * math.factorial((n + m) // 2 - k)
                * math.factorial((n - m) // 2 - k)
            )
        )
        R = R + c * rho ** (n - 2 * k)
    return R


def zernike_nm(n: int, m: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Noll-normalised Zernike Z_n^m on polar coordinates (rho, theta)."""
    norm = math.sqrt(n + 1) if m == 0 else math.sqrt(2 * (n + 1))
    rad = _radial(n, m, rho)
    if m == 0:
        ang = np.ones_like(theta)
    elif m > 0:
        ang = np.cos(m * theta)
    else:
        ang = np.sin(-m * theta)
    return norm * rad * ang


def zernike_basis(
    n_modes: int, grid_px: int, pupil_radius_px: float, start_j: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Stack of ``n_modes`` Zernikes on a square grid + the pupil mask.

    Modes run over Noll indices ``start_j .. start_j + n_modes - 1`` (default
    skips piston, which slopes cannot sense). Polynomials are evaluated
    everywhere (unmasked); the returned boolean ``pupil_mask`` marks rho <= 1.

    Returns ``(basis, pupil_mask)`` with ``basis`` shape (n_modes, grid, grid).
    """
    c = (grid_px - 1) / 2.0
    y, x = np.mgrid[0:grid_px, 0:grid_px].astype(np.float64)
    xr = (x - c) / pupil_radius_px
    yr = (y - c) / pupil_radius_px
    rho = np.sqrt(xr ** 2 + yr ** 2)
    theta = np.arctan2(yr, xr)
    pupil_mask = rho <= 1.0

    basis = np.empty((n_modes, grid_px, grid_px), dtype=np.float64)
    for i in range(n_modes):
        n, m = noll_to_nm(start_j + i)
        basis[i] = zernike_nm(n, m, rho, theta)
    return basis, pupil_mask
