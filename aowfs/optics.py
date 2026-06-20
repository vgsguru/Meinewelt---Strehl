"""Shared optical conversions (hardware physics, not tied to the simulator).

The Shack-Hartmann relation between a spot displacement on the detector and the
mean wavefront gradient over the sub-aperture depends only on the optics
(lenslet focal length, sensing wavelength, pupil->MLA relay magnification,
detector pixel size). Both the synthetic forward model and the reconstruction
path need it, so it lives here rather than in ``sim``.

    delta_px = f * (lambda / 2*pi) * (dphi/dx)_pupil * M / pixel_size

with M = (pupil sub-aperture pitch) / (lenslet pitch).
"""

from __future__ import annotations

import numpy as np

from .config import SimConfig
from .geometry import Geometry


def subaperture_gradients(
    phase_window: np.ndarray, cfg: SimConfig, geom: Geometry
) -> np.ndarray:
    """Mean wavefront gradient per valid sub-aperture [rad/m].

    The SH sampling operator: the best-fit-plane slope of the phase over each
    lenslet block, which on a regular grid separates into independent x and y
    least-squares slopes. Used by both the forward model (simulator) and the
    modal interaction matrix, so it lives here as the single source of truth.

    Returns shape ``(n_valid_subaps, 2)`` with columns (gx, gy).
    """
    n = cfg.mla.n_subaps
    p = cfg.screen_px_per_subap
    scale = cfg.screen_pixel_scale_m

    blocks = phase_window.reshape(n, p, n, p)  # (block_row a, in_row i, block_col b, in_col j)
    xc = (np.arange(p) - (p - 1) / 2.0) * scale
    sxx = float(np.sum(xc ** 2)) * p  # denominator of the separable LS slope

    gx_grid = np.einsum("aibj,j->ab", blocks, xc) / sxx  # (block_row, block_col)
    gy_grid = np.einsum("aibj,i->ab", blocks, xc) / sxx  # (block_row, block_col)

    idx = geom.valid_subap_index
    rows, cols = idx[:, 0], idx[:, 1]
    return np.column_stack([gx_grid[rows, cols], gy_grid[rows, cols]])


def disp_factor(cfg: SimConfig) -> float:
    """Detector pixels of spot motion per unit pupil-plane gradient [rad/m]."""
    M = cfg.mla.magnification(cfg.pupil)
    return (
        cfg.mla.focal_length_m
        * cfg.pupil.wavelength_m
        / (2.0 * np.pi)
        * M
        / cfg.detector.pixel_size_m
    )


def gradient_to_displacement(grad_rad_per_m: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """Wavefront gradient [rad/m] -> spot displacement [detector px]."""
    return grad_rad_per_m * disp_factor(cfg)


def displacement_to_gradient(disp_px: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """Spot displacement [detector px] -> wavefront gradient [rad/m]."""
    return disp_px / disp_factor(cfg)
