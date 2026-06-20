"""Shack-Hartmann WFS forward model: phase screen -> detector spot field.

Physics, kept deliberately explicit so the model is auditable:

  * Each lenslet measures the mean wavefront gradient over its sub-aperture
    (G-tilt). We obtain it as the best-fit plane slope of the phase block,
    which on a regular grid separates into independent x and y least-squares
    slopes.

  * A local phase gradient dphi/dx [rad/m] tilts the wavefront by an angle
    theta = (lambda / 2*pi) * dphi/dx and displaces the focal spot by
    f_lenslet * theta. In detector pixels:

        delta_px = f * (lambda / 2*pi) * (dphi/dx) / pixel_size

  * The spot is a Gaussian of diffraction-limited width (FWHM ~ lambda*f/d)
    carrying ``photons_per_spot`` electrons, rendered into the lenslet's
    detector cell at reference_centre + delta.

  * Noise: Poisson photon statistics + Gaussian read noise, then a fixed
    linear gain to digital numbers and quantisation to the configured bit
    depth (8-bit to match the real rig's .bmp output).

The slope -> displacement map is exact and analytically invertible, so Phase 1
centroiding followed by the inverse map must recover the injected slopes; that
is the Phase 0 validation.
"""

from __future__ import annotations

import numpy as np

from .. import optics
from ..config import SimConfig
from ..geometry import Geometry


# --------------------------------------------------------------------------- #
# Wavefront gradients over the sub-apertures                                   #
# --------------------------------------------------------------------------- #


def subaperture_slopes(
    phase_window: np.ndarray, cfg: SimConfig, geom: Geometry
) -> np.ndarray:
    """Mean wavefront gradient per valid sub-aperture [rad/m].

    Thin alias for :func:`aowfs.optics.subaperture_gradients` (the shared SH
    sampling operator), kept for the simulator's call sites.
    """
    return optics.subaperture_gradients(phase_window, cfg, geom)


def slopes_to_displacement_px(slopes_rad_per_m: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """Convert wavefront gradients [rad/m] to spot displacements [detector px]."""
    return optics.gradient_to_displacement(slopes_rad_per_m, cfg)


def displacement_to_slopes_rad_per_m(disp_px: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """Inverse of :func:`slopes_to_displacement_px` (used by reconstruction)."""
    return optics.displacement_to_gradient(disp_px, cfg)


# --------------------------------------------------------------------------- #
# Reference geometry and spot rendering                                        #
# --------------------------------------------------------------------------- #


def reference_centroids(cfg: SimConfig, geom: Geometry) -> np.ndarray:
    """Flat-wavefront spot positions (cell centres) for valid sub-apertures.

    Shape ``(n_valid_subaps, 2)`` with columns (x, y) in detector pixels.
    """
    q = cfg.detector.det_px_per_subap
    idx = geom.valid_subap_index
    rows, cols = idx[:, 0], idx[:, 1]
    cx = cols * q + (q - 1) / 2.0
    cy = rows * q + (q - 1) / 2.0
    return np.column_stack([cx, cy])


def spot_sigma_px(cfg: SimConfig) -> float:
    """Diffraction-limited Gaussian sigma of a lenslet spot, in detector px.

    Spot size is set by the *lenslet* aperture: FWHM ~ lambda * f / d_lenslet.
    """
    d_lenslet = cfg.mla.lenslet_pitch_m
    fwhm_m = cfg.pupil.wavelength_m * cfg.mla.focal_length_m / d_lenslet
    sigma_m = fwhm_m / 2.3548  # FWHM -> sigma
    sigma_px = sigma_m / cfg.detector.pixel_size_m
    return max(sigma_px, 0.6)  # keep at least sub-pixel-resolvable


def calibrate_gain(cfg: SimConfig) -> float:
    """Fixed electrons->DN digitisation gain (a camera property, not flux-scaled).

    Returned once at startup and reused for every frame, so digitisation is a
    stable linear map. Because it does not depend on the photon flux, lowering
    the flux produces a genuinely dim, read-noise-limited image rather than an
    unphysical read-noise-amplified one.
    """
    return cfg.detector.gain_dn_per_e


def render_spotfield(
    disp_px: np.ndarray,
    cfg: SimConfig,
    geom: Geometry,
    gain: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Render the detector spot field for one frame.

    ``disp_px`` is the per-sub-aperture (dx, dy) spot displacement in detector
    pixels. Returns a quantised detector image of the configured bit depth.
    """
    q = cfg.detector.det_px_per_subap
    det = cfg.detector_px
    sigma = spot_sigma_px(cfg)
    two_sig2 = 2.0 * sigma ** 2
    norm = cfg.noise.photons_per_spot / (2.0 * np.pi * sigma ** 2)

    image_e = np.zeros((det, det), dtype=np.float64)
    refs = reference_centroids(cfg, geom)  # (n_valid, 2) x,y
    idx = geom.valid_subap_index

    # Local pixel coordinate axis within a cell.
    cell_ax = np.arange(q)

    for k in range(refs.shape[0]):
        row, col = idx[k]
        r0 = row * q
        c0 = col * q
        cx = refs[k, 0] + disp_px[k, 0]
        cy = refs[k, 1] + disp_px[k, 1]
        # Local coordinates of this cell's pixels in detector-pixel units.
        xs = c0 + cell_ax
        ys = r0 + cell_ax
        gx = np.exp(-((xs - cx) ** 2) / two_sig2)
        gy = np.exp(-((ys - cy) ** 2) / two_sig2)
        cell = norm * np.outer(gy, gx)  # (y, x)
        image_e[r0 : r0 + q, c0 : c0 + q] += cell

    if cfg.noise.enable:
        if rng is None:
            rng = np.random.default_rng()
        if cfg.noise.dark_e > 0:
            image_e += cfg.noise.dark_e
        image_e = rng.poisson(np.clip(image_e, 0, None)).astype(np.float64)
        if cfg.noise.read_noise_e > 0:
            image_e += rng.normal(0.0, cfg.noise.read_noise_e, size=image_e.shape)

    dn = np.clip(np.round(image_e * gain), 0, cfg.detector.max_count)
    dtype = np.uint8 if cfg.detector.bit_depth <= 8 else np.uint16
    return dn.astype(dtype)
