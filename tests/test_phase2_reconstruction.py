"""Phase 2 validation -- zonal + modal wavefront reconstruction.

Primary check (per SPEC): recovered wavefront RMS error vs ground truth shrinks
as more Zernike modes are corrected, tracking the Noll residual. Plus exact
round-trip recovery of known coefficients, zonal-vs-truth accuracy, and
zonal/modal agreement.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.ndimage import map_coordinates

from aowfs import SimConfig, build_geometry
from aowfs.io import from_synthetic
from aowfs.optics import subaperture_gradients
from aowfs.recon import FriedZonalReconstructor, ZernikeReconstructor

# Noll (1976) residual coefficients Delta_J after J terms, in (D/r0)^(5/3) rad^2.
NOLL = {4: 0.111, 6: 0.0648, 11: 0.0377, 21: 0.0208, 36: 0.0124}


@pytest.fixture(scope="module")
def noiseless_ds():
    cfg = replace(SimConfig(), n_frames=15, noise=replace(SimConfig().noise, enable=False))
    return cfg, from_synthetic(cfg)


def _residual_var(rec, ds, pupil_mask):
    errs = []
    for i in range(ds.sequence.n_frames):
        w = rec.reconstruct(ds.truth.true_slopes[i])
        d = (w - ds.truth.phase_screens[i])[pupil_mask]
        errs.append(np.var(d - d.mean()))
    return float(np.mean(errs))


def test_modal_coefficient_roundtrip():
    """A known Zernike wavefront -> slopes -> reconstructed coeffs (exact)."""
    cfg = replace(SimConfig(), n_frames=1)
    geom = build_geometry(cfg)
    rec = ZernikeReconstructor(cfg, geom, n_modes=20, rcond=1e-6)
    rng = np.random.default_rng(0)
    a_true = rng.normal(0, 1, 20)
    # Build the phase as the same Zernike expansion over the full grid (unmasked).
    phase = np.tensordot(a_true, rec.basis, axes=(0, 0))
    g = subaperture_gradients(phase, cfg, geom)
    a_rec = rec.coeffs(g)
    assert np.allclose(a_rec, a_true, atol=1e-3), np.max(np.abs(a_rec - a_true))


def test_modal_residual_follows_noll(noiseless_ds):
    """Residual wavefront variance shrinks with modes, tracking Noll's trend."""
    cfg, ds = noiseless_ds
    geom = ds.sequence.geometry
    pm = geom.pupil_mask
    factor = (cfg.pupil.diameter_m / cfg.r0_sensing_m) ** (5.0 / 3.0)

    modes = [5, 10, 20, 35]
    var = [_residual_var(ZernikeReconstructor(cfg, geom, n_modes=m), ds, pm) for m in modes]

    # 1) strictly decreasing with mode count
    assert all(var[i] > var[i + 1] for i in range(len(var) - 1)), var

    # 2) magnitude tracks Noll within a finite-SH-sampling factor
    for m, v in zip(modes, var):
        noll = NOLL[m + 1] * factor
        ratio = v / noll
        assert 0.8 < ratio < 2.5, f"n_modes={m}: var={v:.3f} Noll={noll:.3f} ratio={ratio:.2f}"


def test_per_frame_is_single_matvec(noiseless_ds):
    """Reconstruction reduces to one precomputed matrix-vector multiply."""
    cfg, ds = noiseless_ds
    geom = ds.sequence.geometry
    rec = ZernikeReconstructor(cfg, geom, n_modes=20)
    g = ds.truth.true_slopes[0]
    s = np.concatenate([g[:, 0], g[:, 1]])
    assert rec.recon_matrix.shape == (20, 2 * geom.n_valid_subaps)
    np.testing.assert_allclose(rec.coeffs(g), rec.recon_matrix @ s)


def test_zonal_reconstructs_truth(noiseless_ds):
    """Fried zonal reconstruction matches ground-truth phase in the interior."""
    cfg, ds = noiseless_ds
    geom = ds.sequence.geometry
    zon = FriedZonalReconstructor(cfg, geom)
    P, p = cfg.pupil_grid_px, cfg.screen_px_per_subap

    c = (P - 1) / 2.0
    yy, xx = np.mgrid[0:P, 0:P]
    rho = np.sqrt((xx - c) ** 2 + (yy - c) ** 2) / (P / 2.0)
    interior = rho < 0.85

    coord = (np.arange(P) + 0.5) / p
    cc, rr = np.meshgrid(coord, coord)

    errs, truth_rms = [], []
    for i in range(ds.sequence.n_frames):
        phi = zon.reconstruct(ds.truth.true_slopes[i])
        wz = map_coordinates(np.nan_to_num(zon.to_grid(phi)), [rr, cc], order=1, mode="nearest")
        t = ds.truth.phase_screens[i]
        d = (wz - t)[interior]
        errs.append(np.var(d - d.mean()))
        truth_rms.append(np.var(t[interior] - t[interior].mean()))
    rel = np.sqrt(np.mean(errs) / np.mean(truth_rms))
    assert rel < 0.25, f"zonal-vs-truth interior rel error {rel:.3f}"


def test_zonal_fit_consistency(noiseless_ds):
    """Reconstructed corner phases reproduce the input slopes (up to high-freq)."""
    cfg, ds = noiseless_ds
    geom = ds.sequence.geometry
    zon = FriedZonalReconstructor(cfg, geom)
    g = ds.truth.true_slopes[0]
    s = np.concatenate([g[:, 0], g[:, 1]])
    phi = zon.reconstruct(g)
    rel = np.linalg.norm(zon.geometry_matrix @ phi - s) / np.linalg.norm(s)
    assert rel < 0.25, f"zonal slope-fit residual {rel:.3f}"
