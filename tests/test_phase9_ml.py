"""Phase 9 validation -- ML reconstructor + direct-gradient-control naming.

The full zonal/modal/ML accuracy+speed table (which needs multi-realization
training) is produced by docs/phase9_ml_compare.py. Here we validate the pieces:
the ML machinery is correct (ridge recovers the analytic inverse on the linear
forward model), inference is a single matvec, and the folded pipeline exposes
the direct-gradient-control matrix.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from aowfs import SimConfig, build_geometry
from aowfs.io import from_synthetic
from aowfs.pipeline import RealTimePipeline
from aowfs.recon import MLReconstructor, ZernikeReconstructor


@pytest.fixture(scope="module")
def small():
    cfg = replace(SimConfig(), n_frames=30)
    return cfg, from_synthetic(cfg)


def test_ml_ridge_recovers_analytic_inverse(small):
    """On the linear forward model, ridge slopes->Zernike ties the modal pinv.

    This is the core ML result: for a well-conditioned linear problem, a learned
    reconstruction matrix just reproduces the first-principles analytic one.
    """
    cfg, ds = small
    geom = ds.sequence.geometry
    nm = 20
    modal = ZernikeReconstructor(cfg, geom, n_modes=nm)
    D = modal.interaction  # (2*n_valid, n_modes): unit mode -> slopes
    nv = geom.n_valid_subaps

    rng = np.random.default_rng(0)
    A = rng.normal(0, 1, (3000, nm))          # diverse, full-rank coefficient set
    S = A @ D.T                                # exact forward-model slopes (T, 2*nv)
    grads = np.stack([S[:, :nv], S[:, nv:]], axis=2)  # (T, nv, 2)

    ml = MLReconstructor(cfg, geom, n_modes=nm, alpha=1e-6).fit(grads, A)
    # Functional equivalence: on any real slope vector, the learned matrix
    # recovers the coefficients and agrees with the analytic modal inverse.
    # (The matrices need not be equal in the null space the training data never
    # excites; what matters is that they act identically on valid slopes.)
    a_test = rng.normal(0, 1, nm)
    s_test = D @ a_test
    g_test = np.stack([s_test[:nv], s_test[nv:]], axis=1)
    assert np.allclose(ml.coeffs(g_test), a_test, atol=1e-2), \
        np.max(np.abs(ml.coeffs(g_test) - a_test))
    assert np.allclose(ml.coeffs(g_test), modal.coeffs(g_test), atol=1e-2)


def test_ml_inference_is_single_matvec(small):
    cfg, ds = small
    geom = ds.sequence.geometry
    nm = 20
    modal = ZernikeReconstructor(cfg, geom, n_modes=nm)
    nv = geom.n_valid_subaps
    rng = np.random.default_rng(1)
    A = rng.normal(0, 1, (500, nm))
    S = A @ modal.interaction.T
    grads = np.stack([S[:, :nv], S[:, nv:]], axis=2)
    ml = MLReconstructor(cfg, geom, n_modes=nm, alpha=1e-3).fit(grads, A)
    g = grads[0]
    s = np.concatenate([g[:, 0], g[:, 1]])
    np.testing.assert_allclose(ml.coeffs(g), ml.recon_matrix @ s)
    assert ml.recon_matrix.shape == (nm, 2 * nv)


def test_direct_gradient_control_is_one_matvec(small):
    """Folded control matrix: direct_command == reactive step, in one matvec."""
    cfg, ds = small
    geom = ds.sequence.geometry
    pipe = RealTimePipeline(cfg, geom, ds.sequence.reference_centroids, n_modes=45)
    frame = ds.sequence.frames[3]
    np.testing.assert_allclose(pipe.direct_command(frame), pipe.step(frame, predict=False),
                               atol=1e-18)
    # control_matrix is the composed slopes->actuator matrix.
    assert pipe.control_matrix.shape == (geom.n_valid_acts, 2 * geom.n_valid_subaps)
    expected = pipe.M_act @ pipe.R_px
    np.testing.assert_allclose(pipe.control_matrix, expected)


def test_ml_reconstructs_wavefront_on_held_out(small):
    """Trained on measured slopes from a few realizations, ML reconstructs sanely."""
    cfg = small[0]
    geom = build_geometry(cfg)
    from aowfs.optics import displacement_to_gradient
    from aowfs.recon import Centroider

    nm = 30
    modal = ZernikeReconstructor(cfg, geom, n_modes=nm)
    Gm, A = [], []
    for seed in (3, 4, 5):
        c = replace(cfg, turbulence=replace(cfg.turbulence, seed=seed))
        d = from_synthetic(c)
        cen = Centroider(c, geom, d.sequence.reference_centroids)
        for i in range(d.sequence.n_frames):
            Gm.append(displacement_to_gradient(cen.measure(d.sequence.frames[i]), c))
            A.append(modal.coeffs(d.truth.true_slopes[i]))
    Gm = np.array(Gm); A = np.array(A)
    ml = MLReconstructor(cfg, geom, n_modes=nm, alpha=1.0).fit(Gm, A)

    # held-out realization
    held = from_synthetic(replace(cfg, turbulence=replace(cfg.turbulence, seed=99)))
    cen = Centroider(cfg, geom, held.sequence.reference_centroids)
    pm = geom.pupil_mask
    errs = []
    for i in range(held.sequence.n_frames):
        g = displacement_to_gradient(cen.measure(held.sequence.frames[i]), cfg)
        d = (ml.reconstruct(g) - held.truth.phase_screens[i])[pm]
        errs.append(np.sqrt(np.var(d - d.mean())))
    truth_rms = np.mean([np.std(held.truth.phase_screens[i][pm]) for i in range(held.sequence.n_frames)])
    assert np.mean(errs) < truth_rms  # reconstruction reduces the error
