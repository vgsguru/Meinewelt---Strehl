"""Phase 5 validation -- DM actuator mapping with inter-actuator coupling.

Primary check (per SPEC): pushing the computed actuator strokes back through the
forward influence model reproduces the target correction surface within a small
fitting-error tolerance. Plus the coupling model and stroke-limit behaviour.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from aowfs import SimConfig, build_geometry
from aowfs.actuator import ActuatorConfig, ActuatorMapper, influence_function
from aowfs.actuator.influence_matrix import build_influence_matrix
from aowfs.io import from_synthetic
from aowfs.recon import ZernikeReconstructor


@pytest.fixture(scope="module")
def setup():
    cfg = replace(SimConfig(), n_frames=20)
    ds = from_synthetic(cfg)
    geom = ds.sequence.geometry
    rec = ZernikeReconstructor(cfg, geom, n_modes=45)
    mapper = ActuatorMapper(cfg, geom, ActuatorConfig(rcond=1e-2))
    return cfg, ds, geom, rec, mapper


def test_coupling_model_is_exact():
    """A unit poke gives exactly the coupling fraction at one actuator pitch."""
    c = 0.15
    vals = influence_function(np.array([0.0, 1.0, 2.0]), c)
    np.testing.assert_allclose(vals, [1.0, c, c ** 4], rtol=1e-12)


def test_coupling_embedded_in_matrix():
    """The built influence matrix carries the coupling (within pixel sampling)."""
    cfg = replace(SimConfig())
    geom = build_geometry(cfg)
    info = build_influence_matrix(cfg, geom)
    H, act, P = info["H"], info["act_px"], info["P"]
    pidx = info["pupil_idx"]
    yy, xx = np.divmod(pidx, P)
    ppos = np.column_stack([xx, yy])
    ctr = (P - 1) / 2.0
    i = int(np.argmin(np.sum((act - ctr) ** 2, axis=1)))  # central actuator
    dd = np.sqrt(np.sum((act - act[i]) ** 2, axis=1))
    dd[i] = np.inf
    nbr = int(np.argmin(dd))
    col = H[:, i]
    kn = int(np.argmin(np.sum((ppos - act[nbr]) ** 2, axis=1)))
    ratio = col[kn] / col.max()
    # Pixel discretisation broadens this; the coupling must be in the ballpark.
    assert 0.5 * cfg.dm.coupling < ratio < 1.5 * cfg.dm.coupling, ratio


def test_fitting_error_reconstructed_wavefronts(setup):
    """forward(command(W)) reproduces the target correction surface (SPEC check)."""
    cfg, ds, geom, rec, mapper = setup
    rels = []
    for i in range(ds.sequence.n_frames):
        W = rec.reconstruct(ds.truth.true_slopes[i])
        rels.append(mapper.command(W).relative_fitting_error)
    mean_rel = float(np.mean(rels))
    assert mean_rel < 0.10, f"fitting error {mean_rel:.3%}"


def test_low_order_target_fits_well(setup):
    """A smooth low-order mode is reproduced to high accuracy."""
    cfg, ds, geom, rec, mapper = setup
    W = rec.basis[2] * 2.0  # Noll j=4 (focus), amplitude 2 rad
    assert mapper.command(W).relative_fitting_error < 0.10


def test_strokes_within_mechanical_limit(setup):
    """Nominal turbulence keeps strokes within the DM limit (no clipping)."""
    cfg, ds, geom, rec, mapper = setup
    W = rec.reconstruct(ds.truth.true_slopes[0])
    res = mapper.command(W)
    assert np.all(np.abs(res.strokes) <= cfg.dm.max_stroke_m + 1e-18)
    assert res.clipped_fraction == 0.0
    assert np.abs(res.strokes).max() < cfg.dm.max_stroke_m


def test_command_is_single_matvec(setup):
    """Per-frame command is the precomputed matrix-vector multiply (+clip)."""
    cfg, ds, geom, rec, mapper = setup
    W = rec.reconstruct(ds.truth.true_slopes[0])
    res = mapper.command(W)
    target = mapper.target_surface_flat(W)
    expected = np.clip(mapper.command_matrix @ target, -mapper.max_stroke, mapper.max_stroke)
    np.testing.assert_allclose(res.strokes, expected)


def test_stroke_grid_shape(setup):
    """Strokes map onto the Fried (N+1)x(N+1) actuator grid."""
    cfg, ds, geom, rec, mapper = setup
    res = mapper.command(rec.reconstruct(ds.truth.true_slopes[0]))
    grid = mapper.stroke_grid(res.strokes)
    n = cfg.mla.n_subaps + 1
    assert grid.shape == (n, n)
    assert np.count_nonzero(~np.isnan(grid)) == geom.n_valid_acts
