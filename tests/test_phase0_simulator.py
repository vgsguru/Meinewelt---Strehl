"""Phase 0 validation tests -- synthetic SH-WFS simulator against ground truth.

These assert the numeric tolerances reported by ``aowfs.sim.validate`` and add
shape/contract checks for the data structures and the disk round-trip.
"""

from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import pytest

from aowfs import SimConfig, build_geometry
from aowfs.io import from_synthetic, load_sequence, save_dataset
from aowfs.sim import validate as V


@pytest.fixture(scope="module")
def small_cfg() -> SimConfig:
    """A small, fast configuration for the test suite."""
    base = SimConfig()
    return replace(base, n_frames=20)


def test_validation_checks_pass(small_cfg):
    """All Phase 0 closed-loop checks meet their tolerances."""
    checks = V.run(small_cfg)
    failed = [c for c in checks if not c.passed]
    msg = "\n".join(f"{c.name}: {c.value:.4f} {c.unit} (tol {c.tol})" for c in failed)
    assert not failed, f"failed checks:\n{msg}"


def test_forward_model_is_exact(small_cfg):
    """An analytic phase ramp yields exactly the analytic gradient."""
    c = V._check_forward_model_exact(small_cfg)
    assert c.value < 1e-6


def test_flat_wavefront_gives_zero_slopes(small_cfg):
    """Zero turbulence => zero true slopes and spots on the reference grid."""
    cfg = replace(small_cfg, turbulence=replace(small_cfg.turbulence, r0_ref_m=1e6))
    cfg = replace(cfg, noise=replace(cfg.noise, enable=False))
    ds = from_synthetic(cfg)
    assert np.max(np.abs(ds.truth.true_slopes)) < 1e-3
    cent = V._com_centroids(ds.sequence.frames[0], cfg, ds.sequence.geometry)
    assert np.max(np.abs(cent - ds.sequence.reference_centroids)) < 0.05


def test_dataset_shapes_and_contract(small_cfg):
    ds = from_synthetic(small_cfg)
    seq, truth = ds.sequence, ds.truth
    geom = seq.geometry
    det = small_cfg.detector_px
    assert seq.frames.shape == (small_cfg.n_frames, det, det)
    assert seq.frames.dtype == np.uint8  # 8-bit .bmp shape
    assert seq.reference_centroids.shape == (geom.n_valid_subaps, 2)
    assert truth.true_slopes.shape == (small_cfg.n_frames, geom.n_valid_subaps, 2)
    assert truth.phase_screens.shape[0] == small_cfg.n_frames
    # ground truth must be a separate object, not embedded in the sequence
    assert not hasattr(seq, "phase_screens")


def test_tau0_relation(small_cfg):
    """tau0 = 0.314 r0 / V holds for the injected parameters."""
    expected = 0.314 * small_cfg.r0_sensing_m / small_cfg.turbulence.wind_speed_ms
    assert small_cfg.tau0_s == pytest.approx(expected, rel=1e-9)


def test_disk_roundtrip(tmp_path, small_cfg):
    """Frames written as .bmp + metadata reload into an identical sequence."""
    ds = from_synthetic(small_cfg)
    out = os.path.join(str(tmp_path), "run")
    save_dataset(ds, out, save_truth=False)
    reloaded = load_sequence(out)
    assert reloaded.n_frames == ds.sequence.n_frames
    assert reloaded.frames.shape == ds.sequence.frames.shape
    np.testing.assert_array_equal(reloaded.frames, ds.sequence.frames)
    np.testing.assert_allclose(
        reloaded.reference_centroids, ds.sequence.reference_centroids
    )
    # geometry rebuilt from metadata must match
    assert reloaded.geometry.n_valid_subaps == ds.sequence.geometry.n_valid_subaps


def test_geometry_fried_actuator_count(small_cfg):
    """Fried geometry: (N+1) actuators across for N sub-apertures."""
    geom = build_geometry(small_cfg)
    assert geom.n_acts == small_cfg.mla.n_subaps + 1
