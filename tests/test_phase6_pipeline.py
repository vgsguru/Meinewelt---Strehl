"""Phase 6 validation -- real-time pipeline correctness + runtime benchmark."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from aowfs import SimConfig, build_geometry
from aowfs.actuator import ActuatorMapper
from aowfs.bench import run_benchmark
from aowfs.io import from_synthetic
from aowfs.optics import displacement_to_gradient
from aowfs.pipeline import RealTimePipeline
from aowfs.recon import ZernikeReconstructor
from aowfs.recon.fast_centroid import FastCentroider
from aowfs.sim import shwfs


@pytest.fixture(scope="module")
def small():
    cfg = replace(SimConfig(), n_frames=40)
    return cfg, from_synthetic(cfg)


def test_pipeline_matches_explicit_chain(small):
    """Collapsed calibration matrices reproduce the step-by-step chain exactly."""
    cfg, ds = small
    geom = ds.sequence.geometry
    frame = ds.sequence.frames[3]

    pipe = RealTimePipeline(cfg, geom, ds.sequence.reference_centroids, n_modes=45, use_numba=True)
    strokes = pipe.step(frame, predict=False)

    rec = ZernikeReconstructor(cfg, geom, n_modes=45)
    mapper = ActuatorMapper(cfg, geom)
    fc = FastCentroider(cfg, geom, ds.sequence.reference_centroids, 0.2)
    g = displacement_to_gradient(fc.measure(frame), cfg)
    expected = mapper.command(rec.phase_from_coeffs(rec.coeffs(g))).strokes
    np.testing.assert_allclose(strokes, expected, atol=1e-18)


def test_numba_centroid_recovers_truth(small):
    """The JIT centroider recovers the injected displacement within the noise floor."""
    cfg, ds = small
    geom = ds.sequence.geometry
    fc = FastCentroider(cfg, geom, ds.sequence.reference_centroids, 0.2)
    disp = fc.measure(ds.sequence.frames[0])
    true = shwfs.slopes_to_displacement_px(ds.truth.true_slopes[0], cfg)
    rms = np.sqrt(np.mean((disp - true) ** 2))
    assert rms < 0.2, f"numba centroid rms {rms:.3f} px"


def test_pipeline_strokes_within_limits(small):
    cfg, ds = small
    geom = ds.sequence.geometry
    pipe = RealTimePipeline(cfg, geom, ds.sequence.reference_centroids, n_modes=45)
    strokes = pipe.step(ds.sequence.frames[0], predict=False)
    assert strokes.shape[0] == geom.n_valid_acts
    assert np.all(np.abs(strokes) <= cfg.dm.max_stroke_m + 1e-18)


def test_replay_predictive_beats_reactive():
    """Dashboard backing computation: prediction raises mean Strehl."""
    from aowfs.viz.replay import compute_replay

    cfg = replace(SimConfig(), n_frames=200)
    d = compute_replay(cfg, delay=2, n_modes=45)
    assert d.mean_strehl_pred > d.mean_strehl_naive
    assert d.resid_pred.shape == d.resid_naive.shape == d.true_wf.shape
    assert d.t_index.size > 0


def test_benchmark_is_realtime():
    """Runtime path is far faster than the turbulence coherence time."""
    cfg = replace(SimConfig(), n_frames=40)
    res = run_benchmark(cfg, n_iters=60)
    assert res.fps > 500, f"only {res.fps:.0f} fps"
    assert res.frames_per_coherence_time > 5
    assert res.numba_centroid_us < res.numpy_centroid_us  # JIT helps
    # the matvecs are negligible vs centroiding
    assert res.stage_times_us["reconstruct"] < res.stage_times_us["centroid"]
    assert res.stage_times_us["actuator"] < res.stage_times_us["centroid"]
