"""Phase 7 validation -- ISRO real-data adapter.

The real dataset isn't available yet, so we validate the adapter by writing
synthetic frames as a directory of .bmp files (the real-data shape) and loading
them back through the *real-data path* -- reading only the frames plus an
``ISROSpecs`` (never the synthetic metadata) and computing the reference itself.
The loaded sequence must drive Phases 1-6 unchanged and reproduce the native
pipeline's output.
"""

from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import pytest

from aowfs import SimConfig
from aowfs.io import ISROSpecs, from_synthetic, load_isro_dataset, save_dataset
from aowfs.pipeline import RealTimePipeline
from aowfs.recon import ZernikeReconstructor
from aowfs.recon.fast_centroid import FastCentroider


@pytest.fixture(scope="module")
def saved_run(tmp_path_factory):
    cfg = replace(SimConfig(), n_frames=30)
    ds = from_synthetic(cfg)
    out = str(tmp_path_factory.mktemp("isro_like"))
    # Write only the .bmp frames as the real rig would (no reliance on sidecars).
    save_dataset(ds, out, save_truth=False)
    specs = ISROSpecs(
        pixel_size_m=cfg.detector.pixel_size_m,
        mla_n_lenslets=cfg.mla.n_subaps,
        mla_lenslet_pitch_m=cfg.mla.lenslet_pitch_m,
        mla_focal_length_m=cfg.mla.focal_length_m,
        pupil_diameter_m=cfg.pupil.diameter_m,
        wavelength_m=cfg.pupil.wavelength_m,
        frame_dt_s=cfg.detector.frame_dt_s,
        dm_coupling=cfg.dm.coupling,
        det_px_per_subap=cfg.detector.det_px_per_subap,  # explicit (synthetic choice)
        screen_px_per_subap=cfg.screen_px_per_subap,
        frame_resolution=(cfg.detector_px, cfg.detector_px),
    )
    return cfg, ds, out, specs


def test_specs_build_matching_config(saved_run):
    cfg, ds, out, specs = saved_run
    rebuilt = specs.to_config()
    assert rebuilt.mla.n_subaps == cfg.mla.n_subaps
    assert rebuilt.detector.det_px_per_subap == cfg.detector.det_px_per_subap
    assert rebuilt.pupil.diameter_m == cfg.pupil.diameter_m
    assert rebuilt.detector_px == cfg.detector_px


def test_loads_frames_into_sequence(saved_run):
    cfg, ds, out, specs = saved_run
    seq = load_isro_dataset(out, specs, reference_mode="geometric")
    assert seq.frames.shape == ds.sequence.frames.shape
    np.testing.assert_array_equal(seq.frames, ds.sequence.frames)
    assert seq.geometry.n_valid_subaps == ds.sequence.geometry.n_valid_subaps


def test_geometric_reference_matches_native(saved_run):
    cfg, ds, out, specs = saved_run
    seq = load_isro_dataset(out, specs, reference_mode="geometric")
    np.testing.assert_allclose(seq.reference_centroids, ds.sequence.reference_centroids)


def test_mean_reference_is_close(saved_run):
    """Time-averaged reference (no flat frame) lands near the true cell centres.

    This is the approximate fallback when no flat frame is supplied; it converges
    as the sequence lengthens (≈1.3 px over 30 frozen-flow frames, ≈0.4 px over
    200). A supplied flat frame (reference_mode='frame') is exact.
    """
    cfg, ds, out, specs = saved_run
    seq = load_isro_dataset(out, specs, reference_mode="mean")
    err = np.abs(seq.reference_centroids - ds.sequence.reference_centroids).max()
    assert err < 1.5, f"mean-reference offset {err:.3f} px"


def test_pipeline_runs_on_isro_loaded_data(saved_run):
    """Phases 1-6 run unchanged on adapter output and match the native path."""
    cfg, ds, out, specs = saved_run
    seq = load_isro_dataset(out, specs, reference_mode="geometric")

    # Reconstruct via the standard pipeline on the adapter-loaded sequence.
    pipe = RealTimePipeline(cfg, seq.geometry, seq.reference_centroids, n_modes=45)
    strokes_isro = pipe.step(seq.frames[5], predict=False)

    pipe_native = RealTimePipeline(
        cfg, ds.sequence.geometry, ds.sequence.reference_centroids, n_modes=45
    )
    strokes_native = pipe_native.step(ds.sequence.frames[5], predict=False)
    np.testing.assert_allclose(strokes_isro, strokes_native, atol=1e-18)


def test_resolution_mismatch_raises(saved_run):
    cfg, ds, out, specs = saved_run
    bad = replace(specs, frame_resolution=(123, 456))
    with pytest.raises(ValueError):
        load_isro_dataset(out, bad, reference_mode="geometric")
