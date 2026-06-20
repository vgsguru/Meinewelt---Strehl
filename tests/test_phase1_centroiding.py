"""Phase 1 validation -- SH centroiding against ground-truth displacements."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from aowfs import SimConfig
from aowfs.io import from_synthetic
from aowfs.recon import CentroidConfig, Centroider
from aowfs.sim import shwfs


@pytest.fixture(scope="module")
def small_cfg() -> SimConfig:
    return replace(SimConfig(), n_frames=20)


def _true_disp(ds, i):
    return shwfs.slopes_to_displacement_px(ds.truth.true_slopes[i], ds.sequence.cfg)


def test_flat_wavefront_noiseless_is_exact(small_cfg):
    """Flat wavefront, no noise -> spots sit on the reference (offsets ~0)."""
    cfg = replace(small_cfg, turbulence=replace(small_cfg.turbulence, r0_ref_m=1e6))
    cfg = replace(cfg, noise=replace(cfg.noise, enable=False))
    ds = from_synthetic(cfg)
    c = Centroider(cfg, ds.sequence.geometry, ds.sequence.reference_centroids)
    disp = c.measure(ds.sequence.frames[0])
    assert np.sqrt(np.mean(disp ** 2)) < 0.01


def test_flat_wavefront_noise_bounded(small_cfg):
    """Flat wavefront with noise -> residual is small and bounded by the noise floor."""
    cfg = replace(small_cfg, turbulence=replace(small_cfg.turbulence, r0_ref_m=1e6))
    ds = from_synthetic(cfg)
    c = Centroider(cfg, ds.sequence.geometry, ds.sequence.reference_centroids)
    disp = c.measure(ds.sequence.frames[0])
    rms = np.sqrt(np.mean(disp ** 2))
    assert rms < 0.2, f"flat-wavefront noise residual {rms:.4f} px too large"


def test_noiseless_com_unbiased(small_cfg):
    """Noiseless turbulence -> CoM recovers the true displacement (no bias)."""
    cfg = replace(small_cfg, noise=replace(small_cfg.noise, enable=False))
    ds = from_synthetic(cfg)
    c = Centroider(cfg, ds.sequence.geometry, ds.sequence.reference_centroids)
    disp = c.measure(ds.sequence.frames[0])
    err = np.sqrt(np.mean((disp - _true_disp(ds, 0)) ** 2))
    assert err < 0.02, f"noiseless CoM error {err:.4f} px"


def test_noisy_com_accuracy(small_cfg):
    """Realistic noise -> CoM matches ground truth within the noise floor."""
    ds = from_synthetic(small_cfg)
    c = Centroider(small_cfg, ds.sequence.geometry, ds.sequence.reference_centroids)
    meas = c.measure_sequence(ds.sequence.frames)
    errs = [np.sqrt(np.mean((meas[i] - _true_disp(ds, i)) ** 2)) for i in range(ds.sequence.n_frames)]
    assert np.mean(errs) < 0.15, f"noisy CoM RMS {np.mean(errs):.4f} px"


def _seq_rms(centroider, ds):
    errs = []
    for i in range(ds.sequence.n_frames):
        m = centroider.measure(ds.sequence.frames[i])
        errs.append(np.sqrt(np.mean((m - _true_disp(ds, i)) ** 2)))
    return float(np.mean(errs))


def test_correlation_unbiased_and_competitive(small_cfg):
    """At nominal SNR, plain correlation is accurate and no worse than CoM."""
    ds = from_synthetic(small_cfg)
    geom, refs = ds.sequence.geometry, ds.sequence.reference_centroids
    com = Centroider(small_cfg, geom, refs, CentroidConfig(method="com"))
    cor = Centroider(small_cfg, geom, refs, CentroidConfig(method="correlation"))
    com_rms, cor_rms = _seq_rms(com, ds), _seq_rms(cor, ds)
    assert cor_rms < 0.15, f"correlation accuracy {cor_rms:.3f} px"
    assert cor_rms <= com_rms * 1.1, f"correlation {cor_rms:.3f} vs com {com_rms:.3f}"


def test_apodized_correlation_more_robust_at_low_snr(small_cfg):
    """In the read-noise-limited regime, apodised correlation beats plain CoM."""
    cfg = replace(
        small_cfg,
        noise=replace(small_cfg.noise, photons_per_spot=300.0, read_noise_e=3.0),
    )
    ds = from_synthetic(cfg)
    geom, refs = ds.sequence.geometry, ds.sequence.reference_centroids
    com = Centroider(cfg, geom, refs, CentroidConfig(method="com"))
    cor = Centroider(cfg, geom, refs, CentroidConfig(method="correlation", apodize=True))
    com_rms, cor_rms = _seq_rms(com, ds), _seq_rms(cor, ds)
    assert cor_rms < com_rms, f"apodised correlation {cor_rms:.3f} not better than com {com_rms:.3f}"


def test_output_shape_and_convention(small_cfg):
    ds = from_synthetic(small_cfg)
    c = Centroider(small_cfg, ds.sequence.geometry, ds.sequence.reference_centroids)
    disp = c.measure(ds.sequence.frames[0])
    assert disp.shape == (ds.sequence.geometry.n_valid_subaps, 2)
    # sign convention: correlate measured with true displacement -> positive
    td = _true_disp(ds, 0)
    assert np.sum(disp * td) > 0
