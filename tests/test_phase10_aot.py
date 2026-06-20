"""Phase 10 validation -- real AOT (NAOMI) loader + centroiding cross-check.

These tests need the real NAOMI3 AOT file (~68 MB, not in the repo); they skip
cleanly when it is absent. This is a robustness check on real on-sky telemetry,
not a ground-truth accuracy check.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

NAOMI = r"D:\Projects\isro_hackathon_2026\challenge_9\data\real\zenodo\NAOMI3_2019-12-30_DATA_LOGGER-212526.fits"
pytestmark = pytest.mark.skipif(not os.path.exists(NAOMI), reason="real NAOMI AOT file not present")


@pytest.fixture(scope="module")
def loaded():
    from aowfs.io.aot_loader import load_naomi_aot
    return load_naomi_aot(NAOMI)


def test_loads_and_aligns(loaded):
    seq, ref = loaded.sequence, loaded.reference
    # frame-number alignment: frames match stored slopes count, fewer than raw 3095
    assert seq.frames.shape[0] == ref.stored_slopes.shape[0]
    assert 2000 < seq.frames.shape[0] < 3095  # tail frames dropped, most kept
    assert seq.frames.shape[1:] == (64, 64)
    assert ref.loop_rate_hz == pytest.approx(500.149, rel=1e-3)


def test_geometry_matches_telemetry(loaded):
    seq, ref = loaded.sequence, loaded.reference
    # our circular-pupil 4x4 selection must reproduce the instrument's 12 valid sub-aps
    assert seq.geometry.n_valid_subaps == ref.stored_slopes.shape[1] == 12
    assert seq.reference_centroids.shape == (12, 2)


def test_calibration_removed_dark_pedestal(loaded):
    # calibrated frames have a near-zero background (dark subtracted), not ~129 ADU
    seq = loaded.sequence
    bg = np.median(seq.frames[0])
    assert bg < 50, f"background {bg} suggests dark not subtracted"


def test_centroids_track_instrument_rtc(loaded):
    """Our centroids reproduce the instrument's residual slopes in magnitude and
    correlate with them (moderate, residual is near the noise floor)."""
    from aowfs.recon import Centroider, CentroidConfig
    seq, ref = loaded.sequence, loaded.reference
    cen = Centroider(seq.cfg, seq.geometry, seq.reference_centroids,
                     CentroidConfig(method="com", threshold_frac=0.2))
    N = 800
    ours = np.array([cen.measure(seq.frames[i]) for i in range(N)])
    stor = ref.stored_slopes[:N]
    # magnitude agreement within 2x
    assert 0.5 < ours.std() / stor.std() < 2.0
    # high-SNR aggregate tilt mode correlates well
    a = ours[:, :, 1].mean(1); b = stor[:, :, 1].mean(1)
    a = a - a.mean(); b = b - b.mean()
    tilt_corr = (a @ b) / np.sqrt((a @ a) * (b @ b))
    assert tilt_corr > 0.6, f"tilt correlation {tilt_corr:.2f}"


def test_reported_atmosphere_extracted(loaded):
    rep = loaded.reference.reported
    assert rep["seeing_arcsec"] == pytest.approx(0.53, abs=0.01)
    assert rep["r0_500nm_m"] == pytest.approx(0.191, abs=0.01)  # derived from seeing
    assert rep["tau0_s"] == pytest.approx(0.0165, abs=0.001)
