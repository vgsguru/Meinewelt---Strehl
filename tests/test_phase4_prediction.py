"""Phase 4 validation -- predictive layer reduces servo-lag residual error.

Centerpiece comparison: residual wavefront variance (and implied Strehl) with
vs. without linear prediction, swept across loop delay. Also checks that the
linear model's residuals are near-white -- the evidence-based justification for
not escalating to a neural predictor (matching the ALTAIR finding).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from aowfs import SimConfig
from aowfs.io import from_synthetic
from aowfs.optics import displacement_to_gradient
from aowfs.predict import LinearPredictor, PredictorConfig, evaluate_lag_sweep
from aowfs.recon import Centroider, ZernikeReconstructor


@pytest.fixture(scope="module")
def coeff_series():
    cfg = replace(SimConfig(), n_frames=600)
    ds = from_synthetic(cfg)
    seq, truth, geom = ds.sequence, ds.truth, ds.sequence.geometry
    cen = Centroider(cfg, geom, seq.reference_centroids)
    rec = ZernikeReconstructor(cfg, geom, n_modes=66)
    a_true = np.array([rec.coeffs(truth.true_slopes[i]) for i in range(seq.n_frames)])
    a_meas = np.array(
        [rec.coeffs(displacement_to_gradient(cen.measure(seq.frames[i]), cfg))
         for i in range(seq.n_frames)]
    )
    return cfg, a_true, a_meas


def test_prediction_reduces_lag_error_isolated(coeff_series):
    """With true coeffs, linear prediction beats the naive integrator at every delay."""
    _, a_true, _ = coeff_series
    r = evaluate_lag_sweep(a_true, a_true, delays=(1, 2, 3, 4), order=5)
    assert np.all(r.pred_var < r.naive_var)
    assert np.all(r.pred_strehl > r.naive_strehl)
    # Substantial reduction once the lag matters (delay >= 2).
    assert np.all(r.reduction_pct[1:] > 45.0), r.as_table()


def test_prediction_reduces_lag_error_realistic(coeff_series):
    """With measured (noisy) coeffs, prediction still cuts residual error."""
    _, a_true, a_meas = coeff_series
    r = evaluate_lag_sweep(a_meas, a_true, delays=(1, 2, 3, 4), order=5)
    assert np.all(r.pred_var < r.naive_var)
    assert np.all(r.reduction_pct[1:] > 35.0), r.as_table()


def test_strehl_improvement_at_two_frame_delay(coeff_series):
    """Headline number: Strehl improvement at a realistic 2-frame loop delay."""
    _, a_true, a_meas = coeff_series
    r = evaluate_lag_sweep(a_meas, a_true, delays=(2,), order=5)
    assert r.pred_strehl[0] - r.naive_strehl[0] > 0.05


def test_linear_residual_is_near_white(coeff_series):
    """The one-step linear prediction residual is near-white.

    For a single-step forecast an optimal linear model leaves a white residual;
    remaining whiteness here means the linear model has captured the predictable
    (frozen-flow) structure, justifying not defaulting to a neural predictor.
    (Multi-step residuals are MA(d-1)-correlated by construction, so whiteness
    is only meaningful at delay = 1.)
    """
    cfg, a_true, _ = coeff_series
    T, n_modes = a_true.shape
    split = T // 2
    cfgp = PredictorConfig(order=5, delay=1)
    pred = LinearPredictor(n_modes, cfgp).fit(a_true[:split])
    preds, idx = pred.predict_sequence(a_true)
    test = idx >= split
    res = a_true[idx[test]] - preds[test]  # (n_test, n_modes)
    naive_res = a_true[idx[test]] - a_true[idx[test] - cfgp.delay]

    def lag1_autocorr(x):
        # summed-mode residual series, normalised lag-1 autocorrelation magnitude
        s = x.sum(axis=1)
        s = s - s.mean()
        return abs(np.sum(s[1:] * s[:-1]) / np.sum(s * s))

    assert lag1_autocorr(res) < lag1_autocorr(naive_res)
    assert lag1_autocorr(res) < 0.30  # one-step residual substantially whitened


def test_runtime_prediction_is_single_dot_product(coeff_series):
    """Per-frame prediction equals one length-p dot product per mode."""
    _, a_true, _ = coeff_series
    cfgp = PredictorConfig(order=5, delay=2)
    pred = LinearPredictor(a_true.shape[1], cfgp).fit(a_true[:300])
    # Manual single-step prediction matches the vectorised sequence output.
    preds, idx = pred.predict_sequence(a_true)
    t = idx[-1]
    manual = np.einsum("mk,km->m", pred.weights, a_true[t - cfgp.delay : t - cfgp.delay - cfgp.order : -1])
    np.testing.assert_allclose(preds[-1], manual, rtol=1e-9, atol=1e-9)
