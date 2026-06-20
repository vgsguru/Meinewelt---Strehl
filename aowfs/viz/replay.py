"""Pure computation backing the demo dashboard (no Streamlit dependency).

Produces, for a synthetic run, the per-frame residual wavefront maps with and
without the predictive layer plus the headline metrics, so the dashboard (or a
test) can replay the correction side by side.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import SimConfig
from ..io import from_synthetic
from ..optics import displacement_to_gradient
from ..predict.ar_predictor import LinearPredictor, PredictorConfig
from ..recon import Centroider, ZernikeReconstructor
from ..turbulence import characterize


@dataclass
class ReplayData:
    frames: np.ndarray  # (T, det, det) detector frames
    true_wf: np.ndarray  # (Tt, P, P) true wavefront over pupil [rad]
    resid_naive: np.ndarray  # (Tt, P, P) residual without prediction [rad]
    resid_pred: np.ndarray  # (Tt, P, P) residual with prediction [rad]
    strehl_naive: np.ndarray  # (Tt,)
    strehl_pred: np.ndarray  # (Tt,)
    t_index: np.ndarray  # frame indices corresponding to the residual arrays
    pupil_mask: np.ndarray  # (P, P) bool
    delay: int
    # headline numbers
    mean_strehl_naive: float
    mean_strehl_pred: float
    r0_recovered_cm: float
    r0_injected_cm: float
    tau0_recovered_ms: float
    tau0_injected_ms: float
    wind_speed_ms: float


def compute_replay(
    cfg: SimConfig,
    delay: int = 2,
    n_modes: int = 45,
    order: int = 5,
    train_frac: float = 0.5,
    max_display_frames: int = 120,
) -> ReplayData:
    ds = from_synthetic(cfg)
    seq, truth, geom = ds.sequence, ds.truth, ds.sequence.geometry
    cen = Centroider(cfg, geom, seq.reference_centroids)
    rec = ZernikeReconstructor(cfg, geom, n_modes=n_modes)

    a_meas = np.array(
        [rec.coeffs(displacement_to_gradient(cen.measure(seq.frames[i]), cfg))
         for i in range(seq.n_frames)]
    )
    a_true = np.array([rec.coeffs(truth.true_slopes[i]) for i in range(seq.n_frames)])

    # Turbulence characterization (for the dashboard panel).
    grads = np.array(
        [displacement_to_gradient(cen.measure(seq.frames[i]), cfg) for i in range(seq.n_frames)]
    )
    turb = characterize(grads, a_meas, cfg, geom)

    # Fit the predictor on the training segment only.
    split = int(seq.n_frames * train_frac)
    pred = LinearPredictor(n_modes, PredictorConfig(order=order, delay=delay)).fit(a_meas[:split])
    preds, idx = pred.predict_sequence(a_meas)

    # Evaluate on the test segment; cap how many frames we render.
    test = idx >= split
    ti = idx[test]
    pr = preds[test]
    if ti.size > max_display_frames:
        sel = np.linspace(0, ti.size - 1, max_display_frames).astype(int)
        ti, pr = ti[sel], pr[sel]

    # Residual coefficients: true minus the correction the loop would apply.
    res_naive_c = a_true[ti] - a_meas[ti - delay]
    res_pred_c = a_true[ti] - pr

    strehl_naive = np.exp(-np.sum(res_naive_c ** 2, axis=1))
    strehl_pred = np.exp(-np.sum(res_pred_c ** 2, axis=1))

    # Synthesise the wavefront maps for display.
    true_wf = np.array([rec.phase_from_coeffs(a_true[t]) for t in ti])
    resid_naive = np.array([rec.phase_from_coeffs(res_naive_c[i]) for i in range(ti.size)])
    resid_pred = np.array([rec.phase_from_coeffs(res_pred_c[i]) for i in range(ti.size)])

    return ReplayData(
        frames=seq.frames,
        true_wf=true_wf,
        resid_naive=resid_naive,
        resid_pred=resid_pred,
        strehl_naive=strehl_naive,
        strehl_pred=strehl_pred,
        t_index=ti,
        pupil_mask=geom.pupil_mask,
        delay=delay,
        mean_strehl_naive=float(strehl_naive.mean()),
        mean_strehl_pred=float(strehl_pred.mean()),
        r0_recovered_cm=turb.r0_m * 100,
        r0_injected_cm=cfg.r0_sensing_m * 100,
        tau0_recovered_ms=turb.tau0_s * 1e3,
        tau0_injected_ms=cfg.tau0_s * 1e3,
        wind_speed_ms=turb.wind_speed_ms,
    )
