"""Coherence time (tau0) and wind estimation from the slope time series.

We recover the wind by the standard operational method: the spatio-temporal
cross-correlation of the slope field. Under the frozen-flow (Taylor) hypothesis
the whole slope pattern translates rigidly, so the time-lagged cross-correlation
peak displaces by V*tau. Because slopes are a high-pass (gradient) quantity,
that peak is well localised -- unlike the raw phase, whose smooth low-order
content keeps its correlation peak pinned near zero.

The coherence time follows from the Greenwood relation tau0 = 0.314 r0 / V,
which is exactly the quantity injected in the synthetic generator, so the
recovered value is directly comparable to ground truth.
"""

from __future__ import annotations

import numpy as np

from ..config import SimConfig
from ..geometry import Geometry


def _slope_grids(gradients: np.ndarray, geom: Geometry, n: int):
    """Pack (n_frames, n_valid, 2) gradients into dense (n_frames, N, N) maps."""
    idx = geom.valid_subap_index
    T = gradients.shape[0]
    Sx = np.zeros((T, n, n), dtype=np.float64)
    Sy = np.zeros((T, n, n), dtype=np.float64)
    Sx[:, idx[:, 0], idx[:, 1]] = gradients[:, :, 0]
    Sy[:, idx[:, 0], idx[:, 1]] = gradients[:, :, 1]
    return Sx, Sy


def _parabolic(m, o, p):
    denom = m - 2.0 * o + p
    return 0.0 if abs(denom) < 1e-12 else float(np.clip(0.5 * (m - p) / denom, -0.5, 0.5))


def _peak_shift(corr: np.ndarray) -> tuple[float, float]:
    """Sub-pixel (sx, sy) location of the cross-correlation peak (circular)."""
    n0, n1 = corr.shape
    pr, pc = np.unravel_index(int(np.argmax(corr)), corr.shape)
    dy = _parabolic(corr[(pr - 1) % n0, pc], corr[pr, pc], corr[(pr + 1) % n0, pc])
    dx = _parabolic(corr[pr, (pc - 1) % n1], corr[pr, pc], corr[pr, (pc + 1) % n1])
    sy = (pr if pr <= n0 // 2 else pr - n0) + dy
    sx = (pc if pc <= n1 // 2 else pc - n1) + dx
    return sx, sy


def wind_from_slope_correlation(
    gradients: np.ndarray,
    cfg: SimConfig,
    geom: Geometry,
    lags: tuple[int, ...] = (3, 5, 8),
) -> tuple[float, float]:
    """Estimate wind speed [m/s] and direction [deg] from slope correlations.

    Returns ``(speed_ms, direction_deg)``. The per-frame displacement is the
    median over the requested lags of (peak shift / lag), converted from
    sub-aperture pitch to metres per frame and divided by the frame interval.
    """
    n = cfg.mla.n_subaps
    Sx, Sy = _slope_grids(gradients, geom, n)
    T = Sx.shape[0]
    d = cfg.subap_pitch_m
    dt = cfg.detector.frame_dt_s

    per_lag = []
    for L in lags:
        if L >= T:
            continue
        # Time-averaged cross-correlation (sum x and y channels) at this lag.
        acc = np.zeros((n, n), dtype=np.float64)
        for i in range(T - L):
            acc += np.fft.ifft2(np.conj(np.fft.fft2(Sx[i])) * np.fft.fft2(Sx[i + L])).real
            acc += np.fft.ifft2(np.conj(np.fft.fft2(Sy[i])) * np.fft.fft2(Sy[i + L])).real
        sx, sy = _peak_shift(acc)
        per_lag.append((sx / L, sy / L))
    per_lag = np.array(per_lag)
    vx, vy = np.median(per_lag[:, 0]), np.median(per_lag[:, 1])
    shift_per_frame = np.hypot(vx, vy)  # sub-aperture pitches per frame
    speed = shift_per_frame * d / dt
    direction = float(np.degrees(np.arctan2(vy, vx)))
    return speed, direction


def tau0_greenwood(r0_m: float, wind_speed_ms: float) -> float:
    """Greenwood coherence time tau0 = 0.314 r0 / V [s]."""
    if wind_speed_ms <= 0:
        return float("inf")
    return 0.314 * r0_m / wind_speed_ms
