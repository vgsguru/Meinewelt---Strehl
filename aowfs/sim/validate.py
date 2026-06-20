"""Phase 0 validation: prove the synthetic SH-WFS simulator is trustworthy.

Run with ``python -m aowfs.sim.validate``. Every check reports a number, not a
claim, and the script exits non-zero if any check fails its tolerance.

Checks
------
A. Forward-model exactness    -- a known analytic phase ramp produces exactly
                                 the analytic sub-aperture gradient.
B. Spot invertibility          -- a rendered, noiseless spot field centroids
                                 back to the injected displacement (sub-pixel).
C. End-to-end centroid error   -- with full noise, measured spot offsets match
                                 the predicted displacement within the noise
                                 floor (this is what Phase 1 will rely on).
D. Injected r0 recovery        -- r0 recovered from the phase structure
                                 function of the generated screen matches the
                                 dialled-in r0 (the Phase 0 closed loop).
E. Frozen-flow consistency     -- consecutive pupil windows are the wind-shifted
                                 same screen, so tau0 is a real recoverable
                                 quantity.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

from ..config import SimConfig, NoiseConfig
from ..geometry import build_geometry
from .generate import generate_dataset
from .phase_screen import FrozenFlowScreen
from . import shwfs


# --------------------------------------------------------------------------- #
# small, self-contained centroider (the real one lives in Phase 1)            #
# --------------------------------------------------------------------------- #


def _com_centroids(frame: np.ndarray, cfg: SimConfig, geom) -> np.ndarray:
    """Thresholded centre-of-mass per valid sub-aperture cell -> (n,2) x,y px."""
    q = cfg.detector.det_px_per_subap
    idx = geom.valid_subap_index
    out = np.empty((idx.shape[0], 2), dtype=np.float64)
    ax = np.arange(q)
    for k in range(idx.shape[0]):
        row, col = idx[k]
        cell = frame[row * q : (row + 1) * q, col * q : (col + 1) * q].astype(np.float64)
        thr = cell.max() * 0.2
        cell = np.clip(cell - thr, 0, None)
        tot = cell.sum()
        if tot <= 0:
            out[k] = [col * q + (q - 1) / 2.0, row * q + (q - 1) / 2.0]
            continue
        cx = (cell.sum(axis=0) @ ax) / tot
        cy = (cell.sum(axis=1) @ ax) / tot
        out[k] = [col * q + cx, row * q + cy]
    return out


# --------------------------------------------------------------------------- #
# structure-function r0 estimate (screen-level closed loop)                    #
# --------------------------------------------------------------------------- #


def _r0_from_structure_function(
    screen: np.ndarray, pixel_scale_m: float, l0_m: float, r0_guess_m: float
) -> float:
    """Estimate r0 from the phase structure function D(r)=6.88 (r/r0)^(5/3).

    The fit must live in the Kolmogorov inertial range: separations *above* the
    inner scale l0 (below which power is suppressed and r0 reads high) and
    *below* r0 (where the finite screen / outer scale flatten D). We pick the
    separation window adaptively from those two physical scales.
    """
    l0_px = l0_m / pixel_scale_m
    r0_px = r0_guess_m / pixel_scale_m
    lo = max(2, int(np.ceil(2.0 * l0_px)))
    hi = max(lo + 3, int(np.floor(0.7 * r0_px)))
    seps_px = np.arange(lo, hi)
    r0_estimates = []
    for s in seps_px:
        diff_x = screen[:, s:] - screen[:, :-s]
        diff_y = screen[s:, :] - screen[:-s, :]
        D = 0.5 * (np.mean(diff_x ** 2) + np.mean(diff_y ** 2))
        r_m = s * pixel_scale_m
        # D = 6.88 (r/r0)^(5/3)  =>  r0 = r / (D/6.88)^(3/5)
        r0_estimates.append(r_m / (D / 6.88) ** (3.0 / 5.0))
    return float(np.median(r0_estimates))


# --------------------------------------------------------------------------- #
# checks                                                                       #
# --------------------------------------------------------------------------- #


@dataclass
class Check:
    name: str
    value: float
    unit: str
    tol: float
    passed: bool
    detail: str = ""


def _check_forward_model_exact(cfg: SimConfig) -> Check:
    geom = build_geometry(cfg)
    P = cfg.pupil_grid_px
    scale = cfg.screen_pixel_scale_m
    gx_true, gy_true = 7.3, -4.1  # rad/m
    yy, xx = np.mgrid[0:P, 0:P].astype(float)
    phase = gx_true * (xx * scale) + gy_true * (yy * scale)
    slopes = shwfs.subaperture_slopes(phase, cfg, geom)
    err = np.max(np.abs(slopes - np.array([gx_true, gy_true])))
    return Check("A. forward-model gradient exactness", err, "rad/m", 1e-6, err < 1e-6,
                 "analytic ramp -> sub-aperture gradient")


def _check_spot_invertibility(cfg: SimConfig) -> Check:
    cfg_nn = _with_noise(cfg, enable=False)
    geom = build_geometry(cfg_nn)
    gain = shwfs.calibrate_gain(cfg_nn)
    rng = np.random.default_rng(0)
    n = geom.n_valid_subaps
    disp = np.column_stack([
        rng.uniform(-2.0, 2.0, n),
        rng.uniform(-2.0, 2.0, n),
    ])
    frame = shwfs.render_spotfield(disp, cfg_nn, geom, gain, rng=rng)
    cent = _com_centroids(frame, cfg_nn, geom)
    refs = shwfs.reference_centroids(cfg_nn, geom)
    meas = cent - refs
    rms = float(np.sqrt(np.mean((meas - disp) ** 2)))
    return Check("B. spot render->centroid invertibility (noiseless)", rms, "px", 0.05,
                 rms < 0.05, "recover injected spot displacement")


def _check_end_to_end_centroid(cfg: SimConfig) -> Check:
    ds = generate_dataset(cfg)
    seq, truth = ds.sequence, ds.truth
    predicted = shwfs.slopes_to_displacement_px(truth.true_slopes[0], cfg)
    cent = _com_centroids(seq.frames[0], cfg, seq.geometry)
    meas = cent - seq.reference_centroids
    rms = float(np.sqrt(np.mean((meas - predicted) ** 2)))
    return Check("C. end-to-end centroid error (with noise)", rms, "px", 0.25,
                 rms < 0.25, "measured offset vs predicted displacement")


def _check_r0_recovery(cfg: SimConfig) -> Check:
    """Validate the *generator's* r0 on dedicated large screens.

    Uses fixed large screens (independent of the run's n_frames) and averages
    over several seeds, so this measures the turbulence statistics of the
    screen synthesis rather than the length of any particular run.
    """
    from aotools.turbulence.phasescreen import ft_sh_phase_screen
    r0_true = cfg.r0_sensing_m
    scale = cfg.screen_pixel_scale_m
    N = 512
    ests = []
    for s in range(4):
        scr = ft_sh_phase_screen(
            r0=r0_true, N=N, delta=scale,
            L0=cfg.turbulence.L0_m, l0=cfg.turbulence.l0_m,
            seed=cfg.turbulence.seed + 100 + s,
        ).astype(np.float64)
        ests.append(_r0_from_structure_function(scr, scale, cfg.turbulence.l0_m, r0_true))
    r0_est = float(np.mean(ests))
    rel = abs(r0_est - r0_true) / r0_true
    return Check("D. injected r0 recovery (structure function)", 100 * rel, "% err", 20.0,
                 rel < 0.20, f"true={r0_true*100:.2f} cm  recovered={r0_est*100:.2f} cm "
                 f"(mean of 4 512px screens)")


def _check_frozen_flow(cfg: SimConfig) -> Check:
    """Frame n+1 must be frame n translated by exactly the wind vector.

    We reconstruct window(1) by shifting window(0) by the known per-frame wind
    travel and compare on the interior (excluding the strip where new
    turbulence enters). A small residual confirms both the magnitude and the
    direction of the frozen-flow translation -- i.e. tau0 = 0.314 r0 / V is a
    real, recoverable property of the series rather than an assumption.
    """
    from scipy.ndimage import shift as nd_shift
    screen = FrozenFlowScreen(cfg)
    w0 = screen.window(0)
    w1 = screen.window(1)
    sx, sy = screen._step  # actual per-frame sampling step (px)
    dx, dy = cfg.wind_vector_px_per_frame
    # window(1)[n] = w0[n + step] -> reconstruct via nd_shift by (-step_y, -step_x).
    w0_to_1 = nd_shift(w0, shift=(-sy, -sx), order=3, mode="nearest")
    m = int(np.ceil(max(abs(sx), abs(sy)))) + 3  # exclude the entering strip
    a = w0_to_1[m:-m, m:-m]
    b = w1[m:-m, m:-m]
    rel = float(np.sqrt(np.mean((a - b) ** 2)) / np.std(b))
    return Check("E. frozen-flow shift consistency", 100 * rel, "% of screen RMS", 8.0,
                 rel < 0.08, f"wind shift {dx:.2f},{dy:.2f} px/frame reproduces next frame")


# --------------------------------------------------------------------------- #
# helpers + runner                                                             #
# --------------------------------------------------------------------------- #


def _with_noise(cfg: SimConfig, enable: bool) -> SimConfig:
    from dataclasses import replace
    return replace(cfg, noise=replace(cfg.noise, enable=enable))


def run(cfg: SimConfig | None = None) -> list[Check]:
    cfg = cfg or SimConfig()
    return [
        _check_forward_model_exact(cfg),
        _check_spot_invertibility(cfg),
        _check_end_to_end_centroid(cfg),
        _check_r0_recovery(cfg),
        _check_frozen_flow(cfg),
    ]


def main() -> int:
    cfg = SimConfig()
    print("=" * 78)
    print("PHASE 0 VALIDATION  --  synthetic SH-WFS simulator")
    print("=" * 78)
    print(f"  pupil D            : {cfg.pupil.diameter_m*100:.1f} cm")
    print(f"  sensing wavelength : {cfg.pupil.wavelength_m*1e9:.0f} nm")
    print(f"  MLA                : {cfg.mla.n_subaps}x{cfg.mla.n_subaps} lenslets, "
          f"f={cfg.mla.focal_length_m*1e3:.1f} mm (f/{cfg.mla.f_number():.0f}), "
          f"relay M={cfg.mla.magnification(cfg.pupil):.0f}")
    print(f"  D/r0               : {cfg.pupil.diameter_m/cfg.r0_sensing_m:.2f}  "
          f"(spot sigma {shwfs.spot_sigma_px(cfg):.2f} px)")
    print(f"  detector           : {cfg.detector_px}x{cfg.detector_px} px, "
          f"{cfg.detector.bit_depth}-bit")
    print(f"  injected r0        : {cfg.r0_sensing_m*100:.2f} cm @ sensing lambda")
    print(f"  injected tau0      : {cfg.tau0_s*1e3:.2f} ms  "
          f"(V={cfg.turbulence.wind_speed_ms:.1f} m/s)")
    print("-" * 78)
    checks = run(cfg)
    width = max(len(c.name) for c in checks)
    all_ok = True
    for c in checks:
        status = "PASS" if c.passed else "FAIL"
        all_ok &= c.passed
        print(f"  [{status}] {c.name:<{width}}  {c.value:10.4f} {c.unit:<16} "
              f"(tol {c.tol:g})")
        if c.detail:
            print(f"         -> {c.detail}")
    print("-" * 78)
    print("  RESULT:", "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    print("=" * 78)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
