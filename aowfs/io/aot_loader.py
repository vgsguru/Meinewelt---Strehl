"""Adapter for real AOT-format (Adaptive Optics Telemetry) FITS datasets.

Validated against NAOMI on-sky telemetry (ESO VLT Auxiliary Telescope 3,
D = 1.82 m) from the AOT example collection (arXiv:2312.08300). Like the ISRO
adapter, this emits the same :class:`~aowfs.types.WFSFrameSequence` so Phases
1-6 run unchanged; everything real-data-aware lives here.

Three things this loader handles that synthetic/ISRO data did not:

  * **Frame alignment.** Raw detector frames and slopes are logged on different
    cadences (NAOMI: pixels every 10th loop frame) and the pixel stream can
    overrun the loop stream, so we align by the AOT ``FRAME_NUMBERS`` field --
    never by an assumed subsampling ratio -- and drop unmatched frames.
  * **Detector calibration.** Real frames carry a dark pedestal, a flat field,
    and a bad-pixel map; these are applied before the frames are usable.
  * **Closed-loop telemetry.** NAOMI ran closed-loop, so the stored slopes are
    *residual* (post-correction). The raw frames are residual spot fields too,
    which is exactly what the centroiding cross-check needs; recovering the
    *atmospheric* r0 requires a pseudo-open-loop reconstruction (residual +
    DM contribution) handled in the Phase-10 analysis, not here.

Documented instrument facts used (not assumptions): the SPARTA real-time
computer runs NAOMI at ~500 Hz (corroborated here by the FITS header
``ESO AOS LOOP RATE = 500.149``), and the deformable mirror is an ALPAO DM241
with 241 actuators (consistent with the (241, N) telemetry arrays). NB: that
241 is the DM's actuator count and is *unrelated to* — though numerically equal
to — our synthetic 16x16 Fried grid's 241 zonal corner DOF; the coincidence is
flagged so it is not misread as a shared quantity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import (
    DetectorConfig,
    DMConfig,
    MLAConfig,
    NoiseConfig,
    PupilConfig,
    SimConfig,
    TurbulenceConfig,
)
from ..geometry import build_geometry
from ..types import WFSFrameSequence


def _arr(x):
    """Return a plain ndarray from an aotpy Image, list, or array."""
    x = getattr(x, "data", x)
    return np.asarray(x)


@dataclass(frozen=True)
class AOTReference:
    """Instrument-provided telemetry kept for cross-checking (not injected truth)."""

    stored_slopes: np.ndarray  # (n_aligned, n_valid, 2) residual slopes [detector px]
    stored_modal: np.ndarray | None  # (n_aligned, n_modes) residual modal coeffs
    dm_positions: np.ndarray | None  # (n_aligned, n_act)
    frame_numbers: np.ndarray  # (n_aligned,)
    loop_rate_hz: float
    closed_loop: bool
    reported: dict  # seeing_arcsec, r0_500nm_m, tau0_s, wind_speed_ms, wind_dir_deg, atm_wavelength_m


@dataclass(frozen=True)
class AOTDataset:
    sequence: WFSFrameSequence
    reference: AOTReference


# --------------------------------------------------------------------------- #
# NAOMI / generic SH AOT loader                                               #
# --------------------------------------------------------------------------- #


def load_naomi_aot(path: str, calibrate: bool = True) -> AOTDataset:
    """Load a NAOMI-style SH AOT FITS file into a calibrated WFSFrameSequence."""
    from aotpy.io.fits import AOTFITSReader

    sys = AOTFITSReader(path).get_system()
    w = sys.wavefront_sensors[0]
    det = w.detector

    raw = _arr(det.pixel_intensities).astype(np.float64)  # (n_pix, H, W)
    if raw.ndim != 3 or raw.shape[1] != raw.shape[2]:
        raise ValueError(f"expected square raw frames, got {raw.shape}")
    H = raw.shape[1]

    # --- frame-number alignment (never an assumed ratio) ----------------- #
    loop_fn = _arr(w.measurements.time.frame_numbers).astype(np.int64)
    pix_fn = _arr(det.pixel_intensities.time.frame_numbers).astype(np.int64)
    if not np.all(np.diff(loop_fn) == 1):
        raise ValueError("loop frame numbers are not consecutive; alignment assumption invalid")
    idx = pix_fn - loop_fn[0]
    keep = (idx >= 0) & (idx < loop_fn.size)
    sl_idx = idx[keep]
    raw = raw[keep]
    frame_numbers = pix_fn[keep]

    slopes = _arr(w.measurements)  # (n_loop, 2, n_valid)
    stored_slopes = np.transpose(slopes[sl_idx], (0, 2, 1))  # (n_aligned, n_valid, 2)
    n_valid = stored_slopes.shape[1]

    modal_full = _read_named_hdu(path, "MODAL COEFFICIENTS")  # (n_loop, n_modes)
    stored_modal = modal_full[sl_idx] if modal_full is not None else None
    dm_full = _read_named_hdu(path, "DM POSITIONS")  # (n_loop, n_act)
    dm_positions = dm_full[sl_idx] if dm_full is not None else None

    # --- detector calibration -------------------------------------------- #
    if calibrate:
        dark = _arr(det.dark).astype(np.float64) if det.dark is not None else 0.0
        flat = _arr(det.flat_field).astype(np.float64) if det.flat_field is not None else 1.0
        bad = _arr(det.bad_pixel_map) if det.bad_pixel_map is not None else None
        frames = np.clip(raw - dark, 0.0, None) * flat
        if bad is not None:
            frames[:, bad.astype(bool)] = 0.0
    else:
        frames = raw

    # --- geometry / config ----------------------------------------------- #
    n_subaps = int(round(np.sqrt(_count_grid(n_valid))))  # 12 valid -> 4x4 grid
    det_px_per_subap = H // n_subaps
    D = sys.main_telescope.inscribed_diameter or sys.main_telescope.enclosing_diameter
    loop_rate = _header_loop_rate(path, default=500.149)
    cfg = _build_config(D, n_subaps, det_px_per_subap, H, loop_rate)
    geom = build_geometry(cfg)
    if geom.n_valid_subaps != n_valid:
        # Our circular-pupil selection must reproduce the instrument's valid set.
        raise ValueError(
            f"geometry valid sub-aps {geom.n_valid_subaps} != telemetry {n_valid}; "
            "sub-aperture mask mismatch needs explicit handling"
        )

    # Reference spot positions: closed-loop residual is ~zero-mean, so the mean
    # calibrated frame's per-cell centroids are the flat-wavefront references.
    reference = _mean_frame_reference(frames.mean(axis=0), cfg, geom)

    timestamps = (frame_numbers - frame_numbers[0]) / loop_rate
    sequence = WFSFrameSequence(
        frames=frames.astype(np.float32),
        cfg=cfg,
        geometry=geom,
        reference_centroids=reference,
        timestamps_s=timestamps.astype(np.float64),
    )

    reported = _reported_atmosphere(sys)
    ref = AOTReference(
        stored_slopes=stored_slopes,
        stored_modal=stored_modal,
        dm_positions=dm_positions,
        frame_numbers=frame_numbers,
        loop_rate_hz=loop_rate,
        closed_loop=True,
        reported=reported,
    )
    return AOTDataset(sequence=sequence, reference=ref)


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #


def _count_grid(n_valid: int) -> int:
    """Map a valid-sub-aperture count to the full N*N grid size (corners excluded)."""
    # N x N with the 4 corners removed -> N^2 - 4 valid. Invert for small N.
    for N in range(2, 64):
        if N * N - 4 == n_valid:
            return N * N
        if N * N == n_valid:
            return N * N
    raise ValueError(f"cannot infer square grid for {n_valid} valid sub-apertures")


def _read_named_hdu(path: str, extname: str):
    """Read a non-standard telemetry ImageHDU by name -> (n_frames, n_chan) or None.

    FITS stores these transposed (NAXIS1 is the channel axis), so the natural
    numpy read already gives (n_frames, n_channels).
    """
    from astropy.io import fits
    with fits.open(path) as h:
        if extname not in h:
            return None
        return np.asarray(h[extname].data)


def _header_loop_rate(path: str, default: float) -> float:
    from astropy.io import fits
    with fits.open(path) as h:
        for hdu in h:
            for k in hdu.header:
                if "LOOP RATE" in str(k).upper():
                    return float(hdu.header[k])
    return default


def _build_config(D, n_subaps, det_px_per_subap, det_px, loop_rate) -> SimConfig:
    # Sensing wavelength set to the 500 nm seeing-reference wavelength so recovered
    # r0 is directly comparable to the ASM seeing-derived r0 (a reporting choice;
    # the NAOMI WFS itself is a visible broadband EMCCD).
    return SimConfig(
        pupil=PupilConfig(diameter_m=float(D), wavelength_m=0.5e-6),
        mla=MLAConfig(n_subaps=n_subaps, focal_length_m=5.0e-3, lenslet_pitch_m=200e-6),
        detector=DetectorConfig(
            pixel_size_m=24e-6, det_px_per_subap=det_px_per_subap,
            bit_depth=16, frame_dt_s=1.0 / loop_rate, well_depth_e=1e6,
        ),
        dm=DMConfig(coupling=0.3, max_stroke_m=4e-6),
        turbulence=TurbulenceConfig(r0_ref_m=0.19, r0_ref_wavelength_m=0.5e-6),
        noise=NoiseConfig(),
        n_frames=1,
        screen_px_per_subap=det_px_per_subap,
    )


def _mean_frame_reference(mean_frame, cfg, geom) -> np.ndarray:
    q = cfg.detector.det_px_per_subap
    idx = geom.valid_subap_index
    ax = np.arange(q, dtype=np.float64)
    out = np.empty((idx.shape[0], 2), dtype=np.float64)
    for k in range(idx.shape[0]):
        r, c = idx[k]
        cell = mean_frame[r * q:(r + 1) * q, c * q:(c + 1) * q].astype(np.float64)
        cell = np.clip(cell - 0.2 * cell.max(), 0.0, None)
        tot = cell.sum()
        if tot <= 0:
            out[k] = [c * q + (q - 1) / 2.0, r * q + (q - 1) / 2.0]
        else:
            out[k] = [c * q + (cell.sum(0) @ ax) / tot, r * q + (cell.sum(1) @ ax) / tot]
    return out


def _reported_atmosphere(sys) -> dict:
    def scalar(x):
        x = getattr(x, "data", x)
        a = np.asarray(x).ravel()
        return float(a[0]) if a.size else None

    aps = sys.atmosphere_params
    ap = aps[0] if isinstance(aps, list) and aps else aps
    seeing = scalar(ap.seeing) if ap is not None else None
    r0_500 = None
    if seeing:
        r0_500 = 0.98 * 500e-9 / np.radians(seeing / 3600.0)
    return {
        "seeing_arcsec": seeing,
        "r0_500nm_m": r0_500,
        "tau0_s": scalar(ap.tau0) if ap is not None else None,
        "wind_speed_ms": scalar(ap.layers_wind_speed) if ap is not None else None,
        "wind_dir_deg": scalar(ap.layers_wind_direction) if ap is not None else None,
        "atm_wavelength_m": scalar(ap.wavelength) if ap is not None else None,
    }
