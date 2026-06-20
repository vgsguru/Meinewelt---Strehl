"""Adapter for ISRO's real SH-WFS dataset.

Phases 1-6 consume a :class:`~aowfs.types.WFSFrameSequence`; this module produces
exactly that from the real deliverable -- a directory of ``.bmp`` frames plus the
hardware specs from the brief's "Data Required" list -- so nothing downstream
changes. Only this file is real-data-aware.

The brief supplies: detector pixel size + frame resolution; MLA size / lenslet
count / focal length; pupil diameter; DM coupling. Those map one-to-one onto
:class:`ISROSpecs`, which builds the internal :class:`~aowfs.config.SimConfig`
(the turbulence/noise fields are placeholders -- the algorithms never read
them; they use only geometry and optics).

Two real-data concerns this adapter handles explicitly:

  * **Spot-grid registration.** The illuminated N x N spot pattern occupies
    ``N * det_px_per_subap`` detector pixels; if the frame is larger we crop to
    that grid (centred, or at a supplied origin). ``det_px_per_subap`` defaults
    to ``round(lenslet_pitch / pixel_size)`` -- the physical spot spacing.

  * **Reference (flat-wavefront) positions.** Real slopes are displacements from
    a calibrated reference. Provide a flat-illumination frame
    (``reference_mode="frame"``); else use the per-cell geometric centres
    (``"geometric"``) or the time-averaged spot positions (``"mean"``, valid for
    zero-mean turbulence).
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import numpy as np
import yaml
from PIL import Image

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


@dataclass(frozen=True)
class ISROSpecs:
    """Hardware parameters supplied with the ISRO dataset (the brief's list)."""

    pixel_size_m: float
    mla_n_lenslets: int  # N sub-apertures across the pupil (assumed N x N)
    mla_lenslet_pitch_m: float
    mla_focal_length_m: float
    pupil_diameter_m: float
    wavelength_m: float
    frame_dt_s: float
    dm_coupling: float
    dm_max_stroke_m: float = 4.0e-6
    bit_depth: int = 8
    # Detector pixels per spot cell. Default: physical spot spacing on the
    # detector = lenslet pitch / pixel size.
    det_px_per_subap: int | None = None
    # Internal reconstructed-phase-map sampling (synthesis-only; not measured).
    screen_px_per_subap: int = 8
    frame_resolution: tuple[int, int] | None = None  # (H, W), for a sanity check

    def derived_det_px_per_subap(self) -> int:
        if self.det_px_per_subap is not None:
            return int(self.det_px_per_subap)
        return int(round(self.mla_lenslet_pitch_m / self.pixel_size_m))

    def to_config(self, n_frames: int = 1) -> SimConfig:
        q = self.derived_det_px_per_subap()
        return SimConfig(
            pupil=PupilConfig(diameter_m=self.pupil_diameter_m, wavelength_m=self.wavelength_m),
            mla=MLAConfig(
                n_subaps=self.mla_n_lenslets,
                focal_length_m=self.mla_focal_length_m,
                lenslet_pitch_m=self.mla_lenslet_pitch_m,
            ),
            detector=DetectorConfig(
                pixel_size_m=self.pixel_size_m,
                det_px_per_subap=q,
                bit_depth=self.bit_depth,
                frame_dt_s=self.frame_dt_s,
            ),
            dm=DMConfig(coupling=self.dm_coupling, max_stroke_m=self.dm_max_stroke_m),
            # Placeholders -- not used by any estimator on real data.
            turbulence=TurbulenceConfig(),
            noise=NoiseConfig(),
            n_frames=max(n_frames, 1),
            screen_px_per_subap=self.screen_px_per_subap,
        )

    @classmethod
    def from_yaml(cls, path: str) -> "ISROSpecs":
        with open(path, "r", encoding="utf-8") as fh:
            d = yaml.safe_load(fh)
        if isinstance(d.get("frame_resolution"), list):
            d["frame_resolution"] = tuple(d["frame_resolution"])
        return cls(**d)


# --------------------------------------------------------------------------- #
# loading                                                                      #
# --------------------------------------------------------------------------- #


def _read_gray(path: str) -> np.ndarray:
    img = np.asarray(Image.open(path))
    if img.ndim == 3:  # RGB(A) -> luminance
        img = img[..., :3].mean(axis=2)
    return img


def _crop_to_grid(frame: np.ndarray, grid_px: int, origin: tuple[int, int] | None) -> np.ndarray:
    """Crop a frame to the N*q spot grid (centred, or at a given (row, col))."""
    H, W = frame.shape
    if origin is None:
        r0 = (H - grid_px) // 2
        c0 = (W - grid_px) // 2
    else:
        r0, c0 = origin
    if r0 < 0 or c0 < 0 or r0 + grid_px > H or c0 + grid_px > W:
        raise ValueError(
            f"spot grid {grid_px}x{grid_px} at origin ({r0},{c0}) does not fit in "
            f"frame {H}x{W}; check det_px_per_subap / roi_origin"
        )
    return frame[r0 : r0 + grid_px, c0 : c0 + grid_px]


def _absolute_cell_centroids(frame: np.ndarray, geom, q: int, thr_frac: float = 0.2) -> np.ndarray:
    """Per-valid-cell centre-of-mass (absolute detector px), for the reference."""
    idx = geom.valid_subap_index
    ax = np.arange(q, dtype=np.float64)
    out = np.empty((idx.shape[0], 2), dtype=np.float64)
    for k in range(idx.shape[0]):
        r, c = idx[k]
        cell = frame[r * q : (r + 1) * q, c * q : (c + 1) * q].astype(np.float64)
        cell = np.clip(cell - thr_frac * cell.max(), 0.0, None)
        tot = cell.sum()
        if tot <= 0:
            out[k] = [c * q + (q - 1) / 2.0, r * q + (q - 1) / 2.0]
            continue
        out[k] = [c * q + (cell.sum(0) @ ax) / tot, r * q + (cell.sum(1) @ ax) / tot]
    return out


def load_isro_dataset(
    frames_dir: str,
    specs: ISROSpecs,
    *,
    pattern: str = "*.bmp",
    reference_mode: str = "geometric",
    reference_frame_path: str | None = None,
    roi_origin: tuple[int, int] | None = None,
) -> WFSFrameSequence:
    """Load an ISRO ``.bmp`` frame series into a :class:`WFSFrameSequence`.

    ``reference_mode``: ``"geometric"`` (cell centres), ``"mean"`` (time-averaged
    spot positions), or ``"frame"`` (centroid a supplied flat frame at
    ``reference_frame_path``).
    """
    paths = sorted(glob.glob(os.path.join(frames_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"no frames matching {pattern!r} in {frames_dir}")

    cfg = specs.to_config(n_frames=len(paths))
    geom = build_geometry(cfg)
    q = cfg.detector.det_px_per_subap
    grid_px = cfg.detector_px  # N * q

    raw0 = _read_gray(paths[0])
    if specs.frame_resolution is not None and raw0.shape != tuple(specs.frame_resolution):
        raise ValueError(
            f"frame resolution {raw0.shape} != declared {tuple(specs.frame_resolution)}"
        )

    dtype = np.uint8 if cfg.detector.bit_depth <= 8 else np.uint16
    frames = np.empty((len(paths), grid_px, grid_px), dtype=dtype)
    for i, p in enumerate(paths):
        frames[i] = _crop_to_grid(_read_gray(p), grid_px, roi_origin).astype(dtype)

    # Reference (flat-wavefront) spot positions.
    if reference_mode == "frame":
        if reference_frame_path is None:
            raise ValueError("reference_mode='frame' needs reference_frame_path")
        flat = _crop_to_grid(_read_gray(reference_frame_path), grid_px, roi_origin)
        reference = _absolute_cell_centroids(flat, geom, q)
    elif reference_mode == "mean":
        reference = _absolute_cell_centroids(frames.mean(axis=0), geom, q)
    elif reference_mode == "geometric":
        idx = geom.valid_subap_index
        reference = np.column_stack([
            idx[:, 1] * q + (q - 1) / 2.0,
            idx[:, 0] * q + (q - 1) / 2.0,
        ]).astype(np.float64)
    else:
        raise ValueError(f"unknown reference_mode {reference_mode!r}")

    timestamps = np.arange(len(paths), dtype=np.float64) * cfg.detector.frame_dt_s
    return WFSFrameSequence(
        frames=frames,
        cfg=cfg,
        geometry=geom,
        reference_centroids=reference,
        timestamps_s=timestamps,
    )
