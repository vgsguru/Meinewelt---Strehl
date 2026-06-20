"""Top-level synthetic SH-WFS dataset generator.

Ties the frozen-flow phase screen to the SH-WFS forward model and produces a
:class:`SyntheticDataset` -- a realistic, ground-truth-free
:class:`WFSFrameSequence` plus the quarantined :class:`GroundTruth` used only
for validation.
"""

from __future__ import annotations

import numpy as np

from ..config import SimConfig
from ..geometry import build_geometry
from ..types import GroundTruth, SyntheticDataset, WFSFrameSequence
from . import shwfs
from .phase_screen import FrozenFlowScreen


def generate_dataset(cfg: SimConfig, keep_truth: bool = True) -> SyntheticDataset:
    """Generate a full synthetic run from ``cfg``."""
    geom = build_geometry(cfg)
    screen = FrozenFlowScreen(cfg)
    gain = shwfs.calibrate_gain(cfg)
    rng = np.random.default_rng(cfg.turbulence.seed + 1)  # noise stream != screen stream

    det = cfg.detector_px
    dtype = np.uint8 if cfg.detector.bit_depth <= 8 else np.uint16
    frames = np.empty((cfg.n_frames, det, det), dtype=dtype)

    n_valid = geom.n_valid_subaps
    true_slopes = np.empty((cfg.n_frames, n_valid, 2), dtype=np.float64)
    screens = np.empty((cfg.n_frames, cfg.pupil_grid_px, cfg.pupil_grid_px), dtype=np.float32)

    pupil = geom.pupil_mask

    for i in range(cfg.n_frames):
        phase = screen.window(i)
        # Remove piston over the pupil so stored phase has zero mean there.
        phase = phase - phase[pupil].mean()
        masked = np.where(pupil, phase, 0.0)

        slopes = shwfs.subaperture_slopes(phase, cfg, geom)  # gradients over full blocks
        disp = shwfs.slopes_to_displacement_px(slopes, cfg)
        frames[i] = shwfs.render_spotfield(disp, cfg, geom, gain, rng=rng)

        true_slopes[i] = slopes
        screens[i] = masked.astype(np.float32)

    timestamps = np.arange(cfg.n_frames, dtype=np.float64) * cfg.detector.frame_dt_s
    sequence = WFSFrameSequence(
        frames=frames,
        cfg=cfg,
        geometry=geom,
        reference_centroids=shwfs.reference_centroids(cfg, geom),
        timestamps_s=timestamps,
    )

    truth = None
    if keep_truth:
        truth = GroundTruth(
            phase_screens=screens,
            true_slopes=true_slopes,
            r0_sensing_m=cfg.r0_sensing_m,
            tau0_s=cfg.tau0_s,
            wavelength_m=cfg.pupil.wavelength_m,
            wind_speed_ms=cfg.turbulence.wind_speed_ms,
            wind_dir_deg=cfg.turbulence.wind_dir_deg,
        )

    return SyntheticDataset(sequence=sequence, truth=truth)
