"""Turbulent phase-screen generation with frozen-flow temporal evolution.

We generate a single large von Karman phase screen with aotools'
subharmonic-augmented spectral method (``ft_sh_phase_screen``), which corrects
the well-known low-frequency power deficit of plain FFT screens -- important
for getting r0 right. Temporal evolution follows the Taylor frozen-flow
hypothesis: a pupil-sized window is translated across the big screen by the
wind vector each frame. The window is sampled with cubic interpolation so
sub-pixel wind shifts are represented faithfully and no FFT wrap-around
contaminates the edges.

Because the wind speed and r0 are known, the coherence time
tau0 = 0.314 * r0 / V is a real, recoverable property of the generated series
-- which is exactly what Phase 3 is validated against.
"""

from __future__ import annotations

import math

import numpy as np
from aotools.turbulence.phasescreen import ft_sh_phase_screen
from scipy.ndimage import map_coordinates

from ..config import SimConfig


class FrozenFlowScreen:
    """A large static screen sampled through a translating pupil window."""

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.pixel_scale_m = cfg.screen_pixel_scale_m
        self.window_px = cfg.pupil_grid_px

        # Sample the static screen so turbulence features translate across the
        # pupil along +wind_dir (intuitive convention). Stepping the sampling
        # window by -wind_vector each frame makes a fixed feature appear to move
        # in +wind_vector, matching how a real layer blows across the aperture.
        wdx, wdy = cfg.wind_vector_px_per_frame
        dx, dy = -wdx, -wdy
        self._step = np.array([dx, dy], dtype=float)
        n = cfg.n_frames

        # Cumulative travel over the run, in big-screen pixels, plus a margin
        # for the interpolation kernel and the window extent itself.
        travel_x = abs(dx) * max(n - 1, 0)
        travel_y = abs(dy) * max(n - 1, 0)
        margin = 4
        big_w = int(math.ceil(self.window_px + travel_x)) + 2 * margin
        big_h = int(math.ceil(self.window_px + travel_y)) + 2 * margin
        big = max(big_w, big_h)

        r0 = cfg.r0_sensing_m
        self.screen = ft_sh_phase_screen(
            r0=r0,
            N=big,
            delta=self.pixel_scale_m,
            L0=cfg.turbulence.L0_m,
            l0=cfg.turbulence.l0_m,
            seed=cfg.turbulence.seed,
        ).astype(np.float64)

        # Base window origin (top-left) so that, accounting for the full range
        # of positive/negative cumulative shifts, every frame's window stays
        # inside the big screen with the interpolation margin respected.
        min_off_x = min(0.0, dx * (n - 1))
        min_off_y = min(0.0, dy * (n - 1))
        self._origin = np.array(
            [margin - min_off_x, margin - min_off_y], dtype=float
        )

        # Precompute the base sampling grid (row, col) for the window once.
        rr, cc = np.mgrid[0 : self.window_px, 0 : self.window_px]
        self._base_rows = rr.astype(float)
        self._base_cols = cc.astype(float)

    def window(self, frame_idx: int) -> np.ndarray:
        """Phase screen seen through the pupil window at ``frame_idx`` [rad]."""
        off_x = self._origin[0] + self._step[0] * frame_idx
        off_y = self._origin[1] + self._step[1] * frame_idx
        cols = self._base_cols + off_x
        rows = self._base_rows + off_y
        sampled = map_coordinates(
            self.screen, [rows, cols], order=3, mode="nearest"
        )
        return sampled

    def all_windows(self) -> np.ndarray:
        """Stack of every frame's pupil window, shape (n_frames, P, P)."""
        return np.stack(
            [self.window(i) for i in range(self.cfg.n_frames)], axis=0
        )
