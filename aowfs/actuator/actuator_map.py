"""Convert a (predicted) wavefront into deformable-mirror actuator strokes.

The DM must apply the conjugate of the wavefront. We express the target as a
correction surface in length units (OPD), then solve for the stroke vector that
best reproduces it through the influence matrix:

    strokes = C @ target_surface,   C = pinv(H)   (truncated SVD)

The command matrix ``C`` is built ONCE; per frame the mapping is a single
matrix-vector multiply. Strokes are clipped to the mechanical limit and the
clipped fraction is reported. Validation (Phase 5) pushes the strokes back
through ``H`` and checks the reproduced surface matches the target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import SimConfig
from ..geometry import Geometry
from .influence_matrix import build_influence_matrix


@dataclass(frozen=True)
class ActuatorConfig:
    rcond: float = 1e-2  # SVD truncation for the command matrix (kills waffle blow-up)
    # Correction surface = reflection_factor * conjugate OPD. Use 1.0 for a
    # surface-equals-OPD convention, 0.5 for a reflective DM (double-pass),
    # per the DM spec.
    reflection_factor: float = 1.0


@dataclass(frozen=True)
class ActuatorResult:
    strokes: np.ndarray  # (n_acts,) commanded stroke [m], clipped
    raw_strokes: np.ndarray  # (n_acts,) before clipping [m]
    clipped_fraction: float
    fitting_error_m: float  # RMS(reproduced - target) over pupil [m]
    target_rms_m: float  # RMS of the target correction surface [m]

    @property
    def relative_fitting_error(self) -> float:
        return self.fitting_error_m / self.target_rms_m if self.target_rms_m > 0 else 0.0


class ActuatorMapper:
    """Calibrated wavefront-to-stroke mapper. Build once; ``command`` per frame."""

    def __init__(self, cfg: SimConfig, geom: Geometry, acfg: ActuatorConfig | None = None):
        self.cfg = cfg
        self.geom = geom
        self.acfg = acfg or ActuatorConfig()
        info = build_influence_matrix(cfg, geom)
        self.H = info["H"]  # (n_pix, n_acts)
        self.pupil_idx = info["pupil_idx"]
        self.act_px = info["act_px"]
        self.P = info["P"]
        self.n_acts = info["n_acts"]
        # Command matrix (built once).
        self.command_matrix = np.linalg.pinv(self.H, rcond=self.acfg.rcond)  # (n_acts, n_pix)
        self._opd_per_rad = cfg.pupil.wavelength_m / (2.0 * np.pi)
        self.max_stroke = cfg.dm.max_stroke_m

    # ------------------------------------------------------------------ #
    # surfaces                                                           #
    # ------------------------------------------------------------------ #

    def target_surface_flat(self, wavefront_rad: np.ndarray) -> np.ndarray:
        """Conjugate correction surface [m] sampled at pupil pixels (flat vector)."""
        w_pupil = wavefront_rad.ravel()[self.pupil_idx]
        return -self.acfg.reflection_factor * w_pupil * self._opd_per_rad

    def strokes_to_surface(self, strokes: np.ndarray) -> np.ndarray:
        """Mirror surface [m] on the P x P grid (0 outside pupil) from strokes."""
        surf = np.zeros(self.P * self.P, dtype=np.float64)
        surf[self.pupil_idx] = self.H @ strokes
        return surf.reshape(self.P, self.P)

    # ------------------------------------------------------------------ #
    # per-frame command (single matvec + clip)                           #
    # ------------------------------------------------------------------ #

    def command(self, wavefront_rad: np.ndarray) -> ActuatorResult:
        """Actuator stroke command [m] reproducing the conjugate of the wavefront."""
        target = self.target_surface_flat(wavefront_rad)
        raw = self.command_matrix @ target  # (n_acts,)
        strokes = np.clip(raw, -self.max_stroke, self.max_stroke)
        clipped = float(np.mean(np.abs(raw) > self.max_stroke))
        reproduced = self.H @ strokes
        fit_err = float(np.sqrt(np.mean((reproduced - target) ** 2)))
        target_rms = float(np.sqrt(np.mean(target ** 2)))
        return ActuatorResult(
            strokes=strokes,
            raw_strokes=raw,
            clipped_fraction=clipped,
            fitting_error_m=fit_err,
            target_rms_m=target_rms,
        )

    def stroke_grid(self, strokes: np.ndarray) -> np.ndarray:
        """Place a stroke vector on the (N+1)x(N+1) actuator grid (NaN if unused)."""
        n_side = self.cfg.dm.n_actuators_across(self.cfg.mla)
        grid = np.full((n_side, n_side), np.nan, dtype=np.float64)
        valid = self.geom.valid_act_mask
        grid[valid] = strokes
        return grid
