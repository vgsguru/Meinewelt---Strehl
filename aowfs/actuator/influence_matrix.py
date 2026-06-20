"""Deformable-mirror influence functions and the forward influence matrix.

Each actuator deforms the mirror with a localised influence function (IF). We
use the standard Gaussian-coupling model in which the IF width is fixed by the
inter-actuator coupling coefficient ``c`` from the DM spec:

    IF(r) = exp( ln(c) * (r / d_act)^2 )

so a unit poke gives 1.0 at the actuator and exactly ``c`` at the nearest
neighbour (r = d_act), with the Gaussian tail beyond. The forward influence
matrix ``H`` maps an actuator stroke vector to the mirror surface sampled over
the pupil:  surface = H @ strokes.
"""

from __future__ import annotations

import numpy as np

from ..config import SimConfig
from ..geometry import Geometry


def influence_function(r_over_pitch: np.ndarray, coupling: float) -> np.ndarray:
    """Gaussian influence function value at distance ``r/d_act`` for ``coupling``."""
    return np.exp(np.log(coupling) * r_over_pitch ** 2)


def build_influence_matrix(cfg: SimConfig, geom: Geometry):
    """Build the pupil-sampled influence matrix and supporting geometry.

    Returns a dict with:
      ``H``          : (n_pupil_px, n_valid_acts) surface per unit stroke
      ``pupil_idx``  : flat indices of pupil pixels into the P x P grid
      ``act_px``     : (n_valid_acts, 2) actuator positions in detector-grid px (x, y)
      ``d_act_px``   : actuator pitch in phase-grid pixels
      ``P``          : phase-grid side length
    """
    P = cfg.pupil_grid_px
    p = cfg.screen_px_per_subap  # phase-grid px per sub-aperture (= actuator pitch)
    d_act_px = float(p)
    coupling = cfg.dm.coupling

    # Valid actuator positions (subap-pitch units, centred) -> phase-grid px.
    centre = (P - 1) / 2.0
    valid = geom.valid_act_mask.ravel()
    act_xy = geom.actuator_positions[valid]  # (n_act, 2) in subap-pitch units
    act_px = centre + act_xy * p  # (n_act, 2) x, y in grid px

    # Pupil pixel coordinates.
    pupil_idx = np.where(geom.pupil_mask.ravel())[0]
    yy, xx = np.divmod(pupil_idx, P)
    px = np.column_stack([xx, yy]).astype(np.float64)  # (n_pix, 2)

    # Distance from every pupil pixel to every actuator, in pitch units.
    dx = px[:, 0:1] - act_px[:, 0][None, :]
    dy = px[:, 1:2] - act_px[:, 1][None, :]
    r_over_pitch = np.sqrt(dx ** 2 + dy ** 2) / d_act_px
    H = influence_function(r_over_pitch, coupling)  # (n_pix, n_act)

    return {
        "H": H,
        "pupil_idx": pupil_idx,
        "act_px": act_px,
        "d_act_px": d_act_px,
        "P": P,
        "n_acts": act_px.shape[0],
    }
