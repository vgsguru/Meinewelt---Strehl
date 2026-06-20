"""Fried-geometry grid definitions shared by every stage of the pipeline.

The MLA lenslet grid and the DM actuator grid are arranged in a Fried
geometry: lenslets tile an N x N grid over the pupil, and actuators sit on the
(N+1) x (N+1) corners of those sub-apertures. Reconstruction (Southwell),
turbulence statistics, and actuator mapping all index into the structures
built here, so the grid is defined exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SimConfig


@dataclass(frozen=True)
class Geometry:
    """Precomputed grids and masks for a given configuration.

    Coordinates are in units of sub-aperture pitch, with the pupil centred on
    the origin. ``valid_subap`` selects the lenslets whose centres fall inside
    the circular pupil; only those carry usable slope measurements.
    """

    n_subaps: int
    n_acts: int
    subap_centres: np.ndarray  # (n_valid, 2) x,y in subap-pitch units
    valid_subap_mask: np.ndarray  # (N, N) bool
    valid_subap_index: np.ndarray  # (n_valid, 2) int row,col into the N x N grid
    actuator_positions: np.ndarray  # (n_acts_total, 2) x,y in subap-pitch units
    valid_act_mask: np.ndarray  # (N+1, N+1) bool
    pupil_mask: np.ndarray  # (pupil_grid_px, pupil_grid_px) bool, phase-screen sampling

    @property
    def n_valid_subaps(self) -> int:
        return int(self.valid_subap_mask.sum())

    @property
    def n_valid_acts(self) -> int:
        return int(self.valid_act_mask.sum())


def build_geometry(cfg: SimConfig) -> Geometry:
    """Construct the Fried geometry for ``cfg``."""
    n = cfg.mla.n_subaps
    n_act = cfg.dm.n_actuators_across(cfg.mla)

    # Sub-aperture centres on an N x N grid, centred on the pupil. In
    # subap-pitch units the centres run from -(N-1)/2 .. +(N-1)/2.
    coords = (np.arange(n) - (n - 1) / 2.0)
    sx, sy = np.meshgrid(coords, coords, indexing="xy")
    radius_subaps = n / 2.0
    # A lenslet is "valid" if its centre lies within the pupil radius. Use the
    # centre-distance test (standard Fried-grid sub-aperture selection).
    rr = np.sqrt(sx ** 2 + sy ** 2)
    valid_subap = rr <= radius_subaps
    # row, col indices (row = y index, col = x index) of valid sub-apertures
    rows, cols = np.where(valid_subap)
    valid_idx = np.column_stack([rows, cols])
    subap_centres = np.column_stack([sx[rows, cols], sy[rows, cols]])

    # Actuators on the (N+1) x (N+1) corner grid, same centred coordinate frame.
    acoords = (np.arange(n_act) - (n_act - 1) / 2.0)
    ax, ay = np.meshgrid(acoords, acoords, indexing="xy")
    arr = np.sqrt(ax ** 2 + ay ** 2)
    # Actuators within half a pitch of the pupil edge still influence the
    # corrected area, so include the slightly larger radius.
    valid_act = arr <= (radius_subaps + 0.5)
    actuator_positions = np.column_stack([ax.ravel(), ay.ravel()])

    pupil_mask = _circular_pupil_mask(cfg.pupil_grid_px)

    return Geometry(
        n_subaps=n,
        n_acts=n_act,
        subap_centres=subap_centres,
        valid_subap_mask=valid_subap,
        valid_subap_index=valid_idx,
        actuator_positions=actuator_positions,
        valid_act_mask=valid_act,
        pupil_mask=pupil_mask,
    )


def _circular_pupil_mask(grid_px: int) -> np.ndarray:
    """Boolean circular pupil on a square phase-screen grid."""
    c = (grid_px - 1) / 2.0
    y, x = np.ogrid[:grid_px, :grid_px]
    r = np.sqrt((x - c) ** 2 + (y - c) ** 2)
    return r <= (grid_px / 2.0)
