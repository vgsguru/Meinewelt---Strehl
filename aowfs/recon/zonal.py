"""Phase 2 (zonal) -- Fried-geometry wavefront reconstruction.

In the Fried configuration the phase/actuator points sit on the (N+1)x(N+1)
corners of the N x N sub-aperture grid, and each lenslet's mean slope relates
the four surrounding corner phases:

    sx = [(phi(i,  j+1) + phi(i+1, j+1)) - (phi(i, j) + phi(i+1, j))] / (2 d)
    sy = [(phi(i+1, j) + phi(i+1, j+1)) - (phi(i, j) + phi(i,   j+1))] / (2 d)

with d the sub-aperture pitch. Stacking these for every valid sub-aperture
gives the geometry matrix G (2*n_subaps x n_corners). The reconstruction matrix
R = pinv(G) (truncated SVD, which removes the unobservable piston null space)
is built ONCE; per frame, phi = R @ s is a single matrix-vector multiply.

Output corner phases live on the DM actuator grid (Fried), feeding Phase 5
directly.
"""

from __future__ import annotations

import numpy as np

from ..config import SimConfig
from ..geometry import Geometry


class FriedZonalReconstructor:
    """Calibrated Fried zonal reconstructor. Build once; ``reconstruct`` per frame."""

    def __init__(self, cfg: SimConfig, geom: Geometry, rcond: float = 1e-3):
        self.cfg = cfg
        self.geom = geom
        self.n_valid = geom.n_valid_subaps
        self.N = cfg.mla.n_subaps
        self.n_corners_side = self.N + 1
        d = cfg.subap_pitch_m

        # Valid corners = corners touching at least one valid sub-aperture.
        corner_used = np.zeros((self.N + 1, self.N + 1), dtype=bool)
        idx = geom.valid_subap_index
        for (i, j) in idx:
            corner_used[i, j] = corner_used[i, j + 1] = True
            corner_used[i + 1, j] = corner_used[i + 1, j + 1] = True
        self.corner_mask = corner_used
        cr, cc = np.where(corner_used)
        self.corner_index = np.column_stack([cr, cc])  # (n_corners, 2)
        # Lookup from (row, col) corner -> column in G.
        lut = -np.ones((self.N + 1, self.N + 1), dtype=int)
        lut[cr, cc] = np.arange(cr.size)
        self.n_corners = cr.size

        # --- calibration (done once): build the geometry matrix G --------- #
        G = np.zeros((2 * self.n_valid, self.n_corners), dtype=np.float64)
        inv2d = 1.0 / (2.0 * d)
        for k, (i, j) in enumerate(idx):
            c00, c01 = lut[i, j], lut[i, j + 1]
            c10, c11 = lut[i + 1, j], lut[i + 1, j + 1]
            # x-slope row
            G[k, c01] += inv2d
            G[k, c11] += inv2d
            G[k, c00] -= inv2d
            G[k, c10] -= inv2d
            # y-slope row
            r = self.n_valid + k
            G[r, c10] += inv2d
            G[r, c11] += inv2d
            G[r, c00] -= inv2d
            G[r, c01] -= inv2d
        self.geometry_matrix = G
        self.recon_matrix = np.linalg.pinv(G, rcond=rcond)  # (n_corners, 2*n_valid)

    # ------------------------------------------------------------------ #
    # per-frame hot path                                                 #
    # ------------------------------------------------------------------ #

    def reconstruct(self, gradients: np.ndarray) -> np.ndarray:
        """Corner-grid phases [rad] (piston-removed) from (n_valid, 2) gradients."""
        s = np.concatenate([gradients[:, 0], gradients[:, 1]])
        phi = self.recon_matrix @ s
        return phi - phi.mean()  # fix the gauge (piston) for a stable comparison

    def to_grid(self, phi: np.ndarray) -> np.ndarray:
        """Place a corner-phase vector onto the (N+1)x(N+1) grid (NaN elsewhere)."""
        grid = np.full((self.N + 1, self.N + 1), np.nan, dtype=np.float64)
        cr, cc = self.corner_index[:, 0], self.corner_index[:, 1]
        grid[cr, cc] = phi
        return grid
