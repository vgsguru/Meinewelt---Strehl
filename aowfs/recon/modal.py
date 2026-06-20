"""Phase 2 (modal) -- Zernike wavefront reconstruction.

We build the slope-to-Zernike interaction matrix ``D`` by *poking*: each Noll
mode is pushed through the same sub-aperture sampling operator the WFS uses
(`optics.subaperture_gradients`), so the calibration is exactly consistent with
the measured slopes -- this is how a real AO system calibrates against its own
WFS. The reconstruction matrix ``R = pinv(D)`` (truncated SVD to suppress
poorly-sensed / noise-amplifying modes) is built ONCE. Per frame, reconstruction
is a single matrix-vector multiply ``a = R @ s`` plus an optional phase synthesis
``W = a @ basis`` (also a single matvec).

Slopes are wavefront gradients [rad/m]; coefficients and the phase map are in
radians. Piston (Noll j=1) is excluded because slopes cannot sense it.
"""

from __future__ import annotations

import numpy as np

from .. import optics
from ..config import SimConfig
from ..geometry import Geometry
from .zernike import zernike_basis


class ZernikeReconstructor:
    """Calibrated modal reconstructor. Build once; ``reconstruct`` per frame."""

    def __init__(
        self,
        cfg: SimConfig,
        geom: Geometry,
        n_modes: int = 36,
        rcond: float = 1e-3,
    ):
        self.cfg = cfg
        self.geom = geom
        self.n_modes = n_modes
        self.n_valid = geom.n_valid_subaps

        P = cfg.pupil_grid_px
        pupil_radius_px = P / 2.0
        self.basis, self.pupil_mask = zernike_basis(n_modes, P, pupil_radius_px)
        # Flatten basis over pupil pixels for fast phase synthesis (one matvec).
        self._pupil_idx = np.where(self.pupil_mask.ravel())[0]
        self._basis_pupil = self.basis.reshape(n_modes, -1)[:, self._pupil_idx]

        # --- calibration (done once) -------------------------------------- #
        # Interaction matrix D: column k = WFS slopes produced by unit mode k.
        D = np.empty((2 * self.n_valid, n_modes), dtype=np.float64)
        for k in range(n_modes):
            g = optics.subaperture_gradients(self.basis[k], cfg, geom)  # (n_valid, 2)
            D[:, k] = np.concatenate([g[:, 0], g[:, 1]])
        self.interaction = D
        # Truncated-SVD pseudo-inverse: the reconstruction matrix.
        self.recon_matrix = np.linalg.pinv(D, rcond=rcond)  # (n_modes, 2*n_valid)
        self._P = P

    # ------------------------------------------------------------------ #
    # per-frame hot path                                                 #
    # ------------------------------------------------------------------ #

    def coeffs(self, gradients: np.ndarray) -> np.ndarray:
        """Zernike coefficients [rad] from (n_valid, 2) gradients [rad/m]."""
        s = np.concatenate([gradients[:, 0], gradients[:, 1]])
        return self.recon_matrix @ s

    def phase_from_coeffs(self, a: np.ndarray) -> np.ndarray:
        """Synthesise the pupil phase map [rad] (0 outside pupil) from coeffs."""
        flat = np.zeros(self._P * self._P, dtype=np.float64)
        flat[self._pupil_idx] = a @ self._basis_pupil
        return flat.reshape(self._P, self._P)

    def reconstruct(self, gradients: np.ndarray) -> np.ndarray:
        """Full pupil phase map [rad] from gradients (coeffs + synthesis)."""
        return self.phase_from_coeffs(self.coeffs(gradients))

    def pupil_values(self, a: np.ndarray) -> np.ndarray:
        """Reconstructed phase sampled at pupil pixels only (flat vector)."""
        return a @ self._basis_pupil
