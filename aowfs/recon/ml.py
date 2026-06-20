"""Phase 9 -- machine-learning wavefront reconstruction (the fourth method).

This maps slopes directly to Zernike coefficients via a *trained* model, as
opposed to the Phase 4 predictive layer (which forecasts in time). We start with
ridge regression: a learned reconstruction matrix

    W = argmin_W  ||S Wᵀ − A||²  +  α ||W||²        =>   coeffs = W @ slopes

fit on (slopes, coefficients) pairs with an honest train/test split. Trained on
*noisy* slopes against *clean* target coefficients, ridge is a regularised
estimator that can trade a little bias for variance (noise rejection) versus the
analytic pseudo-inverse used by the modal reconstructor.

Per-frame inference is a single matrix-vector multiply -- the same cost as the
analytic modal matrix -- so this is an apples-to-apples comparison. A neural
(MLP) variant is only justified if ridge leaves structured error on this
well-conditioned linear problem; see the Phase 9 comparison note.
"""

from __future__ import annotations

import numpy as np

from ..config import SimConfig
from ..geometry import Geometry
from .zernike import zernike_basis


class MLReconstructor:
    """Ridge-regression slopes->Zernike reconstructor. Fit once, then a matvec."""

    def __init__(self, cfg: SimConfig, geom: Geometry, n_modes: int = 45, alpha: float = 1.0):
        self.cfg = cfg
        self.geom = geom
        self.n_modes = n_modes
        self.alpha = alpha
        self.n_valid = geom.n_valid_subaps

        P = cfg.pupil_grid_px
        self.basis, self.pupil_mask = zernike_basis(n_modes, P, P / 2.0)
        self._pupil_idx = np.where(self.pupil_mask.ravel())[0]
        self._basis_pupil = self.basis.reshape(n_modes, -1)[:, self._pupil_idx]
        self._P = P
        self.W: np.ndarray | None = None  # learned (n_modes, 2*n_valid)

    # ------------------------------------------------------------------ #
    # training (calibration, done once)                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _flatten(g: np.ndarray) -> np.ndarray:
        if g.ndim == 3:  # (T, n_valid, 2) -> (T, 2*n_valid)
            return np.concatenate([g[:, :, 0], g[:, :, 1]], axis=1)
        return np.concatenate([g[:, 0], g[:, 1]])  # (n_valid, 2) -> (2*n_valid,)

    def fit(
        self,
        gradients_train: np.ndarray,
        target_coeffs_train: np.ndarray,
        alpha: float | None = None,
    ) -> "MLReconstructor":
        """Learn W from (gradients, target coefficients) pairs via ridge."""
        a = self.alpha if alpha is None else alpha
        S = self._flatten(gradients_train)  # (T, 2*n_valid)
        A = target_coeffs_train  # (T, n_modes)
        gram = S.T @ S + a * np.eye(S.shape[1])
        self.W = np.linalg.solve(gram, S.T @ A).T  # (n_modes, 2*n_valid)
        return self

    # ------------------------------------------------------------------ #
    # per-frame inference (single matvec)                                #
    # ------------------------------------------------------------------ #

    @property
    def recon_matrix(self) -> np.ndarray:
        if self.W is None:
            raise RuntimeError("MLReconstructor not fitted")
        return self.W

    def coeffs(self, gradients: np.ndarray) -> np.ndarray:
        return self.recon_matrix @ self._flatten(gradients)

    def phase_from_coeffs(self, a: np.ndarray) -> np.ndarray:
        flat = np.zeros(self._P * self._P, dtype=np.float64)
        flat[self._pupil_idx] = a @ self._basis_pupil
        return flat.reshape(self._P, self._P)

    def reconstruct(self, gradients: np.ndarray) -> np.ndarray:
        return self.phase_from_coeffs(self.coeffs(gradients))
