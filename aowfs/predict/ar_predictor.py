"""Linear (auto-regressive) per-mode wavefront predictor.

For a loop delay of ``d`` frames, the controller must set frame t's correction
from data available only up to frame ``t - d``. The predictor forecasts each
Zernike coefficient d frames ahead from its recent history:

    a_hat_j(t) = sum_{k=0}^{p-1} w_{j,k} * a_j(t - d - k)

This is the d-step linear minimum-mean-square (Wiener) forecaster -- exactly
the ALTAIR-style linear forecasting module, with ~p coefficients per mode. The
weights are fit ONCE by least squares on a training segment (calibration); the
runtime prediction is one length-p dot product per mode (a single matvec across
modes), keeping it on the fast path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PredictorConfig:
    order: int = 5  # AR order p (number of past samples per mode; ALTAIR ~5)
    delay: int = 1  # loop delay in frames to forecast ahead

    def __post_init__(self) -> None:
        if self.order < 1:
            raise ValueError("order must be >= 1")
        if self.delay < 1:
            raise ValueError("delay must be >= 1")

    @property
    def min_history(self) -> int:
        """Frames of history required before a prediction can be made."""
        return self.delay + self.order


class LinearPredictor:
    """Per-mode d-step linear forecaster. Fit once, then predict per frame."""

    def __init__(self, n_modes: int, cfg: PredictorConfig | None = None):
        self.n_modes = n_modes
        self.cfg = cfg or PredictorConfig()
        self.weights = np.zeros((n_modes, self.cfg.order), dtype=np.float64)
        self._fitted = False

    # ------------------------------------------------------------------ #
    # calibration                                                        #
    # ------------------------------------------------------------------ #

    def fit(self, coeffs_train: np.ndarray, ridge: float = 1e-6) -> "LinearPredictor":
        """Fit per-mode forecaster weights on a training time series.

        ``coeffs_train`` is (n_train_frames, n_modes). A small ridge term
        stabilises modes whose history is nearly degenerate.
        """
        p, d = self.cfg.order, self.cfg.delay
        T = coeffs_train.shape[0]
        t0 = d + p - 1
        if T <= t0 + 1:
            raise ValueError("training series too short for this order/delay")
        # Design matrix is per-mode; build the lagged feature stack once.
        # features for target t: [a(t-d), a(t-d-1), ..., a(t-d-p+1)]
        targets = coeffs_train[t0:T]  # (n_samples, n_modes)
        n_samples = targets.shape[0]
        feats = np.empty((n_samples, self.n_modes, p), dtype=np.float64)
        for k in range(p):
            feats[:, :, k] = coeffs_train[t0 - d - k : T - d - k]
        # Solve a small ridge least-squares per mode: (X^T X + lam I) w = X^T y.
        for j in range(self.n_modes):
            X = feats[:, j, :]
            y = targets[:, j]
            A = X.T @ X + ridge * np.eye(p)
            b = X.T @ y
            self.weights[j] = np.linalg.solve(A, b)
        self._fitted = True
        return self

    # ------------------------------------------------------------------ #
    # runtime                                                            #
    # ------------------------------------------------------------------ #

    def predict_one(self, history: np.ndarray) -> np.ndarray:
        """Forecast a_hat(t) given history ending at the latest usable frame.

        ``history`` is (>=min_history, n_modes); the most recent row is the
        latest measurement. Returns (n_modes,).
        """
        p, d = self.cfg.order, self.cfg.delay
        # Most recent usable sample is at index -d; then step back p samples.
        feat = history[-d : -d - p : -1] if d + p <= history.shape[0] else None
        if feat is None:
            raise ValueError("insufficient history for prediction")
        # feat[k] = a(t-d-k); weights[:,k] -> sum_k w*feat
        return np.einsum("mk,km->m", self.weights, feat)

    def predict_sequence(self, coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forecast a_hat(t) for every frame with enough history.

        Returns ``(predictions, t_indices)`` where ``predictions[i]`` is the
        forecast for frame ``t_indices[i]`` made from data up to t-delay.
        """
        p, d = self.cfg.order, self.cfg.delay
        T = coeffs.shape[0]
        t0 = d + p - 1
        idx = np.arange(t0, T)
        preds = np.zeros((idx.size, self.n_modes), dtype=np.float64)
        for k in range(p):
            preds += coeffs[t0 - d - k : T - d - k] * self.weights[:, k]
        return preds, idx


class NaivePredictor:
    """Zero-order hold: a_hat(t) = a(t - delay). What an integrator effectively uses."""

    def __init__(self, delay: int = 1):
        self.delay = delay

    def predict_sequence(self, coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d = self.delay
        T = coeffs.shape[0]
        idx = np.arange(d, T)
        return coeffs[: T - d], idx
