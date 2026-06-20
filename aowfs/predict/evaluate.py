"""Quantify the predictive-control benefit: residual wavefront error and Strehl,
with vs. without prediction, swept over loop delay.

Residual variance is summed over Noll-normalised Zernike modes, which by
Parseval equals the pupil-averaged residual phase variance [rad^2]; the implied
Strehl follows the Marechal approximation S = exp(-sigma^2).

The predictor weights are fit on a training segment and evaluated on a disjoint
test segment, so the reported benefit contains no look-ahead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ar_predictor import LinearPredictor, NaivePredictor, PredictorConfig


@dataclass(frozen=True)
class LagSweepResult:
    delays: np.ndarray
    naive_var: np.ndarray  # residual phase variance [rad^2]
    pred_var: np.ndarray
    naive_strehl: np.ndarray
    pred_strehl: np.ndarray
    reduction_pct: np.ndarray  # % reduction in residual variance

    def as_table(self) -> str:
        lines = [
            f"{'delay':>6}{'naive var':>11}{'pred var':>10}{'reduction':>11}"
            f"{'naive Strehl':>14}{'pred Strehl':>13}"
        ]
        for i, d in enumerate(self.delays):
            lines.append(
                f"{d:>6}{self.naive_var[i]:>11.4f}{self.pred_var[i]:>10.4f}"
                f"{self.reduction_pct[i]:>10.1f}%{self.naive_strehl[i]:>14.3f}"
                f"{self.pred_strehl[i]:>13.3f}"
            )
        return "\n".join(lines)


def evaluate_lag_sweep(
    coeffs_input: np.ndarray,
    coeffs_target: np.ndarray,
    delays=(1, 2, 3, 4),
    order: int = 5,
    train_frac: float = 0.5,
    ridge: float = 1e-6,
) -> LagSweepResult:
    """Residual variance / Strehl for naive vs linear prediction over delays.

    ``coeffs_input``  : (T, n_modes) coefficients the controller sees (measured).
    ``coeffs_target`` : (T, n_modes) the true coefficients to be corrected.
    For an isolated prediction benefit pass the same (true) array for both.
    """
    T, n_modes = coeffs_input.shape
    split = int(T * train_frac)
    delays = list(delays)

    nv, pv = [], []
    for d in delays:
        cfg = PredictorConfig(order=order, delay=d)
        pred = LinearPredictor(n_modes, cfg).fit(coeffs_input[:split], ridge=ridge)

        preds, idx = pred.predict_sequence(coeffs_input)
        test = idx >= split
        ti = idx[test]
        res_pred = coeffs_target[ti] - preds[test]
        pv.append(float(np.mean(np.sum(res_pred ** 2, axis=1))))

        # Naive: zero-order hold over the same test frames.
        res_naive = coeffs_target[ti] - coeffs_input[ti - d]
        nv.append(float(np.mean(np.sum(res_naive ** 2, axis=1))))

    nv, pv = np.array(nv), np.array(pv)
    return LagSweepResult(
        delays=np.array(delays),
        naive_var=nv,
        pred_var=pv,
        naive_strehl=np.exp(-nv),
        pred_strehl=np.exp(-pv),
        reduction_pct=100.0 * (nv - pv) / nv,
    )
