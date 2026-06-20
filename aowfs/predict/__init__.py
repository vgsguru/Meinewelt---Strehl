"""Phase 4 -- predictive wavefront control (the differentiator).

Per-mode linear (auto-regressive) forecasting of Zernike coefficients to cancel
servo-lag error, plus a zero-order-hold baseline for comparison. This is the
operationally-proven approach (Gemini ALTAIR used a ~5-coefficient linear
forecaster per mode); a neural predictor is treated as an evidence-gated
stretch, not a default.
"""

from __future__ import annotations

from .ar_predictor import LinearPredictor, NaivePredictor, PredictorConfig
from .evaluate import LagSweepResult, evaluate_lag_sweep

__all__ = [
    "LinearPredictor",
    "NaivePredictor",
    "PredictorConfig",
    "evaluate_lag_sweep",
    "LagSweepResult",
]
