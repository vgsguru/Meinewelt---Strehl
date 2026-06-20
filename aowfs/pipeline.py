"""Real-time AO pipeline: detector frame -> deformable-mirror stroke command.

This is where the architecture pays off. Every calibration matrix is built once
in ``__init__``; the per-frame :meth:`step` collapses to a centroiding pass plus
three small matrix-vector multiplies, because the linear chain
(reconstruct -> synthesise phase -> conjugate -> project onto actuators) folds
into precomputed matrices:

    coeffs   = R_px @ displacement_flat          # slopes -> Zernike coefficients
    a_hat    = predictor(coeff_history)          # forecast `delay` frames ahead
    strokes  = clip( M_act @ a_hat )             # coefficients -> actuator strokes

``R_px`` has the displacement->gradient factor baked in; ``M_act`` folds the
wavefront synthesis, the conjugation, and the DM command matrix into one
(n_acts x n_modes) matrix. Nothing in the loop re-derives a calibration matrix.
"""

from __future__ import annotations

import numpy as np

from .actuator.actuator_map import ActuatorConfig, ActuatorMapper
from .config import SimConfig
from .geometry import Geometry
from .optics import disp_factor
from .predict.ar_predictor import LinearPredictor, PredictorConfig
from .recon.centroiding import CentroidConfig, Centroider
from .recon.fast_centroid import FastCentroider
from .recon.modal import ZernikeReconstructor


class RealTimePipeline:
    """Calibrated AO loop. Build once; call :meth:`step` per frame."""

    def __init__(
        self,
        cfg: SimConfig,
        geom: Geometry,
        reference_centroids: np.ndarray,
        *,
        n_modes: int = 45,
        predictor_cfg: PredictorConfig | None = None,
        actuator_cfg: ActuatorConfig | None = None,
        centroid_threshold: float = 0.2,
        use_numba: bool = True,
    ):
        self.cfg = cfg
        self.geom = geom
        self.n_modes = n_modes
        self.n_valid = geom.n_valid_subaps

        # --- calibration (once) ------------------------------------------ #
        self.use_numba = use_numba
        if use_numba:
            self.centroider = FastCentroider(cfg, geom, reference_centroids, centroid_threshold)
        else:
            self.centroider = Centroider(
                cfg, geom, reference_centroids,
                CentroidConfig(method="com", threshold_frac=centroid_threshold),
            )

        rec = ZernikeReconstructor(cfg, geom, n_modes=n_modes)
        mapper = ActuatorMapper(cfg, geom, actuator_cfg)

        # Fold the displacement->gradient factor into the reconstruction matrix.
        self.R_px = rec.recon_matrix / disp_factor(cfg)  # (n_modes, 2*n_valid)
        # Fold phase synthesis + conjugation + DM command into one matrix.
        opd_per_rad = cfg.pupil.wavelength_m / (2.0 * np.pi)
        rf = mapper.acfg.reflection_factor
        self.M_act = (-rf * opd_per_rad) * (mapper.command_matrix @ rec._basis_pupil.T)
        self.max_stroke = cfg.dm.max_stroke_m
        self.n_acts = self.M_act.shape[0]

        # --- DIRECT GRADIENT CONTROL --------------------------------------- #
        # Composing the two precomputed matrices yields a single slopes ->
        # actuator control matrix. The reactive (non-predictive) per-frame path
        # is then literally ONE matrix-vector multiply on the raw centroid
        # displacements -- this *is* the "direct gradient control" method named
        # in the PS9 explainer (the four being zonal / modal / direct gradient
        # control / ML), and is how SPHERE/Keck-class systems run their fast
        # path. We already had it folded in; ``control_matrix`` names it.
        self.control_matrix = self.M_act @ self.R_px  # (n_acts, 2 * n_valid)

        # Keep component handles for diagnostics / the dashboard.
        self._rec = rec
        self._mapper = mapper

        # --- predictor + rolling history --------------------------------- #
        self.predictor: LinearPredictor | None = None
        self.pcfg = predictor_cfg
        if predictor_cfg is not None:
            self.predictor = LinearPredictor(n_modes, predictor_cfg)
            self._hist_len = predictor_cfg.min_history
        else:
            self._hist_len = 1
        self._hist = np.zeros((self._hist_len, n_modes), dtype=np.float64)
        self._n_seen = 0

    # ------------------------------------------------------------------ #
    # calibration of the predictor (training data, done once)            #
    # ------------------------------------------------------------------ #

    def fit_predictor(self, coeffs_train: np.ndarray, ridge: float = 1e-6) -> None:
        if self.predictor is None:
            raise RuntimeError("pipeline built without a predictor_cfg")
        self.predictor.fit(coeffs_train, ridge=ridge)

    def reset(self) -> None:
        self._hist[:] = 0.0
        self._n_seen = 0

    # ------------------------------------------------------------------ #
    # per-frame runtime path                                             #
    # ------------------------------------------------------------------ #

    def coeffs_from_frame(self, frame: np.ndarray) -> np.ndarray:
        """Centroid + reconstruct to Zernike coefficients (no prediction)."""
        disp = self.centroider.measure(frame)
        disp_flat = np.concatenate([disp[:, 0], disp[:, 1]])
        return self.R_px @ disp_flat

    def direct_command(self, frame: np.ndarray) -> np.ndarray:
        """Direct gradient control: centroids -> strokes in ONE matvec.

        The reactive fast path. ``control_matrix`` already folds reconstruction
        and actuator projection together (see __init__), so this is the
        slopes-to-actuator control matrix applied directly -- identical result
        to ``step(frame, predict=False)`` but making the single-matvec nature
        explicit. This is the "direct gradient control" reconstruction method.
        """
        disp = self.centroider.measure(frame)
        disp_flat = np.concatenate([disp[:, 0], disp[:, 1]])
        strokes = self.control_matrix @ disp_flat
        np.clip(strokes, -self.max_stroke, self.max_stroke, out=strokes)
        return strokes

    def step(self, frame: np.ndarray, predict: bool = True) -> np.ndarray:
        """One real-time iteration: detector frame -> actuator stroke command [m]."""
        a = self.coeffs_from_frame(frame)

        if predict and self.predictor is not None:
            # roll history, append newest
            self._hist[:-1] = self._hist[1:]
            self._hist[-1] = a
            self._n_seen += 1
            if self._n_seen >= self._hist_len:
                a_cmd = self.predictor.predict_one(self._hist)
            else:
                a_cmd = a  # warm-up: fall back to reactive
        else:
            a_cmd = a

        strokes = self.M_act @ a_cmd
        np.clip(strokes, -self.max_stroke, self.max_stroke, out=strokes)
        return strokes
