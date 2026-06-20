"""High-level turbulence characterization: data -> (r0, tau0, wind) report.

Takes the *measured* time series of slopes and Zernike coefficients (whatever
produced them -- synthetic or real) and returns the turbulence parameters. Pure
estimation: it never sees ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import SimConfig
from ..geometry import Geometry
from .r0_estimator import r0_from_slope_variance, r0_from_zernike_variance
from .tau0_estimator import tau0_greenwood, wind_from_slope_correlation


@dataclass(frozen=True)
class TurbulenceResult:
    r0_m: float  # primary r0 (Zernike-variance)
    r0_slope_m: float  # cross-check r0 (slope-variance)
    tau0_s: float
    wind_speed_ms: float
    wind_dir_deg: float

    def __str__(self) -> str:
        return (
            f"r0 = {self.r0_m*100:.2f} cm (slope-var cross-check "
            f"{self.r0_slope_m*100:.2f} cm), "
            f"wind = {self.wind_speed_ms:.2f} m/s @ {self.wind_dir_deg:.0f} deg, "
            f"tau0 = {self.tau0_s*1e3:.2f} ms"
        )


def characterize(
    gradients: np.ndarray,
    coeffs: np.ndarray,
    cfg: SimConfig,
    geom: Geometry,
    zernike_band: tuple[int, int] = (4, 15),
) -> TurbulenceResult:
    """Estimate r0, wind and tau0 from measured slopes + Zernike coefficients.

    ``gradients``: (n_frames, n_valid_subaps, 2) wavefront gradients [rad/m].
    ``coeffs``:    (n_frames, n_modes) Zernike coefficients for Noll j = 2, 3, ...
    """
    r0 = r0_from_zernike_variance(coeffs, cfg, band=zernike_band)
    r0_slope = r0_from_slope_variance(gradients, cfg)
    speed, direction = wind_from_slope_correlation(gradients, cfg, geom)
    tau0 = tau0_greenwood(r0, speed)
    return TurbulenceResult(
        r0_m=r0,
        r0_slope_m=r0_slope,
        tau0_s=tau0,
        wind_speed_ms=speed,
        wind_dir_deg=direction,
    )
