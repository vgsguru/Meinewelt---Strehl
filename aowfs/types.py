"""Core data structures exchanged between the data layer and the algorithms.

These are the *only* objects that cross the io/ boundary. The synthetic loader
(now) and the ISRO loader (later) both produce a ``WFSFrameSequence``; every
downstream stage consumes exactly that, so swapping data sources never touches
algorithm code.

``GroundTruth`` is deliberately a *separate* object that the real loader will
return as ``None``. Keeping the injected truth (phase screens, r0, tau0, true
slopes / Zernikes) out of ``WFSFrameSequence`` makes it structurally
impossible for a reconstruction or estimation routine to peek at the answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import SimConfig
from .geometry import Geometry


@dataclass(frozen=True)
class WFSFrameSequence:
    """A time series of Shack-Hartmann detector frames + the optics metadata.

    This is the realistic, ground-truth-free view of the data -- exactly what
    a real camera + rig hands over.
    """

    frames: np.ndarray  # (n_frames, det_px, det_px), dtype matching bit depth
    cfg: SimConfig
    geometry: Geometry
    reference_centroids: np.ndarray  # (n_valid_subaps, 2) flat-wavefront spot positions [px]
    timestamps_s: np.ndarray  # (n_frames,) acquisition times

    def __post_init__(self) -> None:
        if self.frames.ndim != 3:
            raise ValueError(f"frames must be (T, H, W), got shape {self.frames.shape}")
        if self.frames.shape[1] != self.frames.shape[2]:
            raise ValueError("detector frames must be square")
        if self.reference_centroids.shape != (self.geometry.n_valid_subaps, 2):
            raise ValueError(
                "reference_centroids shape "
                f"{self.reference_centroids.shape} != "
                f"({self.geometry.n_valid_subaps}, 2)"
            )

    @property
    def n_frames(self) -> int:
        return self.frames.shape[0]

    @property
    def detector_px(self) -> int:
        return self.frames.shape[1]


@dataclass(frozen=True)
class GroundTruth:
    """Injected truth for synthetic runs -- used only for validation.

    ``None`` for real ISRO data. No estimation routine may take this as input.
    """

    phase_screens: np.ndarray  # (n_frames, pupil_grid_px, pupil_grid_px) [rad], pupil-masked
    true_slopes: np.ndarray  # (n_frames, n_valid_subaps, 2) noiseless slopes [rad/m]
    r0_sensing_m: float
    tau0_s: float
    wavelength_m: float
    wind_speed_ms: float
    wind_dir_deg: float

    @property
    def n_frames(self) -> int:
        return self.phase_screens.shape[0]


@dataclass(frozen=True)
class SyntheticDataset:
    """Bundle returned by the synthetic generator: realistic data + the truth."""

    sequence: WFSFrameSequence
    truth: Optional[GroundTruth]
