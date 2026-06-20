"""aowfs -- Predictive adaptive-optics wavefront reconstruction (BAH 2026 PS9).

Data ingestion (``aowfs.sim``, ``aowfs.io``) is kept strictly separate from the
algorithms (``aowfs.recon``, ``aowfs.turbulence``, ``aowfs.predict``,
``aowfs.actuator``). Swapping synthetic data for ISRO's real dataset must only
touch the io layer.
"""

from __future__ import annotations

from .config import (
    DetectorConfig,
    DMConfig,
    MLAConfig,
    NoiseConfig,
    PupilConfig,
    SimConfig,
    TurbulenceConfig,
)
from .geometry import Geometry, build_geometry
from .types import GroundTruth, SyntheticDataset, WFSFrameSequence

__all__ = [
    "SimConfig",
    "PupilConfig",
    "MLAConfig",
    "DetectorConfig",
    "DMConfig",
    "TurbulenceConfig",
    "NoiseConfig",
    "Geometry",
    "build_geometry",
    "WFSFrameSequence",
    "GroundTruth",
    "SyntheticDataset",
]

__version__ = "0.1.0"
