"""Configuration model for the AO-WFS pipeline.

Every physical quantity ISRO will eventually supply (detector pixel size and
resolution, MLA size / lenslet count / focal length, pupil diameter, DM
geometry and inter-actuator coupling) is represented here as a field on a
frozen, validated dataclass. Swapping synthetic data for the real dataset must
only mean populating these structures from a different loader -- no algorithm
code changes.

All quantities are SI (metres, seconds, radians) unless the field name says
otherwise. r0 is specified at a reference wavelength and scaled to the sensing
wavelength internally (r0 proportional to lambda**(6/5)).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any

import yaml


# --------------------------------------------------------------------------- #
# Optical / hardware configuration (the "Data Required" list from the brief)   #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PupilConfig:
    """Pupil of the turbulated beam."""

    diameter_m: float = 0.15  # entrance pupil diameter projected onto the MLA
    wavelength_m: float = 1.65e-6  # sensing wavelength (H band, near-IR per PRL NIR AO goal)

    def __post_init__(self) -> None:
        _positive(self.diameter_m, "pupil.diameter_m")
        _positive(self.wavelength_m, "pupil.wavelength_m")


@dataclass(frozen=True)
class MLAConfig:
    """Micro-lens array (Shack-Hartmann lenslet grid).

    In a real SH-WFS the telescope/lab pupil is relayed and demagnified onto a
    physically small MLA, so ``lenslet_pitch_m`` (the actual lenslet size, sub-
    mm) is distinct from the sub-aperture footprint at the pupil
    (``pupil.diameter_m / n_subaps``). The wavefront slope a lenslet sees is the
    pupil-plane slope amplified by the relay magnification
    M = (pupil sub-aperture pitch) / (lenslet pitch). This is why the brief
    lists MLA size and pupil size as separate quantities.
    """

    n_subaps: int = 16  # lenslets across the pupil diameter (N x N Fried grid)
    focal_length_m: float = 5.0e-3  # lenslet focal length (f/# ~ 25 with the pitch below)
    lenslet_pitch_m: float = 200e-6  # physical lenslet pitch on the MLA

    def __post_init__(self) -> None:
        _positive_int(self.n_subaps, "mla.n_subaps")
        _positive(self.focal_length_m, "mla.focal_length_m")
        _positive(self.lenslet_pitch_m, "mla.lenslet_pitch_m")

    def subaperture_pitch_m(self, pupil: PupilConfig) -> float:
        """Physical width of one sub-aperture *at the pupil*."""
        return pupil.diameter_m / self.n_subaps

    def magnification(self, pupil: PupilConfig) -> float:
        """Pupil-to-MLA relay magnification M = pupil subap pitch / lenslet pitch."""
        return self.subaperture_pitch_m(pupil) / self.lenslet_pitch_m

    def f_number(self) -> float:
        return self.focal_length_m / self.lenslet_pitch_m


@dataclass(frozen=True)
class DetectorConfig:
    """Science-grade camera recording the SH spot field."""

    pixel_size_m: float = 5.5e-6  # physical detector pixel pitch
    det_px_per_subap: int = 20  # detector pixels spanning one lenslet's spot cell
    bit_depth: int = 8  # .bmp frames from the real rig are 8-bit
    frame_dt_s: float = 2.0e-3  # inter-frame interval (a few ms per the brief)
    # Electrons mapped to full scale (a FIXED camera/exposure property, like a
    # real detector's gain -- NOT re-scaled per flux). Chosen so the nominal
    # spot fills most of the 8-bit range; lower flux then yields a genuinely
    # dim, read-noise-limited image, which is the correct low-light regime.
    well_depth_e: float = 40.0

    def __post_init__(self) -> None:
        _positive(self.pixel_size_m, "detector.pixel_size_m")
        _positive_int(self.det_px_per_subap, "detector.det_px_per_subap")
        _positive_int(self.bit_depth, "detector.bit_depth")
        _positive(self.frame_dt_s, "detector.frame_dt_s")
        _positive(self.well_depth_e, "detector.well_depth_e")

    @property
    def max_count(self) -> int:
        return (1 << self.bit_depth) - 1

    @property
    def gain_dn_per_e(self) -> float:
        """Fixed digitisation gain: digital numbers per electron."""
        return self.max_count / self.well_depth_e


@dataclass(frozen=True)
class DMConfig:
    """Deformable mirror, in Fried geometry with the MLA.

    In a Fried configuration the actuators sit on the *corners* of the WFS
    sub-apertures, so an N x N lenslet grid drives an (N+1) x (N+1) actuator
    grid. Inter-actuator coupling is the fraction of an actuator's stroke that
    appears at its nearest neighbour, modelled by a Gaussian influence
    function whose width is fixed by this coupling value.
    """

    coupling: float = 0.15  # neighbour coupling fraction (typ. 0.10-0.20)
    max_stroke_m: float = 4.0e-6  # +/- mechanical stroke limit
    # Actuator pitch defaults to the sub-aperture pitch (Fried geometry).
    pitch_m: float | None = None

    def __post_init__(self) -> None:
        if not (0.0 < self.coupling < 1.0):
            raise ValueError(f"dm.coupling must be in (0, 1), got {self.coupling}")
        _positive(self.max_stroke_m, "dm.max_stroke_m")
        if self.pitch_m is not None:
            _positive(self.pitch_m, "dm.pitch_m")

    def n_actuators_across(self, mla: MLAConfig) -> int:
        return mla.n_subaps + 1


# --------------------------------------------------------------------------- #
# Synthetic-only configuration (ground truth we inject, then try to recover)   #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TurbulenceConfig:
    """Injected atmospheric turbulence (synthetic ground truth)."""

    r0_ref_m: float = 0.009  # Fried parameter at reference wavelength (strong lab turbulence)
    r0_ref_wavelength_m: float = 0.5e-6  # reference wavelength for r0 (500 nm convention)
    L0_m: float = 25.0  # von Karman outer scale
    l0_m: float = 0.003  # inner scale (lab generator; keeps a usable inertial range)
    wind_speed_ms: float = 2.0  # frozen-flow wind speed (lab phase-plate equivalent)
    wind_dir_deg: float = 30.0  # wind direction (0 deg = +x)
    seed: int = 1234

    def __post_init__(self) -> None:
        _positive(self.r0_ref_m, "turbulence.r0_ref_m")
        _positive(self.r0_ref_wavelength_m, "turbulence.r0_ref_wavelength_m")
        _positive(self.L0_m, "turbulence.L0_m")
        _non_negative(self.wind_speed_ms, "turbulence.wind_speed_ms")

    def r0_at(self, wavelength_m: float) -> float:
        """Scale r0 to a sensing wavelength: r0 ~ lambda**(6/5)."""
        return self.r0_ref_m * (wavelength_m / self.r0_ref_wavelength_m) ** (6.0 / 5.0)

    def tau0_s(self, wavelength_m: float) -> float:
        """Greenwood-style coherence time for a single frozen layer.

        tau0 = 0.314 * r0 / V, with r0 at the sensing wavelength. This is the
        ground-truth value Phase 3 must recover from the data alone.
        """
        if self.wind_speed_ms == 0.0:
            return math.inf
        return 0.314 * self.r0_at(wavelength_m) / self.wind_speed_ms


@dataclass(frozen=True)
class NoiseConfig:
    """Detector noise model for the synthetic forward simulation."""

    photons_per_spot: float = 2000.0  # total photo-electrons per spot per frame
    read_noise_e: float = 2.0  # RMS read noise (electrons)
    dark_e: float = 0.0  # dark/background pedestal (electrons)
    enable: bool = True  # disable for noiseless ground-truth checks

    def __post_init__(self) -> None:
        _positive(self.photons_per_spot, "noise.photons_per_spot")
        _non_negative(self.read_noise_e, "noise.read_noise_e")
        _non_negative(self.dark_e, "noise.dark_e")


# --------------------------------------------------------------------------- #
# Top-level config                                                             #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SimConfig:
    """Complete configuration for a synthetic SH-WFS run."""

    pupil: PupilConfig = field(default_factory=PupilConfig)
    mla: MLAConfig = field(default_factory=MLAConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    dm: DMConfig = field(default_factory=DMConfig)
    turbulence: TurbulenceConfig = field(default_factory=TurbulenceConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)

    n_frames: int = 200
    # Phase-screen sampling: simulation pixels across one sub-aperture. Higher
    # gives a more accurate gradient at higher cost. This is a synthesis-only
    # knob and never appears in the real data path.
    screen_px_per_subap: int = 8

    def __post_init__(self) -> None:
        _positive_int(self.n_frames, "n_frames")
        _positive_int(self.screen_px_per_subap, "screen_px_per_subap")

    # --- derived geometry (single source of truth, never recomputed ad hoc) --

    @property
    def subap_pitch_m(self) -> float:
        return self.mla.subaperture_pitch_m(self.pupil)

    @property
    def screen_pixel_scale_m(self) -> float:
        """Metres per phase-screen pixel."""
        return self.subap_pitch_m / self.screen_px_per_subap

    @property
    def pupil_grid_px(self) -> int:
        """Phase-screen pixels across the full pupil."""
        return self.mla.n_subaps * self.screen_px_per_subap

    @property
    def detector_px(self) -> int:
        """Detector pixels across the full square spot field."""
        return self.mla.n_subaps * self.detector.det_px_per_subap

    @property
    def r0_sensing_m(self) -> float:
        return self.turbulence.r0_at(self.pupil.wavelength_m)

    @property
    def tau0_s(self) -> float:
        return self.turbulence.tau0_s(self.pupil.wavelength_m)

    @property
    def wind_vector_px_per_frame(self) -> tuple[float, float]:
        """Frozen-flow shift per frame, in phase-screen pixels (dx, dy)."""
        v = self.turbulence.wind_speed_ms
        theta = math.radians(self.turbulence.wind_dir_deg)
        dist_m = v * self.detector.frame_dt_s
        dist_px = dist_m / self.screen_pixel_scale_m
        return dist_px * math.cos(theta), dist_px * math.sin(theta)

    # --- serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SimConfig":
        d = dict(d or {})
        sub = {
            "pupil": PupilConfig,
            "mla": MLAConfig,
            "detector": DetectorConfig,
            "dm": DMConfig,
            "turbulence": TurbulenceConfig,
            "noise": NoiseConfig,
        }
        kwargs: dict[str, Any] = {}
        for key, klass in sub.items():
            if key in d and d[key] is not None:
                kwargs[key] = klass(**d.pop(key))
        kwargs.update(d)
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str) -> "SimConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh))

    def to_yaml(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)


# --------------------------------------------------------------------------- #
# validation helpers                                                           #
# --------------------------------------------------------------------------- #


def _positive(x: float, name: str) -> None:
    if not (x > 0):
        raise ValueError(f"{name} must be > 0, got {x}")


def _non_negative(x: float, name: str) -> None:
    if not (x >= 0):
        raise ValueError(f"{name} must be >= 0, got {x}")


def _positive_int(x: int, name: str) -> None:
    if not (isinstance(x, int) and x > 0):
        raise ValueError(f"{name} must be a positive integer, got {x!r}")
