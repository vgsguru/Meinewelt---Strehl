"""Phase 5 -- deformable-mirror actuator mapping with inter-actuator coupling."""

from __future__ import annotations

from .actuator_map import ActuatorConfig, ActuatorMapper
from .influence_matrix import build_influence_matrix, influence_function

__all__ = [
    "ActuatorMapper",
    "ActuatorConfig",
    "build_influence_matrix",
    "influence_function",
]
