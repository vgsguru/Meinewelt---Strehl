"""Data ingestion layer.

``synthetic_loader`` now; ``isro_loader`` later. Both emit the identical
:class:`aowfs.types.WFSFrameSequence`, so Phases 1-6 never know which produced
the data.
"""

from __future__ import annotations

from .aot_loader import AOTDataset, AOTReference, load_naomi_aot
from .isro_loader import ISROSpecs, load_isro_dataset
from .synthetic_loader import (
    from_synthetic,
    load_sequence,
    save_dataset,
)

__all__ = [
    "from_synthetic",
    "load_sequence",
    "save_dataset",
    "ISROSpecs",
    "load_isro_dataset",
    "load_naomi_aot",
    "AOTDataset",
    "AOTReference",
]
