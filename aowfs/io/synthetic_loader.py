"""Synthetic-data loader and on-disk format.

Two entry points:

  * :func:`from_synthetic` -- generate a dataset in memory (fast dev path).
  * :func:`save_dataset` / :func:`load_sequence` -- write frames as .bmp plus a
    YAML metadata sidecar and read them back into a
    :class:`~aowfs.types.WFSFrameSequence`. This exercises the *exact* disk
    shape ISRO will deliver (a directory of .bmp frames + optics metadata), so
    the future ``isro_loader`` is a drop-in replacement.

The round-trip-from-disk path is the contract the real loader must satisfy; it
returns no ground truth, mirroring real data.
"""

from __future__ import annotations

import os

import numpy as np
import yaml
from PIL import Image

from ..config import SimConfig
from ..geometry import build_geometry
from ..sim.generate import generate_dataset
from ..types import SyntheticDataset, WFSFrameSequence

_META_NAME = "metadata.yaml"
_REF_NAME = "reference_centroids.npy"
_TRUTH_NAME = "ground_truth.npz"
_FRAME_FMT = "frame_{:05d}.bmp"


def from_synthetic(cfg: SimConfig, keep_truth: bool = True) -> SyntheticDataset:
    """Generate a synthetic dataset in memory."""
    return generate_dataset(cfg, keep_truth=keep_truth)


def save_dataset(dataset: SyntheticDataset, out_dir: str, save_truth: bool = True) -> None:
    """Persist a dataset to ``out_dir`` in the real-data on-disk shape."""
    os.makedirs(out_dir, exist_ok=True)
    seq = dataset.sequence

    for i in range(seq.n_frames):
        Image.fromarray(seq.frames[i]).save(os.path.join(out_dir, _FRAME_FMT.format(i)))

    np.save(os.path.join(out_dir, _REF_NAME), seq.reference_centroids)

    meta = dict(seq.cfg.to_dict())
    meta["_n_frames"] = int(seq.n_frames)
    meta["_detector_px"] = int(seq.detector_px)
    with open(os.path.join(out_dir, _META_NAME), "w", encoding="utf-8") as fh:
        yaml.safe_dump(meta, fh, sort_keys=False)

    if save_truth and dataset.truth is not None:
        t = dataset.truth
        np.savez_compressed(
            os.path.join(out_dir, _TRUTH_NAME),
            phase_screens=t.phase_screens,
            true_slopes=t.true_slopes,
            r0_sensing_m=t.r0_sensing_m,
            tau0_s=t.tau0_s,
            wavelength_m=t.wavelength_m,
            wind_speed_ms=t.wind_speed_ms,
            wind_dir_deg=t.wind_dir_deg,
        )


def load_sequence(in_dir: str) -> WFSFrameSequence:
    """Load a directory of .bmp frames + metadata into a WFSFrameSequence.

    This is the ground-truth-free path identical to what ``isro_loader`` will
    provide.
    """
    with open(os.path.join(in_dir, _META_NAME), "r", encoding="utf-8") as fh:
        meta = yaml.safe_load(fh)
    n_frames = int(meta.pop("_n_frames"))
    meta.pop("_detector_px", None)
    cfg = SimConfig.from_dict(meta)
    geom = build_geometry(cfg)

    frames = []
    for i in range(n_frames):
        path = os.path.join(in_dir, _FRAME_FMT.format(i))
        frames.append(np.asarray(Image.open(path)))
    frames_arr = np.stack(frames, axis=0)

    refs = np.load(os.path.join(in_dir, _REF_NAME))
    timestamps = np.arange(n_frames, dtype=np.float64) * cfg.detector.frame_dt_s

    return WFSFrameSequence(
        frames=frames_arr,
        cfg=cfg,
        geometry=geom,
        reference_centroids=refs,
        timestamps_s=timestamps,
    )
