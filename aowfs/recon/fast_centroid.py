"""Numba-JIT thresholded centre-of-mass centroiding for the real-time path.

This is the hot image-processing step of the per-frame loop. The kernel loops
over valid sub-aperture cells doing a thresholded centre-of-mass directly on the
detector buffer -- cache-friendly and free of per-cell temporary arrays, so it
runs an order of magnitude faster than the vectorised numpy version while giving
the same result (threshold at a fraction of the cell peak).
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=True, parallel=True)
def _com_kernel(frame, q, rows, cols, ref_x, ref_y, thr_frac, out):
    n = rows.shape[0]
    for k in prange(n):
        r0 = rows[k] * q
        c0 = cols[k] * q
        # cell peak
        peak = 0.0
        for i in range(q):
            for j in range(q):
                v = frame[r0 + i, c0 + j]
                if v > peak:
                    peak = v
        thr = thr_frac * peak
        sw = 0.0
        sx = 0.0
        sy = 0.0
        for i in range(q):
            for j in range(q):
                v = frame[r0 + i, c0 + j] - thr
                if v > 0.0:
                    sw += v
                    sx += v * j
                    sy += v * i
        if sw > 0.0:
            cx = sx / sw
            cy = sy / sw
        else:
            cx = (q - 1) * 0.5
            cy = (q - 1) * 0.5
        out[k, 0] = c0 + cx - ref_x[k]
        out[k, 1] = r0 + cy - ref_y[k]


class FastCentroider:
    """Calibrated numba CoM centroider matching :class:`Centroider` output."""

    def __init__(self, cfg, geom, reference_centroids, threshold_frac: float = 0.2):
        self.q = cfg.detector.det_px_per_subap
        self.rows = np.ascontiguousarray(geom.valid_subap_index[:, 0])
        self.cols = np.ascontiguousarray(geom.valid_subap_index[:, 1])
        self.ref = np.asarray(reference_centroids, dtype=np.float64)
        self.thr = float(threshold_frac)
        self.n = self.rows.shape[0]
        self._out = np.empty((self.n, 2), dtype=np.float64)
        # Warm up the JIT so the first timed frame is representative.
        dummy = np.zeros((cfg.detector_px, cfg.detector_px), dtype=np.float64)
        _com_kernel(dummy, self.q, self.rows, self.cols,
                    self.ref[:, 0], self.ref[:, 1], self.thr, self._out)

    def measure(self, frame: np.ndarray) -> np.ndarray:
        f = frame if frame.dtype == np.float64 else frame.astype(np.float64)
        _com_kernel(f, self.q, self.rows, self.cols,
                    self.ref[:, 0], self.ref[:, 1], self.thr, self._out)
        return self._out
