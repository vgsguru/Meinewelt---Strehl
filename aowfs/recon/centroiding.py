"""Phase 1 -- Shack-Hartmann spot centroiding.

For each sub-aperture we measure the spot displacement from its calibrated
(flat-wavefront) reference position. That displacement vector is the raw slope
signal the wavefront reconstructor (Phase 2) consumes.

Two methods share one calibrated object:

  * ``"com"``  -- thresholded centre-of-mass. Background-subtracted, with a
    fractional-peak threshold to suppress noise, optionally restricted to a
    window about the reference. Fully vectorised over all sub-apertures; this
    is the real-time path.

  * ``"correlation"`` -- FFT cross-correlation against a data-driven spot
    template with parabolic sub-pixel peak interpolation. More robust at low
    photon flux / high read noise; offered as the noise-hardened option.

Calibration (cell geometry, reference positions, the correlation template) is
done once in ``__init__``; ``measure`` / ``measure_sequence`` are the hot path.

Sign convention matches the forward model: a positive x-gradient moves the
spot to +x, so ``measure`` returns (centroid - reference), i.e. +displacement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import SimConfig
from ..geometry import Geometry


@dataclass(frozen=True)
class CentroidConfig:
    method: str = "com"  # "com" | "correlation"
    threshold_frac: float = 0.2  # zero pixels below this fraction of (peak - background)
    subtract_background: bool = True  # remove per-cell median before centroiding
    window_radius_px: float | None = None  # restrict CoM to a disc about the reference
    # Gaussian apodisation for "correlation": suppresses peripheral read noise
    # (better in the read-noise-limited regime) at the cost of a small
    # centre-pulling bias at high SNR. Off by default -> unbiased.
    apodize: bool = False

    def __post_init__(self) -> None:
        if self.method not in ("com", "correlation"):
            raise ValueError(f"unknown centroiding method {self.method!r}")
        if not (0.0 <= self.threshold_frac < 1.0):
            raise ValueError("threshold_frac must be in [0, 1)")


class Centroider:
    """Calibrated SH centroider. Build once, call :meth:`measure` per frame."""

    def __init__(
        self,
        cfg: SimConfig,
        geom: Geometry,
        reference_centroids: np.ndarray,
        ccfg: CentroidConfig | None = None,
    ):
        self.cfg = cfg
        self.geom = geom
        self.ccfg = ccfg or CentroidConfig()
        self.N = cfg.mla.n_subaps
        self.q = cfg.detector.det_px_per_subap

        # Row/col grid index of every cell, and which are valid (pupil) lenslets.
        self._rows = geom.valid_subap_index[:, 0]
        self._cols = geom.valid_subap_index[:, 1]
        self.reference = np.asarray(reference_centroids, dtype=np.float64)

        # In-cell coordinate axes.
        self._axis = np.arange(self.q, dtype=np.float64)

        # Static window mask (about the cell centre, where the reference sits).
        self._window = None
        if self.ccfg.window_radius_px is not None:
            c = (self.q - 1) / 2.0
            yy, xx = np.meshgrid(self._axis, self._axis, indexing="ij")
            self._window = (
                ((xx - c) ** 2 + (yy - c) ** 2) <= self.ccfg.window_radius_px ** 2
            ).astype(np.float64)

        # Correlation template is built lazily from the first measured frame so
        # the centroider needs no simulator knowledge and works on real data.
        self._template_fft = None

    # ------------------------------------------------------------------ #
    # public API                                                         #
    # ------------------------------------------------------------------ #

    def measure(self, frame: np.ndarray) -> np.ndarray:
        """Return (n_valid_subaps, 2) spot displacements [px] for one frame."""
        cells = self._extract_cells(frame)  # (N, N, q, q)
        if self.ccfg.method == "com":
            cx, cy = self._com(cells)
        else:
            cx, cy = self._correlation(cells)
        # Global spot position = cell origin + in-cell centroid.
        gx = self._cols * self.q + cx[self._rows, self._cols]
        gy = self._rows * self.q + cy[self._rows, self._cols]
        return np.column_stack([gx - self.reference[:, 0], gy - self.reference[:, 1]])

    def measure_sequence(self, frames: np.ndarray) -> np.ndarray:
        """Centroid a stack of frames -> (T, n_valid_subaps, 2) displacements [px]."""
        return np.stack([self.measure(frames[i]) for i in range(frames.shape[0])], axis=0)

    # ------------------------------------------------------------------ #
    # internals                                                          #
    # ------------------------------------------------------------------ #

    def _extract_cells(self, frame: np.ndarray) -> np.ndarray:
        N, q = self.N, self.q
        return frame.reshape(N, q, N, q).transpose(0, 2, 1, 3).astype(np.float64)

    def _com(self, cells: np.ndarray):
        c = cells
        if self.ccfg.subtract_background:
            bg = np.median(c, axis=(2, 3), keepdims=True)
            c = c - bg
        peak = c.max(axis=(2, 3), keepdims=True)
        c = np.clip(c - self.ccfg.threshold_frac * peak, 0.0, None)
        if self._window is not None:
            c = c * self._window
        tot = c.sum(axis=(2, 3))
        safe = np.where(tot > 0, tot, 1.0)
        # cx weights the column axis (x), cy the row axis (y).
        cx = (c.sum(axis=2) @ self._axis) / safe
        cy = (c.sum(axis=3) @ self._axis) / safe
        # Dead cells (no signal) -> place spot at the cell centre (zero displacement).
        centre = (self.q - 1) / 2.0
        dead = tot <= 0
        cx = np.where(dead, centre, cx)
        cy = np.where(dead, centre, cy)
        return cx, cy

    def _apodize(self) -> np.ndarray:
        """Soft Gaussian window centred on the cell, cached after first use."""
        w = getattr(self, "_apod", None)
        if w is None:
            q = self.q
            c = (q - 1) / 2.0
            sig = q / 4.0
            ax = self._axis
            yy, xx = np.meshgrid(ax, ax, indexing="ij")
            w = np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2.0 * sig ** 2))
            self._apod = w
        return w

    def _build_template(self, cells: np.ndarray) -> None:
        """Build a normalised spot template from the data (mean centred spot)."""
        q = self.q
        # Use valid cells only; centre each by its CoM, then average.
        cx, cy = self._com(cells)
        acc = np.zeros((q, q), dtype=np.float64)
        n = 0
        from scipy.ndimage import shift as nd_shift

        centre = (q - 1) / 2.0
        for r, col in zip(self._rows, self._cols):
            cell = cells[r, col]
            s = cell.sum()
            if s <= 0:
                continue
            shifted = nd_shift(cell, (centre - cy[r, col], centre - cx[r, col]), order=1)
            acc += shifted / s
            n += 1
        template = acc / max(n, 1)
        # Zero-mean, unit-norm template -> a matched filter whose correlation is
        # a covariance, sharply peaked at the spot rather than dominated by the
        # broad non-negative baseline.
        template -= template.mean()
        nrm = np.sqrt(np.sum(template ** 2))
        if nrm > 0:
            template /= nrm
        self._template_fft = np.conj(np.fft.fft2(template))

    def _correlation(self, cells: np.ndarray):
        if self._template_fft is None:
            self._build_template(cells)
        N, q = self.N, self.q
        c = cells
        if self.ccfg.apodize:
            # Apodise about the cell centre to keep peripheral read noise out of
            # the matched filter (the spot lives near the centre).
            c = c * self._apodize()
        # Zero-mean each cell so the matched-filter peak is the spot, not the DC
        # overlap of two non-negative images.
        c = c - c.mean(axis=(2, 3), keepdims=True)
        # Cross-correlation peak (circular) gives displacement from template centre.
        R = np.fft.ifft2(np.fft.fft2(c, axes=(2, 3)) * self._template_fft, axes=(2, 3)).real
        M = N * N
        Rf = R.reshape(M, q, q)
        idx = Rf.reshape(M, q * q).argmax(axis=1)
        pr, pc = np.divmod(idx, q)
        ar = np.arange(M)

        def pary(p):
            a = Rf[ar, (p - 1) % q, pc]
            b = Rf[ar, (p + 1) % q, pc]
            o = Rf[ar, p, pc]
            return _parabolic(a, o, b)

        def parx(p):
            a = Rf[ar, pr, (p - 1) % q]
            b = Rf[ar, pr, (p + 1) % q]
            o = Rf[ar, pr, p]
            return _parabolic(a, o, b)

        dy = _signed_index(pr, q) + pary(pr)
        dx = _signed_index(pc, q) + parx(pc)
        centre = (q - 1) / 2.0
        cx = (centre + dx).reshape(N, N)
        cy = (centre + dy).reshape(N, N)
        return cx, cy


def _signed_index(p: np.ndarray, q: int) -> np.ndarray:
    """Map a circular FFT index in [0, q) to a signed shift in (-q/2, q/2]."""
    return np.where(p <= q // 2, p, p - q).astype(np.float64)


def _parabolic(a: np.ndarray, o: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Sub-pixel offset of a parabola through (a, o, b) at offsets (-1, 0, +1)."""
    denom = a - 2.0 * o + b
    out = np.zeros_like(o)
    nz = np.abs(denom) > 1e-12
    out[nz] = 0.5 * (a[nz] - b[nz]) / denom[nz]
    return np.clip(out, -0.5, 0.5)
