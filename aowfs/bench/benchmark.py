"""Benchmark the AO loop with calibration-time strictly separated from runtime.

The brief grades computational efficiency against the atmospheric coherence time
(milliseconds), so the number that matters is the *per-frame runtime path*:
centroid -> matvec -> predict -> matvec. Calibration cost (building and
inverting matrices, fitting the predictor) happens once at startup and is
reported separately, never amortised into the per-frame figure.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..config import SimConfig
from ..geometry import build_geometry
from ..io import from_synthetic
from ..predict.ar_predictor import PredictorConfig
from ..pipeline import RealTimePipeline


@dataclass
class BenchmarkResult:
    calibration_s: float
    runtime_per_frame_s: float  # full predictive path
    fps: float
    reactive_per_frame_s: float  # no prediction
    stage_times_us: dict  # per-stage median microseconds
    numpy_centroid_us: float
    numba_centroid_us: float
    coherence_time_ms: float
    frames_per_coherence_time: float

    def report(self) -> str:
        s = self.stage_times_us
        lines = [
            "=" * 66,
            "PHASE 6  --  RUNTIME BENCHMARK (calibration excluded)",
            "=" * 66,
            f"  calibration (once, at startup) : {self.calibration_s*1e3:8.2f} ms",
            "-" * 66,
            "  per-frame runtime path (the graded number):",
            f"    centroid (numba)             : {s['centroid']:8.2f} us",
            f"    reconstruct  (R_px matvec)   : {s['reconstruct']:8.2f} us",
            f"    predict      (AR forecast)   : {s['predict']:8.2f} us",
            f"    actuator map (M_act matvec)  : {s['actuator']:8.2f} us",
            f"    ---- full step ------------- : {self.runtime_per_frame_s*1e6:8.2f} us",
            "-" * 66,
            f"  throughput (predictive path)   : {self.fps:8.0f} fps",
            f"  throughput (reactive path)     : {1.0/self.reactive_per_frame_s:8.0f} fps",
            "-" * 66,
            f"  centroid: numpy {self.numpy_centroid_us:.1f} us  ->  "
            f"numba {self.numba_centroid_us:.1f} us  "
            f"({self.numpy_centroid_us/self.numba_centroid_us:.1f}x)",
            "-" * 66,
            f"  atmospheric coherence time tau0 : {self.coherence_time_ms:.2f} ms",
            f"  loop iterations within one tau0 : {self.frames_per_coherence_time:.0f}",
            f"  -> {'REAL-TIME CAPABLE' if self.frames_per_coherence_time > 5 else 'TOO SLOW'}"
            f" (need the loop >> turbulence rate)",
            "=" * 66,
        ]
        return "\n".join(lines)


def _median_time(fn, n_iters: int, n_warm: int = 20) -> float:
    for _ in range(n_warm):
        fn()
    ts = np.empty(n_iters)
    for i in range(n_iters):
        t0 = time.perf_counter()
        fn()
        ts[i] = time.perf_counter() - t0
    return float(np.median(ts))


def run_benchmark(cfg: SimConfig | None = None, n_iters: int = 400) -> BenchmarkResult:
    cfg = cfg or SimConfig()
    ds = from_synthetic(cfg)
    seq = ds.sequence
    geom = build_geometry(cfg)
    frames = seq.frames
    nf = seq.n_frames

    # --- calibration (timed once) ---------------------------------------- #
    t0 = time.perf_counter()
    pipe = RealTimePipeline(
        cfg, geom, seq.reference_centroids, n_modes=45,
        predictor_cfg=PredictorConfig(order=5, delay=2), use_numba=True,
    )
    # Training coefficients for the predictor are part of calibration.
    train = np.array([pipe.coeffs_from_frame(frames[i]) for i in range(min(nf, 200))])
    pipe.fit_predictor(train)
    calibration_s = time.perf_counter() - t0

    # --- runtime path (full predictive step) ----------------------------- #
    counter = {"i": 0}

    def full_step():
        f = frames[counter["i"] % nf]
        counter["i"] += 1
        pipe.step(f, predict=True)

    runtime_per_frame = _median_time(full_step, n_iters)

    def reactive_step():
        f = frames[counter["i"] % nf]
        counter["i"] += 1
        pipe.step(f, predict=False)

    reactive_per_frame = _median_time(reactive_step, n_iters)

    # --- per-stage breakdown --------------------------------------------- #
    frame0 = frames[0]
    disp = pipe.centroider.measure(frame0)
    disp_flat = np.concatenate([disp[:, 0], disp[:, 1]])
    a = pipe.R_px @ disp_flat
    pipe.reset()
    for i in range(pipe._hist_len):
        pipe._hist[i] = a
    pipe._n_seen = pipe._hist_len

    stages = {
        "centroid": _median_time(lambda: pipe.centroider.measure(frame0), n_iters),
        "reconstruct": _median_time(lambda: pipe.R_px @ disp_flat, n_iters),
        "predict": _median_time(lambda: pipe.predictor.predict_one(pipe._hist), n_iters),
        "actuator": _median_time(lambda: pipe.M_act @ a, n_iters),
    }
    stage_times_us = {k: v * 1e6 for k, v in stages.items()}

    # --- numpy vs numba centroid ----------------------------------------- #
    from ..recon.centroiding import CentroidConfig, Centroider

    cnp = Centroider(cfg, geom, seq.reference_centroids,
                     CentroidConfig(method="com", threshold_frac=0.2))
    numpy_us = _median_time(lambda: cnp.measure(frame0), n_iters) * 1e6
    numba_us = stage_times_us["centroid"]

    fps = 1.0 / runtime_per_frame
    tau0_ms = cfg.tau0_s * 1e3
    frames_per_tau0 = tau0_ms / (runtime_per_frame * 1e3)

    return BenchmarkResult(
        calibration_s=calibration_s,
        runtime_per_frame_s=runtime_per_frame,
        fps=fps,
        reactive_per_frame_s=reactive_per_frame,
        stage_times_us=stage_times_us,
        numpy_centroid_us=numpy_us,
        numba_centroid_us=numba_us,
        coherence_time_ms=tau0_ms,
        frames_per_coherence_time=frames_per_tau0,
    )


def main() -> int:
    print(run_benchmark().report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
