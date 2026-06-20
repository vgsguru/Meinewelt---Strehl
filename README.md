# Predictive Adaptive-Optics Wavefront Reconstruction (BAH 2026 — PS9)

Fast, physically-correct wavefront reconstruction, turbulence characterization
(Fried parameter r₀, coherence time τ₀) and deformable-mirror actuator-map
generation from Shack-Hartmann WFS time-series data — plus a **predictive
control layer** that estimates the wavefront 1–2 frames ahead to fight
servo-lag, the same technique flown on systems like Gemini ALTAIR.

> Status: **Phases 0-10 complete.** Full real-time predictive pipeline, benchmark,
> dashboard, ISRO real-data adapter (Phase 7), reviewer-flagged fixes (Phase 8),
> all four reconstruction methods incl. ML (Phase 9), and a real-world AOT-format
> stress test against on-sky NAOMI@VLT telemetry (Phase 10,
> [docs/phase10_notes.md](docs/phase10_notes.md)). Phase 11 (optional stretch
> goals) remains.

## Why this design

The brief is satisfied by classical zonal/modal reconstruction alone. We go
beyond it with predictive control because the evaluation explicitly rewards
"speed/efficiency" and "conformance to turbulence characteristics" — and
predictive control is fundamentally about modelling how turbulence *evolves in
time*. We start from the operationally-proven simple approach (linear
auto-regressive / Kalman, ~5 coefficients per mode) and treat a neural
predictor as an evidence-gated stretch goal, exactly as the field's history
(ALTAIR linear forecasting → LQG/Kalman → EOF predictors → CNN control)
suggests.

## Architecture

Data ingestion is kept strictly separate from algorithms, so swapping the
synthetic generator for ISRO's real dataset only touches the `io/` layer.

```
aowfs/
  config.py        Frozen, validated config — one field per ISRO "Data Required" item
  geometry.py      Fried geometry: lenslet grid, (N+1) actuator grid, pupil mask
  types.py         WFSFrameSequence (real-data-shaped) + quarantined GroundTruth
  sim/             [Phase 0] synthetic phase screens + SH-WFS forward model + validate
  io/              synthetic_loader (now); isro_loader (Phase 7) — same output type
  recon/           [Phase 1-2] centroiding, zonal (Southwell), modal (Zernike/Noll)
  turbulence/      [Phase 3] r0 + tau0 estimators
  predict/         [Phase 4] AR / Kalman predictors (NN only if justified)
  actuator/        [Phase 5] influence matrix + coupling-aware actuator mapping
  bench/           [Phase 6] calibration-time vs runtime-path timing harness
  viz/             [Phase 6] streamlit dashboard, plots
tests/             one validation test per phase, against synthetic ground truth
configs/           default.yaml — the single source of optical/turbulence params
```

Two architectural rules are enforced throughout:

1. **Calibration once, runtime is a matrix-vector multiply.** Every
   calibration matrix (reconstruction matrix, DM influence matrix and its
   inverse) is built once at startup; the per-frame path never re-derives one.
2. **Ground truth is quarantined.** Injected truth (phase screens, r₀, τ₀,
   true slopes) lives in a separate `GroundTruth` object that real data returns
   as `None`, so no estimator can accidentally read the answer.

## Phase 0 — Synthetic SH-WFS simulator

Generates data shaped exactly like ISRO's eventual dataset (a directory of
8-bit `.bmp` frames + optics metadata) but with known ground truth:

- **Turbulence:** a large von Kármán phase screen via aotools'
  subharmonic-augmented spectral method (`ft_sh_phase_screen`), translated
  under the Taylor frozen-flow hypothesis by a configurable wind vector — so
  τ₀ = 0.314·r₀/V is a *real, recoverable* property.
- **Optics:** physically-correct SH forward model. The pupil is relayed and
  demagnified onto a small MLA (relay magnification M = pupil-subaperture pitch
  / lenslet pitch ≈ 47), each lenslet measures the mean wavefront gradient, and
  a phase gradient ∂φ/∂x displaces the focal spot by
  `f·(λ/2π)·(∂φ/∂x)·M / pixel_size` detector pixels.
- **Detector:** Gaussian spots (diffraction width λf/d_lenslet), Poisson photon
  noise + Gaussian read noise, fixed linear gain, quantised to 8-bit `.bmp`.

### Validation (every check is a number, against ground truth)

Run `python -m aowfs.sim.validate`:

| Check | Quantity | Result | Tolerance |
|-------|----------|-------:|----------:|
| A. Forward-model exactness | gradient error on analytic ramp | ~0 rad/m | <1e-6 |
| B. Spot invertibility (noiseless) | recover injected displacement | 0.003 px | <0.05 |
| C. End-to-end centroid error (noisy) | measured vs predicted offset | 0.11 px | <0.25 |
| D. Injected r₀ recovery | structure-function r₀ error | ~15 % | <20 % |
| E. Frozen-flow consistency | next-frame reproduced by wind shift | <0.01 % | <8 % |

Default scenario: 15 cm pupil, H-band (1.65 µm), 16×16 lenslets (f/25), strong
lab turbulence r₀ ≈ 3.8 cm (D/r₀ ≈ 4), V = 2 m/s ⇒ τ₀ ≈ 5.9 ms.

## Phase 1 — Spot centroiding

Measures each spot's displacement from its calibrated reference position (the
raw slope signal). One calibrated `Centroider` (cell geometry + reference +
template built once) exposes a vectorised per-frame `measure()`. Two methods:
thresholded **center-of-mass** (the unbiased, fast real-time path) and
**correlation** (FFT matched filter, data-driven template, parabolic sub-pixel)
with optional Gaussian apodisation for the read-noise-limited regime.

Accuracy vs ground-truth displacement (RMS, px) — reported, not assumed:

| Scenario | CoM | Correlation | Corr + apodise |
|---|--:|--:|--:|
| Noiseless turbulence | 0.004 | 0.002 | — |
| Nominal (2000 ph) | 0.122 | **0.108** | 0.240 |
| Moderate (500 ph) | 0.440 | 0.399 | **0.383** |
| Low (300 ph) | 0.640 | 0.676 | **0.497** |
| Flat wavefront, noiseless | **0.00000** (SPEC check) | | |

Takeaway: CoM is the unbiased real-time default; correlation edges it at
nominal SNR; apodised correlation is the robust choice once read noise
dominates. Apodisation biases the high-SNR result, so it is opt-in — we let the
numbers, not a slogan, pick the method.

## Phase 2 — Wavefront reconstruction (zonal + modal)

Both reconstructors **precompute their calibration matrix once** (truncated-SVD
pseudo-inverse); per frame, reconstruction is a single matrix-vector multiply —
the architectural decision that drives the efficiency score.

- **Modal** ([modal.py](aowfs/recon/modal.py)): Noll Zernike basis (self-contained,
  polynomial-continued so edge sub-apertures are exact). The slope-to-Zernike
  interaction matrix is built by *poking* each mode through the same WFS
  sampling operator the simulator uses, so calibration is exactly consistent
  with the measurements — as a real AO system self-calibrates.
- **Zonal** ([zonal.py](aowfs/recon/zonal.py)): Fried-geometry matrix relating
  sub-aperture slopes to phase at the (N+1)×(N+1) sub-aperture corners — i.e.
  directly on the DM actuator grid, feeding Phase 5.

**Validation.** The modal residual wavefront variance shrinks as more modes are
corrected, tracking the Noll Δ_J ∝ (D/r₀)^(5/3) trend (the ~1.4× offset is
finite-SH aliasing, consistent across mode count):

| Noll J | 6 | 11 | 21 | 36 | 46 |
|---|--:|--:|--:|--:|--:|
| measured residual [rad²] | 0.95 | 0.51 | 0.28 | 0.19 | 0.15 |
| Noll Δ_J·(D/r₀)^5/3 [rad²] | 0.65 | 0.38 | 0.21 | 0.12 | 0.11 |

End-to-end (frame → centroid → reconstruct, with full detector noise): residual
wavefront error **0.40 rad RMS vs 2.63 rad** of injected turbulence — an 85%
reduction, implied Strehl ≈ 0.85 (Maréchal). Zonal reconstruction matches the
ground-truth phase to 15.5% in the interior and agrees with modal to ~22%.

## Phase 3 — Turbulence characterization (r₀, τ₀)

Three estimators, all run on the *measured* data (never ground truth):
- **r₀** ([r0_estimator.py](aowfs/turbulence/r0_estimator.py)) — fit the variance
  of reconstructed Zernike coefficients to the Kolmogorov per-mode law
  `⟨a_j²⟩ = c_j(D/r₀)^(5/3)` (Noll coefficients), over mid-order modes 4–15
  (tip/tilt skipped — outer-scale sensitive; high orders skipped — noise). A
  slope-variance estimate (Saint-Jacques) is reported as an independent
  cross-check.
- **Wind V** ([tau0_estimator.py](aowfs/turbulence/tau0_estimator.py)) —
  spatio-temporal cross-correlation of the slope maps; the frozen-flow peak
  displaces by V·τ. (Slopes are high-pass, so the peak localises — raw phase
  does not.)
- **τ₀** = 0.314·r₀/V (Greenwood), exactly the quantity injected in Phase 0.

**Validation — injected ground truth vs recovered**, across four scenarios:

| Scenario | r₀ inj | r₀ rec | τ₀ inj | τ₀ rec | V inj | V rec |
|---|--:|--:|--:|--:|--:|--:|
| nominal | 3.77 cm | 3.43 cm (−9%) | 5.92 ms | 5.45 ms (−8%) | 2.0 | 1.97 |
| strong turbulence | 2.51 cm | 2.29 cm (−9%) | 3.95 ms | 3.64 ms (−8%) | 2.0 | 1.97 |
| fast wind | 3.77 cm | 4.28 cm (+14%) | 2.96 ms | 3.44 ms (+16%) | 4.0 | 3.92 |
| weak turbulence | 5.87 cm | 5.67 cm (−3%) | 12.28 ms | 13.35 ms (+9%) | 1.5 | 1.33 |

Wind direction is recovered to ≈1° in every case. The residual r₀ scatter is
dominated by single-screen realization statistics (frozen flow = one screen
sliding), not estimator bias — exactly the regime the real data will present.

## Phase 4 — Predictive wavefront control (the differentiator)

For a loop delay of `d` frames the controller must set frame *t*'s correction
from data available only up to *t−d*; during that lag the turbulence evolves,
leaving a servo-lag error. We forecast each Zernike mode `d` frames ahead with a
per-mode linear (AR) predictor — `â_j(t)=Σ_k w_{j,k}·a_j(t−d−k)`, ~5 coefficients
per mode, the ALTAIR-style linear forecaster ([ar_predictor.py](aowfs/predict/ar_predictor.py)).
Weights are fit **once** by least squares on a training segment; runtime is one
dot product per mode. Compared against the naive zero-order hold (`â(t)=a(t−d)`,
what an integrator effectively applies).

**Centerpiece result** — residual wavefront variance (→ Strehl via Maréchal),
naive vs predicted, swept over loop delay, on a held-out test segment:

| Loop delay | naive Strehl | predicted Strehl | residual reduction |
|---|--:|--:|--:|
| 1 frame (2 ms) | 0.94 | 0.96 | 43% |
| 2 frames (4 ms) | 0.83 | **0.91** | 51% |
| 3 frames (6 ms ≈ τ₀) | 0.70 | **0.86** | 58% |
| 4 frames (8 ms) | 0.56 | **0.78** | 58% |

(Realistic end-to-end numbers, from *measured* noisy centroids. With true
coefficients the reduction is 59–62%.) The benefit grows with delay — exactly
where servo lag hurts most — and the predictor holds Strehl ≈ 0.8 at a delay
where the naive loop has collapsed to 0.56.

**Why linear, not a neural net.** The one-step prediction residual is near-white
(lag-1 autocorrelation 0.17 vs 0.49 for the naive residual), i.e. the linear
model has already captured the predictable frozen-flow structure. This is the
evidence-based justification for treating a neural predictor as a discussed
stretch rather than a default — consistent with the ALTAIR finding that more
complex ML gave no perceptible gain over linear forecasting.

## Phase 5 — Actuator mapping (with inter-actuator coupling)

The DM applies the conjugate of the predicted wavefront. Each actuator has a
Gaussian influence function whose width is set directly by the DM's
inter-actuator coupling: `IF(r) = exp(ln(c)·(r/d_act)²)`, so a unit poke gives
exactly `c` at one actuator pitch ([influence_matrix.py](aowfs/actuator/influence_matrix.py)).
The forward matrix `H` maps strokes → mirror surface over the pupil; the command
matrix `C = pinv(H)` (truncated SVD, which suppresses poorly-controlled waffle
modes) is **precomputed once**, so the per-frame mapping is one matvec
([actuator_map.py](aowfs/actuator/actuator_map.py)). Strokes are clipped to the
mechanical limit and the clipped fraction is reported.

**Validation** (per SPEC — push the strokes back through the forward influence
model and compare to the target correction surface):

| Check | Result |
|---|--:|
| Coupling at one pitch (`IF(d_act)`) | exactly 0.15 ✓ |
| Fitting error, reconstructed wavefronts (reproduced vs target) | **3.6%** RMS |
| Fitting error, smooth low-order mode | <6% |
| Peak stroke (nominal turbulence) | 0.64 µm (limit 4 µm), 0% clipped |

The 17×17 Fried actuator grid reproduces the conjugate wavefront to ~96%, with
strokes comfortably inside the mechanical range.

## Phase 6 — Benchmarking & demo

The whole linear chain (reconstruct → synthesise phase → conjugate → project
onto actuators) folds into two precomputed matrices, so
[`RealTimePipeline.step`](aowfs/pipeline.py) reduces to: **centroid → `R_px`
matvec → AR forecast → `M_act` matvec**. Calibration cost is built once and
measured separately from the per-frame path ([benchmark.py](aowfs/bench/benchmark.py)).
The centroiding hot loop is numba-JIT'd (parallel over sub-apertures).

**Runtime benchmark** (calibration excluded — the graded number):

| Stage | Time |
|---|--:|
| centroid (numba, parallel) | 163 µs |
| reconstruct (`R_px` matvec) | 3.2 µs |
| predict (AR forecast) | 3.1 µs |
| actuator map (`M_act` matvec) | 4.3 µs |
| **full predictive step** | **≈210 µs → 4 700 fps** |

numba gives a **14×** speedup over vectorised numpy on centroiding (the only
non-trivial per-frame cost; the three matvecs are negligible). Calibration is a
one-time ~0.7 s. At τ₀ ≈ 5.9 ms the loop runs **≈28 iterations per coherence
time** — comfortably faster than the turbulence evolves, and well above typical
0.5–2 kHz AO loop rates.

**Dashboard** — `streamlit run aowfs/viz/app.py` replays the residual wavefront
reactive vs predictive side by side with live Strehl, r₀/τ₀ readouts, and
sliders for turbulence and loop delay (demoable in under two minutes). The
backing computation ([replay.py](aowfs/viz/replay.py)) is pure/tested.

## Phase 7 — ISRO real-data adapter

A thin [`io/isro_loader.py`](aowfs/io/isro_loader.py) reads the real deliverable
— a directory of `.bmp` frames plus the hardware specs — into the same
`WFSFrameSequence` the synthetic loader produces, so **Phases 1–6 run
unchanged**. The brief's "Data Required" list maps one-to-one onto an
`ISROSpecs` dataclass (loadable from [configs/isro_specs_template.yaml](configs/isro_specs_template.yaml));
turbulence/noise config fields are placeholders the algorithms never read.

The adapter handles the two real-data concerns explicitly: **spot-grid
registration** (crop to the `N·det_px_per_subap` grid, with `det_px_per_subap`
defaulting to `lenslet_pitch/pixel_size`; optional ROI origin) and the
**reference (flat-wavefront) positions** (`"frame"` from a supplied flat frame —
exact; `"mean"` time-averaged — approximate; or `"geometric"` cell centres).

```python
from aowfs.io import ISROSpecs, load_isro_dataset
specs = ISROSpecs.from_yaml("configs/isro_specs_template.yaml")
seq = load_isro_dataset("path/to/frames", specs, reference_mode="frame",
                        reference_frame_path="path/to/flat.bmp")
# seq now drives the identical Phase 1-6 pipeline.
```

**Validation.** Since the real dataset isn't yet available, the adapter is
validated by writing synthetic frames as a `.bmp` directory and loading them
back through the real-data path (frames + specs only, computing its own
reference): the loaded sequence is bit-identical to the native one and the
`RealTimePipeline` produces the **same actuator strokes to machine precision**.
When the real data arrives, only this file changes; the r₀/τ₀ and fps numbers
are then re-derived on it for the final submission.

## Phase 9 — Four reconstruction methods (incl. ML)

The PS9 explainer named four reconstruction approaches; all four are implemented:

| Method | Location | Note |
|---|---|---|
| Zonal | [recon/zonal.py](aowfs/recon/zonal.py) | Fried geometry, slopes → corner phases |
| Modal | [recon/modal.py](aowfs/recon/modal.py) | Zernike/Noll, slopes → coefficients |
| **Direct gradient control** | [pipeline.py](aowfs/pipeline.py) `control_matrix` | `M_act @ R_px` — slopes → strokes in one matvec (the reactive fast path; how SPHERE/Keck run) |
| **Machine learning** | [recon/ml.py](aowfs/recon/ml.py) | ridge slopes → coefficients (learned reconstruction matrix) |

**ML, and letting the evidence decide.** The ridge reconstructor is trained on
measured slopes from several independent realizations against clean coefficients
(honest train/test split), validated against the same Phase 2 checks (exact
noiseless tie with modal; matching Noll trend), and compared on a held-out
realization:

| Method | RMS error | rel. | inference/frame | DOF |
|---|--:|--:|--:|--:|
| Zonal | 0.204 rad | 7.0% | 62.7 µs | 241 |
| Modal | 0.357 rad | 12.2% | 20.8 µs | 45 |
| ML (ridge) | 0.361 rad | 12.4% | 22.2 µs | 45 |

ML **ties** the analytic modal reconstructor and beats nothing — the expected
result for a well-conditioned linear inverse, where the optimal linear
reconstructor *is* the analytic pseudo-inverse. So the operational pipeline keeps
the analytic reconstructors, and no neural variant is pursued (no structured
residual to exploit). This is the second "tested the fancier option, evidence
said no" finding, after the Phase 4 AR-vs-NN result. Full note:
[docs/phase9_notes.md](docs/phase9_notes.md).

## Phase 10 — Real-world AOT stress test (on-sky NAOMI@VLT)

A robustness check on real on-sky telemetry (no injected ground truth).
[`io/aot_loader.py`](aowfs/io/aot_loader.py) ingests the AOT FITS format
(arXiv:2312.08300) — validated against **NAOMI3** (VLT AT3, D=1.82 m) — and emits
the same `WFSFrameSequence`, so Phases 1–6 run unchanged. It aligns raw frames to
slopes by the AOT `FRAME_NUMBERS` field (2998/3095 matched; the pixel stream
overruns the loop stream, so a naive 1:10 ratio mis-maps the tail), applies
dark/flat/bad-pixel calibration, and reproduces the instrument's 12-valid-sub-ap
4×4 geometry.

**Findings** (full note: [docs/phase10_notes.md](docs/phase10_notes.md)):
- **Centroiding cross-check (verified):** our centroids on the real frames match
  the instrument's stored residual slopes in magnitude (0.075 vs 0.066 px),
  order, axis and sign, correlating ~0.70 per-sub-ap (0.86 on the high-SNR tilt
  mode). The ~0.7 (not ~0.95) traces to closed-loop residual near the noise floor
  + SPARTA's weighted centroid vs our thresholded CoM + a ~0.05 px readout-
  direction CoM bias (a directional-EMCCD effect synthetic Gaussian spots never
  exposed). The pipeline runs on real frames and tracks the real RTC.
- **r₀/τ₀ (gap documented):** absolute r₀-vs-seeing verification is gated by (1)
  closed-loop telemetry needing a pseudo-open-loop reconstruction and (2) the
  shared AOT file omitting the WFS pixel scale / physical modal units — reported
  as findings, not hidden, per the robustness-check framing.

`detector.frame_rate` is cited from the header (`ESO AOS LOOP RATE = 500.149`,
SPARTA RTC); the ALPAO **DM241** actuator count (241) is flagged as *coincidentally*
equal to our synthetic Fried grid's 241 corner DOF, not a shared quantity.

## Quickstart

```bash
pip install -r requirements.txt
python -m aowfs.sim.validate      # Phase 0 closed-loop validation
pytest tests/ -v                  # per-phase ground-truth tests
python -m aowfs.bench.benchmark   # runtime fps benchmark (calibration excluded)
streamlit run aowfs/viz/app.py    # interactive reactive-vs-predictive demo
```

```python
from aowfs import SimConfig
from aowfs.io import from_synthetic, save_dataset, load_sequence

cfg = SimConfig.from_yaml("configs/default.yaml")
ds = from_synthetic(cfg)                 # in-memory: sequence + ground truth
save_dataset(ds, "data/run01")           # write .bmp frames + metadata (real-data shape)
seq = load_sequence("data/run01")        # reload as the ISRO loader will (no ground truth)
```

## Literature grounding

Gemini ALTAIR linear AR forecasting (≈5 coefficients/mode, with more complex ML
giving no perceptible gain); LQG/Kalman vibration filtering (Petit et al.,
~2008); empirical-orthogonal-function predictors on operational real-time AO
computers (Guyon et al., 2018); CNN-based predictive control as state-of-the-art
context (2021). We deliberately start from the operationally-proven simple
approach and treat the NN version as a discussed, evidence-gated stretch goal.
