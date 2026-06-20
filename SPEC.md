# SPEC: Predictive AO Wavefront Reconstruction Pipeline

## 0. Problem context (for reference — do not re-derive, just implement against this)

Atmospheric turbulence distorts an incoming plane wavefront. A Shack-Hartmann
Wavefront Sensor (SH-WFS) samples it with a microlens array (MLA): each
lenslet focuses its sub-aperture of the wavefront onto a detector, forming a
spot whose displacement from a calibrated reference position is proportional
to the local wavefront slope over that sub-aperture. From the full array of
slopes we reconstruct the wavefront phase map, characterize turbulence
strength (Fried parameter r0, coherence time τ0), and convert the conjugate
of the wavefront into an actuator stroke map for a deformable mirror (DM),
accounting for inter-actuator coupling. The actuator grid and lenslet grid
are arranged in a Fried geometry.

**Dataset ISRO will supply** (treat as the eventual real input; build the
synthetic generator in Phase 0 to match this exact shape so swapping later
is trivial):
- A time series of SH-WFS frames (.bmp), sampled a few milliseconds apart
- Pixel size and frame resolution
- MLA info: physical size, number of lenslets, focal length
- Pupil size of the turbulated beam
- DM info, including the inter-actuator coupling model

**Evaluation criteria stated in the brief** (design every phase to score
well on these specifically):
- Reconstructed wavefront maps must conform to the actual turbulence
  characteristics (i.e., be physically correct, not just plausible-looking)
- Correct derivation of r0 and τ0
- Speed and computational efficiency of the algorithms

**Our added scope** (the differentiator, not requested explicitly but
directly scored well by "speed/efficiency" and "conforms to turbulence
characteristics," since predictive control is fundamentally about modeling
how turbulence evolves in time): a predictive layer that estimates the
wavefront 1-2 frames ahead instead of just reacting to the last measurement,
to reduce servo-lag error — the same problem real systems solve. Ground this
in the literature when writing up results: Gemini ALTAIR's adaptive optics
system achieved strong lag reduction with a simple linear auto-regressive
forecasting module (~5 coefficients per mode), and follow-up testing found
more complex ML models gave no perceptible improvement over that linear
baseline — meaning starting simple is empirically justified, not a
shortcut. The more research-grade lineage (LQG/Kalman vibration filtering,
empirical-orthogonal-function predictors used operationally on real-time AO
computers, and CNN-based predictive controllers showing large Strehl
improvements on faint guide stars) is good context to cite as the field this
sits in, even if you only implement the linear/Kalman version.

## 1. Repository layout

```
project/
  sim/            synthetic phase-screen + SH-WFS frame generator, validate.py
  io/              data loaders — synthetic_loader.py now, isro_loader.py later
  recon/           centroiding.py, zonal.py, modal.py (Zernike/Noll basis)
  turbulence/      r0_estimator.py, tau0_estimator.py
  predict/         ar_predictor.py, kalman_predictor.py, (nn_predictor.py if justified)
  actuator/        influence_matrix.py, actuator_map.py
  bench/           timing harness, separates calibration-time vs runtime-path cost
  viz/             app.py (streamlit dashboard), plots.py
  tests/           one validation test per phase, against synthetic ground truth
  requirements.txt
  README.md
```

## 2. Phase 0 — Synthetic SH-WFS simulator

Goal: generate data shaped exactly like what ISRO will hand over, but with
known ground truth, so every later phase can be validated before real data
exists.

Steps:
- Generate Kolmogorov (or von Kármán) phase screens via the standard
  FFT-based method, or use `aotools`'s built-in turbulence screen generator
  if it fits — don't hand-roll this if a maintained library already does it
  correctly.
- Inject a known, configurable r0 and a wind vector that translates the
  screen frame-to-frame (frozen-flow assumption) so τ0 has a real,
  recoverable value.
- Simulate the MLA: tile the pupil into an N×N grid of sub-apertures,
  compute the local tilt of the phase screen over each sub-aperture, and
  render a synthetic spot (Gaussian blob displaced according to that local
  tilt) into a detector image, with configurable photon and read noise.
- Output frames in the same conceptual shape as the real dataset (pixel
  size, frame resolution, MLA lenslet count/size/focal length, pupil size
  all as configurable parameters, not hardcoded).

Validation: you must be able to dial in a specific r0 and wind speed, run
the generator, and later (Phase 3) recover values close to what you put in.
If you can't close this loop, nothing downstream is trustworthy.

## 3. Phase 1 — Centroiding

- For each sub-aperture, extract its spot sub-image and compute the
  centroid via thresholded center-of-mass against a calibrated reference
  (flat-wavefront) position. This is the baseline — get it working first.
- Stretch: correlation-based centroiding, more robust under low light /
  high noise; only build this after the baseline works end-to-end.
- Output: per-frame array of (dx, dy) slope measurements per sub-aperture.

Validation: feed the generator a flat (zero-turbulence) phase screen —
recovered centroid offsets should be ~0, bounded only by injected noise.

## 4. Phase 2 — Wavefront reconstruction (zonal + modal)

- Zonal: build the geometry matrix relating sub-aperture slopes to phase
  values at grid points under the Fried geometry (Southwell/Hudgin-style
  formulation), solve via least squares.
- Modal: project slopes onto the gradient of Zernike polynomials (Noll
  indexing) via the standard slope-to-Zernike least-squares matrix, solve
  for the Zernike coefficient vector per frame. `aotools` has Zernike basis
  utilities — use them rather than re-deriving polynomials from scratch.
- Precompute the reconstruction matrix (pseudo-inverse, with SVD truncation
  to suppress noise blow-up in poorly sensed / waffle modes) ONCE. Per-frame
  reconstruction from then on is a single matrix-vector multiply — this is
  the single most important architectural decision for the efficiency
  score, do not skip it or compute it implicitly per frame.
- Output: wavefront phase map W(x,y) per frame, plus the Zernike
  coefficient vector per frame.

Validation: on synthetic data, recovered wavefront RMS error vs. the
injected ground-truth phase screen should shrink as you correct more
Zernike modes, matching the expected theoretical trend — if it doesn't, the
reconstruction matrix is wrong before you do anything else.

## 5. Phase 3 — Turbulence statistics (r0, τ0)

- r0: from the variance of the reconstructed (piston/tip-tilt-removed, or
  higher-order) phase or Zernike coefficients, using the standard relation
  between residual phase variance and (D/r0)^(5/3).
- τ0: from the temporal autocorrelation of the Zernike coefficient time
  series (tip-tilt is usually most informative), or by fitting the power
  spectral density and locating where it breaks from the expected
  Kolmogorov −8/3 slope.

Validation: recovered r0 and τ0 on synthetic data should match the
ground-truth values injected in Phase 0 within a defined tolerance (start
with ~10-20%, tighten if you have time). Report this as a table — this is
the single most ISRO-judge-legible piece of evidence that your science is
correct.

## 6. Phase 4 — Predictive wavefront layer (the differentiator)

- Per Zernike mode, fit a low-order linear autoregressive model (AR(p),
  p≈5) on recent coefficient history. This is the baseline predictive
  layer — implement and validate this FIRST.
- Stretch: reformulate as a Kalman filter (state = mode amplitude +
  velocity) for a more principled estimate; this is the LQG-control
  lineage and is a reasonable upgrade if the AR model works but you want
  something more defensible to discuss in the writeup.
- Only build a neural predictor (small GRU/LSTM per mode, or jointly
  across modes) if residual analysis shows the AR/Kalman model is leaving
  structured, non-white residuals — i.e., justify it with evidence, don't
  default to it.
- At each timestep, predict 1-2 frames ahead (matching a configurable loop
  delay parameter) and use that PREDICTED wavefront — not the
  just-measured one — as the input to the actuator-mapping step.

Validation: with frozen-flow turbulence injected in Phase 0, measure
residual wavefront error (or implied Strehl ratio via the Maréchal
approximation, Strehl ≈ exp(−σ²) with σ in radians) with vs. without the
predictive layer, across a sweep of injected loop delays. Report % error
reduction. This comparison plot is the centerpiece of the whole project —
it's the one result nobody else attempting this PS will have, because the
brief never asked for it.

## 7. Phase 5 — Actuator mapping

- Build the influence-function matrix mapping actuator pokes to local
  mirror surface deformation, incorporating the inter-actuator coupling
  model from the supplied DM info.
- Invert it (pseudo-inverse / truncated SVD) to convert a target wavefront
  correction (the conjugate of the predicted wavefront) into actuator
  stroke commands.

Validation: apply the computed actuator map back through the forward
influence-function model — it should reproduce the target correction
surface within a small fitting-error tolerance. Report this residual.

## 8. Phase 6 — Benchmarking and demo

- Strictly separate calibration-time cost (matrix construction, inversion —
  done once at startup) from the runtime path (per-frame: centroid → one
  matrix-vector multiply → predict → one matrix-vector multiply). Report
  frames-per-second for the runtime path only — that's the number that maps
  to the brief's "computational efficiency" criterion and to the real
  atmospheric coherence-time constraint (milliseconds).
- Use numpy vectorization and numba JIT on the hot loop. If time remains,
  porting the hot loop to C/C++ via pybind11 is a legitimate stretch goal,
  since the brief explicitly suggests a low-level language for speed — but
  don't attempt this before the Python version is correct and benchmarked.
- Produce: reconstructed wavefront map plots, the r0/τ0 validation table,
  the predictive-vs-naive residual error comparison plot, and the fps
  benchmark table.
- Build a small dashboard (streamlit is fine) replaying wavefront
  correction with and without the predictive layer side by side, so this
  is demoable in under two minutes.

## 9. Phase 7 — Real-data adapter

Once ISRO supplies the actual frame time series and MLA/DM specs, write a
thin `io/isro_loader.py` that reads their files into the exact same internal
data structures the synthetic loader produces. Phases 1-6 should run
unchanged against it. Re-derive r0/τ0/benchmark numbers on real data before
the official submission — the synthetic numbers are for development and
validation, the real numbers are what go in front of judges.

## 10. Phase 8 — Fix known flaws (do this before anything else below)

Three concrete issues found during review of the dashboard. Fix each, with
a brief written note of what was wrong and what changed — these notes go
straight into the proposal as evidence of a rigorous review process.

- **Label ambiguity, Strehl.** The dashboard's headline "Strehl — reactive /
  predictive" metrics are the mean across all test frames; the wavefront
  panel titles below them ("residual — reactive (Strehl X)") are the value
  for the single frame currently selected by the frame slider. These are
  both correct but currently unlabeled, which reads as an inconsistency to
  anyone who hasn't built the app (it read as one to an outside reviewer
  here). Relabel explicitly: "Strehl — reactive (mean, N frames)" and
  "Strehl (this frame)" respectively.
- **Unexplained Strehl dip.** In the wind=3.0 m/s, loop-delay=2 scenario,
  the reactive Strehl curve drops sharply across roughly frames 40-90
  before partially recovering, while the predictive curve stays flat
  through the same stretch. Trace this back to the actual injected phase
  screen / wind trajectory for that test segment and identify what's
  different there (a locally stronger turbulence patch in the single
  frozen-flow realization, a wind-direction artifact, etc.). Document the
  explanation — an AO judge is very likely to ask "why does the red line
  dip there," and "we traced it to X" is a far stronger answer than "not
  sure."
- **Fast-wind r0 outlier.** The fast-wind validation scenario recovered r0
  at +14% error, the one number outside the 5-10% target band the other
  three scenarios hit. Investigate whether this is purely single-screen
  realization noise (the current working explanation) or whether it
  tightens with a narrower Zernike mode-fitting range or better aliasing
  handling. Either confirm the existing explanation with a quick
  multi-seed check, or fix it — document whichever happens.

## 11. Phase 9 — Fourth reconstruction method: ML, and naming what you already have

The PS9 mentor's explainer session named four valid wavefront
reconstruction approaches: zonal, modal, direct gradient control, and
machine learning. You have zonal and modal (Phase 2). You already have
direct gradient control too, you just haven't labeled it as such: Phase
6's `RealTimePipeline`, which folds reconstruction and actuator projection
into precomputed matrices R_px and M_act, IS a direct slopes-to-actuator
control matrix — exactly what "direct gradient control" means in real AO
systems (this is literally how SPHERE/Keck-class systems run their fast
path). Add a short, explicit note in the code and in the proposal stating
this plainly — no new logic required, just naming what's already there.

What's missing is genuine ML-based reconstruction (mapping slopes directly
to Zernike coefficients or actuator commands via a trained model, as
opposed to the Phase 4 predictive layer, which forecasts in time, not
reconstructs from slopes). Build a fourth reconstructor:

- Train a simple model (start with ridge regression slopes→Zernike
  coefficients; escalate to a small MLP only if ridge underperforms) on
  the same synthetic slope/coefficient pairs already used to validate
  Phase 2, with an honest train/test split.
- Validate it against the exact same checks Phase 2 used: the Noll-variance
  trend on noiseless data, and RMS wavefront error vs. ground truth on
  noisy data, across the same scenarios.
- Report a direct, honest comparison table: zonal vs. modal vs. ML, on
  accuracy and on per-frame inference time. If ML loses to the existing
  linear reconstructors — plausible, since this is a well-conditioned
  linear problem — report that as the result, in the same spirit as the
  Phase 4 AR-vs-NN finding. A second instance of "we tested the
  fancier-sounding option and let the evidence decide" is a stronger
  narrative than the comparison being absent.

## 12. Phase 10 — Real-world data stress test

Before committing real effort, inspect file structure first: AOT-format
files (FITS-based) may contain only already-extracted slopes or
reconstructed coefficients rather than raw WFS pixel frames, depending on
what the original instrument team chose to share. Check the actual FITS
extensions/HDUs for a raw image/pixel array before building a conversion
pipeline — if only slopes are present, this only stress-tests Phases 2
onward, not Phase 1's centroiding, and the plan below should be scoped
down accordingly.

- Pull a real AOT-format dataset (PAPYRUS or NAOMI/ERIS@VLT, via the
  Zenodo DOIs from the AOT paper, arXiv:2312.08300, or the ESO Archive).
  Confirm via inspection whether raw frames are present.
- If raw frames are present, write a `io/aot_loader.py` adapter analogous
  to `isro_loader.py` (FITS → the same internal `WFSFrameSequence`
  structure), and run the full pipeline (Phases 1-6) against it unchanged.
- Important framing difference from synthetic validation: there is no
  injected ground truth for real on-sky data, so this isn't a
  pass/fail accuracy check — it's a robustness check. Does the pipeline
  run without crashing, produce physically plausible r0/τ0/Strehl numbers
  for the seeing conditions reported by the instrument team, and degrade
  gracefully rather than silently? Document whatever you find, including
  anything that breaks — that's the point of the exercise.
- Separately, grab `sh_flat.mat` from github.com/jacopoantonello/mshwfs
  (`examples/data/`) as a five-minute sanity check specifically on grid
  registration against one real, hardware-captured reference frame — note
  this toolbox has reported documentation friction, so don't sink much
  time into it beyond this single check.

## 13. Phase 11 — Optional stretch goals (only if time remains after Phase 10)

- **Live-sensor-ready architecture.** Formalize a `FrameSource` interface
  (`get_next_frame()`) with your existing synthetic and file-based loaders
  as two concrete implementations, plus an `ActuatorSink` interface
  (`send_command(strokes)`) currently implemented as logging/visualization
  only. Add a stub `LiveCameraFrameSource` that raises "not yet
  implemented — see interface contract" rather than failing silently.
  This is an architecture-readiness claim, not a tested-with-hardware
  claim — be explicit about that distinction in any writeup.
- **CPU vs. GPU, empirically.** You've reasoned that GPU would be slower
  than your current CPU/Numba path at this problem's array size (small
  matrices mean kernel-launch and host-device transfer overhead would
  dominate over actual compute). Turn the argument into a measurement:
  benchmark a naive GPU matrix-vector multiply (CuPy or PyTorch, if a GPU
  is available in your environment) at the same array sizes as R_px/M_act,
  and report the comparison. A third instance of "we tested the
  fancier-sounding option and the evidence said no" reinforces the same
  pattern as Phases 4 and 9.

## 14. Deliverables checklist for the ISRO idea-submission writeup

- One-paragraph architecture summary + the repo's module diagram
- r0/τ0 validation table: injected ground truth vs. recovered, with error %
- The predictive-vs-naive residual wavefront error comparison plot, swept
  across loop delay
- The zonal vs. modal vs. ML reconstruction comparison table (Phase 9)
- An explicit statement that the folded real-time pipeline is the
  "direct gradient control" method named in the explainer session
- The calibration-time vs. runtime-path fps benchmark table
- A short note on the real-data stress test (Phase 10): what was tested,
  what held up, what didn't, framed as evidence of rigor rather than hidden
- A short literature-grounding paragraph: Gemini ALTAIR linear AR
  forecasting; LQG/Kalman vibration filtering (Petit et al., ~2008);
  empirical orthogonal function predictors on real-time AO computers
  (Guyon et al., 2018); CNN-based predictive control as state-of-the-art
  context (2021) — framed as "we deliberately start from the
  operationally-proven simple approach and treat the NN version as a
  discussed stretch goal," not as a list of citations for their own sake
