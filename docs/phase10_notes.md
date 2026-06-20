# Phase 10 — Real-world AOT-format stress test (proposal notes)

A robustness check, **not** a pass/fail accuracy check: there is no injected
ground truth for on-sky data. The questions are whether the pipeline runs on
real telemetry, whether what it produces is physically plausible, and whether it
degrades gracefully — and to document whatever is found, including what breaks.

## Dataset

The AOT example collection (arXiv:2312.08300; ESO/AOT teams), 11 files / 5
instruments / 1.2 GB. Shack-Hartmann instruments (in scope): CIAO×4, **NAOMI**×3,
ERIS-NGS, GALACSI. **PAPYRUS is a pyramid WFS — out of scope** (different slope
model). Validated against **NAOMI3** (ESO VLT Auxiliary Telescope 3, D = 1.82 m,
SCAO, natural guide star).

## What the file contains (inspected via aotpy + astropy)

Raw frames **are** present, so the full pipeline (Phases 1–6) is in scope:

| Product | HDU / field | Shape |
|---|---|---|
| Raw detector frames | `pixel_intensities` | (3095, 64, 64) uint16 |
| Residual slopes | `measurements` | (30000, 2, 12) |
| Residual modal coeffs | `MODAL COEFFICIENTS` | (30000, 14) |
| DM positions | `DM POSITIONS` | (30000, 241) |
| Dark / flat / bad-pixel / weight / sky | detector | (64, 64) each |

**Documented instrument facts** (cited, not assumed): NAOMI's SPARTA real-time
computer runs the loop at ~500 Hz — corroborated directly by the FITS header
`ESO AOS LOOP RATE = 500.149`; the deformable mirror is an **ALPAO DM241**
(241 actuators), consistent with the `(N, 241)` telemetry arrays. **Flag:** that
241 is the DM's actuator count and is *unrelated to* — though numerically equal
to — the 241 zonal corner DOF of our synthetic 16×16 Fried grid. The coincidence
is noted so it is not misread as a shared quantity.

## `io/aot_loader.py` — what it does

Emits the same `WFSFrameSequence` as the synthetic/ISRO loaders, so Phases 1–6
run unchanged. Three real-data steps:

1. **Frame-number alignment.** Raw frames and slopes log on different cadences
   (pixels every 10th loop frame) and the pixel stream **overruns** the loop
   stream (last pixel frame 83160 > last loop frame 82196). Aligning by the AOT
   `FRAME_NUMBERS` field — *not* an assumed 1:10 ratio — yields **2998 of 3095**
   matched pairs (97 tail frames dropped); exact frame-number match verified. A
   naive "every 10th slope" would have mis-mapped the tail.
2. **Detector calibration.** Dark pedestal (~129 ADU) subtracted, flat applied,
   bad-pixel map zeroed — before centroiding.
3. **Geometry.** 4×4 lenslets, **12 valid sub-aps** (4 corners excluded); our
   circular-pupil `build_geometry(N=4)` reproduces exactly 12, matching the
   telemetry's `n_valid_subapertures`.

## Result 1 — centroiding cross-check (fully verified)

Our centroider on the calibrated raw frames vs the instrument's own stored
residual slopes (frame-aligned, N = 1500):

- **Magnitude matches**: our residual slope std **0.075 px** vs stored **0.066 px**.
- **Order/axis/sign all match** with no permutation (our sub-ap *k*, x→x, y→y, +sign).
- **Correlation**: per-sub-aperture mean **~0.70**; high-SNR global **tilt mode 0.86**;
  overall flat **~0.59**.

Why ~0.7 and not ~0.95 — a genuine real-data finding synthetic data could not
surface: (a) NAOMI ran **closed-loop**, so the slopes are *residual* (~0.07 px,
near the centroiding noise floor → SNR ≈ 1–3, where two centroiders correlate
only moderately); (b) **algorithm difference** — SPARTA uses a *weighted*
centre-of-gravity with the detector weight map, we use thresholded CoM; (c) our
CoM shows a **~0.05 px common-mode bias in the x (detector-readout) direction**
(our global tip std 0.048 px vs the instrument's 0.009 px) — the classic
directional EMCCD charge-transfer asymmetry that a weighted, calibrated centroid
removes but symmetric synthetic Gaussian spots never exposed. Verdict: the
pipeline **runs on real frames and tracks the real RTC**, and the stress test
surfaced two concrete centroiding refinements (apply the weight map; correct the
readout-direction bias) — exactly the point of the exercise.

## Result 2 — turbulence parameters (status: gap documented)

- **r₀ is exactly defined by the reported seeing**: ASM seeing 0.53″ @ 500 nm ⇒
  r₀ = **19.1 cm @ 500 nm**, D/r₀ ≈ 9.5 — the verification target.
- **τ₀ / wind are approximate references only**: ASM reports τ₀ = 16.5 ms,
  wind 7.18 m/s @ 229°, but these derive from a *multi-layer* turbulence profile
  that a single wind number cannot reproduce, so any single-layer estimate from
  the telemetry can only be expected to agree in order of magnitude.
- **What blocks the absolute r₀ check (two real findings):**
  1. NAOMI ran **closed-loop**, so the telemetry holds *residual* slopes and the
     *applied* DM modes (`DM2M @ DM_positions` reproduces `MODAL COEFFICIENTS`
     exactly — confirming those are the correction, not an independent residual).
     Recovering the *atmospheric* wavefront needs a pseudo-open-loop
     reconstruction `m_open = REC.CM·(residual slopes) + applied modes`, with the
     one-frame measurement→command delay handled correctly.
  2. **Absolute scale is missing from the shared file**: `detector.pixel_scale`
     is `None` and the modal coefficients are in the RTC's internal command
     units (no physical wavefront unit), so converting telemetry to a metric r₀
     requires the WFS pixel scale (rad/px) / modal-basis calibration that this
     AOT export does not include. This is itself a finding about AOT shared
     datasets, consistent with the format making pixel/scale metadata optional.

  Next step (scoped, not yet done): implement the pseudo-open-loop modal
  reconstruction and verify the recovered modal-variance spectrum follows the
  Kolmogorov mode-scaling (a **scale-free** shape check that confirms the
  recovered wavefront is physically turbulence-like even without absolute
  units), then attach an absolute r₀ only if a documented NAOMI pixel scale is
  sourced.

## Bottom line

The pipeline ingests and runs on real on-sky AOT telemetry unchanged through the
`io/` adapter; centroiding magnitude/geometry/sign match the instrument and
correlate ~0.7 with its RTC, with the discrepancy traced to closed-loop residual
SNR, the weight-map/centroid-algorithm difference, and a readout-direction CoM
bias. Absolute r₀/τ₀-vs-seeing verification is gated by closed-loop pseudo-open-
loop reconstruction plus a pixel-scale calibration the shared file omits — both
documented here rather than hidden.
