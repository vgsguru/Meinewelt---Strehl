# Phase 9 — Four reconstruction methods (proposal notes)

The PS9 explainer session named four valid wavefront-reconstruction approaches:
**zonal, modal, direct gradient control, and machine learning.** This project
implements all four. Three were already present; this phase names the third
explicitly and adds the fourth as an honest, evidence-decided comparison.

## (a) Direct gradient control — naming what was already there

The Phase 6 real-time pipeline folds reconstruction and actuator projection into
two precomputed matrices, `R_px` (centroids → Zernike coefficients) and `M_act`
(coefficients → actuator strokes). Their product is a single **slopes-to-actuator
control matrix**:

```
control_matrix = M_act @ R_px            # (n_actuators × 2·n_subaps)
strokes        = control_matrix @ slopes # the reactive fast path: ONE matvec
```

This *is* the "direct gradient control" method — mapping measured gradients
straight to actuator commands through one precomputed matrix, with no explicit
wavefront ever formed in the loop. It is exactly how SPHERE/Keck-class systems
run their fast path. No new logic was needed: the matrix was already implicit in
the folded pipeline; Phase 9 exposes it as `RealTimePipeline.control_matrix` with
a named `direct_command(frame)` entry point ([aowfs/pipeline.py](../aowfs/pipeline.py)),
verified to equal the reactive `step(..., predict=False)` to machine precision.

So the four methods map onto the codebase as:

| Explainer method | Where it lives |
|---|---|
| Zonal | `recon/zonal.py` (Fried geometry, slopes → corner phases) |
| Modal | `recon/modal.py` (Zernike/Noll, slopes → coefficients) |
| **Direct gradient control** | `pipeline.py` `control_matrix` = `M_act @ R_px` (slopes → strokes) |
| **Machine learning** | `recon/ml.py` (ridge slopes → coefficients) — below |

## (b) The fourth reconstructor — ML, and what the evidence says

`recon/ml.py` adds a genuine ML reconstructor: ridge regression mapping slopes
directly to Zernike coefficients (a *learned* reconstruction matrix
`coeffs = W @ slopes`), distinct from the Phase 4 predictive layer (which
forecasts in *time*, not reconstructs from slopes). It is trained on measured
slopes from **five independent screen realizations** (1000 samples — a single
frozen-flow sequence is rank-deficient and starves the fit) against clean target
coefficients, with an honest train/test split, then evaluated on a **held-out
realization** exactly as zonal and modal are. Reproduce with
[phase9_ml_compare.py](phase9_ml_compare.py).

### Validation against the Phase 2 checks
- **Noiseless tie-check.** Trained on noiseless slopes, ML's residual variance
  equals modal's to 4 decimals (0.1206 vs 0.1206 rad²) — ridge recovers the
  analytic pseudo-inverse on this linear problem, as it must.
- **Noll-variance trend (noiseless).** ML residual variance falls 0.955 → 0.518
  → 0.255 → 0.150 rad² over 5 → 10 → 20 → 35 modes, tracking the same Noll trend
  modal passed in Phase 2.

### Comparison table (held-out realization, noisy; lower RMS error is better)

| Method | RMS wavefront error | rel. to truth | inference / frame | DOF |
|---|--:|--:|--:|--:|
| Zonal (Fried) | 0.204 rad | 7.0% | 62.7 µs | 241 |
| Modal (Zernike, analytic pinv) | 0.357 rad | 12.2% | 20.8 µs | 45 |
| **ML (ridge, learned)** | **0.361 rad** | **12.4%** | **22.2 µs** | 45 |

**On the inference times.** The 62.7 / 20.8 / 22.2 µs figures are measured by an
*unoptimized standalone micro-benchmark* (a Python-loop `M @ slopes` timed
identically for all three methods), chosen so the cross-method *relative*
comparison is apples-to-apples for the accuracy study. They are **not** the
production per-frame cost: in the deployed `RealTimePipeline` the reconstruction
is folded into precomputed matrices and JIT-compiled, and the Phase 6 runtime
benchmark measures that folded reconstruct matvec at **~3–4 µs/frame** (with the
full step ~210 µs, dominated by centroiding). Both numbers are real and correct;
they differ because they measure different things — an isolated micro-benchmark
for fair cross-method ranking here, versus the integrated optimized loop in
Phase 6. Whichever reconstructor is deployed, the production path runs at the
Phase 6 figure.

### Result
At the same 45-mode resolution, **ML ties the analytic modal reconstructor**
(12.4% vs 12.2%, identical inference cost) and does not beat it. This is the
expected outcome for a well-conditioned *linear* inverse problem: the optimal
linear reconstructor is the analytic pseudo-inverse, and ridge can at best
re-learn it — at the cost of needing thousands of training samples and a tuned
regularizer that first-principles modal does not. Zonal is the most accurate
here (more degrees of freedom) but the slowest.

This is the second instance — after the Phase 4 AR-vs-neural-network finding —
of deliberately testing the fancier-sounding option and **letting the evidence
decide**: ML earns no accuracy or speed advantage on this linear reconstruction
task, so the operational pipeline stays on the analytic linear reconstructors. A
neural (MLP) variant is therefore *not* pursued: there is no structured residual
for it to exploit on a problem the linear model already solves optimally.
