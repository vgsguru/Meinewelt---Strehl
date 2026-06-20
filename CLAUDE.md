# Project: Predictive Adaptive-Optics Wavefront Reconstruction (BAH 2026 — PS9)

## WHY
This problem statement is presented by Jay Chitroda, Astronomy & Astrophysics
Division, Physical Research Laboratory (PRL), Ahmedabad. PRL's A&A division
has explicitly stated it has initiated "a program on adaptive optics system
development... with the aim to develop adaptive-optics-assisted near-infrared
instrumentation on PRL telescopes" (PRL AATO labs page) — referring to its
1.2m and 2.5m telescopes at the Mt. Abu Infrared Observatory (Gurushikhar).
The 2.5m telescope already runs *active* optics (42 axial + 18 lateral
actuators correcting slow primary-mirror figure) — this problem statement is
about the next step: *adaptive* optics, correcting fast atmospheric
turbulence in real time. This is very likely real groundwork feeding PRL's
actual in-house AO program, not a purely academic exercise — frame the
project accordingly, and be precise about the active-optics-vs-adaptive-optics
distinction since it signals genuine domain understanding.

We are deliberately going beyond the literal ask in the brief (which is
satisfied by classical zonal/modal reconstruction alone) by adding
*predictive* wavefront control to fight servo-lag, a real published
technique in AO research (Gemini ALTAIR linear forecasting, LQG/Kalman
control, EOF predictors, CNN-based predictive control) — not an ML buzzword
bolt-on. See @SPEC.md for full grounding and targets.

## WHAT
A Python pipeline that:
1. Generates synthetic SH-WFS data with known ground-truth turbulence
   (so we can validate correctness before ISRO's real dataset arrives)
2. Centroids spots, reconstructs the wavefront per frame (zonal + modal)
3. Derives turbulence statistics (Fried parameter r0, coherence time τ0)
4. Predicts the wavefront 1-2 frames ahead per Zernike mode (linear AR /
   Kalman, escalate to a small NN only if justified)
5. Converts the predicted wavefront into DM actuator stroke commands,
   respecting inter-actuator coupling
6. Benchmarks runtime performance with calibration-time work strictly
   separated from the per-frame real-time path
7. Swaps in ISRO's real dataset via a thin adapter once supplied

## HOW
See @SPEC.md for the full phase-by-phase build plan, success criteria per
phase, and the deliverables checklist for the ISRO idea-submission writeup.

Work ONE phase at a time. For each phase: propose a short plan, implement
it, run that phase's validation check, show me the result, and stop before
moving to the next phase. Do not implement multiple phases in one pass.

## Tech stack
Python 3.11+, numpy, scipy, numba, aotools (pip install aotools), filterpy
or statsmodels (for AR/Kalman), matplotlib, streamlit (for the dashboard),
pytest.

## Commands
- Install deps: `pip install -r requirements.txt`
- Run unit/validation tests: `pytest tests/ -v`
- Run synthetic end-to-end validation: `python -m sim.validate`
- Launch dashboard: `streamlit run viz/app.py`

## Conventions
- Keep data ingestion (`sim/`, `io/`) strictly separate from algorithms
  (`recon/`, `turbulence/`, `predict/`, `actuator/`). Swapping synthetic
  data for ISRO's real data later must only touch the `io/` layer.
- Precompute every calibration matrix (reconstruction matrix, DM influence
  matrix and its inverse) ONCE at startup. The per-frame runtime path must
  be a single matrix-vector multiply — never re-derive a calibration matrix
  inside the per-frame loop. This is graded explicitly (computational
  efficiency is an evaluation criterion in the brief).
- Don't reach for a neural predictor before checking whether a linear
  AR/Kalman model already captures the benefit — report the comparison,
  don't assume bigger is better.
- Every phase needs a number, not a claim: r0/τ0 recovery error against
  known synthetic ground truth, residual wavefront error with vs. without
  prediction, and measured frames-per-second.
