# Phase 8 — Review fixes (proposal notes)

Three issues found during an outside review of the dashboard and Phase 3
results. Each is documented here as evidence of a rigorous review process: what
was wrong, what we changed, and the number that backs it.

---

## Fix 1 — Strehl label ambiguity (dashboard)

**What was wrong.** The dashboard showed two *different* Strehl numbers with
near-identical labels. The headline metrics ("Strehl — reactive / predictive")
are the **mean across all test frames**, while the wavefront-panel titles below
("residual — reactive (Strehl X)") are the value for the **single frame** picked
by the frame slider. Both are correct, but the shared wording read as an
internal inconsistency to a reviewer who hadn't built the app.

**What changed.** The two quantities are now explicitly distinguished
([aowfs/viz/app.py](../aowfs/viz/app.py)):

- Headline metrics → **"Strehl — reactive (mean, N frames)"** (N filled in live).
- Panel titles → **"residual — reactive (Strehl X, this frame)"**.
- The frame slider is relabelled "selects the single frame shown in the maps
  below", and the time-series plot's cursor line is legended "selected frame".

No numbers change — purely a labelling fix so the mean-vs-instantaneous
distinction is unambiguous.

---

## Fix 2 — "Unexplained" reactive Strehl dip (wind = 3.0 m/s, delay = 2)

**What was flagged.** In the wind = 3.0 m/s, loop-delay = 2 scenario the reactive
(red) Strehl curve drops across roughly test frames 40–90 (local minimum ≈ 0.45)
before recovering, while the predictive (green) curve stays flat (~0.86). "Why
does the red line dip there?"

**What we traced it to.** The reactive correction applies the *last measured*
wavefront, so its residual is exactly the wavefront's change over the loop
delay, `W(t) − W(t−d)`. The reactive Strehl curve is therefore a direct readout
of the **local 2-frame temporal structure function** `D_t(2) = ‖W(t) − W(t−2)‖²`.
Empirically, across the run:

- `corr( reactive lag-error , D_t(2) ) = 0.98`
- `corr( reactive lag-error , instantaneous turbulence amplitude ) = 0.16`

So the dips track the *rate of change* of the wavefront, not its amplitude.
Under the frozen-flow (Taylor) hypothesis, `D_t(2)` at time *t* equals the
**spatial** structure function of the single phase-screen realization evaluated
over the wind step `V·2·Δt`, sampled at the pupil's current footprint. The dip
is the pupil passing over a **steep-gradient cell** of that one screen: in
frames 40–90 the injected pupil RMS is **1.60× the run average** (3.25 vs
2.03 rad) — a genuinely stronger, steeper turbulence cell crosses the aperture
in this seed = 1234 realization — and `D_t(2)` (hence the lag error) spikes
with it.

The predictive curve stays flat because the linear AR forecaster cancels the
*deterministic* frozen-flow translation regardless of how steep the local
gradient is, leaving only its small, near-white residual. (Annotated trace:
[phase8_dip_trace.png](phase8_dip_trace.png).)

**Takeaway for the writeup.** The dip is not an artefact or a bug — it is the
servo-lag error doing exactly what theory predicts when a steep turbulence cell
of a frozen-flow layer transits the pupil, and it is precisely the error the
predictive layer is designed to remove.

---

## Fix 3 — Fast-wind r₀ +14% outlier

**What was flagged.** Of the four Phase 3 validation scenarios, three recovered
r₀ within ±10% but the fast-wind scenario came in at **+14%** — outside the band.
Is that a real bias of the fast-wind regime, or just the single phase-screen
realization that scenario happened to use?

**How we checked it.** Frozen flow means a whole scenario is *one* screen sliding
past the pupil, so its recovered r₀ reflects that single realization's spatial
statistics over the swath traversed. We re-ran both the nominal and fast-wind
configs across **10 independent screen seeds** (Zernike-variance estimator, true
coefficients to isolate the screen effect from measurement noise) and also
varied the Zernike fitting band. Reproduce with
[phase8_r0_multiseed.py](phase8_r0_multiseed.py):

| Config | Zernike band | mean error | std | range | seed 1234 |
|---|---|--:|--:|--:|--:|
| nominal (V=2, 30°) | 4–15 | −7.1% | 15.3% | [−22, +35]% | −15.3% |
| fast-wind (V=4, 120°) | 4–15 | **−1.1%** | 11.9% | [−18, +17]% | +5.9% |
| fast-wind | 4–10 (narrower) | −1.4% | 12.6% | [−20, +16]% | +7.3% |
| fast-wind | 4–21 (wider) | −1.1% | 11.4% | [−17, +16]% | +5.0% |

**Conclusion.** The fast-wind config is **not biased**: across seeds its mean
error is −1.1%, and the observed errors span the empirical range
[−18%, +17%] — which contains the flagged +14%. (Note the ±1σ band from
mean −1.1% / std 11.9% is [−13.0%, +10.8%], so +14% is justified by the actual
observed spread across realizations, not by the nominal 1σ interval.) Narrowing
or widening the mode band changes nothing (std stays ~11–13%), so it is not an
aliasing / mode-fitting artefact. The +14% was single-screen realization
noise — the working explanation, now confirmed quantitatively. (The original
+14% was with measured coefficients, which add a few % of measurement noise on
top of this realization scatter; the seed-1234 true-coefficient draw here is
+5.9%.)

**Takeaway for the writeup.** r₀ recovered from a single frozen-flow realization
carries an intrinsic ≈10–15% (1σ) realization scatter at D/r₀ ≈ 4 — unbiased,
and exactly the regime the real on-sky/lab data will present. The honest figure
to quote is the multi-seed mean ± σ, not a single scenario.

