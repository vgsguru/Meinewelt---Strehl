"""Streamlit demo dashboard: AO correction with vs. without prediction.

Run with:  streamlit run aowfs/viz/app.py

Replays the residual wavefront side by side -- naive (reactive) vs predictive --
across a configurable loop delay, with the turbulence characterization and
Strehl metrics. Designed to be demoable in under two minutes.
"""

from __future__ import annotations

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from aowfs import SimConfig
from aowfs.viz.replay import compute_replay

st.set_page_config(page_title="Predictive AO WFS", layout="wide")

# r0 is conventionally quoted at a 500 nm reference; the sensing band is near-IR,
# and r0 scales as lambda^(6/5). Surface this factor so the slider (500 nm) and
# the recovered/injected metrics (sensing wavelength) are never confused.
_CFG = SimConfig()
_SENSE_NM = _CFG.pupil.wavelength_m * 1e9
_REF_NM = _CFG.turbulence.r0_ref_wavelength_m * 1e9
_R0_FACTOR = (_CFG.pupil.wavelength_m / _CFG.turbulence.r0_ref_wavelength_m) ** (6.0 / 5.0)


# max_entries caps how many cached runs are held resident — keeps the free-tier
# (~1 GB) memory bounded as the user sweeps the sliders.
@st.cache_data(show_spinner=True, max_entries=3)
def _replay(r0_ref_mm: float, wind: float, delay: int, n_frames: int, n_modes: int):
    base = SimConfig()
    cfg = replace(
        base,
        n_frames=n_frames,
        turbulence=replace(base.turbulence, r0_ref_m=r0_ref_mm * 1e-3, wind_speed_ms=wind),
    )
    return compute_replay(cfg, delay=delay, n_modes=n_modes)


def _wf_image(ax, data, mask, vmax, title):
    img = np.where(mask, data, np.nan)
    h = ax.imshow(img, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    return h


st.title("Predictive Adaptive-Optics Wavefront Reconstruction — PS9")
st.caption(
    "Synthetic SH-WFS → centroiding → modal/zonal reconstruction → r₀/τ₀ → "
    "**predictive control** → DM actuator map. Reactive vs predictive correction, side by side."
)

with st.sidebar:
    st.header("Turbulence / loop")
    r0_ref_mm = st.slider(f"r₀ at {_REF_NM:.0f} nm reference [mm]", 5.0, 20.0, 9.0, 0.5)
    st.caption(
        f"≈ **{r0_ref_mm * _R0_FACTOR:.0f} mm** ({r0_ref_mm * _R0_FACTOR / 10:.2f} cm) "
        f"at the {_SENSE_NM:.0f} nm sensing band — r₀ ∝ λ^1.2 (×{_R0_FACTOR:.2f}). "
        f"All r₀/τ₀ readouts below are at the sensing wavelength."
    )
    wind = st.slider("wind speed [m/s]", 0.5, 6.0, 2.0, 0.5)
    delay = st.slider("loop delay [frames]", 1, 4, 2, 1)
    n_frames = st.select_slider("frames", options=[150, 250, 400], value=150)
    n_modes = st.select_slider("Zernike modes", options=[20, 45, 66], value=45)

data = _replay(r0_ref_mm, wind, delay, n_frames, n_modes)

n = data.t_index.size
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Strehl — reactive (mean, {n} frames)", f"{data.mean_strehl_naive:.3f}")
c2.metric(f"Strehl — predictive (mean, {n} frames)", f"{data.mean_strehl_pred:.3f}",
          delta=f"+{data.mean_strehl_pred - data.mean_strehl_naive:.3f}")
c3.metric(f"r₀ recovered @{_SENSE_NM:.0f} nm", f"{data.r0_recovered_cm:.2f} cm",
          delta=f"{data.r0_recovered_cm - data.r0_injected_cm:+.2f} cm vs injected "
                f"({data.r0_injected_cm:.2f})")
c4.metric("τ₀ recovered", f"{data.tau0_recovered_ms:.2f} ms",
          delta=f"{data.tau0_recovered_ms - data.tau0_injected_ms:+.2f} ms vs injected "
                f"({data.tau0_injected_ms:.2f})")

i = st.slider("frame (selects the single frame shown in the maps below)", 0, n - 1,
              min(10, n - 1))

vmax = float(np.nanmax(np.abs(data.true_wf)))
fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
_wf_image(ax[0], data.true_wf[i], data.pupil_mask, vmax, "incoming wavefront [rad] (this frame)")
_wf_image(ax[1], data.resid_naive[i], data.pupil_mask, vmax,
          f"residual — reactive (Strehl {data.strehl_naive[i]:.2f}, this frame)")
h = _wf_image(ax[2], data.resid_pred[i], data.pupil_mask, vmax,
              f"residual — predictive (Strehl {data.strehl_pred[i]:.2f}, this frame)")
fig.colorbar(h, ax=ax, fraction=0.025)
st.pyplot(fig)

st.subheader("Strehl ratio over time (Maréchal)")
fig2, ax2 = plt.subplots(figsize=(13, 2.8))
ax2.plot(data.strehl_naive, color="crimson", label=f"reactive (mean {data.mean_strehl_naive:.2f})")
ax2.plot(data.strehl_pred, color="seagreen", label=f"predictive (mean {data.mean_strehl_pred:.2f})")
ax2.axvline(i, color="gray", ls=":", label="selected frame (maps above)")
ax2.set_xlabel("test frame"); ax2.set_ylabel("Strehl"); ax2.set_ylim(0, 1)
ax2.legend(loc="lower right")
st.pyplot(fig2)

st.caption(
    f"Loop delay {data.delay} frames · wind {data.wind_speed_ms:.2f} m/s · "
    f"predictive layer raises mean Strehl "
    f"{data.mean_strehl_naive:.2f} → {data.mean_strehl_pred:.2f}."
)
