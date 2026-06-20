# Deploying the dashboard on Streamlit Community Cloud

The live workspace (`aowfs/viz/app.py`) deploys as a Streamlit app.

## Steps

1. Go to **https://share.streamlit.io** and sign in with GitHub.
2. **New app** → **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `vgsguru/Meinewelt---Strehl`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
4. (Optional) **Advanced settings → Python version:** `3.12`.
5. **Deploy.** First build installs `requirements.txt` (numpy, scipy, aotools,
   numba, matplotlib, streamlit, …) and may take a few minutes.

The app generates synthetic Shack-Hartmann data and replays the predictive vs.
reactive correction live. The first run computes the dataset (cached afterwards),
so give it ~30–60 s on the free tier.

## Notes

- `streamlit_app.py` is a thin entry point that puts the repo root on
  `sys.path` and loads `aowfs/viz/app.py` — that's why the **Main file path**
  is `streamlit_app.py`, not the app module directly.
- The **landing page** (`landing/index.html`) is a separate static site and is
  *not* part of the Streamlit deployment. Run it locally with
  `python run_workspace.py` (serves the landing page on :8000 and the dashboard
  on :8501).
- The 1.2 GB real-telemetry datasets (`data/`) and the heavy landing background
  videos are git-ignored — they aren't needed to run the dashboard. Phase 10's
  real-data adapter still works locally once those files are present.
