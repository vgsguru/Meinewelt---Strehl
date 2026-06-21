"""Streamlit Community Cloud entry point.

Streamlit runs this file from the repository root. We make the repo root
importable (so `import aowfs` resolves) and then load the dashboard module,
whose top-level code renders the app.

On Streamlit Cloud: set "Main file path" to  streamlit_app.py

If anything fails during import or first render (a missing dependency, an
out-of-memory kill, an environment-specific error), we surface the traceback
in the app instead of letting Streamlit show a blank/black page.
"""

from __future__ import annotations

import os
import sys
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    # Importing the module executes the Streamlit dashboard (st.* calls at module top level).
    import aowfs.viz.app  # noqa: E402,F401
except Exception:  # pragma: no cover - deployment safety net
    import streamlit as st

    try:
        st.set_page_config(page_title="Predictive AO WFS — startup error", layout="wide")
    except Exception:
        # set_page_config may already have run inside the dashboard before it failed.
        pass

    st.error(
        "The dashboard failed to start. The full traceback is below — this is "
        "shown instead of a blank page so the cause is visible."
    )
    st.code(traceback.format_exc(), language="text")
