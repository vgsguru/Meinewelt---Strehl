"""Streamlit Community Cloud entry point.

Streamlit runs this file from the repository root. We make the repo root
importable (so `import aowfs` resolves) and then load the dashboard module,
whose top-level code renders the app.

On Streamlit Cloud: set "Main file path" to  streamlit_app.py
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Importing the module executes the Streamlit dashboard (st.* calls at module top level).
import aowfs.viz.app  # noqa: E402,F401
