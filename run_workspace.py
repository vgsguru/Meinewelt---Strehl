"""Launch the project: animated landing page + the Streamlit workspace.

Starts two local servers and opens the landing page in the browser:

  * landing page  ->  http://localhost:8000   (static, this file's sibling ``landing/``)
  * workspace     ->  http://localhost:8501   (Streamlit dashboard, aowfs/viz/app.py)

The landing page's "Enter Workspace" button points at :8501, so clicking it opens
the live software. This launcher only orchestrates servers -- it does not modify
any pipeline code.

    python run_workspace.py
"""

from __future__ import annotations

import contextlib
import functools
import http.server
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LANDING_DIR = ROOT / "landing"
LANDING_PORT = 8000
WORKSPACE_PORT = 8501


def _free(port: int) -> bool:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def serve_landing() -> threading.Thread:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(LANDING_DIR))
    httpd = http.server.ThreadingHTTPServer(("0.0.0.0", LANDING_PORT), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return t


def start_workspace() -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(ROOT / "aowfs" / "viz" / "app.py"),
        "--server.port", str(WORKSPACE_PORT),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]
    return subprocess.Popen(cmd, cwd=str(ROOT))


def main() -> int:
    if not (LANDING_DIR / "index.html").exists():
        print("landing/index.html not found", file=sys.stderr)
        return 1

    print("Starting workspace (Streamlit) on :%d ..." % WORKSPACE_PORT)
    proc = start_workspace() if _free(WORKSPACE_PORT) else None
    if proc is None:
        print("  (port %d already in use -- reusing existing workspace)" % WORKSPACE_PORT)

    if _free(LANDING_PORT):
        serve_landing()
    print("Serving landing page on :%d" % LANDING_PORT)

    url = f"http://localhost:{LANDING_PORT}"
    print(f"\n  Landing : {url}\n  Workspace: http://localhost:{WORKSPACE_PORT}\n")
    time.sleep(2.0)
    with contextlib.suppress(Exception):
        webbrowser.open(url)

    print("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down.")
        if proc:
            proc.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
