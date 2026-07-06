"""YiAgents web UI (FastAPI + vanilla HTML/JS).

Read-only browsing of past analyses under ``~/.yiagents/logs`` plus a form that
submits a new analysis by spawning ``scripts/run_robust.py`` as a subprocess.
Importing this package has no side effects; the FastAPI app lives in
:mod:`web.app`.
"""
