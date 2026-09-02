#!/usr/bin/env bash
# Azure App Service (trakt-mi-api) startup command.
#
# Serves the FastAPI MI Agent API with gunicorn + uvicorn workers. Set this file
# as the App Service "Startup Command":  bash startup.sh
#
# Tunables (App Service app settings, all optional):
#   MI_API_WORKERS   gunicorn worker processes (default 2)
#   MI_API_TIMEOUT   worker timeout seconds     (default 120)
#   PORT             port to bind               (App Service sets this; default 8000)
set -euo pipefail

# --- Keep the platform's instrumentation agent off the import path ---
# App Service prepends its auto-instrumentation directory to PYTHONPATH, e.g.
#   Updated PYTHONPATH to '/agents/python:/opt/startup/app_logs:.../site-packages'
# That directory ships its own vendored copies of common libraries. Because it
# sits AHEAD of the venv, `/agents/python/common/typing_extensions.py` shadows
# the version the venv installed, and anyio (imported by fastapi.routing) dies
# on `from typing_extensions import sentinel` — HaltServer, worker failed to
# boot, and the site returns 5xx until a later boot happens to omit the entry.
# Observed twice on 2026-09-01 (23:28 and 23:32); the 23:36 boot came up only
# because the platform left the entry out that time.
#
# Dropping the agent's own entries restores the venv's copies. It cannot break
# an app that never imported from them, and the tracing agent itself is loaded
# by the platform, not by this path.
if [[ -n "${PYTHONPATH:-}" ]]; then
  _kept=""
  IFS=':' read -ra _entries <<< "${PYTHONPATH}"
  for _entry in "${_entries[@]}"; do
    case "${_entry}" in
      /agents/python|/agents/python/*) continue ;;
      "") continue ;;
    esac
    _kept="${_kept:+${_kept}:}${_entry}"
  done
  if [[ "${_kept}" != "${PYTHONPATH}" ]]; then
    echo "startup.sh: dropped platform agent path(s) from PYTHONPATH" >&2
  fi
  export PYTHONPATH="${_kept}"
fi

# Worker class: the standalone `uvicorn-worker` package (uvicorn_worker.UvicornWorker)
# is the modern replacement for the deprecated uvicorn.workers module; fall back to
# the classic path if the package isn't present.
if python -c "import uvicorn_worker" >/dev/null 2>&1; then
  WORKER_CLASS="uvicorn_worker.UvicornWorker"
else
  WORKER_CLASS="uvicorn.workers.UvicornWorker"
fi

exec gunicorn mi_agent_api.app:app \
  --worker-class "$WORKER_CLASS" \
  --workers "${MI_API_WORKERS:-2}" \
  --timeout "${MI_API_TIMEOUT:-120}" \
  --access-logfile - \
  --error-logfile - \
  --bind "0.0.0.0:${PORT:-8000}"
