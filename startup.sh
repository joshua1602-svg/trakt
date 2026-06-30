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
