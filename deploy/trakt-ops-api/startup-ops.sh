#!/usr/bin/env bash
# Azure App Service (trakt-ops-api) startup command.
#
# Serves the standalone Operations Control Centre API — operations_control.api.app:app
# — with gunicorn + uvicorn workers. Set as the App Service "Startup Command":
#
#     bash deploy/trakt-ops-api/startup-ops.sh
#
# This is a SEPARATE App Service from trakt-mi-api. The repo-root startup.sh
# (which launches mi_agent_api.app:app) is untouched and still serves the MI API;
# the OCC application is never mounted into it.
#
# Tunables (App Service app settings, all optional):
#   OPS_API_WORKERS   gunicorn worker processes (default 1 — see below)
#   OPS_API_TIMEOUT   worker timeout seconds     (default 300)
#   PORT              port to bind               (App Service sets this; default 8000)
#
# WHY ONE WORKER BY DEFAULT
# The OCC engine owns workflow execution: it runs each workflow on a background
# thread under a blob lease (operations_control/engine.py) and reconciles
# interrupted runs on startup via recover_on_startup() in the FastAPI lifespan.
# Every additional gunicorn worker is another independent engine competing for
# the same leases and running its own recovery pass. The lease design survives
# that, but it makes concurrency a function of the web tier's process count
# rather than a deliberate setting. Keep this at 1 unless you have explicitly
# reasoned about lease contention.
set -euo pipefail

# Serve from the repository root regardless of the invoking working directory.
# This matters: parts of the runtime resolve governed configuration by RELATIVE
# path (e.g. operations_control/adapters.py reads
# "config/system/fields_registry.yaml"), and the Annex 2 delivery stages shell
# out to engine/ scripts with cwd set to the repo root. Anchoring here means the
# service behaves the same under App Service, a container, or a local run.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Worker class: the standalone `uvicorn-worker` package (uvicorn_worker.UvicornWorker)
# is the modern replacement for the deprecated uvicorn.workers module; fall back to
# the classic path if the package isn't present. Same pattern as the repo-root
# startup.sh so both services age the same way.
if python -c "import uvicorn_worker" >/dev/null 2>&1; then
  WORKER_CLASS="uvicorn_worker.UvicornWorker"
else
  WORKER_CLASS="uvicorn.workers.UvicornWorker"
fi

WORKERS="${OPS_API_WORKERS:-1}"
if [ "$WORKERS" -gt 1 ] 2>/dev/null; then
  echo "WARNING: OPS_API_WORKERS=$WORKERS. The OCC engine runs workflows on" >&2
  echo "         background threads and performs startup recovery per process;" >&2
  echo "         every worker is a separate engine competing for the same" >&2
  echo "         leases. 1 is the supported default." >&2
fi

exec gunicorn operations_control.api.app:app \
  --worker-class "$WORKER_CLASS" \
  --workers "$WORKERS" \
  --timeout "${OPS_API_TIMEOUT:-300}" \
  --access-logfile - \
  --error-logfile - \
  --bind "0.0.0.0:${PORT:-8000}"
