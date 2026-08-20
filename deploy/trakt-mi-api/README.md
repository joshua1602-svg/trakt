# trakt-mi-api — Azure App Service (FastAPI MI Agent API)

Deploys the FastAPI MI API (`mi_agent_api.app:app`) as its **own** Azure App
Service named **`trakt-mi-api`** (Linux, Python 3.11), serving with
**gunicorn + uvicorn workers**. It reads the persisted **central platform
canonical** straight from Blob Storage, so the React dashboard's Total / Direct /
Acquired / cohort lenses load from the durable managed-service output — not
`/tmp`.

```
processed-v2/platform/ERE/latest/platform_canonical_typed.csv   (written by the blob trigger / Assembler)
        │   MI_AGENT_PLATFORM_URI = blob://…
        ▼
trakt-mi-api (App Service)  →  GET /health, GET /mi/* , POST /mi/query
        ▲
React dashboard
```

This is a **deployment** of the existing API — no blob-trigger / orchestrator /
onboarding / regime logic is changed.

## Files

| File | Purpose |
|---|---|
| `startup.sh` (repo root) | App Service **Startup Command**: `gunicorn mi_agent_api.app:app -k uvicorn_worker.UvicornWorker` |
| `deploy/trakt-mi-api/requirements.txt` | **the Oryx install set for this App Service.** Only what `mi_agent_api.app` imports |
| `deploy/trakt-mi-api/package_contents.txt` | what the deployment artefact contains — the paths the workflow stages |
| `.github/workflows/deploy-mi-api.yml` | stages that artefact and deploys it (`package:`), rather than the whole checkout |
| `requirements.txt` (repo root) | the **repository's** dependency set — Function App, Streamlit, ESMA delivery engine, mail. Not this App Service's contract |
| `mi_agent_api/requirements.txt` | the server deps on their own (local dev) |
| `deploy/trakt-mi-api/provision.sh` | one-shot `az` provision + deploy |
| `deploy/trakt-mi-api/app_settings.example.json` | the app settings |

> **Why a staged artefact.** Deploying the repo root made Oryx read the root
> `requirements.txt` as the runtime contract, so every deploy installed
> scikit-learn, scipy, streamlit, pyarrow and 24 other packages the API never
> imports — 88 resolved packages instead of 60. That is what left the build on
> `Running uv pip install...` for 20+ minutes, and an unfinished build leaves
> `antenv` without `gunicorn`, which is how `bash startup.sh` exits **127**.
> `tests/test_mi_api_appservice_packaging.py` recomputes the API's import closure
> and fails if either file drifts from what the app actually needs.

## App settings (production)

| Setting | Value |
|---|---|
| `MI_AGENT_PLATFORM_URI` | `blob://processed-v2/platform/ERE/latest/platform_canonical_typed.csv` |
| `TRAKT_BLOB_CONNECTION` | storage connection string (same account as `processed-v2`) |
| `TRAKT_STORAGE_BACKEND` | `blob` |
| `TRAKT_PROCESSED_CONTAINER` | `processed-v2` (only if not the default) |
| `SCM_DO_BUILD_DURING_DEPLOYMENT` | `true` (Oryx builds `requirements.txt`) |
| `WEBSITES_PORT` | `8000` |
| `MI_AGENT_CORS_ORIGINS` | your React host(s), comma-separated |
| `MI_AGENT_ALLOWED_ORIGIN` | the Static Web App origin, when the UI calls the API **cross-origin** (absolute `VITE_AGENT_API_URL`) rather than through the linked backend |
| `MI_AGENT_API_PREFIX` | gateway prefix to accept, default `/api`. Leave unset unless a gateway mounts the API somewhere other than `/api` |
| `MI_API_WORKERS` / `MI_API_TIMEOUT` | `2` / `120` (optional) |

## Which front door is this deployment behind?

The API serves its routes **bare** (`/mi/query`) *and* under the gateway prefix
(`/api/mi/query`), so both supported topologies work without a code change:

| Topology | `VITE_AGENT_API_URL` | Browser calls | App Service receives |
|---|---|---|---|
| SWA **linked backend** (recommended — Easy Auth injects the principal) | `/api` | `/api/mi/query` | `/api/mi/query` |
| Absolute base URL (cross-origin; set `MI_AGENT_ALLOWED_ORIGIN`) | `https://trakt-mi-api.azurewebsites.net` | `…/mi/query` | `/mi/query` |

This is why the chatbot previously returned **404** under the linked-backend
topology: SWA forwards the path intact, and the app served only the bare form.
`mi_agent_api/gateway.py` now normalises the prefix, and
`mi_agent_api/tests/test_mi_query_route_contract.py` asserts every path the React
client calls is servable in **both** forms, so the fault cannot recur silently.

Confirm which form a deployment answers on:

```bash
curl -s https://trakt-mi-api.azurewebsites.net/health | jq .routing
# {"apiPrefix": "/api", "queryPaths": ["/mi/query", "/api/mi/query"]}
```

`MI_AGENT_PLATFORM_URI` is resolved through the storage abstraction: in Azure
(`TRAKT_STORAGE_BACKEND=blob` + connection) it downloads the blob to a local
scratch dir (`MI_AGENT_SCRATCH`, default `/tmp/trakt/mi_platform`); the MI source
kind on `/health` then reads `platform_canonical`.

## Deploy (one command)

```bash
# from the REPO ROOT, az CLI logged in
export RESOURCE_GROUP=trakt-rg LOCATION=uksouth APP_NAME=trakt-mi-api
export TRAKT_BLOB_CONNECTION="<connection-string>"
bash deploy/trakt-mi-api/provision.sh
```

It creates the plan + web app, sets the startup command and app settings, and
zip-deploys the repo root (so `mi_agent_api`, `mi_agent`, `apps/blob_trigger_app`,
`engine`, `config` all ship).

## Deploy (manual, equivalent)

```bash
az group create -n trakt-rg -l uksouth
az appservice plan create -g trakt-rg -n trakt-mi-plan --is-linux --sku B1
az webapp create -g trakt-rg -p trakt-mi-plan -n trakt-mi-api --runtime "PYTHON:3.11"

az webapp config set -g trakt-rg -n trakt-mi-api --startup-file "bash startup.sh"
az webapp config appsettings set -g trakt-rg -n trakt-mi-api --settings \
  SCM_DO_BUILD_DURING_DEPLOYMENT=true WEBSITES_PORT=8000 \
  TRAKT_STORAGE_BACKEND=blob TRAKT_PROCESSED_CONTAINER=processed-v2 \
  TRAKT_BLOB_CONNECTION="<connection-string>" \
  MI_AGENT_PLATFORM_URI="blob://processed-v2/platform/ERE/latest/platform_canonical_typed.csv" \
  MI_AGENT_SCRATCH=/tmp/trakt/mi_platform MI_AGENT_CORS_ORIGINS="https://<react-host>"

# from the repo root:
az webapp up -g trakt-rg -n trakt-mi-api --plan trakt-mi-plan --runtime "PYTHON:3.11" --os-type Linux
```

## Verify

```bash
curl -s https://trakt-mi-api.azurewebsites.net/health | jq
# expect: {"ok": true, "dataSourceKind": "platform_canonical", "dataAvailable": true, ...}
```

`dataSourceKind: "platform_canonical"` confirms the API loaded the persisted
platform canonical from Blob. The `processed-v2/platform/{client}/latest/…` blob
must exist (run/approve a funded pack first).

## Notes / limitations

- The platform canonical is downloaded to scratch on resolution (no TTL cache);
  for very large tapes prefer mounting the `latest/` dir and using
  `MI_AGENT_PLATFORM_DIR` instead of `MI_AGENT_PLATFORM_URI`.
- Code deploys go through `.github/workflows/deploy-mi-api.yml`, which stages the
  artefact described above. `provision.sh` still `az webapp up`s the repo root —
  fine for first-time provisioning, but re-run the workflow afterwards so the App
  Service is left building the API's contract rather than the repository's.
- No auth/RBAC is added here (matches the existing API); put the App Service
  behind your gateway / Easy Auth as needed.
