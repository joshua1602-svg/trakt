# trakt-ops-api — Azure App Service (Operations Control Centre API)

Deploys the standalone Operations Control Centre API (`operations_control.api.app:app`)
as its **own** Azure App Service named **`trakt-ops-api`** (Linux, Python 3.11),
served with **gunicorn + uvicorn workers**.

```
React OCC dashboard  ──VITE_OPS_API_URL──▶  trakt-ops-api   GET /health, /ops/*, /ops/admin/config/*
React MI dashboard   ──VITE_AGENT_API_URL─▶ trakt-mi-api    GET /health, /mi/*, /v1/copilot/*
                                                 ▲
                                    unchanged: bash startup.sh
```

**Two services, deliberately.** `trakt-mi-api` is untouched — it still runs the
repo-root `startup.sh` (`gunicorn mi_agent_api.app:app`), and the OCC application
is **not** mounted into `mi_agent_api.app`. The two have incompatible
authentication models (MI enforces a global Easy Auth / Entra app-role guard; the
OCC uses `X-Operator-Token` against `TRAKT_OPS_OPERATORS`), both define `/health`
and `/`, and the OCC runs workflows on background threads with startup recovery.
Merging them would mean punching a hole in a fail-closed auth boundary and
coupling MI's worker count to the OCC's concurrency requirement.

## Files

| File | Purpose |
|---|---|
| `deploy/trakt-ops-api/startup-ops.sh` | App Service **Startup Command**: `gunicorn operations_control.api.app:app -k uvicorn_worker.UvicornWorker --workers 1 --timeout 300` |
| `deploy/trakt-ops-api/provision.sh` | one-shot `az` provision + deploy; refuses to target `trakt-mi-api` |
| `deploy/trakt-ops-api/app_settings.example.json` | the app settings, with no real secrets |
| `deploy/trakt-ops-api/verify_package.py` | proves an unpacked artefact can import the app and has the files it resolves at runtime |
| `deploy/trakt-ops-api/azure_hosts.sh` | discovers the SCM and public hostnames (never constructs them — secure unique default hostnames carry a hash) |
| `deploy/trakt-ops-api/kudu_status.sh` | the single copy of Kudu's `DeployStatus` enum: `Pending=0, Building=1, Deploying=2, Failed=3, Success=4` |
| `deploy/trakt-ops-api/preflight_checks.sh` | artefact listing plus live App Service configuration, before any upload |
| `deploy/trakt-ops-api/deploy_async.sh` | submits with `isAsync=true` and polls, so the ~230s SCM front-end timeout cannot mask a long build |
| `deploy/trakt-ops-api/oryx_manifest.py` | reads `oryx-manifest.toml` and reports the build mode and which artefact should exist |
| `deploy/trakt-ops-api/verify_oryx_output.sh` | post-deploy evidence: manifest present and its declared output on the site. **Not** the gate |
| `deploy/trakt-ops-api/probe_health.sh` | the gate's `/health` retry loop, sized for a cold start and safe under `set -e` |
| `deploy/trakt-ops-api/collect_diagnostics.sh` | runs on any failure: deployment record, Oryx build log, container logs, site config |
| `.github/workflows/deploy-ops-api.yml` | test + build + OIDC deploy, scoped to `trakt-ops-api` |
| `requirements.txt` (repo root) | Oryx install set — already carries `fastapi` / `uvicorn[standard]` / `gunicorn` / `uvicorn-worker` / `PyYAML` / `pandas` / `lxml` |

No `Dockerfile` is added. `trakt-mi-api` is deployed today through **App Service
source deployment with an Oryx build** (`--startup-file` + zip deploy from the
repo root); its Dockerfile is an alternative path, not the live one. This service
follows the live pattern so there is one build model to reason about.

## Startup command

```
bash deploy/trakt-ops-api/startup-ops.sh
```

Which execs:

```
gunicorn operations_control.api.app:app \
  --worker-class uvicorn_worker.UvicornWorker \   # falls back to uvicorn.workers.UvicornWorker
  --workers ${OPS_API_WORKERS:-1} \
  --timeout ${OPS_API_TIMEOUT:-300} \
  --bind 0.0.0.0:${PORT:-8000}
```

**One worker by default, on purpose.** The OCC engine executes each workflow on a
background thread under a blob lease and reconciles interrupted runs on startup
(`recover_on_startup()` in the FastAPI lifespan). Every extra gunicorn worker is
another independent engine performing its own recovery pass and competing for the
same leases. The script warns on stdout if `OPS_API_WORKERS > 1`.

The script `cd`s to the repository root first. That is load-bearing: parts of the
runtime resolve governed configuration by **relative** path (e.g.
`operations_control/adapters.py` reads `config/system/fields_registry.yaml`), and
the Annex 2 delivery stages shell out to `engine/` scripts with `cwd` set to the
repo root.

## Deployment package root

The package root is the **repository root**. Contents, established by import
trace rather than assumption:

| Path | Why it ships |
|---|---|
| `operations_control/` | the API, engine, stores, versioned configuration packages |
| `engine/` | `orchestrator_agent` (imported) **and** `gate_4b_delivery/annex2_delivery_normalizer.py` + `gate_5_delivery/xml_builder_annex2.py`, which are executed as **subprocesses** — an import check alone would not notice these missing |
| `apps/` | `blob_trigger_app` storage / layout / persistence / approvals |
| `trakt_core/` | governed context, errors, tenancy, policy (hard imports in `engine/annex_delivery_agent/`) |
| `mi_agent/` | reached only through **guarded** (`try/except`) imports in the onboarding path; shipped so risk-limit config and LLM cost estimation are not silently degraded |
| `config/` | every governed YAML the packages are seeded from, plus the Annex 2 XSD at `config/system/DRAFT1auth.099.001.04_1.3.0.xsd` |
| `DRAFT1auth.099.001.04_non-ABCP Underlying Exposure Report_Version_1.3.1.xlsx` | repo-root mapping workbook, passed to the XML builder as `--mapping-workbook` |
| `requirements.txt` | Oryx build input |
| `deploy/trakt-ops-api/startup-ops.sh` | the startup command |

**Excluded:** `frontend/` and `node_modules/` — nothing in the API's import graph
reaches them, and the React app is deployed separately. The workflow fails the
build if either appears in the artefact. Tests, `docs/` and `due_diligence/` are
excluded too. Artefact size: ~8.6 MB.

## Required Azure app settings

See `app_settings.example.json` for the annotated set. The ones that will bite:

| Setting | Value | Consequence if wrong |
|---|---|---|
| `TRAKT_OPS_OPERATORS` | JSON map of token → `{name, clients, role}` | **Fail-closed: every route returns 503 while unset.** `role: "admin"` grants `/ops/admin/config/*` |
| `TRAKT_OPS_CORS_ORIGINS` | exact deployed React origin(s), comma-separated | browser calls blocked; **never** `*` — this API exposes administrator configuration actions |
| `TRAKT_BLOB_CONNECTION` | storage connection string | no storage |
| `TRAKT_STORAGE_BACKEND` | `blob` | falls back to local files |
| `TRAKT_OPS_CONTAINER` | `operations-control` | governance data lands in the wrong container |
| `TRAKT_OPS_STAGING_ROOT` | `/tmp/trakt/ops_staging` | defaults to the **relative** `.ops_state/staging` inside the deployment root |
| `TRAKT_OPS_MEMORY_ROOT` | `/tmp/trakt/ops_memory` | defaults to the relative `.ops_state/client_memory` |
| `PORT` / `WEBSITES_PORT` | `8000` | App Service cannot reach the container |
| `OPS_API_WORKERS` / `OPS_API_TIMEOUT` | `1` / `300` | multiple competing engines / workers killed mid-stage |
| `TRAKT_SOURCE_REGISTRY_URI`, `TRAKT_STATE_CONTAINER`, `TRAKT_PROCESSED_CONTAINER`, `TRAKT_RAW_CONTAINER` | must match the blob trigger / MI API | client resolution and publication read the wrong places |

Store `TRAKT_OPS_OPERATORS` and `TRAKT_BLOB_CONNECTION` as **Key Vault
references**, not literals. Leave `TRAKT_RUNTIME_MODE` unset — `trakt_core`
defaults to `production`, which is what refuses fixture data.

## Deploy (one command)

```bash
# from the REPO ROOT, az CLI logged in
export RESOURCE_GROUP=trakt-rg LOCATION=uksouth
export TRAKT_BLOB_CONNECTION="<connection-string>"
export OPS_OPERATORS='{"<random-admin-token>":{"name":"Administrator","clients":["*"],"role":"admin"}}'
export OPS_CORS_ORIGINS="https://<deployed-react-occ-origin>"
bash deploy/trakt-ops-api/provision.sh
```

It creates its own plan (`trakt-ops-plan`) and web app, creates the
`operations-control` container, sets the startup command and app settings, and
zip-deploys the repo root. It **refuses to run** if `APP_NAME` is `trakt-mi-api`
or `APP_PLAN` is `trakt-mi-plan`, and refuses `OPS_CORS_ORIGINS="*"`.

## Frontend wiring

Traced in the code: `frontend/operations-control-ui/src/api/HttpOpsClient.ts:54`
reads the base URL from `import.meta.env.VITE_OPS_API_URL`, declared at
`src/vite-env.d.ts:4`, defaulting to `""` (same origin). Because the OCC UI calls
this service **directly** on a different host, the variable must be set at
**build** time (Vite inlines it):

```bash
VITE_OPS_API_URL=https://trakt-ops-api.azurewebsites.net
```

And the matching server-side value on `trakt-ops-api`:

```bash
TRAKT_OPS_CORS_ORIGINS=<deployed React origin>      # e.g. https://<swa-name>.azurestaticapps.net
```

These two must agree or the dashboard loads and every request fails CORS. Note
`VITE_OPS_MODE=mock` must **not** be set for a real deployment — it serves canned
data with no backend. `VITE_PROXY_TARGET` is dev-server only.

## Deployment verification checklist

Run after every deploy. Replace the tokens with real ones from
`TRAKT_OPS_OPERATORS`. Expected results below were confirmed against this exact
startup script and artefact.

```bash
OPS=https://trakt-ops-api.azurewebsites.net
MI=https://trakt-mi-api.azurewebsites.net
ADMIN=<admin-token>; OPERATOR=<operator-token>
```

| # | Check | Command | Expect |
|---|---|---|---|
| 1 | Service is up | `curl -s $OPS/health` | `200` · `{"ok":true,"service":"operations-control","auth_configured":true,"storage_ok":true}` |
| 2 | Auth is configured | the same response | `auth_configured: true` — if `false`, `TRAKT_OPS_OPERATORS` is unset and every other route answers **503** |
| 3 | Admin identity | `curl -s -H "X-Operator-Token: $ADMIN" $OPS/ops/me` | `{"ok":true,"principal":{...,"is_admin":true,...}}` |
| 4 | Admin config reads | `curl -s -o /dev/null -w '%{http_code}' -H "X-Operator-Token: $ADMIN" $OPS/ops/admin/config` | `200` |
| 5 | Operator is refused | `curl -s -H "X-Operator-Token: $OPERATOR" $OPS/ops/admin/config` | `403` · `{"detail":{"errorCode":"OPS_ADMIN_REQUIRED",...}}` |
| 6 | Operator cannot mutate | `curl -s -o /dev/null -w '%{http_code}' -X POST -H "X-Operator-Token: $OPERATOR" $OPS/ops/admin/config/system/1/activate` | `403` |
| 7 | No token is refused | `curl -s -o /dev/null -w '%{http_code}' $OPS/ops/admin/config` | `401` |
| 8 | CORS allows the React origin | `curl -s -D - -o /dev/null -X OPTIONS $OPS/ops/admin/config -H "Origin: <react-origin>" -H "Access-Control-Request-Method: GET"` | `access-control-allow-origin: <react-origin>` |
| 9 | CORS is not wildcard | same with `-H "Origin: https://evil.invalid"` | **no** `access-control-allow-origin` header |
| 10 | One worker | App Service log stream | exactly one `Booting worker with pid` per start, and `Using worker: uvicorn_worker.UvicornWorker` |
| 11 | React loads the config pages | open the OCC dashboard as an admin | `Platform configuration` appears in the navigation; Overview / System / Assets / Regimes / History all render |
| 12 | React hides admin from operators | open as an ordinary operator | no `Platform configuration` nav entry; visiting `/admin/config` directly shows the access-denied state |
| 13 | **MI API unchanged** | `curl -s $MI/health` | its own MI payload (`dataSourceKind`, `routing`), **not** `operations-control` |
| 14 | **MI routes unchanged** | `curl -s -o /dev/null -w '%{http_code}' $MI/mi/catalogue` (with MI auth) | same status as before this deploy |
| 15 | **MI startup unchanged** | `az webapp config show -g <rg> -n trakt-mi-api --query appCommandLine -o tsv` | `bash startup.sh` |

Checks 13–15 are the regression gate. If check 13 ever returns
`"service":"operations-control"`, the OCC start command has been applied to the
wrong App Service and the MI API is down.

## GitHub workflow

`.github/workflows/deploy-ops-api.yml` — triggers on `push` to `main` (paths
scoped to the OCC's runtime closure) and `workflow_dispatch`.

Identity: **OIDC federated credential**, no publish profile. `azure/login@v2`
with `permissions: { id-token: write, contents: read }`.

Repository secrets required:

| Secret | Value |
|---|---|
| `AZURE_OPS_API_CLIENT_ID` | app registration (client) ID of the deploying identity |
| `AZURE_OPS_API_TENANT_ID` | Entra directory (tenant) ID |
| `AZURE_OPS_API_SUBSCRIPTION_ID` | subscription containing `trakt-ops-api` |
| `AZURE_OPS_API_RESOURCE_GROUP` | resource group of `trakt-ops-api` |

Steps: install deps → run targeted OCC tests → run the deployment-script tests →
build the artefact → verify the artefact imports and is frontend-free → OIDC login
→ discover hostnames → pre-flight → settle → asynchronous deploy (polled) →
inspect the Oryx build output → **probe `/health`** → optional authenticated probe.
No other App Service is deployed, configured or restarted.

### Post-deployment verification: what is the gate

**`/health` returning `200` is the gate.** If the app serves, its dependencies
imported — whatever layout Oryx published. Everything on the filesystem is
evidence for attributing a failure, not a pass/fail condition.

`verify_oryx_output.sh` asserts only what the manifest itself declares:

1. `oryx-manifest.toml` must exist — the runtime needs it to locate the build
   output.
2. It is parsed (`oryx_manifest.py`) to determine the build mode, and then the
   artefact **that manifest declares** must exist.
3. `antenv/` and `__oryx_packages__/` are printed as observations and are **never
   required**.

Point 3 is a correction. The earlier version asserted:

```bash
if antenv != 200 && __oryx_packages__ != 200; then fail   # WRONG
```

which assumes the uncompressed layout. This service builds with
`--compress-destination-dir`:

```
Copied the compressed output to '/home/site/wwwroot'
Direct compression with zstd done
Manifest file created
Total execution done
```

In that mode `/home/site/wwwroot` holds a single compressed artefact
(`output.tar.zst`) plus `oryx-manifest.toml`, and the runtime extracts it at
startup. No expanded directory is published, so the assertion reported a build
that had installed 91 packages — including `pyyaml==6.0.3`, with the deployment
record showing `complete=true` and `status=4` (Success) — as a hard failure.

#### The probe must outlive a cold start, and must survive its own timeouts

`probe_health.sh` owns the retry loop. Two properties matter, and both were
learned from a failure:

* **A curl timeout is a result, not an abort.** The loop used to live inline as
  `CODE="$(curl … --max-time 20 …)"` under `set -euo pipefail`. A standalone
  assignment takes its command substitution's exit status, so the first
  `--max-time` expiry (curl exit `28`) ended the step on attempt one — defeating
  the eight retries written directly beneath it and skipping the container-log
  dump. The job reported a bare `exit code 28` on a deployment Kudu had recorded
  as `status=4` (Success). curl's failure is now absorbed and surfaced as
  HTTP `000`.
* **The budget is a cold-start budget.** This site runs `alwaysOn=false` with a
  `--compress-destination-dir` build, so the first request after a deploy waits
  for `output.tar.zst` to be extracted and then for `operations_control`,
  `mi_agent`, `analytics_lib` and pandas to import. Twenty seconds only ever
  passed when the container was already warm. The default is now 10 attempts
  × 60s with 20s gaps, returning immediately on the first `200`.

`tests/test_health_probe.sh` stubs curl and pins both, and fails the build if an
unguarded `VAR="$(curl …)"` reappears anywhere in the workflow. It does not scan
`deploy_async.sh` or `collect_diagnostics.sh`: those run under `set -uo pipefail`
with no `-e`, which is why their unguarded substitutions are correct — the poller
has to survive a transient failure and the diagnostics must never abort part-way.

Enabling **Always On** would remove the cold start from the gate's path
entirely. It is a site setting with a cost implication, so it is a decision, not
something this workflow changes.

A completed deployment (`complete=true`, `status=4`) has completed. A verification
failure after it is a fault in the verification until proven otherwise, and is not
evidence that the build was interrupted. The separate "settle" step addresses a
genuine historical incident in which Azure *did* stop a build part-way and said so
in the deployment record; that is a different condition, distinguishable because
the record reports `status=3`.

Optional authenticated probe: when the repository secret
`OPS_SMOKE_OPERATOR_TOKEN` is set, the workflow also calls `/ops/me` with it and
requires `200`. This is the one thing `/health` cannot tell you — `/health` reports
only whether `TRAKT_OPS_OPERATORS` is *set*, not whether a token in it is
accepted. Without the secret the step reports SKIPPED; the token is passed by
environment and never echoed.

If `/health` fails, the step dumps the container log stream (`/api/logs/docker`)
inline before exiting, so the import error is next to the failure rather than only
in the diagnostics artefact.

Two real-component Annex 2 golden tests are **deselected by name** in the test
step (`TestRealComponentsMiniGolden::test_normaliser_and_builder_pass_xsd` and
`::test_builder_interventions_are_captured`). They assert the XML builder's output
validates against the vendored draft ESMA XSD, and they are **already failing on
`main`** — verified against a clean tree. They are deselected so this deployment
gate reflects this service's health rather than a pre-existing product failure;
they are not skipped in the suite and still need fixing separately.

## Remaining manual Azure steps

The provision script does not — and should not — do these:

1. **Create the federated credential** for GitHub OIDC. On the app registration,
   add a credential with issuer `https://token.actions.githubusercontent.com`,
   subject `repo:joshua1602-svg/trakt:ref:refs/heads/main`, audience
   `api://AzureADTokenExchange`. Then grant it **Contributor** (or Website
   Contributor) **scoped to the `trakt-ops-api` resource only** — not the whole
   resource group, so CI cannot reach `trakt-mi-api`.
2. **Add the four repository secrets** listed above.
3. **Convert secrets to Key Vault references** for `TRAKT_OPS_OPERATORS` and
   `TRAKT_BLOB_CONNECTION`, and enable the App Service managed identity with
   `get` access on the vault.
4. **Mint the operator/admin tokens** (long random values) and record who holds
   which. Nothing in the repo generates these.
5. **Set `VITE_OPS_API_URL`** in the React OCC build pipeline, then redeploy the
   frontend so the value is inlined.
6. **Set `TRAKT_OPS_CORS_ORIGINS`** to the real React origin once it is known —
   the placeholder in the example settings is not a working value.
7. **Confirm the `operations-control` container exists** if you skipped
   `provision.sh` and configured the service by hand.
8. Optional: **Always On** (avoids cold starts dropping background workflow
   threads), **health check path** `/health`, and **diagnostic logs** to Log
   Analytics.

## Local run (same entry point)

```bash
export TRAKT_STORAGE_BACKEND=file TRAKT_LOCAL_BLOB_ROOT=.localblob
export TRAKT_OPS_OPERATORS='{"dev-admin":{"name":"Administrator","clients":["*"],"role":"admin"}}'
export TRAKT_OPS_CORS_ORIGINS=http://localhost:5173
export PORT=8100
bash deploy/trakt-ops-api/startup-ops.sh
```

Then `cd frontend/operations-control-ui && npm run dev` (the dev server proxies
`/ops` and `/health` to `VITE_PROXY_TARGET`, default `http://127.0.0.1:8100`).
