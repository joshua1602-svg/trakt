# OCC frontend — Azure Static Web App

Deploys the existing Operations Control Centre React app,
`frontend/operations-control-ui`, to its **own** Azure Static Web App. No
frontend was created or replaced; this is a deployment of what is already there.

```
Browser ──▶ OCC Static Web App                 (static assets only)
             │  bundle carries VITE_OPS_API_URL
             ▼
           trakt-ops-api  (App Service)        GET /health, /ops/*, /ops/admin/config/*
             │  answers only origins listed in TRAKT_OPS_CORS_ORIGINS
```

The OCC calls the API **cross-origin with an absolute base URL**, so two values
must agree or the dashboard loads and every request fails:

| Where | Setting | Value |
|---|---|---|
| Frontend **build** (this workflow) | `VITE_OPS_API_URL` | `https://trakt-ops-api-bveyecbeh6ebarfa.westeurope-01.azurewebsites.net` |
| API **App Service** | `TRAKT_OPS_CORS_ORIGINS` | the deployed Static Web App origin |

`VITE_OPS_API_URL` belongs to the Vite build and is **not** an App Service
setting — the API never reads it. Vite inlines it into the bundle at build time,
so changing it requires a rebuild and redeploy, not a restart.

### Optional: the OCC Agent tab

The OCC Agent (synthetic onboarding) tab is **off** in this deployment and is
deliberately not set by the workflow — it is a pre-scale capability, not part of
Client 1 delivery. Turning it on takes two independent settings, one per side:

| Where | Setting | Value |
|---|---|---|
| Frontend **build** | `VITE_OCC_AGENT_SYNTHETIC_ENABLED` | `true` — shows the tab |
| API **App Service** | `OCC_AGENT_SYNTHETIC_ENABLED` | `true` — mounts the routes |

Both fail closed, and neither depends on the other: without the API setting the
routes do not exist, whatever the bundle was built with. See
`operations_control/occ_agent/README.md`.

## Why a separate Static Web App

`azure-static-web-apps-nice-smoke-067ac7603.yml` already exists but deploys
`frontend/mi-agent-ui` with `VITE_AGENT_API_URL` pointing at the MI API. It does
not build or deploy the OCC. The two apps have different API base URLs and
different authentication models, so they get separate Static Web Apps and
separate workflows. That existing workflow is untouched.

## Files

| File | Purpose |
|---|---|
| `frontend/operations-control-ui/public/staticwebapp.config.json` | SPA fallback + security headers |
| `.github/workflows/deploy-occ-frontend.yml` | build, verify, deploy, smoke test |
| `deploy/trakt-occ-frontend/smoke_test.sh` | post-deployment verification |
| `deploy/trakt-occ-frontend/tests/` | tests for the smoke test, with a fixture frontend + API |

### What gets uploaded, and why `app_location` points at `dist`

The workflow builds locally and deploys with `skip_app_build: true`, so the
artefact that ships is the one whose wiring was just verified. With that flag the
deploy action treats **`app_location` as the already-built content root and
ignores `output_location`** — so `app_location` must be
`frontend/operations-control-ui/dist`, not the project directory.

Pointing it at the project directory uploads the source tree, and the served
`index.html` is then the unbuilt one that references `/src/main.tsx`. That page
still contains `<div id="root">`, so it looks healthy to a probe that only checks
for the root element, but no `/assets/*` exists and the app never boots. The
build step now fails if `dist/index.html` references `/src/main.tsx` or no hashed
bundle, and the smoke test names this case explicitly.

Because the uploaded root is `dist/`, `staticwebapp.config.json` lives in
`public/` — Vite copies `publicDir` contents to the output root, so it lands at
`dist/staticwebapp.config.json`. At the project root it would be left behind and
SPA deep links would 404. The build step asserts it is present in `dist/`.

### Why `staticwebapp.config.json` was needed

The app uses `react-router` with client-side routes including
`/admin/config/system`. Without `navigationFallback`, a refresh or a shared deep
link returns **404** from the static host — the router never gets to run.
`frontend/mi-agent-ui` already had one; the OCC did not. `/health` is
deliberately **excluded** from the fallback so a probe against the frontend host
404s honestly instead of returning `index.html` with a misleading `200`.

## Operator authentication flow (verified)

| Step | Where |
|---|---|
| Token entered | `src/components/SignIn.tsx` — `<input type="password" autoComplete="off">`, shown by `AuthGate` in `src/App.tsx` until a token exists |
| Token stored | `src/lib/token.ts` — `localStorage["trakt_ops_token"]` |
| Header sent | `src/api/HttpOpsClient.ts` — `X-Operator-Token: <token>` on **every** request. The UI never sends `Authorization`, though the API accepts it |
| Rejection handled | A `401` clears the token, fires `UNAUTHORIZED_EVENT`, and the sign-in card reappears |
| Role gating | `GET /ops/me` populates `SessionProvider`; the admin nav entry and `/admin/config/*` routes render only for `is_admin`. The backend re-authorises every request — the UI check only decides what to offer |

**Tokens are never in source control or in the bundle.** They are typed in at
runtime by the operator. Verified against a real build: the bundle contains the
string `trakt_ops_token` (the storage **key** name) and `X-Operator-Token` (the
**header** name), and no token value. The workflow fails the build if the
smoke-test token ever appears in `dist/`, and the smoke test repeats that check
against the deployed assets.

Note that `localStorage` is readable by any script on the origin, so the token is
as strong as the origin's XSS posture. The config sets `X-Frame-Options: DENY`,
`X-Content-Type-Options: nosniff` and `Referrer-Policy: no-referrer`. Moving to
Entra / Easy Auth later changes only `operations_control/api/auth.py` and this
sign-in component.

## Required secrets

| Secret | Purpose |
|---|---|
| `AZURE_STATIC_WEB_APPS_API_TOKEN_OCC` | deployment token of the **new** OCC Static Web App (Static Web Apps' own mechanism; the ops-api OIDC identity is untouched) |
| `OPS_SMOKE_OPERATOR_TOKEN` | a token from the API's `TRAKT_OPS_OPERATORS` map, used only to prove authenticated `/ops/me` and `/ops/dashboard` work |

Without `OPS_SMOKE_OPERATOR_TOKEN` the smoke test **fails** rather than skipping:
"authenticated /ops/me succeeds" cannot be reported as verified when it was not
checked.

## The deployed hostname is discovered, not assumed

The workflow reads `steps.builddeploy.outputs.static_web_app_url` from the deploy
action, so it reports the real hostname — including a PR preview environment,
which gets its own origin. It prints the exact `TRAKT_OPS_CORS_ORIGINS` value to
the job summary.

**PR previews:** each preview origin differs from production, so a preview build
will fail the CORS check unless that origin is also listed on the API. That is
expected — the check is telling the truth about what the browser would do.

## First deployment

```bash
# 1. Create the Static Web App (once). Standard tier if you want preview envs.
az staticwebapp create -g trakt-rg -n trakt-occ-frontend -l westeurope --sku Free

# 2. Take its deployment token and store it as AZURE_STATIC_WEB_APPS_API_TOKEN_OCC
az staticwebapp secrets list -g trakt-rg -n trakt-occ-frontend \
  --query "properties.apiKey" -o tsv

# 3. Store a token from TRAKT_OPS_OPERATORS as OPS_SMOKE_OPERATOR_TOKEN

# 4. Run the workflow. It prints the deployed origin.

# 5. Allow that origin on the API (this is the step that is easy to forget):
az webapp config appsettings set -g trakt-rg -n trakt-ops-api --settings \
  TRAKT_OPS_CORS_ORIGINS="https://<the-origin-the-workflow-printed>"
```

Step 5 changes only `TRAKT_OPS_CORS_ORIGINS`. No route, no startup command, no
authentication model.

## Verification checklist

The smoke test performs all of these; run it by hand with:

```bash
OPS_SMOKE_OPERATOR_TOKEN=<token> \
  bash deploy/trakt-occ-frontend/smoke_test.sh \
    https://<frontend-origin> \
    https://trakt-ops-api-bveyecbeh6ebarfa.westeurope-01.azurewebsites.net
```

| # | Check | Expect |
|---|---|---|
| 1 | frontend `GET /` | 200, and the page contains the app root element |
| 2 | every `/assets/*.js|css` referenced by `index.html` | 200 |
| 3 | the bundle contains the API base URL | present (else `VITE_OPS_API_URL` was unset at build) |
| 4 | the bundle contains no operator token | absent |
| 5 | API `GET /health` | 200, `service: operations-control`, `auth_configured: true` |
| 6 | API `GET /ops/me` **without** a token | 401 |
| 7 | API `GET /ops/me` **with** a token | 200, prints the role |
| 8 | API `GET /ops/dashboard` **with** a token | 200 |
| 9 | CORS preflight from the frontend origin | `access-control-allow-origin` echoes that exact origin, never `*` |
| 10 | CORS allow-headers | includes `X-Operator-Token` |

Then, in a browser: sign in with an **admin** token and confirm `Platform
configuration` appears in the navigation and `/admin/config` renders; sign in
with an **operator** token and confirm it does not, and that visiting
`/admin/config` directly shows the access-denied state.

## Not changed by this deployment

`operations_control` API routes, `startup-ops.sh`, `trakt-mi-api`, the blob
trigger, the ops-api OIDC deployment identity, and the existing MI Static Web App
workflow.
