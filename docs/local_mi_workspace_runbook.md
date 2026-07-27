# Local MI Agent workspace — launch runbook

How to bring up the governed MI workspace locally (API on `:8000`, React on
`:5173`) with **both** the funded platform canonical and the pipeline dataset
loaded. Written for Codespaces, where the browser is not on the API host — the
detail that causes most of the failures below.

---

## 1. One-time

```bash
pip install -r requirements.txt -r mi_agent_api/requirements.txt
cd frontend/mi-agent-ui && npm install && cd -
```

## 2. Start the API (port 8000)

The API needs three things: which **tenant** it serves, the **funded** platform
canonical, and the **pipeline** root. The tenant is deployment configuration —
it is never read from a row in the tape, which is what keeps `direct_001` out of
the Client selector.

```bash
export MI_AGENT_CLIENT_ID=ERE                      # the TENANT (Client = ERE)
export MI_AGENT_PLATFORM_CANONICAL=platform/ERE/latest/platform_canonical_typed.csv
export MI_AGENT_PIPELINE_ROOT=pipeline             # root that CONTAINS the dated cuts

uvicorn mi_agent_api.app:app --reload --port 8000
```

Blob-backed deployments use the URI forms instead — same variables, same
meaning:

```bash
export MI_AGENT_PLATFORM_URI=blob://platform/ERE/latest
export MI_AGENT_PIPELINE_URI=blob://pipeline/ERE/latest
```

Local development only, and only when you are not testing auth:

```bash
export MI_AGENT_AUTH_ENABLED=false      # NEVER set this in a deployed environment
export TRAKT_RUNTIME_MODE=test          # permits fixture sources; refused inside Azure
```

Both default to the safe value. Production stays fail-closed: auth is on unless
explicitly disabled, and `trakt_core.runtime.validate_runtime_mode` refuses a
non-production mode when the Azure markers are present.

### Pipeline layout

Pipeline discovery walks the root for dated weekly cuts and infers the client
from the path, so the root must **contain** the client and date folders:

```
pipeline/                          ← MI_AGENT_PIPELINE_ROOT
└── ERE/
    └── pipeline/
        ├── 2025-11-03/M2L_KFI_and_Pipeline_2025_11_03.csv
        └── 2025-11-10/M2L_KFI_and_Pipeline_2025_11_10.csv
```

Point `MI_AGENT_PIPELINE_ROOT` at `pipeline/`, not at a single dated folder —
evolution and the funnel need every cut, not just the latest.

Confirm both datasets registered:

```bash
curl -s localhost:8000/health | jq '{source: .dataSource, tenant: .governance.tenantId}'
curl -s "localhost:8000/mi/pipeline/snapshot?portfolioId=ERE/latest" | jq '{ok, pipelineRowCount}'
```

## 3. Start the UI (port 5173)

```bash
cd frontend/mi-agent-ui
npm run dev
```

That is the whole command. `.env.development` already sets
`VITE_AGENT_API_URL=/`, so the browser calls the dev server same-origin and Vite
forwards `/mi`, `/me`, `/health` and `/v1` to the API.

If the API is not on `:8000`:

```bash
VITE_PROXY_TARGET=http://127.0.0.1:9000 npm run dev
```

### Do not set an absolute API URL

```bash
# WRONG in Codespaces / behind any forwarded port
echo "VITE_AGENT_API_URL=http://localhost:8000" > .env.local
```

`localhost` resolves in the **browser**, which is not on the API host, so the
request never reaches uvicorn — the port-forwarding proxy answers instead and
the client surfaces `MI Agent API returned 404`. `.env.local` also *overrides*
`.env.development`, so creating that file silently disables the working proxy.
Delete it if it exists:

```bash
rm -f frontend/mi-agent-ui/.env.local
```

The client's 404 message now names this cause directly, so the symptom points at
its own fix.

## 4. Validate

```bash
# through the API
python scripts/validate_local_mi_workspace.py

# through the Vite proxy — exactly the path the browser takes
python scripts/validate_local_mi_workspace.py --base http://127.0.0.1:5173
```

Expected for the current ERE platform:

| Check | Expected |
|---|---|
| Client | `ERE` only — no `direct_001` / `acquired_001` |
| Portfolio | Total · Direct · Acquired · direct_001 · acquired_001 |
| Total | 958 loans |
| Direct / direct_001 | 73 |
| Acquired / acquired_001 | 885 |
| Borrower age | available for every portfolio, including Acquired |
| Region | available for every portfolio, one harmonised category set |
| Pipeline | enabled for Total / Direct, `NON_ORIGINATING` for Acquired |
| Chat | HTTP 200 for every scope, with scope + field coverage |

## 5. After a Codespaces timeout

The container keeps the repo but not the processes or the exported
environment. Re-export and restart both:

```bash
cd /workspaces/trakt

export MI_AGENT_CLIENT_ID=ERE
export MI_AGENT_PLATFORM_CANONICAL=platform/ERE/latest/platform_canonical_typed.csv
export MI_AGENT_PIPELINE_ROOT=pipeline
export MI_AGENT_AUTH_ENABLED=false          # local only
export TRAKT_RUNTIME_MODE=test              # local only

uvicorn mi_agent_api.app:app --reload --port 8000 &
(cd frontend/mi-agent-ui && npm run dev &)

python scripts/validate_local_mi_workspace.py --base http://127.0.0.1:5173
```

Forward both ports (5173 and 8000) in the Codespaces **Ports** panel. Only 5173
needs to be opened in the browser; 8000 is reached through the dev proxy.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `MI Agent API returned 404` | absolute `VITE_AGENT_API_URL` in a forwarded-port environment | `rm frontend/mi-agent-ui/.env.local`, restart `npm run dev` |
| Chat answers canned demo figures | `VITE_AGENT_MODE=mock`, or no API URL in a production build | unset it; the app shows a red misconfiguration banner in prod builds |
| Client selector shows `direct_001` | an API predating the client/portfolio split | pull latest; `/mi/snapshots` is the tenant axis only |
| Pipeline tab disabled for Total | no pipeline root, or no cuts under it | check `MI_AGENT_PIPELINE_ROOT` layout above; the capability `detail` states which |
| Acquired shows no borrower age | the tape carries no borrower DOB **and** no explicit age | the chart says `not_supplied`; check the source mapping for `Borrower 1/2 DOB` |
| A region renders as its raw source value | that value has no governed mapping yet | it is reported in the preparation report's `unresolved_values`; add an approved synonym to `config/mi/region_taxonomy.yaml` |
