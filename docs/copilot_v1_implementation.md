# Trakt Microsoft 365 Copilot — v1 implementation

v1 supports **only** three capabilities, exposed as a Microsoft 365 declarative
agent backed by an API plugin:

1. **Ask a portfolio MI question** (`askTraktMi`) — answered by the existing MI
   Agent deterministic engine.
2. **Retrieve the latest generated investor deck** (`getLatestInvestorDeck`) —
   the current latest PPTX from the existing blob output convention.
3. **Retrieve the latest generated canonical loan tape** (`getLatestCanonicalTape`)
   — the current latest platform canonical CSV from the existing blob output
   convention.

Nothing else is in scope: no ESMA XML, no mapping/validation/exception reports,
no approvals, no SharePoint/Teams/Outlook actions, no Graph indexing, no
connectors, no MCP.

## What was added

| File | Purpose |
|---|---|
| `mi_agent_api/copilot_auth.py` | Entra ID **bearer-token validation** (JWKS signature, issuer, audience, expiry, optional scope/role) for the Copilot routes. Default mode `entra` **fails closed** (503) until configured; `disabled` mode is local-dev/tests only. |
| `mi_agent_api/copilot_actions.py` | The three actions as a FastAPI router (`/v1/copilot/*`) with explicit Pydantic request/response models, plus a short-lived HMAC-signed download endpoint. Contains **no business logic** — it delegates to existing components (see mapping below). |
| `deploy/copilot-agent/manifest.json` | Teams app manifest **template** (declarative-agent host package; tokens substituted per client). |
| `deploy/copilot-agent/declarativeAgent.json` | Declarative agent (schema **v1.7**): instructions (never invent figures; use the actions; state unavailability; include reporting dates; never expose paths/secrets) and the four suggested prompts. |
| `deploy/copilot-agent/ai-plugin.json` | API plugin manifest **template** (schema **v2.4**): exactly the three functions; Entra **SSO** runtime auth (`OAuthPluginVault` + the Trakt-created Enterprise-token-store auth-config id); bound to the OpenAPI spec. |
| `deploy/copilot-agent/trakt-copilot-openapi.yaml` | Hand-authored OpenAPI **3.0.3** contract **template** covering exactly the three actions (the FastAPI auto-spec is 3.1 and covers the whole internal API, so it is not used for the plugin). |
| `deploy/copilot-agent/package_agent.py` | Builds a complete **client-specific** `trakt-copilot-agent-<client>.zip` from a client configuration file; deterministic per-client app GUID (UUIDv5); **fails if any placeholder survives** substitution. |
| `deploy/copilot-agent/client-config.sample.json` | Sample per-client packaging configuration (real configs stay out of the repo). |
| `mi_agent_api/tests/test_copilot_actions.py` | Action tests (see Testing). |
| `mi_agent_api/tests/test_copilot_package.py` | Package/spec structural tests + spec↔routes lock-step. |

## What was modified (all minimal/additive)

| File | Change | Why |
|---|---|---|
| `mi_agent_api/app.py` | `include_router(copilot_actions.router)` at the end of the module (plus a 3-line comment). | Mounts the action layer on the existing API deployment. |
| `mi_agent_api/auth.py` | `COPILOT_PATH_PREFIX = "/v1/copilot"`; `auth_guard` returns early for that prefix. | The Easy-Auth header contract does not exist on Copilot's direct bearer-token calls; the Copilot routes enforce their own fail-closed guard instead. No other route's behaviour changes. |
| `requirements.txt`, `mi_agent_api/requirements.txt` | Added `PyJWT[crypto]>=2.8`. | RS256 token validation. The root file is the Oryx/App Service install set; the API file is the local-dev set. Only dependency added. |
| `deploy/trakt-mi-api/app_settings.example.json` | Added the `TRAKT_COPILOT_*` settings block. | Deployment documentation for the new routes. |

No dashboard, Streamlit, gate, pipeline, deck-generation, or storage-layout
code was touched. The blob folder structure is unchanged.

## How the three actions map to existing Trakt components

**askTraktMi → the existing MI Agent, unchanged.**
`POST /v1/copilot/mi/query` first checks the active data-source kind and
returns a structured **503** if it is `synthetic_demo` or `unavailable` — the
Copilot surface can never answer from demo data (the repo-wide data-source
resolution itself is unchanged, per scope). It then calls the **existing**
`/mi/query` handler function (`mi_agent_api.app.query`) directly — the same
deterministic-first parser (`chat_routing.try_route` → `llm_query_parser`
zero-cost-first), the same `MIQuerySpec` → `mi_query_executor` deterministic
execution, the same metric definitions, filters, and adapter envelope
(`adapters.adapt_workflow_result`). The envelope is reshaped (not recomputed)
into `CopilotMiAnswer`: answer, interpreted, reportingDate, datasetContext,
dataSourceKind/Label, warnings, sourceNotes, and `supportingValues` — a compact
extraction of the KPI/table/chart artifacts (rows capped at 50) so Copilot can
compose its narrative **only from values the deterministic executor produced**.
No narrative layer was rebuilt: Copilot itself verbalises the structured
result, under agent instructions that forbid inventing figures. Follow-ups are
rewritten into standalone questions by Copilot (the MI API is stateless, as it
is for the React client, which does its own follow-up rewriting client-side).

**getLatestInvestorDeck → the existing deck store, unchanged.**
Resolution is exactly the dashboard's: `mi_agent_api.decks.resolve_deck_local /
list_decks` — `MI_AGENT_DECK_ROOT` local mode, else the durable blob store at
`processed-v2/decks/{client}/latest/investor_pack.pptx` with the
`latest_investor_pack.json` pointer supplying `reportingPeriod`/`generatedAt`.
"Latest" is the current blob convention (the mutable `latest/` pointer), by
design for v1. No regeneration, no approval framework, no registry.

**getLatestCanonicalTape → the existing platform-canonical convention, unchanged.**
Resolution reuses `data_source._resolve_platform_canonical()` — i.e.
`MI_AGENT_PLATFORM_URI` → `processed-v2/platform/{client}/latest/platform_canonical_typed.csv`
(blob), or the explicit/local equivalents (`MI_AGENT_PLATFORM_CANONICAL`,
`MI_AGENT_PLATFORM_DIR`, conventional `out_platform/`). It deliberately does
**not** continue down the wider data-source chain: no central-tape substitute,
no demo fallback — absent tape is a structured **404**. The reporting period is
best-effort from the dated platform cuts alongside `latest/`. The file is
served as-is (CSV, the pipeline's existing format — no conversion).

**File delivery.** Both artifact actions return metadata plus a **short-lived
signed URL** (`/v1/copilot/artifacts/download?token=…`): an HMAC-SHA256 token
over `{artifact-kind, client, expiry}` (default TTL 300 s, clamp 60–3600). The
API redeems the token and streams the bytes itself — blob credentials and blob
paths never leave the server; no SAS tokens, no account keys in URLs.

## Authentication

- **Actions** (`askTraktMi`, both artifact lookups): validated **Entra ID
  bearer token** — signature via the tenant JWKS
  (`login.microsoftonline.com/{tenant}/discovery/v2.0/keys`), issuer (v2 or v1
  form), audience, expiry, and optionally a required scope/app-role
  (`TRAKT_COPILOT_REQUIRED_SCOPE`). The API **never trusts a client-supplied
  identity header** on these routes; the existing `X-MS-CLIENT-PRINCIPAL`
  Easy-Auth guard continues to protect `/mi/*` for the dashboard exactly as
  before.
- **Fail-closed defaults**: mode defaults to `entra`; with tenant/audience
  unset every Copilot route returns 503. The routes are not anonymously
  callable in production by omission.
- **Download URL**: authenticated by its signed expiring token (it is a link a
  human clicks from Copilot; a bearer header is not available there).
- **Secrets**: nothing committed; keys and the signing secret are App Service
  settings. Blob credentials stay server-side. Tokens are never logged.
- **Tenancy**: unchanged v1 model — one deployment per client
  (`MI_AGENT_CLIENT_ID` / `MI_AGENT_PLATFORM_URI` select the client); no new
  tenancy architecture was introduced, per scope.

## Required environment variables (Trakt MI API App Service)

| Variable | Value |
|---|---|
| `TRAKT_COPILOT_AUTH_MODE` | `entra` (default; `disabled` only for local dev) |
| `TRAKT_COPILOT_ENTRA_TENANT_ID` | Entra directory (tenant) GUID |
| `TRAKT_COPILOT_ENTRA_AUDIENCE` | `api://<app-id>` (comma-list allowed, e.g. also the bare app id) |
| `TRAKT_COPILOT_REQUIRED_SCOPE` | Optional, e.g. `Trakt.Copilot` (scp or app role) |
| `TRAKT_COPILOT_DOWNLOAD_SIGNING_KEY` | Random ≥32-char secret — **required with >1 gunicorn worker** |
| `TRAKT_COPILOT_DOWNLOAD_TTL_SECONDS` | Optional, default 300 |
| `TRAKT_COPILOT_PUBLIC_BASE_URL` | External API base URL for download links |

(Existing data-source variables are unchanged; see
`deploy/trakt-mi-api/app_settings.example.json`.)

## Run locally

```bash
pip install -r requirements.txt -r mi_agent_api/requirements.txt

# Local dev only: bypass Entra and serve a local deck/tape + explicit dataset
export TRAKT_COPILOT_AUTH_MODE=disabled
export MI_AGENT_AUTH_ENABLED=false
export MI_AGENT_DATA_CSV=$(ls synthetic_demo/**/*canonical_typed.csv | head -1)
export MI_AGENT_DECK_ROOT=/path/to/decks      # {client}/latest/investor_pack.pptx
export MI_AGENT_PLATFORM_DIR=/path/to/platform # platform_canonical_typed.csv

uvicorn mi_agent_api.app:app --port 8000

curl -s localhost:8000/v1/copilot/mi/query -H 'content-type: application/json' \
     -d '{"question": "What is the total current balance?"}' | jq .
curl -s localhost:8000/v1/copilot/artifacts/latest/investor-deck | jq .
curl -s localhost:8000/v1/copilot/artifacts/latest/canonical-tape | jq .
```

## Package the agent (Trakt-side, no manifest editing)

```bash
python deploy/copilot-agent/package_agent.py --config <client>.json
# → deploy/copilot-agent/dist/trakt-copilot-agent-<client_id>.zip
```

The build substitutes every token from the client configuration and **fails if
any placeholder or unresolved template variable remains** in the output. The
client administrator receives a finished zip and only uploads it, grants admin
consent, and assigns users.

## Deployment model and administrator steps

The full private-organisational deployment model — one-time Trakt setup
(multi-tenant Entra app + Entra SSO auth config in Trakt's Teams developer
portal, restricted to the client's organisation), per-client Trakt packaging,
and the exact client-administrator actions (upload custom app, grant admin
consent, assign users — nothing else) — is documented in
**`docs/copilot_client_deployment_runbook.md`**. The client never edits
manifests, never touches the Teams Developer Portal, never builds a package,
and never configures Azure.

## Testing

```bash
python -m pytest mi_agent_api/tests/test_copilot_actions.py \
                 mi_agent_api/tests/test_copilot_package.py -q
# 23 passed

python -m pytest mi_agent_api/tests -q
# 440 collected: 417 passed, 8 failed — the 8 failures are in
# test_platform_discovery.py and PRE-EXIST on main (verified on a clean
# origin/main checkout: same 8 failures). Not touched, per scope.
```

Covered: valid MI question through the deterministic path; governed-source
enforcement (synthetic fallback → 503, unavailable → 503); malformed request
(422); unauthenticated/unconfigured (401/503); deck found/absent/storage-down
(200/404/503) + controlled download; tape found/absent/storage-down + controlled
download; token tamper/expiry (403); manifests structurally valid; OpenAPI
matches the implemented routes; exactly three functions exposed.

## Known v1 limitations

- "Latest" deck/tape is the current mutable `latest/` blob convention — no
  approval state, versioning, or run-ID linkage exists on artifacts yet
  (deliberately out of scope; see the gap-analysis document).
- The MI action is stateless: Copilot must restate follow-ups as standalone
  questions (its normal behaviour); no server-side conversation memory.
- One client per deployment (existing model); `portfolioId` is not checked
  against the caller's identity beyond tenant/audience/scope validation.
- Download links expire (default 5 min) and are single-artifact,
  latest-only; multi-worker deployments must set
  `TRAKT_COPILOT_DOWNLOAD_SIGNING_KEY`.
- The FastAPI-generated `/openapi.json` (public by existing design) now also
  lists the three Copilot routes; the routes themselves require a bearer token.
- The declarative agent surfaces only what the API returns; if the MI Agent
  returns a controlled error, Copilot reports unavailability rather than
  answering.
