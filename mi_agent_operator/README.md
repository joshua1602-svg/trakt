# Trakt Operator Console

A **standalone** approval console for the managed service — deliberately **separate**
from the client-facing MI dashboard (its own app, its own deployment, its own auth).
Client users never see it and, more importantly, cannot reach its API.

It surfaces the human one-click approval queue (new clients/portfolios and material
schema changes, with the AI-proposed mapping pre-filled) plus an audit log of the
auto-approvals that did **not** need a human. It wraps the existing approval
workflow (`apps.blob_trigger_app.approvals` / `ops` / `persistence`) — no canonical
/ MI / ESMA logic is touched.

## What the operator can do

- **Review queue** — every item waiting, badged *New client/portfolio*, *Material
  change*, or *Schema change*, with the source, period, old→new fingerprint,
  detected files, column add/remove diff, and the AI-proposed field mapping.
- **Approve & promote** — one click: accept the mapping and promote the source to
  `active` (pins the fingerprint + header signatures so future identical uploads
  run automatically).
- **Reject** — with a required reason.
- **Choose an alternative mapping** — override the mapping config/id before approving.
- **Auto-approval audit** — the recurring "significantly the same" changes that were
  auto-approved and re-pinned, each with its governance evidence.

## Security (read this)

Authorization is **server-side and fail-closed** — hiding the UI is never the
control. Every `/api/*` route requires the operator token; if the token is not
configured the API refuses all requests (`503`).

- `TRAKT_OPERATOR_TOKEN` — required. Presented as `X-Operator-Token:` (or
  `Authorization: Bearer …`). Store it in Key Vault, not in code.
- `TRAKT_OPERATOR_NAME` — optional; the identity recorded as `decided_by` in the
  audit trail.
- `TRAKT_OPERATOR_CORS_ORIGINS` — optional allowlist; **no wildcard** here.

**Deploy it as its own App Service**, fronted by Entra ID (app role) or an IP
allowlist / private endpoint. The shared token above is the minimum bar for a
locked-down internal surface; Entra ID SSO is the recommended production upgrade
(swap `require_operator` for a JWT/role check — the service layer is unchanged).

## Run locally

```bash
TRAKT_OPERATOR_TOKEN=dev-secret \
TRAKT_STORAGE_BACKEND=file TRAKT_LOCAL_BLOB_ROOT=/path/to/blobroot \
uvicorn mi_agent_operator.operator_app:app --port 8099
# open http://localhost:8099 , paste the token, Connect.
```

Against Azure, set `TRAKT_BLOB_CONNECTION` (+ the usual `TRAKT_*` container
settings) instead of the filesystem vars so it reads the same durable
`trakt-state` the pipeline writes.

## Shape

- `service.py` — `OperatorService`: `queue()`, `item()`, `approve()`, `reject()`,
  `edit()`, `audit()`. Pure Python, no HTTP — fully unit-tested.
- `operator_app.py` — the FastAPI app: token auth + JSON API + serves the UI.
- `static/operator_ui.html` — a self-contained single-page UI (no build step, no
  external dependencies).
