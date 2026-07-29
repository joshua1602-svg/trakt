# Trakt Operations Control Centre (UI)

A calm, non-technical front end for the Trakt Operations Control API
(`operations_control/api/app.py`). Built for a business operator: every screen
answers *what happened, does it matter, and what do I need to do next* — no
technical detail leaks into the interface.

## Run it

```bash
cd frontend/operations-control-ui
npm install

# With a real backend (proxied in dev):
npm run dev
# The dev server proxies /ops and /health to VITE_PROXY_TARGET
# (default http://127.0.0.1:8100), so start the Operations Control API there.

# With no backend at all (canned demo data):
VITE_OPS_MODE=mock npm run dev
```

Other scripts:

```bash
npm run build     # type-check + production build (dist/)
npm run preview   # serve the production build
npm run lint      # tsc --noEmit
npm run test      # vitest (jsdom)
```

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `VITE_OPS_API_URL` | `""` (same origin) | Base URL the HTTP client calls. Leave empty in dev so requests go through the Vite proxy. |
| `VITE_OPS_MODE` | unset | Set to `mock` to use `MockOpsClient` — realistic canned data, no backend, no sign-in. |
| `VITE_PROXY_TARGET` | `http://127.0.0.1:8100` | Where the Vite dev server forwards `/ops` and `/health`. |

## Sign-in / access key

The API authenticates every request with an `X-Operator-Token` header. The UI
keeps that token in `localStorage` under the key `trakt_ops_token`:

- If no token is stored, a minimal full-screen card asks the operator to
  paste their access key.
- Every request sends the stored token.
- If the API answers 401, the token is cleared and the sign-in card
  reappears.

In mock mode there is no sign-in.

## How it talks to the API

- `src/api/types.ts` — payload types for the operator endpoints.
- `src/api/adminTypes.ts` — payload types for `/ops/me` and
  `/ops/admin/config/…`, mirroring the backend read models.
- `src/api/OpsClient.ts` — the client interface (plus `OpsError`, which always
  carries a plain-English message).
- `src/api/HttpOpsClient.ts` — `fetch`-based implementation. API error bodies
  (`{ok: false, message}`) surface their `message` directly; network failures
  show "Trakt could not be reached. Check your connection and try again."
- `src/api/MockOpsClient.ts` — stateful canned data covering every endpoint
  (a workflow needing review with two mapping decisions, one ready to
  publish, a published report, and several rules). `new MockOpsClient(0,
  "operator")` signs in as an ordinary operator.
- `src/api/MockConfigAdmin.ts` — a stateful stand-in for the configuration
  lifecycle that enforces the same refusals the backend enforces (an
  unvalidated draft cannot be activated).
- `src/api/index.ts` — picks Mock when `VITE_OPS_MODE === "mock"`, else Http.

## Screens

| Route | Screen |
| --- | --- |
| `/` | Home — stat tiles, "Needs your attention", "Recently published" |
| `/new` | Start workflow — outcome, client/portfolio/period, files, classification, Start |
| `/workflows` | Workflow list with status filter chips |
| `/workflows/:id` | Workflow detail — stage tracker, stage panels, publish/hold/run-again (polls every 2.5s while running) |
| `/reviews` | Review Centre (filter with `?workflow=...`) |
| `/reviews/:id` | Answer a question — options, scope, confirm/amend/reject |
| `/rules` | Rules Library — search, filters, per-rule history |
| `/history` | Reporting history grouped by client |
| `/admin/config` | **Administrators only** — Platform configuration overview |
| `/admin/config/system` | System package: active, draft, checks, compare, parts, impact, history |
| `/admin/config/assets` | Asset packages, supported regimes, compatibility, package lifecycle |
| `/admin/config/regimes` | Regime packages, schema, policies, compatible assets, lifecycle |
| `/admin/config/history` | Configuration audit trail in plain English |

## Administrator configuration area

`Platform configuration` appears in the navigation only for a principal the
backend reports as an administrator (`GET /ops/me`). That is a convenience, not
a control: every administrator route re-checks the role before rendering, and
every request behind it is authorised again by the backend. A 403 from any
administrator call lands on the same plain-English access-denied state
(`useAdminLoad` separates "refused" from "failed").

The governed lifecycle is the backend's, exactly as it already exists:

```
Draft ──▶ Validate ──▶ Activate        Active ──▶ Roll back to a prior version
```

The interface never mutates an active version, never bypasses validation, and
never offers Activate for a version the backend has not marked valid.
Activation and rollback both require an explicit confirmation dialog that
states which package and version, what it replaces, and that running workflows
stay pinned to the version they started with.

Reusable pieces live under `src/components/admin/`:
`ConfigurationPackageCard`, `VersionHistoryTable`, `ValidationResultPanel`,
`ConfigurationDiff`, `ActivationDialog`, `RollbackDialog`, `DependencyPanel`,
`AuditTimeline`, `CompatibilityTable`, `ImpactPanel`, `FileInspector`.

Comparison and impact are computed by the backend
(`GET /ops/admin/config/{layer}/compare`, `.../impact`) — the browser renders
them and never infers configuration meaning of its own. Drafts are review-only
here: contents are created through the existing supported change process.

## Copy rules

All chrome strings live in `src/lib/copy.ts`. `src/lib/copy.test.ts` fails the
build if any string contains technical wording (paths, format names, internal
identifiers, error dumps).
