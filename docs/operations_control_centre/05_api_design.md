# 05 — Operations Control API design

New FastAPI app: `operations_control/api/` (served separately from
`mi_agent_api`, e.g. `trakt-ops-api`). Scaffold copied from `mi_agent_api`:
`auth_guard` (Easy Auth / SWA `X-MS-CLIENT-PRINCIPAL`, fail-closed), gateway
prefix stripping, `TraktError` exception handlers (no trace leakage), CORS
allowlist, lifespan warm-up. All endpoints require the `operator` role.

Every response is a `trakt_core.envelope.GovernedResult` envelope; payloads
are operator-safe (plain language, no paths/JSON internals — doc 07 §4).

## 1. Resources and endpoints

### Dashboard

| Method | Path | Purpose |
|---|---|---|
| GET | `/ops/dashboard` | Tile counts + recent activity (new deliveries, awaiting review, awaiting approval, blocked, recently published, recent approvals) |

### Workflows

| Method | Path | Purpose |
|---|---|---|
| GET | `/ops/workflows` | List workflow runs; filters: `status`, `client`, `type`, `period` |
| POST | `/ops/workflows` | Start a workflow. Body: `{outcome: "mi" \| "mi_annex2", client_id?, delivery: {...uploaded file refs}}`. Engine determines workflow type; response includes the classification sentence ("existing client, new portfolio") |
| GET | `/ops/workflows/{workflow_id}` | Full run: stage tracker, per-stage Governed Agent Results, open decision count. **Polling target** |
| POST | `/ops/workflows/{workflow_id}/rerun` | Rerun from the earliest affected stage (idempotent; used after decisions and for recovery) |
| POST | `/ops/workflows/{workflow_id}/cancel` | Abandon with reason |
| GET | `/ops/workflows/{workflow_id}/stages/{stage}` | One stage's Governed Agent Result incl. evidence |

### Uploads

| Method | Path | Purpose |
|---|---|---|
| POST | `/ops/deliveries` | Register a delivery (multipart upload or reference to an arrived blob pack); returns delivery slots with plain-language labels |

### Review Centre

| Method | Path | Purpose |
|---|---|---|
| GET | `/ops/reviews` | All open decisions across workflows; filters: `workflow_id`, `client`, `kind`, `blocking` |
| GET | `/ops/reviews/{decision_id}` | One decision: question, recommendation + provenance, options, allowed scopes, evidence |
| POST | `/ops/reviews/{decision_id}/decision` | Body: `{action: "approve" \| "reject" \| "edit", value?, scope: "file" \| "portfolio" \| "client" \| "global", reason?}`. Approve → persist rule, project into agent sinks, schedule stage rerun. Reject requires `reason` |

### Rules Library

| Method | Path | Purpose |
|---|---|---|
| GET | `/ops/rules` | Search: `q`, `kind`, `scope`, `client`, `status` |
| GET | `/ops/rules/{rule_id}` | Current version + where applied |
| GET | `/ops/rules/{rule_id}/history` | All versions with the approvals that created them |
| POST | `/ops/rules/{rule_id}/retire` | Retire a rule (reason required; audited) |

### Publication & history

| Method | Path | Purpose |
|---|---|---|
| GET | `/ops/publications` | Awaiting-approval and recent publications |
| GET | `/ops/publications/{publication_id}` | Reconciliation summary, what changed vs last publication, rule versions used |
| POST | `/ops/publications/{publication_id}/approve` | The **only** path to production `latest` (invokes existing promote) |
| POST | `/ops/publications/{publication_id}/reject` | Hold with reason |
| POST | `/ops/publications/{publication_id}/rollback` | Approval-gated re-publish of the prior version |
| GET | `/ops/history` | Reporting history per client/portfolio: periods, versions, rule versions, publication dates |
| GET | `/ops/history/compare?client=…&period_a=…&period_b=…` | Business-language delivery comparison |

### Reference & audit

| Method | Path | Purpose |
|---|---|---|
| GET | `/ops/clients` | Known clients + portfolios (drives the Start screen) |
| GET | `/ops/audit` | Audit trail; filters: `workflow_id`, `decision_id`, `rule_id`, date range |
| GET | `/health` | Liveness + storage/config checks (open path) |

## 2. Execution model

- **Synchronous API, asynchronous runs.** `POST /workflows` and `/rerun`
  return immediately with the run in `running`; the engine executes
  `run_orchestration()` in a background worker (FastAPI background task /
  single-process worker initially — consistent with the codebase's existing
  in-process invocation in `orchestrator_invoke.py`). The UI polls
  `GET /ops/workflows/{id}`.
- **Idempotency.** Workflow creation takes a client-supplied idempotency key;
  rerun is safe to repeat (orchestrator `--resume` skips completed steps).
- **Concurrency.** One active run per (client, portfolio, period); a second
  start returns the existing run.
- **Upgrade path.** If run volume outgrows in-process execution, the engine's
  invoker is swapped for a queue-backed worker without any API change.

## 3. Error model

`TraktError` codes reused; new codes namespaced `OPS_*`
(`OPS_WORKFLOW_NOT_FOUND`, `OPS_DECISION_ALREADY_RESOLVED`,
`OPS_ILLEGAL_TRANSITION`, `OPS_PUBLICATION_NOT_PREPARED`, …). The UI renders
only the friendly message + reference code; details go to logs/audit.

## 4. AuthZ summary

- Role `operator` required for everything under `/ops`.
- `decided_by` on every decision/publication comes from the authenticated
  principal, never from the request body.
- Tenancy: client scoping enforced server-side via `trakt_core.tenancy` so an
  operator only sees authorised clients.
