# Trakt Operations Control Centre — backend

The governed operational layer above the existing Trakt agents. Design pack:
`docs/operations_control_centre/`. This package is purely additive — it wraps
`engine.orchestrator_agent` through its adapter seam and calls existing
persistence/promotion functions as libraries; no agent, registry or pipeline
code is modified.

## Run the API locally (file-backed storage)

```bash
export TRAKT_STORAGE_BACKEND=file
export TRAKT_LOCAL_BLOB_ROOT=.localblob          # containers emulated on disk
export TRAKT_OPS_OPERATORS='{"dev-token": {"name": "Operator", "clients": ["*"]}}'
python -m uvicorn operations_control.api.app:app --port 8100
```

Then run the React app (`frontend/operations-control-ui/`, `npm run dev`) and
sign in with `dev-token`.

## Environment

| Variable | Default | Purpose |
|---|---|---|
| `TRAKT_OPS_OPERATORS` | *(unset — API refuses all access)* | JSON map of token → `{name, clients, role}`; `"clients": ["*"]` = all, `"role": "admin"` = may administer platform configuration |
| `TRAKT_OPS_CONTAINER` | `operations-control` | Governance container name |
| `TRAKT_OPS_STAGING_ROOT` | `.ops_state/staging` | Orchestrator staging (run outputs before publication) |
| `TRAKT_OPS_MEMORY_ROOT` | `.ops_state/client_memory` | Where approved rules are projected as client-memory YAML |
| `TRAKT_OPS_CORS_ORIGINS` | `http://localhost:5173` | Browser origins |
| `TRAKT_STORAGE_BACKEND` / `TRAKT_LOCAL_BLOB_ROOT` / `TRAKT_BLOB_CONNECTION` | — | Existing storage selection (unchanged) |

In Azure, set `TRAKT_BLOB_CONNECTION` and create the `operations-control`
container; everything else is the same code path.

## Administrator configuration API

The governed system / regime / asset packages are administered through
`/ops/admin/config/…`. Every route calls `require_admin`, so an ordinary
operator is refused with 403 regardless of what the browser offers. The
lifecycle lives in `configuration/packages.py`; `configuration/admin_views.py`
holds read-only presentation (no validation or version-state logic is
duplicated there, and none of it is duplicated in React).

| Endpoint | Returns |
|---|---|
| `GET /ops/me` | `{name, role, is_admin, clients}` — what the UI may offer |
| `GET /ops/admin/config` | Per-layer active/draft/hash/validation, asset + regime catalogues, compatibility matrix, what needs an administrator, recent activations and rollbacks |
| `GET /ops/admin/config/catalogue` | Asset and regime entities, compatibility matrix, compatibility issues |
| `GET /ops/admin/config/audit` | Administrator audit trail with plain-English descriptions, plus `chain_intact` |
| `GET /ops/admin/config/{layer}/{version}` | One version: file inventory with plain names and hashes, dependencies, validation summary, activation blockers; `?file=` adds that file's contents |
| `GET /ops/admin/config/{layer}/compare?from_version=&to_version=` | Grouped added/changed/removed files with a structured line diff |
| `GET /ops/admin/config/{layer}/impact?version=` | Clients, portfolios and workflows pinned to a version |
| `POST /ops/admin/config/{layer}/draft` | Create a draft from a base version |
| `POST /ops/admin/config/{layer}/{version}/validate` | Run the checks; returns the raw result and a plain-English summary |
| `POST /ops/admin/config/{layer}/{version}/activate` | Activate a validated version; 409 `OPS_CONFIG_NOT_READY` if unchecked, 409 `OPS_CONFIG_INCOMPATIBLE` with `blockers` if a dependency or asset-to-regime relationship would break |
| `POST /ops/admin/config/{layer}/rollback` | Re-activate a prior version (the replaced version is kept) |

Activation and rollback refuse a candidate that drops a governed file, or that
would leave an asset reporting under a regulatory package the candidate does
not carry. Running workflows stay pinned to the version they resolved with;
activation only changes what future runs resolve.

## Tests

```bash
python -m pytest tests/operations_control/ -q
python -m pytest tests/operations_control/test_admin_config_api.py -q   # admin config API
```

Covers workflow transitions, restart recovery, all four delivery
classifications, rule-scope precedence, mapping approval persistence, LLM
advisory gating, rejection, rerun, publication gating, duplicate-action
protection, cross-tenant isolation, plain-English failure translation, and
reconstruction from persisted state.
