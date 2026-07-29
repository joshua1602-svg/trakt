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
| `TRAKT_OPS_OPERATORS` | *(unset — API refuses all access)* | JSON map of token → `{name, clients}`; `"clients": ["*"]` = all |
| `TRAKT_OPS_CONTAINER` | `operations-control` | Governance container name |
| `TRAKT_OPS_STAGING_ROOT` | `.ops_state/staging` | Orchestrator staging (run outputs before publication) |
| `TRAKT_OPS_MEMORY_ROOT` | `.ops_state/client_memory` | Where approved rules are projected as client-memory YAML |
| `TRAKT_OPS_CORS_ORIGINS` | `http://localhost:5173` | Browser origins |
| `TRAKT_STORAGE_BACKEND` / `TRAKT_LOCAL_BLOB_ROOT` / `TRAKT_BLOB_CONNECTION` | — | Existing storage selection (unchanged) |

In Azure, set `TRAKT_BLOB_CONNECTION` and create the `operations-control`
container; everything else is the same code path.

## Tests

```bash
python -m pytest tests/operations_control/ -q
```

Covers workflow transitions, restart recovery, all four delivery
classifications, rule-scope precedence, mapping approval persistence, LLM
advisory gating, rejection, rerun, publication gating, duplicate-action
protection, cross-tenant isolation, plain-English failure translation, and
reconstruction from persisted state.
