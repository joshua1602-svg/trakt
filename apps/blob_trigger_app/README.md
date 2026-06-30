# Trakt Blob Trigger app

A thin Azure Functions blob trigger that starts the Trakt pipeline when a file is
uploaded to Blob Storage. It does **routing/inference only** — the Orchestrator
Agent does the work.

```
{container}/{client_id}/{dataset}/{frequency}/{source_portfolio_id}/{reporting_period}/{filename}
  e.g. raw/ERE/funded/monthly/direct_001/2026-09-30/loan_tape.xlsx
```

The watched **container is configurable** via the `TRAKT_BLOB_CONTAINER` app
setting (default `raw`; e.g. `raw-v2`). It is referenced as
`%TRAKT_BLOB_CONTAINER%` in the trigger binding path *and* read in code to anchor
the path parser — so set it once. (`%…%` is resolved by the Functions host, so
the setting must be present; the example settings ship it defaulted to `raw`.)

## Pack completion (READY sentinel)

A reporting pack is usually several files (loan + cashflow + collateral …). The
trigger fires **per blob**, so to avoid starting on the first file (against an
incomplete pack) or running once per file, processing is gated on a **completion
marker**: the uploader writes the data files, then writes a marker file
**last** — `…/{reporting_period}/_READY` (name configurable via
`TRAKT_PACK_MARKER`, default `_READY`).

- **Data-file events** are acknowledged and logged (`status: awaiting_pack`) — the
  Orchestrator is **not** started.
- **The marker event** lists + fingerprints the now-complete folder and starts the
  Orchestrator **exactly once** against the whole pack.
- **Idempotency:** a folder-level processed marker (`out/.../_packs/{pack_key}.json`)
  records the run; a duplicate/re-fired marker with the same schema is skipped
  (`status: already_processed`). New data (different fingerprint) is *not* skipped.

On each upload it:
1. parses the path (fail closed if it doesn't match the convention);
1a. **completion gate** — only the `_READY` marker starts processing; data files
   are logged as pack members;
2. computes a **schema fingerprint** (column names/order, Excel sheet names, file
   type — never cell values);
3. looks the source up in the **source registry**;
4. decides **source onboarding** vs **deterministic processing**;
5. invokes the **Orchestrator Agent** (never the individual agents);
6. writes an **event manifest** to the output/log folder.

## New source vs known source

| Registry state | Schema fingerprint | Decision | What runs |
|---|---|---|---|
| no record / no approved mapping | — | `source_onboarding` | Orchestrator in source-onboarding mode (discovery). Approval is human-gated → **pending_review**, stops before production. |
| active record + approved mapping | **matches** saved | `deterministic` | Orchestrator with the saved mapping (no discovery/LLM). |
| active record + approved mapping | **differs** | `schema_drift` | **Fail closed** → `pending_review`. Never processed with a stale mapping. |

## Target selection (Regime/ESMA is never run for pipeline/forecast)

| dataset | frequency | target | Regime/ESMA |
|---|---|---|---|
| funded | monthly | `mi`, or `all` if `regime_required` on the source record | only when `all` |
| pipeline | weekly | `mi` | never |
| forecast | * | `mi` | never |

## Source registry

File-backed (`config/source_registry.yaml` by default — see
`config/source_registry.example.yaml`), one record per
`client_id/source_portfolio_id/dataset/frequency`, easy to migrate to Azure
Table / Cosmos later (the `SourceRegistry.lookup` contract is stable). Each record
carries the approved mapping id/path, expected schema fingerprint + columns, last
successful run/period, `regime_required`, and `status`
(`active` / `pending_review` / `retired`).

## Deployment entrypoint

Azure Functions loads the Function App from the **root** `function_app.py`. That
file is a **thin shim** — it contains no logic and simply re-exports this app:

```python
# /function_app.py  (repo root)
from apps.blob_trigger_app.function_app import app
```

So the host discovers the trigger here (`apps/blob_trigger_app/function_app.py`)
via delegation; all routing/inference lives in this package and its Azure-free
decision core (`router.py`).

> **Why:** the previous root `function_app.py` was a legacy Event Grid trigger
> bound to the `inbound` container. While it was the deployed entrypoint, uploads
> to the current `raw-v2` container were silently skipped. The shim makes the
> blob-trigger app the live entrypoint, with the watched container configurable
> via `%TRAKT_BLOB_CONTAINER%`.

`.funcignore` (repo root) excludes legacy/unneeded files (the legacy Streamlit
app, `frontend/`, docs, tests, generated outputs, root-level sample data) but
**keeps the runtime packages** the shim imports: `engine/`, `mi_agent/`,
`mi_agent_api/`, `config/`, and `apps/blob_trigger_app/`. Root-level sample-data
globs are anchored with a leading `/` so nested runtime data (e.g.
`config/system/*.xsd`) is preserved.

## Local run

```bash
pip install -r apps/blob_trigger_app/requirements.txt
cp apps/blob_trigger_app/local.settings.example.json apps/blob_trigger_app/local.settings.json
# edit TRAKT_BLOB_CONNECTION + paths, then:
cd apps/blob_trigger_app && func start        # requires Azure Functions Core Tools
```

The decision core (`router.handle_blob_event`) has **no Azure dependency** and is
unit-tested directly (`tests/test_blob_trigger_app.py`).

## Remaining manual step

Mapping **approval is human-gated** and not yet automated. A new or changed source
is taken to `pending_review` and **stops before production**; an operator approves
the mapping (the onboarding `accept-target-advice` / `approve-non-blocking` CLIs)
and sets the registry record to `active` with the approved mapping id/path +
expected fingerprint. After that, routine files route deterministically.
