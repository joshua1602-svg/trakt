# Trakt Blob Trigger app

Starts the Trakt pipeline when files are uploaded to Blob Storage. The **Azure
event source is Event Grid**: an Event Grid subscription on the storage account
delivers blob-created events to the **root `function_app.py`** Event Grid
handler, which delegates to the Azure-free **router** here. It does
**routing/inference only** — the Orchestrator and Assembler Agents do the work.

```
Blob uploaded to raw-v2
   → Event Grid → root function_app.py (Event Grid handler)
   → apps/blob_trigger_app/router
   → pack completeness (_READY.json) → source registry decision
   → Orchestrator Agent → Onboarding / deterministic processing
   → Assembler Agent → MI / Regime / All
```

```
{container}/{client_id}/{dataset}/{frequency}/{source_portfolio_id}/{reporting_period}/{filename}
  monthly funded:  raw-v2/ERE/funded/monthly/direct_001/2026-01-31/LoanExtract.csv
  weekly pipeline: raw-v2/ERE/pipeline/weekly/direct_001/2026-W05/PipelineExtract.csv
  acquired ad-hoc: raw-v2/ERE/funded/ad_hoc/acquired_001/2026-01-31/LoanExtract.csv
```

The accepted **container is configurable** via the `TRAKT_BLOB_CONTAINER` app
setting (default `raw`; **production `raw-v2`**). The Event Grid handler reads it
and **rejects only blobs outside that container** — there is no hardcoded
`inbound` (the old skip-`raw-v2` bug).

## Pack completion (`_READY.json` marker)

A reporting pack is usually several files (loan + property + funder …) and Event
Grid fires **per blob**. To avoid starting on the first file (incomplete pack) or
once per file, processing is gated on a **completion marker**: upload the data
files first, then upload `…/{reporting_period}/_READY.json` **last** (name
configurable via `TRAKT_PACK_MARKER`, default `_READY.json`).

- **Data-file events** are acknowledged and logged
  (`event_decision: ignored_data_file_waiting_for_ready`) — the Orchestrator is
  **not** started.
- **The marker event** lists + downloads the now-complete folder, fingerprints
  **all data files** (never the marker), and starts the Orchestrator **exactly
  once** against the whole pack.
- **Idempotency:** a folder-level processed record (`out/.../_packs/{pack_key}.json`);
  a re-fired marker with the same schema is skipped
  (`event_decision: duplicate_ready_ignored`) **unless** the marker carries
  `force_reprocess: true`.

The `_READY.json` body may carry pack metadata (all optional):

```json
{
  "expected_files": ["LoanExtract.csv", "PropertyExtract.csv", "Funder.csv"],
  "regime_required": true,
  "target": "all",
  "force_reprocess": false,
  "source_portfolio_type": "acquired",
  "acquisition_date": "2025-11-30",
  "seller_name": "BigBank plc"
}
```

If `expected_files` is present and any are missing from the folder, the pack
**fails closed** (`event_decision: incomplete_pack_pending_review`).

On the marker event it:
1. parses the path (fail closed if it doesn't match the convention);
2. **completeness** — checks `expected_files` if declared;
3. computes a **pack schema fingerprint** (column names/order, sheet names, file
   type — never cell values; across all data files, not the marker);
4. **idempotency** — skips a duplicate marker unless `force_reprocess`;
5. looks the source up in the **source registry**;
6. decides **source onboarding** vs **deterministic processing**;
7. invokes the **Orchestrator Agent** (never the individual agents);
8. on a successful **funded** pack, refreshes the **central platform canonical**
   via the **Assembler Agent**;
9. writes an **event manifest** to the output/log folder.

## Event manifest / audit (`event_decision`)

Every event writes a manifest carrying `event_id`, `blob_path`, `container`,
`pack_folder`, `event_type` (`data_file`|`ready_marker`), `client_id`,
`dataset_type`, `frequency`, `source_portfolio_id`, `reporting_date`, `target`,
`orchestrator_run_id`, `central_canonical_path`, `error`, and the audit
**`event_decision`**, one of:

`ignored_data_file_waiting_for_ready` · `invalid_path` ·
`new_source_pending_review` · `known_source_processed` ·
`schema_drift_pending_review` · `duplicate_ready_ignored` ·
`incomplete_pack_pending_review` · `known_source_halted` · `failed`.

## Assembler refresh (central platform canonical)

After a **funded** pack processes successfully, the trigger publishes that
portfolio's accepted canonical into a per-client store
(`out/.../_accepted/{client_id}/{source_portfolio_id}_canonical_typed.csv`) and
re-runs the **Assembler Agent** over the store. The Assembler consolidates the
**latest accepted canonical per `source_portfolio_id`** — so the central platform
canonical grows `direct_001` → `direct_001 + acquired_001` → `+ acquired_002`
**without reprocessing any raw files**. Pipeline/forecast packs are MI-only and do
**not** trigger a platform refresh.

## Historical backfill

Day 1 can upload many months of weekly pipeline / monthly funded packs. Each
folder is made ready **independently** by its own `_READY.json`, so each pack is
one processing event (never one run per file). The accepted store keeps the
latest snapshot per portfolio, so re-running the Assembler over 12 monthly funded
packs yields a central canonical at the latest reporting date per portfolio. A
re-fired marker is idempotent unless `force_reprocess: true`.

## New source vs known source

| Registry state | Schema fingerprint | Decision | What runs |
|---|---|---|---|
| no record / no approved mapping | — | `source_onboarding` | Orchestrator in source-onboarding mode (discovery). Approval is human-gated → **pending_review**, stops before production. |
| active record + approved mapping | **matches** saved | `deterministic` | Orchestrator with the saved mapping (no discovery/LLM). |
| active record + approved mapping | **differs** | `schema_drift` | **Fail closed** → `pending_review`. Never processed with a stale mapping. |

## Target selection (Regime/ESMA is never run for pipeline/forecast)

| dataset | frequency | target | Regime/ESMA |
|---|---|---|---|
| funded | monthly / ad_hoc | `mi`, or `all` if `regime_required` (registry **or** `_READY.json`) | only when `all` |
| pipeline | weekly | `mi` | never |
| forecast | * | `mi` | never |

A `_READY.json` `target` override is honoured **only for funded** packs; a
pipeline/forecast override that asks for Regime is ignored (MI-only is enforced).

## Source registry

File-backed (`config/source_registry.yaml` by default — see
`config/source_registry.example.yaml`), one record per
`client_id/source_portfolio_id/dataset/frequency`, easy to migrate to Azure
Table / Cosmos later (the `SourceRegistry.lookup` contract is stable). Each record
carries the approved mapping id/path, expected schema fingerprint + columns, last
successful run/period, `regime_required`, and `status`
(`active` / `pending_review` / `retired`).

## Deployment entrypoint

- **Azure event source: Event Grid.** An Event Grid subscription on the storage
  account delivers blob-created events to the **root `function_app.py`**, which
  is the deployed Function App entrypoint (the Functions host scans the root).
- The root handler is an **`@app.event_grid_trigger`**; it parses the event
  subject, accepts only the configured container, and **delegates to the blob
  trigger router** (`apps/blob_trigger_app/router.py`). All routing/inference
  lives in this package; the root file does Azure I/O + delegation only.
- **`TRAKT_BLOB_CONTAINER=raw-v2`** controls which container is accepted. The
  previous legacy root handler hardcoded the `inbound` container and silently
  skipped `raw-v2`; the accepted container is now configuration, not a constant.
- **Upload data files first; upload `_READY.json` last** to trigger processing.
  Use **one `_READY.json` per reporting pack**.

> The native `@app.blob_trigger` variant in
> `apps/blob_trigger_app/function_app.py` is **not deployed** (kept for local
> `func start`); it is not imported by the root entrypoint, so only the Event
> Grid trigger is active in production. Both delegate to the same router core.

`.funcignore` (repo root) excludes legacy/unneeded files (the legacy Streamlit
app, `frontend/`, docs, tests, generated outputs, root-level sample data) but
**keeps the runtime packages** the handler imports: `engine/`, `mi_agent/`,
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
