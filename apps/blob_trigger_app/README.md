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

**Input folder structure** (preferred, 7 segments incl. `source_book_type`):

```
{client_id}/{source_book_type}/{dataset_type}/{frequency}/{source_portfolio_id}/{period}/{filename}
  direct funded monthly:   ERE/direct/funded/monthly/direct_001/2025-11-30/LoanExtract.csv
  direct pipeline weekly:  ERE/direct/pipeline/weekly/direct_001/2025-W48/PipelineExtract.csv
  acquired funded monthly: ERE/acquired/funded/monthly/acquired_001/2025-11-30/LoanExtract.csv
  acquired one-off pack:   ERE/acquired/funded/ad_hoc/acquired_001/2025-11-30/LoanExtract.csv
```

`source_book_type` ∈ `direct|acquired`, `dataset_type` ∈ `funded|pipeline`,
`frequency` ∈ `weekly|monthly|ad_hoc`, `period` is `YYYY-Www` (weekly) or a
month-end date `YYYY-MM-DD` (monthly/ad_hoc). Invalid structures are **rejected**
with a clear manifest reason (`event_decision: invalid_path`), and a
`source_book_type` that contradicts the `source_portfolio_id` (e.g. `acquired_001`
under `direct/`) is rejected too. The older 6-segment convention
`{client_id}/{dataset}/{frequency}/{pid}/{period}/` is still parsed as a
**deprecated compatibility** path (book type derived from the portfolio id).

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

> **Azure note:** the registry is **read + written** (`upsert` persists
> last-successful run/period and `pending_review` transitions). The default
> `config/source_registry.yaml` lives under the read-only package mount, so in
> Azure point **`TRAKT_SOURCE_REGISTRY`** at a writable, persistent location
> (e.g. an Azure Files mount) — `/tmp` would not survive across instances.

## Persistent storage (production)

Production must not keep state on `/tmp`. A small **storage abstraction**
(`storage.py`) addresses both backends by `blob://{container}/{key}` URIs:
filesystem locally/tests (`blob://c/k` → `{TRAKT_LOCAL_BLOB_ROOT}/c/k`), real
Azure Blob in the cloud (`TRAKT_STORAGE_BACKEND=blob` + `TRAKT_BLOB_CONNECTION`).
The `/tmp/trakt/blob_trigger` root is **scratch only**; final artifacts are
uploaded durably. Layout (`layout.py`, all container names configurable):

| Artifact | Location (default) |
|---|---|
| source registry | `trakt-state/registry/source_registry.yaml` (`TRAKT_SOURCE_REGISTRY_URI`) |
| pending approvals | `trakt-state/approvals/{approval_id}.json` |
| event manifests | `trakt-state/events/{event_id}.json` |
| accepted per-portfolio canonical | `processed-v2/accepted/{client}/{pid}_canonical_typed.csv` |
| central platform canonical (latest) | `processed-v2/platform/{client}/latest/platform_canonical_typed.csv` |
| central platform canonical (period) | `processed-v2/platform/{client}/{period}/platform_canonical_typed.csv` |
| regime outputs | `processed-v2/regime/{client}/{period}/` |
| MI outputs | `processed-v2/mi/{client}/` |

App settings: `TRAKT_RAW_CONTAINER=raw-v2`, `TRAKT_STATE_CONTAINER=trakt-state`,
`TRAKT_PROCESSED_CONTAINER=processed-v2`, `TRAKT_SOURCE_REGISTRY_URI=blob://…`,
`TRAKT_STORAGE_BACKEND=blob|file`, `TRAKT_BLOB_CONNECTION`, `TRAKT_LOCAL_BLOB_ROOT`.

> **The `trakt-state` and `processed-v2` containers must exist** (the SDK does not
> create them). Every storage write logs its full traceback **and the blob URI**
> on failure — loggers `trakt.blob_trigger.{storage,persistence,source_registry,router}`
> — so a silent Azure "Executed (Failed)" surfaces the first failing persistence
> operation (e.g. `REGISTRY SAVE FAILED uri=blob://trakt-state/registry/…`).

## Approval workflow

New sources and schema changes are **human-gated**. When a pack is detected as a
new source (`new_source_pending_review`) or schema drift
(`schema_drift_pending_review`), the trigger writes a **pending approval
artifact** to `trakt-state/approvals/` carrying the schema fingerprint (old + new
on drift), detected files, suggested mapping, and source metadata
(`source_portfolio_type`, `acquisition_date`, `seller_name`, `regime_required`).
Production processing does **not** proceed until approved. CLI:

```bash
python -m apps.blob_trigger_app.approvals list
python -m apps.blob_trigger_app.approvals show <approval_id>
python -m apps.blob_trigger_app.approvals approve <approval_id> --mapping-id ere_acq1_v1 --mapping-config-path config/acq1.yaml
python -m apps.blob_trigger_app.approvals reject  <approval_id> --reason "seller unverified"
python -m apps.blob_trigger_app.approvals promote <approval_id>   # → active registry entry
```

`promote` writes an `active` registry entry (approved mapping + expected
fingerprint); subsequent packs for that source route deterministically. (Local
runs add `--local-root .localblob` to use the filesystem-emulated store.)

## MI API access to the latest central canonical

After a funded pack, the Assembler-refreshed central canonical is uploaded to
`processed-v2/platform/{client}/latest/platform_canonical_typed.csv`. The MI API
reads it either by **mount** — point `MI_AGENT_PLATFORM_DIR` at the mounted
`…/latest/` directory (no code change) — or by **URI**: set
`MI_AGENT_PLATFORM_URI=blob://processed-v2/platform/{client}/latest/…` and the MI
data source resolves/downloads it via the storage abstraction. The React
dashboard's Total / Direct / Acquired / cohort lenses are unchanged.

## Regime / Projection access

When a funded source is in ESMA scope (`regime_required`, target `all`), regime
projection runs over the **persisted central canonical** and outputs are uploaded
to `processed-v2/regime/{client}/{period}/`. The projector logic is unchanged —
ESMA output stays template-clean and the **provenance companion** is written and
uploaded alongside it.

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
- **Writable output root (Azure-safe).** Azure runs from a read-only package
  mount (`/home/site/wwwroot`), so all runtime output (event manifests,
  orchestrator state/run dirs, accepted + central platform canonicals) derives
  from a single writable root resolved by `runtime_paths.resolve_output_root()`:
  `TRAKT_TRIGGER_OUT` if set, else `/tmp/trakt/blob_trigger` in Azure, else
  `out/blob_trigger` locally. Downloaded pack files use the system temp dir
  (`/tmp`). Nothing writes under repo-root `out/` in Azure. Set
  `TRAKT_TRIGGER_OUT` to an Azure Files mount if you need outputs to persist
  beyond the instance.

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
