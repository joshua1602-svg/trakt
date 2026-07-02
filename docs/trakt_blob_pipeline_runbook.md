# Trakt Blob → Agentic MI → React Runbook

End-to-end operations for the Azure Blob → agentic MI pipeline (LLM-enabled) →
React MI dashboard rendering. This runbook is the single source of truth for the
Azure-side settings you (the human operator) must apply — Claude Code cannot set
Azure App Settings or create Event Grid subscriptions.

The pipeline logic is the same one the Codespaces CLI drove; these settings +
wiring make it fire automatically from blob uploads and render the latest data.

---

## 1. Function App settings (resource group `trakt`, Function App `trakt`)

Set these under **Configuration → Application settings**. Restart the app after
changing them.

| Setting | Value | Why |
|---|---|---|
| `TRAKT_BLOB_CONTAINER` | `raw-v2` | Container the Event Grid handler accepts. The legacy default is `raw`; production uploads land in `raw-v2`. |
| `TRAKT_BLOB_CONNECTION` | *(storage connection string)* | Durable storage for registry/approvals/canonicals. If unset in Azure, `AzureWebJobsStorage` is used as a fallback. |
| `TRAKT_SOURCE_REGISTRY_URI` | `blob://trakt-state/registry/source_registry.yaml` | Durable registry — the single source of truth for known sources + pinned fingerprints. |
| `TRAKT_TRIGGER_OUT` | `/tmp/trakt/blob_trigger` | Writable scratch root (Azure wwwroot is read-only). Ephemeral; final artifacts go to Blob. |
| `TRAKT_PACK_MARKER` | `_READY.json` | Completion marker uploaded last; only this file starts the pipeline. |
| `TRAKT_STATE_CONTAINER` | `trakt-state` | State container (registry, approvals, events, run ledger). |
| `TRAKT_PROCESSED_CONTAINER` | `processed-v2` | Output container (accepted + platform canonicals, regime, pipeline snapshots). |
| `TRAKT_RAW_CONTAINER` | `raw-v2` | Raw container name used by the layout/backfill. |

### LLM (Phase 2 — required for the agentic resolver + approval policy)

| Setting | Value | Why |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(key)* | Enables the LLM mapping resolver + advisor. Absent ⇒ deterministic-only fallback. |
| `TRAKT_LLM_ENABLED` | `true` | Master switch. Off ⇒ deterministic only. |
| `TRAKT_LLM_MODE` | `resolving` | `resolving` wires the mapping resolver into the automated path; `advisory` keeps LLM advisory-only; `off` disables. |
| `TRAKT_LLM_MODEL` | `claude-haiku-4-5-20251001` | Low-cost model for mapping review (override as needed). |
| `TRAKT_APPROVAL_AUTO_CONF` | `0.85` | (optional) Auto-approve deterministic mapping-confidence threshold. |
| `TRAKT_APPROVAL_AUTO_VALUE_MATCH` | `0.98` | (optional) Auto-approve value-match-rate threshold for changed columns. |
| `TRAKT_APPROVAL_AUTO_LLM_CONF` | `0.95` | (optional) Auto-approve LLM mapping-confidence threshold. |

### MI API (App Service `trakt-mi-api` — Phase 4 rendering)

| Setting | Value | Why |
|---|---|---|
| `MI_AGENT_PLATFORM_URI` | `blob://processed-v2/platform/ERE/latest/platform_canonical_typed.csv` | Stable pointer the API reads; overwritten each funded run. |
| `TRAKT_BLOB_CONNECTION` | *(same connection string)* | Lets the API resolve `blob://` URIs. |
| `MI_AGENT_PIPELINE_URI` | `blob://processed-v2/pipeline/ERE/latest/latest_pipeline_snapshot.json` | Stable pointer for the weekly pipeline snapshot. |
| `MI_AGENT_DATA_CACHE_TTL` | `30` | (optional) Seconds; blob ETag re-check cadence so a new run renders without restart. |

---

## 2. Deploy method (CI)

`.github/workflows/main_trakt.yml` builds `deploy.zip` and deploys via
`az functionapp deployment source config-zip --build-remote true`.

**Critical:** the zip now includes **every** runtime package the entrypoint
imports — `apps/ engine/ mi_agent/ mi_agent_api/ mi_agent_pptx/ analytics_lib/
config/` — plus `function_app.py host.json requirements.txt`. The previous zip
shipped only `engine/ config/`, so `import apps.blob_trigger_app` failed at host
start and the trigger never registered (CI passed because `py_compile` never runs
imports).

The CI **sanity check** now unpacks the actual artifact into a clean dir and runs
`python -c "import function_app"` with `azure-functions` installed — an incomplete
package fails the build loudly instead of silently in production.

> If you prefer `func azure functionapp publish`, it honours `.funcignore` (which
> already keeps the runtime packages) and is equivalent. The zip route is kept
> because the OIDC `az login` is already configured for it.

---

## 3. Event Grid subscription (BlobCreated → Function)

The handler is an **Event Grid trigger** (`function_app.on_raw_blob_event`). Event
Grid is **not retroactive** — blobs (and `_READY.json` markers) uploaded *before*
the subscription existed never fire. Two things are required:

1. **Create/confirm the subscription** on the storage account:
   - Event type: `Microsoft.Storage.BlobCreated`
   - Endpoint: the `trakt` Function (Azure Function endpoint type)
   - Subject filter: `/blobServices/default/containers/raw-v2/` (begins-with) so
     only the accepted container fires.
2. **Backfill history** for anything uploaded before the subscription — Event Grid
   will not replay it. See §4.

Confirm delivery under the subscription's **Metrics** (Delivered / Failed) after a
test upload of a `_READY.json`.

---

## 4. Historical backfill (Phase 3)

Run the backfill to process monthly funded packs + weekly pipeline files already
in the containers (Event Grid will never replay them):

```bash
# Against Azure (needs TRAKT_BLOB_CONNECTION + registry settings above):
python -m apps.blob_trigger_app.backfill --container raw-v2

# Dry run first (enumerate packs + planned decisions, process nothing):
python -m apps.blob_trigger_app.backfill --container raw-v2 --dry-run

# Local filesystem copy of the container tree:
TRAKT_STORAGE_BACKEND=file TRAKT_LOCAL_BLOB_ROOT=/path/to/local/blobroot \
  python -m apps.blob_trigger_app.backfill --container raw-v2
```

- Processes packs **chronologically** (oldest reporting period first) so the
  latest pointer ends on the newest data.
- Honours the **approval policy**: recurring "significantly the same" packs
  auto-approve; new sources + material schema changes route to one-click
  `pending_review`.
- **Idempotent**: consults the durable run ledger in `trakt-state`; re-running is
  a no-op unless `--force`.

---

## 5. Approval policy (Phase 2)

- **One-click human approval** for: a new client / new `source_portfolio` (no
  ACTIVE registry record), or a **material** schema change.
- **Auto-approve** (no human) for recurring uploads of the same / "significantly
  the same" dataset, on combined deterministic + LLM evidence.
- Auto-approve writes an audit trail (evidence + old→new fingerprint) to the run
  record + a governance artifact, and **re-pins** the registry so the next
  identical upload is a clean `deterministic` run.

Thresholds are config-driven (see the `TRAKT_APPROVAL_AUTO_*` settings above).

Operator one-click path (unchanged) for new/material:
`ops approve-recommendations <pack_key>` → `ops promote <pack_key>`. The LLM
pre-fills the mapping so approval is genuinely one click.

---

## 6. Verifying render

- After a monthly run: `GET /mi/snapshot` on the MI API returns the new period
  without a restart (blob ETag re-check).
- After a weekly run: `GET /mi/pipeline/snapshot` returns the new weekly extract.
- The React dashboard shows both as the latest data.
