# Trakt Blob Trigger app

A thin Azure Functions blob trigger that starts the Trakt pipeline when a file is
uploaded to Blob Storage. It does **routing/inference only** — the Orchestrator
Agent does the work.

```
raw/{client_id}/{dataset}/{frequency}/{source_portfolio_id}/{reporting_period}/{filename}
  e.g. raw/ERE/funded/monthly/direct_001/2026-09-30/loan_tape.xlsx
```

On each upload it:
1. parses the path (fail closed if it doesn't match the convention);
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
