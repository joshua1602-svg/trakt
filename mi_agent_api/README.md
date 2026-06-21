# MI Agent API

A thin **FastAPI** layer that exposes the existing MI Agent to the React UI. It
**wraps** `mi_agent.mi_agent_workflow.run_mi_agent_query` and projects the real
`mi_semantics_field_registry.yaml` + `MIQuerySpec` enums — it does **not**
introduce any new analytics or a parallel semantic model.

```
User question → POST /mi/query → run_mi_agent_query
   → MIQuerySpec (deterministic parser) → validate → execute → chart factory
   → adapter → artifact response → React renderer
```

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness + data-source status. |
| `GET` | `/mi/catalogue` | Real semantic layer: `states`, `dimensions` (43), `measures` (37), `dates`, `aggregations`, `chart_types` (`bar,line,scatter,bubble,heatmap,treemap`), `temporal_modes`, `risk_monitor_modes`, `filters`. |
| `POST` | `/mi/query` | Run one MI question. Body: `{ question, portfolio?{id,name,entity}, portfolioId?, asOfDate?, filters?, context? }`. |

### Query response shape

Adapts the `run_mi_agent_query` dict into a React-friendly envelope:

```json
{
  "ok": true,
  "answer": "Chart: Bar · Metric: Balance · Dimension: Region — 10 group(s).",
  "interpreted": "Chart: Bar · Metric: Balance · ...",
  "spec": { "chart_type": "bar", "dimension": "...", "...": "..." },
  "validation": { "ok": true, "errors": [], "warnings": [], "resolved_fields": {} },
  "artifacts": [ { "type": "chart", "...": "..." }, { "type": "table", "...": "..." } ],
  "warnings": [],
  "assumptions": [],
  "metadata": { "portfolioId": "...", "asOfDate": "...", "engine": "mi_agent",
                "source": "python", "mock": false, "resultType": "table",
                "rowCount": 10, "chartType": "bar" }
}
```

## Artifact mapping (`adapters.py`)

| MI Agent output | React artifact |
| --- | --- |
| `query_result` `result_type == "summary"` | `kpi` |
| `chart_result` (`bar`/`line`/`scatter`/`bubble`) | `chart` — rebuilt from the result table (lossless on data); the raw **Plotly figure JSON** is carried in `source.figure` for fidelity |
| `chart_result` (`heatmap`/`treemap`) | falls back to `table` (+ warning) until the React renderer supports them |
| `query_result` tabular data | `table` |
| `validation` errors/warnings | `validation` |

Every artifact carries lineage (`source.engine = "mi_agent.workflow"`, `state`,
`spec`, `asOf`, `portfolio`) and `mock: false`.

## Running it

From the repo root, with the repo's Python deps plus the API extras:

```bash
pip install -r requirements.txt
pip install -r mi_agent_api/requirements.txt

uvicorn mi_agent_api.app:app --reload --port 8000
# GET http://localhost:8000/health
```

### Serving a promoted funded central lender tape

By default the API serves the synthetic demo portfolio. To make the React
dashboard reflect a **promoted funded central lender tape** from an onboarding
run instead, point the data source at it (generic by `client_id` / `run_id`):

```bash
# Option A — explicit tape path
export MI_AGENT_CENTRAL_TAPE=onboarding_output/client_001/mi_2025_10/output/central/18_central_lender_tape.csv

# Option B — resolve by client_id / run_id under an onboarding output root
export MI_AGENT_ONBOARDING_OUTPUT_ROOT=onboarding_output
export MI_AGENT_CLIENT_ID=client_001
export MI_AGENT_RUN_ID=mi_2025_10

uvicorn mi_agent_api.app:app --reload --port 8000
# GET /health -> { "dataSourceKind": "funded_central_lender_tape", ... }
```

The promoted tape is period-scoped, so the dashboard inherently shows the funded
universe (e.g. 33 loans for `mi_2025_10`, 73 for `mi_2025_11`) — never the old
2,196-row universe and never pipeline/KFI rows.

**MI preparation (default for the funded path).** The promoted tape is a thin
canonical funded tape. The API runs the existing MI data-prep layer
(`mi_agent_api/funded_prep.py`) before serving it: it derives the bucket source
fields the tape supports (`current_loan_to_value`, `vintage_year`,
`months_on_book`) and then runs the canonical bucket engine
(`analytics_lib.buckets` over `config/mi/buckets.yaml` — the same engine Streamlit
uses) to materialise `ltv_bucket`, `interest_rate_bucket`, `ticket_bucket`,
`time_on_book_bucket`. So the dashboard gets funded **KPIs and stratifications**,
not just thin KPIs. `/health` reports `dataSourceKind`
(`funded_mi_prepared_dataset` | `funded_central_lender_tape_raw`),
`preparationApplied`, `dimensionsAvailable`, `missingDimensions`. See
`funded_mi_data_path_report.md`.

**Enrichment + LTV derivation.** MI availability is decided by the **active MI
target contract + MI enrichment configuration** (`central_lender_tape.mi_enrichment_fields`)
and the source fields actually present — not by the registry category/layer. A
field tagged regulatory/collateral in the registry can still be an MI dimension
(MI contract enrichment using source fields that may also matter for regulatory
reporting — not contract leakage). Raw client fields beyond the core funded set
(borrower age, geography, broker/channel, original advance/valuation) are promoted
into the funded tape via `mi_enrichment_fields` + 04b entity-key linkage
(collateral/loan), so they become MI dimensions. Region/channel are resolved as
groups (obligor/collateral/`collateral_geography`; origination/broker), so a query
"by region" resolves regardless of which field the source supplied.
**Pipeline-only enrichment** is explicit and config-gated
(`allow_pipeline_enrichment` + `pipeline_enrichment_fields`): a pipeline snapshot
NEVER creates funded rows, but a configured pipeline attribute (e.g. broker) may
enrich an existing funded loan when entity-key matched and period-eligible
(funded/collateral sources take precedence). LTV is a
product rule: `current_loan_to_value` / `original_loan_to_value` prefer an explicit
source value, otherwise are **derived** (`current_outstanding_balance /
current_valuation_amount`, `original_principal_balance / original_valuation_amount`)
in the backend prep — never in React. Missing valuation →
`derivation_inputs_missing` (no misleading LTV). `/health` reports
`missingDimensions` with a reason code; see `funded_mi_missing_dimension_trace.md`
for the per-field raw→mapping→scope→eligibility→tape→prep→React trace, and
`funded_mi_data_path_report.md` for the fields added by prep.

Resolution priority: `MI_AGENT_ANALYTICS_DATASET` →
`MI_AGENT_CENTRAL_TAPE` / `MI_AGENT_ONBOARDING_OUTPUT_ROOT`+client/run →
`MI_AGENT_DATA_CSV` → synthetic demo.

Configuration (env):

- `MI_AGENT_ANALYTICS_DATASET` — explicit path to an already MI-prepared CSV.
- `MI_AGENT_CENTRAL_TAPE` / `MI_AGENT_ONBOARDING_OUTPUT_ROOT` (+ `MI_AGENT_CLIENT_ID`,
  `MI_AGENT_RUN_ID`) — serve a promoted funded central lender tape (prepared above).
- `MI_AGENT_DISABLE_PREP=1` — serve the raw thin tape (KPI-only mode).
- `MI_AGENT_DATA_CSV` — path to a canonical `*_typed.csv` (defaults to the
  bundled `synthetic_demo/**/*canonical_typed.csv`).
- `MI_AGENT_SEMANTICS` — path to the semantics registry (defaults to
  `mi_agent/mi_semantics_field_registry.yaml`).
- `MI_AGENT_CORS_ORIGINS` — comma-separated allowed origins (defaults to the
  Vite dev/preview ports).

## Tests

```bash
python -m pytest mi_agent_api/tests -q
```

## Limitations (v1)

- Serves a **single synthetic demo portfolio**; `portfolioId` selects context
  metadata only, not yet a per-portfolio dataset.
- `filters` and `context` are accepted but not yet injected into the spec
  (the NL parser still drives filters from the question).
- LLM parser path is available in the agent but the API runs **deterministic**
  parsing by default (no API key required, zero cost).
- `heatmap`/`treemap` charts fall back to tables in the React renderer.
