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

Configuration (env):

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
