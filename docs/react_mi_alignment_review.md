# React MI Agent — Streamlit Alignment Review

**Status:** Foundation review (precedes the React architecture refactor)
**Audience:** Engineers building the React MI Agent front end and the future
Python MI Agent API.
**Scope:** Catalogue the existing Python/Streamlit MI platform, map it to the
React presentation layer, and recommend a durable artifact schema so that the
React app becomes the presentation/orchestration layer for the existing
analytics — not a parallel dashboard product.

```
User Question → MI Agent (interpreter) → Analytics Engine (states/risk/scenario)
             → Artifact Schema → React Renderer
```

---

## 1. Existing MI capability inventory

The Streamlit app (`analytics/streamlit_app_erm.py`) builds its tabs dynamically
based on module availability (`streamlit_app_erm.py:1047-1074`):

| Workflow / Tab | Source | Notes |
| --- | --- | --- |
| **Pipeline MI** | `analytics/tab_pipeline.py` | Weekly origination flow, funnel snapshots, stage MI, reconciliation to funded tape. Optional (`PIPELINE_TAB_AVAILABLE`). |
| **Funded Exposures** | `streamlit_app_erm.py:1094+` | Portfolio overview KPIs, stratifications, concentration heatmap, bubble charts. Always present. |
| **Forward Exposure** | `analytics/pipeline_forward_risk.py`, `pipeline_expected_funding.py` | Expected funding forecast, run-rate trajectory, weeks-to-£100MM, funded-vs-forward by region/broker. Optional. |
| **Scenario Analysis** | `analytics/scenario_engine.py` | HPI / rate / prepay / mortality / move-to-care assumptions → balance run-off, NNEG losses, LTV projection. Loan-level drill-down. |
| **Static Pools** | `analytics/static_pools_core.py` | Vintage cohorts, months-on-book curves, status-migration Sankey, run-off. |
| **Risk Monitoring** | `analytics/risk_monitor.py`, `mi_agent/risk_monitor/` | Regulatory limit checks, concentration RAG, migration matrices, breach cards. Optional. |
| **Concentration Analysis** | `analytics_lib/concentration.py` | HHI / top-N / single-name / limit-usage RAG; surfaced in heatmaps and risk tab. |
| **Portfolio Overview** | `streamlit_app_erm.py:1094-1242` | 10-card KPI strip (loans, balance, WA LTV/rate/age, avg/max loan, largest region/broker). |
| **Validation & Governance** | `analytics/mi_prep.py:65-111` (`assert_trusted_canonical`), reconciliation control | Canonical-trust checks, data-quality flags, reconciliation summaries. |

The **MI Agent** (`mi_agent/`) is a separate, governed NL→query layer:
question → `MIQuerySpec` (deterministic or LLM) → validate → execute against
canonical data → Plotly chart + result table + metadata + warnings
(`mi_agent/mi_agent_workflow.py:run_mi_agent_query`).

---

## 2. Chart inventory

### MI Agent chart factory — `mi_agent/mi_chart_factory.py`
Canonical, governed chart builders (`_BUILDERS`, line ~720):

| `chart_type` | Builder | Consumes |
| --- | --- | --- |
| `bar` | `_build_bar` | `MIQueryResult.data` (dimension + measure columns) |
| `line` | `_build_line` | time/ordered dimension + measure |
| `scatter` | `_scatter_like(bubble=False)` | x, y |
| `bubble` | `_scatter_like(bubble=True)` | x, y, size, color |
| `heatmap` | `_build_heatmap` | two dimensions + value |
| `treemap` | `_build_treemap` | hierarchy + value |

Output is an `MIChartResult { fig, chart_type, title, subtitle, warnings, metadata }`
with `to_html()/to_json()`. Formatting helpers: `compact_currency` (£1.2m/£450k),
`compact_number`, `format_percent` (respects `percent_scale_detected`).

### Streamlit chart library — `analytics/charts_plotly.py` + app
- **KPI strips** (10-card grid, HTML `.kpi-box`).
- **Stratification dual-bar** (`strat_bar_chart_pure`): balance + count side by side.
- **Treemaps**: geographic (top 10), broker (top 15).
- **Concentration heatmap**: configurable rows/cols, balance/count toggle.
- **Bubble**: balance vs property value; LTV vs borrower age.
- **Vintage**: combo bar (balance) + line (count), Month/Quarter/Year granularity.
- **Scenario**: balance run-off line, NNEG annual bar + cumulative line, LTV line.
- **PPTX export** (`generate_pptx_client.py`): matplotlib equivalents.
- **Waterfall**: pipeline bridge (funded → forecast) — conceptual in forward
  exposure; rendered explicitly in the React prototype.

### Theme constants (single source of truth)
`PRIMARY #232D55` (navy), `SECONDARY #919DD1` (periwinkle), `ACCENT #BFBFBF`,
`TEXT #2D2D2D`, positive `#2E7D5B`, negative `#B23A48`, sequential
light→navy `#F2F4F8 → #919DD1 → #232D55` (`mi_chart_factory.DEFAULT_THEME:65-93`,
`charts_plotly.py:17-22`). The React `theme.ts` mirrors these tokens.

---

## 3. Dimension inventory

Consolidated from `config/mi/stratification_catalogue.yaml` (25 dimensions),
`config/mi/risk_monitor.yaml`, the semantics registry and `mi_prep.add_buckets`:

| Dimension (semantic key) | Bucketed? | Bucket source |
| --- | --- | --- |
| `geographic_region_obligor` / `geographic_region` | categorical | NUTS UKC–UKN |
| `broker_channel` | categorical | — |
| `origination_channel` | categorical | — |
| `erm_product_type` / `product_type` | categorical | — |
| `amortisation_type` | categorical | — |
| `interest_rate_type` | categorical | fixed/variable |
| `account_status` / `arrears_status` | categorical | — |
| `pipeline_stage` | categorical | KFI/APPLICATION/OFFER/COMPLETED/WITHDRAWN |
| `internal_risk_grade` | categorical | A–G |
| `internal_risk_stage` | categorical | watchlist |
| `ifrs9_stage` | categorical | Stage 1–3 |
| `number_of_borrowers` (borrower_structure) | categorical | single/joint |
| `ltv_bucket` | banded | 0-20%…≥100% |
| `original_ltv_bucket` | banded | same |
| `borrower_age` / `youngest_borrower_age` | banded | <55…85+ |
| `interest_rate_bucket` | banded | <2%…≥8% |
| `pd_bucket` | banded | <0.25%…≥25% |
| `lgd_bucket` | banded | <10%…≥75% |
| `ead_bucket` / `balance_band` | banded | <50k…≥1m |
| `time_on_book` | banded | 0-6m…10y+ |
| `vintage_year` (cohort) | derived | year/quarter/month |
| `portfolio_id` / `spv_id` / `acquired_portfolio_id` | segmentation | requires portfolio reference config |

Band edges/labels live in `config/mi/buckets.yaml`; quantile bucketing in
`mi_agent/quantile_buckets.py` (`balance_band`, `interest_rate_bucket`,
`time_on_book_bucket`).

---

## 4. Measure inventory

| Measure (semantic key) | Format | Default agg |
| --- | --- | --- |
| `current_outstanding_balance` / `total_balance` | currency | sum |
| `current_principal_balance` | currency | sum |
| `expected_funded_amount` / `forecast_funded_balance` | currency | sum |
| `loan_count` (derived from row/id count) | integer | count |
| `current_interest_rate` | percent | weighted_avg (by balance) |
| `current_loan_to_value` | percent | weighted_avg |
| `original_loan_to_value` | percent | weighted_avg |
| `youngest_borrower_age` | integer | weighted_avg |
| `arrears_balance` / `interest_arrears_amount` / `principal_arrears_amount` | currency | sum |
| `default_amount` | currency | sum |
| `redemptions_received_in_period` | currency | sum |
| `recoveries_in_period` | currency | sum |
| `expected_nneg_loss` / `cumulative_expected_nneg` | currency | sum (scenario) |
| `balance_share` / `count_share` | percent | derived |
| `probability_of_default` / `loss_given_default` / `exposure_at_default` | decimal/currency | weighted_avg |
| `forecast_funding_probability` | decimal | — |

Aggregation vocabulary (`mi_query_spec.AGGREGATIONS`): `sum, avg, weighted_avg,
count, count_distinct, median, distribution, loan_level, balance_sum`. Default
weight field is `current_outstanding_balance`.

---

## 5. Artifact inventory

Outputs the Python stack already produces, mapped to React artifact types:

| Python output | Shape | React artifact type |
| --- | --- | --- |
| KPI strip (overview measures) | dict of label→value+delta | `kpi` |
| `MIChartResult` / Plotly figures | fig + type + title | `chart` |
| `MIQueryResult.data` / stratification tables | tidy DataFrame | `table` |
| `assert_trusted_canonical` + reconciliation | issues + summary | `validation` |
| `RiskMonitorResult` (concentration/migration/flags) | RAG table / matrix | `risk` |
| `scenario_engine.project_portfolio` | year×metrics DataFrame | `scenario` |
| Concentration top-N / HHI / limit-usage | RAG table | `risk` / `chart` |
| Static pool vintage series / Sankey | MOB×metric / nodes+links | `chart` |
| CSV / Excel / PPTX / HTML exports | bytes | artifact `export` actions |

---

## 6. Existing data contracts

- **Canonical DataFrame**: trusted canonical columns (`mi_prep`), required
  `loan_identifier`, `data_cut_off_date`, `current_principal_balance`,
  `current_valuation_amount`, `current_loan_to_value`, `current_interest_rate`,
  `origination_date`, `youngest_borrower_age`, region/product/broker. Buckets +
  aliases derived on load (`mi_prep.add_buckets`, `add_presentation_aliases`).
- **Filters**: `{dimension: [values]}` over vintage/product/region; applied to
  produce `df_view`; KPIs/charts recompute. No explicit context object —
  implicit in the filtered frame.
- **Reporting date**: YAML `static_reporting_date` or max of
  `data_cut_off_date/as_of_date/reporting_date` in the frame.
- **Portfolio context**: `portfolio_id/spv_id/acquired_portfolio_id` via
  `PortfolioReferenceConfig` (`mi_agent/portfolio_reference.py`).
- **Agent request**: `MIQuerySpec` (v1 chart core + v2 state/temporal/risk
  fields). Enums: `STATES`, `TEMPORAL_MODES`, `RISK_MONITOR_MODES`,
  `BUCKET_STRATEGIES`, `CHART_TYPES`, `AGGREGATIONS`.
- **Agent response** (`run_mi_agent_query`):
  `{ ok, error, question, parser_mode, spec, interpreted, validation,
  query_result{data,row_count,resolved_fields,metadata,warnings},
  chart_result{fig,chart_type,title,subtitle}, warnings, metadata }`.
- **Risk result** (`RiskMonitorResult`): `{ kind, frame, issues, metadata }`
  with RAG `status` per group; thresholds amber 0.20 / red 0.30; limit-usage
  amber 0.80 / red 1.00.

---

## 7. Current React gaps versus Streamlit

The first-pass React prototype (`frontend/mi-agent-ui`) covered: chat panel, KPI
strip, bar/line/area/waterfall charts, one table, a validation summary, and a
keyword mock engine. Gaps versus the Streamlit/Python platform:

1. **No agent client abstraction** — UI imported mock logic directly
   (`data/agentEngine.ts`), so a real backend was not swappable.
2. **Artifact types incomplete** — no **risk** (RAG concentration / migration /
   breaches) or **scenario** (projection) artifacts; charts limited.
3. **No spec/contract alignment** — mock response didn't mirror `MIQuerySpec` or
   the `run_mi_agent_query` response (no `spec`, `interpreted`, `resolved_fields`,
   `metadata`, `warnings`).
4. **Dimensions/measures not modelled** — no catalogue mirroring the semantics
   registry; nothing to drive future query building.
5. **No portfolio/reporting context model** — selectors were cosmetic, not part
   of a typed `PortfolioContext`/`ReportingContext` passed to the agent.
6. **No persistence** — chat/pins/context lost on reload.
7. **Thin states** — limited empty/error/loading handling; no lineage, source
   metadata, or mock-data disclosure per artifact.
8. **No type guards / tests**.
9. **Chart types** — missing heatmap, treemap, scatter/bubble that the factory
   supports.

---

## 8. Recommended future artifact schema

The React app should be **artifact-driven**: the agent returns a list of typed
artifacts and the renderer dispatches by `type`. The schema mirrors the Python
outputs so a future API maps 1:1.

```ts
type ArtifactType = "kpi" | "chart" | "table" | "validation" | "risk" | "scenario";

interface ArtifactBase {
  id: string;
  type: ArtifactType;
  title: string;
  description?: string;
  // lineage — mirrors run_mi_agent_query / RiskMonitorResult metadata
  source: {
    engine: string;          // "mi_agent.workflow" | "risk_monitor" | "scenario_engine"
    state?: MIState;         // total_funded | total_pipeline | ...
    spec?: Partial<MIQuerySpec>;
    resolvedFields?: Record<string, ResolvedField>;
    asOf?: string;           // reporting date
    portfolio?: string;
  };
  createdAt: string;
  mock: boolean;             // mock-data disclosure
  warnings?: string[];
  pinned?: boolean;
}
```

Per-type payloads:

- **KPIArtifact** — `kpis: KPI[]` (label, value, delta, trend, intent, hint).
- **ChartArtifact** — `chartType: bar|line|area|scatter|bubble|heatmap|treemap|waterfall`,
  `rows`, `xKey`, `series[]`, `valueFormat` — mirrors `MIChartResult` + the
  factory's chart types.
- **TableArtifact** — `columns[]` (key/label/align/format/bar), `rows` — mirrors
  `MIQueryResult.data`.
- **ValidationArtifact** — `summary {blockers,warnings,passed,coverage}`,
  `issues[]` (code, severity, scope, detail) — mirrors `assert_trusted_canonical`.
- **RiskArtifact** — `mode: concentration|migration|flags|limits`,
  `groups[]` (name, balance, share, status RAG, approaching), optional
  `matrix` (from/to/movementType) — mirrors `RiskMonitorResult`.
- **ScenarioArtifact** — `assumptions{}`, `projection[]` (year, balance, LTV,
  nneg, cumulativeNneg) — mirrors `scenario_engine` output.

**Agent envelope** (mirrors `run_mi_agent_query`):

```ts
interface AgentRequest {
  question: string;
  portfolio: PortfolioContext;
  reporting: ReportingContext;
  options?: { parserMode?: "deterministic" | "llm"; topN?: number };
}

interface AgentResponse {
  ok: boolean;
  question: string;
  intent: Intent;
  interpreted?: string;          // human-readable spec ("Interpreted as:")
  narrative: string;
  assumptions: string[];
  artifacts: Artifact[];
  warnings: string[];
  spec?: Partial<MIQuerySpec>;
  error?: string;
}
```

The UI talks only to an **`AgentClient`** interface; `MockAgentClient` is the
current implementation and a future `HttpAgentClient` posting an `AgentRequest`
to the MI Agent API drops in without touching components. The mock builders
already emit the lineage (`spec`, `state`, `resolvedFields`, `asOf`) the real
backend will populate, so the renderer is backend-shaped from day one.

---

## 9. Implementation note

This document precedes the React refactor in the same change set. The refactor
introduces `domain/` (typed schema + guards), `api/` (`AgentClient` +
`MockAgentClient`), `state/` (portfolio/reporting context + persistence), an
artifact-driven renderer with empty/error/loading states, risk + scenario
artifacts, and a Vitest suite covering intent routing, response structure,
artifact rendering and type guards.
