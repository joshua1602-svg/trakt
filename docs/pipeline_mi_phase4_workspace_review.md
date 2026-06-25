# Pipeline MI — Phase 4 Workspace Refactor Review (Funded / Pipeline / Forecast views)

> Part 0 review before patching. Scope: a single MI workspace with three aligned,
> schema-driven views (Funded / Pipeline / Forecast), a tab-aware MI Agent query
> context, and a composing view-model endpoint. No data-spine change, no
> funded/pipeline row merge, no parser expansion, no new scenario logic.

## Current architecture

| Concern | Where |
| --- | --- |
| Landing page | `AppShell` stacks `FundedSnapshotPanel` + `ForecastBridgeCard` + `PipelineSnapshotPanel` + `PipelineWatchlist` all at once |
| Workspace state | `useWorkspace` fetches `getSnapshot` (funded) + `getForecastSnapshot` (embeds `pipelineSnapshot` + `forecastBridge` + `watchlist`) on the `portfolioId` effect |
| Selector | `HeaderBar` → `PortfolioSelector` + reporting-date `<select>` → `setPortfolio`/`setRun` |
| Funded snapshot API | `GET /mi/snapshot` → `compute_funded_snapshot` (KPI tiles, MoM, datasetContract) |
| Pipeline snapshot API | `GET /mi/pipeline/snapshot` → `compute_pipeline_snapshot` (separated dates, stage/broker/region breakdowns, completion basis) |
| Forecast bridge API | `GET /mi/forecast/snapshot` → `compute_forecast_bridge` (funded + weighted pipeline, disclosure) |
| MI query | `POST /mi/query` → `run_mi_agent_query(question, df, semantics)` over the funded `get_dataframe()`; adapter → artifacts |
| Parser dataset context | None today — the query always runs over the funded df |
| Reusable components | `FundedSnapshotPanel`, `PipelineSnapshotPanel`, `ForecastBridgeCard`, `PipelineWatchlist`, `pipeline/bits` (StatTile/BarList), `ui` (Card/Badge) |
| displayHints / contract | `datasetContract` + `display_hints` already on each snapshot |

## Key insight — the schema already aligns

The pipeline prepared dataset stores its economic amount under the **funded
canonical name** `current_outstanding_balance` (and shares `geographic_region_obligor`,
`ltv_bucket`, `broker_channel`, `current_interest_rate`, …). So "amount by region"
run over the funded df → funded balance by region; over the pipeline df → pipeline
amount by region — **the same query shape, no parser change**. The Forecast view
needs a derived per-row contribution frame (funded balance + weighted pipeline),
which is the deterministic bridge already implemented in
`states.total_forecast_funded` — composed here at the view level, never merged
into the spine.

## Plan

1. **Toggle** — a `ViewToggle` (Funded/Pipeline/Forecast) in the header; `activeView`
   in `useWorkspace` (default Funded). The frontend already holds all three datasets
   (`snapshot` + `forecast`), so switching tabs is client-side; portfolio/run change
   refreshes all.
2. **Aligned views** — `AppShell` renders ONE active view: Funded → `FundedSnapshotPanel`;
   Pipeline → `PipelineSnapshotPanel` + watchlist; Forecast → `ForecastBridgeCard` +
   forecast breakdowns + watchlist. No stacking.
3. **Lineage** — a compact `LineagePanel` per view (source artefact, metric, dates,
   probability basis) from existing metadata.
4. **Tab-aware query** — `POST /mi/query` accepts `datasetContext` (+ `context.activeView`);
   routes the question to the funded / pipeline / forecast-derived df. Explicit
   "funded/pipeline/forecast" wording in the question overrides the tab. Unsupported
   fields fall through to the existing data-aware validation message.
5. **View-model endpoint** — `GET /mi/workspace/view?portfolioId&runId[&view]` composes
   the three existing snapshot/bridge functions + adds forecast-by-dimension
   breakdowns (funded-by-dim + weighted-pipeline-by-dim). Existing endpoints unchanged.

Funded view stays unchanged in substance; pipeline uses the governed SSoT; forecast
is backend-derived.
