# Pipeline MI — Phase 2 Architecture Review (landing page + forecast bridge)

> Required review (Part 0) completed **before** implementation. Confirms Phase 2
> extends the current production MI Agent architecture (funded snapshot + Phase 1
> pipeline SSoT) and introduces **no** parallel path, no legacy runtime
> dependency, no NL parser, and no scenario-query logic.

## 1. Existing funded snapshot API + frontend consumption

- `GET /mi/snapshot?portfolioId=<client>/<run>` → `snapshots.compute_funded_snapshot`
  returns `{ok, portfolio, prior, loan_count, current_outstanding_balance, kpis[],
  monthly_change, warnings[], diagnostics[], datasetContract}`.
- `GET /mi/snapshots` → discovery index `{portfolios:[{client_id,label,runs[]}]}`.
- Frontend: `useWorkspace` fetches `getSnapshots()` once (data-driven dropdowns),
  then `getSnapshot(portfolioId)` whenever the selected run changes; renders
  `FundedSnapshotPanel` (KPI tiles + warnings + expandable diagnostics).
- **Phase 2 leaves all of this untouched.**

## 2. Existing pipeline snapshot API (Phase 1)

`GET /mi/pipeline/snapshot(s)` → `pipeline_contract.compute_pipeline_snapshot`
returns `{ok, recordType:"pipeline", portfolioId, reportingDate, pipelineRowCount,
pipelineAmount, expectedFundedAmount, weightedExpectedFundedAmount, stageBreakdown[],
expectedCompletionBreakdown[], availableMetrics[], availableDimensions[],
missingDimensions[], dataQuality[], fieldCorrelationToFunded, forecastReadiness,
datasetContract}`.

## 3. Pipeline field contract + forecast-readiness (Phase 1)

`config/mi/pipeline_field_contract.yaml` → `forecast_readiness`:
`economic_amount_field=expected_funded_amount`,
`baseline_completion_probability_field=completion_probability`,
`expected_completion_date_field=expected_completion_date`, correlation axes, and
the formula `forecast_funded_balance = current_funded_balance +
sum(expected_funded_amount * completion_probability)`. `pipeline_prep` already
computes `weighted_expected_funded_amount` and partitions diagnostics
(blocker/warning/info via `diagnostics_by_severity`).

## 4. React landing page structure + data flow

`AppShell` → `useWorkspace(client)` → renders `FundedSnapshotPanel` above
`ArtifactCanvas`. Client boundary is the `AgentClient` interface
(`getSnapshots`/`getSnapshot`/`ask`), implemented by `HttpAgentClient`
(real `/mi/*`) and `MockAgentClient` (offline). **Phase 2 adds one method
`getForecastSnapshot` to that same interface — no parallel client.**

## 5. Snapshot selector / workspace state flow

`HeaderBar` portfolio/run dropdowns are fed by discovered `portfolios/runs`;
selection sets `selectedClientId`/`selectedRunId` → `portfolioId` →
re-fetch of the funded snapshot. Phase 2 hooks the forecast snapshot fetch onto
the **same** `portfolioId` effect so funded + pipeline + forecast move together.

## 6. Reusable card/chart components

- `FundedSnapshotPanel` KPI-tile pattern (reused visually for pipeline tiles).
- `ui.tsx` `Card`/`Badge`/`IconButton`.
- Charts: Plotly/Recharts exist for the AI artifact canvas, but the landing page
  is intentionally lightweight — pipeline stage/region/channel breakdowns render
  as deterministic horizontal bar lists (divs), avoiding a heavy chart dependency
  on the landing page (mirrors the funded panel's tile-only approach).

## 7. Warning/diagnostic handling

`FundedSnapshotPanel` shows business `warnings` inline and `diagnostics` behind an
expandable "Technical details" toggle. Phase 2 reuses this exact pattern for the
pipeline panel and the watchlist (concise business titles; technical detail
expandable).

## 8. Forecast / state assembler (existing)

`mi_agent/states/assembler.total_forecast_funded` already implements the
row-level forecast (`funded + Σ pipeline contribution`, probabilities from
`config/client/pipeline_expected_funding.yaml`). Phase 2's bridge is the
**aggregate** composition of the funded snapshot + pipeline snapshot using the
**same formula** — it does not merge frames row-level (keeping the SSoT
separation), and references the assembler as the canonical formula source.

## 9. Legacy code inspected (reference only)

- `mi_agent/streamlit_mi_agent.py` and `due_diligence/PIPELINE_REVIEW.md` were
  read for business-logic/visual ideas only. **Neither is imported or invoked at
  runtime by any Phase 2 code.** No Streamlit/legacy module is a runtime
  dependency.

---

## Phase 2 plan (decided)

- **Backend:** new `mi_agent_api/forecast_bridge.py` (deterministic composition +
  watchlist) and `GET /mi/forecast/snapshot?portfolioId=<client>/<run>` that
  composes the funded snapshot + pipeline snapshot + config probabilities into a
  `forecastBridge`, embeds the full `pipelineSnapshot`, and a `watchlist`.
  Funded resolution reuses `_resolve_run_dataframe`; pipeline resolution reuses
  `pipeline_contract.resolve_pipeline_source` (matched to the funded run's
  year-month). Missing pipeline → controlled `forecastReadiness.status="blocked"`
  + "No pipeline data available", never a 500.
- **Frontend:** `getForecastSnapshot` on the client interface; `PipelineSnapshotPanel`,
  `ForecastBridgeCard`, `PipelineWatchlist` rendered in `AppShell` below the funded
  panel; `useWorkspace` fetches the forecast snapshot on the same `portfolioId`
  effect; mock support for offline rendering.
