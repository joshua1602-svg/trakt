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
returns `{ok, recordType:"pipeline", portfolioId, pipelineAsOfDate,
pipelineExtractDate, pipelineSourceFolderDate, pipelineRowCount, pipelineAmount,
expectedFundedAmount, weightedExpectedFundedAmount, stageBreakdown[],
expectedCompletionBreakdown[], availableMetrics[], availableDimensions[],
missingDimensions[], dataQuality[], fieldCorrelationToFunded, forecastReadiness,
datasetContract}` (see the date-model revision below — no ambiguous `reportingDate`).

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

---

## Revision — pipeline date model + runtime materialisation

### Pipeline dates are weekly-operational, NOT funded reporting dates

Pipeline files are weekly operational extracts; the pipeline is a
continuous/latest-available view, **not** a monthly accounting cut-off. The
date concepts are now kept strictly separate (no field called simply
`reporting_date` ever refers to both):

| Concept | Field | Example |
| --- | --- | --- |
| Selected MI run | `run_id` | `mi_2025_11` |
| Funded book cut-off | `fundedReportingDate` | `2025-11-30` |
| Pipeline source/scope folder | `pipelineSourceFolderDate` | `2025-11-01` |
| Selected weekly file date | `pipelineExtractDate` | `2025-12-01` |
| Operational as-of | `pipelineAsOfDate` | `2025-12-01` |
| Selected file | `sourceFile` | `M2L KFI and Pipeline 2025_12_01_115711.xlsx` |

`discover_pipeline_sources` now groups weekly files **by source folder (scope)**
and selects the **latest weekly extract** per scope. So `mi_2025_11` may
legitimately use the `2025_12_01` weekly file as its operational as-of date — but
that date is never presented as the funded reporting date or the whole-run date.
The funded run id stays `mi_2025_11` (derived from the folder month, never the
weekly extract month). `/mi/pipeline/snapshot` and `/mi/forecast/snapshot`
expose the funded and pipeline dates as distinct fields.

### Runtime materialisation (governed pipeline output under `onboarding_output`)

Two fixes so a clean E2E onboarding/promote produces governed pipeline outputs:

1. `central_tape_builder._build_pipeline_tape` now recognises the M2L KFI key
   columns (KFI / account number) and the M2L KFI field spellings (loan amount,
   application/offer/funds-released dates), so the central pipeline tape is built
   (was `Central pipeline tape: False (0 applications)`).
2. `build_central_tapes` materialises the governed pipeline **source** files
   (the rich weekly extracts that actually contributed pipeline rows — never a
   funded loan file) under `output/pipeline/<source_folder>/`, with a manifest.
   The MI layer discovers these for `/mi/pipeline/snapshot(s)` and
   `/mi/forecast/snapshot`. Pure file ops — no `engine → mi_agent_api` import.

The funded central lender tape schema and funded book behaviour are unchanged;
pipeline rows are never merged into the funded book.
