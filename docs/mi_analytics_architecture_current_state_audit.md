# MI & Analytics Architecture — Current-State Audit

**Status:** Current-state documentation only. No refactoring, no chart migration,
no MI-route build. This document inventories what exists today across the two
analytics artefacts in the repo, identifies what is reusable, and maps the gaps
against the **target MI route** and **target M&A route**.

**Date:** 2026-06-18
**Scope of inspection:**
- `mi_agent/` — the emerging MI Agent architecture
- `analytics/` — the older pipeline / ERM Streamlit app (legacy/prototype)
- `config/system/fields_registry.yaml` — canonical field model that analytics is built on
- `config/client/risk_limits_config.py`, `config/client/pipeline_expected_funding.yaml`, `config/mi/`
- `exception_db.py`, `exception_queue.py`, `ingest_violations.py` — data-quality exceptions
- `due_diligence/` — review/governance docs

---

## 0. Executive summary

There are **two distinct analytics artefacts** in the repo, and they are
architecturally unrelated (the MI Agent deliberately does **not** import
`analytics/`):

| Artefact | What it is | Maturity | Intended status |
|---|---|---|---|
| **MI Agent** (`mi_agent/`) | A governed, deterministic **NL → validated `MIQuerySpec` → executor → chart factory** engine over a *single* canonical CSV. | v1 foundation. Clean, tested, isolated. | **Future architecture** — to be built out into a route-aware, library-driven analytics engine. |
| **Legacy ERM Streamlit app** (`analytics/`) | A ~10.7k-line multi-tab Streamlit dashboard with funded / pipeline / forward / scenario / static-pools / risk tabs. | Feature-rich but monolithic and hard to manage. | **Legacy/prototype.** Mine it for chart logic, bucketing, field groupings and portfolio views. Do **not** refactor or migrate yet. |

**The single most important structural finding:** the MI Agent today is a
**flat, stateless query engine over one dataset**. It has **no concept of
portfolio "states"** (total pipeline / total funded / forecast funded /
cohort-by-portfolio/SPV/acquired-portfolio) and **no route concept**
(MI vs M&A). The legacy app, by contrast, *does* implement most of those
states and stratifications — but as bespoke Streamlit tab code, not as a
reusable library. The target architecture wants the legacy app's *concepts*
expressed through the MI Agent's *governed, library-driven* design.

A second structural finding: several target dimensions are **not present in the
canonical field registry at all** (internal risk stage, IFRS 9 stage, PD,
SPV id, acquired-portfolio id, acquisition date, single/joint borrower flag,
pipeline stage). These are data-model gaps, not just UI gaps — they block both
the MI risk monitor and the M&A route until the canonical model is extended.

---

## 1. What currently exists in the MI Agent (`mi_agent/`)

The MI Agent is an **isolated, additive** v1 foundation. It does not touch the
ESMA / onboarding / validation pipeline and is not wired into `trakt_run.py`.

### 1.1 Components (the deterministic stack)

```
question → parser (deterministic | LLM) → validator → [repair loop] → executor → chart factory → UI
```

| Component | File | Role |
|---|---|---|
| Curated MI semantic registry | `mi_agent/mi_semantics_field_registry.yaml` (generated) | 72 curated fields (46 core / 26 extended / 10 derived) projected from the canonical registry with MI metadata: `role`, `format`, `business_name`, `synonyms`, `allowed_aggregations`, `allowed_chart_roles`, `weight_field`, `bucket_field`. |
| Registry builder | `mi_agent/build_mi_semantics_registry.py` | Generates the semantic layer from `config/system/fields_registry.yaml` via a hand-curated `CURATION` allowlist. |
| Query spec | `mi_agent/mi_query_spec.py` | `MIQuerySpec` dataclass: `intent`, `chart_type`, `metric`, `dimension`, `x/y/size/color`, `aggregation`, `weight_field`, `filters`, `top_n`, `dimensions`, `hierarchy`. Data-free; names semantic keys only. |
| Validator | `mi_agent/mi_query_validator.py` | Validates a spec against the registry + chart-role/aggregation/structural rules. |
| NL parser | `mi_agent/llm_query_parser.py` | Deterministic pattern parser + optional, mockable Claude path. Repair loop sends validation errors (never data) back to the LLM. |
| Executor | `mi_agent/mi_query_executor.py` | Runs a *validated* spec against a canonical CSV / DataFrame. Deterministic, no LLM, no rendering, no mutation. |
| Chart factory | `mi_agent/mi_chart_factory.py` | Turns an `MIQueryResult` into a themed Plotly figure (isolated copy of the ERM look-and-feel). |
| Workflow + UI | `mi_agent/mi_agent_workflow.py`, `mi_agent/streamlit_mi_agent.py` | Headless orchestration (`run_mi_agent_query`) + a thin Streamlit workbench. |
| Config | `mi_agent/mi_agent_config.py` | LLM enable / provider / model / repair attempts (cost control). |

### 1.2 Supported query/chart surface (today)

- **Intents:** `chart`, `table`, `summary`. **Chart types:** `bar`, `line`,
  `scatter`, `bubble`, `heatmap`, `treemap`, `none`.
- **Aggregations:** `sum`, `avg`, `weighted_avg`, `count`, `count_distinct`,
  `median`, `distribution`, `loan_level`, `balance_sum`.
- Top-N + concentration share; weighted averages (balance-weighted);
  loan-level scatter/bubble with privacy caps (no identifiers).
- **Cohort/vintage:** only as a single derived dimension `vintage_year`
  (and `maturity_year`) with a `cohort` chart role — i.e. "by origination
  year" on whatever single CSV is loaded.

### 1.3 Dimensions the MI semantic registry already carries

From `mi_semantics_field_registry.yaml` (business names), the registry already
exposes groupable dimensions / bucketed measures for:

- **Region** (`collateral_geography`, plus NUTS3 obligor/collateral variants)
- **LTV** + **LTV Bucket** (current & original), **Indexed LTV**
- **Borrower Age** + **Age Bucket** (youngest borrower)
- **Interest Rate** + **Rate Type** (interest-rate type), **Interest Margin**
- **Product Type** + **Sub Product Type**
- **Broker**, **Originator**, **Originator Affiliate**, **Origination Channel**
- **Account Status**, **In Arrears**, **Arrears Bucket**, **Arrears Balance**,
  **Days in Arrears** (total / interest / principal)
- **Vintage** (origination-year cohort), **Maturity Year**, **Term Bucket**
- **Ticket Size**, **Balance**, **Principal Balance**, **Valuation**
- **Occupancy / Tenure / Lien Position / Number of Properties / Bedrooms**
- **Default Amount / Default Date / Losses / Recoveries / Redemptions /
  Prepayments / Equity / Protected Equity / NEG (no-negative-equity guarantee)**

### 1.4 What the MI Agent explicitly does **not** have yet

- **No portfolio "state" concept.** No total-pipeline, total-funded,
  forecast-funded, or cohort-by-portfolio/SPV/acquired-portfolio states. It
  reads one canonical CSV and queries it flat.
- **No route concept.** No MI vs M&A routing; no notion that M&A should be
  funded-only.
- **No risk monitor.** No concentration limits, early-warning flags, migration,
  or period-over-period comparison.
- **No pipeline / forecast inputs.** It has no access to pipeline snapshots or
  expected-funding probabilities.
- **No bucketing engine.** It *reuses* a bucket column if already present in the
  dataframe, otherwise falls back to the raw field with a warning. Buckets are
  produced upstream by `analytics/mi_prep.py` (which the MI Agent does not import).
- **No KPI/summary cards** in the chart factory (`intent=summary` is table-only).
- **No PPTX export** (HTML / JSON / PNG-via-kaleido only).
- **Semantics are generic, not asset/regime-aware.** `config/mi/mi_equity_release_uk_applicability.yaml`
  is a separate config overlay that notes the registry "is GENERIC and not yet
  asset/regime-aware".
- Registry omits some canonical dimensions that *do* exist canonically — most
  notably **`amortisation_type`** (present in `fields_registry.yaml`, absent
  from the curated MI registry).

---

## 2. What currently exists in the old Streamlit app (`analytics/`)

A ~10,700-line, multi-tab dashboard. Main entry: `analytics/streamlit_app_erm.py`
(3,010 lines). Feature flags gate optional tabs (`SCENARIO_ENGINE_AVAILABLE`,
`RISK_MONITORING_AVAILABLE`, `BLOB_STORAGE_AVAILABLE`, `PIPELINE_TAB_AVAILABLE`).

### 2.1 Portfolio "states" / views implemented today

| State / view | Where | Notes |
|---|---|---|
| **Total funded portfolio** | Funded Exposures tab (`streamlit_app_erm.py` ~1090–1815) | KPI strip + full stratification set + concentration. |
| **Total pipeline** | Pipeline tab (`tab_pipeline.py::render_pipeline_tab`, ~525) | Stages KFI → APPLICATION → OFFER → COMPLETED; funnel, weekly flow. |
| **Forecast funded (funded + expected pipeline conversion)** | Forward Exposure tab (`tab_pipeline.py::render_forward_exposure_tab`, ~660) | Combined funded book + expected pipeline; forward WA LTV/rate/age; forward regional concentration. |
| **Completions reconciled to funded** | Pipeline tab (~646) | Match rate / pending / unmatched. |
| **Cohort / vintage by origination date** | Static Pools tab (~2415) + `static_pools_core.py` | Vintage curves by months-on-book (MOB); month/quarter/year granularity. |
| **Cohort / vintage by segment** | Static Pools tab | Segmentable by geography, product, risk bucket (LTV-derived). |
| **Scenario projections** | Scenario tab (~1821) + `scenario_engine.py` | 25-yr runoff, NNEG losses, LTV drift; 6 presets; sensitivity. |
| **Risk monitoring (funded & forward lenses)** | Risk tab (~2785) + `risk_monitor.py` | Limit breach, utilisation gauges, concentration. |

> **Not implemented as states:** cohort/vintage **by portfolio**, **by SPV**, or
> **by acquired portfolio** (no such identifiers in the data model — see §6).

### 2.2 Stratifications implemented today

Produced primarily by `mi_prep.add_buckets` (`analytics/mi_prep.py` ~163–213) and
rendered by `charts_plotly.strat_bar_chart_pure` / inline `px`/`go` calls.

| Target dimension | Legacy support | Field / bucket | Location |
|---|---|---|---|
| Balance & count by **LTV bucket** | ✅ | `current_loan_to_value` → `ltv_bucket` (0-20,…,80%+) | `mi_prep.py`; bars ~1252 |
| Balance & count by **borrower age / youngest** | ✅ | `youngest_borrower_age` → `age_bucket` (<55,…,85+) | `mi_prep.py`; bars ~1358 |
| Balance & count by **geographic region** | ✅ | `geographic_region*` → treemap | ~1308; `portfolio_semantics.py` |
| Balance & count by **interest rate bucket** | ✅ | `current_interest_rate` → `rate_bucket` (<2%,…,8%+) | `mi_prep.py`; bars ~1275 |
| Balance & count by **product type** | ✅ | `erm_product_type` (direct groupby) | bars ~1288 |
| Balance & count by **amortisation type** | ❌ | — | not built |
| Balance & count by **origination channel** | ⚠️ partial | broker channel built; origination channel not a dedicated strat | ~1569 |
| Balance & count by **broker channel** | ✅ | `broker_channel` → top-N treemap | ~1574; `tab_pipeline.py` ~621 |
| Balance & count by **internal risk stage / IFRS 9 stage** | ❌ | no such field | — |
| Balance & count by **PD bucket** | ❌ | no PD field | — |
| Balance & count by **interest rate type** | ⚠️ partial | variable-rate concentration in risk monitor; not a strat chart | `risk_monitor.py` ~232 |
| Balance & count by **arrears status** | ⚠️ partial | account status used in static-pools migration, not a funded strat chart | `static_pools_core.py` |
| Balance & count by **pipeline stage** (KFI/app/offer/completion/funded) | ✅ | `stage` (canonicalised in `pipeline_prep.py`) | funnel `tab_pipeline.py` ~603 |
| Balance & count by **single vs joint borrower** | ❌ | no borrower-count field | — |
| Balance & count by **time on book** | ⚠️ partial | MOB used for vintage curves, not a funded strat | `static_pools_core.py` ~311 |
| Ticket size band | ✅ (extra) | `total_balance` → `ticket_bucket` | `mi_prep.py` |
| Original-LTV bucket | ✅ (extra) | `original_loan_to_value` → `original_ltv_bucket` | `mi_prep.py` |

### 2.3 Chart logic, theme and bucketing (the reusable craft)

- **Chart factory pattern:** `analytics/charts_plotly.py`
  (`apply_chart_theme`, `strat_bar_chart_pure`) — a *pure* (no-Streamlit) factory
  returning `(fig, msg, level)`. This is the cleanest reusable seam.
- **Chart vocabulary in use:** themed bars, top-N treemaps (light-grey→navy
  scale), relationship bubble/scatter (balance vs property value; LTV vs age
  sized by balance), `go.Heatmap` concentration matrix (user-selectable
  dimensions), `px.area`/`px.bar` vintage charts, `go.Funnel` pipeline funnel,
  `go.Indicator` risk utilisation gauges.
- **Theme:** navy `#232D55` / blue `#919DD1` / grey `#BFBFBF`, Calibri, white
  background, horizontal legend, currency formatting (£1.2m / £450k). The MI
  chart factory already keeps an **isolated copy** of this look-and-feel.
- **Bucketing engine:** `mi_prep.add_buckets` (LTV, original-LTV, rate, age,
  ticket) + `pipeline_tab_helpers.add_pipeline_stratification_buckets` (same
  edges for pipeline snapshots). Scale-normalisation logic (decimal vs percent)
  is handled here.
- **Cohort/vintage engine:** `analytics/static_pools_core.py` (575 lines) — a
  **product-agnostic** core: `StaticPoolsSpec` column contract,
  `build_static_pools_panel`, `build_vintage_metric_series` (MOB curves),
  `build_status_migration_sankey` (explicit or derived snapshot-to-snapshot
  transitions), `build_portfolio_runoff_timeseries`.
- **Semantic helpers:** `analytics/portfolio_semantics.py` — UK region
  normalisation to ONS labels/codes, `safe_ltv_percent`.
- **Exports:** CSV / Excel / PowerPoint (`generate_pptx_client.py`, 3,230 lines,
  matplotlib-rendered deck via subprocess) / scenario CSV+JSON.

### 2.4 Risk & forecast engines (legacy)

- **Risk monitor** (`analytics/risk_monitor.py`, 426 lines): 18 configurable
  concentration limits (`config/client/risk_limits_config.py`) over region,
  single/multi-loan borrower, property-value bands, age >85, variable-rate %;
  `check_all_limits()` with green/amber/red `determine_status()`. Runs on
  **funded** and **forward** lenses.
- **Expected funding** (`analytics/pipeline_expected_funding.py`): stage-based
  conversion probabilities (KFI 20% / APP 45% / OFFER 75% / COMPLETED 100%) with
  per-broker / per-product adjustments, clipped; `expected_funded_amount =
  pipeline_amount × final_probability`; high-confidence flag at ≥70%.
- **Forward risk** (`analytics/pipeline_forward_risk.py`): funded + expected
  combined **by region only**.
- **Reconciliation / persistence** (`pipeline_reconciliation.py`,
  `pipeline_persistence.py`): match pipeline to funded book; write combined
  forward-exposure artefact.
- **Scenario engine** (`scenario_engine.py`, 670 lines): survival/exit hazard
  model (mortality age-scaled, prepay, move-to-care), NNEG, LTV drift, 6 presets,
  sensitivity analysis.

---

## 3. What can be reused from the Streamlit app

These are **concepts and self-contained logic** worth lifting into the MI Agent
library later — **not** to be migrated now.

| Reusable asset | Source | Why it's reusable | Reuse note |
|---|---|---|---|
| **Bucket definitions & scale-normalisation** | `mi_prep.add_buckets`, `pipeline_tab_helpers.add_pipeline_stratification_buckets` | Self-contained; edges are exactly the buckets the MI registry references via `bucket_field`. | Becomes the MI Agent's missing **bucketing engine** (materialise `ltv_bucket`, `age_bucket`, `rate_bucket`, `ticket_bucket`, `vintage_year`). |
| **Pure chart factory pattern** | `charts_plotly.strat_bar_chart_pure`, `apply_chart_theme` | No Streamlit coupling; returns figure + message + level. | Pattern already mirrored in `mi_chart_factory`; reuse the *formatting/aggregation* rules, not the file. |
| **Cohort/vintage core** | `static_pools_core.py` | Product-agnostic, spec-driven, already separated from UI. | Strongest reuse candidate — basis for MI cohort/vintage "states" and migration views. |
| **Region semantics** | `portfolio_semantics.py` | Deterministic UK region normalisation + safe LTV. | Lift into MI dimension resolution. |
| **Concentration-matrix heatmap & treemap logic** | inline in `streamlit_app_erm.py` / `tab_pipeline.py` | Encodes useful 2-D concentration views. | Reuse the *grouping/pivot* logic; the MI chart factory already renders heatmap/treemap. |
| **Expected-funding probability model** | `pipeline_expected_funding.py` | Self-contained, config-driven. | Basis for MI "forecast funded" state. |
| **Risk limit catalogue & status thresholds** | `risk_limits_config.py`, `risk_monitor.determine_status` | Encodes real client limits + amber/red logic. | Basis for the MI risk monitor's limit-usage view. |
| **Status-migration (Sankey) builder** | `static_pools_core.build_status_migration_sankey` | Already supports derived snapshot-to-snapshot transitions. | Seed for risk-grade / PD / stage migration once those fields exist. |
| **Field groupings / KPI definitions / portfolio views** | `streamlit_app_erm.py` KPI strips, tab layouts | Curated analyst-facing groupings and weighted-avg KPI formulas. | Reuse as the MI library's default "views" per state. |
| **PPTX deck structure** | `generate_pptx_client.py` | Slide taxonomy (cover, strat, scenario, risk, vintage). | Reuse layout concept for a future MI export path. |

---

## 4. What should be rebuilt into the MI Agent library

These should be **rebuilt natively** in the MI Agent's governed, library-driven
style (validated specs, data-free LLM, isolated chart factory) rather than ported
verbatim from the Streamlit tabs:

1. **A portfolio-"state" layer.** A first-class library of states the agent can
   target: `total_pipeline`, `total_funded`, `total_forecast_funded`
   (funded + expected conversion), and cohort/vintage states keyed by
   `origination_date`, `portfolio`, `SPV`, and `acquired_portfolio`. Today the
   executor only reads one flat CSV; states require a data-source/assembly layer
   above it.
2. **A bucketing engine** inside the agent (the README lists this as a v2 item):
   materialise `bucket_field` hints as real groupable dimensions so
   heatmaps/treemaps can group by banded measures without depending on
   `analytics/`.
3. **A stratification library** — a declarative catalogue of the §2.2 dimensions
   (balance & count) that any state can emit, reusing the bucket engine.
4. **A risk-monitor module** as a route capability (see §5), expressed as
   governed specs + a limits/early-warning config, not bespoke Streamlit.
5. **Route awareness** — an MI route vs an M&A route, with the M&A route
   constrained to funded-only analytics by default (see §7).
6. **Forecast assembly** — bring the expected-funding probability model in as a
   state input so "forecast funded" and "current vs forecast" comparisons are
   first-class.
7. **Asset/regime-aware semantics** — fold `config/mi/...applicability.yaml`
   logic into the registry so dimensions resolve per asset class (e.g. youngest
   borrower for equity release).
8. **KPI/summary cards** in the chart/render layer (currently out of scope in v1).

---

## 5. What is missing for the target MI route

The target MI route = route-aware, library-driven engine supporting the **state
library**, **stratifications**, and an **early-warning risk monitor**.

### 5.1 States

| Target state | Status | Gap |
|---|---|---|
| Total pipeline | ❌ in MI Agent (✅ legacy) | MI Agent has no pipeline ingestion or stage model. |
| Total funded | ⚠️ MI Agent queries a funded CSV flat | No explicit "funded state" abstraction; works only if the loaded CSV *is* funded. |
| Total forecast funded (funded + expected conversion) | ❌ in MI Agent (✅ legacy) | No expected-funding model wired into the agent. |
| Cohort/vintage by **date** | ⚠️ partial | Only `vintage_year` derived dimension; no MOB curves / runoff in the agent. |
| Cohort/vintage by **portfolio** | ❌ everywhere | **No portfolio identifier in the canonical model.** |
| Cohort/vintage by **SPV** | ❌ everywhere | **No SPV identifier in the canonical model.** |
| Cohort/vintage by **acquired portfolio** | ❌ everywhere | **No acquired-portfolio identifier / acquisition date.** |

### 5.2 Stratifications (balance & count)

Already resolvable via the MI registry: **LTV bucket, borrower age, region,
interest-rate bucket, product type, origination channel, broker channel,
interest-rate type, arrears status, time-on-book (via term/MOB), vintage**.

Missing / blocked:

| Stratification | Blocker |
|---|---|
| **Amortisation type** | Canonical field exists (`amortisation_type`) but is **not in the curated MI registry** — registry-curation gap (cheap fix). |
| **Internal risk stage** | **No canonical field** (only SME-only `bank_internal_rating`). |
| **IFRS 9 stage** | **No canonical field.** |
| **PD bucket** | **No loan-level PD field** (only a corporate-guarantor PD, format undefined). |
| **Pipeline stage** | Defined only in legacy `pipeline_prep.py`; not a canonical field and not in the MI Agent. |
| **Single vs joint borrower** | **No `number_of_borrowers` field** (derivable from `borrower_1/2_*` presence). |
| **Time on book (seasoning)** | No dedicated canonical field; derivable from `origination_date` vs reporting date (legacy computes MOB). |

### 5.3 Risk monitor (early-warning capability)

Target = early-warning, not a static dashboard. Current state vs target:

| Capability | Current state | Gap |
|---|---|---|
| Funded concentration monitoring | ✅ legacy (`risk_monitor.py`) | Not in MI Agent. |
| Forecasted concentration monitoring | ⚠️ legacy, **region only** | No borrower/age/LTV/value forecast concentration; nothing in MI Agent. |
| Current vs forecast composition comparison | ⚠️ legacy aggregates region funded+expected; **no deltas** | No composition-shift / delta engine anywhere. |
| Concentration limit usage | ✅ legacy (18 limits, amber/red) | Not in MI Agent; limits are ERM-specific. |
| Early-warning flags | ⚠️ minimal (limit colours, high-confidence flag) | No trajectory/approaching-limit/trend warnings. |
| Risk-grade migration | ❌ | No risk grade field; no migration engine (Sankey builder exists but unfed). |
| PD migration | ❌ | No PD field; no migration. |
| Deterioration/improvement period-over-period | ❌ | Snapshots can be loaded (`prepare_weekly_trend_dataset`) but **no automated period deltas**. |
| Portfolio / SPV / acquired-portfolio / cohort risk views | ⚠️ portfolio-level only; cohort in static pools (no limits) | SPV / acquired-portfolio segmentation absent (data-model gap). |

---

## 6. What is missing for the target M&A route

Target M&A route = **funded-portfolio analysis only** (no pipeline / forecast
unless explicitly requested), with funded stratifications, cohort/vintage by
origination & acquisition date & risk grade & PD bucket & balance band,
portfolio / acquired-portfolio segmentation, diligence data-quality exceptions,
and buyer-relevant concentration.

| M&A requirement | Current state | Gap |
|---|---|---|
| Route exists at all | ❌ | No M&A route / agent. A `MA_Buyside` *consumer* is **declared** in `fields_registry.yaml` (lines 14–20, "Due Diligence Super-Set for Buyers") but has **no analytics backend**. |
| Funded stratifications | ⚠️ available via MI registry / legacy strat charts | Same registry gaps as §5.2 (amortisation type, risk stage, PD). |
| Cohort/vintage by **origination date** | ⚠️ legacy (`static_pools_core`) | Not in an M&A surface; reusable. |
| Cohort/vintage by **acquisition date** | ❌ | **No `acquisition_date` field.** |
| Cohort/vintage by **risk grade** | ❌ | **No core risk-grade field.** |
| Cohort/vintage by **PD bucket** | ❌ | **No loan-level PD field.** |
| Cohort/vintage by **balance band** | ✅ logic exists | `ticket_bucket` in `mi_prep` / `ticket_bucket` in MI registry — reusable. |
| **Portfolio / acquired-portfolio segmentation** | ❌ | **No portfolio / acquired-portfolio identifiers.** |
| **Data-quality exceptions for diligence** | ✅ engine exists, ⚠️ not M&A-wired | `exception_db.py` / `exception_queue.py` / `ingest_violations.py`: SQLite snapshots, findings (BLOCKING/REVIEW/INFO materiality), hash-chained remediation ledger, triage UI. Strong reuse; **not surfaced as a buyer diligence scorecard** and no cross-portfolio comparison. |
| Concentration analysis for a buyer | ⚠️ legacy concentration logic exists | Funded concentration matrix/treemaps reusable; not packaged for M&A; no SPV/acquired segmentation. |

---

## 7. Cross-cutting canonical data-model gaps

Several target capabilities are blocked at the **data model**, not the UI. From
`config/system/fields_registry.yaml` (439 fields):

**Present & usable:** `current_loan_to_value`, `youngest_borrower_age`,
`geographic_region_*`, `current_interest_rate`, `interest_rate_type`,
`product_type`, **`amortisation_type`** (exists canonically; just not curated
into the MI registry), `origination_channel`, `broker_channel`, `account_status`,
`arrears_balance`, `number_of_days_in_arrears`, `origination_date`,
`current_principal_balance`, `default_date`, `credit_impaired_obligor`,
`maturity_date`, `collateral_type`, `employment_status`.

**Missing (block MI risk + M&A):**

| Concept | Status | Impact |
|---|---|---|
| Internal risk stage | MISSING (SME-only `bank_internal_rating`) | Blocks risk-stage strat + migration. |
| IFRS 9 stage | MISSING | Blocks staging strat + migration. |
| Loan-level PD | MISSING (only corporate-guarantor PD, undefined format) | Blocks PD bucket + PD migration. |
| Risk grade (core) | MISSING | Blocks M&A risk-grade cohorts + grade migration. |
| Pipeline stage | Not canonical (legacy `pipeline_prep` only) | Pipeline-stage strat lives outside the canonical model. |
| Single/joint borrower (`number_of_borrowers`) | MISSING (derivable from `borrower_1/2_*`) | Blocks single-vs-joint strat. |
| Acquisition date | MISSING | Blocks M&A acquisition-date cohorts. |
| Acquired-portfolio identifier | MISSING | Blocks acquired-portfolio segmentation (MI & M&A). |
| Portfolio identifier | MISSING | Blocks portfolio-level cohort/risk views. |
| SPV identifier | MISSING | Blocks SPV-level cohort/risk views. |

---

## 8. Capability map (one-screen summary)

Legend: ✅ exists · ⚠️ partial · ❌ missing · 🔁 reusable from legacy · 🧱 data-model gap

| Capability | MI Agent | Legacy app | Reusable? | Target MI | Target M&A |
|---|---|---|---|---|---|
| NL→governed query→validate→chart | ✅ | ❌ | — | core | core |
| Single-CSV flat querying | ✅ | n/a | — | — | — |
| Portfolio "state" layer | ❌ | ⚠️ (per-tab) | 🔁 concepts | needed | funded only |
| Total pipeline | ❌ | ✅ | 🔁 | needed | not required |
| Total funded | ⚠️ | ✅ | 🔁 | needed | **required** |
| Forecast funded | ❌ | ✅ | 🔁 (`pipeline_expected_funding`) | needed | only on request |
| Cohort/vintage by date | ⚠️ (`vintage_year`) | ✅ (`static_pools_core`) | 🔁 strong | needed | **required** |
| Cohort by portfolio / SPV / acquired | ❌ | ❌ | — | 🧱 | 🧱 |
| Strat: LTV / age / region / rate / product / channel / broker / rate-type / arrears | ⚠️ registry-ready | ✅ (most) | 🔁 buckets | needed | needed |
| Strat: amortisation type | ❌ (registry) | ❌ | — | curation fix | curation fix |
| Strat: risk stage / IFRS9 / PD / single-joint / pipeline-stage / time-on-book | ❌ | ⚠️ (some) | partial | 🧱 mostly | 🧱 mostly |
| Bucketing engine | ❌ (reuses cols) | ✅ (`mi_prep`) | 🔁 strong | needed | needed |
| Chart factory + theme | ✅ (isolated) | ✅ (`charts_plotly`) | 🔁 rules | core | core |
| Risk monitor: funded concentration | ❌ | ✅ | 🔁 | needed | concentration only |
| Risk monitor: forecast concentration | ❌ | ⚠️ region only | 🔁 | needed | not required |
| Current vs forecast composition | ❌ | ⚠️ no deltas | partial | needed | not required |
| Concentration limit usage | ❌ | ✅ (18 limits) | 🔁 | needed | buyer view |
| Early-warning flags | ❌ | ⚠️ minimal | partial | needed | — |
| Risk-grade / PD migration | ❌ | ❌ (Sankey builder unfed) | 🔁 builder | 🧱 | 🧱 |
| Period-over-period deterioration | ❌ | ⚠️ snapshots, no deltas | partial | needed | — |
| Scenario projections (NNEG/LTV drift) | ❌ | ✅ (`scenario_engine`) | 🔁 | optional | not required |
| Data-quality exceptions | ❌ | ✅ engine (root) | 🔁 strong | optional | **required (diligence)** |
| PPTX / Excel export | ❌ (HTML/JSON/PNG) | ✅ | 🔁 layout | optional | optional |
| Route awareness (MI vs M&A) | ❌ | ❌ | — | needed | needed |

---

## 9. Headline conclusions

1. **The MI Agent is the right foundation but is currently flat and stateless.**
   It has a clean governed pipeline and a 72-field semantic registry covering
   most strat dimensions, but no states, no routes, no risk monitor, no bucketing
   engine, and no pipeline/forecast inputs.
2. **The legacy Streamlit app already implements most target states and
   stratifications** — funded, pipeline, forecast-funded, vintage/MOB cohorts,
   concentration, risk limits, scenarios — but as monolithic tab code. Its
   **chart logic, bucket definitions, cohort core (`static_pools_core`),
   expected-funding model, and risk-limit catalogue are the highest-value
   reusable assets.**
3. **Three target capabilities are blocked by the canonical data model, not the
   UI:** risk-stage / IFRS 9 / PD (→ risk strat + migration), and
   portfolio / SPV / acquired-portfolio / acquisition-date identifiers
   (→ segmentation and the M&A route). These need field-registry extensions
   before MI-risk and M&A routes can be fully built.
4. **The M&A route does not exist yet** beyond a declared `MA_Buyside` consumer.
   Its funded-only stratifications, balance-band/vintage cohorts, and the
   existing **data-quality exception engine** are reusable; acquisition/portfolio
   metadata and risk-grade/PD fields are the blockers.
5. **Cheap early win:** add `amortisation_type` (and other already-canonical but
   uncurated fields) to the MI semantic registry curation, and materialise the
   `bucket_field` hints — both are registry/curation changes, not data-model work.

---

*This document is descriptive only. Per the brief: the Streamlit app was not
refactored, no charts were migrated, and the MI route was not built.*
