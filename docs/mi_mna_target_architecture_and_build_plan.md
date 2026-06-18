# MI & M&A — Target Architecture and Build Sequencing Plan

**Status:** Design documentation only. No source code is modified, no charts are
migrated, no agents or snapshot layer are built. This document proposes the
target architecture, route contracts, config/file structure, and a build
sequence for the future **MI Agent** and **M&A Agent**, building on the
current-state audit in
[`docs/mi_analytics_architecture_current_state_audit.md`](./mi_analytics_architecture_current_state_audit.md).

**Date:** 2026-06-18

**Product decisions taken as fixed inputs:**
- The current **MI Agent (`mi_agent/`) is the target architecture direction.**
- The legacy Streamlit app (`analytics/`) is **not** the future architecture; it
  is a **source of reusable logic** (charts, buckets, static-pool cohorts,
  risk-limit ideas, forecast logic, data-quality exceptions).
- **Regulatory Annex 2 is a settled, point-in-time contract** and must **not**
  drive MI/M&A design.
- Do not refactor code, migrate charts, build the agents, or build the snapshot
  layer in this task.

---

## 1. Executive summary

The current MI Agent is the right foundation — a governed, deterministic
`question → MIQuerySpec → validate → execute → chart` pipeline with a curated
semantic registry — but it is **flat and stateless**: it reads one canonical CSV
and queries it point-in-time. The target capabilities (portfolio *states*,
forecast-funded, cohort/vintage across periods, risk migration, early-warning
concentration movement) are **inherently time-series**. The MI Agent therefore
needs **three new layers** added *beneath* its existing query engine, without
disturbing the parts that already work:

1. **A snapshot & history layer** — a persistent, append-only store of
   timestamped portfolio snapshots keyed by `(client, snapshot, loan)` with
   explicit, separated dates (upload / reporting / cut-off / origination /
   funding / acquisition / SPV-transfer). This is the single biggest gap.
2. **A portfolio-state library** — `total_pipeline`, `total_funded`,
   `total_forecast_funded`, and cohort/vintage states by date / portfolio / SPV
   / acquired-portfolio, each assembled *from* the snapshot layer.
3. **A route layer** — explicit **MI**, **M&A**, and (already existing)
   **Regulatory** routes with declared contracts. MI is recurring/time-series;
   M&A and Regulatory are point-in-time.

The crucial framing this document introduces is the **point-in-time vs
recurring** split (§2). Regulatory Annex 2 and M&A diligence are *point-in-time*
(one cut-off, one tape); MI, the risk monitor and forecasting are *recurring*
(weekly/monthly snapshots, movement, migration). Conflating these is the root
cause of the "flat/stateless" problem.

On the legacy Azure orchestration: it contains **excellent reusable design
patterns** (deterministic synthetic keys for cross-snapshot tracking, an
etag-idempotent snapshot pointer/manifest, mode-from-path routing) but is
**tightly coupled to the Azure SDK with zero abstraction** and **conflates
upload date with reporting date**. The recommendation is **cloud-agnostic-first**:
model snapshots and history in storage-neutral business logic, with Azure Blob
as one *deployment adapter* behind a `SnapshotStore` interface — not hard-coded
into the analytics, as it is today.

The agents themselves remain thin: the MI Agent gains *states*, *time-series
comparison*, and *route awareness*; the M&A Agent is a **narrower, funded-only**
sibling that reuses the shared library but excludes pipeline/forecast and
recurring-history machinery by contract.

---

## 2. Route taxonomy: point-in-time vs recurring/time-series

This separation is the organising principle of the whole design.

### 2.1 Point-in-time routes (one cut-off, one package)

| Route | Input | Output | Time model |
|---|---|---|---|
| **Regulatory Annex 2** (exists) | One tape at one cut-off date | One regulatory package + XML delivery | Single `cut_off_date`. No history required. **Settled contract — out of scope for redesign.** |
| **M&A / due diligence** (new) | One funded portfolio tape (+ optional historical performance) | Buyer-facing stratifications, cohorts, concentration, data-quality scorecard | Single `as_of_date`. Optional prior periods if the seller supplies them, but **not a recurring feed**. |

Point-in-time routes need **no persistent snapshot history**. They analyse the
tape in front of them. They may *read* the snapshot store if a history exists,
but they do not depend on one.

### 2.2 Recurring / time-series routes (many snapshots over time)

| Route | Input cadence | What it compares |
|---|---|---|
| **MI** (new) | Weekly and/or monthly client uploads, plus backfilled historical uploads | Funded-book evolution, pipeline movement, forecast-funded movement, cohort/vintage across periods, SPV & acquired-portfolio trends |
| **Risk monitor** (new) | Same recurring snapshots | Funded vs forecast concentration, concentration movement period-over-period, limit usage, risk-grade & PD migration, deterioration/improvement flags |
| **Funding / forecasting** (new) | Recurring pipeline + funded snapshots | Expected conversion, forecast-funded book, pipeline→funded movement |

Recurring routes **require** the snapshot & history layer (§6). Their core
operations are **snapshot-to-snapshot comparison** and **cohort tracking across
periods** — neither of which the current flat MI Agent can express.

### 2.3 Why this matters for design

- The **MI Agent must be stateful** (snapshot-aware); the **M&A and Regulatory
  agents must remain stateless** (single-tape). A shared library serves all
  three, but only MI/risk/forecasting depend on the history layer.
- A query in a recurring route always carries a **temporal selector** (a single
  `snapshot_id`, or a pair for comparison, or a range for trend). A query in a
  point-in-time route carries none.

---

## 3. Target MI Agent architecture

### 3.1 Is the current MI Agent too flat/stateless? — Yes, by design, and that must change

The v1 MI Agent (`mi_agent/mi_query_executor.py`) takes a single DataFrame/CSV
and queries it. It has **no snapshot identity, no period selection, no
cross-period comparison, and no state assembly**. The only nod to time is a
derived `vintage_year` dimension on whatever single file is loaded. For the
target MI route this is insufficient: weekly/monthly uploads, funded-book
movement, forecast movement, and migration are all *relationships between
snapshots*, which a single-DataFrame engine cannot represent.

**Conclusion:** keep the v1 query/validate/chart core unchanged, and add three
layers beneath/around it: a **snapshot store**, a **state assembler**, and a
**route/temporal contract**. The `MIQuerySpec` gains a temporal selector and a
state selector; everything downstream of the validator stays as-is.

### 3.2 Target layering

```
                ┌─────────────────────────────────────────────────────────┐
                │  Route layer (MI | M&A | Regulatory)                     │
                │  - resolves a RouteContract: allowed states/dims/temporal │
                └───────────────┬─────────────────────────────────────────┘
                                │
         ┌──────────────────────┴───────────────────────┐
         │  MI Agent orchestration (run_mi_agent_query++)│
         │  question → parser → MIQuerySpec(+temporal,    │
         │  +state) → validator → state assembler →       │
         │  executor → chart factory → UI/export          │
         └───────┬───────────────────────┬───────────────┘
                 │                        │
   ┌─────────────▼─────────┐   ┌──────────▼───────────────┐
   │ State assembler       │   │ Shared analytics library │
   │ (total_pipeline,      │   │ (buckets, stratify,      │
   │  total_funded,        │   │  cohort/vintage, concen- │
   │  forecast_funded,     │   │  tration, forecast,      │
   │  cohort states)       │   │  migration, charts)      │
   └─────────────┬─────────┘   └──────────────────────────┘
                 │
   ┌─────────────▼──────────────────────────────────────┐
   │ Snapshot & history layer (SnapshotStore interface)  │
   │  resolve snapshot(s) → loan-level frame(s) with      │
   │  separated dates + portfolio/SPV/acquired/risk keys  │
   └─────────────┬──────────────────────────────────────┘
                 │
   ┌─────────────▼───────────┐
   │ Storage adapter         │  local-fs (default) | Azure Blob | S3/GCS
   └─────────────────────────┘
```

### 3.3 Portfolio-state library (the MI route's spine)

Each **state** is a deterministic function `(SnapshotSelector, filters) → frame`
that the executor then stratifies. Proposed states:

| State | Definition | Snapshot inputs |
|---|---|---|
| `total_pipeline` | All in-pipeline loans by stage at a snapshot | latest/selected pipeline snapshot |
| `total_funded` | Funded book at a snapshot | latest/selected funded snapshot |
| `total_forecast_funded` | `total_funded` + Σ(pipeline balance × expected conversion probability) | funded + pipeline snapshot + expected-funding config |
| `cohort_by_date` | Funded book grouped by origination/vintage cohort, optionally tracked over snapshots (MOB) | one or many snapshots |
| `cohort_by_portfolio` | As above, segmented by `portfolio_id` | snapshot(s) with portfolio key |
| `cohort_by_spv` | Segmented by `spv_id` | snapshot(s) with SPV key |
| `cohort_by_acquired_portfolio` | Segmented by `acquired_portfolio_id` | snapshot(s) with acquired-portfolio key |

States are **route-gated**: MI may use all; M&A may use only funded + cohort
states (§4); Regulatory uses none of these.

### 3.4 Stratification library (balance & count by dimension)

A declarative catalogue mapping each target dimension to a semantic field and a
bucket rule (reusing legacy bucket *definitions* — see §5). Dimensions:

LTV bucket · borrower age / youngest borrower · geographic region · interest
rate bucket · product type · **amortisation type** · origination channel ·
broker channel · **internal risk stage / IFRS 9 stage** · **PD bucket** ·
interest rate type · arrears status · **pipeline stage** · **single vs joint
borrower** · **time on book**.

Bolded dimensions are blocked by canonical/registry gaps (§9). Each entry
declares: semantic field key, bucket definition (or "categorical"), which states
it applies to, and which asset classes (e.g. *youngest borrower* for equity
release).

### 3.5 MI risk monitor (early-warning)

Built as a route capability over the snapshot store, expressed as governed specs
plus a limits/early-warning config (§10). Capabilities:

| Capability | Mechanism |
|---|---|
| Funded concentrations | stratify `total_funded`, compute group shares |
| Forecasted concentrations | stratify `total_forecast_funded` |
| Current vs forecast movement | delta of (forecast share − funded share) per group |
| Concentration limit usage | compare shares to limit config; amber/red thresholds |
| Migration between risk grades | join two snapshots on `loan_id`, transition matrix on `risk_grade` |
| Migration between PD buckets | same join, transition matrix on `pd_bucket` |
| Deterioration / improvement flags | per-loan grade/PD/arrears delta between snapshots → flag |
| Concentration movement (early warning) | trajectory of group share across ≥3 snapshots → approaching-limit warning |

Migration/movement all reduce to **two-snapshot or N-snapshot joins on a stable
`loan_id`** — which is exactly what the snapshot layer must guarantee.

### 3.6 `MIQuerySpec` extensions (additive, non-breaking)

Add optional fields only; existing specs remain valid:

- `state: Optional[str]` — one of the state-library keys (default: treat the
  loaded frame as `total_funded`, preserving v1 behaviour).
- `temporal: Optional[TemporalSelector]` — `{ mode: "single" | "compare" | "trend",
  snapshot, baseline, range }`. Absent ⇒ point-in-time on the latest/loaded snapshot.
- `segment: Optional[str]` — `portfolio | spv | acquired_portfolio` for cohort states.

---

## 4. Target M&A Agent architecture

The M&A Agent is the **narrow, point-in-time sibling** of the MI Agent. It
**reuses the shared library** but is constrained **by route contract**, not by
re-implementation.

### 4.1 Scope (funded-only, single tape)

| Allowed | Excluded by contract |
|---|---|
| Funded stratifications (the §3.4 catalogue, funded subset) | Pipeline state |
| Cohort/vintage by **origination date** | Forecast-funded state |
| Cohort/vintage by **acquisition date** *(if supplied)* | Recurring snapshot history / weekly-monthly feeds |
| Cohort/vintage by **risk grade / PD** | Risk-grade/PD *migration over time* (needs ≥2 snapshots) |
| Cohort/vintage by **balance band** | Early-warning concentration movement |
| Portfolio & acquired-portfolio segmentation | — |
| Data-quality exceptions for diligence (reuse exception engine) | — |
| Concentration analysis for a buyer/acquirer | — |

The M&A Agent **may** read the snapshot store if historical performance tapes are
provided (treated as a small fixed set of `as_of` cuts), but it does **not**
depend on a recurring feed and does **not** assemble forecast-funded.

### 4.2 What M&A reuses vs what is M&A-specific

- **Reuses:** buckets, stratification catalogue (funded subset), cohort/vintage
  core, concentration logic, chart factory, **data-quality exception engine**
  (`exception_db.py` / `exception_queue.py` / `ingest_violations.py`).
- **M&A-specific:** a buyer **diligence scorecard** (exception materiality
  distribution, field-completeness %, concentration vs buyer limits), and
  **acquired-portfolio segmentation** surfaced as a first-class view.

### 4.3 Why a separate agent rather than an MI flag

Keeping M&A as a distinct route with its own contract (a) prevents accidental
exposure of pipeline/forecast/recurring features to a buyer context, (b) keeps
the M&A surface auditable for diligence, and (c) lets M&A ship without the
snapshot-history layer that MI needs.

---

## 5. Shared analytics library design

A pure, deterministic, UI-free Python package — the common substrate for all
three routes. **No Streamlit, no Azure, no LLM, no I/O.** Functions take frames +
config, return frames/figures.

| Module (proposed) | Responsibility | Seeded from legacy (reuse, not import) |
|---|---|---|
| `buckets/` | Bucket definitions + scale-normalisation (LTV, age, rate, ticket, vintage, term) | `analytics/mi_prep.add_buckets`, `pipeline_tab_helpers.add_pipeline_stratification_buckets` |
| `stratify/` | Balance & count by dimension; top-N + concentration share | legacy inline strat charts + `charts_plotly.strat_bar_chart_pure` |
| `cohort/` | Vintage panels, MOB curves, runoff, status-migration (Sankey) | `analytics/static_pools_core.py` (strongest reuse) |
| `concentration/` | Group shares, 2-D concentration matrix, limit-usage evaluation | legacy heatmap/treemap + `risk_monitor.determine_status` + `risk_limits_config` |
| `forecast/` | Expected-funding conversion model; forecast-funded assembly | `analytics/pipeline_expected_funding.py` |
| `migration/` | Two-/N-snapshot joins → transition matrices; deterioration flags | `static_pools_core.build_status_migration_sankey` (generalised) |
| `semantics/` | Region normalisation, safe LTV, field resolution | `analytics/portfolio_semantics.py` |
| `charts/` | Themed Plotly factory (already isolated in MI Agent) | `mi_chart_factory` + `charts_plotly` theme rules |
| `exceptions/` | Data-quality finding/materiality model (read interface) | `exception_db.py`, `ingest_violations.py` |

Design rules:
- **Library is route-agnostic.** Routes compose it; the library knows nothing
  about MI vs M&A.
- **Legacy is a *reference*, not a dependency.** Reimplement the *definitions*
  (bucket edges, probability defaults, limit thresholds) in clean modules; do not
  import `analytics/`.
- **Buckets become first-class dimensions** (materialised), closing the v1 gap
  where the executor only reuses pre-existing bucket columns.

---

## 6. Snapshot & history layer (the persistent layer MI needs)

This is the **new core** that makes MI recurring. Point-in-time routes can ignore
it; MI/risk/forecasting cannot work without it.

### 6.1 Is a snapshot model needed? — Yes; recommended shape

The proposed snapshot model (from the brief) is **endorsed**, with the date
fields explicitly **separated** (the legacy pipeline's main weakness — see §7).
Recommended two-level model:

**Snapshot header (one row per upload/cut):**

| Field | Purpose |
|---|---|
| `snapshot_id` | Stable surrogate key for the snapshot (e.g. hash of client+cut_off+source_file) |
| `client_id` | Tenant / portfolio owner |
| `upload_timestamp` | When the file landed (operational) — **distinct from reporting date** |
| `reporting_date` | The MI reporting period the upload represents |
| `cut_off_date` | Data cut-off the figures reflect (may differ from reporting_date) |
| `source_file_id` | Stable id of the raw tape (content hash) |
| `source_file_name` | Original filename (audit) |
| `cadence` | `weekly | monthly | adhoc | backfill` |
| `route` | `mi | mna | regulatory` (provenance) |

**Snapshot loan-level row (one row per loan per snapshot):**

| Field | Purpose |
|---|---|
| `snapshot_id`, `client_id` | FK to header |
| `loan_id` | **Stable join key across snapshots** (see §6.3) |
| `portfolio_id`, `spv_id`, `acquired_portfolio_id` | Segmentation keys |
| `pipeline_stage`, `funded_status` | State assignment (pipeline vs funded) |
| `balance` | Core measure |
| `risk_grade`, `pd`, `ifrs9_stage` | Risk migration inputs |
| `origination_date`, `funding_date`, `acquisition_date`, `spv_transfer_date` | **Separated** event dates for cohorting |
| *(+ all canonical stratification fields)* | Dimensions |

> **Storage form is an implementation choice** (Parquet partitioned by
> `client_id`/`reporting_date`, or DuckDB/SQLite, or columnar blobs). The
> *contract* above is what matters; do not hard-code a backend now.

### 6.2 `SnapshotStore` interface (cloud-agnostic)

A thin interface the library/agents depend on, with pluggable adapters:

```
SnapshotStore:
  list_snapshots(client_id, route, cadence=None, since=None) -> [SnapshotHeader]
  resolve(selector) -> SnapshotHeader | (baseline, current) | [SnapshotHeader]
  load_loans(snapshot_id) -> DataFrame      # loan-level rows
  register(header, frame) -> snapshot_id    # idempotent on source_file_id
```

Adapters: `LocalFsSnapshotStore` (default, dev/test), `AzureBlobSnapshotStore`
(prod), future `S3SnapshotStore`. **Business logic imports the interface, never
an SDK.**

### 6.3 Stable join key for cross-snapshot tracking

Migration/movement require a loan to be identifiable across snapshots. Adopt the
legacy pattern but generalise it:
- **Funded loans:** prefer a real, stable `loan_id` (`loan_identifier` /
  `loan_policy_number`).
- **Pipeline opportunities:** reuse the legacy **deterministic SHA1
  opportunity key** (`pipeline_prep.py` — built from KFI/account/broker/product/
  amount/application-date) so the same opportunity is tracked week-to-week and
  can be reconciled to a funded `loan_id` on completion.
- Persist both, plus the `reconciliation_match_rule`, so pipeline→funded
  movement is queryable.

### 6.4 What recurring operations the layer enables

Weekly/monthly uploads · backfilled history (just register with the right
`reporting_date`) · snapshot-to-snapshot comparison · funded-book movement ·
pipeline movement · forecast-funded movement · cohort/vintage across periods ·
SPV-level trend · acquired-portfolio trend · risk-grade & PD migration ·
early-warning concentration movement.

---

## 7. Legacy Azure orchestration: assessment, reuse, and cloud-agnostic recommendation

Evidence base (read-only inspection): `analytics/blob_storage.py` (385 lines),
`function_app.py` (Event Grid trigger), `analytics/pipeline_persistence.py`,
`analytics/pipeline_prep.py`, `host.json`, `Dockerfile.streamlit`,
`deploy-streamlit.sh`.

### 7.1 How the legacy pipeline handles history today (facts)

| Concern | Legacy mechanism | Reference |
|---|---|---|
| Multiple timestamped tapes | List blobs under a prefix, sort by `last_modified` | `blob_storage.list_pipeline_snapshots`, `list_canonical_csvs` |
| Snapshot naming | `{stem}_{etag[:12]}.csv`; pointer manifest `latest_pipeline_snapshot.json` | `blob_storage.py`, `function_app._register_latest_pipeline_snapshot` |
| History depth | Up to **16 weeks** loaded for trends; storage itself unbounded | `tab_pipeline._build_weekly_snapshot_history(max_weeks=16)` |
| Report date | `snapshot_date` column → else max(stage dates) → else blob `last_modified` → else today | `pipeline_prep._choose_snapshot_date` |
| Snapshot comparison | Load N snapshots, compute per-stage `delta = curr − prev` | `tab_pipeline._render_weekly_flow_summary`, `_fmt_delta` |
| Cross-snapshot tracking | Deterministic **SHA1 opportunity key** | `pipeline_prep` (KFI/acc/broker/prod/amt/app-date) |
| Pipeline→funded | String-match `account_number` ↔ `loan_policy_number` / `loan_identifier+'01'` | `pipeline_reconciliation.reconcile_pipeline_to_funded` |
| Route selection | **Mode parsed from blob path/filename** (`mi`/`regulatory`/`annex12`) → `trakt_run.py --mode` | `function_app._parse_mode_from_path` |
| Idempotency | Event dedup on blob **etag** | `function_app._ingest_pipeline_snapshot` |

### 7.2 Date separation — the key weakness

The legacy pipeline **conflates upload date with reporting date** (`snapshot_date`
falls back to blob `last_modified`), and has **no** `cut_off_date`,
`acquisition_date`, or `spv_transfer_date` as distinct concepts. Origination and
the pipeline stage dates (KFI/application/offer/funds-released) *are* separated,
but the rest are missing or merged. The target snapshot model (§6.1) fixes this
by making every date explicit.

### 7.3 Reusable concepts (keep as patterns)

- **Deterministic synthetic key** for cross-snapshot opportunity tracking.
- **Idempotent registration** via a content/etag hash (generalise etag →
  `source_file_id` content hash so it is not Azure-specific).
- **Pointer/manifest** for "latest snapshot" and run provenance
  (`run_timestamp`, `run_mode`, row counts, source paths).
- **Mode/route-from-path** routing concept (but decouple from blob-path
  specifics — make route an explicit input, not an Azure folder convention).
- **Snapshot prefix layout** (`mi/pipeline_snapshots/`, `snapshots/{as_of}-summary.csv`)
  as a *logical* layout, re-expressed as `SnapshotStore` keys.

### 7.4 Concepts to discard / not copy

- **Direct Azure SDK calls throughout business logic** (`BlobServiceClient`,
  `list_blobs`, `download_blob`, `upload_blob`, `event_grid_trigger`) with
  **no abstraction and no local fallback** — `is_azure_configured()` raises if
  Azure is absent. Do not reproduce this coupling.
- **Azure-specific env/config baked into logic:** `DATA_STORAGE_CONNECTION`,
  `AZURE_STORAGE_ACCOUNT`, `TRAKT_OUTBOUND_CONTAINER`, `TRAKT_PIPELINE_*`,
  Event Grid subject parsing, `host.json` function timeout. These belong in a
  **deployment adapter**, not the analytics.
- **Upload-date-as-reporting-date** conflation (§7.2).
- **Etag as the only identity** — replace with a content hash so identity is
  storage-portable.
- **On-session recompute of history** from blob each time — the snapshot store
  should persist normalised snapshots so trends are not re-derived per session.

### 7.5 Recommendation: cloud-agnostic-first, Azure as an adapter

**Yes — orchestration should be cloud-agnostic first.** Model the snapshot store,
state assembly, and analytics in **storage-neutral business logic** that depends
only on the `SnapshotStore` interface (§6.2) and local filesystem by default.
Provide **Azure Blob as one adapter** (re-using the proven blob layout and
etag/idempotency patterns inside the adapter), with S3/GCS as future adapters.
Triggers (Event Grid today) become **thin ingress** that call
`SnapshotStore.register(...)` — they carry no analytics logic. This keeps the MI
Agent testable locally, portable across clouds, and free of the legacy's tight
coupling, while preserving the legacy's genuinely good ideas.

---

## 8. Proposed route contracts

A **RouteContract** is a declarative object (config-backed) the route layer
resolves before any query runs. It bounds what a route may do, making the MI/M&A
distinction enforceable and auditable.

```
RouteContract:
  route_id:            mi | mna | regulatory
  temporality:         recurring | point_in_time
  allowed_states:      [ ... state keys ... ]
  allowed_segments:    [ portfolio | spv | acquired_portfolio | none ]
  allowed_dimensions:  [ ... stratification keys ... ]
  temporal_modes:      [ single | compare | trend ]      # point_in_time ⇒ [single]
  requires_history:    bool
  risk_monitor:        enabled | disabled
  forecast:            enabled | disabled
  exceptions_scorecard: enabled | disabled
```

| Field | MI | M&A | Regulatory |
|---|---|---|---|
| temporality | recurring | point_in_time | point_in_time |
| allowed_states | all 7 | funded + cohort (orig/acq/grade/PD/balance) | n/a |
| allowed_segments | portfolio, spv, acquired | portfolio, acquired | n/a |
| temporal_modes | single, compare, trend | single | single |
| requires_history | true | false (optional read) | false |
| risk_monitor | enabled | disabled | disabled |
| forecast | enabled | disabled (unless requested) | disabled |
| exceptions_scorecard | optional | enabled | (own validation path) |

The **validator** gains a contract check: a spec referencing a state/dimension/
temporal mode outside the active contract fails fast with a clear message
(consistent with the existing repair-loop philosophy — errors, never data).

---

## 9. Required canonical field-registry extensions

These are **data-model gaps** confirmed in the current-state audit (§7 there) and
by registry inspection (`config/system/fields_registry.yaml`, 439 fields). They
block MI-risk and M&A until added. Two buckets: **fields** and **snapshot
metadata**.

### 9.1 Missing canonical loan-level fields

| Concept | Proposed field | Notes |
|---|---|---|
| Internal risk stage | `internal_risk_stage` | core across asset classes (currently only SME `bank_internal_rating`) |
| IFRS 9 stage | `ifrs9_stage` | enum Stage 1/2/3 |
| Loan-level PD | `probability_of_default` (+ derived `pd_bucket`) | only a corporate-guarantor PD exists today |
| Risk grade | `internal_risk_grade` | core; enables M&A grade cohorts + migration |
| Single/joint borrower | `number_of_borrowers` (+ derived `borrower_structure`) | derivable from `borrower_1/2_*` presence |
| Amortisation type (curation) | *already canonical* `amortisation_type` | **registry-curation fix** — add to MI semantic registry; no new field |
| Time on book | derived `months_on_book` | from `funding_date`/`origination_date` vs `reporting_date` |

### 9.2 Missing segmentation / identity fields

| Concept | Proposed field | Notes |
|---|---|---|
| Portfolio identifier | `portfolio_id` | enables portfolio cohort/trend |
| SPV identifier | `spv_id` | enables SPV trend reporting |
| Acquired-portfolio identifier | `acquired_portfolio_id` | enables acquired-portfolio cohorts (MI & M&A) |

### 9.3 Missing / separated dates

| Date | Proposed field | Today |
|---|---|---|
| Funding date | `funding_date` | partly via `date_funds_released` (pipeline only) |
| Acquisition date | `acquisition_date` | **missing** |
| SPV transfer date | `spv_transfer_date` | **missing** |
| Reporting date | `reporting_date` (snapshot header) | conflated with `snapshot_date` |
| Cut-off date | `cut_off_date` (snapshot header) | **missing** |
| Upload timestamp | `upload_timestamp` (snapshot header) | implicit (blob `last_modified`) |

> Origination date (`origination_date`), arrears (`account_status`,
> `arrears_balance`, `number_of_days_in_arrears`), LTV, region, rate, product,
> origination channel, broker channel already exist and need **no** new fields.

### 9.4 Pipeline-stage canonicalisation

`pipeline_stage` (KFI / application / offer / completion / funded) exists only in
legacy `pipeline_prep.py`. Promote it to a canonical enum used by both the
snapshot model and the stratification catalogue.

---

## 10. Proposed config files

Config-driven, route-aware, asset/regime-aware — extending the existing pattern
(`config/mi/mi_equity_release_uk_applicability.yaml`,
`config/client/risk_limits_config.py`, `config/client/pipeline_expected_funding.yaml`).

| Proposed file | Purpose |
|---|---|
| `config/routes/mi_route.yaml` | MI RouteContract (states, dims, temporal modes, requires_history). |
| `config/routes/mna_route.yaml` | M&A RouteContract (funded-only, point-in-time). |
| `config/routes/regulatory_route.yaml` | Reference only; documents that Annex 2 is point-in-time and out of scope. |
| `config/mi/state_library.yaml` | Declarative state definitions (inputs, assembly rules). |
| `config/mi/stratification_catalogue.yaml` | Dimension → semantic field → bucket rule → applicable states/asset-classes. |
| `config/mi/buckets.yaml` | Bucket edges + scale-normalisation (lifted from `mi_prep` definitions). |
| `config/mi/risk_monitor.yaml` | Early-warning rules: limit usage, migration pairs, deterioration thresholds, trajectory windows. |
| `config/client/concentration_limits.yaml` | Per-client limits + amber/red thresholds (generalise `risk_limits_config.py`). |
| `config/client/expected_funding.yaml` | Already exists; reused for forecast-funded. |
| `config/storage/snapshot_store.yaml` | Storage adapter selection (`local_fs` default; `azure_blob` prod) + layout keys. No SDK details in business logic. |
| `config/mna/diligence_scorecard.yaml` | M&A exception-materiality scorecard + buyer concentration limits. |

---

## 11. Proposed module/file structure

Additive packages alongside the existing `mi_agent/`. **Nothing in `analytics/`
is moved or modified.**

```
analytics_lib/                     # NEW shared, pure library (no UI/Azure/LLM/IO)
  buckets/                         # bucket defs + scale normalisation
  stratify/                        # balance & count by dimension, top-N, concentration
  cohort/                          # vintage panels, MOB curves, runoff, migration
  concentration/                   # group shares, matrices, limit-usage eval
  forecast/                        # expected conversion, forecast-funded assembly
  migration/                       # snapshot joins → transition matrices, deterioration
  semantics/                       # region/LTV/field resolution
  charts/                          # themed Plotly factory (theme rules)
  exceptions/                      # data-quality finding/materiality read model

snapshot/                          # NEW snapshot & history layer
  model.py                         # SnapshotHeader / SnapshotLoanRow contracts
  store.py                         # SnapshotStore interface + selector resolution
  keys.py                          # stable loan_id / SHA1 opportunity key
  adapters/
    local_fs.py                    # default
    azure_blob.py                  # prod adapter (reuses legacy blob layout/idempotency)

mi_agent/                          # EXISTING — extended, not rewritten
  mi_query_spec.py                 # + state, temporal, segment (additive)
  mi_query_validator.py            # + RouteContract checks
  mi_agent_workflow.py             # + state assembler + temporal resolution
  states/                          # NEW: total_pipeline/funded/forecast_funded/cohort_*
  risk_monitor/                    # NEW: early-warning capability
  routes/
    mi_route.py
  (executor, chart_factory, parser, semantics registry unchanged)

mna_agent/                         # NEW — narrow funded-only sibling
  mna_workflow.py
  routes/mna_route.py
  diligence_scorecard.py           # reuses analytics_lib/exceptions

config/                            # NEW config trees per §10
  routes/  mi/  mna/  storage/  client/
```

---

## 12. Build sequencing plan

Sequenced so each phase ships independently and de-risks the next. No phase
requires refactoring `analytics/`.

| Phase | Goal | Key deliverables | Depends on |
|---|---|---|---|
| **0. Foundations** | Make the gaps buildable | Canonical field-registry extensions (§9); promote `pipeline_stage`; add `amortisation_type` to MI semantic registry; bucket config (§10) | — |
| **1. Shared library** | Pure, tested analytics substrate | `analytics_lib/` buckets + stratify + cohort + concentration + charts (reimplement legacy *definitions*) | 0 |
| **2. Snapshot layer** | Stateful history | `snapshot/` model + `SnapshotStore` interface + `local_fs` adapter + stable keys; backfill/registration | 0 |
| **3. MI states (point-in-time)** | States without time-series first | `total_funded`, `total_pipeline`, `total_forecast_funded`; MIQuerySpec `state`/`segment`; validator route checks | 1, 2 |
| **4. MI route + temporal** | Recurring MI | TemporalSelector (single/compare/trend); snapshot-to-snapshot deltas; cohort-by-date/portfolio/SPV/acquired; SPV & acquired trends | 3 |
| **5. Risk monitor** | Early warning | concentration (funded/forecast), movement, limit usage, risk-grade & PD migration, deterioration flags | 4 |
| **6. M&A agent** | Funded-only diligence | `mna_agent/` route, funded stratifications, orig/acq/grade/PD/balance cohorts, diligence scorecard (reuse exception engine) | 1, 2 (read) |
| **7. Azure adapter** | Production storage | `azure_blob` SnapshotStore adapter + thin ingress; deployment config | 2 |
| **8. Export & UI polish** | Delivery | KPI cards, PPTX path (reuse legacy deck taxonomy), route-aware UI | 3–6 |

**Quick win, available immediately and independently:** Phase-0 registry
curation of `amortisation_type` + materialising `bucket_field` hints — registry/
config only, no data-model work.

---

## 13. Testing strategy

Mirrors the MI Agent's existing discipline (deterministic, mockable, data-free
LLM). Every layer is unit-tested without a browser, Azure, or an API key.

| Layer | Test approach |
|---|---|
| Shared library | Pure-function unit tests with small synthetic frames; golden-file tests for bucket edges, concentration shares, cohort curves. Property tests for scale-normalisation (decimal vs percent LTV/rate). |
| Snapshot store | Contract tests run against **every adapter** (local_fs now, azure_blob later) via a shared test suite; idempotent `register` (same `source_file_id` ⇒ no dup); selector resolution (single/compare/trend) on fixture snapshots. |
| States | Assemble each state from fixture snapshots; assert `forecast_funded = funded + Σ(pipeline×prob)` exactly; cohort segmentation by portfolio/SPV/acquired. |
| Temporal/migration | Two-snapshot fixtures with known transitions ⇒ assert transition matrices, deltas, deterioration flags. |
| Risk monitor | Limit-usage thresholds (green/amber/red) against config; early-warning trajectory across ≥3 snapshots. |
| Route contracts | Spec referencing a disallowed state/dim/temporal mode ⇒ validator fails with a clear, data-free message. MI vs M&A contract enforcement. |
| Regression vs legacy | For overlapping outputs (e.g. LTV buckets, region concentration), compare new library output to legacy figures on a shared synthetic tape to confirm parity of *definitions*. |
| End-to-end | `question → spec → validate → assemble → execute → chart` on the synthetic demo pack, deterministic mode (no LLM key), plus a mocked-LLM path. |

Synthetic fixtures already exist (`synthetic_demo/`, `synthetic_onboarding_pack*/`,
demo CSVs) and should seed multi-snapshot test data.

---

## 14. Migration strategy from Streamlit to MI Agent

**Principle: strangler-fig, not rewrite.** The legacy app keeps running untouched
while capability moves, view-by-view, into the MI Agent. No charts are migrated in
this task; this is the *plan* for when migration is sanctioned.

1. **Freeze, don't fork.** `analytics/` stays as-is and in service. The MI Agent
   reaches feature-parity for one view at a time.
2. **Lift definitions, not files.** Reimplement bucket edges, probability
   defaults, limit thresholds, cohort mechanics as clean `analytics_lib/` modules
   with parity tests against legacy outputs (§13). Never `import analytics`.
3. **View-by-view cutover order** (lowest-risk first):
   funded stratifications → concentration/heatmap → cohort/vintage (static pools)
   → pipeline/forward → forecast-funded → risk monitor → scenarios (optional).
4. **Dual-run & reconcile.** During each cutover, run legacy and MI Agent on the
   same snapshot and diff key figures; promote only on parity.
5. **Decommission per view.** Once a view is at parity and adopted, retire the
   corresponding Streamlit tab — not the whole app at once.
6. **Exception engine stays shared.** `exception_db.py` et al. are reused by the
   M&A diligence scorecard directly; no migration needed.
7. **Scenarios are explicitly deferred** (optional, last) — `scenario_engine.py`
   is self-contained and low-priority for MI/M&A.

---

## 15. Risks and design decisions

### 15.1 Key design decisions (and rationale)

| Decision | Rationale |
|---|---|
| Keep MI Agent v1 core; add layers beneath | Preserves the tested governed pipeline; avoids a risky rewrite. |
| Separate point-in-time vs recurring routes | Root-cause fix for the flat/stateless problem; lets M&A/Regulatory stay stateless. |
| M&A as a **separate** narrow agent, not an MI flag | Prevents leaking pipeline/forecast/recurring features into a buyer context; auditable. |
| Cloud-agnostic `SnapshotStore`, Azure as adapter | Removes legacy tight coupling; testable locally; portable. |
| Two-level snapshot model with **separated dates** | Fixes legacy upload/report conflation; enables cohorting by every event date. |
| Reuse legacy *definitions*, never import `analytics/` | Captures proven logic without inheriting the monolith. |
| Config-driven route contracts | Makes MI/M&A boundaries enforceable and reviewable. |

### 15.2 Risks

| Risk | Impact | Mitigation |
|---|---|---|
| **Canonical extensions (risk stage, IFRS 9, PD, SPV/portfolio/acquired ids, dates) depend on source data that may not exist** | Blocks MI-risk and M&A cohorts | Sequence registry work in Phase 0; degrade gracefully (dimension "unavailable" rather than failure) when a client lacks a field, mirroring `config/mi/...applicability.yaml`. |
| **Snapshot identity / join-key instability** across uploads | Breaks migration & movement | Standardise stable `loan_id` + SHA1 opportunity key (§6.3); contract-test idempotency. |
| **Upload vs reporting vs cut-off conflation** repeated from legacy | Wrong period attribution | Make all three explicit in the snapshot header; never default reporting_date to upload time. |
| **Scope creep** — rebuilding the whole Streamlit app | Delay, monolith 2.0 | Strangler-fig, view-by-view, parity-gated (§14). |
| **Storage-cost / retention** of unbounded snapshots | Operational cost | Define retention/partitioning in `snapshot_store.yaml`; legacy had none. |
| **Forecast model assumptions** (fixed stage probabilities) treated as truth | Misleading forecast-funded | Keep probabilities in config; surface assumptions in output (as legacy scenario exports do). |
| **Asset-class variability** (e.g. youngest-borrower only for ER) | Wrong dimensions per client | Drive applicability from config (asset_class/jurisdiction), extending the existing overlay. |
| **Regulatory bleed** — MI/M&A design pulling on Annex 2 | Contract risk | Regulatory route documented as point-in-time and explicitly excluded from MI/M&A drivers. |

---

*Documentation only. Per the brief: no source code was modified, no charts were
migrated, the agents and snapshot layer were not built, and the legacy Azure
orchestration was assessed (not copied).*
