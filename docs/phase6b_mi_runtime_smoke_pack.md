# Phase 6B — MI Runtime Smoke Pack

**Status:** Implemented (deterministic smoke pack + fixture + docs). Proves the
Phase 6 MI runtime works end-to-end over canonical snapshot frames via
`LocalFsSnapshotStore` and `run_mi_query`. No multi-artefact consolidation, no
onboarding, no Azure, no Streamlit, no M&A Agent, no Annex 2/regulatory changes,
no new chart types, no LLM.

**Dependency order:** Phase 0 → 0B → 1 → 2 → 3 → 4 → 5 → 6 → **6B** (stacked on
Phase 6 `run_mi_query`).

**Date:** 2026-06-18

---

## 1. What was built

A single, deterministic smoke pack that exercises every Phase 6 runtime path
against small synthetic **canonical snapshot frames** persisted through the real
snapshot layer:

- `tests/test_phase6b_mi_runtime_smoke_pack.py` — 19 tests.
- `tests/fixtures/phase6b_flat_canonical.csv` — a tiny canonical CSV for the
  flat single-CSV path.

The pack registers **three monthly snapshots** (`2024-01-31`, `2024-02-29`,
`2024-03-31`) for one client (`smoke`, route `mi`) into a `LocalFsSnapshotStore`,
then drives `run_mi_query` for each capability. The snapshot layer is never
bypassed — state/temporal/risk queries resolve and load through the store, and a
state query with no store is asserted to fail with `snapshot_store_required`.

### Synthetic dataset (deterministic)
Each snapshot is a canonical loan-level frame with `loan_identifier`,
`funded_status`, `pipeline_stage`, `current_outstanding_balance`,
`geographic_region_obligor`, `broker_channel`, `portfolio_id`,
`internal_risk_grade`, `ifrs9_stage`, `pd_bucket`,
`forecast_funding_probability`, `origination_date`. The data is shaped so the
funded book grows (300 → 620 → 620), a loan migrates (F2: grade B→C, IFRS 1→2,
PD 0.25-0.5%→0.5-1%), a funded loan is added (F3) and a pipeline opportunity
exits (P2).

---

## 2. Proven runtime paths

| # | Capability | Path | Key assertion |
|---|---|---|---|
| 1 | `total_funded` latest | state | 3 rows, balance 620 |
| 2 | `total_pipeline` latest | state | 1 row, balance 50 |
| 3 | `total_forecast_funded` latest | state | forecast total 645 (620 + 50×0.5) |
| 4 | funded compare baseline/current | temporal compare | 300→620, +320, new 1 / retained 2 |
| 5 | funded trend (3 snapshots) | temporal trend | [300, 620, 620], line chart |
| 6 | pipeline trend (3 snapshots) | temporal trend | [90, 90, 50] |
| 7 | forecast-funded trend (3 snapshots) | temporal trend | [335, 663, 645] |
| 8 | funded by **portfolio** | risk/concentration of `total_funded` | "portfolio" → Trakt `portfolio_id`; PF_001=400, PF_002=220 |
| 9 | funded by region | risk/concentration of `total_funded` | North=400, South=220 |
| 10 | pipeline by stage | risk/concentration of `total_pipeline` (as-of S2) | OFFER=50, APPLICATION=40 |
| 11 | forecast-funded by region | risk/concentration of `total_forecast_funded` | North=425, South=220 |
| 12 | forecast-funded by broker | risk/concentration of `total_forecast_funded` | Broker A=425 |
| 13 | concentration warning | risk/concentration | North 64.5% → `status=red` |
| 14 | risk grade migration | risk/migration baseline/current | B→C `deteriorated` |
| 15 | IFRS 9 migration | risk/migration baseline/current | Stage 1→Stage 2 `deteriorated` |
| 16 | PD bucket migration | risk/migration baseline/current | `deteriorated` present |
| 17 | existing flat single-CSV query | flat | grouped bar over the CSV fixture |
| 18 | flat uses governed chart factory | flat (`build_chart`) | `bar` rendered |
| 19 | snapshot layer genuinely used | store + negative case | 3 snapshots listed; no-store → `snapshot_store_required` |

> **Point-in-time "by dimension" views (8–13)** are produced through the
> governed concentration path (`run_concentration` over the selected state),
> which returns balance/count/share per group plus RAG status. This reuses an
> existing Phase 5 capability rather than adding a new grouped-state feature.
> "portfolio" is resolved via the Phase 6 Step 0 `semantic_resolver` using the
> example portfolio reference config, demonstrating the Trakt portfolio
> reference model end-to-end.

---

## 3. What was intentionally NOT built

- **No multi-artefact consolidation fixture** (explicitly deferred) — this pack
  uses canonical snapshot frames only.
- No onboarding orchestration, no Azure adapter, no Event Grid/blob ingestion.
- No Streamlit migration, no M&A Agent runtime, no UI.
- No Annex 2/XML/regulatory changes.
- No new chart types (only the governed `bar`/`line` instructions already used).
- No LLM parsing — specs are constructed directly.

---

## 4. How it uses the snapshot layer

All non-flat queries go through `LocalFsSnapshotStore`:
`run_mi_query` → resolves snapshot(s) via the store's `resolve_latest` /
`resolve_as_of` / `resolve_range` → `load_loans` → Phase 3/4/5 assembly. The pack
asserts the store holds three snapshots and that the latest is `2024-03-31`, and
that omitting the store yields a structured `snapshot_store_required` issue — so
the smoke pack proves the runtime genuinely depends on the snapshot layer.

---

## 5. Files changed

- `tests/test_phase6b_mi_runtime_smoke_pack.py` (new, 19 tests).
- `tests/fixtures/phase6b_flat_canonical.csv` (new fixture).
- `docs/phase6b_mi_runtime_smoke_pack.md` (this file).

## 6. Tests run

`tests/test_phase6b_mi_runtime_smoke_pack.py` — **19 passed**.

## 7. Limitations / deferred items

- Multi-artefact (multi-file) consolidation into a single snapshot is deferred to
  a later phase, by request.
- Grouped "by dimension" point-in-time views are served via the concentration
  path; a dedicated grouped-state stratification in the runtime is a possible
  future ergonomic addition.
- Non-flat chart rendering remains a governed `chart_instruction` (not
  auto-rendered), consistent with Phase 6.
