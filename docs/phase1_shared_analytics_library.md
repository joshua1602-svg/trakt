# Phase 1 — Shared Analytics Library Foundations

**Status:** Implemented (pure library + tests + docs). No orchestration, no
runtime routes, no MI/M&A agent execution, no snapshot/history layer, no Azure,
no Streamlit, no LLM calls, no chart migration, no legacy `analytics/` imports,
no regulatory Annex 2 changes.

**Date:** 2026-06-18

This phase builds the pure analytical substrate (`analytics_lib/`) that the MI
and M&A routes will compose in later phases. It consumes the Phase 0B config
skeletons (`config/mi/buckets.yaml`) and the curated MI semantic registry, and
turns them into deterministic, UI-free functions over pandas DataFrames.

---

## 1. What was built

A new pure package, `analytics_lib/`:

| Module | Responsibility |
|---|---|
| `config_loader.py` | Read-only YAML helpers; `load_bucket_config()` loads & lightly validates `config/mi/buckets.yaml`. |
| `buckets.py` | **Config-driven bucket materialisation engine** — turns numeric source columns into bucket columns per the Phase 0B edge definitions; returns `(df_out, issues, applied)`. |
| `stratify.py` | **Generic balance/count stratification** by one categorical or pre-bucketed dimension. |
| `concentration.py` | **Concentration maths** — group shares, top-N concentration, simple limit-usage RAG status. |
| `cohort.py` | **Point-in-time cohort/vintage foundations** — cohort period derivation (Y/Q/M), cohort tables by any event date, and months-on-book. |
| `migration.py` | **Stub only** — documents the deferred snapshot-to-snapshot migration surface; raises `NotImplementedError`. |
| `__init__.py` | Curated public API surface. |

All functions are pure: input is a DataFrame + config dict/path; output is a
DataFrame or plain dict/list. No file-system writes, no UI, no network.

### 1.1 Bucket engine (config-driven)
- Materialises every bucket declared in `config/mi/buckets.yaml`: **LTV**,
  **borrower age**, **youngest borrower age**, **interest rate**, **PD**,
  **LGD**, **EAD**, **balance band**, **time on book**.
- Returns a **structured issue list** distinguishing:
  `unavailable_field` (column missing) · `invalid_numeric` (non-coercible
  values) · `out_of_range` (numeric but outside every band) · `scale_normalised`
  (info: a decimal/percent conversion was applied) · `config_error`.
- **Decimal-vs-percent normalisation** (LTV, interest rate, PD, LGD) via a
  column-median heuristic, so a single oddball value never flips the column. The
  conversion is recorded as an `info` issue — never silent.
- **Two label conventions supported** (both present in `buckets.yaml`):
  `len(labels) == len(edges) - 1` (capped top band) and
  `len(labels) == len(edges)` (final label is an overflow `[last_edge, +inf)`,
  e.g. LTV `">=100%"`).
- Obviously invalid data is **never silently coerced** without an issue record.

### 1.2 Stratification engine
One row per category with `loan_count`, `balance_sum`, `balance_share`,
`avg_balance`, and optional `{metric}_weighted_avg` columns. Deterministic
ordering (balance descending, then dimension ascending, stable). Explicit
`Unknown` bucket for missing dimension values. Works on categorical **and**
bucketed dimensions. No chart output.

### 1.3 Concentration engine
`group_shares` (balance + count share), `top_n_concentration` (combined
top-N share + table), `limit_usage` (share ÷ limit with green/amber/red
`rag_status` from configurable thresholds). This is the maths only — **not** the
MI risk monitor.

### 1.4 Cohort/vintage foundation
Point-in-time cohorting by **origination**, **acquisition**, or **funding**
date (same function, parameterised by `date_col`), at year/quarter/month
granularity, plus `months_on_book` against a scalar or per-row reporting/as-of
date (negative seasoning clamped to 0 with a warning).

---

## 2. What was intentionally NOT built

- **No orchestration / runtime routes** — nothing resolves a route contract or
  runs a query pipeline.
- **No snapshot/history layer** — no persistence, no `SnapshotStore`.
- **No MI states runtime** — `state_library.yaml` is not executed.
- **No M&A Agent** — `diligence_scorecard.yaml` is not executed.
- **No snapshot-to-snapshot migration** — `migration.py` is a documented stub;
  transition matrices / deterioration flags require the Phase 2 snapshot layer.
- **No chart migration** — the MI Agent owns the governed chart factory; no
  Streamlit/Plotly chart code is reused or copied. *(Future note: multi-line
  cohort/vintage curves would be an **additive** enhancement to the MI Agent
  chart factory — not built here.)*
- **No legacy `analytics/` imports, no Azure, no LLM calls, no Annex 2 changes.**
  Legacy bucket *definitions* were used only as a numeric reference.

---

## 3. How Phase 1 consumes Phase 0B configs

- `config/mi/buckets.yaml` (Phase 0B skeleton) is now **executable**: the bucket
  engine reads its `source_field` / `edges` / `labels` / `scale` and materialises
  the columns. The two labelling conventions in that file are both honoured.
- The bucket `semantic_field` hints map cleanly onto the curated MI semantic
  registry dimensions (e.g. `balance_band → ticket_bucket`,
  `borrower_age_bucket → age_bucket`), so materialised columns can later be
  stratified by registry dimension keys.
- `config/mi/stratification_catalogue.yaml` dimension→bucket→field mapping is the
  contract the stratification engine is designed to satisfy (each catalogue
  dimension resolves to a `stratify()` call over a registry field or a
  materialised bucket column).
- Limit-usage thresholds mirror the RAG shape declared in
  `config/mi/risk_monitor.yaml` / `config/mna/diligence_scorecard.yaml`.

---

## 4. How Phase 1 prepares for Phase 2 (snapshot) and Phase 3 (MI states)

- **Phase 2 (snapshot/history layer):** the library is storage-neutral and
  frame-in/frame-out, so the snapshot store can feed it loan-level frames with no
  changes. `months_on_book` already takes a `reporting_date`/`as_of` input, which
  is exactly the snapshot-header date Phase 2 will supply. `migration.py` reserves
  the import surface that two-/N-snapshot joins will implement once history exists.
- **Phase 3 (MI states runtime):** each declared state (`total_funded`,
  `cohort_by_date`, `cohort_by_*`) becomes `assemble frame → materialise_buckets →
  stratify / concentration / cohort_table`. The state assembler composes these
  pure functions; the functions themselves need no further change. Route
  contracts then gate *which* states/dimensions are allowed — the analytics
  library stays route-agnostic.

---

## 5. Files changed

- `analytics_lib/__init__.py`, `config_loader.py`, `buckets.py`, `stratify.py`,
  `concentration.py`, `cohort.py`, `migration.py` (new package).
- `tests/test_phase1_analytics_lib.py` (new, 26 tests).
- `docs/phase1_shared_analytics_library.md` (this file).

## 6. Tests run

`tests/test_phase1_analytics_lib.py` — **26 passed**. Combined with the existing
Phase 0B and MI Agent suites: **214 passed** (`pandas`/`plotly`/`pytest`
installed in-environment; these are pre-existing environment gaps, not code
changes).

## 7. Limitations / deferred items

- Decimal/percent detection is a column-median heuristic; genuinely ambiguous
  single-unit columns (e.g. a PD column that is uniformly ~1.0) should be
  confirmed by config in a later phase if needed.
- `migration.py` is a stub pending the Phase 2 snapshot layer.
- Multi-line cohort/vintage chart rendering is deferred (future additive MI Agent
  chart-factory enhancement, not in scope here).
