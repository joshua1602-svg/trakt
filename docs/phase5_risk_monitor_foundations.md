# Phase 5 — Risk Monitor Foundations

**Status:** Implemented (migration + per-loan flags + concentration early-warning
+ trajectory + route gating + tests + docs). No full orchestration, no MI Agent
runtime wiring, no LLM query routing, no M&A agent runtime, no Azure, no
Streamlit/charts/UI, no legacy `analytics/` imports, no Annex 2/regulatory
changes.

**Dependency order:** Phase 0 → Phase 0B → Phase 1 → Phase 2 → Phase 3 →
Phase 4 → **Phase 5**.

**Date:** 2026-06-18

Phase 5 adds deterministic **early-warning and migration analytics** on top of
the snapshot / state / temporal foundations. Everything is pure and
frame-in/frame-out; nothing schedules, routes questions, or renders output.

---

## 1. What was built

`mi_agent/risk_monitor/`:

| File | Responsibility |
|---|---|
| `models.py` | `RiskMonitorResult`, movement-type constants, Phase 5 issue codes, and the `config/mi/risk_monitor.yaml` loader + accessors (orderings, thresholds, minimums, trajectory window). |
| `migration.py` | `migration_matrix` (two-snapshot transition matrix) and `per_loan_movement` (per-loan flags) joined on the stable `loan_id`; `classify_change`. |
| `concentration.py` | `funded_concentration` (RAG + approaching-limit), `concentration_movement` (baseline↔current / funded↔forecast), `top_n_concentration`. |
| `monitor.py` | Store-backed entry points (`run_migration`, `run_concentration`, `run_concentration_movement`, `run_funded_vs_forecast`, `run_trajectory`) + `validate_risk_monitor_route`. |
| `__init__.py` | Curated public API. |

### 1.1 Migration matrices
For a dimension, joins baseline/current on `loan_id` and emits rows of
`dimension, from_value, to_value, loan_count, balance_sum, balance_share,
movement_type`. `movement_type ∈ {unchanged, improved, deteriorated, new,
exited, changed, unknown}`. Direction (improved/deteriorated) is taken **only**
from config `deterioration_orderings` (BEST→WORST); **unordered** dimensions are
classified `changed`/`unchanged` and never invent a direction (an
`unordered_migration_dimension` issue is recorded). Supported dimensions:
`internal_risk_grade`, `internal_risk_stage`, `ifrs9_stage`, `pd_bucket`,
`lgd_bucket`, `arrears_bucket`, `ltv_bucket` (any column present on the frames).

### 1.2 Per-loan deterioration/improvement flags
`per_loan_movement` returns `loan_id, baseline_value, current_value,
movement_type, balance_baseline, balance_current, balance_change,
deterioration_flag, improvement_flag`, with missing-key/dimension issues. Covers
risk grade, IFRS 9 stage, PD bucket and arrears (any ordered dimension).

### 1.3 Concentration early-warning
`funded_concentration` reuses `analytics_lib.concentration.group_shares` and adds
`status` (green/amber/red via configured `concentration_thresholds`) and an
`approaching_limit` flag (share within the configured fraction of red but not yet
red). `concentration_movement` compares per-group share between two frames —
baseline↔current (period movement) or funded↔forecast — yielding
`baseline_share, current_share, share_change, increasing, status_current`.
`run_funded_vs_forecast` wires the funded vs forecast-funded comparison.
Dimensions: region, broker/origination channel, product, amortisation type,
risk grade, IFRS 9, PD/LTV buckets, portfolio/SPV/acquired-portfolio ids
(whatever is present). Minimum balance/count thresholds suppress RAG flags on
tiny groups (reported as `concentration_below_minimum_threshold`).

### 1.4 Trajectory (trend-based early warning)
`run_trajectory` reuses the **Phase 4 trend** (stratified by dimension), derives
per-snapshot group shares, and conservatively flags a group whose share is
non-decreasing across the window and rises overall, when its latest share is
amber-or-worse. Requires `trajectory_window` snapshots; fewer →
`insufficient_snapshots_for_trajectory`. No statistical forecasting.

### 1.5 Config-driven rules
`config/mi/risk_monitor.yaml` was extended (additively) with
`monitored_dimensions`, `migration_dimensions`, `deterioration_orderings`,
`concentration_thresholds`, `trajectory_window`, `minimum_balance_threshold`,
`minimum_count_threshold`, and an **illustrative, non-enforced**
`example_concentration_limits` block. No client-specific values are hard-coded in
code; `internal_risk_stage` is intentionally left unordered as an example.

### 1.6 Route eligibility
`validate_risk_monitor_route` reads the route contract: MI (`risk_monitor:
enabled`) is allowed; M&A (`disabled`) is rejected unless
`allow_mna_override=True`; Regulatory Annex 2 (`disabled`) is rejected. All
store-backed entry points enforce this before doing any work.

### 1.7 Structured issues
`missing_baseline_snapshot`, `missing_current_snapshot`,
`missing_stable_key_for_migration`, `missing_migration_dimension`,
`unordered_migration_dimension`, `insufficient_snapshots_for_trajectory`,
`missing_concentration_dimension`, `missing_limit_config`,
`concentration_below_minimum_threshold`, `unsupported_risk_monitor_route`,
`empty_risk_monitor_result`. Optional gaps never crash.

---

## 2. What was intentionally NOT built

- No full orchestration, no MI Agent runtime wiring, no LLM query routing.
- No M&A agent runtime, no Azure, no Streamlit/charts/UI.
- No statistical/predictive forecasting (trajectory is a conservative
  monotonicity check only).
- No legacy `analytics/` imports, no Annex 2/regulatory changes.
- No enforcement of nonexistent client concentration limits (only a documented
  example shape).

---

## 3. How it builds on Phase 4 temporal MI

Migration and concentration-movement are the same **two-snapshot join on a stable
key** that Phase 4's compare introduced, specialised to transition counts and
group-share deltas. `run_trajectory` is a thin early-warning layer over the
Phase 4 `trend` output (stratified by dimension), so trend remains the single
source of multi-snapshot assembly.

## 4. How it uses Phase 0/0B risk fields & config

The migration/concentration dimensions are exactly the Phase 0 canonical risk
fields (`internal_risk_grade`, `ifrs9_stage`, `probability_of_default` →
`pd_bucket`, `loss_given_default` → `lgd_bucket`, …) and the Phase 0B virtual
segmentation dimensions. Orderings, thresholds and windows come from the Phase 0B
`config/mi/risk_monitor.yaml` (extended here), keeping policy in config.

## 5. How it prepares future MI Agent runtime wiring

Each capability is a pure function (frame-in) plus a store-backed wrapper that
takes `(store, client_id, dimension, route, dates/selector, config)` and returns
a `RiskMonitorResult` (frame + issues + metadata). A future MI Agent runtime only
needs to parse a question into one of these calls and render the resulting frame
— no analytics logic moves into the agent. Route gating already mirrors the
contract the runtime will enforce.

## 6. How it differs from M&A diligence

The risk monitor is **recurring** (MI route): it depends on snapshot history for
migration, movement and trajectory. M&A diligence is **point-in-time** and the
risk monitor is disabled for the M&A route by contract (overridable only
explicitly). M&A reuses the *point-in-time* concentration view but not the
recurring migration/trajectory machinery.

---

## 7. Files changed

- `mi_agent/risk_monitor/__init__.py`, `models.py`, `migration.py`,
  `concentration.py`, `monitor.py` (new package).
- `config/mi/risk_monitor.yaml` (additive Phase 5 config sections).
- `tests/test_phase5_risk_monitor.py` (new, 23 tests).
- `docs/phase5_risk_monitor_foundations.md` (this file).

## 8. Tests run

`tests/test_phase5_risk_monitor.py` — **23 passed**. Combined
Phase 0B/1/2/3/4/5 + `mi_agent/` suites — **316 passed**.

## 9. Limitations / deferred items

- Concentration limit *enforcement* against real per-client limits is not built
  (only a documented example shape + RAG on group share).
- Trajectory uses a conservative monotonic-increase rule; no slope/forecast
  modelling.
- Migration balance for a transition uses current balance (baseline for exited
  loans); a dual baseline/current balance per cell is deferred.
- Not wired into the MI Agent runtime (by design).
