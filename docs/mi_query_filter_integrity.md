# MI `/query` Filter Integrity — Fail-Closed Filter Invariant

Extends the fail-closed guarantees from dimensions to **value filters**.

## The issue

The dimension invariant prevented silent dimension drops, but filters were not
covered equivalently. A grouped query + value filter (e.g. *"balance by region
where LTV above 50%"*) parsed the grouping but **silently dropped the filter** —
returning grouped results with `filters_applied=false` and all records included.

## Root cause

Execution **already supported** filter + grouping (the executor applies the mask
before grouping — verified: 156/400 records then grouped). The gap was purely in
the **parser**: the grouped bar / heatmap / treemap / ranking spec builders never
called `_parse_filters`, so a filter phrase alongside a grouping was never
attached to the spec. Per the "do not broaden claims unless execution supports
them" rule, the correct fix was to make the parser attach the filter (execution
genuinely honours it), backed by a fail-closed invariant.

## Changes

1. **Contract** (`mi_agent/mi_query_contract.py`) — `check_filter_invariant` +
   `FilterInvariantResult` with the required fields:
   `parsed_filters`, `applied_filters`, `rejected_filters`, `unavailable_filters`,
   `filters_applied`, and `ok`. A parsed filter is *applied* when its field key
   is in the reconciliation's applied-`filters` map with `filters_applied` true;
   *unavailable* when its field is absent (surfaced on `spec.unavailable_filters`);
   *rejected* when recorded in `rejected_filters` metadata. A parsed filter that
   is none of these is a **silent omission** → `ok=False`. `build_query_trace`
   gains a `filterInvariant` block.

2. **Parser** (`mi_agent/llm_query_parser.py`) — `_grouped_value_filters` extracts
   value filters (numeric, range, and categorical borrower-structure) on the
   grouped paths, excluding any filter whose field is itself the grouping
   dimension. Wired into the bar, two-dimension (heatmap), treemap and ranking
   spec builders. No parser claim is broadened beyond what execution applies.

3. **Workflow** (`mi_agent/mi_agent_workflow.py`) — a fail-closed **filter
   invariant guard** after execution: if a parsed filter was not applied, the
   query is refused (`ok=False`, validation error, reason in `warnings`/`error`)
   rather than returning unfiltered data. `filter_invariant` is attached to the
   result and passed into the trace. `rejected_filters` are surfaced as warnings.

4. **API** (`mi_agent_api/adapters.py`) — the response now carries
   `filterInvariant` (top level) and `queryTrace.filterInvariant`. *(Frontend
   Query Logic wiring is pending the trace being consumed by the client — see the
   dimension-integrity review's open items.)*

## Behaviour now

| Query | Filter | Applied |
|---|---|---|
| `how many loans have LTV above 50%` | `current_loan_to_value > 50` | ✅ (ungrouped KPI) |
| `balance by region where LTV above 50%` | `current_loan_to_value > 50` | ✅ (grouped) |
| `balance by broker where LTV between 40 and 60` | range | ✅ (grouped) |
| `balance by broker for joint borrowers` | `borrower_type = Joint` | ✅ (categorical) |
| `top 5 regions by balance where LTV above 50%` | `current_loan_to_value > 50` | ✅ (ranked) |
| `how many loans have Risk Grade above 700` | field absent | ⛔ refused / surfaced |
| parsed filter not applied by execution | — | ⛔ query refused (fail-closed) |

## Tests

`mi_agent/tests/test_mi_query_invariants.py` (all offline / deterministic):

- `test_filter_simple_filtered_kpi` — ungrouped filtered KPI applies the filter.
- `test_filter_grouped_categorical` — grouping + categorical filter both survive.
- `test_filter_grouped_numeric_range` — grouping + range filter both survive.
- `test_filter_unsupported_shape_is_refused` — absent-field filter refused/surfaced.
- `test_no_silent_fall_through_to_unfiltered_data` — a simulated unapplied filter
  fails the invariant closed.
- `test_both_invariants_hold_across_the_suite`,
  `test_no_silent_filter_omission`, `test_grouped_filter_now_supported` — the
  generative harness (249 cases) holds both invariants; filters exercised on 31.

The calibration report (`docs/mi_query_calibration_report.md`) now reports the
**dimension** and **filter** invariants separately.

## Commands

```bash
python -m pytest mi_agent/tests/test_mi_query_invariants.py -q
python scripts/mi_query_calibration.py     # regenerate the report
python -m pytest mi_agent -q               # full parser/executor/workflow regression
```

## Residual notes

- The filter invariant relies on the executor reconciliation's `filters` /
  `filters_applied` block, present on every filtered path (summary, table,
  loan-level, grouped).
- Time-series (`line`) queries now attach value filters too — a filtered trend
  ("balance by month where LTV above 50%") applies the filter to the mask before
  the trend is built (see `docs/mi_calibration_bank.md`, Priority-1 fixes).
- Frontend does not yet render `filterInvariant` / `queryTrace` — surfacing them
  in the Query Logic disclosure is tracked in the dimension-integrity review.
