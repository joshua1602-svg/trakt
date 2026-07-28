# Duplicate calculation register

Verified by direct code inspection. Line references are to the commit under
review. "Prod" = production-reachable from the FastAPI layer.

## 0. Engine inventory

| Layer | File | Governed? | Prod-reachable |
|---|---|---|---|
| Shared workflow engine | `mi_workflows/engine.py` (CALCULATION_VERSION 1.1.0) | Yes — only module with mixed-currency guard + explicit unknown bucket | Yes via chat_routing prio 65/66 |
| Movement engine | `mi_agent/period_change/calculations.py` | Yes — own currency guard | Yes via chat_routing prio 85 |
| Point-in-time executor | `mi_agent/mi_query_executor.py` | No governance axes, no currency guard | Yes — default fallback path (`mi_service.py:471,550`) |
| Legacy stratifier | `analytics_lib/stratify.py` | No | Yes via `snapshots.py:490`, `risk_limits.py`, risk_monitor |
| Streamlit-era monitor | `analytics/risk_monitor.py` | No | No (Streamlit + legacy pptx only) |

There is no single shared calculation base. `mi_workflows/engine.py` claims to
be it but only 2 of the 12+ analytical routes consume it.

## 1. Sums / totals — 11 implementations

Only S1 (`engine.aggregate`, guarded caller-side) and S2
(`period_change/calculations._aggregate_numeric`, guarded via
`_currency_guard`) have any mixed-currency protection. The following
production-reachable sums add monetary values with NO currency guard:

* S3/S4 `mi_agent/mi_query_executor.py:277-290` — the default `/mi/query` path
* S5 `mi_agent_api/evolution.py:41`
* S6 `mi_agent_api/snapshots.py:76`
* S7 `mi_agent_api/movement_summary.py:137-140` (also nulls → 0 via fillna)
* S8 `analytics_lib/stratify.py:89-91`
* S9 `mi_agent_api/geo.py:171-179`
* S10 `mi_agent_api/cohorts.py:204,230`

`analytics/risk_monitor.py:33` keys on `current_principal_balance` while every
other module keys on `current_outstanding_balance` — two books of record
(legacy path feeds `generate_pptx_client.py`).

## 2. Averages — 7 implementations

A6 `analytics_lib/stratify.py:92` computes avg balance as balance_sum /
loan_count (row/nunique count), A7 `geo.py:191` similarly — diverges from
A1-A5 (sum/non-null-count) whenever balance is sparsely populated.

## 3. Weighted averages — 10 implementations, 4 distinct valid-mask policies

| Impl | Zero-weight behaviour |
|---|---|
| W1 `engine.py:160-172` | weights > 0 required; else None |
| W2 `period_change/calculations.py:295-347` | STATUS_ZERO_DENOMINATOR; "an unweighted average is not substituted" |
| W3 `mi_query_executor.py:295-303` | returns float("nan") — NaN leaks into result frame |
| W4 `analytics_lib/stratify.py:100-109` | **SILENT FALLBACK TO SIMPLE MEAN** (L109) |
| W5 `mi_agent_api/snapshots.py:53-68` | **SILENT FALLBACK TO SIMPLE MEAN** (L66-67, admitted in docstring) |
| W6/W7 `evolution.py:47-56` / `cohorts.py:71-78` | None (verbatim duplicates) |
| W8 `risk_limits.py:220-229` | None, rounds 2dp |
| W9 `analytics/risk_monitor.py:51-58` | raises ZeroDivisionError |
| W10 `geo.py:155-157,193-195` | None |

W4 is reached via `snapshots.py:518`, `mi_agent_pptx/chart_resolver.py:281`,
`mi_agent/states/temporal.py:267,340`. W5 is the `/mi/snapshot` KPI tile path.
Three mutually contradictory contracts for the same metric name, all live.

Weight-field resolution: `mi_query_executor.resolve_weight_field`
(`mi_query_executor.py:201-226`) has a 5-level fallback chain that can
silently weight by a different column than requested; engine/period_change
require the registry weight_field and fail closed.

## 4. Shares — 12 implementations, 3 incompatible denominator policies

Denominator drops unknown/null rows in:

1. `mi_agent_api/geo.py:161` — sharePct is share of geographically-resolved
   balance only (coveragePct returned alongside but shares not re-based).
2. `mi_query_executor.py:761-766` — `missing_policy="exclude"` silently
   re-bases concentration_pct/top_n/shares.
3. `analytics_lib/stratify.py:65-66` — `dropna=True` removes the unknown
   bucket from balance_share denominator.

Five incompatible unknown labels: `"(unknown)"` (engine), `"Unknown"`
(stratify, period_change models), `"Unknown / Missing"` (mi_query_executor,
evolution), `"Unattributed"` (movement_summary).

## 5. Distributions — 6 implementations (see agent notes; includes two
group-by variants inside `evolution.py` with different blank-token lists).

## 6. Movements — 5 engines, 4 relative-change formulas

* M1 `engine.compare_values` — (a−b)/abs(b), fraction
* M2 `period_change/calculations.metric_change` — movement/abs(start), fraction
* M3 `temporal_compare.compare_periods:107-148` — (delta/va)*100, SIGNED denominator
* M5 `states/temporal._pct_change` — (cur−base)/base*100, SIGNED denominator

With a negative base, M1/M2 vs M3/M5 return opposite-sign relative changes;
M3 also returns percent while M1/M2 return fractions.

Two independent balance bridges: `period_change/bridge.balance_bridge`
(loan-identity, currency-guarded, refuses on duplicate/missing keys) vs
`evolution.funded_bridge` (category delta, NO currency guard, no key checks).
Both live, answering "what drove the movement" with different arithmetic.

Materiality thresholds hardcoded in the presentation layer
(`movement_summary.py:66-77`: 0.6pp LTV stability, 0.5 completions
attribution share) drive stated direction and causal wording
("primarily driven by completions", `chat_routing.py:466-467`). Route prio 70
(`period_movement`, thresholded) fires before the governed period-change
route (prio 85).

## 7. Date resolution — 5+ implementations

Governed: `period_change/periods.py` on-or-before with gap guard (T1-T3).
Positional: `temporal_compare._match_period` (periods[-1]/[-2]),
`movement_summary.py:267`, `evolution.funded_bridge:324-333` — "prior" =
previous run in the list, not previous month; no gap guard. Both classes are
production-reachable for the same question class.

## 8. Currency handling

Three identical-but-untied copies of the currency field list
(`engine.py:63-64`, `period_change/calculations.py:102-103` — deliberate,
documented; `mi_agent_api/currency.py:25-26`). Two guards with different
signatures. `currency.resolve_currency_code:54-77` picks the MODAL currency
and formats cross-currency sums with the majority symbol — no guard.

## 9. Scope / lens resolution — two live filter mechanisms

`portfolio_lens.apply_lens` (L4, legacy: filters on portfolio_type string)
still the fallback at `portfolio_lens.py:247` when governed scope is None;
`apply_scope` (L5, governed: explicit portfolio_ids). A newly onboarded
portfolio joins the L5 answer and may not join the L4 answer. Plus a fourth
frame-level dict filter in `evolution._scope_frame_lens` used by
movement_summary.

## 10. Portfolio comparison — single implementation, routes through engine
primitives; keeps private math in `_summary` (favour tally) and
`_compare_distribution`.

## 11. Concentration — 4 parallel production stacks

CA-1 governed (`mi_workflows/concentration_analysis.py`, engine primitives,
currency guard, unknown kept in denominator, top-N = cumulative share capped
at 1−unknown_share) vs CA-2 risk_monitor + CA-3 risk_limits
(`analytics_lib.concentration.group_shares`: no currency guard, Unknown can
occupy a top-N slot, top-N = head(n) sum) vs CA-4 geo (private groupby,
resolved-rows denominator). Same book measured by different top-N definitions
depending on question wording.

## 12. Directionality — two registry-driven implementations with disjoint
output vocabularies (`engine.directionality_verdict` vs
`period_change.interpret`); plus up/down (`temporal_compare`), RAG
(`risk_limits`, `analytics_lib/concentration`).

## 13. Summary/prose — 7 builders; chat_routing PR5 asserts causality
("primarily driven by") gated only by a 0.5 hardcoded share;
`chat_routing.py:434-435,582-583` emit literal `coverage_by_balance_pct:
100.0` regardless of actual population.

## 14. Two independent BSR loaders

`mi_agent/business_semantics.py:361` (entry attr `.field`, applies
source-overrides overlay) vs `mi_workflows/semantics.py:163` (entry attr
`.source_field`, no overlay). Both production-reachable in the same request;
portfolio_risk_comparison/concentration see un-overridden metadata for fields
the period-change workflow sees overridden.

## 15. Priority register

| Rank | Finding | Impact |
|---|---|---|
| 1 | WA silently → simple mean (stratify:109, snapshots:66-67) | wrong number under a "weighted average" label, no flag |
| 2 | Monetary sums with no mixed-currency guard (9 of 11, incl. default /mi/query) | cross-currency addition |
| 3 | Share denominators drop unknowns (geo:161, executor:761, stratify:65) | overstated concentration |
| 4 | 4 relative-change formulas, 2 signed-denominator | opposite-sign answers on negative bases |
| 5 | Thresholded prose route (prio 70) outranks governed workflow (prio 85) | hardcoded materiality decides stated direction |
| 6 | 4 WA valid-mask policies | zero/negative-weight rows counted differently |
| 7 | 5 unknown-bucket labels | results cannot be reconciled |
| 8 | 2 BSR loaders, one applies overrides | governance metadata diverges in one request |
| 9 | 4 top-N concentration definitions | "top 5 = X%" depends on wording |
| 10 | principal vs outstanding balance as book of record (legacy deck path) | different totals |
