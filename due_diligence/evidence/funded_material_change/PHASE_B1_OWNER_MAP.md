# Sprint B — Phase B1 owner map

```
PRODUCT_BASELINE = dcb71a20
LIVE_MODEL_CALLS = 0   LIVE_MI_QUERY_CALLS = 0   DEPLOYMENTS = 0
```

## A correction to my own Phase 1 report

Phase 1 stated that risk/policy status transitions "require calling an existing
single-period owner **twice, once per snapshot, and diffing two governed
outputs**", because the limit and eligibility owners "each take one frame".

**That was wrong.** `concentration_tests.evaluation.evaluate_active_tests` takes
the PAIR and computes the transition itself:

```python
evaluate_active_tests(df, prior_df, config, lib, *,
                      reporting_date, prior_reporting_date, ...)
    """Returns the full governed envelope: per-test results
       (current, prior, movement, status transition), a summary,
       and the configuration provenance."""
```

and every test row already carries `status`, `priorStatus`, `statusTransition`,
`deteriorated`, `headroom`, `utilization`, `breachAmount`, `threshold`,
`warningFraction`, `unit`, `dataStatus`. The earlier claim came from reading
`evaluate_forward_states` and `risk_limits._headroom`, which are single-state,
and generalising from them without checking the pair evaluator.

**Consequence: Phase B4 needs no diffing layer at all** — it normalises an
existing envelope into `Insight`.

## The owner map

| Need | OWNER | STRUCTURED_INPUT | STRUCTURED_OUTPUT | PAIR? | RAW_Q? |
|---|---|---|---|---|---|
| FUNDED_BALANCE, LOAN_COUNT, WA_LTV, WA_RATE | `period_change.workflow.run_period_change_analysis` via `analyse_period_change` | client, root, mode, period_request, requested_fields/concepts, scope | `PeriodChangeResult.metric_changes` | **YES** | **NO** |
| PORTFOLIO_COMPOSITION | same owner | same call | `.distribution_changes` (`CategoryShift`, largest increases/decreases) | **YES** | **NO** |
| ATTRIBUTION / BRIDGE | same owner (`period_change.bridge.balance_bridge`) | `include_bridge=True` | `.balance_bridge` | **YES** | **NO** |
| CONCENTRATION / LIMIT / ELIGIBILITY | `concentration_tests.evaluation.evaluate_active_tests` | df, prior_df, ActiveConfiguration, ConcentrationLibrary, reporting dates, populations | per-test rows with `status`/`priorStatus`/`statusTransition`/`headroom`/`utilization` | **YES** | **NO** |
| MOVEMENT / PERIOD_CHANGE | as above | — | — | YES | NO |

`RAW_QUESTION_REQUIRED = NO` for every owner Sprint B uses. `analyse_period_change`
defaults `question=""` and the only read of `request.question` inside
`period_change/` echoes it into `to_dict()` for provenance.

```
SNAPSHOT_STORE_OWNER  period_change_route.build_snapshots
PERIOD_SELECTOR_OWNER period_change.periods.resolve_periods / PeriodRequest
COMPARABILITY_OWNER   resolve_periods (max_snapshot_gap_days,
                      flow_basis_tolerance_days, from SelectionPolicy)
MATERIALITY_POLICY_OWNER  insight_engine.rank_key / select
MATERIALITY_CONFIG_OWNER  insight_config over config/mi/insights.yaml
```

## What this means for the sprint

`MetricChange` already carries `start_value`, `end_value`, `movement_value`,
`movement_unit`, `relative_change`, `basis_point_change`, `directionality`,
`status`, `confidence` and `evidence` — which is Phase B5's required finding
evidence, already computed. `DistributionChange` is Phase B6. `balance_bridge`
is Phase B7. The concentration envelope is Phase B4.

**So Sprint B is normalisation plus thresholds, and contains no arithmetic.**
Two existing owners are invoked; their governed outputs are mapped into the
existing `Insight` contract; the existing `rank_key`/`select` orders and caps
them; the existing `insight_config` supplies every threshold.

```
ESTIMATED_PRODUCT_LOC = ~320
FILES_EXPECTED_TO_CHANGE = 4 new/changed:
  mi_agent_api/insight_funded.py        NEW   ~190  generators: normalise the two
                                               owners' outputs into Insight
  mi_agent_api/insight_contract.py      ~40   funded finding types + TYPE_PRIORITY
  config/mi/insights.yaml + insight_config  ~40  funded thresholds, existing loader
  mi_agent/plan_material_summary.py     NEW   ~90   thin plan -> composition adapter
NEW_COMPOSITION_FRAMEWORKS = 0
NEW_ANALYTICAL_CALCULATION_OWNERS = 0
```

Within the 400–500 budget, and no second engine, contract or threshold hierarchy.

## Perimeter I expect to have to declare

`evaluate_active_tests` needs an `ActiveConfiguration` and a
`ConcentrationLibrary`. The 135 bank measured that this book answers
borrowing-base questions with *"No funding facility is configured for this
portfolio"*, so concentration/limit configuration may be absent for the test
book. Status-transition findings will then be provable only on synthetic
fixtures (which Phase B10 expects) and will report `DATA_UNAVAILABLE` rather
than a finding on a book without that configuration. That is a configuration
fact, not a capability gap, and it must not be worked around.
