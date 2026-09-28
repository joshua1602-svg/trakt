# change_form completeness repair — offline result

```
PRODUCT_BASELINE  665176864d2f5dbceedc277e351c5d2cd978777f
PRODUCT_HEAD      620e95ba
LIVE_MODEL_CALLS  0     DEPLOYMENTS 0     LIVE_MI_QUERY_CALLS 0
```

## Root cause

The v1 live gate recorded one reading (CF07) that named its measure, its
`weighted_average` statistic and its adjacent-period pair **correctly**, left
`change_form` null, and still compiled to an executable PLAN — because the legacy
`(capability, operation)` pair was sufficient on its own.

The plan that came out was indistinguishable from a change-form plan, while
nothing in it said *which* analytical question had been asked, and therefore which
governed mode and which candidate-set owner applied. `CandidateIntent` is
probabilistic; `GovernedQueryPlan` was accepting semantic incompleteness.

## The rule

`change_form` is REQUIRED when all four hold, read from structured slots only:

| | condition | source |
|---|---|---|
| 1 | `capability` is one the change-form family owns | `CHANGE_FORM_CAPABILITY.values()` — the existing table, not a new list |
| 2 | `operation` states a cross-state difference: `movement`, `bridge`, `compare` | the actions the four forms use |
| 3 | `comparison.kind == "none"` | a population- or dimension-pair comparison is not a change between reporting states |
| 4 | the period denotes more than one governed reporting state, **or** the operation's window is capability-owned | `_CAPABILITY_OWNED_PERIOD_OPERATIONS`, reused |

`summary` is deliberately **absent** from condition 2 even though
`material_summary` canonicalises to it: `concentration`, `pipeline`,
`limit_assessment` and `portfolio_summary` all admit `summary` as a point-in-time
shape, so requiring a form there would refuse ordinary MI that was never about a
change.

`change_form` legitimately remains absent for every non-change request — and,
importantly, for every other family that is change-shaped in its own right
(`pipeline_stage_movement` transitions, `borrowing_base` movements, `forecast`
projections). None of those is one of the four forms, so none may be asked for a
value it has none of. Condition 1 is what guarantees that.

### Measured, not asserted

Sweeping the whole governed `(capability, operation, time.form)` space:

```
19   combinations change behaviour   — all inside the three owned capabilities
148  combinations keep planning      — every other family, untouched
```

## What it does not do

It does **not** infer the missing form. There is no rule from "movement + a period
pair + a metric" to `metric_delta`; that would put semantic inference back in the
compiler and invert the authority model `change_form` exists to state.
`test_the_gate_never_names_one_of_the_four_forms_as_a_suggestion` asserts the
clarification does not even propose one.

`MISSING_REQUIRED_SLOT` already existed and is already in `CLARIFIABLE_CODES`, so
**no reason code, outcome or enum was added**. `change_form` was **not** added to
the JSON schema's `required` array either — it must stay absent for non-change
questions, so the obligation is conditional and deterministic rather than
structural. `test_the_slot_is_optional_so_nothing_recorded_before_it_breaks` still
passes.

## Model contract

One clarification, framing only. `"ONLY when the question is about a CHANGE…"` and
`"Leave empty when the question is not about a change"` became **"REQUIRED
whenever the question is about a CHANGE between reporting states: state the
analytical form the reader asked for, and do not omit it"**, plus the consequence
that an incomplete change request will not be executed. The four form descriptions
are untouched. No example, no phrase list, no regex, no metric name, no benchmark
id.

## Regression — reported, not papered over

**Nineteen pre-existing tests now fail, and the gate is proved to be the sole
cause**: neutralising only `requires_change_form` makes every one of them pass
again. Each asserts the superseded contract — that a change-oriented intent with
no form still plans.

| suite | with gate | gate neutralised | attributable |
|---|---|---|---|
| `tests/interpretation_v2/` + `mi_agent_api/tests/` | 25 FAILED ids | 12 | **13** |
| `mi_agent/tests/` | 22 FAILED ids | 16 | **6** |

The thirteen:

```
test_change_form_contract.py::test_a_plan_with_no_form_records_no_form_binding
test_contract_normalisation.py  (8 tests)
test_corrections_from_the_first_live_run.py::test_a_period_on_period_movement_needs_no_comparison_flag
test_kl_multi_output_and_specialists.py::…[funded_bridge-bridge-funded_balance_movement]
test_slice3_portfolio_affordance.py::test_proof_2_the_signed_off_135_replays_identically
test_slice3_portfolio_affordance.py::test_proof_7_attribution_stays_separate_from_generic_analysis
```

The six: all in `test_plan_temporal_runtime.py`, every one a period-comparison
fixture that omits the form.

Two deserve naming:

- **`test_proof_2_the_signed_off_135_replays_identically`** — ten of the 135
  signed-off cases move PLAN → CLARIFY: **Q18A/B/C, Q19A/B** and the rest of that
  family. These are exactly the nine the B11 evidence already recorded as
  recording `capability=period_movement, operation=movement` with no form, because
  *"it predates the semantic"*. They are incomplete under the new contract.
- **`test_a_plan_with_no_form_records_no_form_binding`** — its `_BASE` fixture is
  `generic_analysis + movement + relative_pair + balance`, i.e. **exactly the CF07
  shape**, and it asserts `result.plan is not None`. That is the literal negation
  of the invariant this sprint was commissioned to establish.

**None of the nineteen has been modified.** Updating a signed-off invariance guard,
or the 135 corpus it replays, is a governance decision and not an implementation
detail. They are listed for a decision rather than quietly made green.

### Protected suites hold

| suite | result |
|---|---|
| Weekly Portfolio Brief | **90 / 90** |
| B10 `test_insight_funded.py` | **51 / 51** |
| B11 `test_change_intelligence_target_controls.py` | **20 / 20** |
| `period_change` (mi_agent + mi_agent_api) | **336 / 336** |
| change-form contract + semantic bank | 63 / 64 (the CF07-shaped test above) |
| **new** completeness controls | **45 / 45** |

## Temporal owner check — read-only, both UNSAFE, not repaired

```
MATERIAL_SUMMARY_CURRENT_ANCHOR_SAFE = NO
ATTRIBUTION_CURRENT_ANCHOR_SAFE      = NO
```

Proved by execution, not by reading:

- **material_summary at `current`** compiles to PLAN with
  `owned_by_capability=False`; `plan_material_summary.claims()` is True and
  `check_eligibility()` then returns **`PERIOD_NOT_A_PAIR`**. `current` is
  deliberately excluded from `PAIR_PERIOD_FORMS` — *"`current` names one period
  and `series` names many; neither is a pair, and neither is silently turned into
  one."* So `current` does **not** resolve to `current_vs_previous`. The control
  at `relative_pair` does: `PeriodRequest(relative_mode='current_vs_previous')`.
- **attribution at `current`** compiles with `owned_by_capability=True` — and
  there is **no governed-plan runtime for `funded_bridge` at all**
  (`plan_temporal_runtime` covers `{point_in_time, breakdown, series, compare}`;
  `plan_material_summary` covers `{summary}`). Nothing keeps that promise, so
  there is no FROM/TO to recover.

This is an **independent temporal-contract defect**: the compiler emits an
executable plan whose period the execution owner then refuses, or for which no
execution owner exists. It is reported here and deliberately **not** repaired — no
temporal semantics were changed, no `period_pair_owner`, no new temporal enum, no
CF05/CF11-specific logic.

It is also why **v2 records the temporal reading and does not gate on it**: gating
on a contract just shown to be defective would make a temporal failure read as a
`change_form` failure, which is the conflation this sprint was told to avoid.

## v2 — pre-registered, NOT run

```
BANK_ID      live_change_form_gate_v2
BANK_SHA256  b9a842c6115ccf16f3091da22615a35c4e978b48663363519d01154a0abad1e4
QUESTIONS    18   (material_summary 4, metric_delta 6, attribution 4, level_comparison 4)
```

Fresh paraphrases throughout. Six metric deltas over **five distinct governed
quantities** — outstanding balance, live loan count, arrears balance,
weighted-average current LTV, weighted-average indexed LTV — so a pass cannot come
from having learned the one metric v1 missed. Scope-bearing: V03 acquired, V10
direct, V14 acquired, V18 direct.

v1 is left **byte-identical**; it remains immutable historical evidence at 17/18.
The v2 runner also persists the verbatim `raw_payload`, the one gap v1 had, and
carries each case's compile outcome — with the gate in place, a PLAN proves the
reading was complete and a CLARIFY naming `change_form` proves the gate caught an
incomplete one.

Harness proved at zero cost: 7/7 adjudicated, 7/7 compiled, the four forms
reaching `period_movement/summary`, `period_movement/movement`,
`funded_bridge/bridge`, `generic_analysis/compare`.

**No live call has been made against v2.**
