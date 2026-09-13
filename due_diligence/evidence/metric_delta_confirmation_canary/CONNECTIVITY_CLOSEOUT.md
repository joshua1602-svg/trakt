# metric_delta connectivity closeout — the last missing arrow

```
PRODUCT_BASELINE  88cf5fde759c29f1cff29b9adbf003a2d6ebd530
PRODUCT_HEAD      (this commit)

METRIC_DELTA_CONNECTED  YES
CALCULATION_OWNER       mi_agent.period_change.workflow.run_period_change_analysis
MODE                    requested_metric   (read from vocabulary.CHANGE_FORM_MODE)

NEW_ANALYTICAL_CALCULATION_OWNERS = 0
RAW_QUESTION_READS_ADDED          = 0
NEW_FRAMEWORKS                    = 0

LIVE_MODEL_CALLS 0   LIVE_MI_QUERY_CALLS 0   DEPLOYMENTS 0   CONFIG_CHANGES 0
```

## What was connected, and to what

The live canary's S03 read the question perfectly — `metric_delta`,
`period_movement`/`movement`, `current_outstanding_balance`, `relative_pair` with
the pair stated, compiled to a plan — and served nothing, because no adapter
claimed the form and `plan_temporal_runtime` claimed it on period form alone and
refused `CAPABILITY_NOT_GENERIC`. `mi_agent/plan_metric_delta.py` closes that and
nothing else.

```
PRODUCT_FILES_CHANGED  3   (1 added, 2 modified)
PRODUCT_LOC_ADDED      161 executable      PRODUCT_LOC_REMOVED  3
    plan_metric_delta.py        +151   the adapter that did not exist
    plan_material_summary.py    +8/-2  `admit_anchor` on the shared period
                                       check; `requested_fields` through the
                                       shared one call to the owner
    plan_serving_canary.py      +2/-1  the import and the dispatch tuple
```

**161 against a ~150 guideline — 7% over, stated rather than rounded down.** The
overage is entirely inside the adapter, and the two blocks that carry it are the
post-execution reconciliation (~30 lines) and the receipt (~20). Neither is
trimmable: the first is what stops a partial or substituted answer being served
and controls D8c/D8d prove it fires, and the second is the evidence contract the
other two forms already meet.

## Three differences from the other two adapters, each forced by the owner

1. **It names a measure.** `requested_metric` mode analyses `requested_fields` —
   BSR registry FIELD names — so the plan's `canonical_field` bindings are passed
   and no concept is resolved twice. A measure with no field (the row itself, a
   capability-owned quantity) has nothing to select and is REFUSED.
2. **It admits no anchor.** `CHANGE_FORM_ABSENT_PERIOD_DEFAULT` authorises a
   default for `material_summary` and `attribution` and withholds one here.
   Admitting a lone `current` would invent the comparison state that table
   deliberately withholds — so `check_period` gained `admit_anchor`, and this
   form passes `False`.
3. **It reconciles the statistic after the fact.** `calculations` reads
   `entry.default_aggregation` and follows it exactly; the owner takes no
   caller-supplied aggregation. So a statistic the READER stated is checked
   against the one applied, and a divergence refuses. Where the compiler
   defaulted it there is no reader intent to violate and the registry stands.

## The measure perimeter, measured rather than assumed

| measure | statistic | bound to the owner |
|---|---|---|
| `current_outstanding_balance` | sum | yes |
| `current_principal_balance` | sum | yes |
| `current_loan_to_value` | weighted_average | yes |
| `indexed_loan_to_value` | weighted_average | yes |
| `current_interest_rate` | weighted_average | yes |
| `arrears_balance` | sum | yes |
| `loan` | count | **no — MEASURE_HAS_NO_GOVERNED_FIELD** |

Loan-count movement is not supported and is refused, not approximated: the row
itself carries no governed field and this owner selects registry fields.

## Two refusals that happen EARLIER than the adapter, found while proving it

Both were assumptions of mine that the evidence corrected, and both are better
answers than the adapter's:

- `metric_delta` + a lone `current` is **`UNSUPPORTED_COMPOSITION` at the
  compiler** — no plan is ever built. The adapter's `PERIOD_NOT_A_PAIR` is
  defence in depth for a plan built without that seam, and is asserted on a
  hand-built one.
- `current_loan_to_value` + `sum` is **`UNSUPPORTED_STATISTIC` at the compiler** —
  the concept publishes its allowed statistics. The adapter's statistic
  reconciliation likewise becomes defence in depth, asserted directly.

## Offline controls

```
METRIC_DELTA_OFFLINE_CONTROLS = 65 / 65
    (tests/interpretation_v2/test_change_intelligence_serving.py — the 41 from
     the connectivity sprint plus 24 for this closeout)
```

| | control | result |
|---|---|---|
| D1 | balance / current LTV / interest rate → requested_metric owner | PASS ×3 |
| D2 | the measure perimeter matrix, 7 measures | PASS ×7 |
| D3 | a measure with no governed field → REFUSE | PASS |
| D4 | absent window → governed refusal at the compiler | PASS |
| D4b | a lone anchor → refused at compiler AND adapter | PASS |
| D8 | filter / dimension / population slots → REFUSE, never widen | PASS ×3 |
| D8b | a statistic the vocabulary forbids never reaches an owner | PASS |
| D8c | an explicit statistic the owner contradicts → REFUSE after execution | PASS |
| D8d | a field the owner excluded is never served as the answer | PASS |
| D9 | the adapter reads no question | PASS |
| C10 | *migrated* — metric_delta now reaches its governed owner | PASS |
| — | the three connected forms stay disjoint; one plan, one owner | PASS |

```
OWNER_PARITY = 3 / 3      NUMERICAL 0 · PERIOD 0 · SCOPE 0 · OWNER 0
```

For each metric delta the served answer is compared against
`run_period_change_analysis` invoked independently in `requested_metric` mode,
with a `PeriodRequest` the test states itself: the published "Metric movements"
table against the owner's own `_metric_rows`, and every `start_value`,
`end_value`, `movement_value` and `aggregation` in the receipt against the
owner's own `MetricChange` objects.

## Regression

| gate | before | after |
|---|---|---|
| B10 | 51 / 51 | **51 / 51** |
| B11 | 20 / 20 | **20 / 20** |
| Weekly Brief | 90 / 90 | **90 / 90** |
| period_change | 335 + 5 subtests | **335 + 5 subtests** |
| completeness + current-anchor + presence | 99 / 99 | **99 / 99** |
| change_form contract + semantic bank | 65 / 65 | **65 / 65** |
| serving canary unit suite | 60 / 60 | **60 / 60** |
| MATERIAL_SUMMARY_REGRESSION | — | **unchanged**, A1–A5 + parity green |
| ATTRIBUTION_REGRESSION | — | **unchanged**, B6–B9 + parity green |
| LEVEL_COMPARISON_REGRESSION | — | **unchanged**, C11 green; no adapter claims it |
| `tests/interpretation_v2/` + `mi_agent_api/tests/` | 12 FAILED ids | **identical 12**, +58 passes |
| `mi_agent/tests/` | 16 FAILED ids | **identical 16**, unchanged |

```
NEW_UNEXPLAINED_FAILURES = 0
```

## Scorer correction — test instrument only

```
SCORER_TEMPORAL_READ_FIXED = YES
PRODUCT_SEMANTICS_CHANGED_BY_SCORER_FIX = NO
```

`_temporal_route` now tests **key membership** — `"interpreted_time_present" in
receipt` — instead of the truthiness of a block that the slice 1 and slice 2
renderer fills with the EXECUTOR's receipt and no temporal keys at all; and its
fallback reads **`time.stated`** instead of key presence in a dataclass `asdict`
where `time` always exists. The raw payload is recorded alongside as
corroboration, since key presence there is authoritative, but the parsed intent's
`stated` remains the governed signal. Self-tested offline on five cases including
both directions of the S07/S08 fault. **The spent canary is not re-scored and not
re-run.**

```
S09_BEHAVIOUR_CHANGED = NO
```

Untouched. No zero baseline, no backfill, no widening. Absence from a historical
governed snapshot is data history and the fail-closed refusal stands.

## The fresh bank — pre-registered, NOT RUN

```
FRESH_CANARY_PREPARED  YES
CANARY_BANK_ID         metric_delta_confirmation_canary
CANARY_SHA256          9705705c7c66a605ccd3f16a08275b33f1b05b201c5f09f99bcd24386eb7d065
CANARY_QUESTIONS       5
RUN                    NOT RUN — no runner and no workflow exist for it
```

```
M01  metric_delta      balance          EXPLICIT_PAIR   governed_plan_metric_delta
M02  metric_delta      WA current LTV   EXPLICIT_PAIR   governed_plan_metric_delta
M03  metric_delta      WA interest rate EXPLICIT_PAIR   governed_plan_metric_delta
M04  material_summary  (none)           OWNER_DEFAULT   governed_plan_material_summary
M05  attribution       bridge target    EXPLICIT_PAIR   governed_plan_attribution
```

Every case pins the ROUTE **and** the CALCULATION OWNER, not merely the reading —
because S03 proved a correct reading that reaches no owner is the failure mode.
No wording from any of the nine earlier banks is reused: verified against 119
historical questions, zero exact matches, maximum token overlap 0.53.

**Two deliberate omissions, both measured rather than assumed.** No S09 retest,
because the acquired portfolio still does not exist in both selected snapshots
and the instruction admits it only on separate factual evidence. No loan count,
because the adapter correctly refuses it — pinning a question the architecture
declines would repeat the v4/W06 authoring defect in a new costume.

## Integrity

```
all six bank manifests hash to their own pins
  v1 · v2 · v3 · v4 · change_intelligence_serving_canary (SPENT) ·
  metric_delta_confirmation_canary (this pre-registration)
serving-canary-run.yml   not modified, not triggered
the spent bank            not edited, not re-scored, not re-run
```
