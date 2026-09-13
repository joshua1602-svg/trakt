# Post-mortem: separating instrument faults from product faults

```
RUN                 34752378644          DEPLOYED  88cf5fde…  (proved, source='artefact')
BANK                change_intelligence_serving_canary  27b7d825…  SPENT, unedited
DIAGNOSTIC RUN      34753888640          read-only sink projection

LIVE_MODEL_CALLS    0     LIVE_MI_QUERY_CALLS 0     DEPLOYMENTS 0
PRODUCT_CODE_CHANGED NO   CONFIG_CHANGES 0
SINK RECORDS READ   4 of 209, one per case, zero duplicates per question
```

## The corrected score

```
ORIGINAL 6 / 10        CORRECTED  8 PASS · 1 SAFE_CONFIG_REFUSAL · 1 PRODUCT_GAP
```

| | form | classification |
|---|---|---|
| S01 | material_summary | **PASS** |
| S02 | material_summary | **PASS** |
| S03 | metric_delta | **PRODUCT_GAP** |
| S04 | metric_delta | **PASS** |
| S05 | attribution | **PASS** |
| S06 | attribution | **PASS** |
| S07 | level_comparison | **PASS** (scored FAIL by instrument fault) |
| S08 | level_comparison | **PASS** (scored FAIL by instrument fault) |
| S09 | material_summary + Acquired | **SAFE_CONFIG_REFUSAL** |
| S10 | negative control | **PASS** |

## S07 / S08 — INSTRUMENT_ONLY, and the product did the right thing at every step

Both were served by the **NEW** path and every stage is correct on the record.

```
                                  S07                          S08
TIME_STATED       true                         true
TIME_FORM         relative_pair                relative_pair
LABELS            ["the current reporting      ["this reporting date",
                   date", "the date before      "the one before it"]
                   it"]
PLAN_OPERATION    compare                      compare
PLAN_PERIOD_FORM  relative_pair                relative_pair
PLAN_PERIOD       stated=true, defaulted=false, periods_back=1,
                  contract=governed_reporting_period_pair, resolved=true
EXECUTION_RUNTIME plan_temporal_runtime (perimeter slice2_temporal, eligible)
CALCULATION_OWNER mi_agent.mi_query_executor.execute_mi_query, once per snapshot
SERVING           decision=NEW, principal_matched=true, reconciled=true
```

**The raw model payload states the pair too** — `time` is present in the payload
with `form: relative_pair` and both labels — so presence is a fact of what the
model emitted, not an artefact of the dataclass.

**What the two groups actually are.** Both selected the same pair, by
`selector_mode = last_n` over the governed catalogue:

```
                  baseline                    current
snapshot          ERE/2025-11-30              ERE/2026-06-30
S07 balance sum   8,903,225.07                159,097,304.07
S08 loan count    73                          958
```

These are the two **adjacent governed snapshots** — 2026-06-30 is the latest the
catalogue holds and 2025-11-30 is the one before it. So yes: the served answer
genuinely answered the current reporting state versus the immediately previous
governed reporting state. That the interval is seven calendar months is the
book's reporting cadence, not a substitution; `current_vs_previous` is defined
over adjacent governed snapshots "whatever cadence the book reports on".

```
S07_CLASSIFICATION = INSTRUMENT_ONLY
S08_CLASSIFICATION = INSTRUMENT_ONLY
```

*(The 13x movement in both figures is a data-content observation about ERE's book
between those two snapshots, not a serving finding. Both executions reconciled.)*

## S09 — SAFE_CONFIG_REFUSAL, and the adapter is exonerated

The trace runs cleanly through every stage this sprint built, and fails at the
period resolver on a true fact about the data.

```
REQUESTED_SCOPE          lens = acquired
PLAN_SCOPE               population.lens = "acquired"
                         scope_predicates = [source_portfolio_type eq "acquired"]
SOURCE_REGISTRY_INPUT    the acquired ROLE, resolved through
                         contract_scope.lens_from_plan -> portfolio_lens
RESOLVED_PORTFOLIO_IDS   NON-EMPTY — proved by which check fired (below)
AUTHORISED_PORTFOLIO_IDS () — none supplied, exactly as the legacy route supplies
                         none; the tenancy boundary is client_id
ELIGIBILITY              eligible = true, perimeter = change_form_material_summary
EXECUTION_RUNTIME        material_summary  (attempted = true)
EXECUTION_ERROR_TYPE     PeriodChangeFailure
EXECUTION_ERROR_DETAIL   "The requested portfolio does not exist at both reporting
                          dates, so the two periods cannot be compared."
                         = FAIL_PORTFOLIO_ABSENT_AT_PERIOD
```

**The adapter claimed it, accepted it, and called the owner.** `eligible = true` at
`perimeter = change_form_material_summary` is the new arrow working. The refusal
came from `period_change.periods.resolve_periods` line 443, *downstream* of a lens
that resolved and was applied.

**That check proves the registry resolved the identity.** It reads
`wanted = tuple(scope.portfolio_ids)` and raises only when a wanted id is missing
from a snapshot. An unresolved role would have produced an empty tuple and no
raise at all — it would have failed earlier and differently, as `LensNotApplied`.
So:

> **Does ERE's governed production configuration contain an Acquired portfolio
> identity that this runtime should resolve? YES — and it did resolve it.** The
> acquired portfolio is absent from one of the two adjacent governed snapshots
> (2025-11-30 / 2026-06-30), so no governed pair exists for that book.

```
S09_ROOT_CAUSE = CLIENT_CONFIG_GAP
```

Data-coverage, specifically: the acquired book does not span both governed
snapshots. Not `SOURCE_REGISTRY_GAP` (it resolved), not `ADAPTER_DEFECT` (the
adapter claimed, passed its perimeter and invoked the owner), not
`AUTHORISATION_SCOPE_GAP`.

**The refusal is correct and must stay.** Comparing a portfolio that exists at one
end and not the other is not a comparison. No fallback, and no whole-book
widening. The offline control A4 passes because its fixture carries the acquired
portfolio at both dates; production does not, and the difference is the data.

*Which end it is missing at is in the exception's `detail` (`missing_at_start` /
`missing_at_end`), which the sink truncated to the message. Reading it needs the
snapshots, not another live call.*

## S03 — PRODUCT_GAP: metric_delta has no governed plan connection

The interpretation is flawless and the plan is complete.

```
CHANGE_FORM   metric_delta      (claimed AND compiled; binding mode = requested_metric)
CAPABILITY    period_movement
OPERATION     movement
MEASURE       current_outstanding_balance, statistic sum (defaulted)
PERIOD        relative_pair, stated=true, periods_back=1, resolved=true
PLAN_ID       plan_6d99df55e11d4dcd
```

Dispatch trace:

```
_change_form_owner(plan)          -> None      no adapter claims metric_delta
plan_temporal_runtime.claims      -> True      it reads period.form alone
check_temporal_eligibility        -> REFUSED   CAPABILITY_NOT_GENERIC
                                   "capability='period_movement' is specialist;
                                    it owns its own input contract and arithmetic"
new_path_eligible                 -> false
response_served_from              -> LEGACY_FALLBACK
legacy_ok                         -> false     the legacy route ALSO refused
```

This is byte-for-byte the refusal traced offline in Phase 1 of the sprint, now
observed in production.

```
S03_METRIC_DELTA_CONNECTIVITY_GAP = YES
```

**The existing calculation owner** is
`period_change.workflow.run_period_change_analysis` in `MODE_REQUESTED_METRIC` —
named by `vocabulary.CHANGE_FORM_MODE["metric_delta"] = "requested_metric"` — with
the plan's measure as `requested_fields`.

**The thinnest existing seam** is the one already built for the other two forms:
a `plan_metric_delta` adapter mirroring `plan_attribution`, reusing
`plan_material_summary`'s shared `analyse()`, `check_slots_honoured()`,
`check_period()`, `change_receipt()` and `governed_envelope()`, and registered in
`plan_serving_canary._change_form_owner`. No new calculation owner, no new
renderer, no new perimeter. **NOT IMPLEMENTED — this is a finding, not a change.**

### Why S04 succeeded where S03 refused

Both took the same plan-path refusal (`CAPABILITY_NOT_GENERIC`) and both fell back
to legacy. They differ only in what LEGACY then did:

```
S04  route = period_change_analysis   = WORKFLOW_ID = ROUTE_NAME -> the SUCCESS envelope
S03  route = period_change            -> one of the four REFUSAL/clarification envelopes
```

`period_change_route` hardcodes `route="period_change"` in exactly four places —
`LensNotApplied`, `PopulationNotApplied`, the "names neither a subject nor a
period" clarification, and the span clarification — and uses `ROUTE_NAME` only for
success. **Which of the four fired for S03 is not in the sink**, because the sink
records the PLAN path and captured only `legacy_ok: false`. That leg is
INCONCLUSIVE and is legacy-estate behaviour outside this sprint.

**Does either path reread the raw question?** The plan path: **no** — asserted
structurally by the offline controls. The legacy path: **yes, by construction** —
`route_period_change` is driven by `recognise_request(req.question)`. That is the
estate this architecture exists to replace, and S03 is a clean example of why.

## Scorer post-mortem

```
SCORER_ROOT_CAUSE
```

Two defects, one masking the other.

1. **The primary fault.** `_temporal_route` treats any truthy
   `metadata.governedPlan.executed` block as a change-form receipt:

   ```python
   if receipt:
       if not receipt.get("interpreted_time_present"):
           return "ABSENT"
   ```

   But `plan_serving_canary.render()` — the slice 1 and slice 2 renderer — builds
   that block from the EXECUTOR's receipt: `applied_predicates`,
   `group_field_keys`, `aggregation`, `balance_field_used`,
   `percent_scale_detected`, `filtered_row_count`, plus `population_base`. It
   carries no temporal keys at all. So for a slice 2 answer the block is truthy,
   `interpreted_time_present` is absent, `.get()` returns None, and the scorer
   returns `ABSENT` **without ever reading the CandidateIntent**. Only the
   change-form receipt has those keys, so the branch is sound only for the two
   forms this sprint connected.

2. **The fault behind it**, which would have produced the same wrong answer via
   the other branch. The fallback tests key presence:

   ```python
   if not isinstance(time_block, dict) or "form" not in time_block: return "ABSENT"
   ```

   `CandidateIntent.to_dict()` is a dataclass `asdict`, so `time` is ALWAYS
   present with a `form` key and presence is carried by `stated` — measured:
   `{"form": "current", …, "stated": false}` for a reading that stated nothing.
   Key presence is the right read for the model's RAW payload and the wrong one
   for the parsed intent. This is the very distinction the temporal presence
   repair exists to make, misapplied by the instrument built to measure it.

```
MINIMUM_CORRECTION
```

- Detect a change-form receipt by **key membership**, not truthiness:
  `if "interpreted_time_present" in receipt:` — then trust it.
- Otherwise read the parsed intent's **`time.stated`** as the owner of presence,
  and the form only when `stated` is true.
- Where the raw payload is available, prefer it for presence, since key presence
  there is authoritative and carries no construction default.

```
CANARY_SCORER_DEFECT_CONFIRMED = YES
```

Not applied here, and the spent canary is not re-scored or re-run. The correction
belongs in future test infrastructure only.

## Integrity

```
product code            unchanged — PRODUCT_FILES_CHANGED = 0
serving-canary-run.yml  not modified, not triggered
the spent bank          read for question identity only; hash asserted in CI;
                        no expectation read, nothing re-scored
credentials             scrubbed; the projection was rescanned before writing
loan-level data         redacted by construction; group keys, period keys and
                        aggregates retained, which is what a comparison IS
```
