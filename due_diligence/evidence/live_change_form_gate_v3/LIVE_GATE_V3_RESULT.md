# Live change_form interpretation gate v3 — FAIL, with the primary gate PASSED

```
PRODUCT_HEAD                    10fa8bc0   (approved 767239c6 + this gate's workflow only)
PRODUCT_CODE_CHANGED_DURING_RUN NO         (asserted in CI, not claimed)
BASELINE                        665176864d2f5dbceedc277e351c5d2cd978777f
BANK_ID                         live_change_form_gate_v3
BANK_SHA256                     2ff59ff12277c66d1814ec496945a338d69fc9ea97098195915c626e65060a98
MODEL                           claude-opus-5   (requested AND served, all 18)
RUN                             https://github.com/joshua1602-svg/trakt/actions/runs/34730778202

LIVE_MODEL_CALLS_ATTEMPTED      18
LIVE_MODEL_CALLS_SUCCESSFUL     18
PROVIDER_FAILURES               0
LIVE_MI_QUERY_CALLS             0     DEPLOYMENTS 0
```

## The primary semantic gate PASSED

```
CHANGE_FORM_TOTAL = 18 / 18        ok

MATERIAL_SUMMARY  = 4 / 4          ok
METRIC_DELTA      = 6 / 6          ok
ATTRIBUTION       = 4 / 4          ok
LEVEL_COMPARISON  = 4 / 4          ok
```

Every one of eighteen fresh paraphrases reached the right analytical form. **The v1
defect is closed**: CF07's weighted-average-LTV shape is V08 here and it now
carries `metric_delta`, as do five further metric deltas over five distinct
governed quantities — outstanding balance, loan count, arrears balance, current
LTV, indexed LTV. A pass could not have come from one learned metric.

**The current-anchor repair also works live.** V12 anchored at `current` for an
attribution and scored `ANCHOR_ACCEPTED`. Under v1's policy that case would have
been a temporal failure; under the closed contract it is a valid reading, and the
deterministic owner supplies the comparison state.

## The secondary gates

| metric | result | required | |
|---|---|---|---|
| EXPLICIT_MEASURE_PRESERVED | **13 / 14** | 14 | **FAIL** |
| MATERIAL_SUMMARY_SPECIFIC_MEASURE_INVENTED | 0 / 4 | 0 | ok |
| SCOPE_PRESERVED | 4 / 4 | 4 | ok |
| UNNECESSARY_CLARIFICATIONS | 0 | 0 | ok |
| INVENTED_FILTERS | **1** | 0 | **FAIL** |
| INVENTED_DIMENSIONS | 0 | 0 | ok |
| SILENT_SEMANTIC_DROPS | **1** | 0 | **FAIL** |

```
TEMPORAL_PAIR_PRESERVED  = 14
TEMPORAL_ANCHOR_ACCEPTED = 1
TEMPORAL_FAILURES        = 3

LIVE_INTERPRETATION_GATE_V3 = FAIL
```

## The smallest evidence-supported residual defect

**The temporal contract still has no way to say "the analytical owner owns this
window", so a model that correctly recognises a capability-owned window expresses
it by OMITTING `time` entirely — and an omitted slot silently becomes `current`
with nothing recording that it was never stated.**

Three of eighteen, one coherent cause, and all three said so in terms:

| | disclosed ambiguity on `time` |
|---|---|
| V10 | *"no period is imposed because funded_bridge defines the movement window itself"* |
| V13 | *"no period is stated on the intent because the funded_bridge capability defines the movement period itself"* |
| V14 | *"No period is stated in the intent because the funded_bridge capability defines the movement period itself"* |

`time` is not in the schema's `required` list, and `SemanticTime.form` defaults to
`"current"`. So an absent slot parses to `current` — which, for `attribution`, is
now a *valid* reading. V13 and V14 therefore compiled to correct plans
(`funded_bridge`/`bridge`, V14 with its acquired scope intact) **by coincidence**:
the default happened to land on the one value the policy accepts.

That coincidence is the defect. The governed layer cannot distinguish a deliberate
`current` from an unstated slot, so it cannot tell a reading that meant
"capability-owned window" from one that simply said nothing.

This is the same reasoning v1's CF05/CF11 showed, one step further on. Accepting
`current` as an anchor fixed the case where the model *says* `current`. It did not
address the model saying *nothing*, which is what it now does for a
capability-owned window.

Not repaired. This run is measurement only.

## Two defects in my own pre-registration, stated plainly

Neither excuses the FAIL — the gate is scored exactly as pinned — but both are
authoring errors in the bank, not model errors, and the record should say so.

**V06, the invented filter.** The question I wrote is *"what is the change in the
number of **live** funded loans since the prior report?"* The model read "live" as
the governed `account_status = Active`, disclosed it as a non-blocking ambiguity,
and named the governed value list. "Live" *is* a restricting adjective, so
`filters: []` was never the right expectation for this question. The bank said it
anyway and the gate counts it. The authoring error is mine.

**V13, the measure.** The question asks *"which **components** explain why the
funded balance ended where it did"*. The model named `bridge_component` — a
governed `funded_bridge` concept, and arguably the better reading of a question
about components. The bank expected the balance family, on the rule that an
attribution's measure is the quantity being decomposed. Both readings are
defensible; the pin decides, and it scores as a miss and a silent drop of the
named quantity.

**V10 and V14 are the clean evidence** for the residual defect: V14's only fault
is the absent slot, and V10's reading was additionally refused by the compiler
(`UNSUPPORTED_COMPOSITION`) for claiming `metric_delta` while naming a
`funded_bridge`-owned measure — the governed layer catching a genuine conflict.

## Integrity

```
v1 manifest hashes to its pin    OK   (immutable historical evidence, 17/18)
v2 manifest hashes to its pin    OK   (immutable, retired UNRUN)
v3 manifest hashes to its pin    OK   (this measurement)
```

No manifest was edited, no expectation was changed after seeing results, no
question was asked twice, no case was re-run, no cheaper model substituted, and no
product code changed during the run — CI asserts that last one by diffing against
the approved head rather than taking it on trust.

The full structured `CandidateIntent` is persisted verbatim for all eighteen cases
in `change_form_v3_result.json`, together with each case's score, temporal verdict,
compile outcome, served model, bank hash and baseline SHA. The v1 `raw_payload` gap
is closed.

This gate measures interpretation. It computes no answer, so it claims **no** actual
FROM/TO snapshot values; period resolution is deterministic runtime evidence,
covered separately and offline.

## One discrepancy in the run instruction, recorded

The instruction's per-form numbers (`MATERIAL_SUMMARY = 5/5`, `METRIC_DELTA = 5/5`,
`EXPLICIT_MEASURE_PRESERVED = 13/13`, `…INVENTED = 0/5`) describe **v1's**
composition. The pre-registered, hashed v3 bank is **4 / 6 / 4 / 4 with 14
measure-bearing cases**, because v3 deliberately rebalanced toward six metric
deltas across five quantities to avoid overfitting to weighted-average LTV. Scoring
followed the pre-registered v3 policy, as instructed. The aggregate gate is
unaffected: `CHANGE_FORM_TOTAL = 18/18` either way, and every case had to be right.
