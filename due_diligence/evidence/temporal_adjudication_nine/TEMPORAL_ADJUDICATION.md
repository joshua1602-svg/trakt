# Temporal adjudication — the nine `relative_pair` -> `previous_reporting_period` cases

    SCOPE           read-only diagnosis. No product code, tests, bank, overlay,
                    scorer or collector touched. No model call. No new bank.
    PRODUCT_SHA     5c436961ebe279bc0820ab006867b9f8a869bde2
    EVIDENCE        b57889bf (V3 certification) and the immutable historical bank
    NINE_CASES_CONFIRMED   YES

Both selection criteria — `period.form` moving `relative_pair` ->
`previous_reporting_period`, and `periods_back` moving `1` -> `None` — select the
IDENTICAL nine questions. Symmetric difference: empty. A tenth case (Q1.1) changed
period form `None -> series` and is outside this scope.

## The finding, in one line

**No deterministic layer changed.** Given the HISTORICAL model output, the CURRENT
shipped normaliser reproduces the HISTORICAL plan period in **9 of 9** cases. The
divergence is entirely upstream of the code, in what the interpreter returned.

## Adjudication

    CASE_ID  QUESTION                                                    ADJUDICATION
    Q18A     "How did the book change in the last month?"                CURRENT_CORRECT
    Q18B     "What changed in the portfolio since last month?"           CURRENT_CORRECT
    Q18C     "...how the funded book moved over the last month."         CURRENT_CORRECT
    Q19A     "How did the Direct book change last month?"                CURRENT_CORRECT
    Q19B     "What changed in the Direct portfolio since last month?"    CURRENT_CORRECT
    Q20A     "How did drawdown loans change last month?"                 CURRENT_CORRECT
    Q20B     "What changed in the drawdown book since last month?"       CURRENT_CORRECT
    Q22A     "Which source portfolio contributed most to balance
              growth last month?"                                        CURRENT_CORRECT
    Q22B     "Did Direct or Acquired add more balance during the
              last month?"                                               CURRENT_CORRECT

    COUNT_CURRENT_CORRECT     = 9
    COUNT_HISTORICAL_CORRECT  = 0
    COUNT_AMBIGUOUS           = 0

Three independent grounds, none of them "the code does it so it must be right":

1. **The wording names one window, not two.** Every one of the nine says "last
   month" / "since last month" / "over the last month" / "during the last month".
   Not one names two periods. The vocabulary's own note — quoted in
   `plan_temporal_runtime._resolve_pair` — records that saying "two periods" is
   the job of `relative_pair`. The readers did not say two periods.

2. **No comparison semantics are lost.** `plan_material_summary.PAIR_PERIOD_FORMS`
   declares `previous_reporting_period` a form that "resolve[s] to TWO snapshots on
   their own", and `period_request_for` returns `PeriodRequest(relative_mode=
   "current_vs_previous")` for it. Proven, not asserted: the five cases that
   executed carry receipts reading `resolution_method: current_vs_previous`,
   `period_from 2025-11-30`, `period_to 2026-06-30`, two `snapshot_references`.

3. **The historical plan was partly a compiler artefact.** In 7 of the 9 the model
   returned `periods_back: null` and the historical plan carried `periods_back: 1`;
   in Q18C the model returned `previous_reporting_period` and the plan carried
   `relative_pair`. That fill came from `normalise` rule 2, whose guard is
   `operation in PAIR_IMPLYING_OPERATIONS` = `{"movement"}`. A default applied with
   no provenance is what the approved contract's "temporal presence/provenance must
   distinguish explicit wording from defaults" forbids. Every current plan records
   `stated: true, defaulted: false, default_method: "", default_owner: ""`.

**The pinned fixtures are stale for this form.** All nine pin
`temporal: relative_pair`, and the string `previous_reporting_period` appears
**zero times** in the whole 135 bank — neither in `questions.json` nor in
`expected_capability.json`. A bank written when the compiler always coerced an
anchor into a pair never needed to express the anchor form, so it never does. The
fixtures were NOT changed and this was not re-scored.

## Root cause

    ROOT_CAUSE_CLASS     SEMANTIC_METADATA_CHANGE
    ROOT_CAUSE_LOCATION  mi_agent/interpretation_v2/intent.py
                         candidate_intent_json_schema() -> the `change_form`
                         property, passed to the model as `input_schema` at
                         mi_agent/interpretation_v2/opus_interpreter.py:379

    ROOT_CAUSE_EVIDENCE
      1. Deterministic layers excluded by control. Historical candidate intents
         replayed through the CURRENT shipped normaliser reproduce the historical
         plan period 9 of 9. `PAIR_IMPLYING_OPERATIONS`, `CANONICAL_PAIR_FORM` and
         `CANONICAL_PAIR_PERIODS_BACK` are byte-identical across the two SHAs.
      2. The divergence is in the model's own candidate intent: `time.form` moved
         in 8 of 9; in the ninth (Q18C) `operation` moved `movement -> summary`,
         which is precisely rule 2's guard, so rule 2 correctly stopped firing.
      3. The prompt file `opus_interpreter.py` is UNCHANGED between the two SHAs.
         The only changed model-facing surface is the JSON schema, which gained
         `change_form`.
      4. Mechanism, in the schema's own words: the `material_summary` description
         reads "the reader asks what changed, or what moved, ... and WITHOUT
         NAMING TWO PERIODS TO COMPARE". A reader told that this form means no two
         periods were named, reporting one named period, is coherent.
      5. NOT the authorised default: `CHANGE_FORM_ABSENT_PERIOD_DEFAULT` fires only
         where `not time.stated`. All nine carry `stated: true` and `defaulted:
         false`. The default never ran.

    LIMIT OF THE CLAIM. Steps 1-3 are proved. Step 4 is a mechanism, not an
    isolated experiment: separating the schema property from ordinary model
    variance needs an A/B with the property removed, which needs model calls.

## Is `previous_reporting_period` sufficient? Two owners, two answers

    A. "one anchor, the owner derives the counterpart"
       plan_material_summary.period_request_for  (shared by plan_attribution and
       plan_metric_delta) -> relative_mode="current_vs_previous". A PAIR.
       This is the path all nine take. Nothing is lost.

    B. "the requested temporal object is only the previous reporting period"
       plan_temporal_runtime._resolve_pair:
           if form == "previous_reporting_period" and operation != "compare":
               return ... shape=SHAPE_POINT
       A POINT.

Two owners hold contradictory definitions of one enum value. It does not bite on
any of the nine — all nine carry a `change_form`, so a change_form adapter claims
them and path A applies. `plan_temporal_runtime` claims a plan on period form only
when no adapter claimed the form. Recorded as a latent inconsistency.

## A separate defect found during the trace — not the cause of the nine

`normalise.canonical_intent` is documented as idempotent ("a canonical intent is
its own canonical form") and `evidence/contract_normalisation_replay.json` records
`idempotence_failures: 0` over 327 cases. Q22B's REAL recorded intent breaks it:

    model returned : change_form=metric_delta  operation=compare
                     time=previous_reporting_period/pb=None
    after pass 1   : operation=movement    time=previous_reporting_period/pb=None
                     applied: change_form: operation 'compare' -> 'movement'
    after pass 2   : operation=movement    time=relative_pair/pb=1
                     applied: relative_period: previous_reporting_period ->
                              relative_pair+periods_back=1
    IDEMPOTENT     : False

Rule 5 (new) rewrites `operation`, which is rule 2's guard, and rule 2 has already
run. Both existing idempotence tests pass unchanged — the corpus does not carry the
`metric_delta + compare + previous_reporting_period` combination.

Consequence is bounded: the resulting plan reaches `plan_metric_delta`, where
`previous_reporting_period` is a PAIR_PERIOD_FORM, so the pair survives. The
SHAPE_POINT branch is not reachable this way, because the combination can only
arise via rule 5, which requires a `change_form`, which routes away from
`plan_temporal_runtime`. What is real is that a documented invariant is false and
plan identity depends on how many times normalisation runs.

**Not fixed. Not tested for. Reported only.**

## Score effect — diagnostic arithmetic, nothing re-scored

    HISTORICAL_FULLY_CORRECT (of the nine)  = 1   (Q22A)
    CURRENT_FULLY_CORRECT    (of the nine)  = 0

    FULLY_CORRECT lost SOLELY to the temporal difference = 1   (Q22A: its only
    current mismatch dimension is `temporal`)

    capability mismatch fixed on               6   Q18A Q18B Q18C Q19A Q19B Q20A
    of those, converted to FULLY_CORRECT       0
    why: all six also carry an `operation` mismatch, because material_summary now
    canonicalises to `summary` while the fixtures pin `movement`. Two of them
    (Q19A, Q20A) had capability as their ONLY baseline mismatch and would be
    FULLY_CORRECT today if neither operation nor temporal had moved.

So the capability improvement is masked mainly by the OPERATION-fixture dispute,
not by the temporal movement. The temporal movement costs exactly one verdict.

## What the answers actually disclose

Checked, because "silent widening" would be the serious version of this. It is not
silent. Q18A's served envelope carries `sourceNotes: "Governed snapshots
2025-11-30 -> 2026-06-30"` and `warnings[0]: "The two snapshots cover reporting
periods of different length (30 and 212 days). Period-flow fields are reported but
not compared."` The window and the mismatch are both disclosed in governed fields.
The `answer` prose does not repeat them.
