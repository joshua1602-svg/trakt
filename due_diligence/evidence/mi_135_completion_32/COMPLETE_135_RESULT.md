# MI 135 release certification — COMPLETE (composite)

    COMPOSITE_BANK_ID   MI_135_RELEASE_CERTIFICATION_V2_COMPLETE
    PRODUCT_SHA         5c436961ebe279bc0820ab006867b9f8a869bde2
    DEPLOYED_SHA        5c436961ebe279bc0820ab006867b9f8a869bde2  (verified at preflight)

    COMPLETION_BANK_ID          MI_135_RELEASE_CERTIFICATION_V2_COMPLETION_32
    COMPLETION_CASE_COUNT       32
    COMPLETION_CASE_LIST_SHA256 9c6ce083e2a21149114db794ea6c5e1a433cf6bada013222ce14fb4f96425df9
    COMPLETION_MANIFEST_SHA256  bea12bd9b7a79cf7133ff70e9160972e9c6ecca88ca85f15687d8ca0ef1ae41d

    QUESTIONS_ATTEMPTED_THIS_TRANCHE  32
    QUESTIONS_COMPLETED_THIS_TRANCHE  31
    PROVIDER_FAILURES                  0
    AUTH_FAILURES                      0
    INFRASTRUCTURE                     1   (Q1.2, HTTP 500 that survived its retry)

    COMPOSITE_CASE_COUNT  135
    DUPLICATES              0
    MISSING                 0
      from SOURCE_103      103
      from COMPLETION_32    32

    run 34784777352, Q1 at 21:46:04Z, last question 22:16Z
    TOKEN_IAT 1789335573 (21:39:33Z)  TOKEN_EXP 1789340328 (22:58:48Z)
    TOKEN_REMAINING_MINUTES 73.0 of 79.2 at Q1   (gate required >= 45)

## Phase 5's precondition is ONE case short, and that is said rather than rounded

The brief admits a composite "ONLY if all 32 receive valid measurable results".
Thirty-one did. **Q1.2 returned HTTP 500 for the second time**, on its one attempt
and again on its one retry, so it has no governed record and no measurement. The
composite is therefore **134 of 135 measured**, not a clean 135, and every rate
below carries a denominator of 134.

Q1.2 is the only case in the estate that has now failed twice for the same
transport reason on two different days. That is recorded as INFRASTRUCTURE, not as
a product result, and it is not re-asked here: the tranche gives each case one live
attempt and Q1.2 has had it.

## FORMAL RESULT — the frozen bank expectations, nothing adjudicated

    FULLY_CORRECT              66
    PARTIALLY_CORRECT          36
    HONEST_REFUSAL             24
    BAD_REFUSAL                 0
    WRONG                       0
    APPROPRIATE_CLARIFICATION   0
    INCONCLUSIVE                8
    INFRASTRUCTURE              1

    non-infrastructure denominator 134

    USER_SAFE_RATE             0.9403    126 of 134
    ANSWERED_USEFULLY_RATE     0.7612    102 of 134
    FULLY_CORRECT_RATE         0.4925     66 of 134
    WRONG_RATE                 0.0000      0 of 134

    GOVERNED_PLAN_COMPLETIONS       25
    LEGACY_FALLBACK_COMPLETIONS    109
    GOVERNED_PLAN_COMPLETION_RATE  0.1866
    (serving provenance also records NONE = 1, the unmeasured Q1.2)

    SILENT_SEMANTIC_ERRORS      0
    MISROUTES                   0
    SILENT_SCOPE_WIDENINGS      0

    INTERPRETATION   CORRECT 90   PARTIAL 44   WRONG 0   NOT_REACHED 1
    EXECUTION        CORRECT_OWNER 5   WRONG_OWNER 0   NOT_EXECUTED 130
    OUTPUT_CONTRACT  NOT_APPLICABLE 135
                     (this bank carries expected SEMANTICS, not numbers; no
                     independent live numeric parity is claimed anywhere)

The scorer's replay gate ran before the composite was scored and reproduced the
historical 52 / 56 / 25 / 2 exactly.

## LIKE-FOR-LIKE against the complete historical baseline

                             baseline    complete    delta
    FULLY_CORRECT                  52          66      +14
    PARTIALLY_CORRECT              56          36      -20
    HONEST_REFUSAL                 25          24       -1
    INCONCLUSIVE                    2           8       +6
    WRONG                           0           0        0
    BAD_REFUSAL                     0           0        0
    APPROPRIATE_CLARIFICATION       0           0        0
    INFRASTRUCTURE                  0           1       +1

    GOVERNED_PLAN                  22          25       +3
    LEGACY_FALLBACK               113         109       -4

    FULLY_CORRECT_RATE        0.3852      0.4925   +10.7pp   (52/135 -> 66/134)
    ANSWERED_USEFULLY_RATE    0.8000      0.7612    -3.9pp   (108/135 -> 102/134)
    USER_SAFE_RATE            0.9852      0.9403    -4.5pp   (133/135 -> 126/134)

Fully-correct rose sharply. Answered-usefully and user-safe fell, and both falls
have one cause: six more questions are now INCONCLUSIVE.

## What the 32 contributed

                        baseline    now
    FULLY_CORRECT              0      14
    PARTIALLY_CORRECT         21       0
    HONEST_REFUSAL            11      10
    INCONCLUSIVE               0       7
    INFRASTRUCTURE             0       1

    served: LEGACY_FALLBACK 31, none served by the governed path

Fourteen questions moved from PARTIALLY_CORRECT (or, for Q2.3, HONEST_REFUSAL) to
FULLY_CORRECT, every one of them "answered by the legacy path, semantics intact".
The ten honest refusals are the same ten as the baseline, refusing for the same
governed reasons — no funding facility configured, a pipeline question this
portfolio cannot answer, a limits/forecast question outside the configured scope.

## NEW_BANK_EXPECTATION_DISPUTE_CANDIDATE — the seven that became INCONCLUSIVE

    Q3.1  "How much do we currently have at offer and how much of it is likely
           to complete?"                       clarification
    Q3.2  "What's the value of outstanding offers, and when do we expect them
           to fund?"                           refusal, adjudicated by reason code
    Q7.1  "How does the front book compare with our older lending from a risk
           perspective?"                       clarification
    Q7.2  "Are older loans riskier than the loans we've originated recently?"
                                               clarification
    Q7.3  "How different is the risk profile of recent originations versus the
           back book?"                         clarification
    Q8.provenance.3  "Are direct and acquired balances developing differently
           over time?"                         clarification
    S05   "What is the headroom?"              clarification

All seven scored PARTIALLY_CORRECT on the previous build. On this build the system
asks for clarification (six) or refuses with a reason code (one) instead of
answering partially, and the frozen scorer routes both to INCONCLUSIVE — it
declines to judge them inline and sends them to separate adjudication by class.
`APPROPRIATE_CLARIFICATION` stays 0 because nothing in this run adjudicated them.

**Recorded as a candidate, not adjudicated, and the scoring is untouched.** Whether
asking "which risk measure did you mean?" is better or worse than a partial answer
is exactly the question the frozen scorer refuses to answer by itself, and this
task is not the place to answer it either. It is the single largest driver of the
user-safe and answered-usefully falls above.

## Provenance of every case

`composite_manifest.json` records, for all 135, which run supplied the record and
that run's sha256. A case comes from the completion tranche if and only if the
frozen `completion_cases.json` lists it, and from the source run otherwise; the two
lists are disjoint by construction, so choosing the better-looking of two results
for a case is impossible rather than merely discouraged.

    source_103     mi_135_release_certification_v3/raw_records.json
    completion_32  mi_135_completion_32/raw_records.json

Neither source run was rewritten. V2, V3 and the historical bank were read only.

---

# ADJUDICATED_OBSERVATIONS

Separate from the formal score above, which incorporates none of them. No new
adjudication is made here; these are the three already established.

## 1. The nine temporal cases — adjudicated CURRENT_CORRECT (commit 553107f3)

    Q18A Q18B Q18C Q19A Q19B Q20A Q20B Q22A Q22B

    DEPRESSES THE FORMAL SCORE.

All nine score PARTIALLY_CORRECT against fixtures pinning `temporal:
relative_pair`, and the string `previous_reporting_period` appears zero times in
the whole bank. One of them, Q22A, is PARTIALLY_CORRECT on temporal ALONE and was
FULLY_CORRECT on the previous build. The adjudication established that the pair is
preserved through the deterministic period contract in all nine, that the
divergence is upstream of the code (the current normaliser reproduces the
historical plan period 9 of 9 on the historical input), and that the fixtures
predate the temporal-authority rule.

## 2. The Q18-Q20 material_summary operation-fixture dispute

    UNRESOLVED, AND DEPRESSES THE FORMAL SCORE.

`material_summary` canonicalises to `operation: summary`; the fixtures pin
`movement`. Nine paraphrases that scattered across compare/movement on the previous
build now agree with each other and disagree with the bank. Deliberately not
resolved in this task.

## 3. Q22B latent `canonical_intent` idempotence issue

    NO FORMAL SCORE EFFECT.

Rule 5 rewrites the `operation` that rule 2 guards on, after rule 2 has run, so a
second normalisation pass changes the plan. Bounded: the resulting plan reaches
`plan_metric_delta`, where `previous_reporting_period` is a pair form, so no pair
is lost and no answer is wrong. Not fixed, per the brief.

## How much the two established disputes depress the formal score

Of the 36 PARTIALLY_CORRECT, **11 are explained ENTIRELY by them** — their only
mismatch dimensions are `operation` (the Q18-Q20 dispute) and/or `temporal` (the
adjudicated nine):

    Q18A Q18B Q18C Q19A Q19B Q20A Q20B Q22B   operation + temporal
    Q19C Q20C                                 operation
    Q22A                                      temporal

Were both resolved in the product's favour, the formal score would read
FULLY_CORRECT 77 and PARTIALLY_CORRECT 25 — a fully-correct rate of **0.5746**
rather than 0.4925. **That arithmetic is illustrative and is NOT the result.** The
remaining 25 PARTIALLY_CORRECT are ordinary mismatches: `statistic` 15,
`weight` 2, `output_structure`+`statistic` 2, `operation`+`statistic` 1,
`temporal` 2 (outside the nine), `operation` 3 (outside the dispute).

# FINDINGS, in the four permitted classes

    RELEASE_BLOCKER               none
      WRONG 0, silent semantic loss 0, silent scope widening 0, misroutes 0,
      and no material violation of an already-frozen semantic contract was
      established. No new release criterion is created here.

    NON_BLOCKING_PRODUCT_QUALITY
      Fully-correct is 0.4925 and governed-plan completion is 0.1866. Both are
      quality facts about the release candidate, not blockers under the frozen
      criteria.

    BANK_OR_TEST_DEFECT
      The Q18-Q20 operation fixtures and the nine temporal fixtures (11 cases,
      quantified above). Plus the NEW_BANK_EXPECTATION_DISPUTE_CANDIDATE: the
      seven clarification/refusal cases the frozen scorer routes to INCONCLUSIVE.

    INFRASTRUCTURE
      Q1.2, HTTP 500 on both its attempt and its retry, on two separate days.
      One case of 135 is unmeasured. The credit exhaustion that blocked the
      previous run did not recur: zero provider failures across all 32.
