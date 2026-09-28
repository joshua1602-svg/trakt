# MI 135 release certification — V3 result

    RUN                34775949733, attempt 1
    Q1 ASKED           2026-09-13T18:52:57Z
    LAST QUESTION      2026-09-13T19:49:00Z
    ELAPSED            56.1 minutes
    LIVE MODEL CALLS   138   (135 questions + 3 infrastructure retries)
    COLLECTOR VERDICT  COLLECTED
    RUN MODE           CANARY_EXPOSED, 103 of 103 measured questions

    PRODUCT_SHA        5c436961ebe279bc0820ab006867b9f8a869bde2  (served, verified)
    BANK_SHA256        9624a0a8f2e7d1c3e59ccf88ad2698be01950fce6e03c111bc428f8272b52122
    OVERLAY_SHA256     a842479642ec2202a61e303df44eeb952fccac5c0fed0d8fbd54f472111cc072
    SCORER_SHA256      bdec83f9f7ad86de3343de72677e672a33b3a40c825b8c7f0afe9360ff3e7be2
    COLLECTOR_SHA256   ebb17776db847bb0402a930bf9fbf994a5d9090beb59200eea7b0ece2712f143
    MANIFEST_SHA256    058fdcd9df49f4c972be44d7a90c152916289c1d5c6ec248d03133ccebad36b0
    OLD_V2_EVIDENCE_MODIFIED   NO
    HISTORICAL_BANK_MODIFIED   NO

## The answer, stated with its denominator

**103 of the 135 questions were measured. 32 were not.** The Anthropic API credit
balance was exhausted at question 106, and every question from 106 to 135 failed
identically with `MODEL_UNAVAILABLE`. That is not a product defect and it is not
a sampling choice — it is a resource that ran out mid-bank.

So this is a certification of **76.3% of the estate**, and the rates below carry a
denominator of 103, not 135:

    FULLY_CORRECT_RATE        0.5049    52 of 103
    ANSWERED_USEFULLY_RATE    0.8544    88 of 103
    USER_SAFE_RATE            0.9903   102 of 103
    WRONG_RATE                0.0000     0 of 103

**The remaining 32 questions are unmeasured on this build.** They are not passes,
they are not failures, and no rate here may be read as an estate figure.

## Frozen scorecard, over all 135

The eight categories were frozen before the run and none was added after it.

    FULLY_CORRECT              52     baseline 52     +0
    PARTIALLY_CORRECT          36     baseline 56    -20
    HONEST_REFUSAL             14     baseline 25    -11
    INCONCLUSIVE                1     baseline  2     -1
    WRONG                       0     baseline  0     +0
    BAD_REFUSAL                 0     baseline  0     +0
    APPROPRIATE_CLARIFICATION   0     baseline  0     +0
    INFRASTRUCTURE             32     baseline  0    +32

    SILENT_SEMANTIC_ERRORS     none detected
    SILENT_SCOPE_WIDENINGS     none detected
    MISROUTES                  0
    BANK_EXPECTATION_DISPUTES  0 raised during the run (see finding 1)

**Those deltas are not like-for-like** and must not be read as regressions: 32 of
the baseline's questions are missing from this column. The valid comparison is
below.

## Like-for-like, on the 103 both runs scored

                          baseline    V3    delta
    FULLY_CORRECT               52     52       +0
    PARTIALLY_CORRECT           35     36       +1
    HONEST_REFUSAL              14     14       +0
    INCONCLUSIVE                 2      1       -1
    WRONG                        0      0       +0

    GOVERNED_PLAN               22     25       +3
    LEGACY_FALLBACK             81     78       -3

Eight questions changed verdict, in both directions:

    Q1.1    INCONCLUSIVE      -> FULLY_CORRECT       clarification now answered
    Q02A    PARTIALLY_CORRECT -> FULLY_CORRECT       governed path, statistic now right
    Q05B    HONEST_REFUSAL    -> PARTIALLY_CORRECT   \  paraphrase pair swapped
    Q05A    PARTIALLY_CORRECT -> HONEST_REFUSAL      /  net zero across the pair
    Q04A    PARTIALLY_CORRECT -> HONEST_REFUSAL      empty-filter refusal instead of a figure
    Q20B    HONEST_REFUSAL    -> PARTIALLY_CORRECT   operation summary, temporal
    Q20C    FULLY_CORRECT     -> PARTIALLY_CORRECT   operation summary  (finding 1)
    Q22A    FULLY_CORRECT     -> PARTIALLY_CORRECT   temporal form      (finding 2)

The 32 unmeasured questions scored PARTIALLY_CORRECT 21 and HONEST_REFUSAL 11 on
the baseline. **None of them was FULLY_CORRECT there**, so the 52 above is not
depressed by the outage — but nor is it a claim about them.

## The four dimensions, recorded separately

    SERVING PROVENANCE     GOVERNED_PLAN 25   LEGACY_FALLBACK 78   (of 103)
                           over all 135: GOVERNED_PLAN 25, LEGACY_FALLBACK 108, NONE 2

    INTERPRETATION         PLAN 102   CLARIFY 1   NOT_REACHED 32
                           WRONG 0

    EXECUTION              material_summary runtime 6, correct owner in all 6
                           no wrong-owner execution detected

    OUTPUT CONTRACT        no independent live numeric parity is claimed:
                           numeric_parity is N/A on all 103, by design —
                           this bank carries expected SEMANTICS, not numbers

Why the governed path declined the other 78, from the plans themselves:

    POPULATION_NOT_EXECUTABLE          33   pipeline / forecast bases
    CAPABILITY_NOT_GENERIC             16   portfolio_summary, limit_assessment
    GEOGRAPHY_REQUESTED                 9   owned by the geography basis resolver
    CAPABILITY_NOT_BRIDGE               4   attribution maps to funded_bridge
    FILTER_NOT_SUPPORTED                3
    RECEIPT_RECONCILIATION_FAILED       3   the measured population is empty
    MEASURE_NOT_SUPPORTED               2
    (remainder)                         8

Every one of those is a declared perimeter refusing rather than answering
approximately. None is a silent narrowing.

---

# Finding 1 — the change_form canonicalisation is live, and the Q18–Q20 fixtures now disagree with the shipped contract

This is the dominant behaviour delta in the run and it is deliberate.

`change_form` appeared on **16 of the 103** measured questions: `material_summary`
10, `attribution` 4, `metric_delta` 2. On 11 of them rule 5 rewrote `operation` to
the form's canonical value, and the effect is visible as a block:

    Q18A  compare  -> summary      Q19A  movement -> summary     Q20A  movement -> summary
    Q18B  compare  -> summary      Q19B  compare  -> summary     Q20B  movement -> summary
    Q18C  movement -> summary      Q19C  movement -> summary     Q20C  movement -> summary
    Q1.1  (none)   -> summary
    Q22B  compare  -> movement     <- metric_delta; this is the M01 fix, live

**The baseline scattered.** Nine paraphrases of one analytical form produced
`compare`, `movement`, `movement`, `compare`, `movement`, `movement`… on the
previous build. On this build all nine produce `summary`. That determinism is
what the canonicalisation was built for, and it is the clearest evidence in this
run that it works.

**The pinned fixtures for that family still expect `movement`.** So eight of the
nine score PARTIALLY_CORRECT on `operation` even though they now agree with each
other and with the contract. One of them, Q20C, was FULLY_CORRECT on the previous
build and is PARTIALLY_CORRECT here for exactly that reason.

    Q20C  "Summarise the month-on-month movement for drawdown loans."
          change_form = material_summary  ->  operation = summary
          fixture expects operation = movement

**The fixtures were not changed and this was not re-adjudicated.** The freeze
contract says a pinned expectation that appears wrong is not touched during the
run, and it was not. This is raised as a BANK_EXPECTATION_DISPUTE for a separate,
deliberate decision after the run: either the Q18–Q20 fixtures are stale against
`CHANGE_FORM_CANONICAL_OPERATION`, or the canonical operation for
`material_summary` is wrong. It cannot be settled by the party that changed the
code, and it is not settled here.

Q19C is the same pattern reaching execution and refusing correctly:

    change_form material_summary, runtime material_summary, owner correct
    refused: start 2026-05-31 is 182 days from the nearest governed snapshot,
    beyond the 45-day limit — "not performed rather than answered about a
    different period"

# Finding 2 — the estate traded 7 capability errors for 9 temporal ones, and the verdict counts hid it

This is the most important product finding in the run, and the frozen scorecard
cannot show it: both the fix and the regression live inside PARTIALLY_CORRECT, so
the category totals barely moved.

**Capability reading improved sharply.**

    questions whose reading differs from the fixture on CAPABILITY
    baseline  8   ->   V3  1

Seven of those eight are the Q18/Q19/Q20 family. On the previous build they were
routed to the wrong capability; on this build they are not. That is the
`change_form` work doing exactly what it was for.

**Temporal reading regressed, in one consistent direction.**

    questions whose reading differs from the fixture on TEMPORAL
    baseline  3   ->   V3  11

    period.form across the measured 103
                                  baseline   V3
        current                         79   79
        relative_pair                   13    4
        previous_reporting_period        1   10
        forward_looking                  8    8
        series                           0    1

Nine questions moved `relative_pair` -> `previous_reporting_period`, and **all
nine also dropped `periods_back` from 1 to None**:

    Q18A  Q18B  Q18C  Q19A  Q19B  Q20A  Q20B  Q22A  Q22B

Nine of nine moving the same way, with the same field going null, is not model
variance between two runs. It is one behavioural change. A movement question
needs a PAIR of periods; `previous_reporting_period` with no `periods_back` names
a single prior period instead, which is a narrowing of the temporal contract, and
every one of these fixtures pins `relative_pair`.

**What caused it is not established by this run and is not guessed at here.** It
is temporally coincident with the `change_form` work and `change_form` adds a slot
to the same intent schema, but the bank was asked once, by design, and one run
cannot separate a schema-shaped effect from a prompt-shaped one. Isolating it
needs a targeted temporal bank, not this one.

Q22A is the single question where the temporal shift alone changed the verdict:

    Q22A  "Which source portfolio contributed most to balance growth last month?"
          operation unchanged (rank), capability unchanged (period_movement)
          period.form  relative_pair -> previous_reporting_period
          FULLY_CORRECT -> PARTIALLY_CORRECT

# Finding 3 — the third consecutive resource failure, and neither guard covers it

    18:52:57Z   Q1
    19:44:48Z   Q106 — HTTP 400 invalid_request_error from the Anthropic API:
                "Your credit balance is too low to access the Anthropic API."
    19:49:00Z   Q135 — the same failure, 30 questions in a row

The product behaved correctly throughout: `MODEL_UNAVAILABLE` from the
interpreter, `compiler.outcome = REFUSE`, `serving.decision = LEGACY_FALLBACK`,
`evidence_persisted = true` on every one of the 30. Nothing was answered on a
guess. That is the governed degradation path working under a dead upstream.

**The harness did not stop, and could not have.** The collector's abort guard
watches for consecutive 401/403 — the V2 failure — and `--abort-after 3` inspects
only the first three questions, for a misconfigured canary. A quota exhaustion
returns HTTP 200 carrying a legacy fallback answer. It is neither an auth failure
nor a transport error, so 30 questions were spent asking a service that could not
reach the model.

This is the third instrument-or-resource failure in three attempts at this bank —
the scorer's intent projection, the bearer's lifetime, and now the model credit —
and each was a different resource silently running out. **No fix is made here.**
The stop policy forbids it and a guard written after seeing the result it would
have caught belongs in a separate, deliberate change.

# Finding 4 — the bearer gate worked, and its margin was 14.8 minutes

    issued        18:45:40Z
    expires       20:03:17Z
    lifetime      77.6 minutes
    at Q1         70.9 minutes remaining   (gate required >= 60)
    run ended     19:49:00Z — 14.3 minutes before expiry

Zero 401s. Zero 403s. The gate that stopped two attempts is the reason this run
reached question 105 instead of question 53.

It is also close. The bank takes 56 minutes and the gate admits a token with 60,
which leaves four minutes for a run that is slightly slower than this one. The
threshold is adequate but not comfortable, and 75 minutes would be the honest
number. **It was not changed after the run.**

The three HTTP 500s at Q21C, Q1.2 and Q1.3 were retried once each; Q1.3 recovered
and the other two did not, giving the two `NONE` provenance rows.

---

## What this run does and does not certify

**Does:** on 103 of 135 questions, against `5c436961`, the estate produced no
WRONG answers, no silent measure, dimension or filter drops, no misroutes and no
scope widenings. Governed-plan completions rose 22 -> 25. Fully-correct held at 52
of the same 103. Capability reading improved from 8 mismatches to 1.

**Also does, and this is not good news:** it establishes one directional
regression. Nine questions narrowed their temporal reading from a period PAIR to
a single previous period, all nine dropping `periods_back`. The category totals
do not show it because it happened inside PARTIALLY_CORRECT, and one question
(Q22A) fell out of FULLY_CORRECT because of it. Finding 2.

**Does not:** it does not certify the estate. 32 questions (23.7%) were never
asked of a working model, and a release certification with a quarter of its
questions unmeasured is a partial certification whatever its rates look like. It
also does not attribute finding 2 to a cause — one run cannot.

    RELEASE CERTIFICATION STATUS   PARTIAL — 103 of 135 measured
                                   NOT SUFFICIENT FOR FULL RELEASE SIGN-OFF
    OPEN REGRESSION                temporal narrowing on 9 questions (finding 2)
    OPEN DISPUTE                   Q18-Q20 operation fixtures (finding 1)

## Nothing was fixed

No product code, no bank, no expectation overlay, no scorecard, no scorer, no
collector, no configuration. The bank was asked once. No second 135 was run. V2's
directory and the historical bank were read and never written.
