# MI 135 release certification — INFRASTRUCTURE_BLOCKED

    BANK_ID                    MI_135_RELEASE_CERTIFICATION_V2
    BANK_SHA256                9624a0a8f2e7d1c3e59ccf88ad2698be01950fce6e03c111bc428f8272b52122
    EXPECTATION_OVERLAY_SHA256 a842479642ec2202a61e303df44eeb952fccac5c0fed0d8fbd54f472111cc072
    SCORING_CONTRACT_SHA256    f15ba5331cf00aff3617659f15d98302664751c97eab2012750e74bc045409af
    PRODUCT_SHA                5c436961ebe279bc0820ab006867b9f8a869bde2
    DEPLOYED_SHA               5c436961ebe279bc0820ab006867b9f8a869bde2  (confirmed)
    run                        34765295922, 15:20:49Z - 15:52:57Z

    RESULT                     INFRASTRUCTURE_BLOCKED

**This run does not certify the estate, and no percentage from it should be
quoted as one.** Two independent failures — one platform, one instrument —
between them make every one of the 135 unscoreable. Neither is a product
finding.

## Blocker 1 — the bearer expired 53 questions in

    questions 1-52   completed        (bank positions 0-52)
    question 53      Q18C  HTTP 401
    questions 53-134 HTTP 401         82 consecutive

    AUTH_FAILURES = 82        INFRASTRUCTURE = 82

The preflight authenticated at 15:20:29 and the 401s begin at bank position 53,
roughly twenty minutes into a thirty-two-minute run. An Entra access token lives
about an hour and this one was minted before the sequence began; the bank simply
outlived it.

Those 82 questions never reached the application, so they cost no interpretation
and tell us nothing about the estate. They are INFRASTRUCTURE by the frozen
definition and no product conclusion is inferred from them.

**The collector should have stopped.** Its `--abort-after` guard checks only the
FIRST N questions for a misconfigured canary; it has no rule for a run that
starts healthy and loses its credential half way. Eighty-two questions were asked
into a 401 one after another. That is a harness gap, recorded below.

## Blocker 2 — the frozen scorer cannot read this build's intents

Every one of the 53 questions that DID complete is INCONCLUSIVE, and all 53 carry
the same reason:

    the recorded intent did not parse:
    IntentParseError: INTENT_SCHEMA_INVALID: time unknown keys ['stated']

The cause, measured:

    this build records   "time": {"form": "current", …, "stated": true}
    00bb3e9d recorded    "time": {"form": "current", …}

`score_135._score_interpretation` re-parses the recorded intent through
`parse_candidate_intent`, and that parser rejects `stated` as an unknown key in
the `time` slot. The field was added by this programme's temporal-presence work —
an authorised, closed change — and the signed-off scorer predates it. So the
interpretation dimension is unscoreable for **every** question on this build, and
because the scorer's user verdict depends on it, all 53 collapse to INCONCLUSIVE.

This is a scorer defect, not a product defect. The product produced a plan for
all 53 (`compiler.outcome = PLAN`, 53 of 53) and answered 39 of them.

**It is not fixed here.** The stop policy forbids modifying the scorecard, and a
scorer edited after seeing a result it produced is not a scorer anyone should
trust. It is reported as a BANK_OR_TEST_DEFECT for a separate, deliberate change.

## What the raw evidence does support

From the 53 that completed — and this is a PREFIX of the bank in its own order,
not a sample of the estate, so no rate is projected from it:

    compiler outcome     PLAN 53 of 53
    serving decision     NEW 23   LEGACY_FALLBACK 30
    envelope ok          39 of 53
    model id             claude-opus-5, 53 of 53 — the approved path, unsubstituted

    GOVERNED_PLAN completions on the prefix   23
    (the whole historical bank produced 22)

That last line is the only thing in this run that points anywhere: the previous
build produced 22 governed-plan completions across all 135, and this build
produced 23 in the first 53. It is suggestive and it is **not** a certification;
the questions are not comparable sets.

## Release metrics — deliberately not computed

    FULLY_CORRECT             not established
    PARTIALLY_CORRECT         not established
    HONEST_REFUSAL            not established
    BAD_REFUSAL               not established
    WRONG                     not established
    APPROPRIATE_CLARIFICATION not established
    INCONCLUSIVE              53   (all for the same instrument reason)
    INFRASTRUCTURE            82

    USER_SAFE_RATE            not established
    ANSWERED_USEFULLY_RATE    not established
    FULLY_CORRECT_RATE        not established
    WRONG_RATE                not established

A denominator of 53 unscoreable cases produces 0.0 for every rate. Publishing
those zeros as an estate measurement would be the most misleading thing this
report could do: they measure the harness, not the product. They are stated as
NOT ESTABLISHED instead.

    SILENT_SEMANTIC_ERRORS    none detected in the 53 completed
    MISROUTES                 0 detected in the 53 completed
    SILENT_SCOPE_WIDENINGS    none detected in the 53 completed
    BANK_EXPECTATION_DISPUTES 0

## The historical baseline is untouched

`mi_135_live_bank_00bb3e9d/` was read and never written. Its results stand:
FULLY_CORRECT 52, PARTIALLY_CORRECT 56, HONEST_REFUSAL 25, INCONCLUSIVE 2,
WRONG 0, with GOVERNED_PLAN 22 / LEGACY_FALLBACK 113. The scoring shim reproduces
those numbers exactly and that gate ran before this run was scored.

No comparison of movement is offered, because there is nothing valid to compare.

## Nothing was fixed

No product code, no bank, no scorecard, no expectation, no configuration. The
bank was asked once. No second 135 was run.
