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

---

# ADDENDUM — both instrument defects fixed, and what the salvaged evidence shows

The two BANK_OR_TEST_DEFECTS above are repaired. The run itself is unchanged and
is not re-run; what follows is the SAME evidence re-read by a scorer that can now
read it.

## Fix 1 — the scorer's projection was one level too shallow

`_score_interpretation` already narrowed the recorded intent to the parser's own
key tuple, with a comment saying exactly why: a recorded intent is
`CandidateIntent.to_dict()`, not model output, and `to_dict` carries fields the
model-facing parser refuses. That narrowing applied to the TOP LEVEL only, and
`stated` lives one level down inside `time`.

**The parser is right and was never the problem.** `_parse_time` DERIVES
`stated` as `"form" in raw`. Admitting it from a payload would let a model assert
that a temporal form was stated when it was not — the exact distinction three
sprints of temporal-presence work established. Adding `stated` to `_TIME_KEYS`
would have been a semantic regression dressed as a fix. **No product code was
changed.**

The projection now narrows every nested block to that slot's own key tuple, read
from the parser rather than listed here, so a derived field added to any slot is
handled automatically and a genuinely unknown key still fails closed.

    score_135.py   f15ba533… -> bdec83f9f7ad86de3343de72677e672a33b3a40c825b8c7f0afe9360ff3e7be2

**The gate that makes this trustworthy:** the historical evidence, re-scored
through the changed scorer, still produces 52 / 56 / 25 / 2 exactly. A scorer
edited after seeing a result it produced is only safe if the run whose answer is
already known still gets that answer.

## Fix 2 — the collector now stops when the credential dies

`--abort-after` inspects only the first N questions, for a misconfigured canary.
Nothing watched a run that starts healthy and loses its credential half way, so
eighty-two questions were asked into a 401 one after another. The cost is not
money — a 401 buys no interpretation — it is that the run LOOKS complete.

    --abort-on-auth-failures  N consecutive 401/403 responses (default 3)

An expired credential is also no longer retried. A second call with the same dead
token buys a second 401, and this programme had already ruled that repeatedly
retrying an expired token is what a harness must not do.

Six self-test rules now prove the predicate, including that a 500 is NOT an auth
failure and is still retried like any transport fault.

    collect_135.py  2ba05353… -> ebb17776db847bb0402a930bf9fbf994a5d9090beb59200eea7b0ece2712f143

## The salvaged 53, re-scored — NOT a certification

    FULLY_CORRECT              18
    PARTIALLY_CORRECT          26
    HONEST_REFUSAL              9
    BAD_REFUSAL                 0
    WRONG                       0
    APPROPRIATE_CLARIFICATION   0
    INCONCLUSIVE                0
    INFRASTRUCTURE             82

    over the 53 that reached the application:
      user safe          1.00
      answered usefully  0.83
      fully correct      0.34
      wrong              0.00

    GOVERNED_PLAN 23   LEGACY_FALLBACK 30
    silent semantic losses none   misroutes 0   silent scope widenings none

**These 53 are the first 53 questions of the bank in its own order.** They are a
PREFIX, not a sample, and the bank is ordered by canonical id — so this covers
roughly the first eighteen canonicals and their three variants, not a spread
across the estate. No rate here is projected to 135 and no movement against the
135-question baseline is computed, because 53 questions and 135 questions are not
comparable sets. The movement column the scorer prints is meaningless for this
run and is deliberately not reproduced.

What can fairly be said: across the 53 that ran, nothing was WRONG, nothing was
silently dropped or widened, nothing was misrouted, and every case was
user-safe. Twenty-three of them completed through the governed plan; the whole
previous bank produced twenty-two.

## Before the next run

The frozen manifest pins the OLD hashes of both files, so its preflight will now
fail the `unmodified: scoring_contract` and `unmodified: collector` checks. That
is correct — the files did change — and it is left failing on purpose: the
manifest is the immutable record of the spent run and is not edited after the
fact. A new run re-freezes the contract first:

    python due_diligence/evidence/mi_135_release_certification_v2/freeze_contract.py --force

and a fresh `MI_BEARER` should be minted immediately before it, because the bank
takes about half an hour and the last token did not survive one.
