# Completion tranche — frozen, and stopped before the first question

    COMPLETION_BANK_ID          MI_135_RELEASE_CERTIFICATION_V2_COMPLETION_32
    COMPLETION_CASE_COUNT       32
    COMPLETION_CASE_LIST_SHA256 9c6ce083e2a21149114db794ea6c5e1a433cf6bada013222ce14fb4f96425df9
    COMPLETION_MANIFEST_SHA256  bea12bd9b7a79cf7133ff70e9160972e9c6ecca88ca85f15687d8ca0ef1ae41d
    PRODUCT_SHA                 5c436961ebe279bc0820ab006867b9f8a869bde2

    SOURCE_PARTIAL_CERTIFICATION_COMMIT  b57889bf
    SOURCE_TEMPORAL_ADJUDICATION_COMMIT  553107f3

    ORIGINAL_103_MODIFIED               NO
    ORIGINAL_PARTIAL_EVIDENCE_MODIFIED  NO

## The 32, read from the evidence and not from question numbering

A case is in this tranche if and only if the committed V3 scorecard recorded it
`INFRASTRUCTURE_FAILURE`. `freeze_completion.py` copies that decision out of
`scored_v3.json`; it does not re-derive it.

    Q21C   Q1.2   Q2.1   Q2.2   Q2.3   Q3.1   Q3.2   Q3.3
    Q4.1   Q4.2   Q4.3   Q5.1   Q5.2   Q5.3   Q6.1   Q6.2
    Q6.3   Q7.1   Q7.2   Q7.3   Q9.1   Q9.2   Q9.3   Q8.provenance.1
    Q8.provenance.2   Q8.provenance.3   C03   C04   S05   B01   B03   B04

    MISSING_CASE_COUNT = 32
    overlap with the measured 103          0
    duplicate ids                          0
    all 32 present in the frozen bank      YES
    question text identical to the bank    YES
    any of the 32 carrying a valid result  NONE
      (no execution receipt; disposition INTERPRETER_FAILURE or no record)

### One discrepancy against the brief, reported rather than smoothed over

The brief asks for the cases that were INFRASTRUCTURE **solely because of Anthropic
credit exhaustion**, and expects 32. The evidence splits:

    credit exhaustion (MODEL_UNAVAILABLE, "credit balance is too low")   30
    HTTP 500 that survived its one retry (Q21C, Q1.2)                     2
    ------------------------------------------------------------------  ---
    total INFRASTRUCTURE                                                  32

Strictly, "solely credit" is 30. The tranche is all 32, because 103 + 32 = 135 is
what a complete certification needs, both causes are INFRASTRUCTURE rather than
product results, and re-asking the two 500s is exactly as valid. The tranche count
the stop rule guards — 32 — is met.

## A second discrepancy: the brief's scorer hash is one repair stale

    brief SCORING_CONTRACT_SHA256   f15ba5331cf00aff3617659f15d98302664751c97eab2012750e74bc045409af
    scorer actually pinned here     bdec83f9f7ad86de3343de72677e672a33b3a40c825b8c7f0afe9360ff3e7be2

`f15ba533…` is `score_135.py` at commit `c783d2b1` — the PRE-REPAIR scorer. It
cannot parse an intent carrying the derived `time.stated` and scored 53 of them
`INCONCLUSIVE` in V2. Scoring the composite with it would reintroduce that defect
AND break like-for-like with the immutable 103, which were scored by `bdec83f9…`.

`bdec83f9…` is pinned instead, and it is the same scorer whose replay reproduces
the historical 52 / 56 / 25 / 2 exactly. **No scorer was edited for this task.**

## What is being reused unmodified

    collector      collect_135.py  ebb17776…  — `--only` is a flag it already had
    scorer         score_135.py    bdec83f9…
    shim           score_v3.py     d40ad2ee…
    bank           questions.json  9624a0a8…
    overlay        expected_capability.json  a8424796…
    preflight      the V3 preflight, at --min-token-minutes 45

The only new instruments are read-only: `freeze_completion.py`, `provider_gate.py`,
`merge_batches.py`, `compose_135.py`. None touches product code.

## Why the tranche is asked in four batches of eight

A provider quota exhaustion returns HTTP 200 carrying a legacy fallback answer. It
is neither an auth failure nor a transport error, so the frozen collector's abort
guards cannot see it — which is precisely how thirty questions were spent against a
dead upstream on 2026-09-13. The collector is pinned for this task, so the stop is
built outside it: four batches, `provider_gate.py` between them. A second
exhaustion costs at most eight cases instead of thirty-two.

The gate also carries `--require-canary` after batch 1. The frozen collector skips
its own canary guard when `--only` is set (`index == args.abort_after and not
wanted`), so a targeted tranche would not otherwise notice a wrong
`MI_AGENT_PLAN_SERVE`.

### Both gates were proved against the run that failed

Pointed at the V3 evidence, `provider_gate.py` reports 30 provider failures and
exits non-zero — it would have stopped that run. The merge, compose and score
pipeline was then dry-run end to end on the same records: composite 135,
duplicates 0, missing 0, and the frozen shim reproduced the committed
52 / 36 / 14 / 1 / 32 exactly.

## Anthropic credit CANNOT be pre-checked from here — stated explicitly

There is no provider endpoint this repository can authoritatively ask about a
credit balance, and the API key that would ask it lives on App Service, not in
these secrets. **Credit availability cannot be established except through
execution.** The first batch of eight IS the check, and the gate stops the tranche
before the other twenty-four are spent.

## What the operator must do before this can run

    1. Top up the Anthropic credit balance.
    2. MI_AGENT_PLAN_SERVE:  off -> canary  on trakt-mi-api
       (single allow-listed principal, no wildcard; this RESTARTS App Service and
       the liveness-twice gate absorbs the restart)
    3. THEN mint a fresh MI_BEARER, LAST, so its life is ahead of the tranche:
           az account get-access-token --scope <the API scope> --force-refresh
       Asking again is not the same as being issued one — Entra and MSAL return a
       CACHED token until it is near expiry, which stopped two attempts at the 135.
       The gate requires TOKEN_REMAINING_MINUTES >= 45.

Nothing else is outstanding. Every check that does not need the operator passes.
