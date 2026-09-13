# Output-contract canary — the five, spent once

    bank      metric_delta_output_contract_canary
    sha256    6465f4ed962f058d9c9e9f63ecb941cb19dcf60a218e27339cf60d92e39a6996
    run       34762507255, 2026-09-13T14:23:26Z → 14:28:30Z
    served    5c436961ebe279bc0820ab006867b9f8a869bde2
    asked     5 of 5, once each, no retries, no rephrasing
    VERDICT   FAIL — 2 PASS, 3 FAIL

The bank is SPENT. The result was committed by the job before it exited.

## The flag, verified behaviourally

Every record that reached serving carries `serving.mode = "canary"` and
`principal_matched = true` against the single allow-listed oid. No wildcard.

## What passed, and it is the point of the sprint

**N03 closes M03, and closes M01 in the same case.**

    question   "Tell me where the funded book's weighted average interest rate
                sat at each of the two most recent reporting dates and what the
                change between them was."
    operation  compare  ->  movement        canonicalised centrally
    route      governed_plan_metric_delta   served from NEW
    mode       requested_metric
    owner      run_period_change_analysis
    field      requested current_interest_rate
               executed current_interest_rate
               served   current_interest_rate
    status     partially_available
    disposition QUALIFIED
    period     2025-11-30 -> 2026-06-30

The model emitted `compare`; the central contract collapsed it to `movement`;
the adapter admitted it; the owner ran in requested_metric mode. That is the M01
repair, live. And the certification gate — which requires the answer to state the
figure the receipt records — passed on a `partially_available` metric. That is
the M03 repair, live, on the very metric that failed before.

**N05 confirms material_summary is unchanged**: `governed_plan_material_summary`,
`portfolio_overview`, `insight_funded.compose`, and no requested-metric block —
a form that names no metric did not acquire one.

## What failed

### N01 — HTTP 500. Not a refusal, and not an interpretation result.

    question       "Set the funded book's outstanding balance at the newest
                    reporting date against the one before it and tell me how far
                    it has moved."
    http_status    500
    record_found   False        (sink polled 120s; no record was ever written)
    seconds        33.5
    answer         "" (empty)

No plan, no reading, no serving decision, no evidence record. The service
returned a server error. **There is therefore no governed evidence of what this
question was read as, and none is inferred here.** The runner recorded
`outcome = REFUSE` only because the envelope was not `ok`; that is the harness's
label for a non-answer, not a governed refusal.

This is the wording closest to M01 — the shape this sprint repaired — so it is
the one case whose failure cannot be attributed to the bank. It needs a separate
read-only diagnostic against the service logs, which this task does not perform.

### N02 and N04 — read as `level_comparison`, and the wording invited it

    N02  "Where does the funded book's weighted average loan-to-value stand now
          compared with the prior reporting date?"
    N04  "Contrast the principal balance on the funded book between the two
          latest reporting dates."

Both were claimed AND compiled as `level_comparison`, served from NEW through the
governed temporal path, and answered "Here is the result for your query, covering
2 groups." No legacy fallback; no wrong figure published.

**The pin was mine and I believe it was the wrong pin.** `vocabulary.CHANGE_FORMS`
defines `level_comparison` as *"What was it in March and in June? Two states,
compared as levels, without movement intelligence or drivers."* N02 asks where a
metric **stands** compared with a prior date; N04 asks to **contrast** two dates.
Neither contains movement language. The interpreter's reading matches the
governed definition; my expectation did not.

That is the v4/W06 authoring defect — pinning an expectation the architecture may
legitimately answer differently — and this bank's own pre-registration document
warns against it by name before committing it twice. The two cases are recorded
as FAIL because the bank pinned `metric_delta` and expectations are not relaxed
after seeing results. What is wrong is identified as the bank, not the product.

## Counts

    QUESTIONS_ATTEMPTED  5      QUESTIONS_COMPLETED  5
    PASS 2   FAIL 3
    METRIC_DELTA_CASES 4 (N01-N04)   OTHER_FORM_REGRESSION_CASES 1 (N05)

    served from NEW                    4 of 5 (N02, N03, N04, N05)
    LEGACY_FALLBACK_POSITIVE_CASES     0
    reached no serving decision at all 1 (N01, HTTP 500)

    REQUESTED_METRIC_ANSWERED          0
    REQUESTED_METRIC_QUALIFIED         1 (N03)
    REQUESTED_METRIC_GOVERNED_REFUSAL  0

    WRONG                    0   no incorrect figure was published anywhere
    SILENT_SEMANTIC_ERRORS   0   every form read is recorded in its own evidence
    MISROUTES                2   N02, N04 against the bank's pin
    UNEXPECTED_REFUSALS      1   N01, and it is a server error, not a refusal
    AUTH_FAILURES            0
    PROVIDER_FAILURES        0   no provider error was recorded; N01's 500 has no
                                 recorded cause and is not attributed to one

    OWNER_DIFFERENCES   0   for the two cases that reached an owner
    PERIOD_DIFFERENCES  0   both resolved 2025-11-30 -> 2026-06-30
    SCOPE_DIFFERENCES   0   both executed tenant ERE, no portfolio narrowing

No independent LIVE numeric parity was measured. Offline owner parity — 3/3 with
zero numerical, period, scope and owner differences — remains the numeric
evidence.

## Non-blocking observations

1. **N02 and N04 published a bare "Here is the result for your query, covering 2
   groups."** for a level comparison. No figure, no metric name, no dates. It is
   not wrong and it violates no pinned contract in this bank, but it is a thin
   answer of exactly the family this sprint repaired for `requested_metric`.
   `level_comparison` has no equivalent output contract. Recorded, not acted on:
   the form is on the do-not-reopen list and no expectation here covers it.

## Nothing was repaired during this run

No question retried, rephrased or substituted; no expectation changed; no product
code touched; the 135 not run.
