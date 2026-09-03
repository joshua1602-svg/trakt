# Why one question answered in one replay and refused in the next

**Status: diagnosed, reproduced deterministically, NOT fixed.** The reproduction
is `mi_agent_api/tests/test_the_language_understanding_step_is_the_variable.py`;
it runs offline against `tests/fixtures/pipeline_transition_2w` and needs no
model, no credit and no network.

## The observation

"How many cases left KFI in the last week" ANSWERED in one 115-question replay
and, on byte-identical deployed code, returned

> I could not complete the language-understanding step for this question, so I
> have not answered it. … Please try again.

in the next.

## What is not the cause

The deterministic reading. The same question parses to the same contract, is
claimed by the same route (`pipeline_stage_movement`) and produces the same
answer, five times out of five, with the concept-merge arm off. Pinned by
`test_the_deterministic_reading_is_stable`.

## The cause

ONE outbound model call.

1. `chat_routing` runs the concept-merge arm after the deterministic contract
   exists (`MI_AGENT_CONCEPT_MERGE=on` plus an API key).
2. `concept_merge_arm.apply` asks the model to propose concepts in registered
   vocabulary. ANY exception from that call — rate limit, overload, timeout,
   exhausted credit, a reply that will not parse — is caught and reported as
   `status: proposal_unavailable`.
3. `mi_service._enforce_model_availability` then turns an otherwise-successful
   envelope into the refusal above.

Step 3 is deliberate and right: an arm that was switched on and did not answer
must not be allowed to silently narrow the question — measured on this build,
with the arm on and the credit exhausted, twenty of twenty runs of one
product-scoped question returned a whole-book answer. The consequence is that
the arm's AVAILABILITY decides the outcome of a question the deterministic path
answers perfectly well, and availability is not a property of the code. That is
the whole non-determinism.

The route still claims and answers the question; the refusal is stamped on top
of a successful envelope. `metadata.conceptMerge.detail` carries the cause
("RuntimeError: overloaded_error: …"), so a run of these names its own fault —
but only to whoever reads envelopes, not to the reader and not to the record.

## The second finding: the record calls it a broken calculation

`_classify_analytical_failure` recognises the coverage gate's marker
(`semanticCoverageRefused`), the declared capability boundary
(`controlledUnsupported`) and an unmapped question. The availability refusal
sets none of them, so it falls through to:

| field | value |
| --- | --- |
| `governance.error.code` | `CALCULATION_FAILED` |
| `governance.error.category` | `capability` |
| `governance.error.retryable` | `false` |
| message | "… Please try again." |

`operations_control.mi_query_telemetry` and `migration_phase0/replay_probe` both
count `CALCULATION_FAILED` as an **ERROR**, so every transient model outage is
recorded as the system having broken — and `retryable: false` contradicts the
sentence the reader is shown. This is precisely the mislabel the coverage gate
was given its own marker to escape on 2026-09-03 (see the comment in
`_enforce_semantic_coverage`: "every time the coverage gate did its job … the
operator's record said the system had broken"). This path never got the same
treatment.

Pinned as it stands by `test_the_record_calls_an_unavailable_model_a_failed_
calculation`, so the gap cannot be lost. Closing it means changing that test
WITH the fix.

## Why it was not fixed here

Every honest fix needs a decision this session did not have:

* **Label it truthfully.** There is no existing error code for "an upstream
  model was unavailable; ask again" — `trakt_core.errors` says code values are
  part of the external contract, and adding one is a governed change. The
  nearest existing shapes (`DATA_SOURCE_UNAVAILABLE`, category `data`,
  retryable, HTTP 503) change the HTTP status a client sees.
* **Retry the call once.** Removes most of the variance, adds latency and cost
  to every question the arm touches, and does not make the remaining failures
  legible.
* **Do not let an unavailable arm refuse a question the deterministic path
  fully covered.** The strongest option, and the one the arm's own docstring
  argues against: "the estate has no completeness proof independent of the
  deterministic parse", so today there is nothing that can certify "fully
  covered". A coverage ledger that could certify it would make this safe.

Recommendation, in order: label it truthfully first (it costs nothing and makes
the ERROR rate readable), then decide between retry and the coverage proof.

## A second, different source — named so the two are not confused

The free-form parser arm (`MI_AGENT_LLM_PARSER`) fails DIFFERENTLY: it falls
back to the deterministic reading and ANSWERS, publishing
`parser_used: deterministic_fallback_after_llm_failure`. That varies the answer
rather than the outcome and carries no refusal sentence — so the observed
failure is the concept-merge arm and not this one. If both arms are on in the
deployment, this is a second non-determinism with no reader-visible signature at
all, and it is worth measuring separately.
