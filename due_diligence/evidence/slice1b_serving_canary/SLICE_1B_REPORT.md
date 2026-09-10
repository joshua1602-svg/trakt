# Slice 1B — tiny serving canary

The final Slice 1 implementation step. The eligible governed plan may now BE the
served answer, for one explicitly allow-listed principal and for nobody else.

## What was built

One new module, two existing production files touched.

| File | Change |
|---|---|
| `mi_agent/plan_serving_canary.py` | NEW. The serving decision, the plan→receipt reconciliation, the render, the record. |
| `mi_agent/plan_shadow_wiring.py` | `record_plan_stages` extracted so the shadow and the serving record have ONE owner for the evidence shape. |
| `mi_agent_api/mi_service.py` | The trusted `ExecutionContext` threaded into `_run_analysis`; one exclusive branch at the existing shadow call site. |

Nothing else. `interpretation_v2`, `CandidateIntent`, the compiler, normalisation,
the Slice 1 eligibility perimeter, `PlanRuntimeAdapter`, the deterministic
executor, geography, filters, dataset ownership, lens ownership, the temporal
architecture and the specialist capabilities are all **byte-unchanged**.

## The principal identity (the blocking precondition)

The brief required a stable authenticated principal, and said to STOP if none
existed. One does:

`ExecutionContext.actor_id`, which is

* the **verified Entra `oid`** on the dashboard bearer path. `react_auth`
  validates the token against the issuing directory's published keys and
  **refuses outright** a token carrying no `oid` — `sub` is deliberately not
  accepted as a fallback because it is pairwise per application. The value is
  normalised lower-case by `trakt_core.principal.normalise_object_id`.
* the platform **`userId`** on the Static Web Apps header path.

It is already what the audit trail records as the actor, precisely so identity
survives a rename or a mailbox reassignment. It is therefore what the allow-list
matches — exactly, and case-folded.

Two shared sentinels exist: `mi_agent_api/identity.py` falls back to
`"unknown-principal"`, and a synthetic local-dev principal resolves to
`"local-dev"`. Neither names an individual, so **both are refused as allow-list
entries** alongside the wildcards. Allow-listing one would have served every
unidentified caller.

## Configuration

```
MI_AGENT_PLAN_SERVE=canary                       # default off
MI_AGENT_PLAN_SERVE_PRINCIPALS=<the oid>         # comma-separated, exact
```

* Default **off**: an unset `MI_AGENT_PLAN_SERVE` serves nothing new, and only
  the exact word `canary` enables anything.
* Fail-closed on the list too: unset or empty serves nobody.
* **No wildcard.** `*`, `all`, `any`, `everyone` are ignored with a warning.
  There is no value of either setting that means "everybody".
* **A client id cannot enable it.** `ERE`, `ere_funding_uk`, `client_001` and
  `acme` are not principal ids and match nothing.
* The Slice 1A shadow settings (`MI_AGENT_PLAN_SHADOW`,
  `MI_AGENT_PLAN_SHADOW_CLIENTS`) cannot enable serving.

## The serving rule

NEW serves only when all of these hold. Otherwise legacy serves, and the record
says which condition failed.

1. `MI_AGENT_PLAN_SERVE=canary`
2. the authenticated `actor_id` is on the allow-list, exactly
3. the frozen interpretation returns PLAN
4. Slice 1 eligibility says eligible
5. the deterministic execution succeeds
6. the plan reconciles against the executor's receipt, and the result renders
   through `adapt_workflow_result` — the contract every channel already renders

CLARIFY and REFUSE introduce **no new serving behaviour in this slice**: both
fall back to legacy and are recorded. So does ineligibility, an interpreter
failure, an execution fault, a render fault, and any unexpected exception.

## Why the branch sits AFTER the legacy answer

The brief's diagram branches before interpretation. This implementation branches
at the existing shadow call site instead — *after* the legacy point-in-time
envelope is complete — and the reason is the fallback guarantee. Falling back to
something already in hand cannot fail; falling back to something that has to be
computed after the new path broke can. The cost is that the canary principal pays
for both paths, which is the right trade for one test account. What is SERVED is
what the serving rule decides, which is the property the brief asks for.

## Why the legacy question-reading guards are not re-run

Four of the five guards between legacy execution and return re-read the question,
which is the raw-text re-reading this architecture exists to remove. They are not
needed on the new path because the eligibility perimeter refuses — structurally,
and before execution — every shape they exist to catch:

| Legacy guard | The perimeter's structural equivalent |
|---|---|
| `_guard_temporal_honouring` | `PERIOD_NOT_CURRENT` — any stated period but `current` is ineligible |
| `_guard_stated_geography_basis` | `GEOGRAPHY_REQUESTED` — any geography binding at all is ineligible |
| `_fail_closed_analytical` | `CAPABILITY_NOT_GENERIC` / `OPERATION_NOT_GENERIC` — the plan must authorise a current-position generic aggregate |
| explicit lens handling | `EXPLICIT_LENS` — any lens but the default population is ineligible |
| `_guard_unknown_category` | reads no question; it is a no-op on a plan-built envelope |

Each of those returns the question to the legacy path, where the guards run as
they always have.

**What the perimeter cannot see** is whether the executor did what the plan said.
So `reconcile` checks that before serving, reading the bound spec and the
executor's own receipt and no text at all:

* every predicate the plan stated appears in `applied_predicates`;
* every axis the plan stated appears in `group_field_keys`;
* the execution produced rows;
* the measured population (`filtered_row_count`) is not empty.

The last of those is a **presentation** decision, not a verdict on the
arithmetic: "0 loans · £0" is the correct answer to a question about a population
this book does not contain, and the legacy path already owns a proven controlled
way to say so. This slice adds no second one. It was found by the tests, not by
reading the code — a `count` over an empty book is a one-row SUMMARY holding
zero, so an emptiness check on the result frame passed it and the canary served
the zero.

## Evidence

The existing recorder, unredesigned. Every serving-canary request adds a
`serving` block:

```
principal_matched        true
principal_id             the matched actor_id
new_path_eligible        true | false
decision                 NEW | LEGACY_FALLBACK
reason                   "" | INTERPRETER_FAILURE | CLARIFY_NOT_SERVED_IN_THIS_SLICE
                         | REFUSE_NOT_SERVED_IN_THIS_SLICE | INELIGIBLE:<reason>
                         | EXECUTION_FAILED | PLAN_RECEIPT_RECONCILIATION_FAILED:<why>
                         | RENDER_FAILED | UNEXPECTED_ERROR
plan_id                  where a plan exists
response_served_from     NEW | LEGACY_FALLBACK
legacy_result_available  whether the legacy envelope answered
legacy_ok / legacy_value the control, for comparison
new_value                the deterministic figure
```

alongside the full `model` / `interpretation` / `compiler` / `eligibility` /
`execution` stages the shadow already records, including grouped cells. No
credential and no borrower row: `plan_shadow_evidence.redact` drops
credential-shaped keys, masks secret markers, and drops any DataFrame or Series
outright.

## Known narrowings, stated rather than discovered later

* **CLARIFY and REFUSE do not serve.** Out of scope for this slice by
  instruction; both fall back and are recorded.
* **Latency.** The canary principal pays the legacy path plus one live
  interpretation (10–26s measured in the live sign-off) plus one deterministic
  execution, synchronously. That is inherent to serving a plan-derived answer and
  is why this is one principal.
* **The shadow does not also run for the canary principal.** Exclusive by
  design: a second interpretation would double the spend and compare the new
  result against itself. Everybody else's shadow is untouched.
