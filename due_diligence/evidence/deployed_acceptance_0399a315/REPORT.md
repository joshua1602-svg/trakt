# Deployed Slice 1 shadow acceptance — Phases 0 and 3 done, Phase 1 blocked

`DEPLOYED_SLICE1_SHADOW_ACCEPTANCE = INCONCLUSIVE`

Not FAIL. Nothing about Slice 1 or 1A was found wanting. The acceptance cannot be
performed from this environment, and the brief's own rule for that case — prove
the preconditions before requests are sent, or STOP — is what was followed.

```
DEPLOY_CANDIDATE_SHA  0399a315b48a7cf6cf39403227a360703121db33   (matches the brief)
SERVED_SHA            NOT ESTABLISHED — nothing was deployed
DEPLOYED              NO
MI_BEARER_USED        NO — none available
MI_BEARER_PERSISTED   NO
SERVING_NEW_PATH      NO
SHADOW_DISABLED_AFTER_TEST  N/A — shadow was never enabled anywhere
```

Two artefacts were produced rather than nothing, because neither depends on
deployment and the brief requires both before the first shadow model call:
the Phase 0 provenance record, and the hashed Phase 3 acceptance bank.

Reproduce both with
`python due_diligence/evidence/deployed_acceptance_0399a315/preregister.py`.

---

## Phase 0 — provenance: PASS

```
DEPLOY_CANDIDATE_SHA                 0399a315b48a7cf6cf39403227a360703121db33
TREE_CLEAN                           YES
ADAPTER == ACCEPTED SLICE 1          YES
INTERPRETATION_V2 FROZEN since 2b00172  YES (code-only diff empty)
QUARANTINED BRANCH IS ANCESTOR       NO
SHADOW DEFAULT                       off
NO WILDCARD OPENS THE CANARY         YES
```

Blob identities for the operator to check against whatever actually gets
deployed:

| file | blob |
|---|---|
| `mi_agent/plan_runtime_adapter.py` | `e73196f69baf4405945263cb89374309eb599922` — identical to accepted Slice 1 |
| `mi_agent/plan_shadow_wiring.py` | `855653aef49297e84e02899c34b556f6c52cbb4c` |
| `mi_agent/plan_shadow_evidence.py` | `45b22898fda4ee5819bb2a6db8878d31ed11adc6` |
| `mi_agent_api/mi_service.py` | `1b4784c79c84186affec6dffa68c038498a4851b` |

The gate perimeter is recorded in the manifest as data —
`generic_analysis`, `{breakdown, point_in_time}`, `{current}`, ≤2 dimensions —
so a later drift is detectable by comparison rather than by memory.

**The no-wildcard property is run, not asserted.** Seven attempts, each with the
resulting allow-list and whether an unlisted client would be shadowed:

| setting | allow-list | unlisted client shadowed |
|---|---|---|
| `*` / `all` / `any` / `everyone` | empty (each logged and ignored) | **No** |
| `*,acme` | `{acme}` | **No** |
| ` ALL , Any ` | empty | **No** |
| `**` | `{**}` — a literal id | **No** |

One honest detail: `**` is not in `FORBIDDEN_CANARY_TOKENS`, so it survives as a
*literal* client id. It still matches nothing, because the canary compares exact
strings and does no globbing. Recorded rather than glossed.

## Phase 3 — the acceptance bank, pre-registered and hashed

`acceptance_manifest.json`, sha256
`f33348a28c08a8cf2ddc24dc83250f82834584101924baee18a318f3e59bdedc`.

```
BANK_SIZE                    12
ELIGIBLE_EXPECTED             8
INELIGIBLE_EXPECTED           4
DISTINCT_INELIGIBLE_REASONS   CAPABILITY_NOT_GENERIC, EXPLICIT_LENS,
                              GEOGRAPHY_REQUESTED, OPERATION_NOT_GENERIC
```

| case | frozen id | covers | expected |
|---|---|---|---|
| A01 | Q01A | count, two numeric filters | `EXECUTED` |
| A02 | Q02A | sum balance, two numeric filters | `EXECUTED` |
| A03 | Q03A | count, governed categorical value + numeric | `EXECUTED` |
| A04 | Q12A | sum by LTV bucket × age bucket, grouped | `EXECUTED` |
| A05 | Q11A | sum by LTV bucket × **ticket** bucket | `EXECUTED` if the deployed frame carries it, else `EXECUTION_ERROR` |
| A06 | Q13A | sum by LTV bucket × **interest-rate** bucket | same conditional |
| A07 | Q01B | paraphrase stability pair with A01 | `EXECUTED` |
| A08 | Q02C | paraphrase stability pair with A02 | `EXECUTED` |
| B01 | Q04A | explicit Direct lens | `INELIGIBLE / EXPLICIT_LENS` |
| B02 | Q14A | governed geography axis | `INELIGIBLE / GEOGRAPHY_REQUESTED` |
| B03 | Q19A | specialist capability, historical | `INELIGIBLE / CAPABILITY_NOT_GENERIC` |
| B04 | Q12C | unsupported shape (`distribution`) | `INELIGIBLE / OPERATION_NOT_GENERIC` |

Every question is verbatim from the successful 20-question local live bank, and
every expectation — interpretation disposition, governed semantics, eligibility,
gate reason, expected `plan_id` — is read out of the frozen run-8 sign-off. None
was invented to fill a category.

### Two things the manifest states up front rather than leaving to be discovered

**Independent numerical truth is NOT available on deployed data.** This harness
cannot see the production portfolio, so there is no oracle figure to compare
against, and every eligible case is pre-registered `TRUTH_UNAVAILABLE` for its
value. A figure matching the legacy answer is a signal and not a pass — the legacy
answer is not the oracle. What does adjudicate, per case, and is named in the
manifest: the interpretation against the frozen expectation, the eligibility
against the gate, the plan-to-spec facet comparison for silent measure/filter/
dimension drops, and the grouped grid reconciled against the execution receipt's
own row counts.

**A05 and A06 are pre-registered with both branches acceptable.**
`ticket_bucket` and `interest_rate_bucket` are governed registry fields that the
local oracle book does not carry; whether the deployed funded frame carries them
is a dataset fact I cannot know from here. Either `EXECUTED` or an
`EXECUTION_ERROR` naming the missing column is correct, and the latter is
`TRUTH_UNAVAILABLE`, never a `DETERMINISTIC_EXECUTION_DEFECT`. Saying so now is
the difference between a pre-registration and a rationalisation.

### The async-evidence policy, fixed before the run

```
interval           2 seconds
timeout            60 seconds per request
on timeout         ASYNC_EVIDENCE_LOST -> that case is INCONCLUSIVE;
                   never a silent pass, never a semantic FAIL
association        evidence.request.question + request.client_id
```

Association is by question and client, **not** by correlation id: the id is
generated server-side inside the shadow and is not returned on the HTTP response,
so the caller cannot quote it. All 12 questions are distinct strings, so that is
unambiguous for this bank — a bank repeating a question would need a time-window
tiebreak, and this one does not. Recorded as a limitation of the current design,
not fixed (no product changes are authorised here).

## Phase 1 — blocked: this environment cannot deploy or reach the deployment

Measured, not inferred, and filed verbatim in `deployment_capability.txt`:

```
az CLI                                            NOT INSTALLED
Azure / deploy / MI_BEARER environment variables   NONE
MI_BEARER recoverable from the session             0 occurrences
GET https://trakt-mi-api.azurewebsites.net/health   http=000
GET https://trakt-ops-api.azurewebsites.net/health  http=000
```

The `000` is the egress proxy refusing, and it says so itself:

```
{"kind": "connect_rejected",
 "detail": "gateway answered 403 to CONNECT (policy denial or upstream failure)",
 "host": "trakt-mi-api.azurewebsites.net:443"}
```

This was checked **once**, because this brief asks for a deployment and a prior
instruction not to retry cannot stand against a new instruction to do the thing.
The result is unchanged from the earlier preflight: no deployment rights, no
credential, no network path.

Consequently: Phase 1 cannot start, Phase 2 (prove the evidence sink is writable
*and* readable from the deployment) cannot be attempted, and Phases 4 through 9
have no deployment to act on. No live model call was made, which is also why
**no shadow needed disabling afterwards** — it was never enabled anywhere.

## Output

```
DEPLOY_CANDIDATE_SHA = 0399a315b48a7cf6cf39403227a360703121db33
SERVED_SHA           = NOT ESTABLISHED

SHADOW_OFF:
  health              = NOT REACHED (proxy denied CONNECT, 403)
  authenticated_smoke = NOT RUN (no MI_BEARER, no network path)
  model_calls         = 0
  evidence_records    = 0

CANARY:
  client_count             = 0 (never configured anywhere)
  isolation                = NOT EXERCISED in deployment; proved locally in 1A
  non_canary_model_calls   = 0

ACCEPTANCE:
  bank_size                 = 12  (pre-registered and hashed)
  successful_http_requests  = 0
  shadow_plans_created      = 0
  eligible                  = 0 executed (8 expected)
  ineligible                = 0 recorded (4 expected)
  async_evidence_lost       = 0
  model_failures            = 0
  transport_failures        = 0
  auth_failures             = 0
  retries                   = 0
  model_substitutions       = 0

INTERPRETATION:  exact = 0   semantically_equivalent = 0   deviations = 0   silent_drops = 0
EXECUTION:       attempted = 0   independently_correct = 0   numerical_errors = 0
                 semantic_drops = 0   grouped_cell_errors = 0
LEGACY_V_NEW:    exact_semantic_parity = 0   new_correct_legacy_differs = 0
                 unexplained_divergence = 0
ISOLATION:       user_visible_diffs = 0   shadow_results_served = 0
                 exception_escapes = 0

RAW_EVIDENCE_COMPLETE = N/A — there were no shadowed requests to evidence
```

**Every zero above is a zero because nothing ran**, not because something was
measured and found clean. None of them is a pass, and the Phase 8 critical proofs
are all **UNPROVEN in deployment**: the served SHA, the deployed model identity,
the legacy-bypass counts, and the in-flight bound. What is established at this
same product SHA, locally and not offered as a substitute: 0 legacy calls by both
the profiler and the 19 entry-point shims; `claude-opus-5` returned on all 20 live
calls with 0 substitutions in the earlier live-integration phase; and the
in-flight cap of one proved by test. None of that is a deployed measurement.

## Next step

`FAILURE_OWNER` = not a failure. One blocker, owned by the execution environment:
no deployment tooling, no credentials, and an egress policy that denies the host.
Unlike the previous preflight, there is **no product-side blocker this time** —
Slice 1A closed the inert-shadow gap, and the wiring, canary, recorder and
failure isolation are all proved locally at this exact SHA.

`LAST_SAFE_PRODUCT_SHA` = `0399a315` — shadow default off, canary fail-closed,
nothing served.

`SHADOW_DISABLED` = YES, trivially: never enabled.

`RECOMMENDED_ACTION` — run this acceptance unchanged from an environment with
deployment rights, egress to `trakt-mi-api.azurewebsites.net`, and a freshly
issued MI_BEARER. Everything that can be prepared in advance is prepared:

1. Deploy exactly `0399a315` and check the four blob hashes above against what
   the deployment reports.
2. Phase 2 first, before any model call: set
   `MI_AGENT_PLAN_SHADOW_EVIDENCE` to a path outside any checkout and prove the
   app can write it **and** that the operator can read it back. The recorder
   refuses a sink inside a git working tree by design, so a checkout path will
   fail closed and count an evidence failure rather than writing.
3. `MI_AGENT_PLAN_SHADOW=shadow`, `MI_AGENT_PLAN_SHADOW_CLIENTS=<one client>`,
   `MI_AGENT_PLAN_SHADOW_MAX_IN_FLIGHT=1`.
4. Run the 12 pre-registered questions, polling the sink at 2s up to 60s per
   request, and associating records by question plus client.
5. Set `MI_AGENT_PLAN_SHADOW` back to off afterwards, pass or fail.

Stopped here. Nothing deployed, nothing served, no product code changed, Slice 2
not begun.
