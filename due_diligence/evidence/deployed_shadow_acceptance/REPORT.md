# Slice 1 — deployed shadow acceptance: STOPPED before Phase 1

`DEPLOYED_SLICE1_SHADOW_ACCEPTANCE = INCONCLUSIVE`

Not FAIL. Nothing previously signed off was found to be wrong. The phase stopped
because it cannot be performed from here, and because a product gap found in
Phase 0 would leave Phases 4-6 with nothing to observe even on a successful
deployment.

```
DEPLOY_CANDIDATE_SHA           5a6035fb17a15963f85cff22afe9e2344c8b4e93
SERVED_SHA                     NOT ESTABLISHED — nothing was deployed
PRODUCT_TREE_EQUIVALENT        YES, to 6f42df67
MI_BEARER_USED                 NO — none available (see Blocker 1)
MI_BEARER_PERSISTED            NO
DEPLOYED                       NO
SERVING_NEW_PATH               NO
```

Reproduce both findings with
`python due_diligence/evidence/deployed_shadow_acceptance/inert_shadow_proof.py`;
its machine-readable output is `phase0_and_finding.json`.

---

## Phase 0 — pre-deploy provenance: PASS

```
DEPLOY_CANDIDATE_SHA                      5a6035fb17a15963f85cff22afe9e2344c8b4e93
TREE_CLEAN                                YES
PRODUCT_TREE_EQUIVALENT_TO_6f42df67       YES
  every differing file is evidence-only   YES — 8 files, all under due_diligence/
INTERPRETATION_V2 frozen since 2b00172    YES (code-only diff is empty)
SHADOW_DEFAULT_OFF                        YES (shadow_mode() == "off" unset)
QUARANTINED_BRANCH_IS_ANCESTOR            NO (3399d2ef absent)
```

The four files that *are* Slice 1 are compared blob by blob, not by directory
diff, so evidence commits cannot mask a product change:

| file | blob at `6f42df67` and at HEAD |
|---|---|
| `mi_agent/plan_runtime_adapter.py` | `e73196f69baf4405945263cb89374309eb599922` |
| `mi_agent_api/mi_service.py` | `e4948bafd1e2e99258d99d496980da1c943b9296` |
| `mi_agent/tests/test_plan_runtime_adapter.py` | `6bb08ed81be9b2de525228f28cc32ba14d4c0526` |
| `mi_agent/tests/test_plan_shadow_does_not_serve.py` | `5c6d37746897066eeac5ce4512336d198c279b73` |

The accepted suites still pass at this SHA: 55 + 12 = **67**.

---

## Blocker 1 — this environment cannot deploy or reach the deployment

Recorded verbatim, not inferred:

```
az CLI                                          NOT INSTALLED
Azure / deploy / MI_BEARER environment vars      NONE
MI_BEARER recoverable from the session           0 occurrences
GET https://trakt-mi-api.azurewebsites.net/health    http=000
GET https://trakt-ops-api.azurewebsites.net/health   http=000
```

The `000` is a **policy denial, not a flaky network**. The agent proxy answered
`403` to the CONNECT, and its own status endpoint records it:

```
{"kind": "connect_rejected",
 "detail": "gateway answered 403 to CONNECT (policy denial or upstream failure)",
 "host": "trakt-mi-api.azurewebsites.net:443"}
```

So Phase 1 (deploy with shadow off) cannot start, Phase 5 (authenticated
`POST /mi/query`) cannot be attempted, and no `SERVED_SHA` can be established.
No MI_BEARER was available and none was sought beyond checking the environment
and the session — a token from an earlier transcript would not be the "freshly
valid" one the brief requires even if one had existed.

This is an environment capability limit. It is not evidence about Slice 1.

---

## Blocker 2 — the deployed shadow is inert, and this is the material finding

**With the flag fully on, a deployed request would produce no plan, no shadow
execution and no ledger row.** Structurally and empirically:

*The call site passes no plan.* `mi_agent_api/mi_service.py:1924`:

```python
_plan_shadow.observe(result=result, frame=df, semantics=semantics, view=view,
                     portfolio_id=authorised.portfolio_id)
```

*The only other way in is unused in production.* `observe` will consult
`_PLAN_PROVIDER` when no `plan` is given, and `set_plan_provider` is the only
thing that installs one. Its callers, repo-wide:

```
mi_agent/plan_runtime_adapter.py:513      the definition
mi_agent/tests/test_plan_shadow_does_not_serve.py   8 call sites
```

**Production callers: none.** And nothing in the request path imports
`interpretation_v2` at all — `plan_runtime_adapter` mentions it only in its
docstring, and a Slice 1 test asserts the module imports no interpreter.

*Empirically, with `MI_AGENT_PLAN_SHADOW=shadow` and a writable ledger, using
exactly the arguments above:*

```
shadow_mode()                 shadow
observe() returned            None
ledger file created           False
ledger rows                   0
served envelope unchanged     True
```

### What this means for the brief's phases

| phase | status |
|---|---|
| 4 — enable shadow only | the configuration mechanism **does** distinguish shadow from serving (there is no serving mode at all), but enabling shadow changes nothing observable |
| 5 — authenticated acceptance | blocked by Blocker 1; and would produce no shadow evidence even if unblocked |
| 6 — offline adjudication | nothing to adjudicate: no interpretation, no plan, no eligibility decision, no shadow result would exist for any request |

The absolute serving rule is satisfied in the strongest possible way — the new
path cannot serve because it does not run — but that is not what the phase set
out to prove.

### This is my error to own

My Slice 1 sign-off recommended exactly this phase: "deploy Slice 1 with the new
path still shadow-only and default-off ... then perform a small authenticated
`/mi/query` shadow acceptance." That recommendation was wrong, and wrong in a way
I could have caught without a deployment: I never checked that the deployed
shadow could obtain a plan. The Slice 1 gates that exercised the shadow all
injected one through `set_plan_provider`, which its own docstring calls a
"test/replay seam", and Gate 3 only ever asserted that the **off** path builds no
plan — never that the **on** path builds one.

The call-site comment I wrote is also misleading and should be corrected whenever
the wiring is added. It says:

> When it is on, the governed-plan path runs against the frame this request
> already resolved and writes a comparison row

That describes the intended seam, not the wired behaviour. Per this brief's
`DO NOT FIX` rule I have changed nothing — not the comment, not the adapter, not
the call site.

---

## Critical deployment checks

Honest status for each, with nothing recorded as a zero that was not measured:

| # | check | status |
|---|---|---|
| 1 | deployed model returned `claude-opus-5` | **UNPROVEN** — nothing deployed; and with the shadow inert, a deployed request makes no model call on the new path at all |
| 2 | deployed code/provenance is the candidate | **UNPROVEN** — no deployment, no `SERVED_SHA` |
| 3 | production config did not enable legacy parsing on the new path | **VACUOUS** — the new path does not execute |
| 4 | the adapter receives the intended caller-resolved funded frame | **PROVEN at source**: the call site passes `frame=df`, the frame this request already resolved, and `view=view` |
| 5 | Slice 1 does not use Direct/Acquired resolution | **PROVEN** (accepted Slice 1): an explicit lens is `EXPLICIT_LENS`, refused |
| 6 | Slice 1 does not invoke temporal/snapshot routing | **PROVEN** (accepted Slice 1): a non-current period is `PERIOD_NOT_CURRENT`, refused |
| 7 | core geography still comes from the asset-config binding | **PROVEN** (accepted Slice 1): a governed geography binding is `GEOGRAPHY_REQUESTED`, refused — the adapter binds no geography |
| 8 | no shadow exception affects the API response | **VACUOUS in deployment**; proven locally in Slice 1 Gate 4 and the live sign-off, including a forced executor failure on a live plan |
| 9 | no shadow result is presented to the user | **PROVEN, trivially** — no shadow result exists |
| 10 | complete raw semantic evidence exists for every call | **NO** — there were no calls |

## Legacy bypass proof

Every item is **UNPROVEN in deployment**, and reported that way rather than as a
zero: `ParsedQuestion.parse`, `llm_query_parser`, `question_interpretation`,
`concept_merge_arm`, `RecogniserRegistry`, `chat_routing` semantic routing,
raw-text lens resolution, raw-text dimension reconstruction. There was no
deployed new-path execution to instrument.

What *is* established, from the live integration sign-off at this same product
SHA: across 20 live interpretations and 8 deterministic executions, a
`sys.setprofile` hook recorded **0** calls into nine legacy semantic modules and
19 named entry-point shims counted **0**. That is a local proof about the same
code, not a deployed one, and it is not offered as a substitute.

---

## Output

```
DEPLOY_CANDIDATE_SHA = 5a6035fb17a15963f85cff22afe9e2344c8b4e93
SERVED_SHA           = NOT ESTABLISHED
PRODUCT_TREE_EQUIVALENT = YES (to 6f42df67; all 8 differing files are evidence)

SHADOW_OFF:
  health                = NOT REACHED (proxy policy denied CONNECT, 403)
  authenticated_smoke   = NOT RUN (no MI_BEARER, no network path)
  legacy_diffs          = NOT MEASURED
  shadow_invocations    = NOT MEASURED in deployment; 0 by construction locally

ACCEPTANCE:
  bank_size             = 0 (not pre-registered — see below)
  successful_requests   = 0
  auth_failures         = 0 (no request was attempted)
  transport_failures    = 0 (no request was attempted)
  retries               = 0
  model_substitutions   = 0 (no model call occurred)

INTERPRETATION:  exact = 0  semantically_equivalent = 0  deviations = 0  silent_drops = 0
ELIGIBILITY:     expected_eligible = 0  actual_eligible = 0  mismatches = 0
SHADOW_EXECUTION: attempted = 0  independently_correct = 0  numerical_errors = 0
                  semantic_drops = 0  grouped_cell_errors = 0
LEGACY_V_NEW:    exact_semantic_parity = 0  new_correct_legacy_differs = 0
                 unexplained_divergence = 0
ISOLATION:       user_visible_diffs = 0  exception_escapes = 0  shadow_results_served = 0

RAW_EVIDENCE_COMPLETE = NO — there were no calls to evidence
```

Every zero above is a zero *because nothing ran*, not because something was
measured and found clean. They are not a pass.

The Phase 2 acceptance bank was deliberately **not** pre-registered. Its
"expected shadow disposition" column cannot be written until the wiring decision
in Blocker 2 is made — whether the request path builds a plan per request, or
asynchronously, or only for a sampled fraction changes what every row should
expect. Pre-registering a manifest now and revising it later would defeat the
point of pre-registration.

`FAILURE_OWNER` = not a failure; two blockers, owned as follows. Blocker 1 is the
execution environment (no deployment tooling, no credentials, network policy
denies the host). Blocker 2 is mine: Slice 1 built the shadow seam for replay and
I recommended a deployed acceptance without checking that a deployed request
could reach it.

`LAST_KNOWN_SAFE_STATE` = `6f42df67` — accepted Slice 1, shadow default off,
nothing served, still byte-equivalent at this HEAD.

`RECOMMENDED_ACTION` — in order:

1. **Decide the production wiring for the shadow** as a small, explicit slice of
   its own: what builds a `GovernedQueryPlan` for a live request, whether it runs
   inline or out of band, and at what sampling rate. This is a design decision
   with a cost attached — one Opus interpretation per shadowed request — so it is
   yours to make, not mine to assume. Correct the call-site comment in the same
   change.
2. **Build the evidence recorder before the wiring goes live**, to the Phase 3
   schema in this brief: the complete raw model response, the CandidateIntent,
   the full CompileResult with bindings and normalisation, the plan and
   `plan_id`, every eligibility reason, the requested and resolved execution
   evidence with grouped cells, and the legacy disposition — and no credentials
   or borrower rows. This is the I04 lesson, and it has to exist before the calls,
   not after.
3. **Then run this phase**, from an environment with deployment rights, network
   access to the app and a freshly issued MI_BEARER.

Stopped here. No product code changed, nothing deployed, nothing served, Slice 2
not begun.
