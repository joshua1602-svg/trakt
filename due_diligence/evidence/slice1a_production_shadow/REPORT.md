# Slice 1A — production shadow wiring

`SLICE_1A_PRODUCTION_SHADOW_WIRING = PASS`

```
START_SHA  e8cc9405   (product tree byte-equivalent to accepted slice 1 6f42df67)
FINAL_SHA  see COMMITS below; product base unchanged at 6f42df67 + this slice
LIVE_MODEL_CALLS 0      MI_BEARER_USED NO      DEPLOYED NO      SERVING_NEW_PATH NO
```

## What changed

```
COMMITS                3, not squashed
  01e0cabe   the evidence recorder + 23 tests
  4a599421   the plan-builder wiring, the canary, the call site + 38 tests
  (this one) failure-isolation and regression evidence

PRODUCT_FILES_CHANGED  1 existing  — mi_agent_api/mi_service.py (+22 / -9)
                       2 new       — mi_agent/plan_shadow_evidence.py   (299)
                                     mi_agent/plan_shadow_wiring.py     (427)
PRODUCT_NET_LOC        +748 / -9
TESTS                  +1065 across 3 files, +1 guard extended
```

Well inside the brief's perimeter of three existing product files. Unchanged and
proved so:

| | |
|---|---|
| `mi_agent/plan_runtime_adapter.py` | **byte-identical** to `6f42df67` — `e73196f69baf4405945263cb89374309eb599922`, asserted by a test in the slice, not just by inspection |
| `mi_agent/interpretation_v2/**` | untouched; still byte-identical to the run-8 sign-off at `2b00172` |
| the eligibility perimeter | untouched; the gate's own constants are recorded into every evidence row so a later drift is visible |

```
PLAN_BUILDER_ENTRYPOINT  mi_agent.plan_shadow_wiring.build_plan
                         -> interpretation_v2.opus_interpreter.OpusInterpreter
                         -> interpretation_v2.compiler.DeterministicCompiler
SHADOW_CALL_SITE         mi_agent_api/mi_service.py :: _run_analysis,
                         immediately before the point-in-time return
CANARY_MECHANISM         MI_AGENT_PLAN_SHADOW_CLIENTS — a comma-separated
                         allow-list of CLIENT ids, following the estate's
                         existing app-setting convention
                         (react_auth._dashboard_directories); fail-closed
EVIDENCE_RECORDER        mi_agent.plan_shadow_evidence, sink
                         MI_AGENT_PLAN_SHADOW_EVIDENCE
```

## Phase 1 — what the flag is allowed to cost

A flag that meant "call Opus for every production MI request" would be a
liability, not a canary. Three independent bounds, each with a test:

| state | what happens |
|---|---|
| `MI_AGENT_PLAN_SHADOW` ≠ `shadow` | nothing. No interpreter is constructed, no evidence row, no ledger row |
| on, client not in the allow-list | no model call, no evidence row. The list is **fail-closed**: unset shadows nobody |
| on, in the allow-list | one interpretation, and **at most one shadow in flight** (`MAX_IN_FLIGHT`, default 1). Concurrent canary traffic is skipped-and-recorded, never queued |

`*`, `all`, `any` and `everyone` are **refused by name** and logged, so there is
no value of the setting that opens the canary to everybody: widening it is an act
of configuration per client and cannot happen by accident.

**The canary cannot read the question.** It sees the client id and nothing else.
A test strips docstrings and comments from `in_canary` and `canary_clients` and
asserts no `question`, `text`, `semantic` or `answer` identifier appears in the
code itself — tying a rollout's coverage to what was asked is exactly the
confusion this architecture exists to remove.

### One architectural decision, stated rather than buried

**The shadow runs off the request path, in a daemon thread.** A live
interpretation measured 10–26 seconds in the live sign-off. Spending that inline
would be influencing the served response by any honest reading, however identical
the content — gateway timeouts and a visibly slower canary client are not
"no user-visible difference". The served answer is already fully computed when the
shadow is dispatched.

Two consequences, both the operator's to know:

* the deployed acceptance harness must **wait for, or poll, the evidence sink** —
  the HTTP response returns before the shadow finishes;
* a worker recycled mid-shadow loses that record. Nothing is corrupted; the
  evidence is simply absent, and the in-flight counter dies with the process.

Dispatch is a module seam so tests run inline and deterministically, and a test
asserts the production default is the background dispatcher.

## Phase 2 — the plan builder

`question → frozen interpreter → CandidateIntent → frozen compiler →
GovernedQueryPlan | CLARIFY | REFUSE`, and nothing else. Exactly the sequence
`interpretation_v2.benchmark.run_benchmark` uses.

The interpreter is built with `AnthropicInterpreterClient()`'s own default model
and an **unnarrowed** `CompilerContext()` — the same construction the 135-question
sign-off and the live integration run used. That is deliberate: a deployed plan
must be the plan those runs proved. Narrowing the vocabulary to the book would be
a semantic change, and this slice is not one.

**Exactly one interpretation per shadowed request**, asserted by a counting test:
no retry, no reworded second attempt, no repair loop. Transport retries remain
the model client's own business.

Nothing between the compiler and the gate edits the plan — a test asserts the
wiring's code contains no `plan[`, `plan.pop`, `plan.update` or `del plan`.

## Phase 3 — the evidence, written before anything could use it

Commit 1 is the recorder, and it landed before the wiring existed. The record
carries every stage whole:

```
EVIDENCE_RAW_PLAN_COMPLETE            YES   compiler.plan, with provenance:
                                            intent_claims, compiler_bindings,
                                            normalisation, vocabulary_version
EVIDENCE_RAW_CANDIDATEINTENT_COMPLETE YES   interpretation.candidate_intent,
                                            plus ambiguities broken out
EVIDENCE_COMPILE_RESULT_COMPLETE      YES   outcome, reasons, reason_codes,
                                            compiler_version, plan_id
EVIDENCE_EXECUTION_COMPLETE           YES   requested_semantics, bound_spec,
                                            value, receipt (with
                                            applied_predicates), grouped cells,
                                            warnings, error
```

Also: the complete raw model response, token usage, the governed metadata lookups
the model made, the gate's own perimeter constants, and the legacy control's
disposition, value and route. A grouped record measures ~10.8 KB.

Never persisted, and structurally rather than by promise: a key whose name reads
as a credential is dropped; a value carrying a secret marker is masked so the
field still shows; **a DataFrame or Series is dropped outright**, because a frame
is the one shape a borrower row could arrive in. Every removal is recorded as a
redaction note. A sink path **inside a git working tree is refused** — production
evidence accumulating in a checkout is how a secret reaches a commit.

A missing sink is a configuration choice and not a fault; an unwritable one
increments a counter and logs, so an evidence failure is observable as a failure
rather than as an absence.

## Phase 4 — failure isolation, measured

`capture_dispositions.py` runs the production orchestration for every reachable
disposition against replayed frozen payloads. `disposition_matrix.json` holds the
matrix and two complete records verbatim.

| scenario | disposition | served envelope | evidence | detail |
|---|---|---|---|---|
| eligible scalar | `EXECUTED` | **SAME** | persisted | 195.0, the independent oracle's figure |
| eligible grouped | `EXECUTED` | **SAME** | persisted | 20 cells |
| ineligible | `INELIGIBLE` | **SAME** | persisted | `EXPLICIT_LENS`, adapter never executed |
| clarify | `CLARIFY` | **SAME** | persisted | no plan, no gate call, no ledger row |
| interpreter failure | `INTERPRETER_FAILURE` | **SAME** | persisted | the failure reason recorded |
| executor failure | `EXECUTION_ERROR` | **SAME** | persisted | `MIQueryExecutionError` captured |
| outside canary | `OUTSIDE_CANARY` | **SAME** | not written | no model call |

```
USER_VISIBLE_RESPONSE_DIFFS   0
SHADOW_RESULT_SERVED          0
SHADOW_EXCEPTION_ESCAPES      0
```

An interpreter that *raises* (rather than failing gracefully) is contained as
`ORCHESTRATION_ERROR` and **still leaves an evidence row** — a contained failure
that left no trace would be indistinguishable from a request that was never
shadowed.

## Phase 5 — nine scenarios, no live Opus

```
FLAG_OFF:            model_calls 0   plan_builds 0   adapter_calls 0
                     evidence_rows 0   served_diffs 0
FLAG_ON_NON_CANARY:  model_calls 0   plan_builds 0   evidence_rows 0
                     served_diffs 0
FLAG_ON_CANARY:      plan_created YES (frozen run-8 plan_id reproduced exactly)
                     eligible_execution YES   ineligible_recorded YES
                     clarify_recorded YES     refuse_recorded YES
                     failures_isolated YES (interpreter, orchestration, adapter,
                                            recorder)
```

The interpreter is injected as a `ReplayClient` over the frozen run-8 payloads, so
a test's plan is the plan the sign-off adjudicated — built through the real
production orchestration, not from a hand-made dict. The eligible case's figure is
checked against `portfolio_truth_oracle`, and the grouped case cell-for-cell.

## Phase 6 — legacy bypass on the new branch

```
NEW_PATH:
  ParsedQuestion.parse              0
  llm_query_parser                  0
  question_interpretation           0
  concept_merge_arm                 0
  RecogniserRegistry                0
  chat_routing semantic routing     0
  raw-text lens resolution          0
  raw-text dimension reconstruction 0
```

Proved two ways at once over **every disposition the branch can reach** —
executed scalar, executed grouped, and ineligible (a refusal must not quietly
fall back to a legacy reading):

* `sys.setprofile` counted 0 calls into nine legacy semantic modules;
* counting shims on 19 named entry points counted 0, and a test asserts **all 19
  were actually installed** — a shim that silently failed to attach proves
  nothing.

And a **control test** deliberately calls `portfolio_lens.names_total_scope` and
asserts both mechanisms register it. Without that, zero would mean nothing.

A static import-closure proof remains unavailable and is not claimed:
`mi_agent/__init__.py` imports `llm_query_parser` and `mi_agent_workflow` eagerly,
so they are in `sys.modules` whether used or not. That predates slice 1.

## Phase 7 — regression

Full capture, no truncation, every failing set diffed against the accepted
slice 1 baseline.

| suite | result | against baseline |
|---|---|---|
| focused slice 1A | **61 passed** (23 evidence + 31 wiring + 7 bypass) | new |
| accepted slice 1 | **67 passed** | unchanged |
| `mi_agent/tests` | 13 failed, **1558** passed, 264 skipped, 7 xfailed | failing set **IDENTICAL**; +61 passing, same 13 pre-existing failures |
| `mi_agent_api/tests` | 20 failed, **1820** passed, 7 skipped | failing set **IDENTICAL** (20 lines, raw) |
| broader changed-surface (33 files under `tests/`, `migration_phase0/`) | 51 failed, 473 passed | failing set **IDENTICAL** at `6f42df67` and at HEAD, run in a separate worktree |
| 65-case plan bank | 47 PASS / 11 FAIL / 1 OUTSIDE / 1 UNJUDGEABLE / 5 NOT_REACHABLE | result file **byte-identical** |
| 28-case scope bank | 4 PASS / 24 FAIL | result file **byte-identical** |
| Gate 5 corpus replay (945 interpretations) | 58 parity / 15 grouped parity / 0 differences | output **byte-identical** |

```
FLAG_OFF_RESULT_DIFFS        0
LEGACY_RESPONSE_DIFFS        0
SLICE1_ELIGIBILITY_DIFFS     0
SLICE1_ADAPTER_RESULT_DIFFS  0
INTERPRETATION_V2_HASH_DIFFS 0
UNEXPECTED_BLAST             0
CORE_GEOGRAPHY               5/5 PASS (C01–C05; C07 OUTSIDE_CURRENT_SCOPE)
```

### On the arithmetic-oracle figure

The brief asks for `ARITHMETIC_ORACLE = 22/22`. **No harness at HEAD produces a
22-case arithmetic tally**, and the figure appears in no committed report — so
rather than recite it, here is what is measurable and was measured:

* the adapter's independent-oracle execution tests: **16/16 pass**
  (`TestEligibleExecution`, figures checked against `portfolio_truth_oracle`,
  which imports nothing from the product);
* the arithmetic families of the 65-case plan bank —
  `AGGREGATION_STATISTIC` + `POPULATION_FILTER` — **28/28 PASS**, result file
  byte-identical to the accepted baseline;
* the Gate 5 replay's 73 independent agreements: **58 exact scalar parity + 15
  grouped cell parity, 0 differences**, output byte-identical.

If 22/22 refers to a harness I should be running, name it and I will.

## Corrections owed

**The call-site comment is fixed**, as the brief requires, and says what was false
about it: it had claimed the governed-plan path ran whenever the flag was on,
when the call passed no plan and nothing installed a provider.

**My Slice 1 Gate 6 diff was narrower than I implied.** It compared `.list` files
built with `grep -E "^(FAILED|ERROR)"`, which omits pytest's `SUBFAILED` lines —
9 of them in the API suite. Both sides were filtered identically, so the
comparison was valid for what it covered and the conclusion stands; but "failing
sets identical, diffed line by line" overstated the coverage. This slice diffs the
**raw** captures, and the API failing set is identical across all 20 lines
including every `SUBFAILED`.

**Two defects the new tests caught before anything ran for real**: asking an
`Ambiguity` for a `to_dict` it does not have, which raised and lost the entire
evidence record; and `getattr(frame, "columns", ()) or ()` in the grouped-cell
capture, where a pandas `Index` raises on a truth test — every grouped capture
would have recorded an empty grid. A third was in a test rather than the product:
a discipline check matching `import re` inside `import refuse`.

## Pass criteria

| criterion | required | measured |
|---|---|---|
| `FROZEN_INTERPRETATION_CHANGED` | NO | NO |
| `SLICE1_ADAPTER_CHANGED` | NO | **NO** — byte-identical, asserted by test |
| `SLICE1_ELIGIBILITY_CHANGED` | NO | NO |
| `SHADOW_DEFAULT_OFF` | YES | YES |
| `FLAG_OFF_MODEL_CALLS` | 0 | 0 |
| `NON_CANARY_MODEL_CALLS` | 0 | 0 |
| `CANARY_SHADOW_PLAN_CREATED` | YES | YES |
| `PLAN_SOURCE` | FROZEN_INTERPRETATION_V2 | FROZEN_INTERPRETATION_V2 |
| `RAW_TEXT_SEMANTIC_REREADS_NEW_PATH` | 0 | 0 |
| `LEGACY_RECOGNISER_CALLS_NEW_PATH` | 0 | 0 |
| evidence complete for PLAN / CLARIFY / REFUSE / INELIGIBLE / EXECUTION_ERROR | YES | YES, one real record each |
| `USER_VISIBLE_RESPONSE_DIFFS` | 0 | 0 |
| `SHADOW_RESULT_SERVED` | 0 | 0 |
| `SHADOW_EXCEPTION_ESCAPES` | 0 | 0 |
| `CORE_GEOGRAPHY` | 5/5 | 5/5 |
| `UNEXPECTED_BLAST` | 0 | 0 |

## What this slice does NOT establish

* **Nothing is deployed and nothing was called live.** `LIVE_MODEL_CALLS = 0` by
  construction — every test and capture replays frozen payloads. The first real
  model call on this path will happen in the deployed acceptance.
* **The background dispatcher has not run under production load.** It is proved
  correct inline and proved to be the default; concurrency beyond the in-flight
  cap of 1 is bounded by design rather than by measurement.
* **`REFUSE` as a compiler outcome is recorded via the interpreter-failure path**
  in the captured matrix. The frozen bank's only non-plan outcomes are CLARIFY,
  so a compiler REFUSE was forced through a malformed payload rather than drawn
  from a real question. The branch is exercised; the question behind it is
  synthetic.

`RECOMMENDED_NEXT_STEP` — perform the already-defined deployed Slice 1 shadow
acceptance from an environment with deployment rights, permitted network access
and a fresh MI_BEARER. No further local architecture work is required first. Set
`MI_AGENT_PLAN_SHADOW=shadow`, `MI_AGENT_PLAN_SHADOW_CLIENTS=<one client>`,
`MI_AGENT_PLAN_SHADOW_EVIDENCE=<a path outside any checkout>`, and remember that
the response returns before the shadow finishes — poll the sink.

Stopped here. Nothing deployed, nothing served, Slice 2 not begun.
