# Slice 1 — live Opus integration sign-off

`START_SHA` `6f42df67` · `END_SHA` `6f42df67` for product code (unchanged); the
evidence commits sit on top of it.

| | |
|---|---|
| `LIVE_SLICE1_INTEGRATION` | **PASS** |
| `TREE_CLEAN` | YES at start and at finish |
| `PRODUCT_CODE_CHANGED` | **NO** — `git diff 6f42df67 HEAD -- mi_agent/ mi_agent_api/ tests/` is empty |
| `INTERPRETATION_V2_CHANGED` | **NO** — all 15 code files byte-identical to the run-8 sign-off at `2b00172` |
| `MI_BEARER_USED` | NO |
| `DEPLOYED` | NO |

## Phase 0 — provenance

```
HEAD                                    6f42df6779ee2eff8d458e510265de30d7f44e71
TREE_CLEAN                              YES (git status --porcelain empty)
interpretation_v2 code vs 2b00172       15/15 files SAME (blob-hash compared one by one)
  opus_interpreter.py                   479003b8d6ea6a99f94a5d29e65825caf47d4680 at both SHAs
  the only diff under interpretation_v2 is the run-8 evidence the sign-off added
plan_runtime_adapter present            YES (e73196f69baf4405945263cb89374309eb599922)
shadow default                           OFF — shadow_mode() == "off" with the var unset
quarantined deterministic branch          NONE — 3399d2ef is not an ancestor of HEAD
```

## Phase 1 — the bank, pre-registered and hashed before any call

`live_bank_manifest.json`, sha256
`9d2044d357c7463536010ae90c080df2d708b257df3edd93327069ce908fb891`, **committed
in `6d35b849` before the first model call**, and verified byte-for-byte by the
live harness at the start of the run.

```
LIVE_BANK_SIZE           20
ELIGIBLE_EXPECTED        10
INELIGIBLE_EXPECTED      10   across 5 distinct reasons
EXECUTABLE_EXPECTED       8
FIXTURE_GAPS_EXPECTED     2
BANK_CHANGED_AFTER_MODEL_CALLS   NO (hash re-verified in the run record)
```

Every question string is taken verbatim from the frozen run-8 sign-off, and every
expectation — disposition, governed semantics, eligibility and reason,
deterministic disposition, independent figure or grouped cells — was read out of
that frozen evidence or computed by the Gate 5 independent control.

**Four coverage items the brief asks for are not available from the frozen bank,
and none was manufactured.** The manifest names each with its reason:

| asked for | why not available |
|---|---|
| mean | no bank question compiles to statistic `average` |
| weighted mean | only Q05 does, and it states the Direct lens → pre-registered as control I02 |
| one filter alone | every eligible bank case carries two predicates or none |
| one dimension alone | every eligible breakdown carries two axes |
| configured core-region axis | Q14/Q16 request a governed geography binding, which became a slice 1 ineligibility when the dropped-axis defect was fixed → pre-registered as control I03 |

Only 5 of the gate's 15 reasons are reachable from this bank. `PERIOD_NOT_CURRENT`
is unreachable (no bank question is `generic_analysis` with a non-current period);
`NOT_SINGLE_OUTPUT` and `TOO_MANY_DIMENSIONS` are masked, because the only
multi-output and three-axis plans (Q17) also state the Direct lens and the lens
check is declared first. Q17A was pre-registered expecting `EXPLICIT_LENS` anyway,
so a gate answering `NOT_SINGLE_OUTPUT` instead would have been caught. Those
three boundaries are covered instead by the 55 focused unit tests, which put
synthetic plans to `check_eligibility` directly.

## Live call budget

```
MODEL_CALLS              20   (authorised 25)
SUCCESSFUL               20
TRANSPORT_FAILURES        0
RETRIES                   0
MODEL_SUBSTITUTIONS       0   — claude-opus-5 returned on all 20
UNAUTHORISED_CALLS        0   — no probe, no rewording, no diagnostic, no follow-up
tokens                   152,812 in / 23,737 out (plus 548,268 cache reads)
latency                  median 16.6s, range 10.1–25.8s
```

The harness was **dry-run first** through `ReplayClient` on the frozen payloads so
its mechanics were proved before a penny was spent — and that dry run found a
harness defect, described under *Harness defects* below.

## Phase 2 — live interpretation

```
INTERPRETATION:
  exact_plan_match          16
  semantically_equivalent    1
  justified_clarify          1
  justified_refuse           0
  deviations                 2
  silent_semantic_drops      0
```

**All ten eligible positives came back EXACT_PLAN_MATCH, with `plan_id`
byte-identical to the frozen sign-off.** A fresh live interpretation of each of
those questions reproduced the signed-off governed plan exactly — measure,
statistic, every filter with its comparator and value, every dimension, the
population base and lens, the period form, the comparison kind and the geography
facet.

All three interpretation movements landed on **ineligible controls**, and in every
one the eligibility verdict still matched the pre-registration:

| case | question | movement | gate verdict |
|---|---|---|---|
| I05 Q17A | "For the Direct book, show balance by LTV bucket, ticket-size bucket and borrower-age bucket." | every projected facet identical, different `plan_id` → representational only | `EXPLICIT_LENS`, as pre-registered |
| I08 Q23A | "When will we reach £100m of funded loans?" | `population_base` `forecast` → `funded` | `CAPABILITY_NOT_GENERIC`, as pre-registered |
| I04 Q12C | "Plot portfolio balance across LTV buckets and borrower-age buckets." | one output with two axes → **two outputs**, the first carrying one axis | `OPERATION_NOT_GENERIC`, as pre-registered |

I08 reads more literally than the frozen run did, not less: the question says
"funded loans" and the live plan's population base is `funded`, where the frozen
plan carried `forecast` — the capability name in the population slot. Both are
`capability: forecast` and both refuse at the same gate.

I04 is the one movement whose full shape **this run cannot settle from its own
evidence**, and the limit is the harness's, not the product's — see below.

## Phase 3 — slice 1 eligibility

```
ELIGIBILITY:
  correct      20 / 20
  mismatch      0
```

Every live plan drew the exact reason the manifest pre-registered, including all
five reachable reasons: `EXPLICIT_LENS` (I01, I02, I05), `GEOGRAPHY_REQUESTED`
(I03), `OPERATION_NOT_GENERIC` (I04), `CAPABILITY_NOT_GENERIC` (I06–I09),
`NOT_A_PLAN` (I10, a live CLARIFY). **No ineligible control became eligible**, and
no eligible positive was refused.

## Phase 4 — deterministic execution against independent truth

```
EXECUTION:
  attempted                 8
  independently_correct     8
  numerical_errors          0
  grouped_cell_errors       0
  semantic_drops            0   (0 filter drops, 0 dimension drops, 0 measure drops)
  fixture_gaps              2   (pre-registered as such)
```

| case | shape | figure | independent control |
|---|---|---|---|
| E01, E02 | count, age > 55 ∧ LTV > 50 | 195 | 195 |
| E03, E04 | sum balance, age > 75 ∧ LTV > 40 | £37,948,448.76 | £37,948,448.76 |
| E05, E06 | count, product = drawdown ∧ LTV > 50 | 93 | 93 |
| E07, E08 | sum balance by LTV bucket × age bucket | 20 cells | 20 cells, every one within 0.01 |

E05/E06 matter beyond their arithmetic: the plan carries the governed canonical
value `drawdown` while the frame carries the display form `Drawdown`, and the
engine resolved it — the same case whose control I got wrong during Gate 5.

E09 and E10 are `UNEXECUTABLE_FIXTURE_GAP` exactly as pre-registered:
`ticket_bucket` and `interest_rate_bucket` are real governed registry fields the
independent replay book does not carry. Neither a pass nor a product failure, and
the fixture was not touched during the run.

## Phase 5 — legacy bypass

```
NEW PATH:
  legacy_parser_calls          0
  recogniser_calls             0
  raw_text_semantic_rereads    0
```

Measured two ways at once, for the whole per-case pipeline **including the live
model call**:

* `sys.setprofile` recorded every call whose file is one of nine legacy semantic
  modules — `parsed_question.py`, `llm_query_parser.py`, `recogniser_registry.py`,
  `chat_routing.py`, `portfolio_lens.py`, `semantic_resolver.py`,
  `mi_agent_workflow.py`, `execution_receipt.py`, `mi_query_contract.py`. **Total
  calls into all nine: 0.**
* 19 named entry points were wrapped in counting shims for the entire run —
  `ParsedQuestion.parse`, four `llm_query_parser` entry points plus `find_field`,
  `RecogniserRegistry.candidates`/`.ordered`, three `chat_routing` semantic
  predicates, five raw-text lens resolvers, `requested_dimension_terms`,
  `dimension_role`, and `run_mi_agent_query`. **Total calls: 0.** All 19 were
  found and wrapped; none was silently missing.

Two disclosures. `sys.setprofile` sees only the calling thread, which is why the
shims exist as well — they count a call from any thread. And a *static*
import-closure proof is not available, because `mi_agent/__init__.py` imports
`llm_query_parser` and `mi_agent_workflow` eagerly, so importing anything under
`mi_agent` puts them in `sys.modules` whether they are used or not. That is a
pre-existing property of the package, not something slice 1 introduced, and the
runtime proof is the operative one.

## Phase 6 — shadow isolation, with live interpretation enabled

```
SHADOW:
  user_visible_response_diffs   0
  exception_escapes             0
```

For all 20 cases the control envelope was serialised before the shadow ran and
compared after: **byte-identical in every one**, and nothing was raised. The paths
exercised with live plans were a live CLARIFY with no plan (I10 — the shadow
returned nothing at all), eight ineligible live plans, eight eligible live
executions, two fixture-gap plans, and one **forced** executor failure, where an
eligible live plan was handed a frame the executor cannot use: captured as
`SHADOW_EXECUTION_ERROR` in the ledger, control untouched, nothing propagated.

The phase 6 control is a deliberate **stub** whose `value` no question would
produce; its only job is to come back unchanged. The ledger classification it
yields is flagged in the record as not a parity result — parity belongs to phase 4
and the independent control.

The live ledger holds 20 rows, carries **none** of the book's 400 loan
identifiers, and carries no question text.

## Adjudication by category

**A. Interpretation deviation — 3, all on ineligible controls, none reaching the
adapter.** I05 representational only; I08 a more literal population base; I04 an
output-shape movement. Zero movements on the ten eligible positives, which all
reproduced the signed-off plan byte for byte. Nothing was changed in response to
any of them.

**B. Eligibility / integration defect — 0.** 20/20 eligibility verdicts matched
the pre-registration, including all five reachable reasons.

**C. Deterministic execution defect — 0.** 8/8 independently correct, 0 numerical
errors, 0 grouped-cell errors, 0 facet drops between plan and spec.

**D. Fixture limitation — 2,** pre-registered as such, fixture untouched.

**E. Harness recording limitation — 1, and it is mine.** The live harness did not
persist the model's raw payloads or the live plan bodies, only a projection of
each plan. Two consequences:

* I04 cannot be fully settled from the recorded evidence. The projection reads
  only `outputs[0]`, so what it recorded — `output_count 1 → 2` and the first
  output's axes going from two to one — is consistent with the second axis moving
  into the second output, and also consistent with its loss. I cannot tell which
  from what was kept, and re-running it would be an unauthorised second call on
  the same question, so it stays open. It does not move any pass criterion: the
  case is an ineligible control, the gate refused it exactly as pre-registered,
  and no figure depended on it. `SILENT_EXPLICIT_SEMANTIC_DROPS = 0` is reported
  on the measured projection, and for the eligible path — where drops would
  matter — the evidence is complete: 0 at the plan level and 0 at the spec level.
* More generally, this run cannot be re-adjudicated offline the way run 8 can.
  Run 8 persisted every raw payload, which is what made the Gate 5 replay of 945
  interpretations possible at zero cost. **Any future live phase should persist
  raw payloads and plan bodies before the calls are made.** Not fixed here: it
  would require a rerun, and reruns cost authorised calls.

### Harness defects found and handled

* **Found in the dry run, before any live call:** the semantic comparison tested
  tuples against the lists a JSON manifest reads back as, and so called nine
  identical replays `INTERPRETATION_DEVIATION` on nothing but the container type.
  Corrected before the live run; no live result was affected, because none
  existed.
* **The recording limitation above**, found after the run, reported and not
  worked around.

## Pass criteria

| criterion | required | measured |
|---|---|---|
| `PROVENANCE` | PASS | PASS |
| `LIVE_BANK_PRE_REGISTERED` | YES | YES, committed in `6d35b849` |
| `BANK_CHANGED_AFTER_MODEL_CALLS` | NO | NO, sha256 re-verified in the run |
| `MODEL` | claude-opus-5 | claude-opus-5 × 20 |
| `MODEL_SUBSTITUTIONS` | 0 | 0 |
| `UNAUTHORISED_MODEL_CALLS` | 0 | 0 |
| `SILENT_EXPLICIT_SEMANTIC_DROPS` | 0 | 0 measured (see limitation E for I04) |
| `SLICE1_ELIGIBILITY_MISMATCHES` | 0 | 0 |
| `NUMERICAL_ERRORS` | 0 | 0 |
| `FILTER_DROPS` | 0 | 0 |
| `DIMENSION_DROPS` | 0 | 0 |
| `MEASURE_DROPS` | 0 | 0 |
| `GROUPED_CELL_ERRORS` | 0 | 0 |
| `LEGACY_SEMANTIC_REREADS_NEW_PATH` | 0 | 0 (profiler and 19 shims) |
| `USER_VISIBLE_RESPONSE_DIFFS` | 0 | 0 |
| `SHADOW_EXCEPTION_ESCAPES` | 0 | 0 |
| `INTERPRETATION_V2_CHANGED` | NO | NO |
| `PRODUCT_CODE_CHANGED` | NO | NO |

`LIVE_SLICE1_INTEGRATION = PASS`

`RECOMMENDED_NEXT_STEP` — deploy Slice 1 with the new path still **shadow-only and
default-off**, verify the exact deployed SHA, then perform a small authenticated
`/mi/query` shadow acceptance using `MI_BEARER` before serving anything. Before
that acceptance run, give its harness what this one lacked: persist every raw
payload and plan body, so the run can be re-adjudicated without buying the calls
again.
