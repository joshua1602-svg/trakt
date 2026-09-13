# Change-intelligence serving connectivity, and the canary it pre-registers

```
PRODUCT_BASELINE  f22d0e467bec66daa2cd3177d7e5c260d847c4e7
PRODUCT_HEAD      88cf5fde759c29f1cff29b9adbf003a2d6ebd530
SPRINT            CHANGE_INTELLIGENCE_SERVING_CONNECTIVITY

LIVE_MODEL_CALLS  0    LIVE_MI_QUERY_CALLS 0    DEPLOYMENTS 0
CONFIG_CHANGES    0    MI_AGENT_PLAN_SERVE untouched (off by default)
```

No semantic contract was reopened. `change_form`, the temporal presence and
default rules, `interpretation_v2`, the Insight Engine and the materiality policy
are exactly as v4 left them.

## Phase 1 — the read-only trace, and the one arrow that was missing twice

```
CURRENT_PLAN_DISPATCH_OWNER
    mi_agent/plan_serving_canary.py::_attempt
    reached from mi_agent_api/mi_service.py::_run_analysis
                 -> _governed_serving_attempt -> plan_serving_canary.serve

MATERIAL_SUMMARY_PLAN_ADAPTER        mi_agent/plan_material_summary.py
                                     EXISTED, WIRED TO NOTHING
MATERIAL_SUMMARY_CALCULATION_OWNER   period_change.workflow.run_period_change_analysis
                                     (mode portfolio_overview, via
                                      period_change_route.analyse_period_change)
MATERIAL_SUMMARY_COMPOSITION_OWNER   mi_agent_api.insight_funded.compose

ATTRIBUTION_PLAN_ADAPTER             DID NOT EXIST
ATTRIBUTION_CALCULATION_OWNER        period_change.bridge.balance_bridge
                                     (via the same workflow, include_bridge=True)

METRIC_DELTA_PLAN_ADAPTER            NONE — the legacy period_change route serves it
LEVEL_COMPARISON_PLAN_ADAPTER        mi_agent/plan_temporal_runtime.py (pre-existing)

RAW_QUESTION_REQUIRED_BY_ANY_CALCULATION_OWNER = NO
```

**The question is not an input to any calculation on this path.** Measured, not
asserted: `period_change/workflow.py` reads `request.question` exactly once, at
line 110, to echo it into `request_interpretation`; `analyse_period_change`
defaults it to `""`; `insight_funded.compose` has no question parameter at all;
and `period_change_route._render` passes it to the envelope builder, where every
channel displays it back to the reader and nothing branches on it.

**Both missing arrows were the same refusal.** A `material_summary` or
`attribution` plan stating a relative pair IS claimed by
`plan_temporal_runtime.claims`, which reads the period form and nothing else. It
then reached `check_temporal_eligibility`, which requires
`capability == generic_analysis`, and was refused `CAPABILITY_NOT_GENERIC` — by a
runtime that speaks for single-measure generic evaluation across snapshots and
never for these two owners. With an absent or `current` period it fell through to
slice 1 and was refused for the same reason. Traced live on compiled plans:

```
material_summary + relative_pair    temporal.claims=True   -> CAPABILITY_NOT_GENERIC
material_summary + ABSENT           temporal.claims=False  -> CAPABILITY_NOT_GENERIC
attribution      + relative_pair    temporal.claims=True   -> CAPABILITY_NOT_GENERIC
attribution      + ABSENT           temporal.claims=False  -> CAPABILITY_NOT_GENERIC
metric_delta     + relative_pair    temporal.claims=True   -> CAPABILITY_NOT_GENERIC
level_comparison + relative_pair    temporal.claims=True   -> ELIGIBLE
```

`level_comparison` was the only one of the four already served by a plan.

## Why `period_change`'s bridge and not `evolution.funded_bridge`

The estate holds two bridges, and picking the wrong one would have made the
temporal contract unenforceable.

| | `evolution.funded_bridge` | `period_change.bridge.balance_bridge` |
|---|---|---|
| decomposes by | a governed DIMENSION | LOAN IDENTITY (new / exited / continuing) |
| resolves its window from | `start_period` / `window_periods`, data-aware | `periods.resolve_periods` |
| honours `current_vs_previous` | no such concept | it is the resolver's own method |

Only the second resolves through the governed period resolver, and the resolver
is what makes this form's authorised default mean **the two adjacent governed
snapshots** rather than a calendar step. A plan whose window the reader left to
the owner can only be honoured there. The sprint's own temporal-presence record
already said so — *"its calculation owner shares `resolve_periods` with every
other period-change figure"* — and this is that statement held to.

The consequence is a real perimeter, not a preference: an attribution plan that
names a dimension is **refused** with the dimension named, because the owner that
could group has a window this form cannot reach.

## What was built

```
PRODUCT_FILES_CHANGED   5   (4 modified, 1 added)
PRODUCT_LOC_ADDED       401 executable   (+607/-20 raw in the four modified
                                          files, plus 266 raw in the new one)
PRODUCT_LOC_REMOVED     19  executable

NEW_ANALYTICAL_CALCULATION_OWNERS = 0
RAW_QUESTION_READS_ADDED          = 0
NEW_SERVING_FRAMEWORKS            = 0
```

**The change budget was exceeded and that is stated, not buried.** The guideline
was ~250–300 lines and the executable count is 401 — about a third over. The
breakdown, so the overage can be judged rather than taken on trust:

| | executable lines | why it exists |
|---|---|---|
| `plan_attribution.py` (new) | 112 | the adapter that did not exist: perimeter, receipt, envelope, execute |
| the shared receipt core (Phase 5) | ~40 | every field Phase 5 names, in one place so two forms cannot receipt differently |
| the shared slot perimeter | ~30 | one refusal per slot the owner cannot honour, each naming its slot |
| `contract_scope.lens_from_plan` | 28 | Direct/Acquired and a named book, mapped through `portfolio_lens` |
| the canary dispatch + attempt helper | 95 | the `_attempt_*` stage contract the other two runtimes already follow |
| envelopes and plan provenance | ~39 | the existing renderer, called, plus the governed receipt stamped into it |
| `analyse`, wrappers, constants, reasons | ~57 | one call to the owner; the parameterised wrappers the extraction needed |
| `mi_service` | 2 | `output_root` and `tenant_id`, from the resolvers it already holds |

Nothing here is a framework: no registry, no new abstraction, no second renderer,
no second calculation. What the lines mostly are is **fail-closed refusals that
name their slot** and **receipt fields Phase 5 requires**. A shorter version
exists, and it is shorter by dropping refusals — which is the trade the estate
has twice paid for.

**One deliberate sharing decision.** The period perimeter, the relative-grain
translation, the slot perimeter and the receipt core live in
`plan_material_summary` and are called by `plan_attribution`, parameterised by
owner name. The two forms are exactly the two that
`vocabulary.CHANGE_FORM_ABSENT_PERIOD_DEFAULT` authorises a `current_vs_previous`
default for, and they execute through one entry point, so they have one temporal
contract. A copy in the new module would be a second place the two could come to
disagree about what the compiler wrote. The precedent is
`plan_temporal_runtime` calling `plan_runtime_adapter.check_capability` and
`check_structure` — slice 2 reusing slice 1's module as the owner of a shared
test.

## Fail closed on every slot the owner cannot honour

`analyse_period_change` takes a client, an output root, a mode, a period request
and a LENS. It passes no `population=` to `build_snapshots`, so a governed row
predicate would reach **neither** snapshot. Left unchecked, a plan saying "what
changed in the back book" would have been computed over the whole book and
published under the reader's words — the defect class the population ledger was
built to close, arriving through a different door.

| plan slot | outcome | reason code |
|---|---|---|
| plan or output `filters` | REFUSE | `FILTER_NOT_SUPPORTED` |
| `geography` | REFUSE | `GEOGRAPHY_NOT_SUPPORTED` |
| `target` | REFUSE | `TARGET_NOT_SUPPORTED` |
| `comparison_kind` | REFUSE | `COMPARISON_NOT_SUPPORTED` |
| `population.seasoning` | REFUSE | `SCOPE_NOT_EXPRESSIBLE` |
| a dimension (material_summary) | REFUSE | `DIMENSION_NARROWS_THE_SUMMARY` |
| a dimension (attribution) | REFUSE | `DIMENSION_NOT_BRIDGEABLE` |
| `population.lens` direct / acquired | **HONOURED**, as the owner's lens | |
| `population.source_reference` | **HONOURED**, as a selection lens | |

Scope is resolved by `mi_agent_api.contract_scope.lens_from_plan`, the same
module that already maps the legacy contract's scope claim, which hands the
answer to `portfolio_lens`' own constructors. Nothing here decides what
"acquired" means. A role the registry cannot resolve to portfolio ids makes
`_apply_lens_filter` **raise**, and the legacy envelope serves — the measured
historical failure was five snapshots in at full size and out at full size, a
whole-book £22.6m published for a book that moved £12.4m, with a receipt
declaring the Direct scope it had not applied. That path has its own control.

## Phase 5 — the execution receipt

Every served answer carries, under `metadata.governedPlan.executed`:

```
plan_id  change_form  capability  operation  mode
population_base  direct_acquired_scope  source_scope  source_portfolio_id
executed_scope {tenant_id, context_id, label, portfolio_ids, asset_classes}
interpreted_time_present  interpreted_period_form  period_completed_by_form
temporal_default_applied  temporal_default_method  temporal_default_owner
period_from  period_to  snapshot_references  period_resolution
calculation_owner  composition_owner  workflow_owner
        material_summary adds  materiality_config  finding_count  omission_count
        attribution adds       bridge_status  bridge_reconciles  bridge_residual
                               bridge_balance_field  bridge_identifier_fields
```

The requested scope and the **executed** scope are separate fields, because an
audit holding only one of them cannot answer whether the book the reader named is
the book the numbers came from.

## Phase 6 — 41 offline controls, and the owner-parity gate

`tests/interpretation_v2/test_change_intelligence_serving.py`. Every control is
built from structured intent; no question is parsed and no model is called.

```
OFFLINE_TARGET_CONTROLS = 41 / 41
```

| | control | result |
|---|---|---|
| A1 | material_summary + explicit relative pair → the runtime | PASS |
| A2 | material_summary + explicit `current` anchor → `current_vs_previous` | PASS |
| A3 | material_summary + absent time → the authorised default, recorded as absent | PASS |
| A4 | material_summary + Direct scope → narrowed to the Direct portfolio ids | PASS |
| A4b | a scope the registry cannot resolve → REFUSE, never widen | PASS |
| A5 | material_summary + unsupported filter → REFUSE (2 cases) | PASS |
| A5b | a named dimension / a governed geography → REFUSE by slot (3 cases) | PASS |
| B6 | attribution + explicit relative pair → the bridge owner | PASS |
| B7 | attribution + explicit `current` → `current_vs_previous` | PASS |
| B8 | attribution + absent time → the authorised default | PASS |
| B9 | attribution + Direct and Acquired scope survives (2 cases) | PASS |
| B9b | attribution across a dimension → REFUSE | PASS |
| B9c | a measure the bridge does not publish → REFUSE | PASS |
| C10 | metric_delta reaches the path it already reached | PASS |
| C11 | level_comparison stays on its temporal path | PASS |
| C12 | a missing `change_form` does not serve | PASS |
| C13 | an incompatible form/operation is refused (3 cases) | PASS |
| C13b | a non-canonical operation reaching the adapter fails closed (3 cases) | PASS |
| C14 | one snapshot → governed refusal, and no date appears anywhere (2 cases) | PASS |
| C15 | a pipeline population never reaches a funded comparison | PASS |
| — | no adapter reads a question (2 modules, structural) | PASS |
| — | the dispatch reads the compiler's form, not the model's claim | PASS |
| — | the served envelope publishes the governed plan receipt | PASS |

```
OWNER_PARITY_GATE = 8 / 8 cases
    NUMERICAL_DIFFERENCES = 0
    SEMANTIC_DIFFERENCES  = 0
    SCOPE_DIFFERENCES     = 0
```

**How parity is proved rather than asserted.** For each positive case the same
structured request is executed twice: once through the serving path, and once by
calling `analyse_period_change` directly with a `PeriodRequest` the test file
states itself — so the comparison is between two independent expressions of the
question, not between the adapter and its own translation. Then:

- the served tables are compared against rows produced by the **owner's own**
  builders (`_metric_rows`, `_distribution_rows`, `_bridge_rows`) from the
  independently-run result, so a rounded, rescaled or re-ordered published value
  would be caught;
- the resolved pair, the resolution method and both snapshot references must
  match;
- `executed_scope` must equal the independent run's `portfolio_scope`;
- for attribution, the bridge status, reconciliation flag, residual and balance
  field must match;
- for material_summary, the brief is recomposed from the independent result and
  every `insight_id` and every finding's `metrics` must match, along with the
  config source and the count.

## Regression

| gate | baseline | after |
|---|---|---|
| B10 `test_insight_funded.py` | 51 / 51 | **51 / 51** |
| B11 `test_change_intelligence_target_controls.py` | 20 / 20 | **20 / 20** |
| Weekly Portfolio Brief (3 files) | 90 / 90 | **90 / 90** |
| `period_change` (13 files) | 335 + 5 subtests | **335 + 5 subtests** |
| completeness + current-anchor + presence | 99 / 99 | **99 / 99** |
| serving canary `test_plan_serving_canary.py` | 60 / 60 | **60 / 60** |
| `tests/interpretation_v2/` + `mi_agent_api/tests/` | 12 FAILED ids | **identical 12**, +41 passes |
| `mi_agent/tests/` | 16 FAILED ids | **identical 16**, unchanged |

```
NEW_UNEXPLAINED_FAILURES = 0
```

The failing-id sets were diffed, not eyeballed. **Two already-red boundary guards
now list two more entries**, and that is stated because it is a change even
though it moves no gate: `test_bank_and_shadow_boundary.py::test_nothing_outside_
the_new_boundary_imports_it` adds `mi_agent/plan_attribution.py` to an
offender list that already holds 79 entries including `plan_serving_canary.py`
and `plan_shadow_wiring.py`, and `test_interpretation_policy.py::test_production_
surfaces_are_untouched` adds `mi_agent_api/contract_scope.py` to a list that
already holds `mi_service.py`. Both guards are shadow-era assertions that
production must not import `interpretation_v2` at all, superseded by three
sprints of deliberate production wiring. They are listed for a decision, not
quietly made green.

## Phase 8 — deployment readiness

```
DEPLOYMENT_SHA                   88cf5fde759c29f1cff29b9adbf003a2d6ebd530
CURRENT_DEPLOYED_SHA             NOT READ — this session cannot reach the host
MI_AGENT_PLAN_SERVE_CURRENT_STATE NOT READABLE by this repository's automation
AUTH_PATH_READY                  YES — the path exists and is exercised
MI_BEARER_AVAILABLE              REFERENCED as a repository secret; not read here
```

**`CURRENT_DEPLOYED_SHA` was not guessed.** The authoritative source is
`GET https://app.traktinfra.io/api/health`, whose `build` field carries the
deployed commit — *"WHICH COMMIT, not which version string"*, added precisely
because the hand-written version was identical across every deploy. This
session's network policy denies `app.traktinfra.io` (the proxy answered 403 to
CONNECT), so it is unread here and readable by a CI step, which is how every
existing acceptance workflow confirms it before asking anything.

**`MI_AGENT_PLAN_SERVE` cannot be read by this repository at all**, and that is
the repository's own finding rather than a limitation of this session:
`MI_AGENT_PLAN_SERVE` and `MI_AGENT_PLAN_SERVE_PRINCIPALS` are Azure App Service
application settings requiring ARM access, and this repo's mi-api automation
holds a publish profile, which grants Kudu and nothing else. Four acceptance
workflows state it. **Per the stop condition, serving is not enabled and nothing
was changed.** Reading or setting it is an operator action by someone with ARM
access to `trakt-mi-api`.

## Phase 7 — the pre-registered canary, NOT RUN

```
CANARY_PREPARED   YES
CANARY_BANK_ID    change_intelligence_serving_canary
CANARY_SHA256     27b7d825a1b3f88e631c5931826fa0b0fdab331205ca7b4ea7d318c43aa627b1
CANARY_QUESTIONS  10
RUN               NOT RUN — awaiting approval. No runner and no workflow exist
                  yet, so nothing can spend by accident.
```

```
S01  material_summary   EXPLICIT_PAIR    ANSWER   governed plan
S02  material_summary   OWNER_DEFAULT    ANSWER   governed plan
S03  metric_delta       EXPLICIT_PAIR    ANSWER   provenance NOT pinned
S04  metric_delta       EXPLICIT_PAIR    ANSWER   provenance NOT pinned
S05  attribution        EXPLICIT_PAIR    ANSWER   governed plan
S06  attribution        OWNER_DEFAULT    ANSWER   governed plan
S07  level_comparison   EXPLICIT_PAIR    ANSWER   provenance NOT pinned
S08  level_comparison   EXPLICIT_PAIR    ANSWER   provenance NOT pinned
S09  material_summary   EXPLICIT_PAIR    ANSWER   governed plan, ACQUIRED scope
                                                  must NARROW, not only label
S10  (no form)          EXPLICIT_PAIR    REFUSE   a portfolio the registry
                                                  does not govern
```

**No number is pinned.** Numeric truth is reconciled afterwards by running the
named deterministic owner independently over the same governed snapshots with the
period request the served receipt records — the offline parity gate, moved to
live inputs. A bank that pinned a figure taken from a model's output and then
scored the model against it would certify nothing.

**None of v1's, v3's or v4's wording is reused.** Verified programmatically
against all 71 historical questions before hashing: zero exact matches, and the
closest token overlap is 0.50 (S06 *"Explain what sits behind the change in
funded balance"* against v1/CF11 *"What drove the change in funded balance this
month?"*). Two first drafts were rewritten for being too close to v3/V01 and
v3/V15.

**The serving provenance of `metric_delta` and `level_comparison` is deliberately
not pinned.** This sprint established nothing about it, so pinning it would pin
an unmeasured fact. What is pinned is the negative: neither may arrive through a
change-form route that does not claim it.

### My fourth chance to repeat the same authoring mistake, declined

Three banks in a row carried an authoring defect of mine, each surfacing while I
corrected the one before: v3/V06 (a restricting adjective is not an invented
filter), v3/V13 (`bridge_component` is a legitimate attribution target), v4/W06
(a concept can be permitted as a filter target and still publish no value
vocabulary, so the restriction is not expressible and clarifying is correct). All
three corrections are carried into this bank **and the third is honoured by
construction**: no question restricts on arrears or on any other concept with an
empty governed value list, and the one question that cannot be satisfied is
pinned as a REFUSAL rather than as an answer with zero clarifications.

## Integrity

```
v1 manifest hashes to its pin                      OK  (immutable, 17/18)
v2 manifest hashes to its pin                      OK  (immutable, retired UNRUN)
v3 manifest hashes to its pin                      OK  (immutable, change_form 18/18)
v4 manifest hashes to its pin                      OK  (immutable, 17/17 + 17/17)
change_intelligence_serving_canary hashes to its pin OK (this pre-registration)
```

No historical evidence was edited. The 135 bank was not run, not rescored and not
touched. No expectation in this bank was written after seeing a result, because
no result exists.
