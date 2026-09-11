# Slice 2 — governed funded-book temporal analysis. Offline result.

```
START_SHA = 53a7b8f6f8294b2c6abb585ec814322f944ce0cb
FINAL_SHA = (see the commit that carries this file)
```

Slice 1 was not reopened. Its perimeter decides exactly what it decided before,
and that is measured rather than asserted — see **Slice 1 non-regression**.

---

## ARCHITECTURE_TRACE

Written before any code changed: `ARCHITECTURE_TRACE.md` in this directory.
Ten numbered findings with file and function references, and the six stop
conditions evaluated one by one. None triggered.

The single most consequential finding: **the compiler already emits governed
temporal intent and needed no change at all.** `vocabulary.TIME_FORMS` already
carries `series`, `range`, `explicit_period`, `previous_reporting_period` and
`relative_pair`; `CAPABILITY_OPERATIONS["generic_analysis"]` already carries
`series` and `compare`; `compiler._PERIOD_CONTRACT` already names the governed
contract per form; `PeriodBinding` already carries `form`, `labels`, `grain`,
`periods_back` and `contract`. Every Slice 2 example shape compiled to a
`GovernedQueryPlan` at `53a7b8f6`, with the temporal intent intact. What was
missing was everything downstream of the plan: nothing resolved a governed
period against a catalogue, and the eligibility gate refused every plan that
named one.

```
TEMPORAL_EXECUTION_OWNER  = mi_agent.mi_query_executor.execute_mi_query
                            (the accepted Slice 1 owner, run once per snapshot
                             from one MIQuerySpec bound once by
                             plan_runtime_adapter.spec_for_plan)
SNAPSHOT_RESOLUTION_OWNER = snapshot.store.SnapshotStore resolvers, reached
                            only through mi_agent.states.selectors.SnapshotSelector
```

No new analytics engine, no second snapshot selector, no second parser cascade.
`mi_agent/states/temporal.py`, `mi_agent/period_change/`, `mi_agent_api/evolution.py`
and `mi_agent/period_request.py` are all untouched; the trace records why none of
them could serve as the Slice 2 measure owner (each carries a fixed
`loan_count`/`balance_sum` pair and cannot express an average or a weighted
average LTV) while `states/temporal.trend` nevertheless supplied the proven
loop *shape* that this slice follows.

---

## FILES_CHANGED

**Product — five files**, one of them new:

| file | net executable lines | what changed |
|---|---:|---|
| `snapshot/store.py` | +12 | `resolve_last_n` — the one selection the four existing resolvers cannot express |
| `mi_agent/states/selectors.py` | +10 | the matching `last_n` mode and factory |
| `mi_agent/plan_runtime_adapter.py` | +28 | `check_eligibility` split into `check_capability` + Slice-1 perimeter + `check_structure`; `reconcile_receipt` and `value_column` extracted so there stays one owner of each |
| `mi_agent/plan_serving_canary.py` | −7 | `reconcile` now delegates its structural half to `adapter.reconcile_receipt` |
| `mi_agent/plan_temporal_runtime.py` | **+525** | NEW — the governed temporal binder and the per-snapshot orchestrator |
| | **+568** | **PRODUCT_NET_EXECUTABLE_LOC** |

**Tests and evidence — five files**, three of them new: `+879` executable lines
(`portfolio_truth_oracle.canonical_history`, `temporal_snapshot_fixture`,
`test_plan_temporal_runtime.py`, `slice2_acceptance.py`, and the rewritten
discipline test in `test_plan_shadow_wiring.py`).

### The budget, declared before editing, and the overrun

```
EXPECTED_PRODUCT_FILES_CHANGED = 4        ACTUAL = 5
EXPECTED_NET_EXECUTABLE_LOC    = ~350     ACTUAL = 568   (+62%)
```

Both are overruns and neither is dressed up. The fifth file is
`plan_serving_canary.py`, changed only so that the structural reconciliation
check has ONE owner rather than two — the alternative was to copy those eight
lines into the temporal runtime, which is the duplication the discipline rule is
protecting against. The line overrun is entirely inside
`plan_temporal_runtime.py` (525 of the 568): the module is 825 lines of which
124 are docstring and 60 are comment, and its largest function is the
122-line orchestration loop.

The rule the discipline section actually states is *stop if the implementation
begins creating duplicate temporal routing, duplicate snapshot selection or
another parser cascade*. Measured against that:

* **temporal routing** — one function decides shape (`resolve_temporal`), and it
  is the only one;
* **snapshot selection** — every selection goes through `SnapshotSelector` into
  a `SnapshotStore` resolver; the new module resolves nothing itself, and a test
  asserts every date it puts on a selector came out of the catalogue;
* **parser cascade** — the module imports no `re`, and a test asserts it off the
  AST. The one string it reads is `period.labels`, which the compiler authored.

---

## SLICE 2 BANK RESULT

`slice2_bank.yaml` → `slice2_acceptance.py` → `slice2_acceptance.json`.

```
SLICE_2_BANK_SIZE = 31        (18 executable questions + 13 negative/safety controls)
ELIGIBLE          = 18
CORRECT           = 31
PARTIAL           = 0
INCORRECT         = 0
CLARIFY           = 6         (4 at the temporal resolver, 2 at the compiler)
REFUSE            = 1         (at the compiler: a forward-looking generic series)
INELIGIBLE        = 6         (at the slice 2 perimeter)

EXECUTED_CASES              = 18
NUMERICALLY_RECONCILED      = 18
GROUPED_CELL_RECONCILIATION = 68 cells across 4 grouped cases, plus 71 scalars
```

Every one of the ten required categories executes and reconciles:

| category | cases | snapshots selected |
|---|---|---|
| funded balance over last N months | S2-01, S2-02 | 6, 3 |
| funded balance each month | S2-03, S2-04 | 8, 8 |
| loan count each month | S2-05 | 8 |
| average LTV over time (simple and weighted) | S2-06, S2-07 | 4, 8 |
| filtered evolution | S2-08, S2-09 | 8, 3 |
| time × one dimension | S2-10, S2-11 | 8, 4 |
| time × two dimensions | S2-12 | 2 |
| explicit period | S2-13, S2-14 | 1, 1 |
| since \<period\> | S2-15, S2-16 | 4, 5 |
| current versus prior period | S2-17, S2-18 | 2, 2 |

Five carry a semantic paraphrase of another case. Paraphrase invariance was not
made the KPI, as instructed.

### The negative controls, and what each does

| control | case | outcome |
|---|---|---|
| unavailable historical period | S2-N1 | CLARIFY `PERIOD_NOT_AVAILABLE` — twelve months asked of an eight-month book; not shortened |
| ambiguous / vague period | S2-N2 | CLARIFY `PERIOD_LABEL_UNRESOLVED` — "last few months" is not resolved by convention |
| future period | S2-N4 | REFUSE `UNSUPPORTED_COMPOSITION` at the compiler |
| unsupported cadence | S2-N5 | CLARIFY `UNSUPPORTED_CADENCE` — weekly against a catalogue that declares monthly |
| pipeline temporal question | S2-N6 | INELIGIBLE `POPULATION_NOT_FUNDED` |
| acquisition attribution | S2-N7 | INELIGIBLE `EXPLICIT_LENS` |
| movement-cause question | S2-N8 | INELIGIBLE `OPERATION_NOT_TEMPORAL` |
| multi-output temporal question | S2-N9 | INELIGIBLE `NOT_SINGLE_OUTPUT` |
| unsupported specialist measure | S2-N10 | INELIGIBLE `CAPABILITY_NOT_GENERIC` |
| a required snapshot absent | S2-N11 | CLARIFY `PERIOD_NOT_AVAILABLE` — "August", which the book does not carry |
| time × geography | S2-N12 | INELIGIBLE `GEOGRAPHY_REQUESTED` — see the boundary note below |
| recorded model interpretations | S2-R1, S2-R2 | COMPILE_CLARIFY, verbatim from run 8 |

None of them widens, narrows, drops the time constraint or substitutes a period.

---

## KEY ACCEPTANCE ASSERTIONS

```
TEMPORAL_INTENT_PRESERVED                 = YES   (31/31)
SNAPSHOT_SELECTION_DETERMINISTIC          = YES
RAW_TEXT_TEMPORAL_REREADS_AFTER_PLAN      = 0
ROW_LEVEL_DATE_FILTER_USED_FOR_HISTORY    = NO
MODEL_AUTHORED_PHYSICAL_SNAPSHOT_BINDINGS = 0
SILENT_TEMPORAL_DROPS                     = 0
SILENT_PERIOD_SUBSTITUTIONS               = 0
SNAPSHOT_SELECTION_ERRORS                 = 0
```

How each is established, rather than asserted:

* **snapshot selection** — `expect.snapshots` is written out by hand in the bank
  against the eight-month fixture and compared with what was selected. The
  product never supplies the expectation.
* **raw-text re-reads** — counted off the ASTs of `plan_temporal_runtime` and
  `plan_runtime_adapter`: no `question` / `sentence` / `parsed` name, and no
  `re` import, in either.
* **physical bindings** — `intent._DATE_LIKE` already refuses a date or a
  snapshot id in any binding slot including `time.labels`, and the harness
  re-reads every emitted plan's period and confirms it carries none.
* **row-level date filter** — the module contains no `to_datetime`, no
  `read_csv`, and no date column name; a test asserts it.
* **silent drops** — no case that failed to resolve produced any point, and no
  executed case ran on a snapshot set other than the one the bank names.

---

## SLICE 1 NON-REGRESSION

```
SLICE_1_REGRESSIONS_ATTRIBUTABLE = 0
```

Two independent measurements:

1. **Decision equivalence over the whole recorded corpus.**
   `test_plan_shadow_wiring.TestDiscipline.test_the_accepted_adapter_decides_exactly_what_it_always_did`
   loads the ACCEPTED `plan_runtime_adapter.py` blob from `6f42df67` alongside
   the current one and puts all 135 recorded plans to both. Every verdict —
   eligible or not, and for which reason — is identical.

   This test replaces a hash of the file, which Slice 2 necessarily broke by
   splitting `check_eligibility` to share its structural half. Replacing a hash
   with a differential test is the honest move here and is a stronger claim, but
   it IS a change to a Slice 1 discipline test and is called out rather than
   buried.

2. **The Slice 1 corpus replay, re-run.**
   `due_diligence/evidence/plan_shadow_slice1/corpus_replay.py` produces a
   result file byte-identical to the one committed at `53a7b8f6`: 945 recorded
   plans, 115 Slice 1 eligible, the same ineligibility histogram, the same 73
   ledger rows, the same 15 grouped-cell parities, 0 live Opus calls.

`mi_agent/tests/test_plan_runtime_adapter.py` (55 tests) and the Slice 1A/1B
suites pass unchanged.

3. **The whole `mi_agent` suite, run twice: once at `53a7b8f6` with the working
   tree stashed, and once with Slice 2 in place.**

   | | base `53a7b8f6` | with Slice 2 |
   |---|---|---|
   | passed | 1623 | **1661** (+38, the new Slice 2 tests) |
   | failed | 24 | 24 |
   | errors | 21 | 21 |
   | skipped / xfailed | 264 / 7 | 264 / 7 |

   The failing NODE IDS are identical between the two runs — `diff` over the
   sorted `FAILED`/`ERROR` lines is empty. All 45 pre-exist this work and none
   touch a module it changed: `test_an_as_at_date_is_accounted_for` (21 errors),
   `test_p1c_ranked_movement` (5), `test_p0_execution_receipt` (3),
   `test_proposal_prompt_caching` (3), `test_a_threshold_that_keeps_every_row`
   (2), and one each in `test_a_time_grain_is_material`,
   `test_case_age_is_not_borrower_age`, `test_mi_predicate_extraction`,
   `test_mi_trust_hardening` and `test_parser_cost_hardening`.

   Three tests under `tests/interpretation_v2/` also fail at base and with Slice
   2 alike — `test_nothing_outside_the_new_boundary_imports_it`,
   `test_only_contract_modules_moved`, `test_production_surfaces_are_untouched`.
   They compare the tree against a historical baseline commit and were already
   red at `53a7b8f6` (their offender lists already name `plan_serving_canary.py`,
   `plan_shadow_wiring.py` and `mi_service.py`). Slice 2 adds four more names to
   those same lists — `plan_temporal_runtime`'s harness and test,
   `states/selectors.py` and `portfolio_truth_oracle.py`. The tests do not go
   from green to red, but this work does widen a list they were already
   complaining about, and that is recorded here rather than left to be found.

---

## MODEL / INTERPRETATION WORK

```
LIVE_OPUS_CALLS = 0
```

`interpretation_v2` was not reopened. The interpreter, the vocabulary, the
intent schema and the compiler are byte-unchanged.

The captured interpretations were replayed first, as instructed, and the finding
is worth stating plainly because it shapes what this bank can and cannot claim.
Of the 135 signed-off interpretations in `run8_135_signoff_2b00172.json`:

* **39** state a period other than `current`;
* **36** of those are specialist capabilities — `period_movement`,
  `funded_bridge`, `forecast`, `limit_assessment`, `pipeline_stage_movement` —
  and every one is correctly `CAPABILITY_NOT_GENERIC` at the Slice 2 perimeter;
* **3** are `generic_analysis`, and all three CLARIFY at the compiler before the
  perimeter is reached (NL1A on a blocking model ambiguity and two missing
  slots; NL1C on a blocking model ambiguity; NL8C on `AMBIGUOUS_PERIOD`);
* **0** are Slice 2 eligible.

So the recorded corpus contains **no funded-book temporal evaluation question at
all** — it was built to measure movement, bridge and forecast questions. Captured
interpretations were therefore not sufficient to populate a Slice 2 bank, and a
fresh interpretation proof (Phase 5) would have been the next step.

**Phase 5 was not run, because this environment holds no `ANTHROPIC_API_KEY`.**
Twenty-nine of the thirty-one bank cases therefore carry an AUTHORED
`CandidateIntent` — a payload written into the bank, parsed by the production
`parse_candidate_intent` and compiled by the production `DeterministicCompiler`.
The remaining two (S2-R1, S2-R2) take the model's own recorded payload whole,
read out of the run-8 evidence file by the harness rather than transcribed.

**What that means for the claim.** This bank measures the compiler, the
perimeter, the temporal resolver, the snapshot selector and the executor. It
does **not** measure whether Opus states Slice 2 temporal intent in the shape
the resolver expects — in particular whether it supplies `periods_back` for "the
last six months" rather than only a label, and whether it labels "since March"
in a form `parse_anchor` reads. That is the open question this slice ends on,
and it is the first thing a Phase 5 run should answer.

---

## PHASE 5 — THE OPUS → TEMPORAL BOUNDARY, MEASURED LIVE

```
LIVE_BANK_SIZE        = 15   (11 positives + 4 controls)
AUTHORISED_LIVE_CALLS = 15
LIVE_CALLS_MADE       = 15
MODEL                 = claude-opus-5
BOUNDARY_HELD         = 7 / 15
```

Tokens: 79,681 input · 16,563 output · 422,436 cache read · 8,988 cache write.
One call per case, no retry, no reworded second try. The manifest sha256 was
verified against the file committed before any call was made.

`7/15` is the honest headline and it is also two different stories added
together, which is not actionable. `temporal_live_findings.py` separates them
from the recorded payloads, with no further calls.

### What held

```
TEMPORAL_INTENT_ABSENT                    = 0
WRONG_SNAPSHOTS                           = 0
MODEL_AUTHORED_PHYSICAL_SNAPSHOT_BINDINGS = 0
```

Not one live interpretation dropped the time constraint, selected a wrong
snapshot, or put a date or snapshot id anywhere in a plan. Every failure below is
**fail-closed**: a refusal or a clarification, never a wrong answer.

Three positives held end to end, through capability, compiler, perimeter and
snapshot selection:

| case | question | what Opus emitted | snapshots |
|---|---|---|---|
| T02 | "…balance in each of the last three months?" | `generic_analysis/series`, `periods_back: 3` | 3, correct |
| T08 | "…by LTV bucket and product type for the last two months." | `generic_analysis/series`, `periods_back: 2` | 2, correct |
| T09 | "What was funded balance in April?" | `generic_analysis/point_in_time`, `explicit_period`, `labels: ["April"]` | 1, correct |

All four controls behaved, including one the pre-registration anticipated: T12
("the next three months") was pre-registered expecting a compiler REFUSE, and the
note said that if the model routed it to `forecast` instead, that would be a
justified refusal at the perimeter. It did, and it was. T13 asked twenty-four
months of an eight-month book — Opus supplied `periods_back: 24`, the resolver
clarified `PERIOD_NOT_AVAILABLE`, and **no shorter series was substituted**.

### FINDING 1 — "How has X changed" is read as a MOVEMENT question (4 cases)

T01, T05, T10 and T11 were refused `CAPABILITY_NOT_GENERIC`. Opus routed every
one of them to `period_movement`:

| case | question | capability / operation |
|---|---|---|
| T01 | "How has funded balance **changed** over the last six months?" | `period_movement` / `movement` |
| T05 | "How has average LTV **changed** over the last four months?" | `period_movement` / `series` |
| T10 | "How has funded balance **changed** since March?" | `period_movement` / `movement` |
| T11 | "…this month **versus** last month?" | `period_movement` / `movement` |

This is not obviously a model defect. "How has X changed" *is* more naturally a
movement question, and `period_movement` is a real governed capability that owns
its own arithmetic. The tension is in the brief: the Slice 2 contract lists "How
has funded balance changed over the last 6 months?" as an in-scope example while
also excluding movement attribution — and the vocabulary resolves that ambiguity
the other way.

**The temporal binding in these four was sound.** A counterfactual re-compile of
the recorded payloads with `capability` forced to `generic_analysis` resolves all
four to exactly the pre-registered snapshots — `periods_back: 6`,
`periods_back: 4`, `labels: ["since March", "March"]`, and a `relative_pair`.
So what failed was **which capability owns the question**, not whether Opus can
state a window the resolver can act on.

> That counterfactual is a counterfactual. It re-compiles recorded payloads with
> one field changed, it is not a re-run, and it is evidence about the temporal
> binding only — never about routing. The headline stays 7/15.

The fix is not in the resolver. It is either a question-phrasing matter, or a
vocabulary change distinguishing "what was X in each period" (evaluation) from
"how much did X move and why" (movement) — and that belongs to a slice
authorised to reopen `interpretation_v2`.

### FINDING 2 — the compiler and my resolver disagree about a bare-cadence span (4 cases)

T03, T04, T06 and T07 ("…each month") were refused `PERIOD_LABEL_UNRESOLVED`.
Opus emitted, for all four, exactly:

```
time: {form: "series", labels: [], grain: "monthly", periods_back: null}
```

It stated the **cadence** and left the **span** open — which is a fair reading of
"each month". Then:

* `compiler._bind_period` raises `AMBIGUOUS_PERIOD` only when a span has *none* of
  labels, count or grain. A grain is present, so **the compiler emitted a plan**;
* `plan_temporal_runtime._resolve_span` requires a count, a recognised
  whole-series label, or an anchor. It has none, so **the resolver refused the
  plan the compiler had just authorised**.

**This is a defect Slice 2 introduced**, and the live run is what found it: two
governed layers disagreeing about what constitutes a stated span. My
`WHOLE_SERIES_LABELS` vocabulary was built for labels like "each month" — and it
never fires, because the model puts nothing in `labels` at all.

Measured: the whole-series reading would select all eight periods, matching the
pre-registration in **4/4**.

The prediction this report made before the run was *directionally* right and
*specifically* wrong. It said the risk was a labels-only interpretation with no
count. The reality is neither labels nor count — the window is carried by `grain`
alone.

**The fix is one branch in `_resolve_span`**: a span form carrying a grain that
matches the catalogue's declared cadence, with no count and no label, is the
whole available series — the same reading the compiler already took when it
emitted the plan. That is a runtime change, which this turn was explicitly not
authorised to make, so it is recorded and not done.

### What this does and does not prove

It proves that Opus, unprompted, states temporal intent the deterministic layer
can act on — a count, an anchor month, or a period pair — in **every case where
it routed to `generic_analysis`**, and that the governed layer never fabricated,
substituted or silently shortened a period.

It does not prove the boundary is production-ready. Two-thirds of the natural
phrasings in this bank either land on a capability Slice 2 excludes, or land on a
resolver branch that does not exist yet.

### To reproduce

```bash
python due_diligence/evidence/plan_temporal_slice2/temporal_live_findings.py   # the two findings, no calls
ANTHROPIC_API_KEY=... python due_diligence/evidence/plan_temporal_slice2/temporal_live_run.py
```

---

## SLICE 2 BOUNDARY CORRECTION — the two issues the live run found

Base for this correction: `515d6c48`. Scope: the two isolated issues and nothing
else. Slice 2 was not broadened, nothing was deployed, `mi_service` was not
wired, Slice 3 was not started, and no second fifteen-question bank was bought.

```
FILES_CHANGED               = 2 product, 2 test, 2 evidence (1 new)
PRODUCT_NET_EXECUTABLE_LOC  = +44
    mi_agent/plan_temporal_runtime.py          +13
    mi_agent/interpretation_v2/vocabulary.py   +31
```

### ISSUE 1 — a bare cadence is a stated span (deterministic, no calls)

`_resolve_span` now reads `{form: series, grain: <catalogue's cadence>,
labels: [], periods_back: null}` as the whole available governed series.

It is not a widening. There is no narrower window being passed over: the request
names a rhythm and no bound, and "every governed period at that rhythm" is the
only window that answers it. It also ends a disagreement Slice 2 itself
introduced — `compiler._bind_period` raises `AMBIGUOUS_PERIOD` only when a span
has *none* of labels, count or grain, so the compiler was emitting plans this
resolver then refused. The correction settles it in the compiler's favour.

The guards that matter are untouched, and each has a test:

| still fails closed | test |
|---|---|
| a grain the catalogue does not keep (weekly against a monthly book) | `test_a_bare_cadence_the_catalogue_does_not_keep_still_fails_closed` |
| a catalogue that declares no cadence at all | `test_a_catalogue_declaring_no_cadence_does_not_get_the_bare_span` |
| a bare `range` — a range states bounds, and one with neither is incomplete | `test_a_bare_range_is_still_an_incomplete_request` |
| a label that is present but unreadable ("the last few months") | `test_an_unreadable_label_still_clarifies_even_with_a_grain` |

The last one is the important one: the branch requires **no label**, not merely
no usable one. "The last few months" names a narrower window than the whole
series, and reading the grain instead would be exactly the substitution this
contract exists to stop.

```
GRAIN_ONLY_REPLAY = 4/4
```

T03, T04, T06 and T07 replayed from their **captured Opus payloads** — the same
bytes the model emitted in the fifteen-call run — now resolve on `basis=cadence`
to all eight governed periods, execute through `execute_temporal_plan`, and
reconcile against `portfolio_truth_oracle`.

### ISSUE 2 — the capability boundary is now stated, not inferred

The orientation block gave the model capability NAMES and no descriptions. That
works while the names are self-separating; the live run measured where they are
not. `vocabulary.CAPABILITY_BOUNDARIES` now states what `generic_analysis` and
`period_movement` each own, and `orientation_payload` carries it.

Both capabilities are real and neither is preferred. The entry is a statement of
governed ownership — level, series, current-versus-previous and the change
*between* snapshot results to `generic_analysis`; cause, attribution,
decomposition, inflow/outflow, transition and which populations drove it to
`period_movement` — closing with "the word 'changed' does not decide it on its
own". Scoped to the one pair the live run measured; a pair with no evidence
behind it does not belong there.

`VOCABULARY_VERSION` moves `2.0.0 -> 2.1.0`, because `PlanProvenance` records what
the model was SHOWN and an interpretation made against a different orientation
block is not comparable with one made against this. Every recorded run before
this change was made at 2.0.0 and says so.

**What was deliberately NOT done.** `interpretation_v2` was not broadly retuned:
`opus_interpreter.py` (which owns `SYSTEM_PROMPT`), `metadata.py`, `outcomes.py`
and the frozen `banks/` are byte-unchanged, and
`test_the_interpreter_policy_did_not_move` still pins all four. No downstream
raw-text routing was created. The compiler does **not** silently replace
`period_movement` with `generic_analysis` — a `period_movement` plan is still
refused at the perimeter, exactly as before. The model is expected to emit the
right capability from the clarified vocabulary, which is why this issue can only
be settled by a live re-ask.

That guard was green before this change and my edit turned it red, so it is
rebuilt rather than deleted, and it now asserts more than it did: the four
policy owners stay pinned by name, and the model-facing orientation block is
compared key-by-key against the pre-correction one — it may gain
`capability_boundaries` and nothing else, lose nothing, and reword no existing
entry.

### Regression, on recorded payloads only

```
ORIGINAL_PASS_REPLAY          = 7/7
GRAIN_ONLY_REPLAY             = 4/4
MOVEMENT_ATTRIBUTION_CONTROLS = T14 INELIGIBLE:CAPABILITY_NOT_GENERIC (funded_bridge)
                                T12 INELIGIBLE:CAPABILITY_NOT_GENERIC (forecast)
UNAVAILABLE_PERIOD_CONTROL    = T13 CLARIFY:PERIOD_NOT_AVAILABLE
OVERLONG_PERIOD_CONTROL       = T13 selected NOTHING — no shortened series
MULTI_OUTPUT_CONTROL          = T15 INELIGIBLE:NOT_SINGLE_OUTPUT
SCALARS_RECONCILED            = 28
CELLS_RECONCILED              = 60
PHYSICAL_BINDINGS             = 0
```

Slice 2 offline acceptance: **31/31 CORRECT**, unchanged, including the
"last few months" clarify control and the weekly-cadence control.

```
SLICE_1_REGRESSIONS_ATTRIBUTABLE = 0
```

`corpus_replay.json` reproduces **byte-identical** to the committed file (945
recorded plans, 115 eligible, same histogram, same 73 ledger rows). The Slice 1
and 1B suites pass: 197 tests across the adapter, the serving canary, the shadow
wiring, the evidence recorder and the replay.

### The live retest — four calls, and one case still open

```
LIVE_OPUS_CALLS            = 4   (4 authorised, 4 made, no retries)
GENERIC_ANALYSIS_ROUTING   = 4/4
LIVE_TEMPORAL_PLAN_CORRECT = 3/4
```

Tokens: 16,433 input · 5,252 output · 102,630 cache read. One call per case, no
retry, no paraphrase. The manifest sha256 was verified first; the subset result
is written to its own file and cannot overwrite the fifteen-case run.

**The capability correction worked, completely.** All four moved from
`period_movement` to `generic_analysis` — the vocabulary change did the whole
job it was asked to do:

| case | before | after | snapshots |
|---|---|---|---|
| T01 "…changed over the last six months?" | `period_movement/movement` | `generic_analysis/series`, `periods_back: 6` | 6, correct, executed |
| T05 "…average LTV changed over the last four months?" | `period_movement/series` | `generic_analysis/series`, `periods_back: 4` | 4, correct, executed |
| T11 "…this month versus last month?" | `period_movement/movement` | `generic_analysis/compare`, `relative_pair` | 2, correct, executed |
| T10 "…changed since March?" | `period_movement/movement` | `generic_analysis/**movement**` | refused |

Three executed and reconciled against the independent oracle — 12 scalars, zero
discrepancies. And across all four:

```
TEMPORAL_INTENT_ABSENT                    = 0
WRONG_SNAPSHOTS                           = 0
MODEL_AUTHORED_PHYSICAL_SNAPSHOT_BINDINGS = 0
```

### T10 — the same finding, one level down

T10 now carries the right capability and the wrong OPERATION:
`generic_analysis` + `operation: movement`, which the Slice 2 perimeter refuses
as `OPERATION_NOT_TEMPORAL`. Its temporal binding is fine —
`labels: ["March", "since March"]` resolves to the four periods from March
onwards — and the refusal is fail-closed, not a wrong answer.

`CAPABILITY_BOUNDARIES` separated the two CAPABILITIES and said nothing about
which OPERATION a "how has X changed" question takes. `CAPABILITY_OPERATIONS`
lists `movement` under `generic_analysis`, so "changed" can still land there.
The boundary statement fixed the layer it addressed and the ambiguity moved down
one.

The obvious next step is to extend the same statement to the operation —
`series` for a level across periods, `movement` only where a cause or
decomposition is asked for — and re-ask T10. **That was not done here.** This
turn authorised two named issues, one call per case and no retries, and a third
edit followed by a third re-ask of the same four questions starts to be tuning
against a fifteen-question bank rather than fixing an architecture. It is a
decision to take deliberately, not one to slide into.

### Final accounting

```
7 originally passing        (replayed, 7/7)
4 grain-only corrected      (replayed, 4/4)
3 capability corrected      (live, 3/4)
= 14/15

SLICE_2_TEMPORAL_BOUNDARY = FAIL against the stated 15/15 criterion
                            14/15 settled; T10 open on operation vocabulary
```

Reported as FAIL because the success criterion was 15/15. Every other assertion
held: no temporal intent absent, no wrong snapshots, no silent drops, no period
substitutions, no physical bindings, no raw-text rereads after the plan, and no
Slice 1 regression.

```bash
python .../temporal_boundary_regression.py                                   # 15-case, PASS
python .../temporal_boundary_regression.py temporal_live_retest_result.json  # retest, FAIL on T10
```

---

## SLICE 2 SERVING INTEGRATION — offline

```
INTEGRATION_BASE_SHA = f107022f   (Slice 2 with main merged in)
COMMITS_RECONCILED   = 5 from main, 50 from Slice 2, zero conflicts of substance
```

### The lineage

`main` carried five commits Slice 2 did not: three hardening
`borrowing_base_acceptance.py` and two registering acceptance workflows. **None
touches product code.** One file appears on both sides —
`.github/workflows/deployed-shadow-acceptance.yml` — and it is byte-identical on
each (blob `601ae007`), so the merge resolved without a judgement call. A merge
was the right operation rather than a rebase: the branch carries fifty commits
that are already pushed and reviewed, and rewriting them to absorb five
evidence-only commits would trade real history for a tidier graph. Both lineages
are ancestors of the result, `53a7b8f6` included.

### The seam

```
SERVING_DISPATCH_SEAM = plan_serving_canary._attempt, immediately after
                        `plan = compiled.plan.to_dict()` and before
                        `adapter.check_eligibility(plan)`
```

One structural read — `plan_temporal_runtime.claims(plan)` — decides which
runtime owns the plan, and it is not a judgement. Slice 1's perimeter accepts
`period.form == "current"` and nothing else; `SLICE_2_PERIOD_FORMS` excludes
`current` and nothing else. The two are **disjoint by construction**, so the plan
says which runtime owns it and no new semantic owner appears at the seam.

`claims()` returning True does not mean eligible. A temporal plan outside the
Slice 2 contract is claimed and then refused *by Slice 2*, with a temporal
reason — T10 comes back `OPERATION_NOT_TEMPORAL` rather than falling through to
Slice 1 to be refused for not being current, which would describe the wrong
thing.

```
FILES_CHANGED              = 2 product, 1 test, 2 evidence (1 new)
PRODUCT_NET_EXECUTABLE_LOC = +140   (declared ~120)
    mi_agent/plan_serving_canary.py    +98
    mi_agent/plan_temporal_runtime.py  +42
```

### The catalogue is the caller's to supply — and production supplies none

`serve()` gains `snapshot_store` / `snapshot_client_id` / `snapshot_route`,
exactly as it already takes `frame` and `semantics` from its caller. When none is
supplied the temporal path returns `TEMPORAL_STORE_UNAVAILABLE` and the legacy
envelope serves.

**`mi_service` is byte-unchanged and passes none.** Production has no
`SnapshotStore` wired — none has ever existed in `mi_agent_api` — so no
production request can take the temporal path, and nothing a production caller
can observe has changed. Building a store over the onboarding run catalogue would
be new production configuration, which is a declared stop condition; choosing a
catalogue inside the canary would be that module deciding which book a question
is about. So the seam is wired and provable, and the deployment decision stays
where it belongs.

### One response contract, not two

A series is N results and `adapt_workflow_result` takes one, so the per-snapshot
results are stacked into a single frame — `reporting_date` + any axes + the value
column the executor itself names — wrapped in the existing `MIQueryResult` and
rendered through the existing contract. No parallel temporal envelope.

Governance evidence rides on the existing `metadata.governedPlan` block
additively: `requested` carries the plan's temporal semantics, and `executed`
carries per-snapshot identity, applied predicates, grouping, aggregation, row
counts, values/cells and the comparison arithmetic. A Slice 1 answer passes
nothing new and is byte-identical to what it always was.

### Proved through the real serving path

`temporal_serving_integration.py` calls `plan_serving_canary.serve` — the
production entry point — with a canary principal, and lets the real path do the
rest: the real allow-list, the real interpreter seam (replaying **recorded Opus
payloads**, zero calls), the real compiler, the real dispatch, the real
perimeter, the real runtime, the real renderer.

Figures are read back off the **final API payload**, not off the runtime's return
value — an answer that reconciles internally and loses its figures on the way out
is exactly what a serving proof is for.

```
TEMPORAL_POSITIVE_CASES = 5      TEMPORAL_POSITIVE_PASS = 5
SERIES_VALUES_RECONCILED       = 24
GROUPED_CELLS_RECONCILED       = 40
PERIOD_COMPARISONS_RECONCILED  = 1
```

| case | what | served |
|---|---|---|
| S2-1 | funded balance over last N months | NEW, 6 snapshots, 6 rows |
| S2-2 | loan count each month | NEW, 8 snapshots, 8 rows |
| S2-3 | filtered temporal series | NEW, 8 snapshots, predicate on every one |
| S2-4 | time × one dimension | NEW, 8 snapshots, 40 cells |
| S2-5 | current versus previous period | NEW, 2 snapshots, change reconciled |

### Negative controls, through the same seam

```
T10_CONTROL                  = LEGACY_FALLBACK  INELIGIBLE:OPERATION_NOT_TEMPORAL
MOVEMENT_ATTRIBUTION_CONTROL = LEGACY_FALLBACK  INELIGIBLE:CAPABILITY_NOT_GENERIC
UNAVAILABLE_PERIOD_CONTROL   = LEGACY_FALLBACK  TEMPORAL_NOT_RESOLVED:PERIOD_NOT_AVAILABLE
OVERLONG_PERIOD_CONTROL      = LEGACY_FALLBACK  no snapshots selected — not shortened
MULTI_OUTPUT_CONTROL         = LEGACY_FALLBACK  INELIGIBLE:NOT_SINGLE_OUTPUT
NON_CANARY_PRINCIPAL         = not handled, no record written, legacy unchanged
```

T10 remains exactly the safe refusal it was. Nothing converts it to a series.

### Regression

```
SLICE_1_REGRESSIONS_ATTRIBUTABLE = 0
LEGACY_REGRESSIONS_ATTRIBUTABLE  = 0
```

269 tests across the serving canary, the adapter, the shadow wiring, the
evidence recorder, the replay and the deployed-acceptance harness. The Slice 1
corpus replay reproduces **byte-identical**. Slice 2's offline acceptance is
31/31 and the boundary regression is 7/7 + 4/4.

```
INTERPRETATION_V2_CHANGED   = NO   (0 files)
COMPILER_CHANGED            = NO
TEMPORAL_VOCABULARY_CHANGED = NO
NEW_SERVING_FLAG_ADDED      = NO   (two env vars before, the same two after)
LIVE_OPUS_CALLS             = 0

SLICE_2_SERVING_INTEGRATION_OFFLINE = PASS
```

---

## SLICE 2 PRODUCTION SNAPSHOT-STORE BINDING

The blocker named at the end of the serving integration — `serve()` takes a
`snapshot_store` and `mi_service` supplied none — is closed. This is dependency
wiring: no new catalogue, no new dates, no new setting.

### Phase 1 — the trace

```
PRODUCTION_FRAME_OWNER      = mi_agent_api/datasets.py
                              _resolve_query_frame  (the active/latest frame)
                              _resolve_run_dataframe (one dated run; blob + on-disk)
PRODUCTION_CATALOGUE_OWNER  = the index {portfolios:[{client_id, label,
                              runs:[{run_id, reporting_date, loan_count, …}]}]}
AVAILABLE_SNAPSHOT_SOURCE   = blob dated platform canonicals
                              -> on-disk onboarding central tapes
                              -> the loaded platform canonical
SNAPSHOT_IDENTITY           = client_id/run_id — the existing production
                              portfolioId form; reporting_date is the period
SNAPSHOT_STORE_BINDING_SEAM = mi_service -> serve(snapshot_store=…)
```

Production **does** have an authoritative catalogue, so the stop condition did
not trigger. That index is what `/mi/snapshots` serves, what the portfolio and
reporting-date dropdowns are built from, what `evolution.py` builds its periods
from and what `_resolve_run_dataframe` loads a dated run against. It is real and
it is already trusted; inventing a second list of a book's months would have been
the defect.

**One thing had to move.** The three sources all live in `datasets.py`, but the
resolution ORDER between them lived only inside the `/mi/snapshots` route, so
consuming it from anywhere else meant either importing a route or writing the
order out twice. It is extracted verbatim to `datasets.snapshot_index()` and the
route now delegates. That is a reduction in owners, not an addition: a dropdown
that offers a month the temporal runtime cannot resolve is precisely the
disagreement two copies would produce.

### What was built

```
FILES_CHANGED              = 4 product (1 new), 1 test (new), 2 evidence (1 new)
PRODUCT_NET_EXECUTABLE_LOC = +136
    mi_agent_api/governed_snapshot_store.py  +129   NEW — the adapter
    mi_agent_api/datasets.py                  +28   snapshot_index(), extracted
    mi_agent_api/app.py                       −25   the route delegates
    mi_agent_api/mi_service.py                 +4   the binding

NEW_CATALOGUE_CREATED = NO
NEW_ENV_VAR_ADDED     = NO
HARDCODED_SNAPSHOTS   = 0
```

`GovernedFundedSnapshotStore` implements the four `SnapshotStore` primitives over
the production catalogue and the production dated-run loader, injected rather
than imported so it can be exercised against a production-shaped index. It
contains no reporting date, no environment read and no directory walk of its own
— a test asserts all three off the AST. Blob and onboarding-root knowledge stays
in `datasets.py` where it already was; `plan_temporal_runtime` and
`SnapshotSelector` learn nothing about storage and are byte-unchanged.

### Scope is a property of the object — found by its own control

A store built for a request is **bound** to that request's client. Any other
client is refused by `list_snapshots`, `get_snapshot` and therefore `load_loans`,
whatever id a caller supplies.

That was not the first design. `get_snapshot` derived the client from the
snapshot ID and then listed that client's runs, so a store built for one tenant
would resolve and load another tenant's run if simply handed its id. Nothing on
the temporal path does that — the runtime only ever asks for the request's own
client — but "no caller happens to do it" is not a boundary, and the
cross-portfolio control is what found it. Both the binding and the hole it closed
are asserted in
`mi_agent_api/tests/test_governed_snapshot_store.py`.

### Approval semantics were not invented

The catalogue records `run_id`, `reporting_date` and size per run and carries
**no per-run approval or status field**; `evaluate_source_approval` governs the
ACTIVE source, not each historical run. So the adapter applies the scoping that
exists — client, funded route, a resolvable reporting date — and nothing else.
A run the catalogue does not list is not selectable, and that is the only
approval semantics the estate keeps today. Adding more would be new onboarding
governance, which this task was told not to build.

### Proved through `serve`, against a production-shaped catalogue

`temporal_production_binding.py` builds the exact index contract
`snapshot_index` returns, over the independent oracle's history, **with two
clients in it** — a binding that scopes correctly against a single-client index
has proved nothing about scoping.

```
CURRENT_SLICE1_CONTROL     = PASS  a current-period plan never reaches the
                                   temporal runtime; its figure is unchanged
TEMPORAL_SERIES_CONTROL    = PASS  6 snapshots, values reconciled
FILTERED_TEMPORAL_CONTROL  = PASS  8 snapshots, predicate applied on every one
GROUPED_TEMPORAL_CONTROL   = PASS  8 snapshots, 40 cells reconciled
UNAVAILABLE_PERIOD_CONTROL = PASS  TEMPORAL_NOT_RESOLVED, zero points produced
CROSS_PORTFOLIO_CONTROL    = PASS  another client, the pipeline route and an
                                   unlisted run are all unreachable
NO_CATALOGUE_CONTROL       = PASS  slice 1 keeps working; slice 2 fabricates
                                   nothing from an empty index
NO_STORE_CONTROL           = PASS  TEMPORAL_STORE_UNAVAILABLE — production today

SERIES_VALUES_RECONCILED  = 14
GROUPED_CELLS_RECONCILED  = 40
SNAPSHOT_SELECTION_ERRORS = 0
SILENT_PERIOD_SUBSTITUTIONS = 0
```

Figures are read off the **final API payload**, not the runtime's return value.

### Regression

```
SLICE_1_REGRESSIONS_ATTRIBUTABLE = 0
```

186 tests across the serving canary, the adapter, the temporal runtime and the
shadow wiring. Slice 2's offline acceptance is 31/31 and the serving integration
is 11/11. On the API side, the whole `mi_agent_api` suite was run at
`13ccac54` and with the binding: the failing node-id set is **identical**, 62
entries either way, and every one is a missing optional dependency (`rapidfuzz`,
`jwt`) rather than a behaviour. `test_snapshots.py` and `test_mi_service.py`
pass — 38 tests over the route and the service this change touched.

```
INTERPRETATION_V2_CHANGED  = NO
COMPILER_CHANGED           = NO
TEMPORAL_RUNTIME_CHANGED   = NO   (plan_temporal_runtime.py byte-unchanged)
SNAPSHOT_SELECTOR_CHANGED  = NO   (states/selectors.py, snapshot/store.py unchanged)
LIVE_OPUS_CALLS            = 0

SLICE_2_PRODUCTION_SNAPSHOT_BINDING = PASS
```

### What is still true about deployment

Nothing was deployed and no flag was changed. `MI_AGENT_PLAN_SERVE` still
defaults off and the principal allow-list is still empty, so the temporal path
remains unreachable in production until someone deliberately enables the canary
for a named principal. What has changed is that the catalogue is no longer the
thing standing in the way.

The proof runs against a production-**shaped** catalogue, not a production one:
this environment has no blob root and no onboarding output, so what is
demonstrated is the adapter's reading of the contract, not a live storage
round-trip. That is the remaining gap, and it is a deployment-environment
question rather than a code one.

---

## KNOWN BOUNDARY: TIME × GEOGRAPHY IS INELIGIBLE

"Show funded balance by region over the last three months" is listed as an
in-scope Slice 2 shape and comes back INELIGIBLE (`GEOGRAPHY_REQUESTED`).

That is deliberate and it is not a silent narrowing — the question is refused
whole, with a reason, and nothing is computed on a different axis in its place.

The reasoning: the accepted Slice 1 adapter binds no geography at all. A plan's
`GeographyBinding` carries a basis and level resolved by
`config/asset/mi_geography.yaml`, and `spec_for_plan` places none of it into the
`MIQuerySpec`; Slice 1 recorded five eligible plans (Q14A/B/C, Q16A, Q16C) that
were being grouped on their OTHER axis alone, and closed the hole by refusing
them. The Slice 2 contract says *dimensions: existing governed Slice 1
dimensions*, and Slice 1's dimensions exclude geography. Binding a geography
axis here would be new capability, not a temporal extension, and the instruction
was not to widen eligibility merely because the interpreter can describe the
question.

Governed geography semantics are preserved exactly by leaving them alone: no
ITL3, no dashboard geography, no second region owner. Making time × geography
available is a geography-owner task, not a temporal one.

---

## HOW TO RE-RUN

```bash
# offline acceptance bank (zero model calls)
python due_diligence/evidence/plan_temporal_slice2/slice2_acceptance.py

# unit suite
python -m pytest mi_agent/tests/test_plan_temporal_runtime.py -q

# slice 1 non-regression
python -m pytest mi_agent/tests/test_plan_runtime_adapter.py \
                 mi_agent/tests/test_plan_shadow_wiring.py -q
python due_diligence/evidence/plan_shadow_slice1/corpus_replay.py
```

---

```
SLICE_2_OFFLINE_READY = YES
```

Offline only. Nothing is deployed, nothing is served, no flag was added, and
`mi_service` has no call site for any of this — `plan_temporal_runtime` is
reachable only from a test or from this harness. Production canary and Slice 3
were not started.

---

## PRODUCTION ACCEPTANCE TOOLING

Built after the snapshot-store binding, and separately authorised. It changes no
product file and asks production nothing until an operator dispatches it.

### Why a new script was needed

Neither existing harness can adjudicate a temporal answer, and neither was
altered:

| harness | why it cannot do this |
|---|---|
| `slice1b_serving_canary/run_serving_acceptance.py` | reads `serving.decision` correctly, but its bank is `[c for c in manifest["cases"] if c["expected_slice1_eligible"]][:6]` from a **sha256-pinned slice 1 manifest** with no bank parameter. It cannot ask a temporal question, and swapping the manifest would break the integrity gate that is the point of pinning it. |
| `mi_api_certification/time_dimension_acceptance.py` | asks a different fixed matrix and contains **zero** references to `serving` or `snapshot`, so it can tell neither NEW from LEGACY nor which periods were selected. |

Neither reconciles a per-period value, a grouped cell or a comparison against
the execution receipt.

### What the new script adds, and only this

The six-question bank and the *reading* of a temporal answer. Everything else is
imported from machinery already accepted in this estate:

```
transport / MI_BEARER    certify_mi_api._live_asker
zero-cost provenance     run_serving_acceptance.served_commit  (GET /health)
HTTP figure extraction   run_serving_acceptance.response_figures  (pinned to)
Kudu evidence sink       run_acceptance.Sink
correlation-aware poll   run_acceptance.poll_for
publish-profile creds    run_acceptance.publish_profile_credentials
model-id gate            run_acceptance.is_required_model
secret hygiene           run_acceptance.scrub / _save / MIN_SECRET_LENGTH
```

`certify_mi_api.preflight` was deliberately **not** reused: it POSTs *"What is
the total balance?"*, which against a switched-on canary costs a live Opus
interpretation. Provenance must cost nothing, so it is established by `GET
/health` and a mismatch stops the run with zero questions asked.

### The execution receipt is the numeric oracle

No portfolio figure is written down. Each case asserts that what the **caller
received** equals what the deterministic runtime **recorded having computed**,
period by period and cell by cell. A harness carrying its own expected balances
would go stale the moment the book moved and would be testing its own memory
rather than the service.

### Two offline modes, neither of which calls anything

`--self-test` — 32 rules, stdlib only (no pandas, no `mi_agent`), so it runs on
a bare runner. Every reconciliation is proved twice: that it accepts a faithful
response *and* that it **catches** its own failure — a drifted figure, a dropped
period, a period the runtime never executed, a wrong grouped cell, a dropped
cell, a change the comparison's own two values do not imply, a wrong percentage,
a shortened series on the negative control. It also pins the restated
`value_column` rule against `plan_runtime_adapter`'s source, so the harness
cannot silently drift from the adapter that owns that naming.

`--shape-check` — drives the **real** `plan_serving_canary.serve` offline
(recorded Opus payloads, replayed through the sanctioned `set_interpreter_factory`
seam) and feeds the records and envelopes the *product* writes through this
file's own adjudicator. It answers the one question a synthetic self-test
cannot: whether the harness reads the shape the product actually emits.

```
PASS T06  series      decision=NEW             periods=8 rows=8  reconciled=8
PASS T07  grouped     decision=NEW             periods=8 rows=40 reconciled=40
PASS T11  comparison  decision=NEW             periods=2 rows=2  reconciled=4
PASS T13  negative    decision=LEGACY_FALLBACK periods=0 rows=0  reconciled=0
4/4 kinds adjudicated against records the product wrote
```

Two defects were found this way and fixed before any live use:

1. **Rows counted twice.** `response_figures` returns *copies* of the artefact
   rows, so merging its output with a sweep of the artefacts could not tell its
   rows from the originals and doubled every period. `http_rows` now sweeps the
   artefacts once and is pinned to `response_figures` by a self-test rule
   asserting the two agree on a single-grid envelope.
2. **The negative control failed for the wrong reason.** It treated *any*
   dated rows in the response as a defect. Once the decision is a fallback the
   caller is holding the **legacy** answer, and whether that path ships dated
   rows of its own is not a slice 2 property and was not changed by slice 2 —
   failing on them would report a legacy behaviour as a temporal defect. The
   row check is now gated on the decision being NEW; the control still turns on
   the two facts that decide it (not served, and no period executed).

### The push trigger asks nothing

`slice1b-serving-acceptance.yml` uses a self-scoped `push` trigger because a
dispatchable workflow must be registered on the default branch before the API
will accept a dispatch. The same device is used here — but this push path runs
the **offline self-test only**. Slice 2 is not deployed and the canary is off,
so asking the bank now would spend six live Opus interpretations to prove that
the legacy path still serves: a guaranteed and meaningless FAIL. The live bank
requires an explicit dispatch with `mode: bank`, a deliberate operator act taken
after the deployment and the canary are both in place. A bank run without a
pinned `expect_commit` is refused before the first question.

Confirmed on the runner. The push that published this fired run
[34582097118](https://github.com/joshua1602-svg/trakt/actions/runs/34582097118):

```
success  Prove the harness before it is pointed at anything   (1s)
skipped  Refuse to run the bank without the credentials it needs
skipped  Run the six-question temporal acceptance
skipped  Keep the evidence whatever the verdict
skipped  Turn the serving canary back off
```

Which settles two things at once: every live step stayed skipped, and the
self-test really does run on a bare runner — the job has no `pip install` and
finished the harness proof in a second, so nothing in the acceptance path
reaches pandas or `mi_agent`.

```
SLICE_2_ACCEPTANCE_TOOL_READY = YES
PRODUCT_FILES_CHANGED = 0
ACCEPTANCE_TOOL_LOC = 808 (script) + 178 (workflow); 556 of the script is code,
                      179 of that the self-test and 64 the shape check
```

Nothing was deployed, `MI_AGENT_PLAN_SERVE` was not enabled, Opus was not
called, Slice 2 was not modified and Slice 3 was not started.
