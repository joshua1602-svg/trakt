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

## PHASE 5 — THE OPUS → TEMPORAL BOUNDARY: PRE-REGISTERED, HARNESSED, NOT RUN

The one open question Slice 2 ended on — *does Opus state temporal intent in a
shape the deterministic layer can act on?* — now has a bank, a manifest and a
harness. It does not yet have an answer, and the reason is a missing credential,
not a missing test.

```
LIVE_BANK_SIZE          = 15   (11 positives + 4 controls)
AUTHORISED_LIVE_CALLS   = 15
LIVE_CALLS_MADE         = 0
BLOCKER                 = no ANTHROPIC_API_KEY in this environment
```

The product's own interpreter says so, rather than this report asserting it:

```
AnthropicInterpreterClient().available  ->  False
OpusInterpreter(...).interpret(q).reason ->
    {'code': 'MODEL_UNAVAILABLE', 'subject': 'interpreter',
     'detail': 'no ANTHROPIC_API_KEY in the environment'}
```

`ANTHROPIC_BASE_URL` is set to the public `https://api.anthropic.com` and carries
no credential; there is no `.env`, no key vault fallback, and every call site in
the estate reads `os.environ["ANTHROPIC_API_KEY"]` directly. The harness detects
this, prints `NOT RUN`, exits 2 and writes no result file — it does not fabricate
one.

### The boundary under test, and where it stops

```
raw question -> OpusInterpreter.interpret        (live, frozen interpreter)
             -> CandidateIntent                   (the model's only output)
             -> DeterministicCompiler.compile     (frozen)
             -> GovernedQueryPlan
             -> check_temporal_eligibility
             -> resolve_temporal -> SnapshotSelector -> resolved snapshots
             STOP
```

No MI API call, no `MI_BEARER`, no execution against a book, no serving surface,
and **no product module changed** — the whole of Phase 5 is new files under
`due_diligence/evidence/plan_temporal_slice2/`.

### What is pinned, and what deliberately is not

A pre-registration that demanded a particular `time.form` or `operation` would be
scoring Opus against this repository's guess at its wording. `series` and `range`
are both span forms the perimeter admits; `series`, `breakdown` and
`point_in_time` are all operations it admits.

| pinned | not pinned |
|---|---|
| slice 2 eligibility | which span form the model chose |
| the resolved snapshot set | which operation it chose |
| measure concept, statistic (from an allowed set), filters, dimensions, population base | its wording of the period label |
| the absence of any physical date or snapshot id | |

`expected_snapshots` is hand-written against the eight-month fixture and never
read back from the product. Relative windows resolve against the **catalogue's**
latest period, not wall-clock, so the expectation does not drift with the date.

### The harness is proved before a penny is spent — both ways

`--dry-run` replays authored stand-in payloads through the identical pipeline:
**15/15 boundary held** (11 `TEMPORAL_BOUNDARY_HELD`, 2 `JUSTIFIED_INELIGIBLE`,
1 `JUSTIFIED_REFUSE`, 1 `JUSTIFIED_CLARIFY`).

A harness whose only demonstrated outcome is PASS has demonstrated nothing, so
`--degraded` replays the same fifteen questions with four payloads deliberately
broken. All four are caught, each named precisely:

| injected fault | verdict | what the harness reported |
|---|---|---|
| T01 window in words only, no `periods_back` | `PERIOD_UNRESOLVABLE` | `PERIOD_LABEL_UNRESOLVED: no governed period is named by ['the last six months']` |
| T02 four periods where three were asked | `WRONG_SNAPSHOTS` | expected 3 dates, got 4 |
| T05 time constraint dropped (`form: current`) | `TEMPORAL_INTENT_ABSENT` | "the time constraint did not survive the model boundary" |
| T06 filter dropped | `SEMANTIC_DEVIATION` | `filters: [['erm_product_type','eq','drawdown']] -> []` |

The degraded run found a real defect in the harness's own diagnostics, which is
recorded rather than quietly fixed: a dropped time constraint was first reported
as `UNEXPECTED_INELIGIBLE` with no detail, because the perimeter refuses
`current` as `PERIOD_NOT_TEMPORAL` before any model-facing check ran. Those are
the same event, and only one of them is a finding *about the model*. The
`temporal_required` check now runs **before** eligibility so the finding is named
as one.

### The single most likely live failure, named in advance

T01's pre-registration says it plainly: *"The resolver can only honour it if the
model supplies `periods_back=6`; a labels-only interpretation clarifies, and that
would be the finding."* The recorded corpus is not reassuring on this — NL1A came
back with `labels: ["last few months"]` and `periods_back: null`. If the live run
returns `PERIOD_UNRESOLVABLE` on the count-bearing cases, the fix is a prompt or
vocabulary change in `interpretation_v2`, not a widening of the resolver, and it
belongs to a slice that is authorised to reopen interpretation.

### To run it

```bash
python due_diligence/evidence/plan_temporal_slice2/temporal_live_run.py --dry-run    # harness, free
python due_diligence/evidence/plan_temporal_slice2/temporal_live_run.py --degraded   # harness catches faults, free
ANTHROPIC_API_KEY=... python due_diligence/evidence/plan_temporal_slice2/temporal_live_run.py
```

The third command is the only one that spends anything: at most 15 calls, one
per case, no retry and no reworded second try. It verifies
`temporal_bank_manifest.sha256` against the committed manifest first and refuses
to run if the expectations have moved since they were registered.

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
