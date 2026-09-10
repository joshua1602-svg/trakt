# Deterministic execution baseline, under the frozen interpretation layer

Measurement only. No deterministic code was changed, no branch merged, no commit
cherry-picked, no model called.

```
BASE_SHA                               = ea8c65b592ae5d2203d924bf66afa051618641e6
TEST_SHA                               = 2b00172c0b033c97de8afe5fa5b60ba43e1443ec
DETERMINISTIC_BRANCH                   = origin/claude/borrowing-base-mi-query-cvplz0
DETERMINISTIC_BRANCH_HEAD              = 101ac7aa927760600bc926f84a3a6ee35a9008a8
deterministic files baseline-identical = YES
```

## Phase 0 — provenance gate

Repository `joshua1602-svg/trakt`, branch
`claude/mi-plan-interpretation-compiler-v1-jfonpp`, clean tree. Both SHAs exist.

`ea8c65b` → `2b00172` changes **53 files, every one an ADD (status `A`), and every
one inside `mi_agent/interpretation_v2/` or `tests/interpretation_v2/`.** There is
not a single `M` in the diff. Proved three ways:

1. `git diff --name-only ea8c65b 2b00172` excluding those two directories returns
   nothing.
2. Tree-object comparison of every entry under `mi_agent/` and `tests/`: the only
   differing entry in each is the added `interpretation_v2`; nothing was removed.
3. Top-level tree objects for every other path are byte-identical.

So the deterministic engine at the test SHA **is** the accepted baseline, and the
gate passes. Confirmed empirically afterwards: the bank scores 16/20 identically
when run against the repo root and against the detached `2b00172` worktree.

Note on the deterministic branch: it contains `ea8c65b` and is 33 commits ahead,
touching 101 files outside the isolated work. `origin/main` (4 commits) is a
subset — 3 of its commits are ancestors of the deterministic branch. `2b00172`
contains neither, so the interpretation work branches cleanly off the baseline.

## Phase 1 — test environment

Detached worktree at `2b00172`, production code unmodified. Five test
dependencies were installed into the environment (`fastapi`, `cffi`,
`python-pptx`, `matplotlib`, `rapidfuzz`, `python-multipart`); no repository file
was touched.

The boundary actually exercised:

```
QueryPlan (constructed directly)
    -> compile_query_plan -> MIQuerySpec
    -> execute_mi_query / execute_query_plan
    -> MultiResultEnvelope + metadata receipt
    -> compared against independent truth
```

No `llm_query_parser`, no `ParsedQuestion`, no recogniser cascade, no
raw-question routing, no Opus. Every plan is built from `AnalyticalScope`,
`Predicate` and `PlannedOutput` in code.

## Phase 2 — what already existed, and what was added

Inventory first. The estate already has the hard part: **`mi_agent/tests/portfolio_truth_oracle.py`**
is an independent oracle that imports nothing from the product — pandas and
explicit column names over a deterministic 400-row book — and
`test_portfolio_truth_bank.py` already drives plans against it with no natural
language. It passes 22/22 at the frozen SHA.

So the bank reuses that oracle rather than re-deriving truth, and adds the
surfaces it does not reach: empty populations, absent dimensions, ITL3 geography,
the balance bridge, period honouring and receipt disclosure. Bridge truth is
recomputed here longhand (continuing / new / redeemed from two loan-level maps);
no expected number anywhere comes from the engine's own answer.

Also inventoried and deliberately **not** converted into bank cases, because
their truth is question-level rather than plan-level:
`config/mi/golden_questions/ere_mi_calibration_250.yaml`,
`migration_phase0/MI_ACCEPTANCE_BANK_ANSWERS.json`,
`migration_phase0/{BORROWING_BASE_MI,STAGE_MOVEMENT,SIMPLE_COMPOSITION}_BANK.yaml`,
`SOURCE_SCOPE_TRUTH_TABLE.json`, the stage-movement fixtures. Feeding those
through the parser would have measured the thing this phase is trying to exclude.

```
BANK_SIZE                  = 20
PASS                       = 16
FAIL                       = 4
GOVERNED_REFUSAL_EXPECTED  = 3   (B4 empty population, C3 absent dimension,
                                  F1 unhonourable period)
UNEXPECTED_REFUSAL         = 0
SILENT_SEMANTIC_ERROR      = 3   (B4, D2, F1)
NUMERICAL_ERROR            = 0
EXECUTION_ERROR            = 0
```

Evidence: `plan_execution_bank.py` (re-runnable; `DET_BANK_ROOT` points it at any
worktree) and `plan_execution_bank_result.json` (per-case plan, expected result,
actual result, receipt, verdict, classification).

### Capability breakdown

| surface | pass / cases |
|---|---|
| AGGREGATION/STATISTIC | 6 / 6 |
| PLAN_BINDING/DIMENSIONS | 2 / 2 |
| SPECIALIST_CAPABILITY (bridge) | 3 / 3 |
| POPULATION/FILTER | 3 / 4 |
| GEOGRAPHY | 1 / 2 |
| RECEIPT/GOVERNANCE | 1 / 2 |
| TEMPORAL_RESOLUTION | 0 / 1 |

What is solid: sums, counts, simple and weighted means, single and conjoined
filters, one- and two-dimensional grids cell by cell, grids reconciling to their
population total, three measures over one population, and the balance bridge —
whose components (`continuing_movement` 40, `new_loan_balance` 400,
`exited_loan_balance` 300, residual 0.0, `reconciles` True) match a longhand
decomposition exactly, including full redemption and an identical-snapshot case
that correctly reports zero.

A filter true of every row is applied **and** disclosed: the receipt carries
`filter current_loan_to_value gt 0.0 kept 400/400 rows`. An absent dimension is
never substituted by another field.

## Failure inventory

Each failure gets exactly one primary owner.

**1. B4 — an empty population returns a confident zero.** `SUM(balance)` with
`LTV > 1000` matches no row. The executor returns `0.0` with
`filtered_row_count=0` and no unavailability warning, so a question about a
population that does not exist is answered with a number. **Owner:
RECEIPT/GOVERNANCE.** Silent semantic error: numerically defensible, semantically
wrong — the brief's own rule is that unavailable inputs must produce a governed
unavailable, not an invented zero. Note the estate already refuses an empty
*aggregation* (`test_empty_aggregation_refuses` passes, 8/8) — that guard covers
an all-null series, not a filter that removes every row.

**2. D2 — raw ITL3 codes presented as region labels.** Grouping on
`geographic_region_obligor_itl3` returns group keys `TLI31`, `TLJ21` verbatim. The
arithmetic is right and the labels are machine codes offered as places. **Owner:
GEOGRAPHY.** Silent semantic error.

**3. F1 — a period the book cannot honour is silently ignored.** The plan states
`period="1999-12-31"`; the flat book carries no period column. The executor
returns the whole-book total `115,450,800.70` — the same figure as for
`2026-06-30` — with no refusal and no disclosure. **Owner: TEMPORAL_RESOLUTION.**
Silent semantic error, and the most serious of the four: the answer is a
plausible number attached to the wrong period.

**4. F2 — the receipt does not state which period was executed.** Result metadata
carries `dataset`, `applied_predicates`, `applied_filter_fields`,
`filtered_row_count`, `input_row_count` and more, but no `period`,
`reporting_date`, `as_of_date` or equivalent. **Owner: RECEIPT/GOVERNANCE.** This
is F1's root cause as much as a defect of its own: the plan's period never
reaches the receipt, so nothing downstream can notice it was not honoured.

Against the brief's hypotheses: *zero comparable metrics returning success*
reproduced (B4). *ITL3 geography binding* reproduced (D2). *Relative
period-pair / observation-gap semantics* reproduced in the adjacent and more
basic form that a stated period is not honoured at all (F1/F2). *Funded movement
/ bridge execution* did **not** reproduce — the bridge is correct on every case
tried. *Portfolio-summary behaviour* and *specialist capability execution* beyond
the bridge were not reached at plan level in this phase and remain unmeasured;
they are stated as not-covered rather than as passes.

## Reconciliation against the deterministic branch (read-only)

The single most useful result of this phase:

> **The deterministic branch does not modify `mi_query_executor.py`,
> `query_plan_execution.py` or `query_plan.py` at all.** `git diff --name-only
> ea8c65b origin/claude/borrowing-base-mi-query-cvplz0 --` those three files
> returns nothing.

It works one layer up — `mi_agent_api/*` routes, `question_interpretation/*`,
`period_change/recognition.py`, `llm_query_parser.py`, `mi_geography.py`,
`mi_query_spec.py`, `query_plan_adapter.py`, `execution_receipt.py`. So the two
workstreams are complementary rather than overlapping.

| failure | addressed on the branch? | relevant commit | evidence |
|---|---|---|---|
| B4 invented zero | **NO** | none found | No commit mentions an empty population or an invented zero; the executor is untouched |
| D2 ITL3 labels | **PARTIAL / UNKNOWN** | `101ac7aa` "a column of codes is not a column of place names" | Exactly the right diagnosis — but it changes `mi_agent_api/geo.py` only (+54/−5), fixing ITL3 basis acceptance on the `exposure_by_itl3` route. The executor's grouped labelling is not in that diff, so whether this case converges is unproven |
| F1 unhonourable period | **NO at this layer** | `05dbe95f` "one period-pair owner" | Changes `mi_agent_api/temporal_compare.py`, `movement_summary.py`, `question_interpretation/*` — which period a QUESTION means, not whether the executor honours a period a PLAN states |
| F2 receipt omits the period | **NO** | `2ef7fb3e` (only `execution_receipt.py` change) | That commit is about a weighting naming its weight and a parallel question being one list; it adds no period field |

`0 of 4` observed defects are demonstrably addressed; `1 of 4` is partially
addressed at a different layer.

## Supplementary: the existing suite at the frozen SHA

Not the bank, but useful context. `mi_agent/tests` gives **1404 passed, 22 failed,
21 errors, 264 skipped** (5m29s). Deduplicated, that is 16 distinct failing
groups — among them three `test_p0_execution_receipt` cases where the engine
**refuses** ("not available in this dataset … no value was fabricated") while the
test expects a success carrying a disclosure. That is the engine being more
conservative than its test, not a widening defect, and it is pre-existing at the
baseline. The 21 errors are all one module, `test_an_as_at_date_is_accounted_for`,
failing at fixture setup — which the deterministic branch independently
identified (`fe7fea03`, `2ac7bffb`: "my test fixtures were changing the meaning of
other files' tests").

`tests/` and `mi_agent_api/tests` did not finish inside this phase's window and
are reported as not-measured rather than estimated.

## Verdict

```
DETERMINISTIC_BASELINE_ESTABLISHED           = YES
CURRENT_EXECUTION_CORRECT_RATE               = 16/20 (80%) on the plan-level bank
                                               22/22 on the pre-existing truth bank
SILENT_SEMANTIC_ERRORS                       = 3  (B4, D2, F1)
EXISTING_DETERMINISTIC_BRANCH_APPEARS_TO_ADDRESS = 0/4 observed defects
                                               (1 of 4 partially, at another layer)
```

**RECOMMENDED_NEXT_STEP.** Do not merge the deterministic branch expecting it to
close these four. It is aimed at the routing and interpretation layers and leaves
the plan-level executor untouched, so it would neither fix nor disturb what this
bank measures — which also means it can be assessed on its own merits separately.

Fix F2 first, and F1 falls out of it: the plan's period must reach the execution
receipt, because until it does, nothing can detect that a stated period was
ignored. That is one field on the receipt plus a guard that refuses when the
dataset cannot honour the period. B4 is the same shape of fix — a governed
unavailable where the population is empty — and both belong in
`mi_query_executor.py`, which neither workstream currently owns. D2 needs the
`101ac7aa` diagnosis applied at the executor's labelling step, not only on the
geo route.

Extending the bank is worth doing before any of that: borrowing base,
concentration, ranking, distribution and portfolio summary are unmeasured at
plan level, and three of four defects found here were in the surfaces that had
the least plan-level coverage.
