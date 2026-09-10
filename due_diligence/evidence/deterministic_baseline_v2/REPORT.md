# Deterministic execution baseline v2 — broader, and geography-corrected

Measurement only. No product code changed, nothing merged, no model called, no
deployed API touched. The previous reconnaissance evidence in
`../deterministic_baseline/` is untouched.

```
BASE_SHA             = ea8c65b592ae5d2203d924bf66afa051618641e6
TEST_SHA             = 2b00172c0b033c97de8afe5fa5b60ba43e1443ec
TREE_CLEAN           = YES   (fresh detached worktree, 0 modified files)
PRODUCT_CODE_CHANGED = NO
```

Phase 0 proofs, all re-established on a fresh worktree:

* HEAD exactly `2b00172c0b033c97de8afe5fa5b60ba43e1443ec`, 0 modified files.
* Deterministic production files identical to the accepted baseline: the
  `ea8c65b..2b00172` diff excluding `mi_agent/interpretation_v2` and
  `tests/interpretation_v2` is **empty**.
* **0** commits from the deterministic branch are ancestors of HEAD; `101ac7aa`
  is not an ancestor. Nothing merged, nothing cherry-picked.
* The harness contains **0** references to `interpretation_v2` and imports no
  parser, `ParsedQuestion`, `question_interpretation`, recogniser or Opus path.

```
ORIGINAL_RECON_BANK   = 20 cases, 16 PASS / 4 FAIL
REVISED_RECON_RESULT  = 20 cases, 17 PASS / 2 FAIL / 1 OUTSIDE_CURRENT_SCOPE
```

The original bank was re-run unchanged at this SHA and reproduced
**byte-identically** — `git diff` on its committed result is empty afterwards. The
16/20 is preserved as history; the revisions below are recorded here, not by
editing it.

| recon case | was | now | why |
|---|---|---|---|
| D2-itl3-codes | FAIL (GEOGRAPHY) | **OUTSIDE_CURRENT_SCOPE** | tested a secondary lens, not the MI region route — repository-confirmed below |
| B4-filter-keeps-no-row | FAIL (invented zero) | **PASS** | my judge's error, not the engine's. See the correction below |
| F1-as-at-unhonourable | FAIL | FAIL (root cause now located) | confirmed, re-owned to PLAN_BINDING |
| F2-period-is-recorded | FAIL | FAIL (same root cause) | confirmed |

**Correction to the reconnaissance, stated plainly.** B4 claimed the engine
returns "a confident 0.0 with no unavailability warning" for an empty population.
That was wrong. The receipt carries `filter current_loan_to_value gt 1000.0 kept
0/400 rows` **and** `filtered_row_count: 0`. The population is disclosed as empty
twice over; my judge only searched for the words "empty"/"no rows"/"unavailable"
and missed the engine's own vocabulary. The expanded bank tests this three ways
(B07, B08, B10) and all three pass. **There is no invented-zero defect.**

## Geography — the authoritative contract, proved from configuration

```
CORE_REGION_ASSET_CONFIG_OWNER = config/asset/mi_geography.yaml
                                 :: primary_basis_by_asset_class
                                 (+ config/asset/product_defaults_ERM.yaml for
                                 equity_release / lifetime_mortgage, which declare
                                 their basis in their own asset pack)
                                 resolved by mi_agent.mi_geography
                                 .default_primary_basis -> axis_fields
                                 -> field_for_basis
                                 overridable per portfolio via the governed
                                 portfolio registry (mi_geography.primary_basis)
CORE_REGION_TEST_RESULT        = 5 PASS / 0 FAIL
ITL3_MI_QUERY_STATUS           = OUTSIDE_CURRENT_SCOPE
```

Resolved, not assumed:

| | |
|---|---|
| asset class | `equity_release` |
| `default_primary_basis` | `collateral` |
| `axis_fields('collateral')` | `('collateral_geography', 'geographic_region_collateral')` |
| `field_for_basis(..., frame=book)` | **`collateral_geography`** — the field the bank uses |
| `tiers_for_basis('collateral')` | reporting: `collateral_geography`, `property_region` · code: `geographic_region_collateral`, `geographic_region_collateral_itl3` |

The config file says so itself: *"'balance by region', asked with no qualifier,
has no answer until somebody has decided WHICH geography this book reports on.
That decision is a property of the ASSET, established once at onboarding."* An
asset class absent from the table gets **no** default — `default_primary_basis`
returns `None` rather than guessing.

ITL3 appears only in the **code** tier, never the reporting tier, and
`field_for_basis` returns the most readable field first. So the repository
**confirms** the stated product contract rather than contradicting it: MI does not
resolve an ordinary "region" request to ITL3. Case C07 records this and is scored
`OUTSIDE_CURRENT_SCOPE`, not pass or fail. No ITL3 MI-query translation was
required or tested.

## Expanded bank

```
EXPANDED_BANK_SIZE        = 65
PASS                      = 47
FAIL                      = 11
EXPECTED_GOVERNED_REFUSAL = 4   (B07, B08, B11, F06 — all honoured)
UNEXPECTED_REFUSAL        = 0
SILENT_SEMANTIC_ERROR     = 11
NUMERICAL_ERROR           = 0
RECEIPT_GOVERNANCE_ERROR  = 2
NOT_PLAN_LEVEL_REACHABLE  = 5
UNJUDGEABLE               = 1
```

Every case declares an assertion descriptor — the object, the field and the truth
source — and the runner captures the actual runtime type it read. `pick()` never
defaults: reading an absent field records `<ABSENT>`, which is what made the
missing period and the missing dataset visible instead of invisible. A judgement
that cannot be made honestly is `UNJUDGEABLE`, never a pass.

| surface | pass | fail | not reachable | key finding |
|---|---|---|---|---|
| AGGREGATION_STATISTIC | 18 | 0 | — | arithmetic is sound across sum, count, mean, weighted mean, median, min, max, grids, shares and nulls |
| POPULATION_FILTER | 10 | 0 | — | filters narrow correctly and are disclosed; empty populations ARE disclosed |
| CORE_GEOGRAPHY | 5 | 0 | — | the asset-configured region field is bound and reconciles; unresolved regions are excluded **and warned about** |
| RECEIPT_GOVERNANCE | 7 | 2 | — | predicates, row counts, grouping, aggregation and measure all evidenced; **dataset and period are not** |
| TEMPORAL_RESOLUTION | 1 | 6 | — | only the current period is right, because it is the only one the executor can see |
| PLAN_BINDING | 0 | 3 | — | **the root cause**: three of four scope facts never reach the spec |
| SPECIALIST_CAPABILITY | 6 | 0 | 5 | the bridge is correct on five cases; no-facility refuses; five capabilities are not plan-level expressible |

## The eleven failures are one defect

`AnalyticalScope` carries four population-defining facts — `dataset`,
`portfolio_lens`, `period`, `filters`. `mi_agent/query_plan_compiler.py` binds
**only `filters`**. The words `dataset`, `portfolio_lens` and `period` appear in
that file **zero times each**, verified by count. Proof, from B13:

```
plan asks : dataset='funded'  portfolio_lens='direct'  period='2026-03-31'  filters=1
survives  : filters only
LOST      : dataset, portfolio_lens, period
```

So the executor is never told which period, which dataset or which lens was
requested. It cannot honour them, and the receipt cannot evidence them. Every one
of the eleven failures is a symptom of that single binding gap:

| # | case | requested semantics | actually executed | expected | actual | receipt evidence | primary owner | silent |
|---|---|---|---|---|---|---|---|---|
| 1 | **B13** | dataset + lens + period + filters | filters only | all four bound | 1 of 4 | — | **PLAN_BINDING** | YES |
| 2 | D07 | period `2026-03-31` into the spec | nothing | a temporal field set | `reporting_date=None`, `as_of_date=None`, `temporal_mode=None`, `execution_mode=None`, `compare_periods=[]` | PLAN_BINDING | YES |
| 3 | B12 | lens `direct` into the spec | nothing | `portfolio_lens='direct'` | `None` | PLAN_BINDING | YES |
| 4 | D02 | balance at `2026-03-31` | whole current book | refusal or a period-specific figure | `115,450,800.70`, no period, no warning | TEMPORAL_RESOLUTION | YES |
| 5 | D03 | balance at `2026-05-31` | whole current book | as above | same figure | TEMPORAL_RESOLUTION | YES |
| 6 | D04 | balance at `2026-02-30` (a date that cannot exist) | whole current book | refusal | same figure | TEMPORAL_RESOLUTION | YES |
| 7 | D05 | balance at `1999-12-31` | whole current book | refusal | same figure | TEMPORAL_RESOLUTION | YES |
| 8 | D06 | balance at `2099-12-31` | whole current book | refusal | same figure | TEMPORAL_RESOLUTION | YES |
| 9 | D08 | two different stated periods | one figure for both | different figures, or a refusal | both `115,450,800.7` | TEMPORAL_RESOLUTION | YES |
| 10 | E06 | a receipt evidencing the period | no temporal key under any spelling | the period | `<ABSENT>` | — | RECEIPT_GOVERNANCE | YES |
| 11 | E01 | a receipt evidencing the dataset | `dataset` key present, value `None` | `'funded'` | `None` | — | RECEIPT_GOVERNANCE | YES |

Independent truth source for all eleven: longhand inspection of the plan object
against the compiled `MIQuerySpec`, plus `portfolio_truth_oracle.total` for the
figure the temporal cases return. All eleven are **silent** — no refusal, no
warning, and a number that looks entirely plausible. D05 is the sharpest: a
question about 1999 returns today's book with nothing anywhere saying so.

## What is solid

Not one numerical error in 65 cases. Specifically verified against independent
truth: sum, count, simple mean, weighted mean (and that weighted ≠ simple, so the
weight is genuinely used), median, min, max, one- and two-dimensional grids cell
by cell, grids reconciling to their population total, the volunteered
`concentration_pct` share column summing to 100 and matching longhand shares, and
null measures excluded rather than zero-filled — A17 checks the zero-filled
alternative explicitly and the engine does not take it.

Filters: one, two and three conjunctive predicates; a predicate true of every row
(applied *and* recorded); one matching exactly one row; one matching none; one
over a null-bearing measure. A predicate on a column the book lacks refuses
rather than being dropped.

Governance that works: `applied_predicates`, `input_row_count`,
`filtered_row_count`, `group_field_keys`, `aggregation` and the measure column are
all present and correct. Unresolved grouping values are excluded **with a
warning** naming the field and the row count — which is why C05 passes and why the
reconnaissance-era suspicion of a silent geography drop does not survive.

The balance bridge is correct on mixed churn, full redemption, an all-new book,
an identical-snapshot zero case, and a 27-loan case where the components
recompose to the closing balance with residual 0.0.

`borrowing_base.calculate(book, None)` refuses rather than inventing a facility.

## Not reachable, and one unjudgeable

`NOT_PLAN_LEVEL_REACHABLE = 5`. The governed plan's operation vocabulary is
exactly `{avg, count, max, median, min, sum, weighted_avg}`. Ranking,
distribution, portfolio summary, movement/bridge and concentration are **not
operations of the plan contract**, so they cannot be driven from this boundary.
They are reported as not reachable and counted as neither pass nor fail. No
adapter was invented to make them testable. (The bridge *is* exercised, via its
own direct capability entry point, and the share column shows concentration
arithmetic reaching grouped results — but neither is expressible as a plan
operation.)

`F07-borrowing-base-configured` is **UNJUDGEABLE**. The capability returns a
result for a hand-built `FacilityConfiguration`, but no independent facility
fixture exists to verify eligibility and advance against, and authoring expected
eligibility myself would be the product marking its own homework. Owner recorded
as DATA_CONFIG: what is missing is a governed fixture, not a fix.

## Reconciliation against the deterministic branch — read-only

```
DETERMINISTIC_BRANCH      = origin/claude/borrowing-base-mi-query-cvplz0
DETERMINISTIC_BRANCH_HEAD = 101ac7aa927760600bc926f84a3a6ee35a9008a8
```

| defect | semantic owner | addressed? | commit(s) | same execution layer? | evidence |
|---|---|---|---|---|---|
| Scope binding (dataset, lens, period dropped) — B13, D07, B12 | PLAN_BINDING | **NO** | none | n/a | `query_plan_compiler.py` is **UNTOUCHED** on the branch |
| Stated period ignored — D02–D06, D08 | TEMPORAL_RESOLUTION | **NO** | `05dbe95f` is the nearest | **NO** | that commit changes `mi_agent_api/temporal_compare.py`, `movement_summary.py`, `question_interpretation/*` — which period a *question* means, never whether the executor honours one a *plan* states |
| Receipt has no period — E06 | RECEIPT_GOVERNANCE | **NO** | `2ef7fb3e` is the only `execution_receipt.py` change | NO | it adds no period/reporting_date/as_of line; grep of its added lines returns none |
| Receipt has no dataset — E01 | RECEIPT_GOVERNANCE | **NO** | none | n/a | — |

All four of `query_plan_compiler.py`, `query_plan_execution.py`, `query_plan.py`
and `mi_query_executor.py` are **UNTOUCHED** on that branch, and
`query_plan_adapter.py` — which the branch does modify — gains **0** added lines
mentioning period, reporting_date, dataset or lens.

```
EXISTING_DETERMINISTIC_BRANCH_ADDRESSES = 0/4 confirmed defects
PARTIALLY_ADDRESSES                     = 0
DOES_NOT_ADDRESS                        = 4
```

An upstream `question_interpretation` fix is not a plan-level execution fix, and
this is the clearest case of that distinction in the programme so far.

## Verdict

```
DETERMINISTIC_BASELINE_ESTABLISHED = YES

CORE_ARITHMETIC_ASSESSMENT   = SOUND. 0 numerical errors in 65 cases across sum,
                               count, means, weighted means, median, min, max,
                               grids, shares, nulls and the balance bridge, every
                               figure against independent truth.
SEMANTIC_EXECUTION_ASSESSMENT = DEFECTIVE ON SCOPE. The engine computes the right
                               number over the wrong population whenever the plan
                               names a period, a dataset or a lens, because the
                               plan->spec compiler binds only filters.
SILENT_SEMANTIC_ERRORS        = 11, all one root cause, none detectable from the
                               answer or the receipt.

CONFIRMED_DEFECTS_TO_FIX = 1 root cause, 4 reportable defects:
    DEF-1 PLAN_BINDING      AnalyticalScope.period -> MIQuerySpec            (D07, D02-D06, D08)
    DEF-2 PLAN_BINDING      AnalyticalScope.dataset -> MIQuerySpec           (E01)
    DEF-3 PLAN_BINDING      AnalyticalScope.portfolio_lens -> MIQuerySpec    (B12)
    DEF-4 RECEIPT_GOVERNANCE the receipt must evidence period and dataset,
                             and execution must refuse a period the data
                             cannot honour rather than answering as current   (E06, E01)

UNMEASURED_CAPABILITIES = ranking, distribution, portfolio summary, concentration
                          (not plan-level expressible); borrowing-base eligibility
                          (no independent fixture); multi-period / snapshot
                          execution (untestable until DEF-1 lands — there is no
                          way to ask for a second period today)
```

**RECOMMENDED_FIX_ORDER**

1. **DEF-1.** Carry `scope.period` into `MIQuerySpec` in
   `query_plan_compiler.py`. Nothing temporal can be tested or trusted until a
   stated period reaches the executor — today a multi-period request is
   unexpressible, so this blocks the other temporal work rather than merely
   preceding it.
2. **DEF-4, refusal half.** With the period arriving, make a period the dataset
   cannot honour a governed unavailable. This is the safety-critical half: the
   current behaviour answers 1999 with today's book.
3. **DEF-4, receipt half, plus DEF-2.** Put period and dataset on the receipt.
   Cheap once the values exist, and it is what makes the rest auditable.
4. **DEF-3.** The lens, for completeness of the same contract.

Do 1 and 2 together: landing 1 alone turns a silently-wrong answer into a
differently-wrong one, because the executor would then see a period it has no
policy for.

**RECOMMENDED_NEXT_STEP.** Fix DEF-1 and DEF-2 in `query_plan_compiler.py` with
the bank re-run as the gate — it is deterministic and reproduced byte-identically
across two worktrees, so a diff in its result is attributable. Do not merge the
deterministic branch for these: it does not touch this layer, so it would neither
fix nor disturb them, and it can be assessed separately on its own merits. Before
the fix, consider adding a multi-period fixture to the oracle so the temporal
surface has somewhere to land — six of the eleven failures currently cannot be
tested positively, only negatively.
