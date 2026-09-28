# Phase A — the gate, hardened before any product code moves

Test and evidence only. No product file changed.

```
PHASE_A_SHA          = (this commit)
BANK_SIZE            = 65 (frozen v2) + 28 (new scope bank) = 93 plan-level cases
                       + 22 independent arithmetic oracle cases
CURRENT_SCOPE_FAILURES = 11 (v2) + 24 (scope bank) = 35
ORACLE_INDEPENDENCE  = PASS
```

## Why the scope bank had to exist

Baseline v2 could only prove the scope defect NEGATIVELY. Its book carried no
period, no portfolio lens and no dataset identity, so it could show that a stated
period was ignored and never that one was honoured. Phase B cannot be gated on
that: "the figure changed" is not evidence the figure is right.

`scope_oracle.py` supplies the positive fixtures. It imports **nothing** from the
product — `pandas`, `numpy`, `itertools`, `typing` and explicit column names, as
an AST check of its imports confirms. Its book carries two governed registry
fields on every row:

* `reporting_date` — three periods (`2026-03-31`, `2026-05-31`, `2026-06-30`)
* `source_portfolio_type` — `Direct` / `Acquired`

Both are real fields in `mi_agent/mi_semantics_field_registry.yaml`, so a scope
request against them is expressible in the existing contract rather than needing
a new one.

Row counts differ per period on purpose — a book grows and redeems — so a
period-blind sum is wrong by an amount no rounding could explain:

| figure | £ |
|---|---|
| all three periods, both lenses | 136,132,618.88 |
| 2026-03-31 | 24,851,104.67 |
| 2026-05-31 | 41,321,206.13 |
| 2026-06-30 | 69,960,308.08 |
| Direct (all periods) | 109,449,739.44 |
| Acquired (all periods) | 26,682,879.44 |
| 2026-03-31 × Acquired | 5,129,308.35 |
| 2026-06-30 × Direct | 56,243,599.21 |
| dataset `pipeline` | 85,445,021.33 |

## Accidental equality is the enemy, and the guard caught two of mine

`assert_materially_distinct()` runs at import and refuses to let the fixture load
if any two figures the bank compares are within £1m of each other. A fixture whose
March total happened to equal its current-Acquired total would let a scope-blind
engine pass, and "the period was ignored" would be indistinguishable from "the
period was honoured".

It earned its place twice while this fixture was being built:

1. The first shape put the March total **£332,722 from** the current-Acquired
   total. Rejected; the row/scale shape was rebuilt so all fourteen figures sit
   at least £3m apart.
2. The first version of the guard compared the two datasets only against each
   other, and `pipeline` landed **£83k from** the Acquired lens total. The guard
   now compares every pair, datasets included, and `pipeline` was moved to ~£85m.

One identity is exempt and named in the source: `dataset_funded` *is* the scoped
book, so their equality is the fixture being consistent, not a coincidence that
could mask anything — no case distinguishes those two figures.

## Current product still exhibits the defect

| bank | result |
|---|---|
| 22-case independent arithmetic oracle | **22 passed**, 18 subtests |
| frozen v2 bank (65 cases) | 47 PASS / 11 FAIL / 1 outside scope / 1 unjudgeable / 5 not reachable — **identical to `7d199138`, 65/65 cases byte-equal** |
| new scope bank (28 cases) | **4 PASS / 24 FAIL**, 12 silent |

Scope bank by dimension:

| dimension | pass | fail |
|---|---|---|
| PERIOD | 0 | 10 |
| DATASET | 2 | 2 |
| LENS | 0 | 6 |
| COMPOSITION | 2 | 6 |

Every period case returns `136,132,618.88` — the sum of all three periods —
whatever period was asked for. Both lens cases return the same figure, so Direct
and Acquired are indistinguishable. Nothing in any receipt names a period, a lens
or a dataset.

The four passes are honest but weak, and are labelled as such in the bank: they
are the dataset cases, and they pass because `execute_query_plan(plan, data, ...)`
takes the frame from its CALLER. The declared `dataset` is not doing any work — it
is simply not contradicted. That is Phase C's first question, not a Phase A
result.

## What the architecture investigation found, for Phase B

The period semantic owner **already exists** and is not the flat executor:

* `mi_runtime.run_mi_query` routes by `infer_execution_mode(spec)` across
  flat / state / temporal / risk.
* `_run_temporal` resolves `SnapshotSelector.as_of(client, spec.as_of_date)`.
* Governed refusal codes already exist: `SNAPSHOT_STORE_REQUIRED`,
  `TEMPORAL_SELECTOR_INCOMPLETE`, `MISSING_SNAPSHOT_CLIENT_ID`,
  `FALLBACK_TO_FLAT_EXECUTOR`.
* `mi_query_executor.py` contains **no** `reporting_date`, `as_of_date`,
  `temporal_mode` or `execution_mode` handling — the flat path is period-blind by
  design, and `execute_query_plan` calls `execute_mi_query` directly, bypassing
  the router.

The lens owner already exists too: `mi_agent/portfolio_lens.py` with
`apply_lens(spec, lens)` — language-free, keyed on `source_portfolio_type`, with
`LENS_TOTAL` / `LENS_DIRECT` / `LENS_ACQUIRED`.

So none of Phases B–D should need a new semantic owner. What is missing is the
binding at the governed-plan boundary, which is where the fix belongs.

## Files this commit adds

```
due_diligence/evidence/deterministic_baseline_v2/scope_oracle.py
due_diligence/evidence/deterministic_baseline_v2/scope_bank.py
due_diligence/evidence/deterministic_baseline_v2/scope_bank_result.json
due_diligence/evidence/deterministic_baseline_v2/PHASE_A.md
```

Product files changed: **none**.
