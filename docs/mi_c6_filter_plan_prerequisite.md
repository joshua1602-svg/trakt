# C6 filter wiring — `SELECT_POPULATION(kind=row_predicates)`

Base `0417f68`. **Filter plan prerequisite only.** `evolution` is not converted
to the compositional path, Funnel/Stage raw ownership is untouched, no C6
thresholds are set, `period_change` is not migrated, T3–T7 are not enabled.

---

## 1. Regression gate (§1) — PASSED

Full suite, same tree, before and after the predicate-parity change:

```
before (Steps 1–2) : 203 failed, 10320 passed, 36 skipped, 16 xfailed, 28 errors
after  (parity)    : 203 failed, 10350 passed, 36 skipped, 16 xfailed, 28 errors

failing-name sets  : 231 vs 231
INTRODUCED         : 0
FIXED              : 0
silent drops       : 0   (skips, xfails and errors all identical)
```

The +30 passes are exactly the 30 tests the parity task added. Parity re-confirmed
at **119/119**, contract at **121/121**, immediately before wiring began.

---

## 2. No remaining semantic work (§2) — proved before implementing

The proof was run **read-only, against the un-wired tree**, so it could not be
made true by the wiring it was meant to justify:

```
question : balance trend where LTV above 50%
spec.filters : {'current_loan_to_value': {'op': 'gt', 'value': 50.0}}
claims       : [('current_loan_to_value', 'gt', 50.0)]

period    rows(spec)  rows(claims)  ids identical        bal(spec)      bal(claims)
2026-04         1721          1721           True   432,425,355.79   432,425,355.79
2026-05         1799          1799           True   450,969,362.11   450,969,362.11
2026-06         1889          1889           True   472,527,483.38   472,527,483.38

PER-PERIOD IDENTITY: YES
corpus: 121 spec.filters entries, 121 material predicates, 0 non-material
```

So the target flow needs **no** new field resolver, normalisation rule, operator
vocabulary, categorical vocabulary, filter registry, or raw-question read. The
field was bound once by `_filter_field_of` upstream; the meaning is owned once by
`governed_predicate_mask`. What was missing was only the carriage between them.

---

## 3. The second mode (§3)

`select_population` now has three named modes, where it had three string
literals:

```python
KIND_SOURCE_PORTFOLIO_LENS = "source_portfolio_lens"   # narrow by IDENTITY
KIND_ROW_PREDICATES        = "row_predicates"          # narrow by VALUE
KIND_WHOLE_DATASET         = "whole_dataset"
```

They are separate structures, not one filter bag, and `lens_filters` was not
overloaded. The reason is a measured one: the lens narrows by governed portfolio
identity, which the registry decides — Phase 1C measured the two readings
diverging at £300 against £1,200 on a book with two portfolios of one type.
Putting value predicates into `lens_filters` would move identity back into the
value channel, which is the P1I-A ruling in reverse.

**A reader had to be fixed to make room for the second mode.** `lens_label` took
the *first* `select_population` step. That was correct while a plan could only
carry one; the moment it can carry two, "the first one" is no longer the lens.
It is now kind-aware, with a control that fails if it regresses.

---

## 4. Built from the contract, and only from the contract (§4)

`analytical_plan.row_predicate_step(interpretation)` reads
`interpretation.row_predicates` and nothing else. `spec`, the question text and
provenance strings are not in scope — there is no English within reach, so a
route planning from this cannot re-derive a filter's meaning even by accident.

`row_predicates(plan)` returns the executor's own `Predicate` objects rather
than dicts, because the one thing every caller must not do is re-interpret them.

In the route, `_filtered_funded_evo` **no longer receives the spec at all**. Its
signature takes `predicates`; it executes them through `apply_population`, which
since the parity work runs `governed_predicate_mask` — the same owner
`_apply_filters` uses. That equivalence is why the substitution is row-for-row
identical rather than merely similar.

The answer prose and the population ledger are also described from the plan:
the prose through `Predicate.describe()`, the ledger from the fields the plan
applied. Prose, evidence and receipt now cannot say three different things about
one narrowing.

Preserved exactly, and asserted: bare `"over 50"` still binds to
`current_outstanding_balance`; a repeated field still collapses to the last
predicate because `spec.filters` is a dict; an unappliable predicate still
defers to the controlled point-in-time path.

---

## 5. Old vs new (§5, §6)

Filtered — the delivered case, compared field by field:

```
balance trend where LTV above 50%   IDENTICAL
   £432,425,355.79 → £450,969,362.11 → £472,527,483.38
   route evolution · 1,889 rows in the final period
   ledger: current_loan_to_value (applied within each period), unavailable []
```

Non-vacuity is asserted, not assumed: the filtered series must be strictly below
the unfiltered one in every period, or a filter that silently did nothing would
satisfy every other assertion.

Unfiltered — funded, count, pipeline and funnel evolution all unchanged, and
the 882-question census below covers stage evolution too.

---

## 6. Blast (§9) — 882 questions, zero movement

Diffed against the snapshot taken **before the predicate-parity change**, so the
figure covers parity and wiring together:

```
interpretation 0 · dataset 0 · route 0 · predicates 0 · answer 0 · population 0
route mix identical, evolution = 28
```

---

## 7. Mutation controls (§8)

| mutation | result |
|---|---|
| drop one `RowPredicateClaim` before it becomes a predicate | 6 of 19 fail |
| bind the predicate to a different field | 6 of 19 fail |
| alter the predicate value | 4 of 19 fail |
| route re-reads `spec.filters` — a second owner | 5 of 19 fail |
| unappliable predicate | fails closed; the route defers, control asserts the raise |

The architectural guard needed rewriting before it was worth anything. Its first
cut checked `"spec.filters" not in source` and failed against **its own
docstring**, which explains what it replaced — the third time a substring guard
in this programme has flagged a mention rather than a use. It now parses the
function with `ast` and asserts there is no *call* to `_apply_filters` and no
*attribute read* of anything on `spec`.

---

## 8. Duplicate filter ownership (§7) — partly consolidated, and stated plainly

`chat_routing` no longer calls `_apply_filters` anywhere; the import is gone.
For **meaning**, the evolution filter has one owner: the contract, executed by
`governed_predicate_mask`.

One read of `spec.filters` remains, and it is not filter interpretation:

```python
requested = dict(getattr(spec, "filters", None) or {})
filtered = bool(predicates)
if requested and len(predicates) != len(requested):
    return None
```

This is a **presence-and-coverage gate**. The keys the contract excludes by
design are scope and reporting-basis keys, which are not row predicates at all;
if any remain, this route cannot honour the whole request and defers to the
point-in-time path — the same fail-safe an invalid filter already took. Measured
on the corpus: 121 entries, 121 material predicates, 0 excluded, so it defers on
nothing that ships today.

**This is not full consolidation and is not reported as such.** Removing the
gate needs the scope channel to reach this route as a plan step, which is
`evolution`'s own conversion and is explicitly out of scope here.

---

## 9. Cost (§10) — this task only

Raw added + raw deleted production diff lines.

| bucket | file | raw lines |
|---|---|---|
| shared wiring | `analytical_plan.py` | 73 |
| route-specific wiring | `chat_routing.py` | 88 |
| cleanup | (the dead `_apply_filters` import, counted in the 88) | 2 |
| **production total** | | **161** |
| tests | `test_c6_filter_plan.py` | 267 (new) |

The predicate-parity work's **401 raw production lines are NOT included** in this
task's cost. They are retained in the migration chronology as **C6 discovery
burden**: C6 surfaced a latent inconsistency in a shared semantic owner that four
production consumers already depended on and that predates the compositional
plan.

---

## 10. Verdict

# C6 FILTER PLAN PREREQUISITE CLOSED

`RowPredicateClaim` drives `SELECT_POPULATION(kind=row_predicates)`; filtered
populations and economics are identical to the penny; the compositional path
needs no route-level filter interpretation; fail-closed behaviour is intact.
