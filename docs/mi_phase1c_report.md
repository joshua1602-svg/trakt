# Phase 1C — report

# STOP — PRODUCTION SOURCE-SCOPE SEMANTICS ARE INCONSISTENT

**The interpretation contract was not extended.** No production module was
changed. `portfolio_summary` remains unconverted.

§2 of this task says: *"If production behaviour itself is ambiguous or
inconsistent, stop and report before changing the contract."* It is, in a way
that decides what the contract must say — so the contract change is held.

Commit: `3697d94` (truth table + 8 measured-behaviour tests).

---

## The blocking finding

§3 asked for **Case D — show that production does not silently widen to Total.**

**It does.**

```
"Summarise the acquired_001 book"
  ok=True   route=portfolio_summary   lensApplied=True   facets=[]   warnings=[]
  "At 30 June 2026 the portfolio (acquired_001) holds 11,035 loans
   with a funded balance of £1.96bn."
```

`acquired_001` is not in the governed registry. `resolve_scope` falls back to
Total, correctly recording `fell_back_to_total=True` and
`requested_context_id='acquired_001'`. Then `_resolve_lens` keeps the lens
**name and label** while taking the **empty filters**:

```python
return PortfolioLens(name=lens.name, label=lens.label,
                     filters=dict(scope.filters), cohort_id=lens.cohort_id)
```

So the answer carries the requested scope's label against whole-book figures.
**`lensApplied=True` is a false claim**, no facet is raised, no warning is
emitted, and the `fell_back_to_total` flag the scope object holds never reaches
the reader.

This is the P0 defect class verbatim — *a materially different question answered
silently* — and it is the exact behaviour §5 calls non-negotiable:

> *"An unresolved state must never be interpreted as Total."*

**Production interprets it as Total.**

### Why this blocks the contract change rather than sitting beside it

The contract must represent *unresolved* distinctly from *Total* (§5). But §9
requires the shadow to reproduce production. Those are now in direct conflict:

* encode the safe semantics → the shadow diverges from production on this case;
* encode production faithfully → the contract carries a defect forward, and the eventual conversion ships it with a compositional blessing.

Which one is correct is **a decision about product behaviour**, not an
implementation detail, and it is not mine to take inside a contract-completeness
task. Fixing it is also explicitly out of scope here: it changes a user-visible
answer and needs its own authorisation and measurement, exactly like the arity
disclosure defect.

### Reachability

The registry holds `alp_origination`, `alp_acquired`, `spv1_sponsored`. Any
cohort-shaped identifier the lens recognises but the registry does not hold —
a decommissioned book, a renamed SPV, a typo, an id from another tenant's
vocabulary — takes this path. `_is_portfolio_summary` claims such questions, so
the route owns them.

Pinned as measured behaviour in
`tests/test_source_scope_production_semantics.py::TestProductionPrecedence`
(`test_case_d_an_unresolvable_scope_widens_to_the_whole_book`,
`test_case_d_the_widening_is_not_disclosed_to_the_reader`) — **passing**, because
they assert what production does, not what it should do. If someone fixes the
defect, those two tests fail and say why.

---

## 1. Production source-scope truth table

`python -m migration_phase0.source_scope_truth_table` — 28 rows, 7 phrasings ×
4 caller defaults, executed end to end. Full record in
`migration_phase0/SOURCE_SCOPE_TRUTH_TABLE.json`.

The chain, traced from source rather than named:

```
1 DETECT      portfolio_lens.resolve_lens(question)              NL recognition
2 DEFAULT     lens_from_selection(req.source_portfolio_lens)     the dropdown
3 PRECEDENCE  resolve_lens_with_default(question, default):
                  if mentions_portfolio(question):  QUESTION wins
                  else:                             DEFAULT  wins
4 GOVERN      context_id(lens) -> resolve_context(...).scope     semantic -> ids
5 APPLY       scope.filters -> {source_portfolio_id: [...]}      -> _scope_frame_lens
```

| question | default | mentions | q-scope | effective | won by | governed ids | rows | balance |
|---|---|---|---|---|---|---|---:|---:|
| silent | — | False | total | total | neither | all 3 | 11,035 | 1,964,886,258.21 |
| silent | acquired | False | total | **acquired** | default | `[alp_acquired]` | 3,909 | 579,377,675.23 |
| silent | direct | False | total | **direct** | default | `[alp_origination]` | 7,126 | 1,385,508,582.98 |
| explicit direct | acquired | True | direct | **direct** | question | `[alp_origination]` | 7,126 | 1,385,508,582.98 |
| explicit acquired | direct | True | acquired | **acquired** | question | `[alp_acquired]` | 3,909 | 579,377,675.23 |
| **explicit total** | acquired | True | total | **total** | question | all 3 | 11,035 | 1,964,886,258.21 |
| **disclaimed** | acquired | True | total | **total** | question | all 3 | 11,035 | 1,964,886,258.21 |
| **explicit cohort** | any | True | cohort | cohort | question | **all 3 (fell back)** | **11,035** | **1,964,886,258.21** |

Branches exercised: **question wins 20, default wins 6, neither 2.**

The last row is the defect. The two bold `total` rows are the Phase 1B blocker.

---

## 2. Phase 1B blocker — proved by enumeration

**6 divergences** where a contract carrying only the resolved scope cannot
reproduce production:

```
'portfolio summary across all portfolios'            + default=acquired|direct|alp_acquired
'summarise the portfolio excluding the acquired book' + default=acquired|direct|alp_acquired
```

In each, the question **mentions** a portfolio and **resolves to `total`**, so
the question wins and the dropdown is ignored. `source_scope` reports
`filled / total / narrows=False` — **identical to Case A**, where `total` means
"silent" and the dropdown *must* win.

Case A and Case B2 produce the same claim and require opposite behaviour. That
is the blocker, and it is why provenance — not more scope values — is what the
contract needs.

---

## 3. Contract extension — NOT MADE

Held pending a ruling on the Case D defect (above). The design was settled
before stopping, and is recorded so the next task starts from it rather than
re-deriving it:

* extend the **existing** `SourceScopeClaim` rather than adding a parallel claim family;
* add a provenance value distinguishing **stated by the question** / **supplied by the caller** / **no narrowing stated** / **unresolved**;
* `state` continues to separate `EMPTY` (owner not consulted) and `UNRESOLVABLE` from any `FILLED` scope, so §5's contract-level requirement is already met by Phase 1A's shape;
* `mi_agent.portfolio_lens` stays the single owner — the projection carries `mentions_portfolio` alongside `resolve_lens`, both from the owner, and no phrase list or planner-side text check is introduced.

**What is not settled, and is the ruling needed:** what the contract should say
when the governed registry falls back to Total for a scope the question stated.
Encoding it as `total` reproduces production and launders the defect. Encoding
it as `unresolved` is correct and diverges from production.

---

## 4. Governed portfolio resolution

**Old (Phase 1A shadow):** `resolve_lens(question).filters` →
`{'source_portfolio_type': 'acquired'}` → dataframe filter.

**Production:** `context_id(lens)` → `resolve_context(...).scope.filters` →
`{'source_portfolio_id': ['alp_acquired']}` → dataframe filter.

Measured on the shipped book, they select **identical rows** — because it holds
exactly one portfolio per type. That coincidence is the whole reason Phase 1A's
numbers matched while proving nothing, and it is now pinned as a test
(`test_on_the_shipped_book_both_paths_coincide`) so nobody re-derives comfort
from it.

---

## 5. Multi-portfolio discrimination test — built and passing

`test_the_registry_not_the_data_decides_group_membership`, on a **focused unit
fixture** rather than the governed book (§7's preference).

Two portfolios both carrying `source_portfolio_type='acquired'` **in the data**;
only one typed `acquired` **in the registry** — the same shape as
`spv1_sponsored`, which is present on the shipped book and typed `None`, so it
belongs to no group.

| path | selects | balance |
|---|---|---|
| governed `{'source_portfolio_id': ['acq_one']}` | 2 rows | **300.00** |
| raw `{'source_portfolio_type': 'acquired'}` | 3 rows | **1200.00** |

**A compositional path filtering on the raw lens would answer for a book the
registry does not place in the group**, and no economic check on the shipped
book could catch it. `resolve_scope`'s own docstring states the rule this
enforces: *"Group membership is computed from the registry every time… `direct`
is whatever is currently typed `direct`."*

---

## 6. Precedence tests

Five, all passing, all measuring production:

| case | assertion |
|---|---|
| A — no explicit scope, default present | default wins (`acquired`) |
| B — explicit scope vs different default | question wins (`direct`) |
| B2 — explicit *whole-book* vs default | question wins (`total`) — **and the same claim as A** |
| C — neither | `total`, no filters |
| D — unresolvable | **widens to the whole book, undisclosed** |

---

## 7–8. Equivalence and regression — not reached

No change was made, so there is nothing to compare and no gate to move. Running
them would produce a result that means nothing. The Phase 1B conversion baseline
(`migration_phase0/CONVERSION_BASELINE.json`, 11 cases, stable at 0 differences)
stands unchanged and remains the yardstick.

---

## 9. Final status

# STOP — PRODUCTION SEMANTICS INCONSISTENT

Not "contract still incomplete" — the missing contract fact is known, designed
and small. The blocker is upstream of it: **production's own source-scope
semantics contain a silent widening that the contract would otherwise be asked
to encode as correct.**

**What is needed to proceed:** a ruling on the Case D defect.

1. **Fix it first** — an unresolvable governed scope refuses, or discloses, rather than answering the whole book under the requested label. A user-visible governance correction, separately authorised and measured, in the same class as the arity-2 disclosure defect. Then the contract encodes `unresolved` honestly and the shadow reproduces it.
2. **Encode production as-is** — the contract carries `total` for a fallback, the conversion reproduces the defect exactly, and the defect is recorded as known-open with the conversion explicitly not making it worse.

**Option 1 is the one consistent with §5's non-negotiable**, but it is a product
decision and it is not mine to take here.

---

## 10. Measured effort

| | |
|---|---|
| production lines changed | **0** |
| tests added | 8 (all passing; they pin production, not an aspiration) |
| instruments added | 1 (truth table, 28 rows) |
| commits | 2 |
| baselines updated | 0 |

### Dependencies discovered

1. **`_resolve_lens` discards `fell_back_to_total`.** The governed scope knows it widened; the lens handed to the route does not carry it. Any conversion needs that signal, and it exists one call upstream.
2. **The registry holds a portfolio the type groups do not** (`spv1_sponsored`, type `None`). So "the acquired book" and "everything acquired-ish in the data" are already different populations on the shipped book — the discrimination case is not hypothetical.
3. **`lensApplied` is computed without checking whether the lens resolved.** It is a route-level claim about a governed fact, which is the same route-identity-versus-execution-evidence pattern the study found in the receipt layer.

### Does this change estimated migration cost?

**Yes, and this is the fourth instance of one pattern.** `lens_from_selection`
defaults to Total; `resolve_lens` returns provenance type where the route uses
portfolio ids; `resolve_lens` omits the precedence bit; and now `resolve_scope`
silently widens while the route drops the flag that says so.

Every one is the same shape: **a governed concept has a strict path and a
convenient path, production uses a mixture, and the mixture is only visible by
executing it.** The scoping study costed conversions as *re-expressing a route
in primitives*. The measured work is **auditing each governed concept's real
resolution chain end to end before it can be represented at all** — and on this
route, that audit found a shipped defect rather than just a gap.

I would now expect route conversions to be dominated by that audit, not by the
composition, and the first three conversions will not give a stable median until
each has been through it.
