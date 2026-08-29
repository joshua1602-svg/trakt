# Phase 1B — report

# STOP — CONTRACT PREREQUISITE

**`portfolio_summary` was not converted.** No production module was changed. The
conversion stopped at a property the interpretation contract cannot represent,
and the only two ways past it were both forbidden by this task's constraints.

Commits: `9011d61` (conversion baseline), `60c5bd6` (blocker, pre-registered).

---

## The blocker

`portfolio_lens.resolve_lens_with_default` decides precedence between the
question and a caller-supplied default like this:

```python
if mentions_portfolio(text):   return resolve_lens(text)    # the question wins
return default or total_lens()                              # the dropdown wins
```

**The contract carries *which* scope the owner resolved. It does not carry
*whether the question named one*** — and that second fact is what decides
precedence.

Two questions the route **owns** mention a portfolio and resolve to `total`:

* *"portfolio summary across all portfolios"*
* *"summarise the portfolio excluding the acquired book"*

For those, `source_scope` is `filled / total / narrows=False` — **identical to
what a question that says nothing about source scope produces**.

### Verified end to end, not inferred

`execute_governed_mi_query(question, source_portfolio_lens="acquired")`:

| | shipped today | a contract-only plan |
|---|---|---|
| *portfolio summary across all portfolios* | **whole book** — £1,964,886,258.21 / 11,035 loans | Acquired — £579,377,675.23 / 3,909 |

The shipped route honours the question's explicit whole-book reading over the
dropdown. A plan reading only `source_scope` cannot see that the question spoke
to scope at all, falls back to the dropdown, and **narrows**.

That is a silent population narrowing on an owned question — the defect class
this programme exists to remove.

### Why it is not workaround-able within this task

Exactly two ways to preserve the shipped behaviour:

1. **Re-read the raw question in the plan path** — forbidden by the conversion constraints, and the defect the whole programme has been removing.
2. **Carry one more fact from the same owner** — an interpretation-contract change, listed under "Do not" for this task.

Neither is available, so it is reported rather than taken.

### Reachable, not theoretical

`source_portfolio_lens` is a live field on `MiQueryRequest`
(`mi_service.py:100`), populated by `app.py:1785` from the workspace UI
dropdown. A user with "Acquired" selected who asks across all portfolios hits
this.

### It is a distinct fact, not a defect in Phase 1A

Phase 1A carried **which scope**. This needs **whether the question spoke to
scope**. Neither implies the other. Phase 1A is not wrong; it is incomplete for
this purpose, and could not have known — the gap only appears when a caller
supplies a default.

### The nine frozen cases do not diverge — which is the point

A1–A6 name no portfolio; L1–L3 name one and resolve to it. **A conversion
validated only on the frozen surface would have passed every gate and shipped
the defect.** The surface was found by asking which questions the recogniser
owns, not which ones were convenient to test.

---

## 1. Conversion baseline — frozen (`migration_phase0/CONVERSION_BASELINE.json`)

The exact production answer for all 11 cases (9 owned + X1/X2 deliberately not
owned), captured before any change: `ok`, `route`, answer text, payload keys,
every artifact, reconciliation, source notes, warnings, the full
`executionSummary`, facets as `(kind, label, field, status)`, `notApplied`,
metadata keys, `portfolioScope`, `lensApplied`, `engine`.

**Stability proven, not assumed:** re-capture against the frozen file gives **0
differences across 11 cases**. Two corrections were needed to get there, both of
which would have made the baseline useless:

* `route` is on `metadata`, not top-level, and `executionSummary` is top-level, not under metadata. The first draft recorded `route: None` for all 11 — a baseline that cannot detect a change of route ownership.
* the point-in-time path renders a fresh `kpi_<8 hex>` per run. Stripped **by value shape, not key name** — dropping every `id` would hide a real difference.

Registered surfaces at the same commit are unchanged from Phase 1A: calibration
267; robustness 32/6/4/2 with Q1 4 · Q7 4 · Q8 12; shipped shapes 0 wrong;
routed 31 + `rt_004`; recognition 15/7/10/29; **silent drops 0**;
`question_interpretation/tests` 566 + 1 pre-existing.

---

## 2. The switch point — traced

| # | | |
|---|---|---|
| 1 | recognised | `chat_routing.py:2851` registry entry → `_is_portfolio_summary` (`:320`) |
| 2 | execution begins | `_route_portfolio_summary` (`:433`) → `movement_summary.portfolio_summary(...)` |
| 3 | where the plan replaces it | **that one call**. Everything after — prose, KPI/chart/table artifacts, envelope — is shape, not economics |
| 4 | result → envelope | `_envelope(..., route="portfolio_summary")` at the end of the handler |
| 5 | receipt/facets | **downstream, in `mi_service.py:644–674`** — `reconcile_routed_facets`, `build_routed_receipt`. The route returns an envelope; governance is applied to it afterwards |
| 6 | fall-through | `return None  # defer to the existing point-in-time summary path` (`:442`) |

The switch is genuinely small — one call — which is why the blocker sits in the
**input** to that call, not in the call itself.

---

## A second finding: Phase 1A's equivalence was measured against the wrong side

Recording this because it qualifies a claim I made:

```
'Summarise the acquired book'
   raw  resolve_lens      filters={'source_portfolio_type': 'acquired'}
   routed _resolve_lens   filters={'source_portfolio_id': ['alp_acquired']}
```

`_resolve_lens` additionally resolves the lens **through the governed portfolio
registry** (`portfolio_context.resolve_context`) into an explicit portfolio-id
list. The Phase 1A harness used the **raw** lens on both sides, so it compared
raw-against-raw and never touched what the shipped route actually filters on.

The economics matched because this fixture has **exactly one book per
provenance type**, so `type == 'acquired'` and `id in ['alp_acquired']` select
the same rows. **On a book with two acquired vehicles they would not.**

So Phase 1A's "economics identical" was true of the engine, not of the route.
The Phase 1B conversion baseline — captured through `execute_governed_mi_query`
— is the first measurement against the real thing, and it is why freezing it
before converting was worth the step.

**Consequence for the migration:** a converted plan must reconstruct the lens
through the registry step, not just the raw lens. That is the *second* instance
of the same lesson as Phase 1A's `lens_from_selection` finding — **the owner's
convenience paths are not the route's paths**, and every governed concept a plan
reconstructs needs checking against what the route actually used.

---

## 3. What was NOT done

Per the constraints, and stated so the boundary is auditable:

* no production module changed;
* no new primitive, no phrase list, no route-specific economic branch;
* rank residual policy untouched; arity disclosure untouched; filter-clause join untouched;
* no recogniser changed, no interpretation semantics changed;
* no capability enabled — T3–T7 remain closed;
* no duplicate parser/recogniser/executor decision removed.

**Receipt/envelope equivalence (§5) and answer-text handling (§6) were not
reached.** They are downstream of a switch that did not happen, and running them
against an unconverted path would produce a result that means nothing.

---

## 4. Measured effort

| | |
|---|---|
| production lines changed | **0** |
| test files touched | 1 (2 tests added, both `xfail(strict=True)`) |
| conversion commits | **0** — 2 commits, neither a conversion |
| new primitives | 0 |
| route-identity decision sites removed | 0 |
| baselines updated to absorb a change | 0 |

### Dependencies and blockers encountered

1. **The precedence bit** — the blocker above.
2. **The registry lens resolution** — `_resolve_lens` ≠ `resolve_lens`; the route filters on portfolio ids, not provenance type.
3. **Governance is applied downstream of the route**, in `mi_service`, not inside the handler. Good news for conversion: the receipt layer sees an envelope and does not care how it was produced — so §5's receipt equivalence is likely reachable once the input problem is solved.

### Does this change the expected cost of later conversions?

**Yes, upwards, and specifically.** Two of the three findings above are the same
shape: *a governed concept has more than one resolution path, and the route uses
the stricter one*. `lens_from_selection` defaults to Total; `resolve_lens`
returns provenance type where the route uses portfolio ids; `resolve_lens`
returns a scope where the route also needs the precedence bit.

**A plan cannot be built from "the owner's answer" alone — it must be built from
the owner's answer *as the route consumes it*.** That is a per-concept
verification cost the scoping study did not anticipate, and it will recur for
every governed concept in every route conversion.

---

## 5. What would clear this

One deliberate contract addition, in the same shape as Phase 1A: carry the
owner's `mentions_portfolio` reading alongside the resolved scope, so a
consumer can distinguish

* *the question said whole-book* (question wins over any default), from
* *the question said nothing about source scope* (default wins).

Pre-registered as declared-failing at
`question_interpretation/tests/test_source_scope_claim.py::test_the_contract_says_whether_the_question_named_a_source_scope`
— two cases, `xfail(strict=True)`, so it fails loudly if the property ever
arrives without being claimed.

That is a Phase 1A-shaped contract task, not a conversion task, and it should be
authorised as one.
