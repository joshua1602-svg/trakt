# Conversion 2 — `period_movement` — report

## Verdict

# STOP — COST ASSUMPTION BREACHED

**282 production lines against the pre-registered cap of 240.**

The cap was registered before any production change (`054307b`) and is
**unchanged**. It is not revised here. Conversion 3 does **not** proceed
automatically.

The conversion itself is correct and shipped: `period_movement` now executes as
a plan over the interpretation contract, the answers are unchanged across the
whole owned surface, and every non-cost stop condition is clear. What failed is
the **economic hypothesis**, and §10 says exactly how.

| | |
|---|---|
| base | `f56bd35` |
| production lines changed | **282** (cap **240** — breached by 42) |
| — shared infrastructure | **138** |
| — route-specific | **144** |
| Conversion 1, for comparison | **383** |
| new primitives | **0** ✓ |
| envelope pairs compared | 36 |
| envelope leaf fields compared | 7,633 |
| differences | **0** |
| A5 surfaces re-run | 5 of 5, all identical by name |
| silent drops | **0** |

---

## 0. Base and HEAD

| | |
|---|---|
| branch | `claude/clause-splitting-scoping-38ahbz` |
| base | `f56bd35` — Conversion 1 complete, working tree clean |
| Conversion 1 commits present | `7d6a921`, `6b5c7db`, `a2af2df`, `437484f` ✓ |
| Conversion 1 verified **live**, not falling back | 54 of 54 owned renders contract-supplied, 54 compositional plans built, **0 deferrals** |

That last line was checked *before* starting, and it matters: Conversion 1's own
first equivalence run was **vacuous** — 0 differences across 54 pairs while the
plan path was taken **zero** times, because an exception in the interpretation
builder was being swallowed. Confirming the converted route is genuinely live is
now a precondition of every conversion, not an assumption.

### Commits to equivalence

| commit | |
|---|---|
| `054307b` | pre-register stop conditions, cap **unchanged** at 240 |
| `00e6074` | instruments for the surface and envelope diffing |
| `724d6c9` | `period_movement` executes as a plan over the contract |
| `7d9b821` | tests that the switch cannot silently come undone |
| `9945922` | the unit test supplies a contract, as production does |

**5 commits**, one change each.

## 1. What was converted

One route: `chat_routing._route_period_movement`. The call it used to make —

```
lens  = _resolve_lens(question, source_lens)          # reads the question
span  = _period_request.requested_span(question)      # reads the question again
mv    = movement_mod.period_movement(..., lens_filters=..., span_periods=...)
```

— is now

```
mv    = _plan.period_movement(output_root, client_id, interpretation=..., ...)
span  = _period_request.span_from_claim(interpretation.time)
```

Both reads of the raw question are gone from the route. The plan derives the
population and the window from the contract, and the executor delegates to the
**existing** `movement_summary.period_movement` implementation — the periods,
deltas, regional bridge and cohort attribution are not re-derived, because a
second implementation of the same economics is an A2 risk for no gain.

## 2. The S8 prerequisite — was the contract sufficient?

The target-state closure recorded that `period_movement` re-decides exactly
**two** semantic concepts downstream, and that both are carried by the contract:

| concept | contract field | who owns it now |
|---|---|---|
| source scope | `source_scope.{base_population, portfolio_ids, provenance, state}` | `projection._source_scope`, once, upstream |
| the stated window | `time.{window_periods, window_governed, trend_window}` | `projection._time` → `period_request.requested_span`, once, upstream |

**No third concept appeared.** The closure finding was complete for this route.
S8 not breached — and this is the condition that mattered most, because a third
concept would have meant the closure inventory was wrong, which is worse than
an overrun.

Measured, in production code:

```
chat_routing callers of _resolve_lens        5  →  4
chat_routing callers of requested_span       1  →  0
```

`_resolve_lens` is still alive for the eight **unconverted** routes; that is the
migration's remaining work, not a defect of this conversion. The window has no
second production owner left at all.

## 3. Route ownership, by enumeration

`migration_phase0/route_ownership_period_movement.py` — 17 candidate questions ×
3 caller scopes (`None`, `direct`, `acquired`).

| | |
|---|---|
| claimed by the `period_movement` recogniser | **12** |
| deliberately excluded, and verified excluded | 5 (`X1`–`X5`) |
| window mismatches, contract vs shipped owner | **0** |

The five `X` cases are questions that *look* like movement questions and are
answered by other routes (`Summarise the portfolio`, `How has the funded balance
evolved over time`, `Compare June 2026 with May 2026`, `What has changed?`,
`Show the balance by region`). They are listed and asserted **not claimed**, so
the surface cannot silently grow to include a route this conversion did not
touch — the same discipline that stopped Conversion 1 measuring a surface it did
not own.

The zero window mismatches are the pre-switch proof that
`time.window_periods` already carried what `requested_span` read, across every
owned case and every scope. Without that, the switch would have been a guess.

## 4. The plan

`build_period_movement_plan(interpretation, *, region_column, has_portfolio_column)`
— **the question is not a parameter**, and that is enforced over the AST, not
over the source text.

Primitives used, all from the existing seven:

```
stack_periods      take="pair", span_periods=<from the contract>
select_population  _population_step(...) — THE SAME FUNCTION portfolio_summary uses
resolve_measure    x5 governed headline metrics
compare            the five metrics, prior vs current, as absolute delta
group              region, and source_portfolio_id — "of": "the delta"
```

**S2 — new primitives: 0.** `compare` was already one of the seven derived by
the scoping study; Conversion 2 is the first route to *use* it, which cost one
line for its id constant. `project` remains unused.

## 5. Payload and receipt equivalence

`migration_phase0/envelope_snapshot.py`, taken **before** and **after** on the
real tree at the two commits — not in a worktree. (A worktree snapshot was
attempted first and was invalid: without the governed multi-period dataset every
case answered "at least two funded reporting periods are needed", which would
have compared two refusals and proved nothing.)

| | |
|---|---|
| pairs (12 owned cases × 3 caller scopes) | **36** |
| envelope leaf fields compared per pair, total | **7,633** |
| **differences** | **0** |

Compared in full, not by summary: `answer`, `artifacts`, `route`, `ok`,
`controlledRefusal`, `error`, `verdict`, `facets`, `guardFacets`, `notApplied`,
`payloadKeys`, `metadataKeys`, `executionSummary`, `portfolioScope`,
`portfolioCoverage`, `reconciliation`, `sourceNotes`, `warnings`.

The receipt-bearing fields are genuinely populated, so the zero is not the zero
of two empty structures: `executionSummary` non-empty in 27/36, `facets` 27/36,
`reconciliation` 30/36, `verdict` 30/36, `payloadKeys` and `metadataKeys` and
`portfolioScope` 36/36.

> A first pass at this diff keyed the entries by `(case, lens)` where `lens` was
> not a field, silently collapsing 36 entries to 12 and comparing a third of the
> surface while reporting "36 pairs". It was caught by asserting the key count.
> The figures above are the corrected run. This is the fourth time in this
> programme that an instrument has reported a pass it had not earned, and the
> reason every instrument now asserts its own denominator.

**S4 — bespoke exception: none.** There is no branch anywhere in the payload or
receipt path naming this route, a period, a book or a dimension. The executor
returns the same dict the shipped engine returned, so everything downstream is
unchanged *by construction* rather than by adaptation — which is why the receipt
cost of this conversion is zero lines.

## 6. Structural guarantees, checked over the AST

```
_route_period_movement       calls _resolve_lens = False   calls requested_span = False
build_period_movement_plan   calls _resolve_lens = False   calls requested_span = False
period_movement (plan)       calls _resolve_lens = False   calls requested_span = False
span_periods                 calls _resolve_lens = False   calls requested_span = False

build_period_movement_plan params: ['interpretation','region_column','has_portfolio_column']
takes a question parameter: False
```

Over the parsed tree, never over the text. Three guards in this programme have
previously passed by matching their own docstrings — prose that *denied* the
thing the guard was looking for.

The route **defers rather than falls back** when no contract is available
(`interpretation is None → return None`). Keeping the lens-resolved path as a
fallback would leave a second population owner reachable exactly when the first
one failed, which is the worst possible moment for two owners to disagree.

## 7. Stop conditions, each answered

| # | condition | result |
|---|---|---|
| S1 | production lines > 240 | **BREACHED — 282** |
| S2 | a new primitive is required | clear — 0 new; `compare` was already one of the seven |
| S3 | economics breach A2 (£0.005) | clear — 0 differences across 7,633 fields; the tolerance was not approached |
| S4 | a bespoke `period_movement` exception in payload/receipt | clear — none; no route-naming branch exists |
| S5 | any silent drop | clear — `time_series_surface` **silent drops 0** |
| S6 | any silent population widening | clear — precedence and sufficiency matrices unchanged; the plan refuses on EMPTY scope rather than reading it as Total |
| S7 | any unexplained regression | clear — see §8 |
| S8 | a generic concept the contract does not carry | clear — the two named concepts were both carried; no third appeared |

**One breach, and it is S1.**

## 8. Regression, by name

The five registered A5 surfaces, re-run after the switch and compared **by case
name**, not by total:

| surface | baseline | after Conversion 2 |
|---|---|---|
| robustness 44 | 32 CORRECT / 6 UNHELPFUL_REFUSAL / 4 SAFE_REFUSAL / 2 DISCLOSED_LIMITATION | **identical** |
| — by intent Q1–Q9 | `Q1 4C · Q2 3C,1U · Q3 2D,2U · Q4 2C,2U · Q5 3C,1U · Q6 4C · Q7 4C · Q8 12C · Q9 4S` | **identical, every family** |
| shipped shapes | correct 15, wrong 0, total 15 | **identical** |
| routed surface | passed 31, failed 1 (`rt_004`) | **identical** — same known-open case |
| recognition 61 | DELIVERED 15; by shape T1 6 · T2 1 · T3 0 · T4 0 · T5 0 · T6 0 · T7 3 · T8 5 | **identical, every shape** |
| time-series | T1 PROVEN · T2 PARTIAL · T3–T8 ABSENT; **silent drops 0**; honest refusals 20/29 | **identical** |

`rt_004` ("funded balance by quarter") is the pre-registered known-open defect,
failing before this programme began. It is not introduced here.

### The whole repository, by name

The registered gate was a "core suites" scope. This conversion ran the **full
repository — 10,373 tests** — as four shards, which is a strict superset, and
attributed **every** failing name rather than comparing totals.

| shard | scope | passed | failed | errors |
|---|---|---|---|---|
| A | `mi_agent` + `question_interpretation` | 1,811 | 7 | 0 |
| B | `mi_agent_api` (isolated) | 1,231 | 12 | 0 |
| C | `tests/` first half | 3,756 | 51 | 4 |
| D | `tests/` second half | 3,326 | 45 | 24 |

Attribution, name by name:

* **Shard A — 7 failures, and they are exactly the 7 pre-existing baseline names**
  (`test_complex_query_executes_all_filters`, `test_C_joint_borrowers_count_and_balance`,
  three `test_p0_execution_receipt` disclosure cases,
  `test_layered_question_routes_to_llm_even_when_deterministic_parses`,
  `test_the_wording_that_asked_is_returned[balance by each month]`). Nothing added,
  nothing fixed.
* **Shard B — 12, which is the count Conversion 1 left behind.** It was 13 before
  one convention fix; see below.
* **Shard C — all 55 pre-existing.** One is baseline name #8
  (`test_q7_compares_the_two_governed_sides_and_reconciles`); the other 54 are in
  `tests/mail/`, `tests/test_annex*`, `tests/operations_control/` and
  `tests/test_evidence_manifest.py`, and were verified failing identically at
  `f56bd35`.
* **Shard D — all 69 pre-existing, measured directly.** The identical 157-file
  shard was re-run against the base production files in a full copy of the tree:
  **45 failed, 24 errors — the same names, exactly.** Set difference in both
  directions is empty. Nothing introduced, nothing fixed.

Two attribution traps were hit and corrected while doing this, both worth
recording because both would have produced a false alarm:

1. A fresh `git worktree` at the base commit **skips** the evidence-manifest and
   MI tests rather than running them — it lacks the generated `onboarding_output`
   artefacts — so two tests looked "introduced" when they had simply not run at
   base. Symlinking the data in was not enough; the base comparison was
   therefore taken in a **full copy of the working tree** with only the
   production files reverted, which reproduces HEAD's behaviour exactly (89
   passed standalone, same as HEAD).
2. `tests/test_portfolio_name_resolution.py`,
   `tests/test_portfolio_summary_conversion_prerequisites.py` and
   `tests/test_portfolio_identity_alignment.py` fail inside the 157-file shard
   and **pass standalone at HEAD** — order-dependent cross-test state. "It
   passes on its own" is an explanation, not evidence, so it was not accepted as
   one: the whole shard was re-run at base and came back identical by name.

**Introduced failing names: 1, fixed, and it was the switch working.**

### The one by-name movement, and why it is the switch working

`mi_agent_api/tests/test_movement_summary.py::TestRouting::test_the_movement_route_produces_a_reconciling_answer`

It called `_route_period_movement` **directly with no interpretation**.
Post-conversion that is a deferral by design: the route takes both its semantic
inputs from the contract, and keeping the lens-resolved path as a fallback would
leave `_resolve_lens` and a second reading of the question reachable exactly
when the contract failed.

In production the handler has one caller — the recogniser registry — and it
always supplies a contract. The test used a calling convention production no
longer uses, so it now uses production's. **Its assertions are unchanged**: same
route, same movement, same South East attribution, same kpi/chart/table
artifacts. `mi_agent_api/tests` isolated: **13 → 12**, back to the count
Conversion 1 left behind, and this was the only new name.

This is the same movement, for the same reason, that Conversion 1 recorded for
the summary route's unit test — which is itself a small piece of evidence that
the two conversions are doing the same kind of thing.

### Conversion 1 is still live and still correct

`tests/test_conversion1_portfolio_summary.py` — **15 passed**, unchanged. This
matters more than it looks: Conversion 2 **renamed and restructured the module
Conversion 1 shipped**, extracting `_population_step` out of `build_plan`. If
that generalisation had shifted `portfolio_summary`'s behaviour by a step, the
migration would be trading one converted route for another.

`tests/test_conversion2_period_movement.py` — **17 passed**, new. Five
properties, each one a defect already paid for once in this programme:

* the plan cannot read the question — checked over the AST;
* both semantic inputs come from the contract — scope **and** window;
* the span wording stays with its owner — `span_from_claim` rebuilds every field
  from the claim rather than guessing a label;
* the plan layer is shared, not copied — one module, one `_population_step`;
* the answers are unchanged — the envelope for the movement surface.

## 9. Cost, in the two registered parts

Measured with `git diff --numstat -M f56bd35 HEAD`, production only — tests and
docs excluded, the rename counted at its real changed lines rather than as
churn, exactly as S1 registered.

```
 24    0   mi_agent/period_request.py
170   42   mi_agent_api/{portfolio_summary_plan.py => analytical_plan.py}
 29   17   mi_agent_api/chat_routing.py
                                        TOTAL  282     (cap 240)
```

Split at the hunk level — not estimated, attributed hunk by hunk:

| | lines |
|---|---|
| **SHARED INFRASTRUCTURE** | **138** |
| `analytical_plan.py` — rename, module docstring, the `compare` primitive id, and `_population_step` extracted so both route plans share one owner of EMPTY / UNRESOLVABLE / FILLED | 110 |
| `period_request.span_from_claim` — the contract → `SpanRequest` bridge | 24 |
| `chat_routing.py` — module-rename churn | 4 |
| **ROUTE-SPECIFIC (`period_movement`)** | **144** |
| `analytical_plan.py` — `build_period_movement_plan`, `span_periods`, the `period_movement` executor | 102 |
| `chat_routing.py` — the switch in `_route_period_movement` and its registration | 42 |
| **TOTAL** | **282** |

Recorded separately, and **not** counted as production-line cost:

| | |
|---|---|
| test files | 1 new (`test_conversion2_period_movement.py`, 17 tests) |
| instruments | 2 (`route_ownership_period_movement.py`, `envelope_snapshot.py`) |
| docs | 2 (stop conditions, this report) |
| production modules changed | 3 (0 new — the plan module was renamed, not added) |
| commits | 4 |

## 10. The hypothesis, answered

Conversion 1 measured **383** and offered this explanation:

> the overrun carries a one-off cost — promoting the plan layer from a shadow
> instrument into production. On that reading the switch itself was 94 lines and
> the rest is infrastructure the next conversions inherit.

Conversion 2 was registered as the experiment that would test it, with the cap
held at 240 so the evidence could answer.

**The evidence.** One caveat on the comparison first, because it changes what
can honestly be concluded from it: **Conversion 1 and Conversion 2 did not use
the same cost accounting.** The two-part split was registered *for Conversion 2*,
after Conversion 1's overrun raised the question it answers. Conversion 1's own
recorded figures are total **383**, "the switch itself" **94**, and "the plan
layer, one-off" **~150 raw / ~95 code** — and those categories do not map onto
shared/route-specific, because C1's "plan layer" bundles `portfolio_summary`'s
own plan and executor (route-specific, by the registered definition) together
with the generic `Step`/`Plan` machinery (shared). Reporting a
shared-cost *trend* across the two would be inventing a comparison.

Two things are genuinely like-for-like, and they are enough:

| | Conversion 1 | Conversion 2 |
|---|---|---|
| **total production lines** (S1's definition, same instrument) | 383 | **282** |
| **shared infrastructure** (registered for C2; first measurement) | not recorded in this form | **138 of 282** |

And the hypothesis does not need Conversion 1's split at all. It made a
prediction about *this* conversion: the next conversions **inherit** the
infrastructure. That predicts a shared component of approximately **zero**.

Measured: **138.**

### The hypothesis as stated is FALSIFIED

Shared infrastructure did **not** go to zero. It went to 138 — roughly as large
as the route-specific cost, and around half of everything Conversion 2 spent. A
second conversion that inherits a finished generic layer should have paid
approximately nothing for it. It paid 138.

### But the naive alternative is falsified too

"Every route conversion is expensive, and nothing is shared" also fails: total
cost fell **383 → 282, a 26% reduction**, and the second route genuinely reused
`_population_step`, `Plan`, `Step`, `lens_filters`, `lens_label` and the whole
blocked-plan refusal path rather than reimplementing them.

### What the evidence actually shows

**A plan layer built for one route is not yet a generic plan layer — it is a
route-specific procedure that happens to live in a module.** Conversion 1 did
not build shared infrastructure; it built `portfolio_summary`'s execution and
put it somewhere reusable. Generalising it was deferred, not done, and
Conversion 2 paid for it:

* the module had to be **renamed** — `portfolio_summary_plan` was a name that
  denied a second route existed;
* the population step had to be **extracted** — inline in `build_plan`, it would
  have been copied, and a copy is how two routes come to disagree about which
  loans are in the book;
* a **second contract input** had to be bridged — `span_from_claim` exists
  because this is the first route needing a *window*, and nothing in Conversion 1
  needed one.

Each is a real generalisation, paid once, at the point where a new *kind* of
demand first appeared. None of them is per-route work.

### The falsifiable prediction, stated in advance

If that reading is right, the shared component must **decay**:

* Conversion 3's shared component should be **materially below 138**, and close
  to zero if it needs no contract input beyond scope and window;
* route-specific cost should stay near **~144**, since that is what converting a
  route actually costs.

If Conversion 3's shared component is again near 138, then "shared
infrastructure" is a mislabel, the true cost is per-route, and A1's economics
fail. **That is the measurement Conversion 3 exists to take** — and, on the
route-specific figure alone (144 < 240), it is the only remaining question about
whether this migration is affordable.

## 11. Where the overrun came from

The four categories the conversion brief named, with the measured split:

| source | lines | share |
|---|---|---|
| **shared generic architecture** | **138** | **49%** |
| **route-specific semantics** | **144** | **51%** |
| receipt / payload adaptation | **0** | 0% |
| testing / governance requirements | **0** production lines | — recorded separately |

Two of the four cost nothing, and both for the same structural reason: the
executor returns the shipped engine's own result dict, so no downstream layer
had to be adapted; and tests and instruments are not production lines by the
registered definition.

**The overrun is 42 lines. Shared architecture alone is 138.** Had the plan
layer been generic when Conversion 1 finished it, Conversion 2 would have come
in at approximately **144 — well inside the cap**. The cost that breached S1 is
work Conversion 1 should have done and deferred, not work `period_movement`
required.

That is a real finding and not a rationalisation of the number: it is stated as
a **prediction Conversion 3 will falsify or confirm**, and the cap is not being
moved on the strength of it.

## 12. What must happen before Conversion 3

The brief is explicit — no automatic progression, and the migration-cost thesis
must be **re-baselined** first. The re-baselining needs one input this report
cannot supply: a third observation.

Recommended, in order:

1. **Do not move the 240 cap.** It has now produced two informative results. A
   cap that moves to fit the measurement stops measuring anything.
2. **Re-baseline A1 against two numbers, not one** — the thesis should predict
   *shared* and *route-specific* cost separately, because they behave
   differently and only one of them scales with the number of routes.
3. **Register Conversion 3's stop conditions with a split expectation**: shared
   ≤ 50 and route-specific ≤ 240. That is the shape of the prediction in §10 and
   makes it falsifiable rather than narrative.
4. **Do not pre-generalise the plan layer** in anticipation. Generalising ahead
   of a second demand is how the layer becomes wrong in a way no measurement
   catches.

### Should Conversion 3 proceed?

**Yes — but only as a deliberately taken third measurement, not as continued
migration, and only after the re-baselining above is committed.**

The reasoning, stated plainly so it can be disagreed with:

* Nothing in the *architecture* argues against it. Two routes are converted, the
  economics are identical, no primitive was added, no receipt exception was
  needed, and the contract carried every semantic fact both routes required.
* The *economics* are unresolved, and one more observation resolves them. §10's
  prediction is sharp enough to be wrong: shared cost must fall well below 138.
* Stopping now would leave the programme with two data points and a hypothesis
  it has already falsified once — the least useful place to stop.

What would change the answer: if the re-baselining cannot state a falsifiable
split expectation for Conversion 3, do not run it. An unfalsifiable third
conversion is not a measurement, and A1 would then have to be decided on the two
observations already in hand.

## 13. Abort conditions A1–A5

| | condition | status after Conversion 2 |
|---|---|---|
| **A1** | migration economics do not hold | **NOT TRIGGERED — but now the live question.** A1 needs three conversions; two are done. Route-specific cost (144) fits the cap; shared cost (138) is what must decay. Conversion 3 decides it. |
| **A2** | economics diverge ≥ £0.005 | **NOT TRIGGERED.** 36 pairs, 7,633 leaf fields, 0 differences. The tolerance was not approached on any field. |
| **A3** | a governed concept loses its single owner | **NOT TRIGGERED — improved.** The window's second production owner is gone (1 → 0 callers of `requested_span` in `chat_routing`); `_resolve_lens` retired from a second route (5 → 4). |
| **A4** | a new primitive is required | **NOT TRIGGERED.** 0 new. `compare` used for the first time, from the existing seven; `project` still unused. |
| **A5** | unattributable regression | **NOT TRIGGERED.** All five registered surfaces identical by name; silent drops 0; introduced failing names 0. |

## 14. Position after two conversions

```
compositional   2 of 15   portfolio_summary, period_movement
specialist     13 of 15
```

Both converted routes now share one plan module, one population step, one
refusal path, and take every semantic decision from the interpretation contract.
Neither reads the raw question.

**The architecture is working. The economics are not yet proven, and this
conversion is why we know that rather than assume it.**
