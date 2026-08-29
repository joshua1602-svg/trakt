# Conversion 1 — `portfolio_summary`

# STOP — COST ASSUMPTION BREACHED

**383 production lines against a pre-registered limit of 240.**

The conversion itself is **complete and proven**: production `portfolio_summary`
executes as a plan over the contract, payload and receipt equivalence is 0
differences across 54 rendered answer pairs, economics are identical, the
duplicate semantic owner is deleted, and regression is clean by name.

But S1 fired, and I pre-registered *"do not rationalise overruns after the
fact."* So this reports the breach and asks for a ruling rather than deciding
unilaterally that 383 is acceptable. The work is committed and working; nothing
needs to be undone to rule either way.

Base `44bc90c`; converted from `a56b7eb` (`44bc90c` + the three target-state
closure commits, whose contract fields the conversion depends on).

---

## 1. The breach, measured honestly

| | lines |
|---|---|
| `mi_agent_api/portfolio_summary_plan.py` (new) | **+280** |
| `mi_agent_api/chat_routing.py` | +66 / −14 |
| `mi_agent_api/movement_summary.py` | +17 / −6 |
| **total** | **383** |
| pre-registered limit (S1) | **240** |

Code-only, excluding blank lines, comments and docstrings: **~261** — still over.
**Both readings breach**, so the overrun is not an artefact of documentation
density.

### Why it is larger, stated as a finding rather than an excuse

**Conversion 1 pays a one-off cost the next conversions will not: promoting the
plan layer into production.** Of the 280-line new module, roughly 95 code lines
are `Step`, `Plan`, `build_plan` and the executor scaffold — the *plan artefact
itself*, which existed only in `migration_phase0/` as a shadow instrument.

The **switch** — the part that is actually "converting a route" — is the 94
lines across two existing modules, comfortably inside budget.

My 240 was set before I knew a first conversion means shipping the plan layer.
That is a real A1 datum, and it argues the budget was measuring the wrong thing
for conversion 1 — **but the threshold was pre-registered and I am not moving
it retroactively.** The ruling is yours.

## 2. Dataset-conflation check — no live defect

Run **before** any production change, because a conversion that finds a wrong
number afterwards cannot tell whether it caused it.

> *"The balance by seasoning segment excluding pipeline cases"*

| owner | resolves to |
|---|---|
| `view_named_by_question` | `None` |
| `resolve_active_view` | **funded** |
| `chat_routing._dataset_for` | **funded** |
| contract claim | **funded** (`default`) |

Answer: *Total Balance, grouped by Seasoning Segment, 2 groups, 11,035 loans.*

**The exclusion is honoured at every owner.** Three controls behave: the same
question without the disclaimer returns the identical figure; an explicit
*"pipeline balance"* resolves to pipeline and refuses for want of pipeline data;
*"ignoring the forecast"* stays funded. **Contract representation only** —
recorded, and the conversion continued.

## 3. Payload and receipt equivalence — the boundary Phase 0 could not reach

**18 route-owned cases × 3 caller defaults = 54 rendered answer pairs, 0
differences.**

Compared per pair: route identity, payload keys, metadata keys, governed
population declaration, `lensApplied`, `portfolioScope`, `portfolioCoverage`,
reconciliation, source notes, warnings, every artifact (type, title,
description, columns, rows, series, KPIs), execution summary, requested and
applied facets, receipt verdict, guard facets, limitation and refusal state, and
the answer text itself.

Both renders are the **shipped handler**; only the branch differs. No second
handler exists and nothing is copied.

### The first run of this measurement was worthless, and that is the finding

It reported **0 differences across all 54 pairs while the compositional path was
taken zero times.**

Phase 1G's routed wiring raised on every question — `frame.columns or []` is a
`ValueError`, not a falsy fallback — and the `try/except` around the provider
returned `None` silently. **A construction site that never constructed, for an
entire phase**, asserted by a unit test that passed it a lambda.

My Phase 1G report said the routed path "now has a construction site". It had
one that always failed. That claim was wrong and this corrects it.

The instrument now **refuses to report a pass it did not earn**: it counts which
branch each render took and raises if a compositional render fell back.

## 4. A design change the measurement forced

**`UNRESOLVABLE` must not block the plan.**

* `EMPTY` — nobody looked. Nothing can be planned from it, and reading it as
  Total would widen a population the question may have narrowed. **It blocks.**
* `UNRESOLVABLE` — the owner looked and found a name this book does not hold.
  That is a **refusal**, and the refusal already has a single route-independent
  owner: the facet layer raises a LOST narrowing and `assess` declines (Phase
  1E, proved across three routes).

Blocking it in the plan put a **second refusal owner** in the path, and measured,
it moved **23 payload and receipt fields** on `acquired_001` and *Highgate
Mortgages Book*: the route deferred, the question fell through to the
point-in-time path, and route identity changed from `portfolio_summary` to
`None`. Both still refused — so the defect was invisible in the answer and
visible only in the payload.

The step is still recorded with `unresolved`, so the plan **declares** what it
could not do and the receipt decides.

## 5. The switch

| # | | |
|---|---|---|
| 1 | recognition | unchanged — `_is_portfolio_summary` |
| 2 | contract | `RouteRequest.resolve_interpretation()`, passed at the registration site |
| 3 | **replaced** | `movement_summary.portfolio_summary` → `portfolio_summary_plan.portfolio_summary` |
| 4 | envelope | unchanged — same result dict, so prose, artifacts and `_envelope` are untouched **by construction** |
| 5 | receipt | unchanged — `mi_service` reconciles and builds as before |
| 6 | fall-through | no contract → the route defers |

**Five primitives, no new one:** stack periods, select population, resolve
measure ×5, group ×2, rank. `compare` and `project` are not needed.

`cohort_balances` was **extracted** from `movement_summary`, not copied — one
calculation, one owner.

## 6. `_resolve_lens` — deleted from this route, with proof

§5's four requirements, each proved **before** the deletion:

1. **the contract supplies everything it supplied** — the 54-pair, 0-difference
   comparison above, measured while both paths existed;
2. **every former call path reaches the contract** — 54 of 54 owned renders were
   contract-supplied; the fall-through fired **zero** times;
3. **no consumer relies on their disagreement** — the payload comparison *is*
   that test, and it found none;
4. **no by-name regression** — §8.

Deleted rather than left unreachable: keeping the lens path as a fallback would
leave `_resolve_lens` in this route as a **second population owner reachable
exactly when the first one failed** — the worst moment for two owners to
disagree. One owner, or none.

**Surgical.** The five `_resolve_lens` calls owned by other routes are untouched,
and a test pins that they remain.

`portfolio_summary` is now **CLEAN** in the semantic-owner inventory. The generic
estate drops from **13 downstream semantic decisions to 12**.

### Two guards in this conversion were fooled by their own prose

The plan module's import guard and the deletion guard were both first written as
substring checks, and both matched the sentences that *deny* the thing they look
for — `"source_portfolio_lens"` is the step kind, and the docstring says
"`_resolve_lens` is deliberately NOT reachable from here". Both now read the AST.
**A guard that reads prose is not reading code.**

## 7. Economics — 0 differences

| case | figure | route |
|---|---|---|
| named direct | 7,126 / £1.39bn | `portfolio_summary` |
| governed id | 3,909 / £579.4m | `portfolio_summary` |
| direct category | 7,126 / £1.39bn | `portfolio_summary` |
| acquired category | 3,909 / £579.4m | `portfolio_summary` |
| funded category | 11,035 / £1.96bn | `portfolio_summary` |
| no scope named | 11,035 / £1.96bn | `portfolio_summary` |
| zero-row governed | governed no-data outcome | `portfolio_summary` |
| storage id / unknown label | controlled refusal | `portfolio_summary` |

Shadow harness: 9 cases, **0 economic differences**, 0 blocked, 0 external
injection, populations equal on rows. A2 tolerance (£0.005) not approached on any
field.

## 8. Regression, by name

| surface | result |
|---|---|
| calibration bank | unchanged |
| robustness 44 | **byte-identical** — 32/6/4/2 |
| — seasoning by name | **Q1 4 · Q7 4 · Q8 12 CORRECT** |
| shipped shapes | **byte-identical** — 15 correct, 0 wrong |
| routed surface | **byte-identical** — 31 passed, `rt_004` known-failing |
| recognition | **byte-identical** — 15/7/10/29 |
| time-series | **byte-identical** — **silent drops 0** |

Core suites: **8 failed, 2117 passed** — the same eight pre-existing failures by
name.

`mi_agent_api/tests/`, isolated before and after (`a56b7eb` vs HEAD): **12 → 13
→ 12**. The one new name was
`test_movement_summary::test_the_summary_route_names_the_governed_metrics`,
which called the handler **directly with no contract** — a convention production
no longer uses, and the deletion working. Its calling convention is updated with
the reason recorded in it; its assertions are unchanged.

**Introduced failing names: 0. Silent drops: 0. Silent population widening: 0.**

## 9. Conversion 1 cost — the first A1 observation

| | |
|---|---|
| production lines changed | **383** (S1 limit 240 — breached) |
| — of which the plan layer, one-off | ~150 raw / ~95 code |
| — of which the switch itself | **94** |
| production modules changed | **3** (1 new, 2 edited) |
| test files touched | 2 (1 new, 1 convention update) |
| commits to equivalence | **4** |
| **new primitives** | **0** ✓ |
| semantic decision sites removed | **1** (13 → 12 in the generic estate) |
| bespoke route exceptions | **0** ✓ |

**No median is inferred.** One observation is not an estimate, and A1 needs
three.

**Blockers found:** the Phase 1G provider bug (§3) and the `UNRESOLVABLE`
blocking design (§4). Both were found by measurement, before shipping.

## 10. The target architecture — a hybrid, stated explicitly

**The end state is a compositional funded-book analytical core plus governed
specialist capabilities — not one universal execution mechanism.**

**Within the compositional model (7 of 15):**
`portfolio_summary` (converted), `period_movement`, `temporal_compare`,
`evolution`, `funded_bridge`, `geo_exposure`, `period_change_analysis`.

**Outside it, deliberately (8 of 15):**

| route | why it stays outside |
|---|---|
| `scenario` | stress semantics over a derived frame |
| `cohort_conversion` | pipeline-to-funded conversion model |
| `forecast_extrapolation` | run-rate projection |
| `cohort_progression` | static-pool progression |
| `risk_limits` | governing-document limits |
| `concentration_tests` | eligibility/limit tests |
| `portfolio_risk_comparison` | own workflow brief, own recognition and rejection rules |
| `concentration_analysis` | same |

The last two are *funded-book* routes that hand the whole question to a workflow
which re-interprets it. They are not compositional today and forcing their
semantics into the generic contract is what the closure task forbade.

**Documentation only. No specialist was redesigned.**

## 11. Final status

# STOP — COST ASSUMPTION BREACHED

Every other condition passed:

| | |
|---|---|
| S1 production lines ≤ 240 | **FIRED — 383** |
| S2 no new primitive | passed — 0 |
| S3 no bespoke route exception | passed — 0 |
| S4 A2 economic tolerance | passed — 0 differences |
| S5 no unexplained regression | passed — every movement attributed by name |
| S6 no silent drop or widening | passed — 0 and 0 |

No A1–A5 abort condition fired. A1 remains **not yet measurable**; this is
observation 1 of 3.

## 12. Should Conversion 2 proceed?

**Not until S1 is ruled on.** Three options, and my recommendation is the second:

1. **Accept 383 for conversion 1** and re-register the budget for conversions
   2–3 at the *switch* cost, which is what they actually pay now that the plan
   layer ships. On this evidence that is ~**120 lines** each.
2. **Accept 383 and hold 240 for conversion 2** — no re-registration, and
   conversion 2 becomes a real test of the "one-off cost" claim in §1. If
   `period_movement` lands inside 240, the claim is evidenced; if it does not,
   the budget was wrong rather than the conversion. **This is the one I would
   choose: it tests my own explanation rather than assuming it.**
3. **Reject and trim** — the module can lose perhaps 40 raw lines of
   documentation without losing code. It would not reach 240, and I do not
   recommend shaving comments to fit a number.

When conversion 2 does proceed, **`period_movement`** remains the right choice:
it shares `movement_summary` with the route just converted, both its downstream
semantic reads are already carried, and it is the cheapest available test of
whether a second conversion costs less than the first.
