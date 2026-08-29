# Phase 0 + first migration slice — report

**Nothing client-visible ships from this task.** T3, T4, T5, T6 and T7 remain
closed. No production module was changed. The `portfolio_summary` shadow path
does not replace production execution.

Base: `42cef00`. Scoping study: `9f2d256`. This task: `c88a792`, `d4862b5`,
`c7bc798`.

**Recommendation: NO-GO on switching `portfolio_summary`.** The economics
reconcile exactly; the plan cannot be built from the interpretation contract.
§6 gives the specific blocker and what would clear it.

---

## 1. Phase 0 migration baseline

**Artefact:** `migration_phase0/BASELINE.json` — regenerate with
`python -m migration_phase0.freeze_baseline`.

**Attribution is exact.** The product code at HEAD is byte-identical to
`42cef00`: `git diff --name-only 42cef00 HEAD` outside `docs/`,
`compositional_plan_scoping/`, `migration_phase0/` and the new test file returns
nothing. Every figure below is therefore a measurement **at the base commit**,
not near it. The fixture's four tape hashes also match the recorded MI_40Q
pre-launch baseline exactly, so the book is the recorded one, not a rebuild that
drifted.

| surface | result | record |
|---|---|---|
| calibration bank | **267 passed, 0 failed** | `pytest mi_agent/tests/test_mi_calibration_bank.py` |
| robustness 44 | **32 CORRECT / 6 UNHELPFUL_REFUSAL / 4 SAFE_REFUSAL / 2 CORRECT_WITH_DISCLOSED_LIMITATION** | `robustness_deterministic.json` |
| — seasoning families, by name | **Q1 4 CORRECT · Q7 4 CORRECT · Q8 12 CORRECT** | same |
| shipped shapes | **15 correct, 0 wrong, 0 refusals** | `shipped_shapes.json` |
| routed surface | **31 passed, 1 failed (`rt_004`)** | `routed_surface.json` |
| recognition (61 phrasings) | **DELIVERED 15 / WORDING 7 / UNPARSED 10 / CAPABILITY 29**; 13 reach no route | `recognition_diagnosis.json` |
| time-series surface | T1 PROVEN, T2 PARTIAL, T3–T8 ABSENT; **silent drops 0**; 20 of 29 honest refusals | `time_series_surface.json` |

The baseline keeps four categories apart and never collapses them: **delivered
behaviour**, **governed refusals**, **known failures**, **known-open governance
defects**.

### Known failures — pre-registered, not the migration's

1. **`test_q7_compares_the_two_governed_sides_and_reconciles`** — `assert answer['ok'] is True` → `assert False is True`. Verified to fail identically on a clean tree at `42cef00`.
2. **`routed_surface::rt_004`** (*"funded balance by quarter"*) — expects `route=None`; now reaches `route='evolution'`, `verdict='refuse'`. The expectation was last edited at `1b90fe4` and its own note pins it as a deliberate "before". `42cef00` widened the time-axis vocabulary and moved the behaviour; the expectation was not updated with it.

### Known-open governance defects — pre-registered

3. **Arity-2 disclosure** (below, §3).
4. **Interpretation lens gap** (below, §4) — found while building the shadow plan.
5. **Filter clause join** — 71 questions carry unjoined halves; `clause_id` set on 0. Pre-dates this programme; does not block `portfolio_summary`.

---

## 2. Abort conditions

**`docs/mi_migration_abort_conditions.md`**, committed before any route
migration. Five conditions, each with an instrument and a threshold.

Two are worth restating here because they shape the rest of this report.

**A1 (cost explosion) is expressed as a multiple of an observed median, not an
absolute.** The first three conversions establish *m* (production lines) and *c*
(commits to equivalence); the migration stops when a later route needs more than
`2 × m` or `2 × c`. No conversion has been measured yet, so any absolute figure
written today would be an invention. **`portfolio_summary` did not reach a
conversion, so *m* and *c* are still unknown.**

**A4 distinguishes two interpretation cases, and only one is an abort.** The
contract owns a decision and the plan rereads the question anyway → abort. The
contract *cannot represent* the decision → stop, report, extend deliberately.
**The second fired in this task.**

**A5's P0 property is absolute:** silent drops stay 0, and no equivalence result
overrides it.

---

## 3. Governance prerequisite A — arity-independent disclosure

### What it is, precisely

`mi_agent/mi_query_executor.py::_execute_grouped`, lines 1191–1201. Inside
`if len(group_cols) == 1:` sit three things:

* `out["loan_count"]` — the denominator column, always;
* `out[f"{value_col}_total"]` — the total, for `avg` only;
* the thin-sample **warning**, for `avg`/`weighted_avg` only.

All three are withheld at arity ≥ 2.

### Measured

```
arity 1  [collateral_geography]                12 groups,   1 thin ( 8.3%)  disclosed: True
arity 2  [collateral_geography, ltv_bucket]    88 groups,  16 thin (18.2%)  disclosed: False
arity 2  [ltv_bucket, ticket_bucket]           50 groups,  11 thin (22.0%)  disclosed: False
```

End to end, the same question at two arities:

| question | `loan_count` | thin warning |
|---|---|---|
| *average borrower age **by region*** | yes | **yes** — "1 group(s) have fewer than 5 loans" |
| *average borrower age **by region and LTV band*** | **no** | **no** |

**`ltv_bucket × ticket_bucket` is `shipped_shapes` C1–C5, which currently
PASS.** The defect is live on a shipped, passing shape.

### It is three shapes of one defect, not one guard

1. `_execute_grouped` — withholds denominator **and** warning at arity ≥ 2.
2. `_execute_grouped_measure_set` — attaches `loan_count` at **every** arity, but raises the thin-sample warning at **no** arity.
3. `mi_query_executor.py:1045` (contribution) — resolves only `group_keys[0]`. A latent arity assumption, **currently unreachable** because the two-dimension contribution question refuses upstream. Stated as latent rather than live, because it is.

### Can it be generalised without changing answer semantics?

**Yes, and the mechanism is already arity-independent.** `_align` handles
`MultiIndex`; `_group_sum`, `_maybe_concentration`, `_apply_top_n` and
`_coverage_block` all take `group_cols` as a list. The generalisation is
`work.groupby(group_cols)` instead of `group_cols[0]`, with the guard removed —
no new helper, no new threshold.

**No policy value changes.** `LOW_GROUP_COUNT = 5` and the `avg`/`weighted_avg`
restriction are policy. This generalises **where** the policy applies. The
pre-registered tests read `LOW_GROUP_COUNT` from the module rather than
restating it, so a future policy change moves the tests with it.

**One semantics change is unavoidable and must be authorised separately:**
adding `loan_count` to an arity-2 result **adds a column to a shipped
artifact**, and `shipped_shapes` C1–C5 assert on those artifacts. That is a
user-visible change. Per the sprint's rule it is permitted only as an
"explicitly authorised governance correction, separately measured and
attributable" — it is **not** authorised by this task and was not made.

### Pre-registered tests

`tests/test_migration_preregistered.py::TestArityIndependentDisclosure` — 4
tests, all committed, `xfail(strict=True)`:

* arity 1 attaches the denominator — **passes today** (control)
* arity 1 discloses thin groups — **passes today** (control)
* arity 2 attaches the denominator — **declared failing**
* arity 2 discloses thin leaf groups — **declared failing**

The controls exist so a later failure is attributable to a baseline move rather
than to the target.

---

## 4. Governance prerequisite B — receipt/facet proof independent of route identity

### The count, verified

`python -m migration_phase0.route_identity_inventory`.

**14 production decision sites** across **11 route-keyed constants** in three
modules:

| module | decision sites | functions |
|---|---|---|
| `mi_agent/execution_receipt.py` | 12 | `reconcile_routed_facets` (7), `reconcile_facets` (2), `detect_measure_substitution`, `route_time_grain`, `granularity_disclosure`, `build_routed_receipt` |
| `mi_agent_api/chat_routing.py` | 1 | `_disclose_lens_scope` |
| `mi_agent_api/mi_service.py` | 1 | `_route_requires_run` |

### This corrects the scoping study, in both directions

The study reported **54 route-name literals**. That count included
**declaration** sites — a route naming *itself* on its own envelope, and the
allowlist definitions. Those are the channel through which execution declares
what it was; a compositional layer replaces *what* declares, not the fact of
declaring. Counting only consumers that **branch** on identity gives **14**.
**The coupling is smaller and more tractable than the study implied.**

It is also **broader in kind**. The study named 7 allowlists. The inventory
finds **11 route-keyed constants**, including four the study missed —
`ROUTE_FIXED_MEASURE`, `_ROUTE_GRANULARITY`, `_ROUTE_TIME_GRAIN`
(`execution_receipt`), `_ROUTE_NOUN` (`chat_routing`) — plus `_RUN_SCOPED_ROUTES`
(`mi_service`), which decides **whether an answer may proceed** against a
run-scoped dataset. That last is a permission, not a label, and the study did
not have it.

### One property is already in the target shape

`KIND_GROUPING` is proven by `grouping_proven(facet, declared_axes, fields)`,
where `declared_axes` comes from `declared_group_fields(envelope, route)` —
which reads `metadata.groupedBy`, a **per-step declaration the route makes about
itself**. Measured:

```
grouping_proven(groupedBy=['geographic_region_obligor']) -> False
grouping_proven(groupedBy=['collateral_geography'])      -> True
```

It accepts or rejects on what was declared, not on who declared it. This is the
existence proof that the target property is reachable: one facet kind already
holds it.

### Smallest migration path (proposed, not taken)

Nothing was removed. The smallest path, in dependency order:

1. **Make the plan artefact a declaration channel.** A shadow plan already emits `declares_grouped_by`; publish it into the envelope the way `metadata.groupedBy` is published. No consumer changes.
2. **Move one facet kind at a time**, starting with the two that already read declarations (`KIND_GROUPING`, `KIND_POPULATION`), then the four allowlist-driven ones (`KIND_STATISTIC`, `KIND_SHARE`, `KIND_GRANULARITY`, comparison).
3. **Convert the four descriptive constants last** (`_ROUTE_LABELS`, `_ROUTE_NOUN`, `_ROUTE_TIME_GRAIN`, `_ROUTE_GRANULARITY`) — they are prose selection, lower risk, and touching them early would churn answer text for no governance gain.
4. **`_RUN_SCOPED_ROUTES` is separate work**, because it is a permission gate rather than a receipt claim, and it sits in `mi_service` rather than the receipt layer.

**Measurement of success (A3's threshold):** the decision-site count for a
property must fall to **zero** after that property's generalisation commit.
Anything above zero means it did not generalise.

---

## 5. Interpretation-contract gap report

**Not "none". One missing concept, and it blocks the first slice.**

Measured with `migration_phase0/probe_interpretation_gap.py`:

| population concept | contract carries it? | evidence |
|---|---|---|
| **seasoning segment** (front / back book) | **YES** | `PopulationClaim(concept='seasoning_segment', state='filled')` |
| **source-portfolio lens** (direct / acquired / SPV id) | **NO** | `population: []` for *"Summarise the acquired book"*, while `resolve_lens` returns `source_portfolio_type=acquired` |
| **row filter** (`balance > 150k`) | **half** | two claims, `('wording',)` and `('field','bound')`, `clause_id=None` |

**The missing concept:** a claim carrying the source-portfolio lens.
`mi_agent.portfolio_lens.resolve_lens(text)` is its only owner and it reads raw
question text. `PopulationClaim` exists and has a `concept` field; nothing
populates it for a lens.

**Why this blocks all nine cases, not just the lens-scoped ones.** An empty
`population` list cannot be read as *"Total"* while the lens is unrepresentable.
Absence of a claim is not evidence of no narrowing. A plan that assumed Total
would answer the whole book for *"Summarise the acquired book"* — the silent
population widening the P1L work exists to prevent.

**What was not done, deliberately:** no phrase list was added to the planner, the
contract was not extended on the fly, and `build_plan`'s signature carries no
question parameter — asserted structurally by `assert_no_question_read`.

---

## 6. `portfolio_summary` decomposition and shadow equivalence

### Primitives used — 5 of 7

`compare` and `project` are not needed.

| primitive | what it does here | existing implementation |
|---|---|---|
| stack periods | all governed snapshots; take the latest; disclose `periodCount` | `evolution.funded_frames` |
| select population | narrow to the portfolio lens | `evolution._scope_frame_lens` |
| resolve measure ×5 | `funded_balance` sum, `loan_count` count, `wa_ltv` weighted_avg, `wa_interest_rate` weighted_avg, `avg_borrower_age` avg | `assemble_funded_evolution` |
| group ×2 | balance by region; balance by source portfolio | `movement_summary._regional_exposure`, `_cohorts` |
| rank | largest first, truncated to `TOP_REGIONS` (8), 3 named in prose | `_regional_exposure`'s sort + head |

**One property worth recording:** the `rank` step truncates to top-8 **with no
residual**, so the regional list does not reconcile to the book. Shares are
computed against the full scope total, so the shares are right — the *list* is
partial by design. `funded_bridge` solves the same problem with an explicit
`"Other"` bucket. Phase 4 will have to reconcile the two residual policies.

### Equivalence result

`python -m migration_phase0.equivalence_portfolio_summary`

**9 cases on the surface. 0 economic differences.**

| scope | funded balance | loans | period |
|---|---|---|---|
| Total (A1–A6) | £1,964,886,258.21 | 11,035 | 2026-06 |
| Acquired (L1, L3) | £579,377,675.23 | 3,909 | 2026-06 |
| Direct (L2) | £1,385,508,582.98 | 7,126 | 2026-06 |

Compared field by field: `available`, `period`, `reportingDate`, `periodCount`,
`regionColumn`, all five metrics, every `topRegions` entry (`region`, `balance`,
`share`) and the cohort set. **Zero differences on every case.**

**The surface is verified, not assumed.** Each case is checked against
`chat_routing._is_portfolio_summary` before comparison. Two questions that look
like portfolio summaries are listed deliberately and **excluded** by that check:
*"Summarise the front book"* (a seasoning population — answered by the
point-in-time path at 1,177 loans, not by this route) and *"What is the
portfolio position for the direct book?"*. Comparing on a question the route
does not own would have manufactured an equivalence that means nothing.

### And the plan cannot be built

**9 of 9 cases BLOCKED** at `select_population`, for the reason in §5.

Where a case is blocked, the lens was supplied to the shadow executor **from
outside**, for measurement only, and the result records it as
`lensFiltersSuppliedExternally`. **Identical economics on a blocked case prove
the composition; they do not prove the plan could be built.** The distinction is
the whole result of this task.

### Observed differences

**Economic: none.** **Structural: one, and it is the blocker.** No other
difference was observed in `available`, periods, metrics, regions, shares or
cohorts.

Not compared, and stated rather than glossed: **answer text**, because the
baseline does not record `portfolio_summary` prose as stable, and **receipt
facets**, because the shadow path produces no envelope — it is a plan and a
result, not a route. Both are work for the conversion commit, not for a shadow.

---

## 7. Go / no-go on switching `portfolio_summary`

# NO-GO.

Not because the composition is wrong — it reproduces the shipped economics
exactly on every case, at every scope. **Because the plan cannot be constructed
from the interpretation contract**, and the only ways to switch it today are:

* **reread the raw question** for the lens — forbidden by the sprint's
  interpretation rule, and abort condition A4's first case;
* **assume `Total` from an empty population claim** — a silent population
  widening on *"Summarise the acquired book"*;
* **pass the lens in from outside the plan** — which is what the equivalence
  harness does for measurement and is not a production path.

**What would clear it:** one deliberate, tested extension to
`question_interpretation` — a claim carrying the source-portfolio lens, with
`mi_agent.portfolio_lens` as its single owner supplying it, and the projection
populating it. That is a contract change with its own before/after measurement,
and it is a **Phase 1 item**, not something to improvise inside a route
conversion.

**A clean no-go is the successful outcome here.** The slice was scoped to
determine whether the first migration can be achieved under the safety
constraints. It cannot, yet, and the reason is specific, measured and fixable.

---

## 8. Measured effort

| | |
|---|---|
| **production modules changed** | **0** |
| files added | 17 — 8 Python instruments/shadow, 6 JSON surface records, 2 docs, 1 test module |
| tests added | 7 (2 controls passing, 5 `xfail(strict=True)`) |
| tests changed | 0 |
| baselines updated to make something green | **0** |
| commits | 3, separable: baseline+aborts, governance prep, shadow plan |

### Production dependencies discovered

* `portfolio_summary` is **not** self-contained: it reads `evolution.funded_frames`, `evolution._scope_frame_lens`, `evolution.assemble_funded_evolution`, and four `movement_summary` helpers. Converting it touches the same `evolution` module Phase 5 converts — the routes are **not** independent, and the blast-radius ordering does not capture that.
* `_route_portfolio_summary` returns `None` to defer when the summary is unavailable, so the fall-through is proven and a conversion can keep it.
* **`_ROUTE_LABELS` has no entry for `portfolio_summary`** — consistent with the study's observation that it is the only route with zero receipt literals.

### Signals the migration is more expensive than the study suggested

**Three, stated plainly:**

1. **The route-identity coupling is broader in kind than counted.** Four constants the study missed, one of which (`_RUN_SCOPED_ROUTES`) is a permission gate rather than a receipt label. The *site count* is smaller (14, not 54); the *kinds* to generalise are more numerous.
2. **The arity defect is three defects.** The study found the `_execute_grouped` guard. Two more of the same family were found here, and one of them (`_execute_grouped_measure_set`) never discloses at any arity — so "generalise the arity-1 rule" understates it.
3. **The first slice did not reach a conversion at all.** The study's blast-radius ordering scored `portfolio_summary` cheapest on tests, handler size and receipt literals. None of those predicted the actual blocker, which is an interpretation-contract gap. **Blast radius did not predict conversion cost, and the migration order should not be trusted to.**

### Signals it is cheaper than the study suggested

Stated for balance, because they are equally real:

* **The economics reconciled first time**, on 9 cases and 3 scopes, with no bespoke exceptions and no new primitive.
* **The governance coupling is 14 sites, not 54.**
* **One facet kind already holds the target property** — `grouping_proven` proves from declaration, not identity.

---

## 9. What ships

**Nothing.** No production module changed. T3–T7 closed. The shadow path is not
wired. The arity-2 disclosure correction is **not** authorised by this task and
was not made — it changes a shipped artifact's columns and needs its own
authorisation and its own measurement.
