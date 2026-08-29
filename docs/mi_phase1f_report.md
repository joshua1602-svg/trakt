# Phase 1F — `portfolio_summary` production conversion

# STOP — PREREQUISITE

**`portfolio_summary` was not converted. No production module was changed.**

The prerequisite is the one Phase 1B stopped on. It is still open, it is now
**measured across the whole owned surface rather than inferred from two cases**,
and the surface it applies to has doubled since Phase 1B saw it.

**Named precisely:**

> The interpretation contract carries **which** source scope the owner resolved.
> It does not carry **whether the question named one** — and that second fact is
> what decides precedence against a caller-supplied `source_portfolio_lens`.
> A compositional plan reading only `source_scope` widens the population on
> **14 of 54** (question, caller default) combinations the route owns.

Base: `a264754` (Phase 1E). Instruments added, production untouched.

---

## 1. Pre-switch route surface — 18 owned cases, verified

`python -m migration_phase0.route_ownership_portfolio_summary`

Ownership is asked of the shipped recogniser (`chat_routing._is_portfolio_summary`),
never assumed from a list. The candidate bank deliberately includes questions
expected **not** to be owned, because a bank containing only owned questions
cannot detect ownership drift.

**18 of 23 candidates claimed.** Phase 0 measured 9; the other 9 arrived with
Phases 1C and 1E and had never been on this surface before.

| case | question | contract claim | production, no dropdown |
|---|---|---|---|
| A1–A6 | *portfolio summary* and five paraphrases | `filled/total` | 11,035 / £1.96bn |
| L1, L3 | *the acquired book* | `filled/acquired` | 3,909 / £579.4m |
| L2 | *the direct book* | `filled/direct` | 7,126 / £1.39bn |
| P1 | *across all portfolios* | `filled/total` | 11,035 / £1.96bn |
| P2 | *excluding the acquired book* | `filled/total` | controlled refusal |
| N1 | *the ALP Origination Book* | `filled/cohort=alp_origination` | 7,126 / £1.39bn |
| N2 | *the alp_acquired book* | `filled/cohort=alp_acquired` | 3,909 / £579.4m |
| N3 | *the ALP Acquired Back Book* | `filled/cohort=alp_acquired` | controlled refusal (§7) |
| Z1 | *the spv1_sponsored portfolio* | `filled/cohort=spv1_sponsored` | governed no-data outcome |
| U1 | *the acquired_001 book* | `unresolvable` | controlled refusal |
| U2 | *the Highgate Mortgages Book* | `unresolvable` | controlled refusal |
| F1 | *the funded book* | `filled/total` | 11,035 / £1.96bn |

**Correctly excluded (5).** `X1` *the front book* (a seasoning population),
`X2` *portfolio position for the direct book*, `X3` *summarise the portfolio by
region* (a stratification), `X4` *what changed since last month* (routes to
`period_movement`), `X5` *for the London book…* (a governed value of
`collateral_geography`, not a portfolio). Comparing on a question the route does
not own manufactures an equivalence that means nothing — Phase 0's rule, kept.

## 2. The exact production switch point

| # | | location |
|---|---|---|
| 1 | recognition completes | `chat_routing.py:2886` registry entry → `_is_portfolio_summary` (`:320`) |
| 2 | interpretation available | **nowhere on this path — see §4c** |
| 3 | route-specific calculation begins | `_route_portfolio_summary` (`:450`) → `_resolve_lens` (`:454`), then `movement_summary.portfolio_summary` (`:455–457`) |
| 4 | where the plan replaces it | **that one call**, `:455–457`. Everything after — prose, KPI/chart/table artifacts, envelope — is shape, not economics |
| 5 | result → envelope | `_envelope(..., route="portfolio_summary")` (`:564`) |
| 6 | receipt / governance | downstream in `mi_service.py:644–674` (`reconcile_routed_facets`, `build_routed_receipt`), then the two Phase 1E guards |
| 7 | fall-through | `:458` — Phase 1E's narrowing branch first, then `return None` to defer |

The switch is genuinely one call. **The blocker is in the input to that call,
not the call itself** — unchanged from Phase 1B, and that is the point: nothing
about the switch has got harder, and nothing about it has got easier.

## 3. The blocker, measured

`python -m migration_phase0.contract_sufficiency_portfolio_summary`

Per (question, caller default), comparing what production resolves
(`chat_routing._resolve_lens`) against what a contract-only plan would select
(`shadow_portfolio_summary.build_plan`, whose signature carries no question — the
structural guarantee `assert_no_question_read` exists to protect).

**34 of 54 combinations plan correctly. 14 WIDEN. 6 block.**

### The 14

| cases | caller default | contract plans | production ships |
|---|---|---|---|
| A1–A6, **F1** | `acquired` | `total` | `acquired` |
| A1–A6, **F1** | `direct` | `total` | `direct` |

On A1 with the workspace scoped to Acquired, a contract-only plan answers
**11,035 loans / £1,964,886,258.21** where production answers **3,909 /
£579,377,675.23**. The gap is **£1,385,508,582.98** — abort condition **A2**'s
threshold is £0.005.

### Why the contract cannot fix it by looking harder

```
A1  "Please provide a portfolio summary"        claim: filled / total / narrows=False
P1  "portfolio summary across all portfolios"   claim: filled / total / narrows=False
```

`as_dict()` on the two is **equal**. Under `default=acquired`, production
answers A1 over Acquired and P1 over the whole book. **Identical claims,
opposite required populations.** No amount of reading `source_scope` more
carefully separates them, because the distinguishing fact was never written
down.

The fact production uses is `portfolio_lens.mentions_portfolio(question)` —
since Phase 1E, `mentions_portfolio(text) or names_governed_portfolio(text,
registry)`. Both are the owner's readings of the raw question, and the contract
carries neither.

### The 6 blocked cases are not defects

U1/U2 (`acquired_001`, *Highgate Mortgages Book*) reach the plan as
`state=unresolvable`, and `build_plan` emits a **blocked** `select_population`
step — which by Phase 0's design is a refusal, not an answer with the step
quietly omitted. **Production also refuses.** The outcomes agree, so this is the
contract working. It is recorded separately only because §12 asks literally for
"route-owned cases construct from contract: 100%", and a deliberate refusal is
a construction outcome rather than a construction success.

### Why this is a STOP and not an abort

Abort condition **A4** distinguishes two cases and only one aborts:

* the contract owns a decision and the plan rereads the question anyway → **ABORT**;
* the contract **cannot represent** the decision → **STOP AND REPORT, extend deliberately**.

This is the second. Phase 0 §7 says what to do with it, and says it about this
exact gap: *"a contract change with its own before/after measurement, and it is
a Phase 1 item, not something to improvise inside a route conversion."*

The three ways past it today are the same three Phase 0 named, and all three are
forbidden by this task's own gate:

| workaround | forbidden by |
|---|---|
| reread the raw question in the plan | §12 "downstream raw-question semantic reads: 0"; A4 case 1 |
| pass the lens in from the route | §12 "external lens injection: 0"; Phase 0's `lensFiltersSuppliedExternally` |
| read the empty/`total` claim as Total | the 14 widenings above |

## 4. Two further prerequisites found, and one finding

### 4a. The plan selects a RAW type filter, not the governed id list

`shadow_portfolio_summary.lens_for` rebuilds the lens with
`lens_from_selection(scope)`:

| plan scope | plan rebuilds | production resolves |
|---|---|---|
| `acquired` | `{'source_portfolio_type': 'acquired'}` | `{'source_portfolio_id': ['alp_acquired']}` |
| `direct` | `{'source_portfolio_type': 'direct'}` | `{'source_portfolio_id': ['alp_origination']}` |

On the shipped book these select **identical rows**, because it holds exactly
one portfolio per type. Phase 1C measured the divergence on a two-portfolio
fixture: **governed £300.00 against raw £1,200.00**. So no economic check on
this book can catch it, and a conversion that carried `lens_for` unchanged would
ship a governed-population defect that only appears on the next onboarded
portfolio.

This is not a contract prerequisite — it is a correction the conversion commit
must make, in the instrument rather than in production. Recorded so it is made
deliberately rather than discovered afterwards. §8's requirement — *positive
governed population evidence before applying a narrowed scope* — is exactly what
the governed path provides and the raw path does not.

Related, and weaker than its docstring claims: `lens_for`'s guard compares
`lens.name != scope`, and `lens_from_selection('nonesuch_book')` returns a
**cohort** lens for a book the registry does not hold. `cohort == cohort`, so the
guard passes. It is unreachable today (the contract blocks those cases first)
but it is not the protection it is documented to be.

### 4b. The routed path constructs no interpretation at all

§4 asks where interpretation is already available. On this route: **nowhere**.

The single production construction site is `mi_agent/mi_agent_workflow.py:1003`,
on the **point-in-time** path, where its own comment states the position
plainly: *"Nothing reads it: it is carried so Stage 3 can convert consumers onto
it one at a time."* A routed question never reaches
`run_mi_agent_query`, so `portfolio_summary` never builds a
`QuestionInterpretation`.

That site also calls `from_parts(...)` **without** `registry=`, so even there the
`source_scope` claim is the pre-1E reading — named portfolios unresolved,
unheld names indistinguishable from Total.

A conversion therefore needs a construction site on the routed path, built with
the registry. Small, but it is production wiring that does not exist today and
must not be improvised inside the switch.

### 4c. A finding: the whole-book vocabulary is uneven

Two explicit whole-book phrasings, opposite precedence:

| question | in `_TOTAL_TERMS`? | with the workspace scoped to Acquired |
|---|---|---|
| *portfolio summary across all portfolios* | yes | **11,035** — the question wins |
| ***Summarise the funded book*** | **no** | **3,909** — the dropdown wins |

Under this phase's stated semantics, *Funded Book* **is** the complete funded
population, Direct and Acquired together. The route owns the question, and with
a dropdown selection it answers 3,909 of 11,035 loans while the contract records
the same `filled/total` claim it records for a question that named no scope at
all.

**Reported, not fixed.** Adding a term to `_TOTAL_TERMS` changes which
population a shipped question answers over — a user-visible product decision
needing its own authorisation, exactly like the arity disclosure defect. §14's
"do not add new parser vocabulary" also forbids it here. It belongs with the
precedence ruling, because both are answers to the same question: what counts as
the question speaking to scope.

## 5. Deliverables the conversion could not produce

Stated rather than glossed, and each with what *is* known:

| asked | status |
|---|---|
| old-vs-new economic comparison | **not produced** — no new path exists. Phase 0's shadow comparison stands: 9 cases, 3 scopes, **0 economic differences**, no bespoke exception, no new primitive. It was measured with the lens supplied externally and therefore proves the composition, not the plan. |
| population-resolution comparison | **produced** — §3 and §4a. 34/54 match, 14 widen, and the raw-vs-governed filter difference is measured. |
| payload / receipt comparison | **not produced.** Phase 0 could not compare receipts because a shadow emits no envelope; that is still true. Nothing here says receipt equivalence is hard — only that it is unmeasured. |
| answer-text differences | **not produced**, and would have been the last thing to check, not the first. |
| route-conversion cost | **0 production lines, 0 production modules, 0 conversion commits.** *m* and *c* for A1 remain **unknown** after two attempts. |

## 6. Regression gate — nothing moved, because nothing production changed

`git diff a264754 --stat` outside `docs/`, `migration_phase0/` and `tests/`
returns nothing.

| surface | pre-switch | now |
|---|---|---|
| calibration bank | 267 passed | unchanged |
| robustness 44 | 32 / 6 / 4 / 2 | unchanged |
| — seasoning by name | **Q1 4 · Q7 4 · Q8 12 CORRECT** | unchanged |
| shipped shapes | 15 correct, **0 wrong** | unchanged |
| routed surface | 31 passed, `rt_004` known-failing | unchanged |
| recognition (61 phrasings) | 15 / 7 / 10 / 29 | unchanged |
| time-series surface | **silent drops 0** | **silent drops 0** |
| estate suites | 8 pre-existing failures, by name | unchanged |

**Introduced failing names: 0. Silent drops: 0. Baselines updated to absorb
changed behaviour: 0.**

The eight, all pre-existing and all previously named:
`test_mi_predicate_extraction::test_complex_query_executes_all_filters`,
`test_mi_trust_hardening::test_C_joint_borrowers_count_and_balance`,
`test_p0_execution_receipt` ×3,
`test_parser_cost_hardening::test_layered_question_routes_to_llm_even_when_deterministic_parses`,
`test_p0_time_axis_request::test_the_wording_that_asked_is_returned[balance by each month]`,
and `test_analytical_capability_layer::TestSecondBookAcceptance::test_q7_compares_the_two_governed_sides_and_reconciles`
— the last being **known failure 1** in the abort-conditions document, verified
there to fail identically on a clean tree at `42cef00`. It appears in this run
and not in Phase 1E's because this selection includes its suite.

New tests: `tests/test_portfolio_summary_conversion_prerequisites.py` — 27
passed, 2 `xfail(strict=True)`. The two declared failures are the prerequisite
(§3) and the raw-vs-governed filter (§4a); both announce themselves the moment
someone closes the gap.

## 7. What would clear this

One contract extension, whose design Phase 1C settled before stopping and which
this phase's evidence does not change:

* extend the **existing** `SourceScopeClaim` rather than adding a parallel claim family;
* add a provenance value distinguishing **stated by the question** / **supplied by the caller** / **no narrowing stated** / **unresolved**;
* `mi_agent.portfolio_lens` stays the single owner — the projection carries `mentions_portfolio` / `names_governed_portfolio` alongside `resolve_lens`, both from the owner, with no phrase list introduced anywhere else.

Phase 1C held this pending a ruling on the Case D defect. **Phase 1E closed Case
D** — an unheld scope is now `UNRESOLVABLE` in the contract and a controlled
refusal in production — so the ruling that blocked it has been made, and the
extension is unblocked.

Two smaller items travel with it: the routed-path construction site (§4b) and
the governed-vs-raw filter (§4a).

## 8. Should `temporal_compare` proceed as conversion 2?

**No — not yet, and not because of `temporal_compare`.**

`portfolio_summary` was chosen first because the study scored it cheapest. It
has now failed to convert **twice**, both times on the interpretation contract
rather than on anything about the route. `temporal_compare` consumes the same
contract and the same portfolio lens, so it inherits this prerequisite whole and
would stop at the same place — and it carries a second scope axis (two periods)
that `portfolio_summary` does not.

Converting a harder route to avoid a blocker that applies to both would replace
a measured stop with an unmeasured one. **Close the contract prerequisite, then
retry `portfolio_summary`** — it remains the cheapest route to convert, and it
is the only one whose economics are already proven to reconcile exactly.

Phase 0 recorded the lesson this repeats: *"Blast radius did not predict
conversion cost, and the migration order should not be trusted to."* Two failed
attempts on the cheapest route say the same thing again. **A1's *m* and *c* are
still unknown, so no conversion cost has been established, and none should be
quoted.**

---

## Final status

# STOP — PREREQUISITE

**The interpretation contract cannot represent whether a question named a source
scope**, which is the fact that decides precedence against a caller-supplied
`source_portfolio_lens`. 14 of 54 owned (question, default) combinations would
silently widen; the largest single divergence is £1,385,508,582.98 against an
A2 threshold of £0.005.

No abort condition fired. No production module changed. No baseline moved.
