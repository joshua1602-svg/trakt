# Target-state contract closure

# READY FOR SYSTEMATIC MIGRATION

The generic funded-book estate was inspected **as an estate**, not one route at
a time. Six concepts were still being decided downstream of interpretation; two
were generic gaps and are closed here; four are accounted for by name. The
contract now carries every semantic fact the shipped generic routes need,
with one route-specific exception that is recorded rather than closed.

Base: `44bc90c` (Phase 1G). **157 production lines, 3 files.** Nothing
client-visible changed.

---

## 1. Semantic concepts the funded-book estate requires

Derived from shipped behaviour, not from a wish list. Fourteen routes, split by
whether the seven primitives claim them.

**Generic funded-book estate (9):** `portfolio_summary`, `period_movement`,
`temporal_compare`, `evolution`, `funded_bridge`, `geo_exposure`,
`period_change_analysis`, `portfolio_risk_comparison`, `concentration_analysis`.

**Specialist, deliberately outside the compositional contract (6):**
`scenario`, `cohort_conversion`, `forecast_extrapolation`,
`cohort_progression`, `risk_limits`, `concentration_tests`.

The concepts the generic estate needs:

| concept | owner | in the contract |
|---|---|---|
| measure / subject | parser | `subject`, `operation` |
| statistic / aggregation | parser | `operation.type` |
| filters | parser + facet layer | `filters` |
| grouping dimensions | receipt layer | `dimensions[].role` |
| time grain | `period_request` | `time.grain` |
| **time window magnitude** | `period_request.requested_span` | **added here** |
| comparison period | parser | `time.comparison_period` |
| movement | parser | `operation.type = movement` |
| ranking | parser + facet layer | `operation.type = ranking` |
| source population — Funded / Direct / Acquired | `portfolio_lens` | `source_scope.base_population` |
| governed portfolio / SPV ids | registry via `portfolio_lens` | `source_scope.portfolio_ids` |
| cohort / vintage | parser | `population[concept=cohort_vintage]` |
| explicitness / provenance | `portfolio_lens` | `source_scope.provenance` |
| caller / UI context | request | folded into provenance |
| unresolved / ambiguous | `portfolio_lens` | `state=UNRESOLVABLE` |
| **dataset — funded / pipeline / forecast** | `workspace.resolve_active_view` | **added here** |
| comparison **sides** | `mi_workflows.portfolio_risk_comparison` | **no — §5** |

## 2. Contract-coverage matrix

`python -m migration_phase0.semantic_owner_inventory`

| route | semantic decisions still made downstream | in contract? | raw-text reread? | blocker |
|---|---|---|---|---|
| `portfolio_summary` | source scope + precedence | **yes** | yes — a duplicate | no |
| `period_movement` | source scope + precedence; time window | **yes** (window added here) | yes ×2 | no |
| `funded_bridge` | source scope + precedence | **yes** | yes | no |
| `geo_exposure` | source scope + precedence | **yes** | yes | no |
| `evolution` | dataset selection | **yes** (added here) | yes | no |
| `temporal_compare` | dataset selection | **yes** (added here) | yes | no |
| `period_change_analysis` | source scope; time window; **ranking subject + direction**; route shape | ranking **no** | yes ×4 | ranking |
| `portfolio_risk_comparison` | **whole-question delegation to a workflow** | comparison sides **no** | wholesale | yes |
| `concentration_analysis` | **whole-question delegation to a workflow** | n/a | wholesale | yes |

Specialist routes: **all six clean** at the handler boundary — the question
reaches them for prose and audit, not for re-interpretation.

## 3. Downstream semantic-owner inventory

**13 decisions, 6 distinct concepts**, inside the generic path:

| concept | routes | class | disposition |
|---|---|---|---|
| source scope + caller precedence | 5 | **B** duplicate | contract carries it since 1G — the conversion deletes the duplicate |
| dataset selection | 2 | **B** duplicate | **closed here** |
| time window | 2 | **B** duplicate | **closed here** |
| ranking subject + direction | 1 | **B** | route-specific → left for migration (§7) |
| route shape (recognition) | 1 | **B** | a route's own claim decision, not a plan input |
| whole-question delegation | 2 | **B** structural | those two routes are not compositional today — §7 below |

### The instrument under-reported twice, and both are recorded

This inventory was wrong twice before it was right, and the corrections are the
most useful thing in it:

1. **Its default bucket was benign.** Anything not on the semantic list landed
   in `A_EXECUTION`. That hid `_dataset_for` and both whole-question
   delegations — all three read as execution at the call site. The default is
   now `UNCLASSIFIED` and the instrument prints a loud block until every site
   is judged.
2. **It scanned one module.** `period_change_analysis`'s handler lives in
   `period_change_route.py`, so a whole generic route was missing — and with it
   the ranking concept. **A route whose handler lives elsewhere is not a route
   that decides less.**

Both are why this task was worth doing before another conversion: each would
have been discovered mid-conversion, which is exactly the pattern the programme
has been paying for.

## 4. Target-state minimum contract

No parallel model. Two additions to existing claim types.

### `TimeClaim.window_periods` / `window_governed`

| | |
|---|---|
| owner | `mi_agent.period_request.requested_span` |
| representation | `int` periods, plus a `governed` flag |
| states | `None` (no window named) or ≥ 1 |
| provenance | the flag: a question that named a countable span vs a vague recency a convention settled |
| consumers | `period_movement`, `period_change_analysis` |
| validation | a window shorter than one period is refused by the schema |

`trend_window` already said a window was named; it did not say **which one**, so
both consumers asked the owner again.

### `DatasetClaim` — `dataset` + `provenance`

| | |
|---|---|
| owner | `mi_agent_api.workspace.view_named_by_question` (extracted from `resolve_active_view`) |
| representation | `funded` / `pipeline` / `forecast` |
| states | `FILLED` / `UNRESOLVABLE` |
| provenance | `explicit_user` / `caller_context` / `default` — **`source_scope`'s vocabulary, reused** |
| consumers | `evolution`, `temporal_compare`, and the point-in-time path |
| validation | a filled claim must name a dataset; an unknown dataset is refused |

**A different axis from `source_scope`.** A question picks a *tape* and,
separately, a *portfolio scope* within it. Conflating them is how *"the balance
by seasoning segment excluding pipeline cases"* reached a route with
`dataset='pipeline'` — narrowed to the very thing it excluded.

**The owner stays single.** `view_named_by_question` was **extracted** from
`resolve_active_view`, which now calls it; it was not copied beside it. A second
vocabulary is the defect B21 fixed, and a test asserts the resolver contains no
second copy.

## 5. What was NOT closed, and why

**Comparison sides.** Which two governed populations are being compared. Owned
inside `mi_workflows.portfolio_risk_comparison`, which is one of the two routes
that hand the whole sentence to a workflow. **One route needs it**, so §7's rule
applies: a route-specific change is recorded and left for migration.

**Ranking subject + direction.** `period_change_route.resolve_rank_intent`.
Also one route. Same disposition.

**Two routes are not compositional today.** `portfolio_risk_comparison` and
`concentration_analysis` pass the raw question into workflows that run their own
recognition predicates and rejection rules. That is not a missing contract
*field* — it is a structural fact about those two routes, and forcing their
semantics into the generic contract is what §3 forbids. **They are not in the
first migration wave.**

## 6. Contract bank — by combination, not by route

`python -m migration_phase0.contract_bank` — **20 of 21 combinations fully
representable**, the exception being comparison sides above.

The bank found five gaps on its first run and **three were its own fault**,
which is worth stating because it is the same discipline the inventory needed:

* *"acquired balance by month"* — it required a grouping dimension; **"by month"
  is a time grain**. A required-facts list read off intuition rather than off
  what the question asks.
* *"How has balance moved by region?"* and *"…moved this year?"* — the contract
  records `operation=amount`, not `movement`, because the deterministic parser
  does not read that phrasing as a movement. **A recognition limit upstream, and
  the contract faithfully carries what the owner said.** It carries `movement`
  for the bridge and compare phrasings that do set it.
* *"Which regions grew the most?"* — `operation=ranking` **is** carried; the
  dimension's *role* is `unresolved` with a stated reason, which is the
  pre-registered Stage-1 finding and is the contract correctly saying "I do not
  know" rather than guessing.

## 7. Migration entry criteria

| # | criterion | status |
|---|---|---|
| 1 | every generic semantic concept is representable | **yes**, except comparison sides — one route, recorded |
| 2 | no generic planner needs raw question text | **yes** — `build_plan` has no question parameter; `assert_no_question_read` holds |
| 3 | governed portfolio identity is contract-driven | **yes** — registry ids in `portfolio_ids`, 1G |
| 4 | Funded / Direct / Acquired / named portfolio are deterministic | **yes** — 54/54 precedence cells |
| 5 | unresolved concepts fail closed | **yes** — `UNRESOLVABLE`, 0 widenings |
| 6 | provenance available wherever precedence depends on it | **yes** — scope and dataset both |
| 7 | specialist routes explicitly out of scope | **yes** — 6 named, all clean at the boundary |
| 8 | regression baseline clean by name | **yes** — §10 |

## 8. Revised migration order

Ranked on this task's evidence, not on the study's blast radius.

| route | contract | execution reuse | governance | local semantics left | verdict |
|---|---|---|---|---|---|
| **1. `portfolio_summary`** | complete | 5 primitives, all existing | no receipt literals | **1** (scope, carried) | **first** |
| **2. `period_movement`** | complete | same `movement_summary` module | low | **2** (scope + window, both carried) | second |
| **3. `geo_exposure`** | complete | `geo` + lens | low | **1** (scope, carried) | third |
| 4. `funded_bridge` | complete | own bridge module | rank residual differs | 1 | later |
| 5. `evolution` | complete | shared with 3 routes | run-scoped | 1 (dataset, carried) | later |
| 6. `temporal_compare` | complete | shares `evolution` | run-scoped | 1 (dataset, carried) | later |
| 7. `period_change_analysis` | **ranking missing** | own workflow | own envelope | 4 | after ranking |
| 8–9. comparison / concentration | **sides missing** | own workflows | own briefs | wholesale | not compositional today |

**`portfolio_summary` is still first, and the evidence is stronger than the
study's was.** Not because it was scored cheapest — that prediction failed twice
— but because it is now the only route where every input is measured: its
economics reconcile exactly (9 cases, 0 differences, 5 existing primitives, no
new primitive), its populations match on rows through the governed path, its
plan constructs from the contract with 0 external injection, and it has zero
receipt literals to generalise.

**`period_movement` is second and is new to the top three.** It shares
`movement_summary` with `portfolio_summary`, so the second conversion reuses the
first's execution work; both its downstream semantic reads are now carried; and
it is the cheapest way to test whether a *second* route costs less than the
first, which is what A1 needs.

**`temporal_compare` is demoted from second to sixth.** It shares `evolution`'s
module with three other routes, so converting it commits to the shared
`evolution` machinery Phase 5 also converts — Phase 0 recorded that dependency
and the original order ignored it.

## 9. Migration economics, re-baselined

### The split

**Product hardening — valuable regardless of composition (873 production lines):**

| phase | lines | what |
|---|---|---|
| 1A | packaging | the App Service failed to start; found by the migration, not caused by it |
| 1E | 515 | registry-aware portfolio identity; unknown-scope refusal made route-independent; a governed portfolio with zero rows no longer answers whole-book |
| 1G | 363 | Funded/Direct/Acquired semantics; scope provenance; a workspace selection of a real portfolio no longer resolving to the whole book |

Every one of these was a **live correctness defect on shipped answers**,
reachable by a user today, and none needed a compositional plan to be worth
fixing. `acquired_001` returning 11,035 loans under one portfolio's name was a
wrong answer with `ok=True` and no disclosure.

**Compositional migration cost so far: 157 lines, all of it contract work, and
zero route conversions.**

### Is it still economically sensible?

**Yes, but the justification has changed and should be stated honestly.**

The programme has so far bought **product correctness**, not composition. Six
phases produced no converted route. If the value were solely "fewer duplicated
shape decisions", the return would not yet justify the spend.

What makes it sensible now is that the *discovery* phase is over and measured:
the estate has **6 downstream semantic concepts, not an unknown number**, four
are carried and two are named and bounded. The next conversion is the first that
starts from a complete map rather than from a hypothesis.

### Cost of the next three

Evidence-based, and each is a **prediction to be scored**:

| route | production lines | commits to equivalence | basis |
|---|---|---|---|
| `portfolio_summary` | 40–80 | 2–3 | one call replaced; envelope and receipt unchanged; the plan already exists and reconciles |
| `period_movement` | 30–60 | 1–2 | same module, contract complete, execution reuse from conversion 1 |
| `geo_exposure` | 30–60 | 1–2 | one lens read, one existing `geo` call |

**No calendar estimate is offered.** No conversion has been measured, so any
date would be an invention — the same discipline A1 was written under.

### Assumptions now invalid

1. **Blast radius predicts conversion cost.** Failed twice on the route it
   scored cheapest. Contract completeness predicts it; blast radius does not.
2. **Routes are independent.** `evolution` is shared by four; `movement_summary`
   by two. Order must follow shared execution, not test counts.
3. **"54 route-name literals" was the coupling.** It is 14 branching sites — but
   the real coupling was never route names, it was **semantic decisions**, of
   which the study counted none.
4. **The interpretation contract was nearly complete.** It was missing the
   source-portfolio lens (1A), its provenance (1G), governed identity (1E/1G),
   the time-window magnitude and the dataset axis (here).

### Assumptions strengthened

1. **The economics reconcile.** 9 cases, 3 scopes, 0 differences, no bespoke
   exception, no new primitive — and now on the *governed* population path.
2. **The primitives are sufficient.** 5 of 7 for `portfolio_summary`; nothing in
   this estate needed an eighth.
3. **Governance can generalise from declaration.** `grouping_proven` still holds;
   the two new claims carry provenance the same way.
4. **The contract is the right boundary.** Every blocker found since Phase 0 has
   been a missing *contract field*, not a missing capability — and each has been
   closable in tens of lines.

## 10. Abort conditions

| | condition | verdict | evidence |
|---|---|---|---|
| **A1** | cost explosion | **NOT YET MEASURABLE** | *m* and *c* need three conversions; **zero** have happened. No figure may be quoted. |
| **A2** | failure to reconcile | **NOT FIRED** | 0 economic differences, 0 bespoke exceptions, 0 new primitives, on every measurement including the governed-population rerun |
| **A3** | governance cannot generalise | **NOT YET MEASURABLE** | its threshold is the decision-site count falling to zero *after a property's generalisation commit*; no such commit has been made. The count is unchanged at 14. |
| **A4** | interpretation ownership | **NOT FIRED** | fired twice as its *second* case (contract cannot represent), which is the prescribed path, not an abort. Both extensions were made deliberately and tested. No plan rereads the question. |
| **A5** | unattributable regression | **NOT FIRED** | every movement across 1E–1G and this task is attributable by name; A5's absolute property — silent drops 0 — has held at every measurement |

**No abort condition has fired.** Two are not yet measurable, and saying so is
the honest answer rather than scoring them favourably.

## 11. Regression

Production changed in three files, so every registered gate was rerun.

| surface | result |
|---|---|
| calibration bank | unchanged |
| robustness 44 | **byte-identical** — 32 / 6 / 4 / 2 |
| — seasoning by name | **Q1 4 · Q7 4 · Q8 12 CORRECT** |
| shipped shapes | **byte-identical** — 15 correct, 0 wrong |
| routed surface | **byte-identical** — 31 passed, `rt_004` known-failing |
| recognition (61 phrasings) | **byte-identical** — 15 / 7 / 10 / 29 |
| time-series surface | **byte-identical** — **silent drops 0** |
| `portfolio_summary` shadow | 9 cases, **0 economic differences**, 0 external injection |
| Phase 1F matrix | 48/54, **0 widenings** |
| precedence matrix | **54/54** |
| contract bank | **20/21** |

Suites: **9 failed, 2101 passed, 3 skipped, 15 xfailed.** Eight are the
pre-existing failures recorded since Phase 1F, by name. The ninth,
`test_schema::test_as_dict_round_trips_every_slot`, is the contract's own
completeness guard correctly detecting the new `dataset` claim — it is updated
with the reason recorded in it, which is what the guard exists for.

**Introduced failing names: 0. Silent drops: 0. Unexplained economic movement:
0.**

## 12. Recommended next task

**Convert `portfolio_summary`** — the Phase 1F task, re-run against a contract
that now carries everything it needs.

Its gates are already green in shadow. What remains genuinely unmeasured is what
a shadow structurally cannot measure — **payload and receipt equivalence** — and
that should be the conversion's first work, not its last.

Two things to carry in:

* record production lines and commits-to-equivalence as they happen; they are
  the first data point for A1's *m* and *c*, which are still unknown;
* the conversion is expected to **delete** the duplicate `_resolve_lens` call,
  not merely stop using it — a duplicate left in place is a second owner that
  will drift.
