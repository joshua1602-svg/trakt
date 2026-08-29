# Phase 1G — source-scope provenance and scalable portfolio selection

# GO-READY — SOURCE SCOPE + PORTFOLIO SELECTION COMPLETE

`portfolio_summary` was **not** switched. The prerequisite Phase 1F stopped on
is closed, measured on the surface that exposed it, and `portfolio_summary` can
be retried for production conversion in the next task.

Base: `918d3c4` (Phase 1F). Five production files, 375 lines.

---

## 1. The final scope model

Three independent facts, where there was previously one conflated value:

```
base_population   funded | direct | acquired      the broad business population
portfolio_ids     ()  |  (governed registry ids)  the specific selection
provenance        explicit_user | caller_context | default | unresolved
```

Read together:

| request | base_population | portfolio_ids | means |
|---|---|---|---|
| *the funded book* | `funded` | `()` | the complete funded population, **unrestricted** |
| *the direct book* | `direct` | every direct id | the category, as the registry currently defines it |
| *the acquired book* | `acquired` | every acquired id | likewise — all of them |
| *SPV2* | **`funded`** | `('spv2',)` | one governed portfolio |

**`SPV2` is deliberately not `base=acquired`.** Which category a named portfolio
belongs to is a property of the *portfolio*, held by the registry, not of the
*request*. On the phase's fixture, reading SPV1 as its category overstates it
from £4.00 to £234.00.

**`funded` carries no ids on purpose.** The complete funded population is
unrestricted, not an enumeration: listing today's members would silently exclude
a portfolio onboarded tomorrow from a question that asked for the whole book.

There is **no hierarchy**. `Funded → Acquired → Acquired Book 1 → SPV1` does not
exist anywhere in the model; a request is a base population, optionally narrowed
to governed portfolio ids, with cohort and filter claims independent of both.

## 2. Provenance

`SourceScopeClaim.provenance`, plus a derived `stated_by_user`, and the schema
now **refuses** a `FILLED` claim that carries no provenance — a resolved scope
whose origin is unknown is the Phase 1F blocker in object form, and defaulting
it inside the claim would pick a side.

| value | meaning | precedence |
|---|---|---|
| `explicit_user` | the question named this scope | **wins**, over any caller context |
| `caller_context` | the question was silent; this is the workspace selection | applies |
| `default` | neither; the complete funded population | applies |
| `unresolved` | the question named a scope that could not be resolved | **refuse — never Funded** |

Precedence is applied **once**, in the projection, from the owner's own two
readings (`resolve_lens` and `mentions_portfolio` / `names_governed_portfolio`).
No phrase list was added to `question_interpretation`, and the planner still
cannot reach the question — `build_plan`'s signature carries no question
parameter and `assert_no_question_read` still holds.

## 3. Precedence truth table

`python -m migration_phase0.scope_precedence_matrix --registry fixture`
— 9 question kinds × 6 caller contexts, against a registry with 1 direct,
2 acquired and 2 SPVs.

**54 of 54 cells meet the §5 rule** (was 41 of 54).

| question | caller | provenance | selects |
|---|---|---|---|
| silent | — | `default` | complete funded population |
| silent | Acquired | `caller_context` | all 3 acquired ids |
| silent | **SPV1** | `caller_context` | `['spv1']` |
| *the funded book* | Acquired | `explicit_user` | complete funded population |
| *across all portfolios* | Acquired | `explicit_user` | complete funded population |
| *the acquired book* | Direct | `explicit_user` | all 3 acquired ids |
| *Show SPV2* | Acquired | `explicit_user` | `['spv2']` |
| *SPV9* / *acquired_001* | any | `unresolved` | **nothing — refuse** |

Four families of divergence were closed:

1. **A caller-selected portfolio became Total.** `_SELECTABLE_COHORT_ID_RE`
   required an underscore, so a workspace scoped to `spv1` fell through every
   branch of `lens_from_selection` and resolved to the whole book. **The registry
   now decides what is selectable**, not a naming convention — the same defect
   class Phase 1D found on the text side, on the selection side.
2. **"The funded book" deferred to the dropdown** — §5 below.
3. **An unknown SPV widened to Funded** — §5 below.
4. **`resolve_scope`'s fallback to Total** reached the plan. The claim now
   resolves governed ids itself and **never takes that list**; the fallback
   remains in `trakt_core`, disclosed by `fell_back_to_total`, as Phase 1F
   recorded.

## 4. Category versus specific portfolio — proven numerically

The fixture gives each portfolio a distinct power-of-ten balance, so any subset
sums to a value unique to that subset and a mistake is a **wrong number**, not a
wrong-looking id list.

| request | selects | balance |
|---|---|---|
| no scope named | everything | **£51,234.00** |
| *the funded book* | everything | **£51,234.00** |
| *the direct book* | `direct_a`, `spv2` | **£51,000.00** |
| *the acquired book* | `acquired_a`, `acquired_b`, `spv1` | **£234.00** |
| *Acquired Portfolio A* | `acquired_a` | **£200.00** |
| *SPV1* | `spv1` | **£4.00** |
| *SPV2* | `spv2` | **£50,000.00** |
| *SPV9* | — | refuses |

The two mistakes §8 forbids are each a different number here: a category
collapsing onto one portfolio is £200 instead of £234, and a named portfolio
read as its category is £234 instead of £4.

**The live book cannot show any of this** — it holds one portfolio per category,
so every one of these mistakes selects identical rows on it. That is why the
fixture exists.

## 5. Two client-visible changes, both authorised, both measured

### "The funded book" is now an explicit scope

Phase 1F reported this as a finding it would not fix. §1 makes the business
meaning authoritative, so the phrase is now explicit and overrides a workspace
selection, like its synonym *across all portfolios*.

`python -m migration_phase0.funded_book_precedence_change`

| question | no selection | UI = Acquired | UI = Direct |
|---|---|---|---|
| **Summarise the funded book** | 11,035 | **11,035** *(was 3,909)* | **11,035** *(was 7,126)* |
| **Summarise the funded portfolio** | 11,035 | **11,035** | **11,035** |
| *portfolio summary* (silent) | 11,035 | 3,909 | 7,126 |
| *across all portfolios* | 11,035 | 11,035 | 11,035 |
| *the acquired book* | 3,909 | 3,909 | 3,909 |
| **What is the funded balance?** | 11,035 | **3,909** | **7,126** |
| **Show funded balance by region** | 11,035 | **3,909** | **7,126** |

The last two are the constraint that keeps the vocabulary honest: bare *funded*
names a **measure**, and reading a measure as a scope is the silent mutation this
vocabulary exists to prevent. Only the noun phrases that name the **book** were
added — `funded book`, `funded portfolio`, `funded loan book`.

#### It supersedes a prior product ruling — flagged, not buried

§5 says to implement the rule *"unless executable production behaviour exposes a
contradiction requiring escalation"*. One surfaced, and it is a **prior ruling
recorded in a test docstring**, not an ambiguity in production:

> `tests/test_p1i_scope_resolution.py::test_funded_scope_keeps_the_active_selection`
> **"Product ruling: 'funded book' names the active dataset, not a scope
> override. With one book selected the answer stays on that book."**

That reading took *funded* to distinguish the funded **tape** from the pipeline
tape — a dataset distinction, which says nothing about portfolio scope, so a
workspace selection survived it. It is coherent, and it is the opposite of §1.

**§1 was taken as authoritative**, as this phase and the two before it state it,
and §5's worked example is this exact scenario:

```
UI = Acquired
Question = "Summarise the funded book"
→ Funded
```

The test is renamed and rewritten to assert the new ruling **with the old one
quoted in it**, so the supersession is visible at the point of change rather
than absorbed. The rest of the P1I ruling is untouched and still asserted:
"the funded book" creates no `funded_status` row predicate, no geography filter
and no grouping axis. **Only the precedence changed.**

This is the one place in the phase where a recorded decision was reversed. If
the P1I reading is still wanted, this is the change to revert — it is one test
and three vocabulary entries.

### An unknown portfolio name refuses instead of widening

`Summarise SPV9` resolved to Total and answered for all five portfolios under
the name of one that does not exist. Phase 1E's detector needed a
"Book"/"Portfolio" head noun, which that sentence has not got.

The fix is **derived from the registry**, not from a vocabulary: a registry
holding `spv1` and `spv2` demonstrates an `spv<n>` naming family, so `spv9` is a
member of a family MI can see and a name it cannot resolve. There is no `spv`
literal anywhere in the resolvers — asserted — so a client whose portfolios are
`pool1` and `pool2` gets `pool7` recognised for the same reason, and a client
with no numbered family sees no change at all.

## 6. Adding a portfolio is data, not code

`SPV3` appears in no resolver. Registering it makes it resolvable:

```
registry without spv3   "Summarise SPV3"  ->  UNRESOLVABLE   (the control)
registry with    spv3   "Summarise SPV3"  ->  portfolio_ids ('spv3',)
                        "the acquired book" -> 234.00 becomes 238.00
```

Four properties are asserted: the name resolves once registered; it does **not**
resolve before (so the first test measures registration, not a coincidence); a
new category member joins its category with no resolver change; and no portfolio
name is hard-coded in `portfolio_lens`, `projection` or `schema`.

Adding SPV4 needs a registry entry. It needs no parser branch, no route, and no
new scope value.

## 7. Vintage and portfolio identity are separate axes

```
"How has the 2025 vintage of SPV2 progressed?"
   source_scope : cohort, portfolio_ids ('spv2',), base_population funded
   population   : [('cohort_vintage', '2025')]
```

Both claims survive; neither overwrites the other; the scope vocabulary is
unchanged at four values. A hierarchy would have needed a scope value per
vintage per portfolio, which is the model §2 forbids.

`cohort_vintage` is **carried from the owner that already reads it**
(`spec.cohort_vintage`, set by the deterministic parser). No vintage capability
was built and nothing reads the question.

**The known limit, unchanged:** the parser sets `cohort_vintage` only when the
question also carries a progression marker, so a **point-in-time** vintage
("Show the 2025 vintage for SPV2") is dropped upstream of the contract. That is
the Phase 1D defect, pre-registered, and it is an **owner** gap rather than a
structural one — the contract represents both axes simultaneously, which the
progression phrasing proves. Pinned `xfail(strict=True)`.

## 8. Routed interpretation wiring

`try_route` now supplies a `QuestionInterpretation` through `RouteRequest`,
built from the same spec and facets the receipt layer already produces, with the
**governed registry** and the **caller's workspace selection** as inputs.

A lazy provider with a per-request memo — the same shape `history_model` uses,
and for the same reason: assembling it detects the request's facets, which reads
the frame. Recognition never touches it; only a handler that asks pays. A
provider that raises yields `None`, so a plan that cannot be built refuses on
the contract's own terms rather than costing the request.

**Nothing reads it yet.** Every handler is byte-for-byte unaffected; the
contract is carried so the first conversion has something to plan from. The
point-in-time site in `mi_agent_workflow` still passes neither registry nor
caller scope — a known, separate gap, and not on the path being converted.

## 9. Phase 1F matrix, before and after

`python -m migration_phase0.contract_sufficiency_portfolio_summary`

| | Phase 1F | Phase 1G |
|---|---|---|
| plan correctly from the contract | 34 / 54 | **48 / 54** |
| **WIDENS** | **14** | **0** |
| BLOCKED (unresolvable; production also refuses) | 6 | 6 |
| identical claim, different shipped population | **yes — the blocker** | **none** |

The instrument's own closing line changed from listing 27 divergent rows to:

> *IDENTICAL CONTRACT CLAIM, DIFFERENT SHIPPED POPULATION: none — the claim
> determines the population.*

Every changed cell by name: the 14 widenings were `A1`–`A6` and `F1`, each at
`default=acquired` and `default=direct`. `A1`–`A6` are now `caller_context` and
follow the selection; `F1` is now `explicit_user` and does not.

## 10. `portfolio_summary` shadow equivalence

`python -m migration_phase0.equivalence_portfolio_summary`

```
cases on the surface     : 9
cases NOT claimed        : 2 -> ['X1', 'X2']
economic differences     : 0
cases the plan BLOCKS    : 0
externally supplied lens : 0
plan population == shipped population on every compared case: yes
```

Two corrections were needed to make this result mean anything, and both were
made before the result was read:

* **the harness was measuring the pre-1E reading.** It projected without a
  registry, so its "0 differences" said nothing about the governed path. It now
  passes the registry, exactly as the routed path does.
* **`lens_for` rebuilt a raw type filter.** With governed ids in the claim it now
  selects `{'source_portfolio_id': [...]}` — Phase 1F §4a, closed. Its guard
  compares the **selected ids** rather than the lens name.

Population equivalence is compared **on the rows**, not on the lens name: the
plan now legitimately says `cohort` where the shipped route says `acquired`.
Those are different names for the same population on this book and different
populations on a book with two portfolios of one type (Phase 1C: £300 against
£1,200). Comparing names would have reported the correction as a regression.

## 11. Regression

Every A5 surface re-captured and diffed against the pre-change capture:

| surface | result |
|---|---|
| calibration bank | unchanged |
| robustness 44 | **byte-identical** — 32 / 6 / 4 / 2 |
| — seasoning by name | **Q1 4 · Q7 4 · Q8 12 CORRECT** |
| shipped shapes | **byte-identical** — 15 correct, 0 wrong |
| routed surface | **byte-identical** — 31 passed, `rt_004` known-failing |
| recognition (61 phrasings) | **byte-identical** — 15 / 7 / 10 / 29 |
| time-series surface | **byte-identical** — **silent drops 0** |

Suites: **8 failed, 2022 passed, 3 skipped, 15 xfailed** — the same eight
pre-existing failures, by name, that Phase 1F recorded.

Wider estate, run **before and after in the same tree** with only the five
production files swapped to `918d3c4` for the baseline, so data and fixtures are
identical — the 30 scope/lens/receipt suites in `tests/` plus the whole of
`mi_agent_api/tests/` (2,429 tests):

| | before | after |
|---|---|---|
| | 16 failed, 2413 passed | 17 failed, 2412 passed |

**One movement, and it is attributable:**
`test_p1i_scope_resolution::test_funded_scope_keeps_the_active_selection` — the
superseded ruling in §5. Nothing else in the estate moved in either direction.

**Introduced failing names: 0. Silent drops: 0. Baselines updated to absorb
changed behaviour: 0.**

Nine tests changed **by decision**, each re-pinned to the new behaviour rather
than deleted: the two Phase 1B `xfail(strict=True)` precedence tests and the
Phase 1F `test_the_contract_can_decide_precedence` now **assert** the property
instead of pre-registering its absence; `test_both_produce_the_same_contract_claim`
became `test_the_two_readings_resolve_to_the_same_scope`;
`TestTheTotalScopeVocabularyIsUneven` became `...IsEven` and gained the
funded-measure control; `TestTheRoutedPathBuildsNoInterpretation` became
`...BuildsAnInterpretation`; two contract tests moved from asserting an
empty `portfolio_ids` to asserting the governed ids; and
`test_funded_scope_keeps_the_active_selection` became
`test_funded_scope_overrides_the_active_selection`, quoting the ruling it
supersedes.

## 12. Files changed

| file | lines | why |
|---|---|---|
| `question_interpretation/schema.py` | +76 | `base_population`, `provenance`, `stated_by_user`, the provenance invariant |
| `question_interpretation/projection.py` | +115 −6 | precedence applied once; governed ids for categories; vintage carried |
| `mi_agent/portfolio_lens.py` | +107 −3 | registry-backed selection; funded-book vocabulary; registry-derived naming family |
| `mi_agent_api/chat_routing.py` | +42 | the routed construction site |
| `mi_agent_api/recogniser_registry.py` | +35 | `interpretation_provider` / `resolve_interpretation` |
| **production total** | **375 (+358 / −17)** | |

Tests: `tests/test_scope_model_and_portfolio_selection.py` (new, 25 + 1 xfail),
plus the re-pinned tests above. Instruments: `scope_precedence_matrix.py` and
`funded_book_precedence_change.py` (new), `contract_sufficiency_portfolio_summary.py`,
`equivalence_portfolio_summary.py` and `shadow_portfolio_summary.py` (updated).

## 13. Out of scope, untouched

Back Book span collision, arity disclosure, filter-clause joins, Phase 4
deletion, T3–T7, economic definitions, LLM interpretation. No SPV-specific
parser branch and no storage-folder vocabulary was added — asserted by test.

## 14. Is `portfolio_summary` safe to convert?

**Yes, in the next task.** Every condition Phase 1F named is met:

| Phase 1F prerequisite | status |
|---|---|
| the contract cannot decide precedence | **closed** — `provenance`, 0 widenings on the matrix |
| the plan selects a raw type filter | **closed** — governed ids, compared on rows |
| the routed path builds no interpretation | **closed** — `RouteRequest.resolve_interpretation` |

The conversion's own gates are already green in shadow: 100% construct from
contract on the answering cases, 0 external injection, 0 raw-question rereads,
0 economic differences. What remains unmeasured is what a shadow structurally
cannot measure — **payload and receipt equivalence** — and that is the
conversion task's work, as it has been since Phase 0.
