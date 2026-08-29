# Phase 1E — registry-aware portfolio resolution

**Branch** `claude/clause-splitting-scoping-38ahbz` · **Base** `cbc95fc` (Phase 1D addendum)
**Scope** Make natural-language portfolio resolution use the governed registry, and
prove that a portfolio reference MI cannot resolve does not silently widen to
Funded/Total. `portfolio_summary` is **not** switched to the plan layer here.

**Verdict: GO-READY.** The two Phase 1D gates are closed on measured behaviour,
no gate regressed, and one further defect was found, fixed and pinned. One
pre-existing defect is reported and deliberately **not** fixed — it is
fail-closed and its fix is outside 1E's remit. `portfolio_summary` can be
retried in Phase 1F.

---

## 1. The business rule this phase encodes

Confirmed for this phase, and it corrects an assumption the earlier phases were
carrying:

> **Funded Book** means all funded assets **across both the Direct and the
> Acquired book** — and any other governed funded category. It is **not** a
> synonym for Direct.

So:

| term | means |
|---|---|
| Funded Book | every governed funded portfolio, all categories |
| Direct Book | every governed funded portfolio classified `direct` |
| Acquired Book | every governed funded portfolio classified `acquired` |
| a named book | exactly that one governed portfolio |
| a name not held | clarify — never widen |

## 2. What was wrong

Phase 1D established that React and MI share an identity *model* (the governed
registry) but not an identity *vocabulary*. MI's text side recognised the
**storage convention** — `acquired_001`, which is a blob folder name — and
nothing else. Every name a client can actually see resolved to the whole book.

Measured end to end against the live book before this phase
(`alp_origination` 7,126 loans / £1.39bn; `alp_acquired` 3,909 / £579.4m;
`spv1_sponsored` 0 funded rows; whole book 11,035 / £1.96bn):

| question | answer before 1E |
|---|---|
| Summarise the **ALP Origination Book** | 11,035 loans / £1.96bn |
| Summarise the **alp_acquired** book | 11,035 loans / £1.96bn |
| Summarise the **spv1_sponsored** portfolio | 11,035 loans / £1.96bn |
| Summarise the **acquired_001** book | "the portfolio **(acquired_001)** holds **11,035** loans …", `ok=True`, no facets, no warnings |
| Summarise the **Highgate Mortgages Book** | 11,035 loans / £1.96bn |

The `acquired_001` row is the whole defect in one line: the requested scope's
**name** printed against the **whole book's** figures, `ok=True`,
`lensApplied=True`, `executionSummary.facets == []`, `warnings == []`. The one
object that knew — `portfolioScope.fell_back_to_total` — never reached the
reader.

## 3. What changed

Four production files. Reproduce with
`python -m migration_phase0.name_resolution_answers`.

**`mi_agent/portfolio_lens.py`** — `resolve_lens` takes an optional governed
`registry`. With one:

* a portfolio named by its **governed display label** or its **governed id**
  resolves to that portfolio (`_named_portfolio_lens`, longest match first, so a
  label that contains a category word does not lose to the category);
* a cohort-shaped id the registry does **not** hold becomes the new
  `LENS_UNRESOLVED` instead of a cohort lens that then falls back to Total;
* a **capitalised book name** the registry does not hold likewise becomes
  `LENS_UNRESOLVED` (`_unknown_named_book`).

Without a registry every decision is byte-for-byte what it was. That is
asserted (`TestNoRegistryIsExactlyThePreviousBehaviour`), because a migration
that changed answers by omission would be worse than one that changed them by
decision.

**`mi_agent_api/chat_routing.py`** — `_resolve_lens` builds the governed
registry once, resolves the lens against it, and hands the *same* registry to
`resolve_context`, so the scope a route discloses and the scope its lens
resolved against are one object rather than two that happen to agree.

`_route_portfolio_summary` no longer defers to the point-in-time path when a
**narrowing** lens is in force. See §5.

**`mi_agent/execution_receipt.py`** — new `unresolved_scope_facets()` raises an
unresolved scope as a `LOST` narrowing facet. LOST is the status that fails
closed, and `reconcile_routed_facets` promotes a narrowing only on execution
evidence, which a scope that resolved to nothing cannot produce. The mechanism
that already adjudicates dropped narrowings therefore adjudicates this one; no
new refusal path was written.

**`mi_agent_api/mi_service.py`** — new `_guard_unresolved_scope`, applied at
**both** answer sites, beside the temporal-honouring guard that is already
applied at both. It builds the facets, calls `assess`, and refuses — so the
refusal sentence is the one every other dropped-narrowing refusal on this
surface is written in, not a second author of that wording. Route-independence
is the point; see §7a. The registry is built from **the frame the answer was
computed from**, never from the process-wide active dataset — see §7c.

## 4. Result

| case | question | before | after |
|---|---|---|---|
| named direct | Summarise the ALP Origination Book | 11,035 / £1.96bn | **7,126 / £1.39bn** *(ALP Origination Book)* |
| governed id | Summarise the alp_acquired book | 11,035 / £1.96bn | **3,909 / £579.4m** *(ALP Acquired Back Book)* |
| zero-row governed | Summarise the spv1_sponsored portfolio | 11,035 / £1.96bn | *"There are no funded loans in spv1_sponsored … I have not answered for the whole book instead."* |
| storage id | Summarise the acquired_001 book | 11,035 / £1.96bn **labelled acquired_001** | **controlled refusal** |
| unknown label | Summarise the Highgate Mortgages Book | 11,035 / £1.96bn | **controlled refusal** |
| direct category | Summarise the direct book | 7,126 / £1.39bn | unchanged |
| acquired category | Summarise the acquired book | 3,909 / £579.4m | unchanged |
| funded category | Summarise the funded book | 11,035 / £1.96bn | unchanged |
| no scope named | Please provide a portfolio summary | 11,035 / £1.96bn | unchanged |
| named acquired | Summarise the ALP Acquired Back Book | refusal | refusal — **see §6** |

Against a fixture registry with **two** acquired portfolios (the live book has
one per category and cannot show a category collapsing),
`python -m migration_phase0.identity_resolution_table` reports **10 of 10**
cases meeting the target semantics, up from 4 of 10 at the start of the phase.

## 5. A defect this phase found, and fixed

`spv1_sponsored` is a governed portfolio: it is in the registry and carries an
asset class in the portfolio metadata. It has **zero rows** in the funded
frame.

Once the lens resolved it, `movement_summary.portfolio_summary` correctly
returned `available: False, reason: "no governed reporting period is available
for this scope"` — and `_route_portfolio_summary` then did
`return None  # defer to the existing point-in-time summary path`. The
point-in-time path cannot see a lens parsed from the question, so the answer
came back as the whole book's 11,035 loans with the requested scope mentioned
nowhere.

**The layer was honest and the caller discarded it.** Deferring is safe only
when the lens is Total, because only then is there no narrowing to lose. The
route now states the empty scope instead — the same shape `geo_exposure`
already uses for the same condition:

> There are no funded loans in spv1_sponsored at the current governed reporting
> date, so there is no position to summarise for it. I have not answered for
> the whole book instead.

This is the fifth instance of one pattern across Phases 1A–1E: **a governed
concept has a strict path and a convenient path, production uses a mixture,
and it is visible only by executing it.** It is also the second that this
programme has now closed rather than only recorded.

## 6. A defect this phase found, and deliberately did NOT fix

`ALP Acquired Back Book` is a governed display label. The lens layer resolves
it correctly to `alp_acquired` (asserted). The **population parser**, reading
the same sentence with a different vocabulary, separately reads the substring
`"Back Book"` as `seasoning_segment = Back Book`; `portfolio_summary` cannot
apply a population filter, so the answer is a controlled refusal.

* It is **pre-existing** and **unchanged** by 1E — measured on the stashed tree.
* It is **fail-closed**: a refusal, not a wrong number, and not a widening. No
  §11 hard requirement is violated.
* Asked by its governed **id** (`Summarise the alp_acquired book`) the same
  portfolio answers correctly, which is what makes the collision the *label's*
  and not the portfolio's.

The fix is for `portfolio_lens.mask_scope_phrases` / `scope_phrase_spans` — the
span-masking layer that already exists to stop one parser eating another's
words — to know the registry. That is a change to every parser call site, and
threading a registry through them is a scoped piece of work, not an improvised
one at the end of a phase. **Pinned as `xfail(strict=True)`** in
`tests/test_portfolio_name_resolution.py::TestKnownDefectALabelThatCollidesWithSeasoningVocabulary`,
so it cannot be fixed or worsened silently.

## 7a. A second widening the first fix did not reach

The first cut wired the disclosure into `_guard_routed_answer` only. That guard
runs on **routed** answers; a question that falls through to the point-in-time
path never reaches it. Measured with that fix alone in place:

| question | route | answer |
|---|---|---|
| What is the funded balance **by region** for the Highgate Mortgages Book? | *(none)* | `ok=True` · "Total Balance, grouped by Region, 12 groups, **11,035 loans**" |
| What is the funded balance of the **acquired_001** book? | *(none)* | `ok=True` · "Total Balance, entire funded portfolio, **11,035 loans**" |

The whole book, under the name of a book this platform has never onboarded.

**Which route happens to claim a question is not a fact about whether its scope
resolved.** Phase 0 recorded exactly this as a governance prerequisite — a
receipt proof that holds only on the routed path is not a proof — and the first
fix reproduced the defect it was closing, one layer down.

`_guard_unresolved_scope` is therefore applied at **both** sites, and is the
**single owner** of this refusal: the facet append inside `_guard_routed_answer`
was removed rather than left beside it, because two mechanisms for one decision
is the second-owner defect this codebase keeps warning about. Asserted across
four questions and three routes in `TestTheRefusalIsRouteIndependent`, together
with the two controls that matter — a name the registry **does** hold still
answers (7,126 loans), and a question naming no portfolio is untouched (11,035).

## 7b. A false refusal the second fix caused, and fixed before commit

Once the guard was route-independent, eight tests across
`test_p1e_golden_bank`, `test_p1e_measure_safety` and `test_p1e_multi_measure`
failed — every one of them on the phrase **"the London book"**, as in

> For the London book, give me balance, number of loans, weighted-average LTV
> and average borrower age.

which is an answered CFO question in this estate's own golden bank. `London` is
a governed **value** of `collateral_geography`. The lens layer has no vocabulary
for what values the tape carries, so it read `London Book` as a book name it
could not find, and the guard refused.

**A false refusal on a question the system answers correctly is a worse failure
than the widening this facet exists to stop.** The guard now consults
`execution_receipt.dimension_values(frame, semantics)` — the existing owner of
what values this book carries, built from the *loaded book*, so the check is
the profiled-allowlist discipline rather than a word list maintained here. A
requested scope that reduces to a value the tape carries (`london`,
`south east`) is a population, not a portfolio, and raises no facet.

Note where the fix lives: the **guard**, not the lens. `LENS_UNRESOLVED` carries
no filters, so an over-eager lens verdict computes exactly the pre-1E answer;
only the guard turns it into a refusal. Making the layer that *decides* precise
is enough, and it keeps the vocabulary out of the lens, which owns none.

Pinned as a control in
`test_a_value_this_book_carries_is_a_population_not_a_book_name`.

## 7c. A regression this phase caused, and fixed before commit

The first cut of the disclosure step called
`portfolio_context.build_registry()` with no frame. That falls through to
`active_frame()`, which populates `data_source._ACTIVE_CACHE` under a TTL.
Reaching for it on **every** governed answer leaked one test's fixture frame
into the next and broke five unrelated receipt tests
(`test_d6_book_scoped_availability`, `test_d7_grouping_evidence`,
`test_d8_population_evidence` ×2, `test_dimension_role_owner`) — each of which
passed in isolation and failed in a full run, which is the shape of defect a
per-file run never catches.

Fixed by building the registry from the frame the guard already holds. That is also the more correct object: the registry should describe the
population the answer was computed from. **A disclosure step has no business
changing what the next request reads.**

## 8. Canonical identity in the interpretation contract

`SourceScopeClaim` now carries governed identity explicitly:

* `portfolio_ids` is documented as **governed portfolio ids** — the identity the
  registry keys on, and the only one a consumer may filter or join on. Before
  1D this was the storage convention.
* new `portfolio_label` — the governed **display label**, kept separate from
  `raw_text`, which is **the wording that asked**. "the alp_acquired book" and
  "ALP Acquired Back Book" are the same portfolio said two ways; an audit of
  what was asked needs the first, an explanation of what was answered needs the
  second, and collapsing them loses one.
* the schema now **refuses** `state=FILLED, scope=cohort` with no id. A cohort
  claim with nothing to narrow by would make `narrows` true with no narrowing;
  the honest reading of that state is UNRESOLVABLE, which the owner must state.
* an unheld name projects as **UNRESOLVABLE** naming the wording, never
  `scope=total`. UNRESOLVABLE and "explicitly the whole book" are the two
  readings this claim exists to keep apart.

`project()` / `from_parts()` take an optional `registry`, passed in rather than
discovered — this module reaches into no application state, which is what keeps
it a transport object. `mi_agent.portfolio_lens` remains the single owner; no
vocabulary was added here.

The contract is still **carried, not acted on**: nothing in production consumes
`QuestionInterpretation` yet, so §8 changes no answer.

## 9. Stated residual — the governed contract still widens

`resolve_scope(registry, "acquired_001")` still returns **every** portfolio id
with `fell_back_to_total=True`. The contract *discloses* the widening; it does
not *prevent* it. What prevents the answer widening is the facet layer
downstream.

This is recorded rather than fixed because `resolve_scope` is `trakt_core` and
its blast radius is every channel. It is pinned twice, deliberately:

* `test_the_governed_contract_discloses_the_widening_it_does_not_prevent` —
  the flag is actually raised, so a fallback that stopped setting it would leave
  the downstream refusal as the *only* line of defence, undetectably;
* `test_case_d_an_unresolvable_scope_widens_to_the_whole_book` — kept from
  Phase 1C, unchanged.

`migration_phase0/identity_resolution_table.py` prints the raw scope alongside
the lens for exactly this reason: scoring the lens alone would have recorded a
pass the governed contract does not earn on its own.

## 10. Vintage identity

Re-proved against a **named** portfolio, which Phase 1D could not do because no
name resolved: `"Show the 2023 vintage of the NBS Acquired Book"` resolves the
source scope to `nbs_acquired` and leaves `2023` outside every lens phrase span,
so the seasoning axis still has its year to read. A bare
`"Summarise the 2023 vintage"` still scopes to the whole book. Vintage and
source provenance remain independent axes.

## 11. Gates

Baseline taken on a **fully stashed tree** (`-p no:randomly`, selection
`mi_agent/tests question_interpretation/tests` + the identity/scope/migration
suites): **7 failed, 1825 passed, 1 skipped, 14 xfailed**. Those 7 are
pre-existing and untouched by this phase.

After 1E, the same selection: **7 failed, 1874 passed, 1 skipped, 15 xfailed**
— the *same seven*, by name. No new failures, and none fixed by accident. The
one test that changed did so by decision:

* `test_case_d_the_widening_is_not_disclosed_to_the_reader` →
  `test_case_d_the_widening_is_now_refused_rather_than_printed`. It pinned the
  Phase 1C defect verbatim; 1E closes it, so it now pins the closure. The
  `fell_back_to_total` assertion is deliberately **kept** in its sibling — see
  §9.

The seven pre-existing failures, for the record:
`test_mi_predicate_extraction::test_complex_query_executes_all_filters`,
`test_mi_trust_hardening::test_C_joint_borrowers_count_and_balance`,
`test_p0_execution_receipt` ×3,
`test_parser_cost_hardening::test_layered_question_routes_to_llm_even_when_deterministic_parses`,
`test_p0_time_axis_request::test_the_wording_that_asked_is_returned[balance by each month]`.

New coverage: `tests/test_portfolio_name_resolution.py` (39 passed, 1 xfail),
`question_interpretation/tests/test_source_scope_identity.py` (12 passed).

A second, wider selection was run **before and after in the same tree**, with
only the six production files swapped to `cbc95fc` for the baseline, so the
data and fixtures are identical: the 30 `tests/test_*` suites that reference
`portfolio_lens`, `portfolio_context`, `portfolio_scope`, `chat_routing`,
`mi_service`, `execution_receipt`, `resolve_lens` or `source_portfolio`, plus
the whole of `mi_agent_api/tests/`.

| selection | baseline | after 1E |
|---|---|---|
| scope/lens/receipt suites in `tests/` | 5 failed, 1181 passed | **4 failed, 1182 passed** |
| `mi_agent_api/tests/` (1,244 tests) | 12 failed, 1231 passed | **12 failed, 1231 passed** — same tests by name |

**No new failures in either.** The single difference is the rewritten Case D
test, which by design fails on pre-1E code and passes on 1E. The four
remaining failures in the first selection are pre-existing and named:
`test_analytical_capability_layer::test_q7_...` (the Q7 failure Phase 0
**pre-registered**), `test_p1e_measure_safety::test_a_routed_capability_may_satisfy_a_share_by_stating_one`
(a postcode/ITL3 disclosure question, confirmed pre-existing by running it
against `cbc95fc`), and two in `test_p1l_population_propagation`.

Hard requirements, each asserted:

| requirement | test |
|---|---|
| a named React portfolio resolves to its governed id | `test_react_display_label_resolves_to_its_governed_id` |
| no named portfolio broadens to category scope | `test_a_named_portfolio_does_not_broaden_to_its_category` |
| a category does not collapse onto one portfolio | `test_acquired_book_is_every_acquired_portfolio` |
| Funded is Direct **and** Acquired | `test_funded_book_is_direct_and_acquired_together` |
| an unknown explicit label does not widen | `test_an_unknown_name_resolves_to_unresolved_not_total`, end to end in `test_case_d_the_widening_is_now_refused_rather_than_printed` |
| ... on **every** path, not just the routed one | `TestTheRefusalIsRouteIndependent` (4 questions, 3 routes, 2 controls) |
| ordinary questions are untouched | `TestOrdinaryQuestionsAreUntouched` (10 cases) |
| a governed dimension value is not read as a book name | `test_a_value_this_book_carries_is_a_population_not_a_book_name` |
| no registry ⇒ pre-1E behaviour | `TestNoRegistryIsExactlyThePreviousBehaviour` |

## 12. Known limit, stated rather than papered over

`_unknown_named_book` requires the book name to be **capitalised** in the
original text. `"summarise the highgate mortgages book"`, all lower case,
carries no proper-name signal and still widens to Total. Requiring
capitalisation is precisely what stops the detector refusing ordinary lending
English ("the loan book", "the back book", "the retirement interest only
book"); recognising a lower-case unknown name needs a vocabulary check this
layer does not own. Documented in the function, not left to be discovered.

## 13. Abort conditions

None of A1–A5 tripped. In particular **A5** (a governed answer changes value
without a decision) did not: every value that changed in §4 changed because a
narrowing the question asked for is now applied, and each is asserted against
the governed population it names.

## 14. Recommendation

`portfolio_summary` **can be retried** in Phase 1F. The blocker Phase 1B
stopped on — the precedence prerequisite — and the blocker Phase 1C stopped on
— unresolvable scopes widening without disclosure — are both closed on measured
behaviour, and the identity flowing into the plan layer is now the governed one
rather than the storage one.

Carry forward, unfixed:

1. the label/seasoning vocabulary collision (§6) — pinned `xfail(strict=True)`.
   Needs registry-aware span masking, and should be scoped as its own piece of
   work. A second face of the same collision: on the **point-in-time** path
   `"What is the funded balance by region for the ALP Origination Book?"`
   refuses, because the filter parser reads the governed label as a
   `source_portfolio_label` predicate it cannot apply. Also fail-closed; also
   fixed by registry-aware span masking. Both are the same root cause — two
   parsers reading one sentence with different vocabularies and no arbitration
   — and fixing that root cause is the natural scope of a Phase 1G.
2. `resolve_scope`'s fallback to Total (§9) — a `trakt_core` decision, pinned
   twice;
3. the lower-case unknown-name gap (§12) — stated in the function.
