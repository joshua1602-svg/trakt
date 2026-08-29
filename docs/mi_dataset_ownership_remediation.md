# Dataset ownership remediation — question-driven, tab-agnostic

Base `d927963`. Conditions pre-registered at `7edc197` before any production
change. **Net executable production lines: +0** — 44 added, 44 deleted. This is
a relocation of semantics, not an addition, which is what consolidating two
owners into one should look like.

---

## 1. The rule now in force

```
question
  → workspace.resolve_dataset(question)        ← the one semantic owner
  → FUNDED | PIPELINE | FORECAST
  → interpretation contract  (qi.dataset)
  → every downstream consumer
```

`resolve_dataset` takes **one parameter**. A parameter it does not have cannot
quietly become an input again, and `test_the_resolver_cannot_be_handed_a_tab`
pins the signature.

Precedence: `forecast` > `pipeline` > `funded` > pre-funding artefact > default.

## 2. Semantic-owner inventory

| what | before | after |
|---|---|---|
| `workspace.resolve_dataset` | did not exist | **SEMANTIC OWNER** — the only one |
| `workspace.view_named_by_question` | semantic owner (view names) | the view-name half of the owner; called by it, by nothing else |
| `workspace.resolve_active_view` | **semantic owner** — folded the question and the tab together | delegating shim; `dataset_context` accepted and inert |
| `chat_routing._dataset_for` | **second semantic owner** — its own wider tape vocabulary | **retired**; vocabulary moved to `workspace.PIPELINE_ARTEFACTS` |
| `chat_routing._PIPELINE_WORDS` | the second owner's vocabulary | **retired** |
| `projection._dataset` | consumer that re-derived from the question + caller tab | consumer; carries the owner's answer |
| `mi_service` line 261 | called the tab-folding owner | calls `resolve_dataset` directly |
| `_route_compare`, `_route_evolution` | called the second owner | call the one owner |
| `_resolve_frame`, envelope `datasetContext`, `MiQueryRequest.dataset_context` | consumers | consumers, unchanged |
| `app.py /mi/evolution/compare` | consumer with a caller-supplied `dataset` query parameter | unchanged — a typed API argument, not natural-language MI |
| `mi_workflows.analytical.intent` | third *reader* — decides refusability, never the dataset | unchanged, and deliberately so (§4) |

**One owner, not two that agree.** `test_the_second_owner_is_gone` and
`test_no_production_module_re_decides_the_dataset_from_raw_text` assert it
structurally; the latter was proved non-vacuous by planting a copy of the
vocabulary in `evolution.py` and watching it fail.

## 3. Why this vocabulary, measured rather than asserted

The brief warns against a simplistic keyword rule where the governed
interpretation already expresses the concepts structurally. It does —
`mi_workflows.analytical.intent` carries `REQ_PIPELINE_DATASET` and
`REQ_FORECAST`. That candidate was **censused, not assumed**:

| candidate rule | worked-example failures | corpus movements (882) |
|---|---|---|
| today, point-in-time | 4 of 15 | — (the baseline) |
| today, routed (`_dataset_for`) | 4 of 15 | 29 disagreements with the point-in-time rule |
| **R1 — narrow union (chosen)** | **0 of 15** | **5 (0.6%)** |
| R2 — intent requirements | 0 of 15 | **59 (6.7%)** |

R2 is the wrong tool and the numbers say why: those requirements decide whether
a question is **refusable** and are checked *against* a dataset, never used to
select one, so their vocabularies are far wider. Used as a selector they send
*"top brokers by expected funded amount"* and *"expected funded by stage"* to
FORECAST, and demote genuine pipeline questions to forecast as well.

R1 is the narrowest union of the two existing owners: steps 1–3 are
`view_named_by_question` unchanged, so nothing it already decided can move; the
artefact step fires only where it returned `None`, which is exactly the gap the
retired second owner was covering alone.

## 3b. One requirement is NOT met: a bare `case`

The brief lists **"How many cases are there?" → PIPELINE** among its worked
examples. **It is not satisfied, deliberately**, and this section exists so the
choice is visible rather than looking like an oversight.

The retired second owner did list `case`, but it was reachable only from the
compare and evolution routes, so its ambiguity never decided anything much.
Read by the ONE owner it decides every question — and in this estate a bare
`case` means a **funded loan** at least as often:

```
"Which region gained the most cases since last month?"
```

sits in the P1C golden bank under `# -- loan count --`, directly beside
*"Which region added the most loans month-on-month?"*, and expects a ranked
**FUNDED** movement. Classified as pipeline it becomes a refusal:

> *I understood that you asked for comparison period (last month), but that
> could not be applied to the calculation … the answer is a single point in
> time; no period comparison was calculated.*

That is an unrelated question moving, which is registered STOP condition **B7**.

**The evidence for dropping the word rather than keeping it:** across the 882
distinct corpus questions, **not one** reaches the pipeline through `case`.
Every artefact-driven movement comes from `application`, `kfi` or `offer`, all
unambiguous. `case` bought nothing and cost a golden-bank answer.

A question that means the pipeline case still says so — *"how many **pipeline**
cases are there?"* resolves through the view name, unaffected.

The governed intent layer agrees with the brief and not with the golden bank:
`intent.classify("Which region gained the most cases since last month?")`
returns `families=('PIPELINE',)`, `requirements=('pipeline_dataset',)`,
`matched=('cases',)`. So production is **internally inconsistent about this one
word**, and consolidating owners is what made the inconsistency decide
something. Which way it should be settled is a **product decision about house
vocabulary**, not a dataset-ownership question, and it is left open:
`test_a_bare_case_is_a_funded_loan_in_this_estate` and the golden-bank entry
move together if it is taken the other way.

`migration_phase0.dataset_rule_census` reports the live owner as failing
**1 of 15** worked examples, with this reason printed. The instrument was not
edited to agree with the code.

## 4. The disagreement, before

`migration_phase0.dataset_ownership_disagreement`, 14 cases × 4 tabs = **56
executions**.

| | before | after |
|---|---|---|
| tab-sensitive cases | **6 of 14** | **0 of 14** |

The two worth naming:

* `"How many cases are there?"` — funded tab → the funded book, pipeline tab →
  the pipeline, forecast tab → the forecast. One unchanged sentence.
* `"What is the balance by seasoning segment excluding pipeline cases?"` —
  served from the **pipeline** on the pipeline tab. The question rules the
  pipeline out in words and the tab put it back.

## 5. Tab independence

```
questions whose dataset VARIED BY TAB before : 769 of 882
questions whose dataset varies by tab AFTER  : 0
```

## 6. 882-question blast-radius census

`migration_phase0.dataset_census_882` — every distinct corpus question through
both retired rules, on every tab, classified against the pre-registered classes.

| class | movements |
|---|---|
| **M1** tab influence removed | 2302 (769 distinct questions) |
| **M2** one owner rather than two | 5 |
| **M3** forecast precedence restored | 96 (24 distinct questions) |
| **UNEXPLAINED** | **0** |

The five M2 questions are exactly the artefact set:

```
applications over the last four weeks
What completion rate is assumed from KFI to completion?
How much do we currently have at offer and how much of it is likely to complete?
What's the value of outstanding offers, and when do we expect them to fund?
How much is sitting at offer today and what do we expect to complete from it and when?
```

The previously observed eight cases, checked individually: the **three forecast
questions remain FORECAST**, the five artefact questions resolve **PIPELINE**.

M3 is the retired routed rule's precedence bug being undone. `_dataset_for`
tested its tape words *before* reading any view name, so `"Show forecast balance
by region."` was `pipeline` to it while the active view was `forecast`.

## 7. Answer-state movement, reported not hidden

Across the 56 probe executions, 15 moved. **No answer became a refusal.**
Three refusals became answers:

| case | tab | before | after |
|---|---|---|---|
| `"...excluding pipeline cases?"` | pipeline | pipeline tape, `ok=False` | funded tape, `ok=True` |
| `"...excluding pipeline cases?"` | forecast | forecast tape, `ok=False` | funded tape, `ok=True` |
| `"What is the total balance?"` | pipeline | pipeline tape, `ok=False` | funded tape, `ok=True` |

Both are the defect being fixed: the question asked for the funded book and the
tab sent it elsewhere.

### One narrowing, recorded

`"amount by region"` on the pipeline tab used to be answered from the pipeline.
It is now answered from the funded book — the intended rule — and **the
shorthand has no replacement**, because naming the tape in the sentence feeds
`pipeline` to the *measure* parser, which rejects it:

> `'pipeline' is not a governed measure in this dataset`

That parser behaviour is **pre-existing** — `"pipeline amount by region"` was
already `ok=False` before this change, on every tab — but the tab shortcut
masked it and it is now load-bearing. Fixing it means teaching measure
resolution to ignore a word the dataset owner has already consumed: a separate
task with its own blast radius, out of scope here.
`test_an_unqualified_measure_on_the_pipeline_is_now_unreachable` asserts the
current state so the day it is fixed, that test fails and says so.

## 8. Tests changed, by name, and why

Every one encoded the retired semantics. None was deleted; each was re-pointed
at the rule that replaced it.

| test | change |
|---|---|
| `test_workspace_views::test_tab_context_used_when_no_explicit_wording` | renamed `…the_tab_no_longer_decides_when_the_question_is_silent`; now asserts `funded` on every tab |
| `test_workspace_views::test_unqualified_amount_routes_to_active_dataset` | renamed `…an_unqualified_question_means_the_same_thing_on_every_tab` |
| `test_workspace_views::test_funded_balance_differs_from_pipeline_amount` | renamed `test_naming_the_dataset_still_reaches_a_different_tape`; tabs deliberately crossed |
| `test_workspace_views::test_unsupported_query_fails_gracefully` | reaches the pipeline by wording instead of by tab |
| `test_b21_disclaimed_view::test_the_view_resolver_still_selects_a_genuine_view` | the one tab-dependent assertion retired, with a comment saying so |
| `test_b21_disclaimed_view::test_the_dataset_resolver_reads_the_same_test` | renamed `…the_second_owners_wider_vocabulary_survived_its_retirement`; same assertions, re-pointed at the owner |
| `test_contract_target_state::test_the_workspace_tab_applies_when_the_question_is_silent` | renamed `…no_longer_applies…`; asserts `caller_context` is unreachable |
| `test_contract_target_state::test_the_contract_agrees_with_the_owner` | passes the tab and asserts it makes no difference |
| `test_contract_target_state::test_the_view_reading_lives_in_one_place` | asserts the shim decides nothing — a stronger property than before |

New: `tests/test_dataset_ownership.py`, **53 tests** — the matrix across all
tab values, population independence, contract handoff, provenance, and the
structural single-owner proofs.

## 9. C5 dependency verifier

```
readings                                   : 26
dataset disagreements, contract AS BUILT    : 0   (was 10)
dataset disagreements, WITH the view wired  : 0   (was  3)
measure disagreements (at the same dataset) : 0
readings whose periods are STRUCTURAL       : 0
```

**Read the zero correctly.** The two sides now share one owner, so this is a
**wiring** check — does the contract carry the owner's answer to the route? —
not an agreement check between two rules. The instrument's docstring says so,
because a zero that reads as "two independent readings coincide" would be a lie.

## 10. Is `comparison_period` now the only remaining C5 prerequisite?

**Yes.** The dataset blocker is closed. `time.comparison_period` still carries
`", ".join(compare_periods)` and `Slot` has no list field, so the period pair
exists nowhere structurally — 0 of 26. That is the same closure `window_periods`
already made once for `trend_window`, Regime B, ~20 lines.

## 11. Regression

### Named suites

791 passed, 1 failed:
`question_interpretation/tests/test_p0_time_axis_request.py::test_the_wording_that_asked_is_returned[balance by each month-by each month]`
— present **verbatim in the baseline's 214 failing names**, so pre-existing and
not introduced.

Suites run: the full `question_interpretation` tests, portfolio identity,
C1–C4 conversion guards, the funded-bridge grouping declaration, the
pre-registered migration guards, contract target state, the new dataset
ownership matrix, MI recognition diagnosis, and workspace views.

Separately: `test_chat_routing_e2e` reported three failures —
`test_pipeline_amount_evolution_by_week_e2e`, `test_kfi_trend_by_week_e2e`,
`test_cumulative_cohort_conversion_routes` — all three present verbatim in the
baseline.

### Full estate

Baseline at `d927963` (same environment, whole repo):
**186 failed, 10168 passed, 35 skipped, 16 xfailed, 28 errors** — **214 failing
names** recorded for attribution.

*The after-run is in progress at the time of writing and its by-name
attribution is appended below when it completes. It was restarted from scratch
after the `view`-parameter removal so that a single run attributes the final
state rather than an intermediate one.*

## 12. Production lines changed

| | executable | comment / docstring / blank |
|---|---|---|
| added | 48 | 113 |
| deleted | 48 | 38 |
| **net** | **+0** | +75 |

Four production files: `mi_agent_api/workspace.py`,
`mi_agent_api/chat_routing.py`, `mi_agent_api/mi_service.py`,
`question_interpretation/projection.py`, plus a three-line field comment in
`mi_agent_api/recogniser_registry.py`.

Net executable **+0** is the result worth reading: consolidating two owners into
one should move code, not add it.

## 13. Scope held

Not touched: C5 itself, `comparison_period`, subject/operation accessors,
Direct/Acquired semantics, the portfolio registry, economic calculations, T3–T7,
LLM logic, UI tabs. `MiQueryRequest.dataset_context` and the workspace tabs
still exist and still select what the UI displays — this changed semantic
ownership, not UI behaviour.
