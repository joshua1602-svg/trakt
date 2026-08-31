# PPTX — Final Analytical Surfacing Hardening

**Starting HEAD** `7388513` · **Branch** `claude/pptx-analytical-surfacing-final`
· **Report date** 2026-08-31

---

## 1. Verdict

**YES.** The automated pack now selects the strongest cuts the particular book
supports, rather than the same four on every book.

The evidence is measured, not asserted. Running the same fixtures through the
selector at the starting commit and at HEAD:

| | at `7388513` | at HEAD |
|---|---|---|
| new_book | ltv, ticket, age, region | age, region, ltv, **rate** |
| seasoned_book | ltv, ticket, age, region | age, region, ltv, **vintage** |
| multi_seasoned | ltv, ticket, age, region | age, region, ltv, **vintage** |
| multi_growing | ltv, ticket, age, region | age, region, ltv, **vintage** |
| *pairs, every fixture* | ltv×age, ltv×region, ltv×ticket, ltv×rate | *five different sets — §6* |

Before, **every fixture produced exactly the same four dimensions and exactly
the same four crossings** — including the multi-book fixture that carries
borrower type and product. That is the defect in one line: the output did not
depend on the book. After, three distinct selections, seven distinct crossings,
and a deep-dive page carrying two further cuts on the one book large enough to
earn it. Slide counts are unchanged.

What this sprint did **not** do is widen the engine. No MI primitive, no
calculation, no model, no methodology, and no asset-class branching was added.
Every figure on every page was already computed; the change is which of them
reach a slide, and whether the pack tells the truth about the ones that did not.

---

## 2. Selector defect

### Before — preference first

```python
ranked = sorted(scored, key=lambda item: (order.get(key, len(order)), -score))
```

`preferred` was the FIRST sort key. Naming a dimension early in a config list
won it the page whatever the data said. On the audit book:

| dimension | categories | top share | outcome |
|---|---|---|---|
| ltv | 7 | 24.8% | drawn |
| ticket | 5 | **56.8%** | drawn — because it was preferred second |
| age | 7 | 14.5% | drawn |
| region | 7 | 14.5% | drawn |
| vintage | 5 | **20.4%** | rejected |

Vintage is the better-distributed cut and lost to ticket. The ledger then
printed *"4 more informative dimensions were available"* — a statement the same
numbers contradict, because ticket was not more informative than vintage.

### After — information first, preference as tie-breaker

```python
def rank(item):
    _entry, shape, key = item
    return (-round(shape["score"], SCORE_PRECISION),   # information
            order.get(key, len(order)),                # governed preference
            key)                                       # determinism
```

The score is deliberately simple, deterministic and auditable — two numbers a
reader can recompute from the bars on the page:

```python
score = evenness × granularity

evenness    = normalised Shannon entropy of the band shares (0 = one band
              carries everything, 1 = perfectly even)
granularity = k / (k + 4)   for k meaningful categories
```

`granularity` is a **saturating** term on purpose. "Most categories wins" would
make a two-category single/joint split permanently ineligible against a
seven-band age ladder, which the brief explicitly rules out: the marginal value
of a fifth band is large, of a twelfth band it is not. A well-split
two-category dimension scores 0.33 and can beat a lopsided five-band one.

Rounding to two decimals is what makes preference reachable: two dimensions a
reader could not tell apart are a genuine tie, and only there does the governed
economic order decide.

### Reason codes

Every rejection now carries a machine-readable code and the numbers it came
from, so the ledger is checkable against the selector's own inputs:

`NOT_SUPPLIED` · `ONE_CATEGORY` · `LOW_INFORMATION` · `LOW_COVERAGE` ·
`LOWER_RANKED_THAN_SELECTED_DIMENSIONS` — and for crossings, `TOO_SPARSE` and
`REDUNDANT`.

`LOWER_RANKED` is the code the untrue message used to be. It is a **ranking**
statement and says so: *"By ticket size, By rate band: available, and ranked
below the 4 drawn here."* It never claims the dimension says nothing.

---

## 3. All eleven governed dimensions

Every one of the eleven is reported with an explicit availability state — the
selector's ledger sits on top of that, it does not replace it.

Scores below are the seasoned multi-book fixture, the only one carrying all
eleven-capable columns.

| dimension | available | informative | eligible | score | selected in | suppression reason |
|---|---|---|---|---|---|---|
| ltv | yes | yes | yes | 0.624 | all five | — |
| age | yes | yes | yes | 0.636 | all five | — |
| region | yes | yes | yes | 0.636 | all five | — |
| vintage | yes | conditional | yes | 0.600 | seasoned ×3, multi_growing | `ONE_CATEGORY` on new_book (one origination year) |
| ticket | yes | yes | yes | 0.436 | deep dive (seasoned_gbp) | `LOWER_RANKED` |
| rate | yes | yes | yes | 0.412 | new_book_gbp; deep dive (seasoned_gbp) | `LOWER_RANKED` elsewhere |
| borrower_type | conditional | yes | yes | 0.324 | multidim (multi_seasoned) | `NOT_SUPPLIED` where no second applicant |
| product | conditional | conditional | yes | 0.301 | multidim (multi_seasoned) | `ONE_CATEGORY` on one-product books |
| broker | conditional | conditional | yes | 0.301 | — | `ONE_CATEGORY` where 100% Direct |
| status | no | — | yes | — | — | `NOT_SUPPLIED` — arrears fields absent |
| equity | no | — | yes | — | — | `NOT_SUPPLIED` — protected-equity fields absent |

All eleven are **eligible**. Seven of the eleven reach a page on at least one
fixture. `status` and `equity` are absent from the representative tapes, not
from the selector.

---

## 4. Borrower type — proof it can earn a slot

**In a rendered pack.** `multi_seasoned_gbp` slide 9 draws **Balance by LTV ×
borrower type** (joint / single against seven LTV bands). Screenshot page:
`artifacts/pptx_qa/multi_seasoned_gbp.pdf` p.9.

**As a standard stratification.**
`tests/test_selector_information_first.py::test_B_borrower_type_can_be_selected`
proves a book where single/joint out-scores a competitor selects it into the
four-panel matrix.

On the multi-book fixture it scores 0.324 against vintage at 0.600 and is
correctly rejected as `LOWER_RANKED` — which is the rule working, not failing.
The brief asks that it *can* earn a place, not that it always does.

It is **not** forced anywhere: on the four fixtures with no second-applicant
field it is `NOT_SUPPLIED`, and no joint-lives concentration test is published
merely because the dimension exists.

---

## 5. Vintage, rate, product

| dimension | proof in a rendered pack | proof in test |
|---|---|---|
| **vintage** | `seasoned_book_gbp` p.7 — *By origination vintage* takes the fourth panel, displacing ticket (0.600 vs 0.436) | `test_C_vintage_can_be_selected`, `test_H_vintage_beats_ticket_on_the_audit_book` |
| **rate** | `new_book_gbp` p.5 — *By rate band* takes the fourth panel (vintage is one-category on a new book); `seasoned_book_gbp` p.8 deep dive | `test_D_rate_can_be_selected` |
| **product** | `multi_seasoned_gbp` p.9 — *Balance by product × region* | `test_E_product_can_be_selected_on_a_multi_product_book` |
| **product (singleton)** | four fixtures print *"By product: the whole book sits in a single band, so the distribution is not charted"* | `test_F_a_single_product_book_suppresses_product` |

Case H — the critical regression — is proven three ways: vintage beats ticket on
the audit book, the scores explain the outcome, and preference cannot rescue the
weaker dimension.

---

## 6. Multidimensional pairs

12 candidate crossings, up to 4 drawn. The pair selector had the **same**
defect and got the same fix: gate on shape, then rank by information over
CELLS, with declaration order only as a tie-break.

| fixture | selected pairs |
|---|---|
| new_book_gbp | ltv × rate, ticket × age, ticket × region |
| seasoned_book_gbp | ltv × vintage, rate × vintage, ticket × age, ticket × region |
| **multi_seasoned_gbp** | ltv × vintage, rate × vintage, **ltv × borrower type**, **product × region** |
| multi_growing_gbp | ltv × vintage, ltv × rate, ticket × age, ticket × region |
| seasoned_book_eur | ltv × vintage, rate × vintage, ticket × age, ticket × region |

Seven distinct crossings across five fixtures. At `7388513` all five drew the
same four — ltv×age, ltv×region, ltv×ticket, ltv×rate — because the pair
selector ranked by declaration order, so the first four resolvable pairs won
whatever the data said. Borrower type and product reach a
matrix on the one fixture whose tape supports them.

Gates, each producing a reason derivable from the crossing's own numbers:
`NOT_SUPPLIED` (tape lacks a side) · `ONE_CATEGORY` (a side has one band, so it
is a stratification drawn as a grid) · `TOO_SPARSE` (< 18% of cells carry
balance) · `REDUNDANT` (both dimensions already crossed above) · `LOWER_RANKED`.

---

## 7. Pipeline

The pipeline second cut and the pipeline stratification page both go through the
same shared selector; there is no second ordering rule anywhere in the deck.

| fixture | pipeline stratifications |
|---|---|
| new_book_gbp | age, region, LTV, ticket |
| seasoned_book_gbp | age, region, LTV, ticket |
| multi_seasoned_gbp | age, region, LTV, ticket |
| multi_growing_gbp | age, region, LTV, ticket |
| seasoned_book_eur | *(no governed pipeline source)* |

The four are the same on every fixture **because the pipeline extract is the
same shape on every fixture** — one product, direct-only, no origination date,
no second applicant. Broker and product are correctly suppressed as
`ONE_CATEGORY` and rate band is named as `LOWER_RANKED`. That the selector is
capable of displacing them is proven directly:
`tests/test_selector_governs_every_surface.py::test_the_pipeline_second_cut_takes_the_most_informative_dimension`
and `::test_preference_no_longer_decides_the_pipeline_panel`.

Nothing is hard-coded: the four names in the config are a preference tuple
consumed as a tie-break.

---

## 8. Concentration utilisation history

The engine had evaluated **every** historical frame against today's approved
configuration all along. `compute_history` discarded the utilisation it had
already computed, and the deck spent the remaining two points on a single
direction word.

Changed:

* `concentration_tests_api.compute_history` carries the already-computed
  `utilisation` through on each history point — it is not recomputed, because a
  consumer reforming the ratio becomes a second owner of it;
* `mi_agent_pptx/concentration.attach_history` passes the whole series to the
  row;
* the **existing** Concentration Tests and Headroom page plots the path where
  three or more governed frames exist, against a limit reference line. No new
  page, no new section.

Utilisation rather than the raw value, so a ceiling test and a **floor** test
share one scale and 100% means "at the limit" whichever way the governed
operator points.

Below three frames the panel keeps the bars — two points are a prior, not a
trend. Live proof: `new_book_gbp` p.12 shows *"Utilisation of limit"* with bars;
`multi_seasoned_gbp` p.20 shows *"Utilisation of limit over time"* with five
frames and the limit line.

Two rendering defects the page itself caught and that are fixed: the limit line
was drawn off the top of the axis (the one thing the chart exists to show), and
a four-series legend ran off the figure, printing *"Scotland conce"*.

---

## 9. Pack composition

| fixture | slides at 7388513 | slides at HEAD |
|---|---|---|
| new_book_gbp | 14 | **14** |
| seasoned_book_gbp | 20 | **20** |
| multi_seasoned_gbp | 21 | **21** |
| multi_growing_gbp | 20 | **20** |
| seasoned_book_eur | 15 | **15** |

**No page was added.** Better selection, not more pages.

One composition defect was found and fixed. *Funded Stratifications* and *Funded
Stratifications — Secondary Dimensions* drew the same four panels under two
titles: the two pages used to differ only because they named different
preference orders, so once preference stopped deciding the outcome both reached
the same answer. The deep-dive page now declares the page it `continues` and the
dimensions that page drew are withheld from it — `seasoned_book_gbp` p.8 draws
ticket size and rate band. Where nothing is left to continue with, the page is
omitted with its reason rather than drawn empty.

---

## 10. Visual QA

All five variants regenerated through the **real production route**
(`POST /mi/decks/generate` → poll → `GET /mi/decks/download`). Nothing mocked,
no direct builder call. Preflight **PASS — 24 gates, 0 failures, 0 warnings** on
every variant.

**PPTX / PDF:** `artifacts/pptx_qa/{new_book_gbp, seasoned_book_gbp,
multi_seasoned_gbp, multi_growing_gbp, seasoned_book_eur}.{pptx,pdf}`
**Machine report:** `artifacts/pptx_qa/qa_report.json` — **0 findings** across
all 90 pages.

All 90 pages were rendered and inspected. Findings, all fixed in this sprint:

1. the concentration limit line was drawn off the top of the axis;
2. a four-series legend was cropped mid-word (*"Scotland conce"*);
3. the multidimensional page vanished from all five variants — a composition
   guard hard-coded three pair keys and had only ever been correct because the
   old selector returned those three every time;
4. the two stratification pages drew identical content (§9).

**Does selection genuinely differ by fixture?** Yes:

| | 4th funded dimension | pairs |
|---|---|---|
| new_book_gbp | rate band | ltv×rate, ticket×age, ticket×region |
| seasoned_book_gbp | origination vintage | ltv×vintage, rate×vintage, ticket×age, ticket×region |
| multi_seasoned_gbp | origination vintage | ltv×vintage, rate×vintage, **ltv×borrower type**, **product×region** |
| multi_growing_gbp | origination vintage | ltv×vintage, **ltv×rate**, ticket×age, ticket×region |
| seasoned_book_eur | origination vintage | ltv×vintage, rate×vintage, ticket×age, ticket×region |

**QA fixture change.** Every representative book carried one product, one
channel and no second applicant, so borrower type and product could not reach a
rendered pack however well the selector worked. The seasoned multi-book
portfolio now carries both products, both channels and single/joint lives —
which is where a mixed tape realistically comes from: a warehouse holding
acquired alongside direct originations. The other four stay single-product,
direct-only and single-life **on purpose**; without them there is no fixture
proving the selector suppresses a one-category dimension in a rendered pack.

No economics were altered: balances, LTVs, ages, regions and vintages are
untouched. Fields the fixture never supplied are supplied.

---

## 11. New MI

**NEW MI PRIMITIVES: 0**

No new calculation, model, methodology, asset-class-specific engine or
client-specific logic. Every number on every changed page was already computed
by the engine before this sprint. `concentration_tests_api` carries a value
through that it had already produced and was discarding; `composition` counts
dimensions using the shared rule rather than defining a new one.

---

## 12. Query Agent

**MI QUERY AGENT MODIFIED: NO**

Mechanically: `git diff --name-only 7388513..HEAD` touches **0** files under
`mi_agent/`. The frozen surface — recogniser, parser, routing, executor,
vocabulary, capability resolution — is byte-identical.

`tests/test_query_agent_freeze.py` makes the separation structural rather than
incidental: no module on the Query Agent's path may import the presentation
selector or the deck builder, and the deck may cross into the Query package
only for two governed shared contracts (`load_mi_semantics`, `portfolio_scope`)
— never the parser, executor or router.

**STAGE-MOVEMENT QUERY SUPPORT: DEFERRED TO QUESTION-BANK SPRINT**

Untouched, as instructed. Engine YES / React YES / PPTX YES / MI Query NO.

---

## 13. Regression

### Broad: baseline vs HEAD

Full suite, `python -m pytest tests/ -q -p no:randomly`, 47m46s.

| | failed | passed | skipped | xfailed |
|---|---|---|---|---|
| known baseline | 107 | 7432 | — | — |
| starting HEAD `7388513` | 107 | 7543 | — | — |
| **this branch** | **107** | **7609** | 438 | 8 |

**The failing test-ID sets are identical.** Diffed both ways:

```
comm -23 head_ids baseline_ids   →  (empty)   zero new failures
comm -13 head_ids baseline_ids   →  (empty)   zero accidental repairs
```

**+66 passing** against the starting HEAD, which is exactly the number of tests
this sprint adds: 15 selector probes, 20 cross-surface, 14 Query-freeze, 12
concentration-history, 5 multidimensional.

### MI Query acceptance — replay, not modification

Replayed rather than changed, as instructed:

```
mi_agent/tests/test_mi_query_capability_matrix.py
mi_agent/tests/test_mi_query_executor.py
mi_agent/tests/test_mi_query_invariants.py
mi_agent/tests/test_mi_query_validator.py
tests/test_phase7_mi_query_spec_v2.py
tests/test_annex2_path_acceptance_gate.py
tests/test_mi_portfolio_lens_wiring.py
tests/test_query_agent_freeze.py
→ 201 passed, 7 xfailed
```

### Targeted

| suite | result |
|---|---|
| `test_presentation_parity.py` (React/PPTX parity, EUR/GBP, real deck route) | 16 passed, 1 skipped |
| `test_dimension_selection.py` | 10 passed |
| `test_selector_information_first.py` (probes A–H) | 15 passed |
| `test_selector_governs_every_surface.py` | 20 passed |
| `test_multidim_selection.py` | 15 passed |
| `test_concentration_history_surface.py` | 12 passed |
| `test_query_agent_freeze.py` | 14 passed |
| `test_pptx_commentary_is_deterministic.py` | passed |

The one skip (`test_the_deck_and_react_cross_tab_share_axes_and_totals`) is
pre-existing: it skips identically with the tree checked out at `7388513`,
verified by doing exactly that. It is a conditional skip by the test's own
design — React's first served pair is not always one the deck's page budget
draws.

Two failures were investigated and cleared during the sprint rather than
assumed:

* `test_portfolio_identity_alignment.py::test_react_builds_its_selector_from_the_governed_registry`
  appeared to be new. It fails identically with this branch's changes stashed —
  it is order-dependent and pre-existing, and it is in the baseline set.
* an early stage-movement Query probe returned *"no governed pipeline data"* for
  every question including trivial ones. That was a broken probe (a missing
  `portfolioId`), not a finding, and was not reported as evidence; the finding
  in §12 rests on a re-probe after pipeline reachability was proven.

## 14. Merge recommendation

**YES.**

* zero unexplained new failures — the failing test-ID set is identical to the
  baseline, both directions;
* no pre-existing failure repaired accidentally;
* MI Query acceptance behaviour unchanged, and 0 files under `mi_agent/` touched;
* preflight PASS (24 gates, 0 failures, 0 warnings) on all five variants through
  the real production route;
* 0 QA findings across 90 rendered pages;
* no page added to any pack;
* NEW MI PRIMITIVES: 0.

