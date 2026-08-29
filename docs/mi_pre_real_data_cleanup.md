# MI Agent — Final Pre-Real-Data Cleanup

**Verdict: NOT READY FOR REAL-CLIENT-DATA ACCEPTANCE.**
One specific blocker, named in §9. Everything else in the brief is closed and
measured; the blocker is a guard I was told not to change without reporting
first, and this is that report.

Branch `claude/mi-query-agent-c7-2tlhr6`. Baseline for every "before" number is
commit `b138cc4`.

---

## 1. Safety baseline, reproduced before any change

Reproduced exactly at `b138cc4`, so the STOP condition ("CLEANUP BASELINE NOT
REPRODUCED") did not fire:

| control | baseline |
|---|---|
| post-claim analytical-meaning census | 0 in every category |
| route substitution detector | SUBSTITUTIONS 0 of 2 |
| architecture guard suite | 68 passed |
| CFO acceptance bank (91) | EXACT 64 · FALSE_REFUSAL 12 · TRUE_REFUSAL 13 · DISCLOSED 2 · **WRONG 0** |
| generalisation supplement (24) | CORRECT 19 · SAFE REFUSAL 5 |
| categorical sweep (69) | CORRECT NARROWING 49 · HONEST REFUSAL 16 · UNCLEAR 4 · 0 silent drops |
| MI regression (278 modules) | 5957 passed · 81 failed · 711 skipped · 15 xfailed · 4 errors · 0 hung · **85 failing names, frozen** |

---

## 2. The Gamma Direct collision, closed generically

### What was wrong

    "How many Gamma Direct loans do we have?"   ->   104

The categorical parser correctly claimed `Gamma Direct` as a value of
`broker_channel` (147 loans). The portfolio-lens resolver, reading the SAME raw
string with its own qualifier/noun grammar, independently matched `Direct
loans` and narrowed the population to the `direct_001` book as well — 104 of
those 147. Two resolvers, one span, and neither could see the other.

### Span ownership, traced

The first point at which the categorical span is reused as scope evidence is
`mi_agent/portfolio_lens.py::lens_phrase_spans`, consulted by `resolve_lens`:

    'How many Gamma Direct loans do we have?'
       categorical value span   (9, 21)  'Gamma Direct'
       lens_phrase_spans       [(15, 27, 'Direct loans')]     <-- overlaps
       resolve_lens -> direct  {'source_portfolio_type': 'direct'}

### The invariant, and where it lives

> Once a contiguous span has been claimed as a governed categorical value, the
> tokens INSIDE that span must not independently create another semantic claim,
> unless the grammar explicitly establishes a second meaning.

`mi_agent/categorical_spans.py` is the ONE owner: which spans a value has
claimed, and which values may claim one. `llm_query_parser._categorical_value_
field` is now a thin alias for it, not a second resolver.

Three limits keep the claim honest, each of them measured rather than assumed:

* **Unambiguous only.** A value two governed fields both claim has not been
  claimed by anything, so it may not silence anything.
* **Multi-word only, counted on WHITESPACE.** A single-token span has no
  "inside". Counting underscores as separators made `direct_002` two words and
  let a book value mask an explicit cohort id —
  `test_mi_query_lens_matrix::test_an_exact_cohort_id_in_the_question_wins`
  caught it.
* **Never a resolver's OWN fields.** The rule governs a collision between two
  DIFFERENT owners. A fixture whose `source_portfolio_label` is literally
  "Direct Book" turned "show direct book balance" into Total until the scope
  owner was excluded from its own catalogue.

### Where it is applied

Every reader that matches a vocabulary of its own against the raw sentence now
reads the owned sentence:

| reader | why it owns no book field |
|---|---|
| `portfolio_lens.resolve_lens` + `mentions_portfolio` (the precedence gate) | scope, minus its own four fields |
| `question_interpretation.projection` — `SourceScopeClaim`, `DatasetClaim` | the contract the 7 migrated routes plan from |
| `mi_agent_api.workspace.resolve_dataset` (masked by the CALLER — see §5) | forecast / pipeline / funded |
| `recogniser_registry.RouteRequest.for_recognition` | recognition is pre-claim; handlers keep the raw sentence |
| `mi_workflows.analytical.intent.settle` and `mi_service._fail_closed_analytical` | analytical families |
| the parser's governed intent recognisers (bridge, cohort, compare, limit, forecast) | route vocabularies |
| `execution_receipt` facets: geography (minus its own fields), comparison period, ranking, stress, threshold, relationship | none |

### Both claims still survive when the grammar states both

    "How many Gamma Direct loans are in the Direct book?"
       -> Broker = Gamma Direct · Source Portfolio in direct_001 · 104 loans

`tests/test_governed_span_ownership.py` (19 tests) pins all of it, including a
structural test that no business value appears in the owner's executable code.
Run against the pre-fix tree with the owner present but unwired, 8 of them fail
— the tests discriminate.

### Four more defects of the same class, closed

1. **The word-level ownership list overruled the book.** `_NON_PLACE_TERMS`
   vetoed a value the book carries because ONE of its words belongs to another
   vocabulary, so `how many Gamma Direct loans` resolved and `what is the total
   balance for Gamma Direct loans` refused. The book's values now decide first;
   the list still runs for a value nothing claims, which is what it was for.
2. **The attributive scan could not see a value containing its head noun.**
   "Pipeline Mortgage Club" split on `mortgage`; "London Bridge Loans" lost its
   trailing `Loans`. The span owner is asked first, and the head noun is now
   captured so a tail may be tried with it.
3. **The dataset was decided before any tape could be opened.** A broker called
   "Pipeline Mortgage Club" served 8 pipeline cases in place of its 63 funded
   loans. The same owner is now asked once more once the book is readable, and
   can only ever return to the default.
4. **The measure-set and ranking branches resolved a category with no
   catalogue.** "Which broker channel has the largest balance for LUMP SUM
   loans?" filtered `geographic_region_obligor = 'Lump Sum'`, matched nothing,
   and refused naming a field the reader never mentioned. Both were the last
   unwired sites of that P0 class.

---

## 3. Collision sweep

Two sweeps. The first is the gate as the brief states it — **real book values**.
The second is a harder probe I built because the real book carries exactly ONE
colliding value, and gating a generic rule on one value is how a patch passes
for a fix: `/tmp/coll_env` is the same 640 loans and the same five snapshots
with the broker column rewritten to ten names that each straddle a DIFFERENT
vocabulary (scope, dataset, stage, geography, period, ranking, movement), every
one of them spanning both books so a leaked claim changes the number.

| sweep | questions | CORRECT | HONEST REFUSAL | **SILENT DROP** | **WRONG ADDITIONAL CLAIM** |
|---|---|---|---|---|---|
| real book values | 46 | 42 | 2 | **0** | **2** |
| synthetic colliders | 58 | 54 | 2 | **0** | **2** |

* **SILENT DROP = 0. Gate met.**
* **WRONG ADDITIONAL CLAIM = 2. Gate NOT met** — and it is the same single case
  in both sweeps, reported in §9 rather than traded away.

Truth is computed from the tape, never from the agent. Two oracle faults were
found and fixed before any of these numbers were believed: the classifier read
the formatted KPI string (`"£24.3MM"`) instead of `rawValue`, scoring every
balance question against its own blindness; and it compared `"second home"` to
the book's `second_home`, turning six correct answers into drops.

---

## 4. Scalar and refusal presentation

**A single figure now says the figure**, led by the measure the SPEC says the
question asked for, built from the same row and the same formatter the KPI
artifact uses so the prose and the artifact cannot disagree:

    before   Here is the result for your query, covering 1 group(s).
    after    Current Outstanding Balance: £24.3MM · 93 loans.
    after    147 loans · Current Outstanding Balance: £40.9MM.      (a count question)
    after    Current Outstanding Balance Share Pct: 38.7% · 244 loans · Population Total: 640.

**Refusals name fields as the registry names them for a reader.** Four
identifier leaks closed:

| before | after |
|---|---|
| `The MI book for this client does not include nneg_flag` | `'NNEG' is not available in this dataset. This book does not report it…` |
| `the population current_loan_to_value gt 50.0` | `loans where Current LTV over 50` |
| `No loans in this book match that filter (geographic_region_obligor)` | `…(Obligor Region (NUTS3))` |
| `originator-specific vocabulary (requires_scale_alignment)` | `each originator spells this dimension's categories in its own vocabulary…` |

`Predicate.spoken()` is a SECOND rendering beside `describe()`, not a
replacement: `question_interpretation/b5_reachability.py` proves a guard
unreachable **because** every population facet's label leads with the field
name, so the label may not become prose. `RequestedFacet.spoken` carries the
prose beside it and only user-facing text reads it.

---

## 5. One guard corrected, and one signature restored

**Corrected — the cohort-comparison trigger.** A third arm matched a bare
`how does the … compare with …`, with no book anywhere in it, and claimed

    "How does the current month compare with the previous month?"

as *"a comparison between two books"*, then refused it for not having compared
them. The correction is measured, not argued: `_COHORT_COMPARISON_FRAMING_RE`
already carries a SUPERSET of that arm and the caller raises the facet on it
whenever `cohort_concepts_named` finds a cohort — and over the estate's **848
corpus questions, exactly one** reached the facet through the third arm and
through nothing else, the month-on-month question itself. Cohort comparisons
still route: "How does the direct book compare with the acquired book?" reaches
`portfolio_risk_comparison`, and the back-book case's refusal is byte-identical
to baseline.

**Restored — `resolve_dataset` takes one argument.** An earlier cut of the
dataset fix added `available_values=`, and two architecture guards failed:
`test_the_resolver_cannot_be_handed_a_tab` ("not 'it ignores the tab' — it has
nowhere to put one") and `test_the_view_reading_lives_in_one_place`. Both are
right and neither was weakened: the masking is applied to the QUESTION by the
caller, and the owner's signature is unchanged.

---

## 6. False refusals

CFO bank, 91 questions:

| | baseline | now |
|---|---|---|
| EXACT | 64 | **66** |
| FALSE_REFUSAL | 12 | **10** |
| TRUE_REFUSAL | 13 | 13 |
| DISCLOSED | 2 | 2 |
| **WRONG / SILENT** | **0** | **0** |

Closed: current-vs-previous month (§5); share of a named category ("what share
of the book is drawdown?" → 38.7%, via the copular categorical form and the
matching copular case of the existing population-qualifier rule); and, outside
the bank, the mis-bound `lump sum` ranking filter.

Also closed, and it is the more important one: **an unknown category now
refuses instead of widening.** Handing the workflow the book's catalogue meant
it correctly declined to invent a geography for "in Atlantis" — and declining
silently would have answered over the whole book. A category the reader named
that no governed field claims is recorded as unresolved and refused, with the
scope owner left to speak for an unheld PORTFOLIO name, which has a refusal of
its own.

**Deferred, with reasons:**

| questions | why deferred |
|---|---|
| "Which region has the largest/smallest balance?" (2) | `geo_exposure` owns them and cannot build an ITL3 view on this tape. Deliberately reverted in an earlier pass; `test_geographic_exposure_degrades_honestly_without_itl3_or_postcode` forbids handing a specialist capability's failure to another answer. **See §9.** |
| filtered ranked movement (2) | the period-change capability applies no row predicate. Closing it means adding population filtering to that execution path — a capability extension, not a bounded fix. It refuses honestly and names the predicate in the reader's words. |
| "Show broker concentration." | `broker_channel` is not governed as a concentration dimension in this deployment's Business Semantics Registry. A registry entry, not code. |
| "Show product concentration." | **not a false refusal.** The registry's declared asset class is now honoured (that blocker is fixed), and what remains is a correct governance refusal: `erm_product_type` categories are originator-specific and the scope spans two portfolios. Scoped to one book it answers. |
| "What proportion of the book is in the acquired portfolio?" | a share whose narrowing is a portfolio SCOPE rather than a row predicate; the share route computes against a predicate. |
| the three pipeline questions | the pipeline extract wired into this environment carries `Status`, but the governed pipeline view exposes no `pipeline_stage` dimension for it. An onboarding mapping, not a semantic owner. |

---

## 7. Architecture controls, after every change

| control | result |
|---|---|
| post-claim analytical-meaning census | **0** in all eight categories |
| route substitution detector | **SUBSTITUTIONS 0 of 2** (boundary derived from the detector's own run) |
| architecture guard suite | **214 passed**, 45 skipped |
| MI regression, 278 modules | **5957 passed · 81 failed · 711 skipped · 15 xfailed · 4 errors · 0 hung** |
| introduced failures vs the frozen baseline | **0** |

The regression is byte-identical to the frozen baseline: same 85 failing names,
same totals. Three failures introduced mid-work were each fixed rather than
absorbed — an Atlantis widening, and the two `resolve_dataset` signature guards.

*(An intermediate regression run against a tree I was still editing is not
reported here; its numbers were not a measurement of anything.)*

---

## 8. Manual commercial review, 15 questions

Read as a lender would read them. Twelve are good. Three carry a note:

| question | reading |
|---|---|
| "What is the total book value?" | answers £172.1MM but the receipt says *Count of loans* — the parse takes "how much" from "value" as a count. The number is right and the receipt is honest about what it did. **Cosmetic.** |
| "How does the current month compare with the previous month?" | answers, and opens with *"Youngest Borrower Age −0 (74 → 74)"* — a true but uninformative lead. **Cosmetic.** |
| "Which region has the largest balance?" | returns **`ok: true`** with *"I can't build a geographic exposure view for this book"*. A success-shaped non-answer. **This is the blocker.** |

---

## 9. THE BLOCKER

**A route that could not build its view returns `ok: true` with a non-answer.**

    "Which region has the largest balance?"
       route geo_exposure · ok TRUE · artifacts []
       "I can't build a geographic exposure view for this book:
        no ITL3 field and no property postcode on the tape."

Meanwhile "Show the balance by region" answers over the same book's seven
governed regions. So a lender asking the ranking form of a question the platform
can answer is told it cannot be answered, in a response the API reports as a
success. Two of the ten remaining bank false refusals are this, and it is the
first thing a real client will hit.

**Why I have not changed it.** Two things stand in the way, and both were put
there deliberately:

* `mi_agent_api/tests/test_chat_routing_e2e.py::test_geographic_exposure_
  degrades_honestly_without_itl3_or_postcode` asserts `ok is True` for exactly
  this envelope.
* An earlier pass made this route DEFER so the generic path could answer, and
  reverted it on the ground that a specialist capability's failure must not be
  handed to a different answer. The revert is documented in
  `chat_routing.py:2198`.

**What I believe the fix is**, offered as a finding rather than a change: the
comment says *"no contract field separates 'where is the book concentrated' from
'which region has the largest balance'"*. There is one now. The contract carries
an `OperationClaim` of type `RANKING` with `ordering_*` fields; the first
question is not a ranking and the second is. That is a governed field, available
from `RouteRequest.resolve_interpretation()`, and it distinguishes them without
any new vocabulary. Whether the geo recogniser should decline a RANKING it
cannot serve — and whether the envelope should be `ok: false` regardless — is a
decision about a guard someone else formulated, and the brief says to stop and
report before changing it. This is that report.

---

## 10. Carry-forward checklist for real-client-data acceptance

Ordered: a wrong answer, then a misleading one, then a refusal, then polish.

**Before the first real book**

1. **Resolve the blocker in §9.** Decide, with the guard's author, whether the
   geographic route declines a RANKING it cannot serve, and whether an
   un-built specialist view may report `ok: true`.
2. **Decide the single-token scope/value precedence.** `origination_channel`
   carries the value `direct`; so does the scope vocabulary. "How many direct
   loans do we have?" answers 441 (the direct book) where the value is 146.
   Its sibling `broker` already does the honest thing — *"I could not tell how
   you meant broker"* — so the mechanism exists. The span-ownership rule is
   deliberately silent here (a single-token span has no "inside"), and the
   scope precedence is pinned by eleven tests across P1I/P1J-1/P1L/P1N. **This
   is the only WRONG ADDITIONAL CLAIM left in either sweep.** Check the real
   book's `origination_channel` values before go-live: if it does not carry
   `direct`, the collision does not arise on that book.
3. **Run both sweeps against the real book on day one.** `coll_sweep.py`
   generates its questions from the book's own values, so it needs no
   editing. Gate: SILENT DROP 0, WRONG ADDITIONAL CLAIM 0.
4. **"What is the balance in Atlantis?" on the API path.** The workflow path
   now refuses; the API path answered over the whole book at baseline and
   still does for questions whose category the prepositional pattern does not
   reach. Not introduced here, not closed here.
5. **Business Semantics Registry entries for the client's dimensions.**
   `broker_channel` is not governed as a concentration dimension, and the
   registry's `display_name` for `erm_product_type` renders as *"erm product
   type"* in prose. Both are configuration.
6. **Pipeline mapping.** The governed pipeline view must expose the client's
   stage column as `pipeline_stage`, or the three pipeline questions stay
   unanswerable.

**Bounded work, once the above is settled**

7. Filtered ranked movement: apply the contract's `RowPredicateClaim` in the
   period-change execution path.
8. A share whose narrowing is a portfolio scope rather than a row predicate.
9. "What is the total book value?" — "value" read as a count.
10. The period-comparison summary should lead with the movement that matters,
    not the first metric in registry order.

**Standing**

11. Freeze a new MI regression baseline from the real book's fixture, and keep
    the 85-name frozen list as the denominator until it is deliberately
    replaced. Never report "introduced 0" from a run against a tree still
    being edited — one such run is excluded from §7 for exactly that reason.
