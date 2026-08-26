# MI Agent — Basic Compositional Completeness

**READY FOR REAL-CLIENT-DATA ACCEPTANCE.**

Branch `claude/mi-query-agent-c7-2tlhr6`; baseline `2309555`.

---

## 0. Baseline, reproduced before any change

| surface | expected | reproduced |
|---|---|---|
| CFO bank (91) | EXACT 66 · DISCLOSED 2 · TRUE_REFUSAL 13 · FALSE_REFUSAL 10 · WRONG/SILENT 0 | ✅ exact |
| supplement (24) | WRONG/SILENT 0 | ✅ |
| categorical sweep (69) | SILENT DROP 0 | ✅ |
| MI regression | 278 modules · 5957 passed · 81 failed · 4 errors · 85 names · 0 introduced | ✅ exact |
| census · detector · canary · guards | 0 · 0/2 · 0 · green | ✅ |

---

## A. Contract extension

`OperationClaim.type` says what KIND of answer is wanted. It does not say
whether the reader named an **analysis**. Measured field by field on the shipped
path, these two carried identical contracts:

```
                          "Which region has the      "What is the largest geographic
                           largest balance?"          area concentration?"
operation.type            ranking                    ranking
ordering_direction        increase                   increase
ordering_basis            absolute                   absolute
ordering_limit            None                       None
ordering_of / modifiers   None / ()                  None / ()
subject.candidate         current_outstanding_balance current_outstanding_balance
subject.span              whole question             whole question
dimensions                [(collateral_geography,    [(collateral_geography,
                            'grouping'), …]            'grouping'), …]
residue                   []                         []
```

So nothing governed separated a ranked stratification from a specialist
concentration analysis, and route ownership fell back to wording tests inside
the routing layer.

**`OperationClaim.analytic`** is the minimum distinction, with
`ANALYTICS = ("concentration",)`.

**Why it is generic, not geography-specific:**

* it names a **shape of analysis** — how the book distributes across a governed
  dimension family — and says nothing about *which* family. `schema.py` contains
  no geography, product or broker word, and `tests/test_analytic_intent_contract
  .py::test_the_rule_names_no_dimension_family` asserts the entitlement rule's
  executable body contains none either;
* the vocabulary is **not new**. It is
  `mi_workflows.concentration_analysis`'s own, split out as
  `names_a_concentration_analytic` — the four positive tests that module already
  used, with route precedence removed. `is_concentration_question` still applies
  precedence first and then asks it, so the concentration route's recognition is
  behaviourally unchanged;
* it is **orthogonal to `type`**, which is what the mixed case requires: *"the
  largest geographic area concentration"* carries `type=ranking` **and**
  `analytic=concentration`, and both survive.

One vocabulary addition inside the owner: `<family> exposure` and
`most/least exposed`, in the same shape as the existing `<family> mix`
construction. Bounded to a family word, so *"what is our exposure to Wales"* —
exposure **to** a value — is untouched.

**Discrimination, 20/20:** ten generic shapes read `None`, ten specialist shapes
read `concentration`.

---

## B. Basic composition — before and after

| question | required primitives | before | after |
|---|---|---|---|
| Which region has the largest balance? | measure + axis + direction | `geo_exposure` → "I can't build a geographic exposure view" | **Scotland has the highest Current Outstanding Balance: £28.9MM (7 groups)** |
| Which region has the smallest balance? | as above, ascending | as above | **South West has the lowest … £20.5MM** |
| What are the top three regions by balance? | + ordering limit | 7 rows returned, limit dropped | **3 rows** |
| Show the pipeline by stage. | dataset + axis + default measure | "couldn't map this question to a governed analytic" | **Total Balance grouped by Pipeline Stage, 5 groups** |
| What is the balance of offer stage cases? | dataset + stage population + measure | "Offer (Pipeline Stage) — this narrowing was not applied" | **Total Balance · Pipeline Stage = OFFER · 3 loans** |
| How many cases are at offer stage? | + count | "'cases offer' is not a governed measure" | **3 loans** |
| What proportion of the book is in the acquired portfolio? | share + scope population | "an absolute figure was calculated … not its proportion" | **31.8%** (direct 68.2%; they sum to the book) |
| pipeline amount by region | dataset word ≠ measure | `ok=false`, "'pipeline' is not a governed measure" | **delivered** |
| What is the value of outstanding offers? | measure genuinely ambiguous | "I couldn't map this question to a governed analytic" | **"'value' could mean more than one governed measure in this dataset (Balance or Valuation). Say which one…"** |

**Reasons**, all of the same kind — a fact that already existed, carried to
where it was needed:

1. **Route entitlement** now reads the contract instead of wording.
2. **`"stage"`** named no governed dimension: every synonym was two words, and
   the bare word sits on the over-generic token list. Curated exactly as
   `"region"` is — no other governed field claims it.
3. **`"pipeline"`** was fed to the measure parser while `"book"` and
   `"portfolio"` were analytical framing. It names the dataset the answer is
   built *from*, so it is framing too.
4. **The governed stage** was read once, carried to the contract, consumed by
   the routed paths and lost on the point-in-time path. Now applied where the
   loaded dataset carries the column — which preserves the measured decision not
   to assert it globally (35 of 39 stage-naming questions route elsewhere, 10 to
   forecast, where "completion" is a time concept).
5. **A share** whose population is a governed portfolio scope rather than a row
   value now requests the share; the predicate arrives from the scope owner
   before execution, so the numerator is the scope and the denominator stays the
   whole book.
6. **A book value** is no longer read as an unresolved measure; an ambiguous
   measure word is named rather than met with a blank refusal.

**Two ordering defects were found and fixed while wiring this**, both caught by
existing guards rather than by inspection: the dimension-term loop claimed
`pipeline_stage` ahead of its owner and suppressed the role (which moved
"pipeline evolution by stage" off its route), and moving the owner ahead of the
loop let it outrank a genuine dimension — it also reads *"declined"* as
`WITHDRAWN`, so a region period-change question was ranked by pipeline stage.
**The compound canary caught that one.** The loop now yields that single key to
its owner; every other claim order is untouched.

---

## C. Top-N carried end to end

`lexical.ordering_request` "owns direction, basis and limit for the whole
estate" — its words — and the parser had a **digits-only regex of its own**.
They disagreed:

```
"What are the top THREE regions by balance?"
    contract   ordering_limit = 3
    spec       top_n          = None      ->  all seven regions, nothing said
```

The parser now asks the owner. Not a widening: over the numeric forms the owner
returns exactly what the regex did; it additionally reads the words a reader
writes.

| case | rows | | case | rows |
|---|---|---|---|---|
| Top 1 region by balance | **1** | | Smallest 3 regions by balance | **3** |
| Top 2 regions by balance | **2** | | Largest 3 regions by balance | **3** |
| Top 3 regions by balance | **3** | | Bottom 3 brokers by balance | **3** |
| top **three** regions by balance | **3** | | Show balance by region (no limit) | **7** |
| Top 5 regions by balance | **5** | | | |

Direction and limit compose: *smallest 3* returns South West / London / South
East; *largest 3* returns Scotland / North / Midlands.

---

## D. Success envelopes

`ok: true` must mean the requested analysis was delivered. A structural scan of
every `_envelope(ok=True, …, artifacts=[])` call site found **26**:

| disposition | count | examples |
|---|---|---|
| **converted to `ok:false` + `controlledUnsupported`** | **23** | "I can't build a geographic exposure view…", "No weekly pipeline extracts are available…", "Contractual risk limits are unavailable…", "I couldn't resolve a dimension to attribute the bridge by." |
| **left as delivered** | 3 | two empty-population statements ("There are no funded loans in *X*") — an analysis that ran over an empty population; and the run-rate branch that reports the current balance and discloses that it could not extrapolate |

No new public taxonomy: `ok:false` + `metadata.controlledUnsupported` (HTTP 200)
is the estate's existing contract for "I will not answer that", already used by
`_capability_unavailable_envelope`. All three existing markers are stamped.

**Two guards were widened to match.** `_guard_unresolved_scope` and
`_guard_routed_answer` only ran on delivered answers — sound while a route that
could not deliver still returned `ok=True`. Once such an envelope became a
controlled refusal, the guards stopped running and a reader asking about a book
this platform has never onboarded was told *"at least two funded reporting
periods are needed for a bridge"*. They now also run on a controlled
non-delivery. An **execution failure is still excluded**: a route that broke has
adjudicated nothing.

**Success-shaped non-answers across every measured surface: 0.**

Three tests asserted the old envelope and were updated with the reason recorded
in each. One of them —
`test_an_unqualified_measure_on_the_pipeline_is_now_unreachable` — had written
its own instruction: *"Asserted so that the day it is fixed, this test fails and
says so."*

---

## E. Safety surfaces

| surface | before | after | gate |
|---|---|---|---|
| CFO bank (91) | EXACT 66 · FALSE_REFUSAL 10 · TRUE 13 · DISCLOSED 2 · **WRONG/SILENT 0** | EXACT **68** · FALSE_REFUSAL **8** · TRUE 13 · DISCLOSED 2 · **WRONG/SILENT 0** | ✅ |
| simple-composition bank (36, new) | — | **35 EXACT · 1 pre-registered refusal · 0 wrong** | ✅ |
| supplement (24) | CORRECT 20 · **WRONG/SILENT 0** | CORRECT **21** · **WRONG/SILENT 0** | ✅ |
| categorical sweep (69) | 56 correct · **SILENT DROP 0** | 56 correct · **SILENT DROP 0** | ✅ |
| collision sweep, real values (46) | 42 correct · SILENT DROP 0 · WRONG ADDITIONAL CLAIM 2 | identical | SILENT DROP ✅ |
| collision sweep, synthetic (58) | 54 correct · SILENT DROP 0 · WRONG ADDITIONAL CLAIM 2 | identical | SILENT DROP ✅ |

**The two WRONG ADDITIONAL CLAIMs are unchanged and are the documented
single-token `origination_channel = direct` residual**, which this task was
instructed to preserve and did not touch. Named here rather than buried: it is
the one wrong-claim count on any surface that is not zero.

**Explicit constraints, across all banks: ignored measure 0 · ignored population
0 · ignored filter 0 · ignored period 0 · ignored ranking direction 0 · ignored
ranking limit 0.**

Preserved and re-checked: Gamma Direct span ownership (147 broker-only; 104 with
the Direct book), lump_sum 396, drawdown 244, largest/smallest grouped ranking
direction, leading filter clauses, `broker_channel`, scalar-answer UX, average
balance over time, WA interest rate at 6.26%, forward forecast horizon, pipeline
population labelling, source-portfolio scope.

### Mutation controls

| mutation | expected | observed |
|---|---|---|
| 1 — remove the entitlement exclusion | generic ranking recaptured | `geo_exposure` claims "Which region has the largest balance?" again |
| 2 — a specialist request read as generic ranking | specialist ownership fails | `test_geographic_exposure_routes_to_itl3_engine` fails |
| 3 — fault the specialist handler after claim | fail closed, no second route | `ok:false`, *"I have not answered your question with a different analysis instead"* |
| 4 — reverse the ranking direction | truth test fails | 5 of 9 direction tests fail |
| 5 — drop the ordering-limit delegation | limits ignored | 5 simple-bank rows become WRONG/SILENT |

All restored; suites green afterwards.

---

## F. False refusals remaining — all **ACCEPTABLE CAPABILITY LIMIT**

| question | classification | why |
|---|---|---|
| Which region added the most balance since last month **for loans with LTV above 50%** (×2 phrasings) | ACCEPTABLE | The period-change route selects a population **by scope, not by row predicate** — a documented structural property. Filtering both snapshots before a movement ranking is implemented nowhere; this is not a route table failing to compose an existing primitive. It refuses naming the predicate in the reader's words. |
| Show product concentration. | ACCEPTABLE | A **correct governance refusal**: the registry marks `erm_product_type` originator-specific and the scope spans two portfolios, so combining them would compare unlike categories. Scoped to one book (`…for the direct book`) the same request delivers — which is what makes it a decision, not a gap. |
| Show broker concentration. | ACCEPTABLE | Same rule, same field flag on `broker_channel`. Pre-registered as `REFUSE` in the new bank. |
| How has the pipeline evolved? | ACCEPTABLE | Names **no measure**. The estate refuses measure-less trend questions by an explicit, measured guard — widening the axis vocabulary must not quietly answer "Total Balance". *"How has the pipeline balance evolved?"* delivers a 5-period series. |
| What is the value of outstanding offers? | ACCEPTABLE | **Genuinely ambiguous**: this tape carries both `current_outstanding_balance` and `current_valuation_amount`. Now an actionable clarification naming both, not a blank refusal. Both unambiguous forms deliver. |
| Are any of our concentration limits at risk? · Which of our limits are currently most at risk? | ACCEPTABLE | **Data genuinely absent** — no Schedule 8 limits extracted for this fixture. These moved EXACT → FALSE_REFUSAL *because of section D*: they were success-shaped non-answers, and the frozen bank pre-registered `DELIVER` when the envelope still looked like one. The behaviour is more correct; the oracle counts an honest refusal as a false one. **The frozen bank was not edited.** |

**COMMERCIAL BASIC-GAP: none.**

---

## G. Regression

| | frozen | after |
|---|---|---|
| passed | 5957 | **5957** |
| failed | 81 | **81** |
| skipped / xfailed / errors | 711 / 15 / 4 | **711 / 15 / 4** |
| hung | 0 | **0** |
| failing names | 85 | **85** |

**introduced = 0 · resolved = 0**, exact-name diff empty both ways, 278 modules,
run alone. Seven failures appeared mid-work and every one was resolved rather
than absorbed — two guard preconditions, one series-vocabulary conflict, one
claim-ordering defect and three superseded assertions.

### Architecture controls

| control | result |
|---|---|
| post-claim raw-question semantic reads | **0** in all eight categories |
| route-local semantic vocabularies | **0** — the new distinction lives on the contract and routes read it there |
| route substitution detector | **0 of 2** |
| compound canary | **intact**, 0 breaches |
| semantic / migration guard suites | **green** |

---

## H. Manual CFO review — 15 questions

Twelve read as a CFO would want. Three carry a note, none of them wrong:

| question | reading |
|---|---|
| What is our total funded balance? | £172.1MM · 640 loans ✅ |
| What is our weighted average LTV? | 36.3% ✅ |
| Summarise the portfolio. | balance, LTV, rate, age, largest regional exposures ✅ |
| Which region has the largest / smallest balance? | Scotland £28.9MM / South West £20.5MM ✅ |
| What are the top three regions by balance? | 3 groups, led by Scotland ✅ |
| Show broker concentration. | governed refusal, explained ✅ |
| What share of the book is drawdown? | 38.7% of 244 of 640 ✅ |
| How does the current month compare with the previous month? | delivered; **opens with "Youngest Borrower Age −0 (74 → 74)"** — true, uninformative. Cosmetic. |
| What is the pipeline balance? | £3.6MM · 8 cases ✅ |
| Show the pipeline by stage. | 5 stages ✅ |
| What is the value of outstanding offers? | actionable clarification naming Balance or Valuation ✅ |
| How many Gamma Direct loans? | 147 ✅ |
| How many lump sum loans? | 396 ✅ |
| Which region added the most balance since last month? | Scotland £16.5m → £28.9m (+£12.4m, +75.0%) ✅ |

No answer exposes a route name or implementation rationale.

---

## Verdict

**READY FOR REAL-CLIENT-DATA ACCEPTANCE.**

Generic region ranking composes; specialist geographic exposure remains
specialist and is proven load-bearing by mutation; no post-claim route
substitution was introduced and fail-closed dispatch is re-proven; explicit
ordering limits are honoured end to end; no success-shaped non-answers remain on
any surface; no wrong/silent or wrong/disclosed answers on the frozen bank, the
supplement, the categorical sweep or the new simple-composition bank; every
remaining false refusal is a genuine capability or data limit; the authoritative
MI regression is identical to its frozen baseline; architecture controls are
green.

**Synthetic MI capability development is frozen. No additional synthetic feature
work is authorised before the real-client-data acceptance run.**

---

## Real-data handoff checklist

*(from `docs/mi_pre_real_data_cleanup.md`, unchanged and not executed here.
Items 1 and 3 are now closed by this task and the previous one and are marked
so.)*

**Before the first real book**

1. ~~Resolve the geographic route-ownership blocker.~~ **CLOSED** — the contract
   now separates a ranked stratification from a specialist concentration
   analytic, and an un-built specialist view no longer reports `ok: true`.
2. **Decide the single-token scope/value precedence.** `origination_channel`
   carries the value `direct`; so does the scope vocabulary. "How many direct
   loans do we have?" answers 441 (the direct book) where the value is 146. Its
   sibling `broker` already discloses the ambiguity. Deliberately untouched and
   pinned by eleven tests across P1I/P1J-1/P1L/P1N. **The only non-zero
   wrong-claim count on any surface.** Check the real book's
   `origination_channel` values before go-live: if it does not carry `direct`,
   the collision does not arise.
3. **Run both collision sweeps against the real book on day one.**
   `coll_sweep.py` generates its questions from the book's own values, so it
   needs no editing. Gate: SILENT DROP 0, WRONG ADDITIONAL CLAIM 0.
4. **"What is the balance in Atlantis?" on the API path.** The workflow path
   refuses; the API path answered over the whole book at baseline for categories
   the prepositional pattern does not reach. Not introduced here, not closed
   here.
5. **Business Semantics Registry entries for the client's dimensions.** The
   registry's `display_name` for `erm_product_type` renders as *"erm product
   type"* in prose. Configuration.
6. **Pipeline mapping.** The governed pipeline view must expose the client's
   stage column as `pipeline_stage`. On this fixture it does; on the client tape
   it must be mapped at onboarding.

**Bounded work, once the above is settled**

7. Filtered ranked movement: apply the contract's `RowPredicateClaim` in the
   period-change execution path.
8. ~~A share whose narrowing is a portfolio scope rather than a row predicate.~~
   **CLOSED** by this task.
9. "What is the total book value?" — "value" read as a count.
10. The period-comparison summary should lead with the movement that matters,
    not the first metric in registry order.

**Standing**

11. Freeze a new MI regression baseline from the real book's fixture, and keep
    the 85-name frozen list as the denominator until it is deliberately
    replaced. Never report "introduced 0" from a run against a tree still being
    edited, or from one taken under load — two such runs were excluded from this
    programme for exactly that reason.
