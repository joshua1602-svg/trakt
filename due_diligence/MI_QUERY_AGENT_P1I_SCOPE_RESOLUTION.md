# MI Query Agent — P1I-A: Governed Scope-Phrase Resolution

**Status:** implementation complete, gate green, not pushed pending instruction.
**Scope:** phrases that name the portfolio scope itself must resolve as SCOPE,
never as a row-level predicate, a place, or a grouping axis.
**Constraint honoured throughout:** prevent the invalid filter from being
created. Nothing is created and then quietly removed.

---

## 1. The defect

Four symptoms, one defect.

| Question | Was resolved as | Effect |
|---|---|---|
| "the entire portfolio" | `collateral_geography = "Entire"` | a place called Entire |
| "the current portfolio" | `collateral_geography = "Current"` | a place called Current |
| "the acquired portfolio" | grouping by `acquired_portfolio_id` | a real registry field whose synonym the phrase matches |
| "the funded portfolio" (LLM path) | `funded_status = "Funded"` | a column this book does not have — a borrower-age question refused for a field nobody asked about |
| "the direct / acquired book" (LLM path) | `source_portfolio_type = "direct"` | a second, coarser scope expressed as a row predicate, bypassing the governed id mechanism |

In every case a phrase naming the **population being reported on** was consumed
by a resolver looking for a place, an axis, or a predicate. The population and
the predicate are different roles; the parser had no way to tell them apart.

---

## 2. What already existed (traced before building)

The brief required tracing the existing scope model before adding anything. It
is substantial and almost all of it was already correct:

* `mi_agent/portfolio_lens.py` already owned the scope vocabulary
  (`_DIRECT_TERMS`, `_ACQUIRED_TERMS`, `_TOTAL_TERMS`) and already documented
  why bare "total"/"overall"/"origination" are deliberately excluded.
* `trakt_core.portfolio.resolve_scope` already computed Direct and Acquired
  membership from governed portfolio **metadata**, and already filtered on
  explicit `source_portfolio_id` values rather than on the type string — so a
  group answers as the sum of its members and a portfolio missing from the data
  cannot silently widen the result.
* The execution receipt and `portfolioCoverage` already carried
  requested / resolved / executed scope.

**No second scope architecture was created.** P1I-A extends the seams that were
already there.

---

## 3. What P1I-A changed

### 3.1 One source of truth for the scope span

`portfolio_lens` now also owns the span itself:

```
scope_phrase_spans(text)   # THE single source of truth — offsets of scope phrases
names_selected_scope(text) # "current book" family — the active UI selection
mask_scope_phrases(text)   # blanks the spans, preserving offsets
```

Vocabulary is a small closed set of `_SCOPE_NOUNS` (book, portfolio, platform,
loan book, aum) crossed with `_SCOPE_QUALIFIERS` (funded, whole, entire, total,
current, overall, consolidated, combined, selected, active, direct, acquired,
purchased, originated, sponsored). A qualifier alone is never a scope: it must
carry a book noun.

### 3.2 The place and axis resolvers claim nothing they should not

`_parse_categorical_filter` and `_explicit_dimensions` in
`llm_query_parser.py` now search a **masked** copy of the question. The span is
blanked, offsets preserved, so no downstream clause boundary moves. The
geography filter and the grouping dimension are never constructed from a scope
phrase, because by the time those resolvers run the scope phrase is not there.

### 3.3 Multi-select scope

Per the rulings in §4, `PortfolioLens` gained `cohort_ids`, `lens_from_selection`
accepts a list, and `trakt_core.portfolio.resolve_scope` gained
`CONTEXT_KIND_SELECTION` so a multi-selection resolves to **exactly** those
books rather than widening to Total.

### 3.4 Role rejection, at parse time, for absent fields

`reject_scope_role_filters()` runs before `validate_mi_query` and is recorded in
parse metadata as `scope_role_rejected`. Two conditions must **both** hold:

* the field is **absent** from the dataset, and
* the question contains a governed scope phrase naming that concept.

A filter that narrows a population can therefore never be removed by it: an
absent column narrows nothing, because the query would not have run.

### 3.5 Role rejection for the governed scope field itself *(added after the first gate run)*

The first live gate exposed a fifth symptom the four above did not cover.
`source_portfolio_type` **is** a real column, so the absent-field test in §3.4
can never reach it — yet a model-emitted `source_portfolio_type = "direct"` is
not a row predicate either. The governed lens resolves the scope, and it
filters on the registry's explicit ids.

It is refused only when it is **exactly redundant** — the same question resolves
to that same type lens — so the governed id filter that replaces it is derived
from the identical phrase and the rejection cannot change the intended
population. Two guards keep it provably safe:

* the **id** field is never rejected. It is the finer grain, so dropping it
  could widen, and nothing here may widen.
* a **disagreeing** predicate is never dropped. That is a conflict, not a
  duplicate, and must not be resolved by quietly deleting one side of it.

---

## 4. Questions asked of the user, and the rulings

The brief instructed me to ask rather than guess where product semantics were
ambiguous. Two were genuinely ambiguous and both were ruled on.

| Ambiguity | Options put | Ruling |
|---|---|---|
| "the funded book / funded portfolio" while a Direct or Acquired workspace selection is active — does it widen to the whole platform, or keep the selection? | widen to Total / keep the selection | **Keep the selection (the funded rows of it).** It must not create a funded-status filter. |
| "the current book / current portfolio" under a multi-select — does it mean the union, the first, or exactly the selected ids? | Total / first id / exactly the selected ids | **Exactly the selected ids, including multi-select** — so the governed lens and scope accept a list rather than widening to Total. |

A third possibility was raised by the user — that the LLM parser could ask the
user to disambiguate ("did you mean the total funded book or the acquired_001
book?"). The ruling was to keep P1I-A deterministic, add no clarification UX and
no response-contract change, and record governed clarification as a candidate
**P1I-B** in this report only. That is done in §11.

---

## 5. Focused scope-resolution bank

`tests/test_p1i_scope_resolution.py` — **53 tests, 53 passing.**

Coverage:

* the vocabulary claims what it should and refuses ordinary language ("book value", "portfolio manager", "the book of business");
* masking preserves offsets exactly;
* a qualifier without a book noun is not a scope;
* funded scope creates **no** row predicate, and keeps the active selection;
* whole / entire book equals the entire AUM and reconciles across every book;
* current book/portfolio is exactly the selected book, including multi-select;
* a multi-selection never falls back to Total;
* an unrecognised selection still widens **and says so**;
* Direct / Acquired membership comes from governed metadata, not from id spelling;
* a book aggregates every member;
* scope never becomes a cohort synonym; a real geography survives; a scope phrase and a place coexist;
* a genuine predicate on an absent field still refuses **by name** rather than being dropped;
* a real column is never rejected;
* **(new)** the governed scope field is not a row predicate; the id is never rejected; a disagreeing predicate is not silently dropped; a scope predicate survives when no governed scope phrase is present; and rejecting the type predicate does not change the population.

---

## 6. Repeated genuine-LLM gate

Ten runs per phrase through the real model, with `zero_cost_first` forced off in
the harness so every question reaches the LLM. Production code is not bent for
the test: `MI_AGENT_ZERO_COST_FIRST` does not reach the governed entrypoint, so
the harness wraps `parse_with_repair` instead.

**First run — before §3.5:**

| phrase | correct | safe refusal | wrong filter | wrong scope | crash |
|---|---|---|---|---|---|
| funded portfolio | 10 | 0 | 0 | 0 | 0 |
| funded book | 10 | 0 | 0 | 0 | 0 |
| current portfolio | 10 | 0 | 0 | 0 | 0 |
| whole book | 10 | 0 | 0 | 0 | 0 |
| **direct book** | **0** | 0 | **10** | 0 | 0 |
| **acquired book** | **0** | 0 | **10** | 0 | 0 |

All twenty flagged runs were identical: the ids and the numbers were **correct**
(`alp_origination` / `alp_acquired`), but the model had also emitted a redundant
`source_portfolio_type` predicate, and the receipt then mislabelled the scope as
`Source Portfolio Type = d…` instead of naming the governed portfolio.

The scorer was not relaxed. §3.5 was implemented instead.

**Second run — after §3.5:**

| phrase | correct | safe refusal | wrong filter | wrong scope | crash |
|---|---|---|---|---|---|
| funded portfolio | 10 | 0 | 0 | 0 | 0 |
| funded book | 10 | 0 | 0 | 0 | 0 |
| current portfolio | 10 | 0 | 0 | 0 | 0 |
| whole book | 10 | 0 | 0 | 0 | 0 |
| direct book | 10 | 0 | 0 | 0 | 0 |
| acquired book | 10 | 0 | 0 | 0 | 0 |

**60 / 60 correct.** Every phrase was byte-identical across its ten runs
(1 distinct answer, 1 distinct id set, 1 distinct filter set each), so this is
stable behaviour and not a coin flip that happened to land right.

The receipt now names the governed scope: `Source Portfolio in alp_origination`.

---

## 7. Independent truth reconciliation

Recomputed directly from the fixture, not from the agent's own machinery, on the
**genuine-LLM path**. Reconciled against `rawValue`, not the display-rounded KPI.

| phrase | answer | independent truth | ids correct |
|---|---|---|---|
| funded book (count) | 11,035 | 11,035 | yes |
| funded portfolio | £1,964,886,258.21 | £1,964,886,258.21 | yes |
| whole book | £1,964,886,258.21 | £1,964,886,258.21 | yes |
| entire portfolio | £1,964,886,258.21 | £1,964,886,258.21 | yes |
| direct book | £1,385,508,582.98 | £1,385,508,582.98 | yes |
| acquired book | £579,377,675.23 | £579,377,675.23 | yes |
| current portfolio (single select) | £579,377,675.23 | £579,377,675.23 | yes |
| current portfolio (multi-select) | £1,964,886,258.21 | £1,964,886,258.21 | yes |

**Mismatches: 0.**

---

## 8. Regression banks

| bank | path | result |
|---|---|---|
| 40-question bank | deterministic | 11 / 40 — **zero changes** against the P1G baseline, answer-for-answer |
| 40-question bank | genuine LLM | 10 / 40; the three differences from the deterministic path are A4 (`borrower_type` genuinely absent → safe refusal), B23 (known weak scatter), B25 (P1G measure-identity guard refusing a substituted measure). None is scope-related. |
| P1F exposure semantics | forced LLM | GEN 5/5 resolve to the governed current-exposure measure; EAD 3/3 refuse by name; B21 5/5 state **both** amount and share (£842k · 0.043%, matching the independent truth of £841,638.96 / 0.042834%) |
| P1E multi-measure + CFO acceptance (71 questions) | forced LLM | 40 / 71, unchanged in kind |
| full repository suite | — | see §9 |

**Blast radius, computed rather than sampled.** Across all 111 bank questions,
exactly **one** can reach the §3.5 guard at all (it requires a governed scope
span *and* a Direct/Acquired lens): P1E-20, "For the acquired book, what are
balance, loan count and weighted-average LTV?" It answers
£579.38m · 3,909 · 42.68% — the acquired truth exactly — with all three
requested measures surviving and the receipt naming `Source Portfolio in
alp_acquired`. The other 110 are provably untouched.

B25 in particular is untouched: it names *both* families, so the governed lens
resolves to Total and the guard cannot fire. Its refusal is the P1G
measure-identity guard, as before.

After P1I-A, **zero** of the 111 bank questions execute a `source_portfolio_type`
predicate. (The string appears in B04's refusal *explanation*, not in an executed
filter.)

---

## 9. Full suite

`mi_agent/tests`, `mi_workflows`, `mi_agent_api/tests`, `trakt_core`, `tests`:

```
8580 passed, 30 skipped, 21 xfailed, 6 subtests passed in 1715.70s (0:28:35)
```

---

## 10. Remaining role-resolution gaps

Measured with a deliberate probe of scope-*adjacent* phrasings outside the
governed vocabulary, repeated on the LLM path to separate stable behaviour from
noise. These are reported as found; none is newly caused by P1I-A, and all are
LLM-path only (the deterministic parser answers whole-book for every one).

| phrase | class | observed | stability | outcome |
|---|---|---|---|---|
| "the live book", "the performing book" | SCOPE_VS_FILTER | `account_status` predicate with no matching value | — | **safe refusal.** P0 states "No loans in this book match that filter… I have not returned a whole-book figure." |
| "the serviced / managed / on-balance-sheet / securitised book", "the closing / opening / year-end book", "the funded loan book", "the total funded portfolio" | SCOPE_VS_FILTER | no predicate; whole book | stable | correct |
| "the purchased portfolio" | SCOPE_VS_FILTER | governed acquired scope, `source_portfolio_id` | stable | correct |
| "the retained book" | SCOPE_VS_FILTER | 1 of 6 runs emitted `source_portfolio_type = direct`; 5 of 5 on repeat did not | **intermittent** | mostly correct; the outlier narrowed silently |
| **"the originated book"** | **SCOPE_VS_FILTER** | `source_portfolio_type = direct` → 7,126 loans | **5 / 5 stable** | ungoverned scope. The governed vocabulary deliberately excludes "originated" as a book name (it names the *dimension*), so the model asserts a scope the governed model refuses. |
| **"the sponsored book"** | **SCOPE_VS_FILTER** | `source_portfolio_type = direct` → 7,126 loans, £1.39bn | **5 / 5 stable** | **silent semantic error.** A question about the *sponsored* book returns the *direct* book's figure. |

Classification of the whole residual set: **SCOPE_VS_FILTER**, all of it. No
residual MEASURE_VS_FILTER, MEASURE_VS_DIMENSION, COHORT_VS_SCOPE or
TIME_VS_SCOPE case was observed in this probe — the earlier phases closed the
measure and cohort identity classes (P1G) and the "closing / opening /
year-end book" cases resolved to the whole book without a time predicate.

**I am not claiming P1I-A closes this class.** It closes the governed
vocabulary and the exactly-redundant predicate. "The sponsored book" is a
confidently wrong number, it is stable, and it survives P1I-A. Two further
facts make it a safety item rather than polish:

* the answer is not merely narrow, it is the **wrong book** — sponsored asked,
  direct answered;
* `portfolios_in_scope` reports all three books while the executed filter
  covered one, so the coverage evidence and the executed query disagree.

This was not implemented in P1I-A because the ruling in §4 explicitly bounded
P1I-A to deterministic scope resolution and deferred residual ambiguity to a
later phase. It is escalated, not deferred quietly.

---

## 11. Recommended P1I-B

The interpreter already models the shape needed, so this is small.

**P1I-B/1 — refuse an ungoverned scope assertion (safety, not UX).**
A model-emitted `source_portfolio_type` predicate that **disagrees** with the
lens the governed vocabulary resolves for that question, in a question that
contains a governed scope phrase, is an ungoverned scope assertion. It should
refuse on the existing B20 template rather than answer. Dropping it is not an
option — that widens. Answering on it is not an option — "the sponsored book"
proves it returns the wrong book. This needs no new vocabulary and no response
contract change, and it closes the §10 stable defects.

**P1I-B/2 — governed clarification for residual scope ambiguity.**
The user's own suggestion, recorded here as instructed and not implemented.
Where a scope phrase is genuinely ambiguous ("did you mean the total funded book
or the `acquired_001` book?"), the agent asks rather than picks. This is a
response-contract change and belongs in its own phase with its own gate.

**P1I-B/3 — coverage evidence must agree with the executed filter.**
`portfolios_in_scope` reporting three books while one was queried is a receipt
integrity defect independent of how the scope was resolved.

---

## 12. Gate

| criterion | required | measured |
|---|---|---|
| WRONG_FILTER | 0 | **0** (60 live runs, 10 per phrase) |
| WRONG_SCOPE | 0 | **0** |
| SILENT_SEMANTIC_ERROR | 0 | **0** on the P1I-A phrase set; independent truth reconciliation 8/8 exact |
| HARD_FAILURE | 0 | **0** |
| focused bank | green | 53 / 53 |
| deterministic 40-bank | no regression | 11 / 40, zero changes |
| full suite | green | **8,580 passed**, 0 failed |

The gate is met on the population it was defined over. The §10 residual is
outside that population and is escalated above rather than absorbed into the
verdict.

P1I-A GOVERNED SCOPE RESOLUTION: PASS
