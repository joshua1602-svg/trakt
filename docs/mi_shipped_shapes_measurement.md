# Three shipped shapes, scored four ways

Measured, not fixed. Nothing in this pass changes product behaviour.

Bank: `config/mi/golden_questions/shipped_shapes.yaml` (3 shapes × 5 variations)
Runner: `question_interpretation/shipped_shapes.py`
Entry point: `execute_governed_mi_query`, routing exactly as shipped.

---

## 1. The counts

```
   correct               9
   wrong answer          1     <- the only outcome that reaches a reader as fact
   honest refusal        0
   unhelpful refusal     5
   ------------------------
   total                15
```

| shape | correct | wrong | honest | unhelpful |
|---|---|---|---|---|
| A portfolio summary | 4 | 0 | 0 | 1 |
| B a value in a filter position | 1 | 1 | 0 | 3 |
| C two groupings at once | 4 | 0 | 0 | 1 |

**Zero honest refusals.** Every refusal in these fifteen declined something the
book can express and the sentence supplies. That is the number to move.

## 2. A correction to the premise, which strengthens it

The brief said these run "through `execute_governed_mi_query` — not through
`nl_harness`". `/mi/query` (`mi_agent_api/app.py:1737`) **already calls
`execute_governed_mi_query`**, and the robustness runner posts there. So the
robustness bank was on the shipped path all along.

What was never on the shipped path is the **grading**. `nl_score.grade` reads an
outcome label:

* line 151 — *any* refusal carrying a stated reason returns `SAFE_REFUSAL`. All
  five unhelpful refusals below grade `SAFE_REFUSAL`, indistinguishable from a
  refusal that was right to decline.
* `INCORRECT_SUCCESSFUL` exists but is reached only by per-intent heuristics on
  the answer TEXT. Nothing compares a figure to the book. B5 below returns a
  complete, well-formatted, wrong number and no label-grader catches it.

So the conclusion stands and the reason is sharper: **the questions were right,
the path was right, and the grader could not tell a wrong number from a right
one.** This surface computes the answer from the book with pandas and compares.

## 3. THE WRONG ANSWER, IN FULL

### B5 — "What LTV are we running on loans bigger than £150k?"

```
  route     : None            view: funded
  verdict   : ok              ok: True
  facets    : []                          <- nothing was raised, so nothing could be honoured
  measure   : Current LTV (weighted_avg)
  filters   : []                          <- the threshold is gone
  dimensions: []
  population: 11035                       <- the narrowing selects 5,857
  artifacts : [('kpi', 0)]

  ANSWER AS THE READER RECEIVES IT:
      Here is the result for your query, covering 1 group(s).

      Calculated: Weighted-average Current LTV · entire funded portfolio ·
      11,035 loans · as at 30 June 2026.
```

**Correct answer: 44.32% simple / 45.40% weighted, over 5,857 loans.
Returned: 40.27% / 43.15%, over 11,035.** A 2.25-point error on a
credit-risk headline, presented as complete, with `ok: True` and an empty
facet list.

The receipt is not merely silent — *"entire funded portfolio"* is an
affirmative claim about a scope the reader did not ask for.

### It is not one phrasing. Probed across the comparator vocabulary:

```
  over          filter applied   5,857   ok
  above         filter applied   5,857   ok
  more than     filter applied   5,857   ok
  of at least   filter applied   5,857   ok
  >             filter applied   5,857   ok
  greater than  refuses
  exceeding     no filter       11,035   ok=False   (accidentally safe)
  in excess of  no filter       11,035   ok=False   (accidentally safe)
  bigger than   NO FILTER       11,035   ok=True    <- silent wrong number
  larger than   NO FILTER       11,035   ok=True    <- silent wrong number
  higher than   NO FILTER       11,035   ok=True    <- silent wrong number
```

**Three phrasings return the whole book as fact.** B5 caught one; the probe
found two more the bank does not contain.

## 4. Root cause: one lexical decision, two vocabularies that disagree

"Is this a threshold?" has **two independent owners**, and neither is a superset
of the other:

| owner | role | has | lacks |
|---|---|---|---|
| `llm_query_parser._FILTER_COMPARATORS` | **applies** the filter | more than, greater than, older than, over, above, at least, > | bigger/larger/higher than, exceeding, in excess of |
| `execution_receipt._THRESHOLD_PATTERNS` | **raises** the facet | over, above, more than, greater than, exceeding, in excess of | bigger/larger/higher than |

The disagreement is exactly what decides whether a failure is safe or silent:

* in **both** → the filter is applied. Correct.
* in the **receipt only** (`exceeding`, `in excess of`) → the facet is raised,
  execution cannot honour it, the guard refuses. **Accidentally safe — and safe
  only because the two lists differ.**
* in **neither** (`bigger than`, `larger than`, `higher than`) → no filter, no
  facet, nothing to honour, nothing to disclose. **The whole book, as fact.**

This is the census shape exactly, with a new consequence: the honour-or-clarify
guard cannot protect a narrowing that was never recorded. B16a established that
for values; this is the same hole for thresholds.

## 5. The five unhelpful refusals, and what they share

```
A5  Tell me the basics about this book
    "I couldn't map this question to a governed analytic"
    -> four other phrasings of the same question answer correctly.

B1  What is the LTV for loan tickets above £150k?
    "I could not tell how you meant ticket. Split by it, or narrowed to one value?"
    facets: threshold applied, unresolved_role LOST | filter applied | pop 5,857
    -> IT HAD THE ANSWER. Filter applied, 5,857 loans, measure Current LTV.
       It declined over the word "ticket", which the sentence uses in neither
       of the two roles it offers.

B3  Show me the LTV for loans with a balance above £150,000
    "the answer reports balance, but the question asked about ltv"
    measure resolved to BALANCE | filter applied | pop 5,857
    -> the field word in the FILTER position was taken as the measure.

B4  For tickets larger than £150k, what is the LTV?
    same "how did you mean ticket" clarify, and here the threshold is also lost
    ("larger than" — §3), so pop 11,035.

C4  Give me a breakdown of balance across LTV and ticket size
    "the answer reports ltv, but the question asked about balance"
    measure resolved to CURRENT LTV | dims ['Ticket Size']
    -> the exact MIRROR of B3. A field word in an AXIS position was taken as
       the measure. Changing one word fixes it:
         "...across LTV and ticket size"  -> refuses
         "...by LTV and ticket size"      -> answers correctly
```

**Four of the five (B1, B3, B4, C4) are one defect: a field word named outside
the measure position is assigned the wrong ROLE.** D2 built `dimension_role` to
own *filter vs axis*. Nothing owns *measure vs not-measure*, so `balance` in a
filter clause and `LTV` behind `across` both get taken as the measure.

In B3 and C4 the substitution guard **catches it and refuses** — the guard is
working. The refusal is unhelpful because the role assignment upstream was
wrong, not because the guard was.

## 6. What went right, stated with the same care

* **Shape C works.** Four of five phrasings return both dimensions, a heatmap
  and a table carrying both, 50 cells, reconciling to £1,964,886,258.21 — the
  exact book total. The brief expected no pair handling; there is.
* **The product is more careful than my first ground truth.** A plain
  `groupby(["ltv_bucket","ticket_bucket"])` drops the two loans with a null LTV
  bucket and £337,343.21 with them. The product surfaces them as
  `Unknown / Missing` and reconciles to the book. My first grader called that a
  defect in four cases.
* **Shape A works** in four of five, and answers in the KPI card as well as the
  prose — which my first grader also could not see, calling A4 a summary with no
  balance in it.

**Three of my first seven "wrong answers" were the grader's fault.** Recorded
because it is this surface's own failure mode, found in this surface: a grader
that cannot see where the product puts its answer reports the product wrong.
`--self-test` now probes every grader in both directions.

## 7. kpi_028-031, as raised

Confirmed. `config/mi/golden_questions/ere_mi_calibration_250.yaml:553` —
`expected_metric: null`, `expected_metrics: null`,
`expected_columns_include: []`, `expected_min_columns: 1`, and the note
*"Whole-book summary (count + balance)"*. **Nothing asserts the balance.** All
four pass on an answer containing no balance at all. A4's phrasing is not in
that bank, and A4 is where the prose omits it.

## 8. Shortest path, ordered by what the measurement shows

**1 — One owner for "is this a threshold?"** *(the only wrong number)*
Consolidate `_FILTER_COMPARATORS` and `_THRESHOLD_PATTERNS` onto one vocabulary,
the way B21 consolidated the disclaiming window. Add `bigger/larger/higher than`.
This is the only item that removes a wrong number from the shipped product, and
it converts three silent whole-book answers into correct ones. The two
accidentally-safe phrasings become correct rather than refused.
*Instrument first: the surface must assert `population` and `filtersApplied` per
comparator — it does; the probe in §3 becomes a case per phrasing.*

**2 — One owner for the MEASURE role.** *(four of five unhelpful refusals)*
Extend D2's role owner from *filter vs axis* to *measure vs filter vs axis*, so a
field word in a filter clause or behind an axis marker is not taken as the
measure. Diagnose B3 and C4 together before fixing either — they are one shape
in mirror image, and the `across`/`by` asymmetry says the axis vocabulary is
also multi-owned (`lexical.AXIS_MARKERS` has six markers,
`lexical._BY_RE` matches only `by`, `llm_query_parser:857` has its own six).
Do not fix inside item 1.

**3 — `unresolved_role` should not fire on a word the sentence never put in
either role.** *(B1, B4)* "ticket" in "loan tickets above £150k" is neither an
axis nor a value-selector; it is the noun the threshold is on. B1 held the
correct answer and declined. Smaller than item 2 and possibly closed by it —
re-measure after 2 rather than scheduling it now.

**4 — "the basics" / "headline numbers" reach `portfolio_summary`.** *(A5)*
A phrase-list gap, one unhelpful refusal, no wrong number. Lowest.

### What this does not change

Nothing in the backlog blocks items 1–4. **D10, D9, D14, the segmented series,
B24, B20, B16b, B9 and B10 all stay in the backlog** — none of them is on the
path to any of these four, and item 1 is the only one that moves a number in the
shipped product.

Item 1 outranks D10 on the same test that put B21–B23 ahead of it: it changes
the number a reader is shown.
