# Item 3 (B1/B4) — one owner for "which field is this threshold on"

Pre-registration. No code touched. Base: HEAD `a6b4b42`.

---

## 1. The wrong disclosure SHARES the cause. Folded in.

The ruling was: fold in the wrong label *if it shares the cause; if not, leave
it.* Measured, it does — and they are the same two owners.

| | `llm_query_parser._filter_field_of` | `execution_receipt._threshold_subject` |
|---|---|---|
| decides | which FIELD a threshold binds to | which SUBJECT the receipt names |
| vocabulary | `_FILTER_SUBJECT_PATTERNS` — ltv, age, rate, balance (incl. `ticket`), valuation | `_THRESHOLD_SUBJECTS` — LTV, borrower age, balance (incl. `ticket`), interest rate, valuation |
| rule | **the subject nearest BEFORE the comparator** | **the first entry in a fixed priority list**, over a 42-character window |
| output | a field key | a display name |

Near-identical vocabularies, two different rules, and the rules disagree whenever
a measure is named earlier in the sentence than the threshold's own noun:

```
question                                          receipt label   execution binds
what is the ltv for loans with a balance above…    LTV over …      current_outstanding_balance   DISAGREE
what is the ltv for borrowers over 75              LTV over 75     youngest_borrower_age         DISAGREE
what is the balance for loans with an interest…    balance over 5  current_interest_rate         DISAGREE
```

**Three of eight probed sentences disclose a threshold on a field execution did
not filter.** B1's is one of them: the receipt says `"LTV over 150000"` for a
filter applied as `Balance > 150000`.

So this is the **sixth instance** of the shape, and the unresolved role is the
same decision *never asked*: `dimension_role` consults neither owner.

## 2. What is one fact and what is two

* **One fact** — *which noun is this threshold on?* The rule is proximity to the
  comparator, which `_filter_field_of` already implements and item 1 already
  hardened (the currency-window fix).
* **Two** — the renderings. A field key is not a display name, exactly as an
  operator was not a receipt word in item 1.

So: one vocabulary and one rule in `question_interpretation.lexical`, returning a
**kind**; the parser maps the kind to a field, the receipt maps it to a display
name, and `dimension_role` reads it as a fifth source.

## 3. The fifth source, not a duplicated mapping

`dimension_role`'s four sources are reader 1's filter slot, reader 1's axis slot,
the sentence's grouping cut, and "the book cannot express it". Source 1 misses B1
because the facet is `ticket_bucket` and the applied filter is
`current_outstanding_balance` — the facet's own `satisfied_by()` keys never reach
the balance field.

**Source 5:** the facet's label names a word the owner recognises as a threshold
subject, and the FIELD that subject resolves to is in `filters` → `ROLE_FILTER`
on that key. The word is doing the job of naming the thing being thresholded, and
that is a settled role, not an unresolved one.

## 4. The two-detector separation is preserved

`_detect_thresholds(q)` keeps its signature and keeps reading the SENTENCE. The
owner is a text-level rule needing no semantics, no spec and no applied filters,
so `test_the_detectors_stay_separate` holds unchanged — deliberately, because it
guards the property that makes a missed threshold refuse rather than answer.

## 5. Pre-registered prediction

| id | today | predicted |
|---|---|---|
| B1 *"the LTV for loan tickets above £150k"* | unhelpful refusal | **answered, 5,857 loans, LTV 45.40%** |
| B4 *"For tickets larger than £150k, what is the LTV?"* | unhelpful refusal | **answered, 5,857 loans** |
| the three disagreeing labels | name the wrong field | **name the field execution bound** |
| A5 | unhelpful refusal | **unchanged** — item 4 |

`shipped_shapes` predicted **14 correct · 0 wrong · 0 honest · 1 unhelpful.**

### Must not move

1. `answer_diff` 729 of 729 identical.
2. The 44, both books `32/6/4/2`; seasoning by name Q1 4, Q7 4, Q8 12.
3. Calibration 259/259; routed 32/32.
4. Item 1's 48 and item 2's 19 tests, including
   `test_the_detectors_stay_separate`.

### Stop conditions

* a threshold binding to a different field than it does today in any case where
  the receipt and execution already AGREE;
* `_detect_thresholds` acquiring any parameter beyond `q`;
* a corpus answer moving.

### Constructed coverage, declared before the number

The corpus contains no question naming a bucketed dimension as a threshold's
subject — B1 and B4 are constructed. A clean `answer_diff` will mean the fix did
not reach the corpora and nothing more. **The label corrections are the part that
may reach them**, and that is the half worth watching.

## 6. Recorded, not opened

* The receipt gap: closing the population check means the analytical composition
  layer emitting, per composed capability, the population it measured and the
  whole-book total it measured against — the two numbers the point-in-time path
  already publishes. A receipt change on the analytical path, not a grader
  change. **Not now.**
