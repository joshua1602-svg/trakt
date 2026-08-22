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

---

# MEASURED — appended after implementation

## 7. Results against §5

| declared | measured | |
|---|---|---|
| B1 answered, 5,857 loans | **answered, `Balance > 150000`, 5,857 loans** | ✅ |
| B4 answered, 5,857 loans | **answered, identical** | ✅ |
| the disagreeing labels name the bound field | all eight probed sentences agree | ✅ |
| A5 unchanged | unchanged | ✅ |
| shipped shapes **14 / 0 / 0 / 1** | **14 / 0 / 0 / 1** | ✅ |
| `answer_diff` 729 identical | **729 of 729** *(after §8)* | ✅ |
| the 44, both books 32/6/4/2 | unmoved; seasoning Q1 4, Q7 4, Q8 12 | ✅ |
| calibration 259/259, routed 32/32 | both | ✅ |
| `test_the_detectors_stay_separate` | passes; `_detect_thresholds(q)` unchanged | ✅ |
| lexical decisions | **697 of 697 identical** | ✅ |

## 8. The stop condition fired, and it was a REGRESSION — caught by one corpus answer

The prediction named the label corrections as "the half that may reach the
corpora, and that is the half worth watching". One did:

```
  service_path/ranking_concentration_023
    "What percentage of the book is above 50% LTV?"
      threshold facet label:  "LTV over 50"  ->  "over 50"
```

**The subject was lost, not corrected.** `"above 50% LTV"` is a POSTFIX
construction — the subject follows the value — and `_filter_field_of` has always
had a rule for it: *"a subject stated immediately AFTER the number binds
tightest"*. I shared the vocabulary and implemented only the nearest-BEFORE half
of the rule.

So the owner now carries both halves, postfix first. This is **item 1's lesson a
second time**: a consolidation is complete when the consumers have been exercised
across the vocabulary's full range, and *prefix-only was not the full range*. The
difference is that here the surface caught it — one corpus answer, on the differ,
before it shipped.

Recorded as a can-fail: `test_a_postfix_subject_binds_tightest`.

## 9. A test that measured nothing

`test_a_word_naming_a_threshold_subject_is_a_filter` failed on its first run
while the end-to-end path worked. The fixture used the label `"ticket size"`; the
sentence says `"loan tickets"`, so `_named_term_span` found nothing and source 5
correctly declined. **A unit test whose fixture does not match the data the
function receives measures nothing** — the real label is `"ticket"`, which the
refusal message had been printing all along.

## 10. Constructed coverage

> **729 of 729 identical means the fix did not reach the corpora — nothing
> more.** B1 and B4 are constructed; the corpus contains no question naming a
> bucketed dimension as a threshold's subject. The claim rests on the 12 tests
> here and the shipped-shapes surface.

**The label half did reach the corpora**, once, and it reached them as a
regression that the differ caught. That is the corpora doing the one job this
pack says they can do: not proving a fix correct, but catching a change that
touches them.
