# Item 1 — `_FILTER_COMPARATORS` and `_THRESHOLD_PATTERNS`, diagnosed as one decision

Diagnosis and pre-registration. No code touched. Base: HEAD `a046de7`.

---

## 1. The measurement, generated from the code rather than read off it

Every phrase probed against both vocabularies mechanically, collecting **all**
matches rather than the first — a first-match probe got three rows wrong and is
recorded in §6.

```
phrase                     parser       receipt                outcome
------------------------------------------------------------------------------
over                       gt           over                   applied + disclosed
above                      gt           over                   applied + disclosed
more than                  gt           over                   applied + disclosed
greater than               gt           over                   applied + disclosed
bigger than                -            -                      SILENT WHOLE BOOK
larger than                -            -                      SILENT WHOLE BOOK
higher than                -            -                      SILENT WHOLE BOOK
exceeding                  -            over                   not applied -> guard refuses
in excess of               -            over                   not applied -> guard refuses
at least                   ge           at least               applied + disclosed
no less than               ge,lt        under,at least         applied + disclosed
minimum of                 -            at least               not applied -> guard refuses
greater than or equal to   ge,eq        -                      applied, NOT disclosed
under                      lt           under                  applied + disclosed
below                      lt           under                  applied + disclosed
less than                  lt           under                  applied + disclosed
fewer than                 lt           under                  applied + disclosed
smaller than               -            -                      SILENT WHOLE BOOK
lower than                 -            -                      SILENT WHOLE BOOK
beneath                    -            under                  not applied -> guard refuses
at most                    le           at most                applied + disclosed
no more than               le,gt        over,at most           applied + disclosed
up to                      -            at most                not applied -> guard refuses
maximum of                 -            at most                not applied -> guard refuses
capped at                  -            at most                not applied -> guard refuses
between                    between      between                applied + disclosed
N+                         ge*          -                      applied, NOT disclosed
N or above                 ge*          or above               applied + disclosed
older than                 gt           over                   applied + disclosed
younger than               lt           under                  applied + disclosed
------------------------------------------------------------------------------
both 16 · parser only 2 · receipt only 7 · neither 5
```

**They agree on 16 of 30.**

### The five silent ones, confirmed end to end

```
smaller than £150k   ->  ok=True   11,035 loans   filters []
lower than £150k     ->  ok=True   11,035 loans   filters []
bigger than £150k    ->  ok=True   11,035 loans   filters []
larger than £150k    ->  ok=True   11,035 loans   filters []
higher than £150k    ->  ok=True   11,035 loans   filters []
   (control) less than £150k -> ok=True  5,178 loans  ['Balance < 150000']
   (control) over £150k      -> ok=True  5,857 loans  ['Balance > 150000']
```

**Five, not the three the shipped-shapes bank found.** The downward direction has
the same hole and no case reached it.

## 2. Are these one decision or two?

**Two decisions that must stay separate, over one vocabulary that must be shared.**

The two owners do genuinely different jobs:

| | `_FILTER_COMPARATORS` | `_THRESHOLD_PATTERNS` |
|---|---|---|
| yields | an **operator** (`gt`, `ge`, `lt`, `le`, `between`) | a **word** for the receipt (`over`, `at least`, `under`, `at most`) |
| purpose | build a predicate that narrows rows | record that the SENTENCE asked for a narrowing |
| consumer | execution | the honour-or-clarify guard |

**The receipt must keep detecting from the sentence, independently.** If its facet
were derived from the parser's output, a threshold the parser missed would never
be raised — and the guard could never catch it. That is precisely what protects
the seven "receipt only" rows today: `exceeding`, `in excess of`, `minimum of`,
`beneath`, `up to`, `maximum of`, `capped at` are refused rather than answered
wrongly **because the receipt sees a threshold the parser does not.** Two
independent detectors is the design. Collapsing them into one owner would
convert those seven from safe to silent.

What is NOT defensible is that they detect from **different word lists**. *"Is
'bigger than' a comparator, and in which direction?"* is one fact about English,
not two. The five silent phrasings are exactly the rows where both lists happen
to be missing the same word, and nothing made that a decision — it is the
residue of two lists maintained apart.

### The answer to the question asked: a shared VOCABULARY, not a shared owner

And this is the **inverse** of the `_qualified_span_re` precedent, which matters
enough to state rather than let pass:

| | B22 `_qualified_span_re` | item 1 |
|---|---|---|
| shared | the **implementation**, parameterised | the **vocabulary** |
| distinct | the **vocabularies** — scope nouns are genuinely not lens nouns | the **implementations** — an operator is not a receipt word |

The precedent is not "always parameterise the implementation". The precedent is
**share what is genuinely one fact and separate what is genuinely two**, and
hard-coding is wrong whenever it forces those apart. In B22 the vocabularies had
to differ and hard-coding one list dropped five governed phrases. Here the
vocabularies must be identical and keeping two lists dropped five comparators.
Same lesson, opposite mechanics.

So: **one table of `(phrase -> operator, receipt word)`, two consumers building
their own regexes from it and detecting independently.**

## 3. Two further defects found, recorded and NOT fixed here

1. **Applied but not disclosed.** `greater than or equal to` and `N+` apply a
   filter and raise no threshold facet. The number is right and the receipt does
   not mention the narrowing. Item 1 closes this incidentally — both consumers
   reading one table means anything applied is disclosable.
2. **Subject binding, out of scope.** *"loans no more than £150k"* binds the
   threshold to **Current LTV**, not Balance: `['Current LTV <= 150000']`,
   population 11,033, and it refuses. The direction is right; the **column** is
   wrong. That is `_threshold_subject`'s decision, not the vocabulary's.
   **Recorded as its own item, not opened inside item 1** — the B24 precedent.

## 4. What the grader claims, unchanged

Item 1 does not extend the grader. Six of the nine intents stay unverified and
the pack still says so. The success criterion uses the population check that
already exists: **a grader claiming to check a forward figure or a limit headroom
would be the defect this whole sequence is unwinding, wearing a better label.**

## 5. Pre-registered prediction

### 5.1 What moves

| today | predicted |
|---|---|
| 5 silent whole-book phrasings | **applied + disclosed**, narrowed population |
| 7 accidentally-safe refusals | **applied + disclosed**, answered |
| 2 applied-but-undisclosed | **disclosed** |
| `over`, `above`, `less than`, `at least`, `at most`, `between`, `older than`, `younger than` | **unchanged** |

`shipped_shapes`: **B5 correct**, B4 still an unhelpful refusal (its threshold
will now apply, but "ticket" is still unresolved — that is item 2). Predicted
counts **10 / 0 / 0 / 5**.

`nl_score`: the three WRONG_FIGURE phrasings stop being flagged, because the
narrowing now reaches the figure.

**Corpus: 2 of 676 questions contain a newly-covered comparator with a number,
and both are cases this work constructed.** Predicted answer_diff movement:
0 corpus answers, and only the routed/constructed cases.

### 5.2 What must not move

1. The 44, both books: `CORRECT 32 · UNHELPFUL 6 · SAFE 4 · DISCLOSED 2`.
2. Calibration 259/259, 0 hard failures, 0 known gaps.
3. Routed surface 32/32.
4. Seasoning families by name, both books: Q1 4, Q7 4, Q8 12.
5. No lexical decision moves.

### 5.3 Stop conditions

* any control phrasing (`over`, `less than`, `at least`, `at most`, `between`)
  changing its population;
* a direction inverting — `no more than` must never apply `gt`;
* a corpus answer moving;
* the two-detector separation collapsing, so a threshold the parser misses stops
  being raised.

## 6. The restatement checked as its own artefact

Carried in: *"treat the measurement and its restatement as two artefacts, both
verified."*

The first version of §1's table was **wrong in three rows**, and the errors were
in the probe rather than the code:

* it probed `between` as `"loans between 150000"` — not a valid `between`
  clause — and reported `neither`. Both vocabularies know it. **A control was
  miscounted as a defect.**
* it reported only the **first** match per vocabulary, while
  `_detect_thresholds` collects **all**. That hid the two direction ambiguities
  (`no less than` raising both `under` and `at least`).

Corrected before the table was written down, not after. The count moved from
`both 14 · neither 6` to `both 16 · neither 5`, and the headline claim — the
lists disagree, and where both are blind the answer is silent — survived both
versions. **This is the fourth time a restatement has needed its own check, and
the second time the check found something.**

---

# MEASURED — appended after implementation; prediction above left as written

## 7. The success criterion

Stated: *"the three failing phrasings return the narrowed population, and 'over
£150k' continues to."*

```
bigger than £150k    ok=True   5,857 loans   ['Balance > 150000']
larger than £150k    ok=True   5,857 loans   ['Balance > 150000']
higher than £150k    ok=True   5,857 loans   ['Balance > 150000']
over £150k           ok=True   5,857 loans   ['Balance > 150000']   (control)
above / more than    ok=True   5,857 loans   ['Balance > 150000']   (controls)
```

**Met.** And the vocabulary now agrees on **29 of 30** phrases, up from 16, with
**zero silent** and **zero direction ambiguities**.

## 8. It was not met by the vocabulary alone, and that is the finding

Unifying the lists moved the three phrasings from **silent wrong number** to
**refusal** — safer, and not the criterion. The reason:

```
No loans in this book match that filter (current_loan_to_value), so there is
nothing to calculate. I have not returned a whole-book figure in its place.
```

The threshold was binding to **`current_loan_to_value`** instead of the balance.
`_filter_field_of` decides which column a threshold attaches to, and it probed a
**fixed twelve characters** after the comparator for a currency sign:

```
"over "        5 chars   £ inside the window   -> balance
"more than "  10 chars   £ inside the window   -> balance
"bigger than " 12 chars  £ ONE CHARACTER OUTSIDE -> falls through to LTV
"smaller than "13 chars   £ outside              -> falls through to LTV
```

Every phrase the old vocabulary held was short enough. **Item 1 added the longer
phrases and the fixed window became the binding constraint** — a hard-coded span
around a variable-length vocabulary, which is this programme's recurring shape
one layer below the lists themselves. The caller knows where the value ends, so
it now says so, and the window is the match rather than a guess.

That the guard REFUSED rather than answering is the honour-or-clarify contract
working: a filter matching zero rows is not silently replaced by the whole book.
The intermediate state was safe. It was still wrong.

## 9. Results against §5

| declared | measured | |
|---|---|---|
| 5 silent phrasings → applied + disclosed | all 5, narrowed populations | ✅ |
| 7 accidentally-safe → applied + disclosed | all 7 | ✅ |
| 2 applied-but-undisclosed → disclosed | **1 of 2** — see §10 | ❌ |
| controls unchanged | `over`, `above`, `less than`, `at least`, `at most`, `between`, `older than`, `younger than` all unmoved | ✅ |
| shipped shapes 10 / 0 / 0 / 5 | **10 / 0 / 0 / 5** — B5 correct, zero wrong answers | ✅ |
| B4 still an unhelpful refusal | yes: threshold now applies (5,857), "ticket" still unresolved — item 2 | ✅ |
| corpus answers unmoved | **answer_diff 729 of 729 identical, 0 moved** | ✅ |
| the 44, both books | `CORRECT 32 · UNHELPFUL 6 · SAFE 4 · DISCLOSED 2`, identical on both | ✅ |
| calibration 259/259 | 259/259, 0 hard failures, 0 known gaps | ✅ |
| routed surface 32/32 | 32 passed, 0 failed | ✅ |
| seasoning by name, both books | Q1 4, Q7 4, Q8 12 | ✅ |
| no lexical decision moves | 688/690; the 2 moves pre-date this work | ✅ |
| estate | 60 failures, **zero new, zero fixed** vs `a046de7` | ✅ |

## 10. The half of the prediction that was wrong

§5.1 said both "applied but not disclosed" rows would close. **One did not.**

The bare `N+` form — *"how many borrowers are 70+"* — applies its filter
correctly (`Borrower Age >= 70`, 6,862 loans) and raises **no threshold facet**.
The receipt's postfix pattern ends `(?:\+|\s+or (?:above|over|...))\b`, and a
`\b` after `\+` can never match: `+` is a non-word character and what follows is
a space or end-of-string. **A branch that cannot fire** — the same class as the
dead guard found in B16a, and it belongs to the B20 mutation pass over the guard
set rather than to item 1's vocabulary.

It is a disclosure gap, not a wrong number: the narrowing is applied and the
figure is right; the receipt does not mention it. **Recorded, not fixed.**

## 11. Also recorded and not fixed: subject binding without a currency marker

```
"loans no more than £150k"   -> Balance <= 150000    5,178   correct
"loans no more than 150000"  -> Current LTV <= 150000  11,033  wrong column
```

With a currency sign the window fix binds it correctly. Without one, the nearest
-subject heuristic picks the measure named earlier in the sentence. That is
`_filter_field_of`'s precedence rule, a genuinely separate decision from the
vocabulary, and **it is not opened here** — the B24 precedent.

## 12. Constructed coverage, in the established form

> **729 of 729 identical means the fix did not reach the corpora — nothing
> more.** 2 of 676 corpus questions contain a newly-covered comparator with a
> number, and both are cases this work constructed. The corpora would read the
> same whether the change were correct, inert, or wrong in a direction the
> constructed cases do not probe. The claim rests on the 48 tests in
> `test_item1_threshold_vocabulary.py`, the 30-phrase vocabulary probe, and the
> shipped-shapes surface, where B5 moves from wrong answer to correct while B2
> holds.

The one figure that did move for a reader: **45.40% weighted LTV over 5,857
loans** where **43.15% over 11,035** was returned before, for five phrasings of
the question. That is a real correction to a credit-risk headline, and it is
measured on this book rather than constructed.
