# The coordinated-axis role read — diagnosis

Base `31c3257`, tree clean, **nothing built and no design proposed**. Every
number below comes from calling the readers directly and from the delivered
contract, never from the answer prose.

The two headline answers first:

> **It is a bounded rule, not coordination parsing.** The coordination in Q17C
> is already parsed correctly, by an owner that already resolves all three axes.
> Nothing needs to learn to read "A, B and C".
>
> **It is worth one question, not twenty.** Across 1,446 corpus questions, three
> lose an axis their own segmenter resolved. One of them delivers a wrong answer
> (Q17C); the other two already refuse.

---

## 1. Where Q17A and Q17C first diverge

**Earliest point: the term-map lookup inside `_explicit_dimensions`** (line 486),
called at `_deterministic_parse:3405`.

```
Q17A  _explicit_dimensions -> ['ltv_bucket', 'ticket_bucket', 'age_bucket']
                              terms: 'ltv bucket', 'ticket', 'age bucket'
Q17C  _explicit_dimensions -> ['ticket_bucket']
                              terms: 'ticket size'
      _detect_metric(remaining) -> youngest_borrower_age (avg)
```

**What decides it:** `terms_map` = `_registry_dimension_terms(semantics)` (251
terms) + `EXPLICIT_DIMENSION_TERMS` (56). `"ltv bucket"` and `"age bucket"` are
in it; bare `"ltv"` and `"borrower age"` are not. `"ticket size"` *is*, because it
happens to be a declared synonym of `ticket_bucket` — which is why exactly one of
Q17C's three axes survives.

**But that is not the operative cause**, and reading only this far is what makes
this look like a vocabulary gap. Two lines further on there is a second reader
that gets Q17C completely right:

```
Q17C  _grouping_segments  -> metric_part='break direct portfolio balance down'
                             segments=['ltv', 'ticket size', 'borrower age.']
      _classify_segments  -> [('numeric',     'current_loan_to_value',  'ltv_bucket'),
                              ('categorical', 'ticket_bucket',           None),
                              ('numeric',     'youngest_borrower_age',  'age_bucket')]
```

**All three axes, correctly, with their bands.** The estate already knows that
this sentence names three axes and that bare "ltv" means `ltv_bucket`.

**The operative cause is branch order.** Tracing which spec builder fires:

| question | builder | note | dims |
|---|---|---|---|
| `balance by ltv and ticket size` | `_build_two_dim_spec` | — | `ltv_bucket, ticket_bucket` ✓ |
| `balance by ltv, ticket size and borrower age` | *(none)* | **`multi_measure`** | `ticket_bucket` ✗ |
| `Break Direct portfolio balance down across LTV, ticket size and borrower age` | *(none)* | **`multi_measure`** | `ticket_bucket` ✗ |
| `balance by ltv bucket, ticket size bucket and borrower age bucket` | `_build_multi_dim_table_spec` | — | all three ✓ |

The **multi-measure branch at line ~3066 returns before the grouping branch at
~3448 is ever reached.** `detect_measure_set` finds ≥2 measures (balance, LTV,
borrower age), masks their spans, reads dimensions from what is left — and keeps
**one**: `dimension=dims[0]`. `_classify_segments`, which had all three, is never
consulted. `_build_multi_dim_table_spec` — which exists, and handles `len(full_dims) >= 3`
— is never reached.

So: two readers of the same words, both internally right, and **branch order is
the tie-break**. That is the defect.

---

## 2. Bare/banded, or coordination? Tested apart

| shape | question | dims parsed | metric | rows |
|---|---|---|---|---:|
| **bare, 1 axis** | Show balance by **LTV**. | `[]` | LTV | 0 |
| **bare, 1 axis** | Show balance by **borrower age**. | `[]` | age | 0 |
| bare, 1 axis | Show balance by **ticket size**. | `ticket_bucket` | balance | 5 |
| banded, 1 axis | Show balance by LTV **bucket**. | `ltv_bucket` | balance | 6 |
| banded, 1 axis | Show balance by borrower age **bucket**. | `age_bucket` | balance | 7 |
| bare, 2 coord | Show balance by LTV and ticket size. | `ltv_bucket, ticket_bucket` | balance | 30 ✓ |
| **bare, 3 coord** | Show balance by LTV, ticket size and borrower age. | `ticket_bucket` | balance | 5 ✗ |
| banded, 2 coord | …by LTV bucket and ticket size bucket. | both | balance | 30 ✓ |
| banded, 3 coord | …by LTV bucket, ticket size bucket and borrower age bucket. | all three | balance | ✓ |

**A bare axis fails ALONE, with no coordination present at all.** "Show balance by
LTV." parses zero dimensions and reads LTV as the measure. So the primary variable
is **bare vs banded**, not coordination.

**Coordination enters only as an arity effect, and in the opposite direction to
the intuition.** The *two*-item bare coordination **works** (30 rows, correct) —
because only one bare measure term is present, `detect_measure_set` finds fewer
than two measures, and the two-dimensional branch wins. The *three*-item bare
coordination **fails** — because the third bare term is the second measure, which
is what tips `detect_measure_set` over its threshold and hands the sentence to the
multi-measure branch.

So coordination does not break the parse. **It changes which branch claims the
sentence.**

---

## 3. Is this Q22C's defect? No — and the contrast is clean

```
Q17C  _grouping_segments -> ['ltv', 'ticket size', 'borrower age.']     3 segments
      _classify_segments -> three axes, all resolved

Q22C  _grouping_segments -> []                                          NO segments
      _classify_segments -> []
      mask_scope_phrases -> 'Which of the Direct and                drove more…'
```

| | Q17C | Q22C |
|---|---|---|
| owner | `llm_query_parser` branch order | `portfolio_lens` scope mask + `_detect_lost_narrowing` |
| the coordination | **parsed correctly, all conjuncts resolved** | **not parsed at all — zero segments** |
| failure | a competing reader claims the sentence first | the mask eats "Acquired books" and strands a bare "Direct" |
| what is missing | a precedence decision between two existing readers | distributing an elided head noun over both conjuncts |

Q22C needs a reader that can see *"the Direct and Acquired **books**"* as two
scopes sharing one head noun. Nothing in the estate does that, which is why it is
open. **Q17C needs no such thing** — its conjuncts are already separate, already
resolved, already banded. It does **not** join Q22C on the open list.

---

## 4. Does the `derived_from` correction already cover part of this? No — measured

The correction shipped in `31c3257` is in `question_interpretation/completeness.py`
— **the check, not the parser.** It cannot influence a parse, and does not.

The parser holds its own copies of the same relation, and they are lists:

| where | entries | covers |
|---|---:|---|
| registry `derived_from` | **20** | every derived field |
| `_NUMERIC_AXIS_BUCKET` (llm_query_parser:1962) | 9 terms | 4 buckets — ltv, age, interest rate, ticket |
| `_explicit_dimensions(grouping=True)` extra map | **1** | `{"age": "age_bucket"}` |

Neither reads `derived_from`. The bare→banded knowledge exists three times, in
three different sizes, and the registry's declaration — the only one that is
complete — is not among the readers. F1's shape again, and this time in triplicate.

---

## 5. How many readers decide axis-vs-measure, and do they agree?

**Five, and they give three different answers on the same word.**

| term | `_explicit_dimensions` | `…(grouping=True)` | `_classify_segment` | `_NUMERIC_AXIS_BUCKET` | `_detect_metric` |
|---|---|---|---|---|---|
| `ltv` | — | — | numeric → `ltv_bucket` | `ltv_bucket` | `current_loan_to_value` |
| `borrower age` | — | `age_bucket` | numeric → `age_bucket` | `age_bucket` | `youngest_borrower_age` |
| `interest rate` | — | — | numeric → `interest_rate_bucket` | `interest_rate_bucket` | `current_interest_rate` |
| `balance` | — | — | numeric → `ticket_bucket` | `ticket_bucket` | `current_outstanding_balance` |
| `ticket size` | `ticket_bucket` | `ticket_bucket` | categorical → `ticket_bucket` | — | `current_outstanding_balance` |

Note the two *internal* disagreements, which are not about the sentence at all:

- `_explicit_dimensions(grouping=True)` resolves `borrower age` but **not** `ltv`,
  because its extra map has one entry; `_NUMERIC_AXIS_BUCKET` resolves both.
  **Two bare→banded maps in one module, of different sizes.**
- `_metric_side_residue` (line 836) folds `_NUMERIC_AXIS_BUCKET`'s nine terms
  **into the measure vocabulary**. The same nine strings are simultaneously the
  bare-axis map and part of what makes a word a measure. The collision is not
  between two files — it is in one module, and the two uses sit 1,100 lines apart.

`execution_receipt.requested_dimension_terms` and
`concept_proposal.vocabulary` both delegate to `_explicit_dimensions`, so they
inherit reader 1's answer — which is why the facet layer reports no lost axis on
Q17C: **it is looking at the reader that agrees with the wrong branch.** That is
why `notApplied` is empty.

---

## 6. Blast radius, and what a fix would recover

Measured by comparing, for every question, the axes `_classify_segments` resolves
against the dimensions the delivered contract carries.

| surface | name 2+ resolvable axes | **lose at least one** |
|---|---:|---:|
| review pack, 166 | 19 | **2** |
| corpus, 1,446 | 152 | **3** |

Every affected question, in full:

| question | segmenter resolved | contract carries | today |
|---|---|---|---|
| **Q17C** Break Direct portfolio balance down across LTV, ticket size and borrower age. | ltv, ticket, age | ticket | **WRONG** off arm (5 rows vs 143); CORRECT on merge |
| **Q12C** Plot portfolio balance across LTV buckets and borrower-age buckets. | ltv, age | ltv | FALSE_REFUSAL both arms |
| Show no-negative-equity risk by age and LTV. | age, ltv | — | refuses (NNEG is not governed) |

**No currently CORRECT answer is in the affected set**, on either arm, on either
surface.

**Recovery: one wrong answer and one false refusal.**

- **Q17C** — the whole reason for asking. Off arm only; the merge arm already
  answers it correctly at 143 cells.
- **Q12C** — a *different* sub-cause worth naming: it contains the word "plot",
  which sets `explicit_plot` and skips the grouping branch entirely. Adjacent, not
  the same trigger.
- The third already refuses for an unrelated reason and would not move.

### The risk is not in the affected set

The three above bound the **recovery**. They do not bound the **risk**: a change
to precedence between `detect_measure_set` and the grouping segmenter would be
evaluated against the **149 questions that name 2+ axes and get them all today**,
and against every genuine multi-measure question — "For the London book, give me
balance, number of loans, weighted-average LTV and average borrower age" is a real
multi-measure request whose bare terms must **stay** measures. Any rule here has to
separate that sentence from Q17C, and both put bare measure words after a
preposition.

That separation is the actual difficulty, and it is not measured here because
measuring it means building. Stated rather than glossed.

---

## 7. The answer to the question you asked

**Bounded rule.** The evidence:

1. The coordination is already parsed — `_grouping_segments` returns three
   segments and `_classify_segments` resolves all three axes with their bands.
2. The correct builder already exists and already handles three or more axes
   (`_build_multi_dim_table_spec`, reached via `len(full_dims) >= 3`).
3. The failure is one branch returning before another, decided by an arity
   threshold in `detect_measure_set`.
4. A bare axis fails **alone**, which rules coordination out as the cause.
5. Q22C's coordination, by contrast, produces **zero** segments — genuinely
   unparsed, genuinely a different defect.

**Worth: one question.** Q17C on the deterministic arm, plus Q12C from an adjacent
trigger. Not twenty. If the merge arm is the shipping configuration, Q17C is
already correct there and the recovery is one false refusal.

Nothing built. No design proposed. The gate, `scopeApplied` and the dataset class
are untouched.

### Environment
`MI_AGENT_LLM_PARSER=off` throughout (F2), run from the repository root (F6).
**Successful model responses: 0.**
