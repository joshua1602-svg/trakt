# The fragment guard, and the value-collision rule

Base `9968025`. Two changes, in the order ruled. Both measured on the
166-question pack (both arms), 98 further provenance/channel questions in two
fixtures, 1,446 distinct corpus questions, the 226-question data-claim audit and
the 278 registered test modules.

**Read §4 first if you read nothing else.** The fragment fix is correct and it
exposes a defect underneath it. One question moves from a false refusal to a
wrong answer on the deterministic arm, and that class is the one this programme
ranks above everything else. It is pre-existing and was being masked, but you
should decide whether it changes the order.

---

## 1. The fragment guard — YES, it is a property

> *A token whose capital is explained by SENTENCE POSITION carries no
> proper-name evidence, so it is not part of the name.*

`mi_agent/portfolio_lens.py` · `_SENTENCE_INITIAL_RE` + `_unknown_named_book`.
No verb vocabulary. No list to keep up to date. Only the **first** token of the
run is discounted, so a multi-word name that opens a sentence keeps the rest of
its run to be judged on.

### It was short by three, not two

Measured over **1,446 distinct corpus questions**, against the live registry:

| | |
|---|---:|
| fragments removed | **3** |
| fragments newly raised | **0** |
| genuine unknown book names still caught | all |

```
REMOVED  'Break Direct portfolio'   <- Break Direct portfolio balance down across LTV…
REMOVED  'Plot portfolio'           <- Plot portfolio balance across LTV buckets and…
REMOVED  'Which portfolio'          <- Which portfolio grew the most in the acquired book?
KEPT     'Highgate Mortgages Book'  'ALP Origination Book'  'NBS Acquired Book'
         'ALP Acquired Back Book'   'London book'
```

`break` and `split` were the two you named. `plot` and `which` were also
missing, and `which` is not even a verb. That is F1 exactly: the list was not
two words short of complete, it was short by however many words nobody had
thought of, and there was no way to know which.

### What the two other removals became

Both now refuse **for true reasons**, which is strictly better than refusing
with a fake book name:

```
Plot portfolio balance across LTV buckets and borrower-age buckets.
  -> "I understood that you asked for age buckets, but that could not be
      applied to the calculation (age buckets (Age Bucket) — the requested
      breakdown was not applied). I have not substituted a broader figure."

Which portfolio grew the most in the acquired book?
  -> "…this asks how something changed, which needs two governed reporting
      snapshots to compare. I have NOT substituted a current-position figure…"
```

### Two limits, recorded in the docstring rather than discovered later

- **A one-word book name opening a sentence** ("Highgate Book summary") is
  discounted along with the verbs. At that position a name and a verb are
  genuinely indistinguishable by capitalisation, which is the only signal this
  layer has. Measured trade: 3 fragments removed, 0 raised.
- **A trailing separator still evades the guard.** "Direct-book" yields the
  token `Direct-`, and `direct-` matches no entry in *any* word list, so the run
  is judged a name whatever the list holds. Same shape of defect — a token the
  guard cannot match rather than a word nobody listed — and **not fixed here**,
  because it is coupled to the qualifier/noun separator in `_qualified_span_re`,
  which is unshipped. Four questions still refuse with `'Direct- book'`.

So: **the class is closed for the verb case and open for the separator case**,
and I am not claiming otherwise. Simulated, adding edge-punctuation
normalisation removes those four as well — 7 removed, 0 raised — but it changes
behaviour on questions the unshipped hyphen fix also touches, so it waits for
your ruling rather than arriving as a side effect.

`QUOTES_A_MANGLED_PHRASE` in the data-claim audit: **2 → 1**. The one left is
`'Direct- book'`.

---

## 2. The value-collision rule — segmentation key wins

`mi_agent/execution_receipt.py` · `_value_owner`, read by `dimension_values`.

```
origination_channel     source_criteria: [curated]        146 rows of `direct`
source_portfolio_type   source_criteria: [segmentation_key]  441 rows of `direct`
```

> When a value resolves to more than one governed field, the field the registry
> declares a `segmentation_key` wins. Between fields of equal standing the value
> stays ambiguous and is dropped — so a collision this rule cannot decide is
> still not decided by iteration order.

No new registry key. No new binder. `direct` is not named anywhere in the code.

**The map was built with `setdefault`, so the winner was whichever field the
registry happened to iterate first.** That is what this closes. The worse half
was never the wrong field but the disagreement: `categorical_spans.value_field`
*refuses* the same token as ambiguous by design — "an ambiguous narrowing must
be disclosed, never resolved by preference" — while this map silently picked the
losing field. Two owners of one word, one refusing and one guessing.

**Not also in the filter binder**, as ruled. `source_portfolio_type` is a
scope-owned field and narrowing it is `portfolio_lens`'s job. The second site is
where the redundant predicate would live.

**It recovers no questions, and that is not why it is here.** Three refusals
stop naming the wrong field:

```
before   Direct (Origination Channel) — this narrowing was not applied…
after    Direct (Source Portfolio Type) — this narrowing was not applied…
```

---

## 3. What moved

### The 166-question pack

| | CORRECT | FALSE_REFUSAL | NO_COMPUTABLE_TRUTH | TRUE_REFUSAL | WRONG |
|---|---:|---:|---:|---:|---:|
| **off** before | 102 | 22 | 22 | 15 | 5 |
| **off** after | 102 | **21** | 22 | 15 | **6** |
| **merge** before | 108 | 21 | 20 | 15 | 2 |
| **merge** after | **109** | **20** | 20 | 15 | 2 |

Four answers moved on each arm. Three are the collision rule's wording change
(Q04A, Q05B, Q17B). The fourth is Q17C — §4.

**No previously CORRECT answer changed, on either arm.**

### Everything else

| surface | result |
|---|---|
| 98 provenance/channel questions × 2 fixtures | 1 moved (`"What is the direct funded balance?"`, wording only). 0 stopped answering, 0 started, 0 population changes. |
| 1,446 corpus questions, fragment guard | 3 removed, 0 raised |
| data-claim audit, 226 questions | `FALSE_about_the_book` 0. `QUOTES_A_MANGLED_PHRASE` 2 → 1. AUDIT HOLDS. |
| 278 registered test modules | 85 failures before, 85 after — the frozen baseline restored exactly, none new, none fixed |

---

## 4. THE FRAGMENT FIX EXPOSES A DEFECT UNDERNEATH IT

**Q17C — "Break Direct portfolio balance down across LTV, ticket size and
borrower age."**

| arm | before | after |
|---|---|---|
| merge | FALSE_REFUSAL | **CORRECT** — 143 cells, matching the frozen truth |
| off | FALSE_REFUSAL | **WRONG** — 5 rows |

The off arm now answers:

```
Here is the bar for your query, covering 5 groups.
Calculated: Balance · Average Borrower Age · Source Portfolio in direct_001
            · grouped by Ticket Size · 441 loans · as at 30 June 2026.
```

The question named **three** axes. The spec carries **one** (`ticket_bucket`).
`notApplied` is empty. No facet mentions LTV or borrower age. `borrower age` was
carried in the **wrong role** — read as a measure ("Average Borrower Age"), not
an axis — and `LTV` vanished entirely. Truth is 143 cells; the answer is 5 rows.

**A wrong figure with a receipt that reads clean.** The class this programme has
ranked above everything else.

### It is pre-existing, and the false refusal was masking it

The parse has always dropped those two axes. Its sibling Q17A — *"For the Direct
book, show balance by LTV **bucket**, ticket-size **bucket** and borrower-age
**bucket**"* — is CORRECT on the off arm today. Q17C's phrasing names the axes
bare, and bare "LTV" and "borrower age" are read as measures. The fragment
refusal was standing in front of that, refusing with a false statement about the
reader's own sentence.

So reverting the fragment fix does not remove the defect. It restores a lie in
front of it.

### The estate can already see part of the loss

The Stage 1 completeness check, run on the same envelope:

```
Q17C  STATED:     dimension 'ticket size' · scope · measure 'borrower age'
      UNRESOLVED: measure 'borrower age'          <- fires
Q17A  UNRESOLVED: (none)                          <- correctly silent
```

It fires on Q17C today, deterministically, with no new reader. It does **not**
see bare "LTV" — no owner resolves it as a dimension in that sentence, which is
the recall limit recorded when the check was built: *the check's recall is the
owners' recall*. This is the strongest case yet for wiring it.

### The ruling I need from you

Both changes are pushed, because the fix is right and the exposure was already
there. But if you rank "no new WRONG on the deterministic arm" above "no false
refusal", the ordering should be:

1. **Wire the completeness check**, or fix the coordinated-axis role read
   (`"break X down across A, B and C"` — the same coordination family as Q22C),
2. **then** the fragment fix.

Say the word and I will revert the fragment fix pending that. I did not make
that call for you.

---

## 5. Q22C — left where it is

Elided coordination: *"Which of the Direct and Acquired books drove more…"*. The
scope mask consumes "Acquired books" and leaves a bare "Direct", which raises a
lost narrowing. `_COMPARISON_MARKERS` does not contain "drove more". Unchanged
by both fixes, as expected. It stays on the open list as needing coordination
parsing rather than a phrase.

---

## 6. Q07B — logged separately, and my earlier label was wrong

I called it "a merge-arm duplicate dimension". **It is not a duplicate.**
Measured on the envelope:

```
spec.dimensions = ['source_portfolio_type']   spec.dimension = 'source_portfolio_type'
```

That is the documented convention — `_apply_to_spec` deliberately puts a single
governed axis on both fields. My "duplicate" came from the review pack's own
capture concatenating the two spec fields. The pack's `spec_dimensions` column
is a measurement artefact on every single-axis row, and I have said so here
rather than leaving it to be read as data.

**The real cause is a recogniser gap:**

| question | route | outcome |
|---|---|---|
| "Compare the Direct and Acquired books." | `portfolio_risk_comparison` | applies the axis, answers |
| "How do the Direct and Acquired portfolios differ?" | **none** | falls to the generic path, which cannot apply the axis |

The merge arm filled a concept the sentence genuinely states; the generic path
could not honour it; the silently-dropped-dimension guard refused. **Every part
of that behaved correctly.** The gap is that the comparison recogniser does not
accept *"How do X and Y differ?"*.

Worth noting which arm is actually worse: the **off** arm answers that same
question with a plain 640-loan count — an answer to a different question, which
escapes a WRONG grade only because there is no computable truth for it. The
merge arm's refusal is the better behaviour of the two.

Nothing to do with the `direct` collision. Logged as its own item.

---

### Environment

`MI_AGENT_LLM_PARSER=off` throughout (F2). The merge arm replayed recorded
proposals with `llm_query_parser._call_llm` replaced by a tripwire that kills the
run rather than letting a live call answer quietly. **Successful model responses:
0.**

Runs are executed from the repository root. A run from any other working
directory silently loses two capability-aware refusals: `trakt_core.capability.
load_registry` resolves `config/system/mi_capability_registry.yaml` relative to
the process cwd, and `_capability_explanation` swallows the `FileNotFoundError`
and returns `None`. Found while checking pack fidelity; logged as its own item.
