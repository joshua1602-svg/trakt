# Release-candidate baseline — re-established on the correct tree

**The previous baseline is VOID.** It was measured against a tree that does not
contain the release candidate. This document replaces it entirely.

| | |
|---|---|
| Release candidate | `28ece25` — *Honour-or-clarify for populations* |
| Tag | **`mi-query-agent-rc`** — created here; there was no tag before |
| Branch base | `4e051f3` on `claude/mi-analytical-capability-layer-vlkjfw` |
| Book | real funded tape, 11,035 loans, £1,964.89m — never `build_fixture` |
| pandas | 2.3.3 (the repo pins `<3.0.0`) |

---

## Why the previous baseline was void

Four independent faults, any one of which invalidates it:

1. **Wrong base.** The previous branch was cut from `main` at `1351c51`. None of
   `28ece25`, `32c263a`, `b88286d` or `4e051f3` is an ancestor of it. The
   merge-base with the correct branch is `5dbda29`, **136 commits back**.
2. **`4e051f3` and `28ece25` are not in `main` at all**, so no base cut from
   `main` could ever have contained them. The clone's remote refs held only
   `main` and the working branch; the analytical-layer branch was never fetched.
3. **The parser measured was 1,640 lines short.** `llm_query_parser.py` differs
   by +1640/−48 between the old base and the release candidate. The calibration
   bank differs by +647/−148; the registry by +159/−6.
4. **The bank was graded against `build_fixture`.** The release candidate
   deliberately refuses that fallback — it "is what let the bank report 251/252
   while scoring 125/252 on a real book". The old run used exactly the synthetic
   frame the tree rejects, on a pandas version the repo does not support.

The revert commit `b88286d` records the release candidate's own figures —
*calibration bank, real book: 252/252* and *correct/disclosed 91.0%*. Those are
the numbers the brief quoted. The previous report called them unmatchable and
"corresponding to nothing this repository produces". **They match exactly**, on
the correct tree with the real book. That misreading was a consequence of the
wrong base, and it is withdrawn.

Reproduced here: the calibration bank runs **260 passed, 0 skipped, 0 xfailed**,
and the bank comparison below shows **252/252**.

---

## The four defect occurrences — all four are FIXED

Re-checked case by case on the release candidate. The previous report claimed
occurrences 1 and 2 were live and that occurrence 1 was "worse than described".
**Both claims were artefacts of the wrong tree and are withdrawn.**

| Occurrence | Question | Release candidate | Verdict |
|---|---|---|---|
| 1 — `_detect_metric` | *balance where LTV above 50%* | metric `current_outstanding_balance`, agg `sum`, filter `current_loan_to_value gt 50.0` | **FIXED** |
| 2 — `wants_balance_too` | *how many loans have a balance above £250k* | agg `count`, metric `None`, filter `current_outstanding_balance gt 250000.0` | **FIXED** |
| 3 — `answer_type.asked` v1 | *balance by region where borrower age is over 70* | subject balance, grouping `collateral_geography`, filter `youngest_borrower_age` | **FIXED** |
| 4 — `answer_type.asked` v2 | *balance by LTV bucket* | subject balance, grouping `ltv_bucket` | **FIXED** |

The LTV filter-drop is fixed, exactly as recorded. The subject stays `balance`,
the aggregation stays `sum`, and the filter is applied rather than dropped.

**The occurrence list in the previous report is rewritten to: none of the four
is live at the release candidate.** That removes the primary evidence the
previous report offered for building the layer, and it has to be said plainly
rather than replaced with something else.

---

## The three "fabricated bindings" — one artefact, two survive, neither silent

| Claim | Release candidate | Verdict |
|---|---|---|
| *how many loans are in the book* binds a geography filter | `filters={}`, `ok=True` | **Artefact of the wrong tree. Withdrawn.** |
| *how much is in the good book* binds a geography filter | binds `geographic_region_obligor='Good'` | **Survives at parse level** |
| *funded balance by day* binds an arrears grouping | binds `arrears_bucket` | **Survives at parse level** |

But the previous report's framing — "each narrows or slices the population on a
condition the user never stated, and each is invisible to both banks" — is
**wrong on this tree**. Both survivors **fail closed**:

* *how much is in the good book* → `ok=False`, and the message is
  *"No loans in this book match that filter (geographic_region_obligor), so
  there is nothing to calculate. I have not returned a whole-book figure in its
  place."*
* *funded balance by day* → `ok=False`, and the message is
  *"'Arrears Bucket' is not available in this dataset … no value was
  fabricated."*

Neither produces a wrong number. Both are **right-for-wrong-reason**: the block
is correct, the stated reason is not. A user who asks about "the good book" — a
credit-quality phrase — is told no loans match a *geography* filter. A user who
asks for a daily series is told about *arrears*, rather than that the book is
month-end snapshots and a daily grain does not exist.

That is a real and reportable finding. It is a much weaker one than "fabricated
bindings both banks pass", and it is stated at its true strength.

---

## The three instruments, re-run

### (a) Adversarial probe set — 22 / 40 (55.0%)

Previously reported as 15/40 on the void tree.

| Shape | RC | previously (void) |
|---|---|---|
| a measure word inside a filter clause | **7 / 9** | 2 / 9 |
| a measure word inside a grouping clause | **10 / 10** | 8 / 10 |
| a dimension value misreadable as an axis | 5 / 11 | 5 / 11 |
| a vague phrase with no bound | 0 / 10 | 0 / 10 |

The two shapes the four named defects belong to are now 17/19. The release
candidate handles the defect class this layer was proposed to retire.

### (b) Vocabulary blast radius — 0 regressions, again

| Bank | compared | pass before | pass after | regressions |
|---|---:|---:|---:|---:|
| calibration bank (real book) | 252 | **252** | 252 | **0** |
| generated harness | 249 | 249 | 249 | **0** |
| adversarial probes | 40 | 22 | 22 | **0** |
| time-series probes | 24 | 1 | 1 | **0** |

252/252 corroborates the release candidate's own recorded figure, which is a
useful check that the harness is now pointed at the right thing.

The flat result survives the correction. One disclosure: **`coupon` is already a
synonym of `current_interest_rate` on this tree**, so one of the five synonyms
adds nothing and only four are real additions. The five are left as authored
rather than retuned.

### (c) Time-series probe set — 1 / 24 (4.2%)

Unchanged by the correction. The single pass is *loan count over time*. Grain is
never carried; *by quarter* and *each month* are not recognised as a time axis
at all; a second grouping axis is dropped when a time axis is present; relative
windows bind nothing.

**This is the one instrument whose previous result stands.** The capability the
brief says has no home still has no home.

---

## The instruments themselves are not yet fit for this tree

This is the most important finding of the re-baseline, and it is a fault in my
work rather than in the release candidate.

Both probe sets score `_deterministic_parse` and nothing else. **The release
candidate does a large part of its classification after parsing**, in
`mi_agent/execution_receipt.py` — a facet layer that detects material facets
from the RAW QUESTION before any parsing decision, carries a field key per
facet, and reconciles each against what execution actually did. Its statuses map
onto this model's span states directly:

| receipt status | span state |
|---|---|
| `applied` | filled |
| `unavailable` / `unsupported` / `rejected` | filled-but-unresolvable, disclosed |
| `lost` | filled-but-unresolvable, fails closed |

**So the assertion in the previous baseline that "the release candidate has no
unresolvable channel" is false on this tree.** It has one; it is not in the spec,
which is why a parse-only reading cannot see it.

Four probes scored as failures produce correct governed clarifications end to
end:

| Probe | Parse-level | End to end |
|---|---|---|
| *balance for large loans* | agg `count`, no filter | *"'large loans' does not state a threshold I can apply. Give a bound (for example 'over 80') and I will filter on it"* — **exactly the filled-but-unresolvable filter the probe asked for** |
| *balance excluding London* | filter dropped | *"I understood that you asked for London, but that could not be applied"* — the unapplied filter is surfaced, not dropped |
| *total balance where days in arrears exceed 90* | filter dropped | multi-measure facet raised; refuses rather than answering the whole book |
| *balance in London by broker* | filter dropped | refuses — `broker_channel` is not in this dataset |

So **22/40 understates the release candidate**, and by an unknown margin.

### The workflow-aware projection is started but not valid yet

`rc_workflow` reads the spec plus the reconciled `execution_receipt`. It scores
**21/40 and 0/24**, which is *lower*, and the reason is a gap not a result:
**a query that fails closed produces no receipt at all**, because the receipt is
built on the successful path. So the tree loses precisely the clarifications
that motivated building it. Its classification on those cases lives in
`validation` and `error` instead.

Reading the failed path means choosing how to map free-text refusals onto spans,
and that choice moves the headline number. **I am not making that choice
unilaterally after seeing results.** It is the first thing to settle at review.

### Two probes are mis-authored against this tree's domain

Found while re-checking, and disclosed rather than quietly corrected:

* `adv_vag_11` *balance for the back book* expects an unresolvable filter.
  **`back book` is a governed synonym of `seasoning_segment`** here, along with
  `front book`, `new lending` and `legacy book`.
* `adv_vag_10` *what is the balance of the recent vintages* expects an
  unresolvable filter. **`vintages` is a governed synonym of `vintage_year`.**

In both cases the field is governed and only the *value* is unbounded — the
two-level distinction the rules file already draws for rule 16. The correct
expectation is a filter on the named field with an unresolvable value, not an
unresolvable filter.

**The scores above are reported as authored, with these two counted as
failures.** Correcting them would raise the release candidate's score. Moving an
instrument after seeing the number is what the ordering exists to prevent, so
the correction is proposed here and left for review.

---

## What this changes

The previous report recommended building the layer, resting on: two live defect
occurrences, three fabricated bindings, and a time-series gap.

On the correct tree:

* **The two live defect occurrences are not live.** All four are fixed.
* **One fabricated binding does not exist**; the other two fail closed and are
  right-for-wrong-reason, not silent wrong answers.
* **The time-series gap is real and unchanged** — 1/24.
* **The blast radius is flat**, on a valid tree and a real book.
* **A facet layer already exists** that detects from the raw question before
  parsing decisions, carries per-facet field keys, and has unresolvable-equivalent
  states — which is a substantial part of what §1 proposes building.

The case for the layer is now materially weaker than the previous report claimed
and rests mainly on the time-series gap and on the right-for-wrong-reason class.
**No conclusion is drawn here.** The rules were not revised and Phase 2 was not
started, per instruction. The corpus validation report is withdrawn pending
re-run on this tree; its rule `corpus_support` counts were measured against a
calibration bank that differs by +647/−148 and are stale.

## Next decisions, for review

1. **How to read the failed path** into spans, so `rc_workflow` becomes a valid
   instrument. This moves the headline number and should not be settled by
   whoever is holding the result.
2. **Whether to correct `adv_vag_10` and `adv_vag_11`.**
3. **Whether the existing facet layer already is the clause layer**, or a
   partial one worth completing in place rather than replacing.
4. Re-run the corpus validation on this tree once 1 and 2 are settled.
