# Wiring the completeness check — scope, and a stop

Base `3de7008`. **The wiring is not shipped.** You asked me to stop and report
if it would convert anything currently CORRECT into a refusal. It would, and the
reason is not the one either of us expected.

One change did ship: a correction to the **check itself**, which changes no
answers because the check is not wired. It is what cleared the stop condition on
the review pack — and it is also what made the real blocker visible.

---

## The prediction, recorded before measuring

> *It fires on 20 of 21 type-(c) losses and 0 of 157 delivering questions, so I
> expect the wiring to convert false refusals into better-worded false refusals
> and Q17C into a refusal.*

**The first half is exact.** The calibration still reads 19/20 type-(c), 0 of 31
75-bank EXACT, 0 of 73 CFO-91 EXACT, 0 of 53 composition DELIVER — 157 delivering
questions, no false positives.

**The second half does not follow, and the error is in the mechanism.** A guard
only ever looks at a **delivering** envelope — that is the condition every
existing guard in `mi_service` applies, for the reason that converting a refusal
into a differently-worded refusal is not what a guard is for. So wiring cannot
reword a false refusal. What it does is convert **answers** into refusals. The
question is therefore not "which refusals improve" but "which answers stop".

---

## Where it would sit

`mi_agent_api/mi_service.py`, as a **sixth guard**, at both answer sites
(lines 1233–1243 routed, 1309–1317 point-in-time) after the five that exist:

```
_guard_routed_answer · _guard_temporal_honouring · _guard_unresolved_scope
_guard_unknown_category · _fail_closed_analytical
```

The template is `_guard_unresolved_scope`, and its three design rules carry over
without argument:

- **only a delivering envelope** is examined;
- **the refusal sentence is not written in the guard** — the concepts are raised
  as LOST facets and `execution_receipt.assess` produces the wording, so the
  refusal reads like every other dropped-narrowing refusal on the surface. A
  second author of that sentence is the defect this programme spent seven
  consolidations removing;
- **`except Exception` around the whole thing** — a disclosure step never breaks
  a governed answer.

`semantics`, `frame` and `_book_values(df, semantics)` are all already in scope
at both sites, which is everything `stated_concepts` needs. `from_envelope` takes
the finished envelope, so **all routes are covered**, not just the workflow path.

### It is a third invariant of a family that already exists

`mi_agent_workflow` already refuses on two fail-closed invariants:

| invariant | compares | refuses when |
|---|---|---|
| `check_dimension_invariant` | **parsed** dimensions vs executed | a parsed axis is neither applied nor rejected |
| `check_filter_invariant` | **parsed** filters vs executed | a parsed filter is silently not applied |
| *(this one)* | **stated** concepts vs executed | a stated concept is not positively recorded |

Q17C is the case that shows why the third is needed: its two lost axes were never
*parsed*, so neither existing invariant has anything to check. This one sits one
step upstream, at the sentence.

## What it would do when it fires

`ok=False`; `answer` and `error` set to the wording `assess` produces;
`artifacts` emptied; `executionSummary` set to `None` (so the figure the refusal
says it will not substitute is not left on the envelope for a channel to render);
`controlledRefusal` and `semanticGuard` stamped; the message appended to
`warnings`. Byte-identical plumbing to `_guard_unresolved_scope`.

---

## Blast radius

### The review pack, 166 questions, both arms

| arm | delivering | would fire | of those, currently CORRECT |
|---|---:|---:|---:|
| off | 130 | 8 | **0** |
| merge | 131 | 3 | **0** |

Off arm, all eight are answers that should not stand: five WRONG (Q03A, Q05C,
Q16B, Q17C, Q19A) and three NO_COMPUTABLE_TRUTH (Q07B, Q10A, Q25C). Q17C fires
and refuses its 5-row answer — exactly what you asked for.

**This is after the correction below. Before it, Q17C on the merge arm — CORRECT,
143 cells — would have been refused.**

### The 1,446-question corpus — and this is the blocker

| | |
|---|---:|
| asked | 1,446 |
| delivering | 854 |
| would fire | **39** (4.6% of delivering) |

| kind | n | verified |
|---|---:|---|
| `dataset` | 15 | **all genuine losses** |
| `value` | 9 | genuine (drawdown, lump sum, acquired) |
| `scope` | 8 | **at least 3 are correct answers** |
| `measure` | 5 | mixed |
| `dimension` / `facet:grouping_dimension` | 4 | a threshold read as an axis |

**The `scope` class blocks the wiring.** Measured against the tape:

```
Summarise the acquired book         -> 199 loans, £54.7m   CORRECT and NARROWED
Summarise the direct book           -> 441 loans, £117.4m  CORRECT and NARROWED
portfolio summary for the acquired book -> 199 loans        CORRECT and NARROWED
                                       ...and all three publish scopeApplied: None
```

The check is **not wrong** about them. The envelope genuinely records no
narrowing, and Q19C proved that an envelope indistinguishable from these carried
a figure wrong by £10.2m. That is the finding the check exists to make. But as a
**gate** it would refuse three correct answers, and eight are in the class.

**The remedy is the one Stage 1 already applied.** Its own record reads: *"the two
scope routes now publish `metadata.scopeApplied`"*. These are further routes —
`portfolio_summary` and `funded_bridge` — with the same silence, invisible on the
157-question calibration surface and visible on 854. Make them publish, and the
class empties.

### An unasked-for finding that is worth more than the wiring

**Fifteen questions naming the pipeline or a forecast are answered against a
different dataset**, with clean receipts, and the contradiction is already sitting
in the envelope:

```
"Summarise the current pipeline."
   metadata.datasetContext : pipeline
   reconciliation.dataset  : funded
   -> "At 30 June 2026 the portfolio holds 640 loans with a funded balance
       of £172.1m…"
```

The reader asked about the pipeline and was given the funded book, in full
confidence. Every one of the fifteen has `datasetContext` disagreeing with
`reconciliation.dataset`. None is graded WRONG, because none has a computable
truth — which is how fifteen of them survived every bank this programme has run.

---

## The correction that shipped, and why it was needed

`question_interpretation/completeness.py` — `_bands_of` + `StatedConcept.carried_by`.

> **A band of a field is that field.** "Show balance by borrower age bucket"
> states `youngest_borrower_age`; the contract groups by `age_bucket`, which the
> registry declares `derived_from: youngest_borrower_age`. The concept reached
> the contract — banded, and as an axis rather than a measure. Role disagreement
> is not loss, which the check's own docstring already says for the grouping kind
> and did not apply to the measure kind.

Read from `derived_from`, declared for all nineteen derived fields. No list; a
bucket added tomorrow is covered without an edit.

**It cured fourteen false positives** across the corpora, every one a bucket
naming its own source field:

```
Show balance by borrower age bucket.      lost: measure 'borrower age'   (7 groups by age_bucket)
Show balance by interest rate bucket.     lost: measure 'interest rate'  (5 groups by interest_rate_bucket)
balance by ltv band                       lost: measure 'ltv'            (6 groups by ltv_bucket)
loan count by ticket size                 lost: measure 'ticket size'    (5 groups by ticket_bucket)
… and ten more
```

It is also what cleared the stop condition: on the merge arm Q17C groups by
`age_bucket`, so the concept is carried and the check is now correctly silent —
while on the off arm, where only `ticket_bucket` is applied, it still fires. The
right answer is left alone and the wrong one is caught, which is the whole point.

**The calibration is unchanged**: 19/20 type-(c), 0 false positives on all 157.

---

## What I did not do

**I did not wire it.** The corpus says three correct answers would be refused, and
you asked me to stop.

**I did not start the coordinated-axis role read.** It was ordered after the
wiring, and the wiring is blocked. It is now the more attractive of the two: it
recovers Q17C by parsing the sentence correctly rather than refusing it, and it
needs no gate.

## The order I would propose

1. **Make `portfolio_summary` and `funded_bridge` publish `scopeApplied`** — the
   change Stage 1 already made for two other routes. Empties the `scope` class
   and is the precondition for gating on anything.
2. **The dataset class** — fifteen questions answered from the wrong dataset is a
   larger finding than the wiring, and it is not a disclosure problem.
3. **The coordinated-axis role read** — recovers Q17C properly.
4. **Then wire**, when the delivering-firing set is only genuine losses.

Ruling yours. Nothing below step 4 depends on it.

---

### Environment

`MI_AGENT_LLM_PARSER=off` throughout (F2). Merge arm replayed with
`llm_query_parser._call_llm` replaced by a tripwire. All runs executed from the
repository root (F6). **Successful model responses: 0.**
