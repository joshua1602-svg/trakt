# ESCALATION — a fabricated population produces a plausible incorrect successful answer

**Found during:** Second-Book Commercial Generalisation review (build fixture / review / measure only).
**Status:** OPEN — escalated, not fixed. No production code, registry, parser or route changed.
**Affects:** the production LLM path on **both** books. The deterministic path is correct on both.
**Severity:** proposed **BETA BLOCKER** — see §5.

---

## 1. The defect

> **"What is the balance of the sponsored book?"**

The governed ruling (P1I-A follow-up) is explicit: *"sponsored book" is a governed scope phrase
meaning the sponsor's full AuM across all directly originated and acquired portfolios — it is
equivalent to `ENTIRE_AUM`.*

On the LLM path the model instead invents a **seasoning population** and the answer is computed
over the front or back book:

```
Calculated: Total Balance · Seasoning Segment = back book · 9,858 loans · as at 30 June 2026.
```

`ok=True`. `semanticGuard: ok`. `warnings: []`. No facet raised.

## 2. Measured, both books

| Book | Runs | Delivered | Correct answer | Error |
|---|---|---|---|---|
| **Alderbridge** (production demo book) | **3 of 3** → back book | **£1,793,150,141.49** | £1,964,886,258.21 | **−8.7%** |
| **Kestrelmoor** (second book) | **2 of 3** → front book | **£238,685,188.37** | £1,331,647,994.86 | **−82%** |
| **Kestrelmoor** | **1 of 3** → back book | **£1,092,962,806.49** | £1,331,647,994.86 | **−18%** |

Two things make this worse than a single wrong answer:

1. **It is non-deterministic.** The same question returned *two different wrong figures* across
   runs on Kestrelmoor, and never the correct one.
2. **The Alderbridge error is the plausible one.** £1.79bn against a true £1.96bn is an 8.7%
   difference on a headline AuM figure. Nothing on screen would prompt a CFO to question it.

**The deterministic path answers correctly on both books** (£1,964,886,258.21 and
£1,331,647,994.86), so this is purely an LLM-path defect.

## 3. Trigger

The fault is phrase-specific, which is why it survived earlier gates:

| Question | Result |
|---|---|
| "What is the balance of the sponsored **book**?" | **WRONG** — seasoning population invented |
| "What is the sponsored book worth?" | correct — entire funded portfolio, 3/3 |
| "How large is the sponsored portfolio?" | correct — entire funded portfolio, 3/3 |

The word *book* adjacent to *sponsored* appears to pull the model toward the front/back book
vocabulary, which P1J-1 governs as a seasoning segment.

## 4. Why the existing guards do not catch it

This is the **mirror image of P1K/P1L**, and the architecture only covers one direction.

P0's facet ledger and P1L's population evidence are built to detect a **lost** intent — something
the question asked for that execution did not honour. Here the opposite happens: execution
honours a population the question **never requested**. The predicate was applied, the frame
narrowed, rows-before/rows-after were recorded, so the population facet is legitimately
`APPLIED`. Every guard reports success.

Nothing in the chain asks the question that would catch it:

> *did the question actually request this population?*

P1I-A's `mask_scope_phrases` exists precisely to stop a governed scope phrase becoming a row
filter, and `names_total_scope` already contains "sponsored book". But that machinery governs
what the **deterministic** parser builds; it does not adjudicate a filter the **model** invented,
and no cross-check reconciles the LLM's emitted population against the governed scope resolution
for the same phrase.

## 5. Severity

**Proposed BETA BLOCKER.**

- It returns a wrong number as the headline answer, with `ok=True` and no disclosure.
- It reproduces on the **production demonstration book**, not only on the new fixture.
- The Alderbridge error (−8.7% on AuM) is entirely plausible and undetectable by the reader.
- "The sponsored book" is the client's own term for their whole book — a first-session question,
  not an adversarial probe.
- It is non-deterministic, so it cannot be relied on to fail visibly in testing.

The only mitigation is the calculation trace, which does say `Seasoning Segment = back book`.
That is the same mitigation P1M considered and the product owner overruled: naming the
substitution in the trace is not the same as not making it.

## 6. What I have NOT done

No production code, semantic registry, parser rule or route changed. No fix attempted. Nothing
pushed. The second-book fixture is isolated under the review scratchpad and the repository tree
is clean; P1I/P1L/P1M/P1N gates pass 193/193 unchanged.

## 7. Reproduction

```
scratchpad/secondbook/k25.py alderbridge     # 3/3 wrong, back book
scratchpad/secondbook/k25.py kestrelmoor     # 2/3 front book, 1/3 back book
```

## 8. Direction requested

1. Confirm the **BETA BLOCKER** classification.
2. Confirm the intended fix shape. My reading is that this is a **SAFETY_FIX**, not a breadth
   change: a governed scope phrase that resolves to `ENTIRE_AUM` must not be answerable with an
   invented row population, on either parser path.
3. Confirm whether the general invariant — *a population that the question did not request may
   not be applied silently* — should be taken as its own gate, since the current ledger provably
   only guards the losing direction.
