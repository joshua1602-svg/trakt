# Opus semantic layer — shadow, activation attempt, and STOP

Base `bb360fe`, clean tree. **No production code was changed at any point in
this work.** The arm was activated by environment flag only
(`MI_AGENT_CONCEPT_MERGE=on`, `MI_AGENT_CONCEPT_MERGE_MODEL=claude-opus-5`),
precisely so that a failed gate could be undone by not setting it.

**Result: the first Opus run fails two hard gates. Stopped before any code
change, as briefed.**

---

## 1. Returned model identifier

`claude-opus-5` — read from the runtime response on **every** call, never from
the environment variable. 967 successful completions across four runs, all
`stop_reason: end_turn`:

| run | calls | successful | returned model |
|---|---|---|---|
| shadow, 24 CR4 | 24 | 24 | `claude-opus-5` ×24 |
| shadow, 166 + 3 must-refuse | 169 | 169 | `claude-opus-5` ×169 |
| live, 166 acceptance | 166 | 166 | `claude-opus-5` ×166 |
| live, 1,446 surface | 608 | 608 | `claude-opus-5` ×608 |

## 2. Shadow phase — clean

Shadow ran the real production seam with the ONE mutation suppressed
(`_apply_to_spec` against a deep copy), so the model was called with the
production prompt, vocabulary and deterministic interpretation while the spec
that executed was untouched. **Non-alteration was proven, not assumed: 166/166
answers byte-identical to the deterministic baseline.**

| | 169-question shadow |
|---|---|
| attempted / successful | 169 / 169 |
| valid semantic proposals | 168 (1 non-JSON reply → `proposal_unavailable`, degraded safely, contract untouched) |
| concepts proposed | 275 |
| accepted governed bindings | 275 |
| rejected — unregistered | **0** |
| rejected — ambiguous | **0** |
| unsupported concepts | **0** |
| conflicts | 30, every one fail-closed (27 person-chosen, 2 governed default, 1 unrecorded provenance) |
| model-selected canonical fields | **0**, by construction |
| ungrounded source spans | 3 / 275 |

The three ungrounded spans are elisions (`"Direct ... books"`,
`"the book ... acquired portfolio"`) rather than fabrications — both fragments
appear in the question, the middle is dropped. Two were declined for unrelated
reasons; **one (CFO65) would have reached the contract.** The schema does not
require a contiguous span and nothing checks it.

**Must-refuse controls: the model proposed NOTHING for all three** — no metric,
no period, no compare target invented for "What changed?", "Show me the trend."
or "Compare us with the market."

## 3. Activation — TWO HARD GATES FAILED

### Gate "no previously correct answer regresses" — FAILED, 5 answers

| id | question | was | now |
|---|---|---|---|
| Q23A | When will we reach £100m of funded loans? | CORRECT | refusal |
| Q23C | When does the funded book reach the £100m milestone? | CORRECT | refusal |
| CFO74 | At the current run rate, when do we reach £250m of loans? | CORRECT | refusal |
| CFO63 | What share of the book is drawdown? | CORRECT | refusal |
| CFO65 | What proportion of the book is in the acquired portfolio? | CORRECT | refusal |

### Gate "six protected pipeline answers do not regress" — FAILED, 1 answer

Q10B changed. See §5 — it is not a regression, and what it exposes is worse.

### Why — two governed-vocabulary gaps, neither a prompt defect

**Class A · a portfolio target becomes a row predicate.** For "when will we
reach £100m", Opus proposes `threshold · balance · at least · 100000000`,
correctly grounded on the span "£100m". The binder finds exactly ONE governed
owner for "balance" and `population.predicate_of` makes it a **per-loan**
filter. The guard then correctly refuses, because a filter selecting loans each
worth ≥£100m cannot be applied to a run-rate extrapolation.

The binder obeyed rule 2B exactly. The defect is that *"the book reaches £100m"*
and *"loans of £100m each"* are different concepts sharing one governed owner,
and the estate has no aggregate-target slot to tell them apart. This is F1 in a
new place: **a concept the vocabulary cannot express produces the nearest
expressible thing rather than nothing.**

**Class B · a share question gains an axis.** "What share of the book is
drawdown?" is answered deterministically as `Share of Balance · Product Type =
drawdown` — a FILTER plus a share. Opus proposes `drawdown` as a DIMENSION, the
binder binds it, the merge fills the empty `dimensions` slot, and the receipt
guard refuses: *"parsed dimension(s) neither applied nor rejected"*.

The axis-or-filter distinction (D2) is load-bearing elsewhere in this estate.
The proposal schema separates `dimension` from `category_value`, but nothing
constrains the model's choice to what the governed route can use, and the
merge's "fill an empty slot" rule has no notion of which slots a route accepts.

## 4. The 24 CR4 cases

| verdict | count | ids |
|---|---|---|
| **RECOVERED CORRECTLY** | **7** | Q01C, Q02B, Q03C (false refusal → correct); Q03A, Q05C, Q16B, Q17C (wrong → correct) |
| SAFE REFUSAL | 14 | Q01B Q04A Q05B Q07B Q10C Q12C Q15B Q15C Q17B Q20B Q21B Q21C Q23B Q24B |
| WRONG | 2 | Q04C, Q19A — both unchanged from the deterministic baseline |
| **REGRESSED** | **1** | Q23A (was CORRECT) |

Q07B moved WRONG → refusal, which is safer but is still movement.

Headline grades across the 166 move `CORRECT 118→119, WRONG 7→2,
FALSE_REFUSAL 22→25`. **That improvement must not be banked.** It is largely
failures moving between grading categories, and it was bought with five correct
answers — exactly what the brief forbids.

## 5. Q10B is not a regression — it exposes a grader defect

> *"Give me an overview of the pipeline **by size and stage**."*

| | grouped by | groups |
|---|---|---|
| deterministic | Pipeline Stage | 5 |
| Opus | Ticket Size **and** Pipeline Stage | 8 |

The deterministic answer **silently dropped the "size" axis the question
names**, and is graded CORRECT — on a frozen human verdict, with
`independent_truth: null`. Opus restored it.

So one of the six protected pipeline answers has been losing a requested axis,
and the instrument blessed it. That finding stands regardless of what happens
to the Opus layer, and it is the second time in this programme that a frozen
human grade has covered a silent loss.

## 6. The 1,446 surface — PARTIALLY MEASURED, and it says so

**608 of 1,446 measured (42%). The remaining 838 were never exercised.**

The API credit balance was exhausted mid-sweep at question 608; every question
from that index on returned `proposal_unavailable` with **no model call**, in
one contiguous block. The arm degraded safely — those answers are simply the
deterministic ones — but they are **NOT MEASURED**, not "no movement" (F3).

Within the measured 608: **25 movements (4.1%)**.

| | n | |
|---|---|---|
| answered → REFUSED | 3 | the £250m run-rate question (class A); "How do the Direct and Acquired portfolios differ?" and "Show the loans included in Unknown / Missing age" (class B) |
| refused → answered | 12 | seven `Balance by region for <region> loans.`, plus the numeric-threshold family that CR4 covers |
| answered, text changed | 8 | see below |
| refused, reason changed | 2 | |

The 8 changed answers are a mix this run cannot separate:

* **apparent corrections** — "How many drawdown loans have LTV above 50%?"
  144 loans/£37.1m → 45 loans/£11.3m (the drawdown filter was not being applied
  before); "WA LTV for lump sum lending in the Direct portfolio" 441 → 278 loans;
* **unverified and suspicious** — "Drill into the 50%+ LTV bucket" 6 groups → 2;
  "Break Direct portfolio balance down across LTV, ticket size and borrower age"
  5 groups → **143**; "How did balance change since last month for drawdown
  loans?" *"4 of 4 governed metrics"* → *"1 of 1"*.

None has independent truth. **No claim is made that any of the 8 is correct.**

## 7. What did NOT break

* must-refuse controls, arm live: **3/3 still refuse**, and the model proposed
  nothing for any of them;
* **0** must-refuse → answered;
* **0** dataset substitutions;
* **0** dropped deterministic positive claims — 206 agreed, 30 declined, none
  overwritten;
* **0** model-selected canonical fields;
* **0** model-invented required metric or period;
* frozen 278-module regression: **85 failing names, EXACT**;
* the deterministic arm is untouched — no production file was modified.

## 8. funded_bridge — ruling

**Not narrowed, and not revisited.** Its precondition is an ACTIVE semantic
layer. Activation failed its gates, so the replacement path for "What is the
weighted expected pipeline contribution?" and "Show funded vs pipeline
contribution." is not established, and narrowing on an unproven replacement is
exactly the move the brief prohibits.

## 9. What would have to change (NOT done — no code was touched)

Recorded so the next attempt does not start from scratch. Both are governed
vocabulary gaps, and neither is a prompt fix:

1. **An aggregate-target concept.** "The book reaches £X" needs an owner that is
   not a row predicate. Until one exists, a threshold on a measure that the
   question applies to the PORTFOLIO must bind to nothing and be reported
   unsupported — never to the row predicate that shares its name.
2. **A route-aware axis-or-filter rule at the merge.** A dimension fill must be
   refused where the selected capability cannot express an axis, instead of
   being written into the contract for a downstream guard to reject. The estate
   already owns this distinction; the merge does not consult it.

A third, smaller: the proposal schema should require a **contiguous** source
span, and the binder should reject one that is not. 3 of 275 were elisions and
one of those reached the contract.

---

# DO NOT FREEZE

The architecture held everywhere it was designed to. Across 967 live Opus calls
the model never selected a canonical field, never overwrote a human claim, never
invented a metric or period for a question that lacked one, never produced an
unregistered or ambiguous binding, and degraded safely when its own reply was
malformed. Every one of those is a gate, and every one passed.

It is still not safe to activate. Two governed-vocabulary gaps turn five correct
answers into refusals, and the wider surface adds eight changed answers that
nothing can currently verify. Seven genuine CR4 recoveries do not buy that.

The deterministic substrate is unchanged and remains where `bb360fe` left it:
frozen regression exact, must-refuse holding, no production file modified.
