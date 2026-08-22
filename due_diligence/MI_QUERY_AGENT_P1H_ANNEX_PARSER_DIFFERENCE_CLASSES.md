# P1H Annex — Five-Class Parser Difference Analysis

**Input for the next phase.** Companion to
`MI_QUERY_AGENT_P1H_SAFE_INTENT_RECOVERY.md` (NO-GO, premise disproven).

**Method.** The LLM spec is captured at the merge seam *before* any
carry-forward, so each classification describes the model's own output. 79
live-model parses across the unchanged 40-question bank, the P1E golden bank and
the curated P1H bank. Every conflict is adjudicated on the merits — which parser
is **semantically correct** — not by assuming deterministic precedence.

---

## 1. Distribution

Per question, worst class wins:

| Class | Questions | Share |
|---|---:|---:|
| AGREEMENT | 39 | 49% |
| ELABORATION *(LLM fills a slot the deterministic parser left empty, with a field the book has)* | 18 | 23% |
| OVER_EMISSION | 12 | 15% |
| TRUE_OMISSION | 6 | 8% |
| ROLE_MISASSIGNMENT | 3 | 4% |
| CONFLICT | 1 | 1% |

Per-slot observations: AGREEMENT 83 · ELABORATION 54 · OVER_EMISSION 14 ·
TRUE_OMISSION 8 · ROLE_MISASSIGNMENT 3 · CONFLICT 1.

**ELABORATION is benign and was separated out deliberately.** The deterministic
parser routinely leaves measure and dimension empty because it routed the
question to a specialist capability rather than building a chart spec. The model
filling those slots with fields the book carries is not a defect, and counting it
as over-emission — as a first pass did — overstates the problem threefold.

---

## 2. TRUE_OMISSION — adjudicated, 8 observations

| Question | Omitted | Was the deterministic parser right? |
|---|---|---|
| balance below 75% LTV | `current_loan_to_value` | **No** — LTV is the filter subject. LLM correct. Answers correctly today. |
| older borrowers / bigger loans vs property value | `current_valuation_amount` | **No** — LLM correct. Answers correctly today. |
| which borrower type has grown most | `collateral_geography` | **No** — geography was a deterministic misfire. |
| balance by LTV by borrower type | `amortisation_type` | **No** — a deterministic guess. |
| growing fastest by loan count rather than balance | `account_status` | **No** — a deterministic guess. |
| which product type has highest ticket size | `current_outstanding_balance` | Moot — refuses on absent `erm_product_type`. |
| cohorts closest to NNEG | `equity` | Moot — concept absent from book. |
| **balance by region by borrower type** | `collateral_geography` | **Yes** — region genuinely requested. Refuses anyway on absent `borrower_type`. |

**One of eight omissions had the deterministic parser right, and that question
refuses for an unrelated reason.** Carry-forward would have recovered nothing
and, on two of these, would have overwritten a correct LLM reading with a wrong
deterministic one.

---

## 3. ROLE_MISASSIGNMENT — 3 observations, and a correction

This class did not exist in my earlier report, and finding it **revises what I
told you about B25.**

### B25 — the concept is not dropped, it is misfiled

> *"How does the direct book compare with the acquired book on borrower age?"*

| | |
|---|---|
| deterministic | `youngest_borrower_age` as **measure** |
| LLM | `youngest_borrower_age` as **dimension**; measures `balance` [, `loan_count`] |

I previously reported that the model *"drops borrower age"*. It does not. **It
recognises borrower age and assigns it to the wrong slot** — it wants to group
*by* age, then picks balance and count as the things to measure.

Your conflict ruling still holds and the refusal is still correct: the model did
emit measures, and they are not the requested one. But the underlying defect is
narrower and more tractable than "the model ignores the question": the semantic
concept survives the parse with the wrong **role**.

**Adjudication: deterministic is correct.** Age is the measure being compared.

### B15 — the same shape, opposite verdict

> *"What proportion of the book is eligible for a 75% LTV securitisation?"*

| | |
|---|---|
| deterministic | `current_loan_to_value` as **measure** |
| LLM | `current_loan_to_value` as **filter**; measures `balance`, `loan_count` |

**Adjudication: the LLM is correct.** "Eligible for a 75% LTV securitisation"
makes LTV the eligibility predicate, not the thing being measured. Neither
parser expressed the *share* aggregation the question asks for, so the P1F share
guard refuses — correctly.

**These two cases point in opposite directions.** Any future role-resolution
work must adjudicate per question; neither parser is the standing authority.

---

## 4. CONFLICT — 1 observation

> *"If origination continues at the current rate, what will the balance be at
> year end?"*
> deterministic `[current_interest_rate, current_outstanding_balance]` · LLM
> `[current_outstanding_balance]`

**Adjudication: the LLM is correct.** "The current rate" is the rate of
origination — a run-rate — not the interest rate. The deterministic parser
matched the word "rate" to the interest-rate field. Refuses on projection
capability regardless.

---

## 5. OVER_EMISSION — 14 observations, two very different populations

Fields the model asserted that the book does not carry:

| Field | Count | Is the model wrong? |
|---|---:|---|
| `funded_status` | **3** | **Yes — recoverable.** Invented from "the funded portfolio / funded book / funded loans". |
| `reporting_date` | 2 | **Yes — likely same class.** A temporal scope read as a predicate. |
| `vintage_year` | 3 | No — the question genuinely asks about vintage; the book lacks it. |
| `borrower_type` | 2 | No — genuinely requested, absent. |
| `broker_channel` | 2 | No — genuinely requested, absent. |
| `erm_product_type` | 1 | No — genuinely requested, absent. |
| `negative_equity_guarantee` | 1 | No — genuinely requested, absent. |

**9 of 14 are the model correctly naming a concept the book does not carry** —
a correct refusal, not a parser defect, and not addressable by parser work.

**5 of 14 are genuine over-emission**, and they share one shape: *a phrase that
names the dataset's own scope, read as a predicate over it*. The whole governed
tape **is** the funded book; "the funded portfolio" is not a filter.

This is the mirror of the P1F defect where "the funded book" was read as a
**region**. Same phrase, three parsers' worth of wrong answers — deterministic
geography (fixed in P1F), LLM filter (open), and the temporal variant.

---

## 6. What this says about the next phase

Ranked by recoverable questions, with adjudication applied:

| Opportunity | Evidence | Tractability |
|---|---:|---|
| **Scope-phrase resolution** (`funded_status`, `reporting_date`) | 5 observations | **High** — one governed concept: a phrase naming the dataset scope is not a predicate. Precedent exists in P1F's `_NON_PLACE_TERMS`. |
| **Semantic role resolution** (measure vs filter-subject vs dimension) | 3 observations, verdicts split | **Medium** — must adjudicate per question; neither parser is authoritative. B25 is the worked example. |
| Carry-forward of omitted intent | 1 useful observation | **None** — rejected, §2. |
| Missing governed concepts | 9 observations | **Not parser work** — the book lacks the fields. |

**Recommended shape for the next phase: semantic role resolution, with
scope-phrase resolution as its first and simplest case.** Scope phrases are a
degenerate role problem — the phrase has no role at all, and the parsers keep
assigning it one. Doing them together means one governed mechanism rather than
two, and the harder measure/filter/dimension adjudication lands behind the
easier case rather than in front of it.

Caution carried forward from P1F: **dropping a filter silently broadens a
population**, which is the P0 cardinal sin. The safe form is a governed
scope-phrase registry that prevents the filter being created, not a heuristic
that deletes filters after the fact.

---

## 7. Caveats

* **Single-run sampling.** One live parse per question. Model output is
  stochastic — P1G established that a single sample cannot distinguish a stable
  behaviour from a coin flip. The *classes* are stable enough to direct a phase;
  the exact counts are not a contract.
* **`ELABORATION` is a judgement call.** It marks the LLM filling a slot the
  deterministic parser left empty with an available field. Benign here because
  the deterministic parser had routed the question elsewhere — worth
  re-examining if a future phase changes routing.
* **Adjudications are mine.** Each is stated with its reasoning above so it can
  be disagreed with individually rather than taken on trust.
