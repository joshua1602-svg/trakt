# TRAKT MI Query Agent — P1F Exposure Semantics & B21 Completion

**Branch** `claude/mi-query-agent-review-n8d33r` · **Base** `1905b54` (P1E, accepted)
**Fixture** `demo_platform` / alderbridge — 11,035 loans, £1,964,886,258.21, as at 30 June 2026
**40-question bank** sha256 `e0fc0b61…3194` — **unmodified**

---

## 1. Root cause

### Why the two parsers disagreed about "exposure"

The registry already settles the meaning. `current_outstanding_balance` carries
**`exposure`** among its synonyms; `exposure_at_default` carries `ead`,
`exposure at default`, `default exposure`, `exposure amount`. The deterministic
parser reads that list in full and has always resolved "total exposure" to
balance.

The model never saw it. `compact_catalogue` truncated every field's synonym list
to the first three:

```python
syn = ",".join((entry.get("synonyms", []) or [])[:3])
```

`exposure` is **fifth** on the balance field, so it was cut. `exposure at
default` is **third** on the EAD line, so it survived. The only exposure word in
front of the model belonged to EAD, and that is what it chose.

**This corrects the P1E report.** §11.3 there concluded "not a catalogue defect
… it over-read the generic word". That was wrong. The model was not misreading
the question; it was never shown the word that settles it. Which three synonyms
survived was an accident of list order, and 99 synonyms across 51 core fields
were hidden the same way.

### Why B21 answered half a question

Three separate faults, none of them arithmetic — the concentration workflow was
already computing the right numbers.

1. **The question never reached it.** `_TOP_EXPOSURE_RE` and `_SINGLE_NAME_RE`
   both required the noun to follow the superlative immediately, so "largest
   **single-loan** exposure" and "biggest **individual** loan" matched neither.
   The question fell through to the point-in-time path, which cannot express a
   single-name share, and P1E's share facet correctly refused it.
2. **The share printed as zero.** `_share_pct` formatted to one decimal, so
   0.042834% of the book rendered as **"0.0% of exposure"** — a real
   concentration displayed as nothing.
3. **The amount was never stated.** The answer named the loan's identifier and
   its share, which answers neither half of the question and puts loan-level
   data in a portfolio headline.

---

## 2. Governed exposure semantics

| Language | Resolves to | Behaviour on this book |
|---|---|---|
| exposure · total exposure · current exposure · portfolio exposure · book exposure · outstanding exposure | `current_outstanding_balance` | answered |
| EAD · exposure at default · regulatory EAD | `exposure_at_default` | **refused** — not carried; no substitution |

The distinction is enforced in both directions and asserted: the same book
answers "What is the total exposure?" and refuses "What is the exposure at
default?", so the two concepts are genuinely distinct rather than aliased.
"exposure band" (a `ticket_bucket` synonym) does not capture the measure.

**How it is carried.** Lifting the catalogue truncation was implemented, tested,
and **reverted** — see §8. The convention now lives in prompt rule 11, which
names the current-outstanding-balance concept directly rather than pointing at a
synonym the model cannot see. One instruction changed, not 99 synonyms.

A test pins the arrangement rather than leaving it to be rediscovered: the
catalogue **still** hides `exposure`, and rule 11 is asserted to compensate.

---

## 3. B21 calculation

Independent truth, pandas over the canonical fixture — the executor was not its
own oracle:

| | Value |
|---|---|
| Loans in scope | 11,035 |
| **Largest single-loan exposure** | **£841,638.96** |
| **Total book exposure** | **£1,964,886,258.21** |
| **Share of book** | **0.0004283398 → 0.042834%** |

| | Independent truth | Agent | Variance |
|---|---|---|---|
| Largest exposure | 841,638.96 | 841,638.96 | **0** |
| Total exposure | 1,964,886,258.21 | 1,964,886,258.2099957 | 0 (float) |
| Share | 0.00042833978632774816 | 0.00042833978632774816 | **0** |

Displayed: **"The largest single-loan exposure is £842k, representing 0.043% of
exposure."**

Receipt: **"Calculated: Largest single-loan current exposure · share of total
current exposure · 11,035 loans."**

---

## 4. B21 execution evidence

The route now declares structured evidence instead of leaving the receipt and
the share facet to infer from prose:

```json
{"kind": "loan", "grainField": "loan_identifier", "basis": "exposure",
 "population": 11035, "distinctNames": 11035,
 "topExposure": 841638.96, "topShare": 0.00042833978632774816,
 "totalExposure": 1964886258.2099957}
```

**Grain.** `kind == "loan"`, `grainField == "loan_identifier"`,
`distinctNames == 11,035` — one row per loan, and the numerator equals the
maximum single value in the book. Not a region, not a portfolio, not a grouped
aggregate.

**Denominator consistency.** `basis == "exposure"` on both sides;
`totalExposure` equals the sum of the same governed field over the same 11,035
loans.

**The share is re-derived, not trusted.** `_single_loan_share_proven` recomputes
`topExposure / totalExposure` and compares to the reported `topShare` within
1e-12. A mismatched denominator, a non-loan grain, or a missing share cannot
satisfy the facet — each is asserted.

Before this, the share facet was discharged by finding a percentage in the
answer text, which proves only that the text mentions one.

---

## 5. Exposure bank

| Case | Deterministic | LLM (forced) |
|---|---|---|
| GEN-1..5 generic exposure | 5/5 → `current_outstanding_balance` | 5/5 → `current_outstanding_balance` |
| EAD-1..3 explicit EAD | 3/3 refuse, no substitution | 3/3 refuse, no substitution |
| B21-1..5 paraphrases | 5/5 correct | 5/5 correct |
| B21-AMT amount only | correct, no share required | correct |
| B21-SHR share only | same numerator and denominator | same |
| B21-EAD explicit EAD | refuses | refuses |

### Parser agreement

| Phrase | Deterministic | LLM | Agreement |
|---|---|---|---|
| total exposure | `current_outstanding_balance` | `current_outstanding_balance` | ✅ |
| current exposure | `current_outstanding_balance` | `current_outstanding_balance` | ✅ |
| EAD | `exposure_at_default` | `exposure_at_default` | ✅ |
| exposure at default | `exposure_at_default` | `exposure_at_default` | ✅ |

---

## 6. P1E bank regression

Forced LLM, every question through the model:

| | Before (P1E) | After (P1F) |
|---|---|---|
| Five CFO questions | 5/5 | **5/5** |
| P1E 26-question bank | 25/26 | **26/26** |

The gain is P1E-02 — "Show me total exposure, loan count, weighted-average loan
to value and weighted-average interest rate" — the question that started this
phase. It now answers `Balance: £1.96bn · Loans: 11,035 · Weighted-average
Current LTV: 43.16% · Weighted-average Interest Rate: 6.56%`.

No P1E question regressed.

---

## 7. 40-question bank

Unmodified, both paths.

| Path | P1E baseline | P1F |
|---|---|---|
| Deterministic | 10/40 | **11/40** |
| Genuine LLM (production) | 11/40 | 11/40 |

**Deterministic: B21 is the only change** — refusal → correct. Nothing else
moved.

**LLM: B21 correct; B04 is the open failure** (§9).

---

## 8. Prompt blast-radius analysis

Measured, not estimated. The harness was run against two code states that differ
in **one file only** — `mi_agent/llm_query_parser.py` reverted to `1905b54` —
holding routing, share precision, receipts and grain evidence constant, so any
parse that moved is attributable to the parser change alone.

### Attempt 1 — lifting the catalogue truncation (REVERTED)

7 of 71 forced parses changed. Six were the target fix or improved refusal
messages. One was not:

> **B04 — "Is the credit quality of new origination better or worse than the
> back book?"** Refusal → **answered, grouped by Source Portfolio Type.**

That is a vintage question. This book carries no vintage, so the refusal was
correct. Direct-versus-acquired is not new-lending-versus-seasoned-lending: the
numbers were right for a question nobody asked. The mechanism was exact —
`source_portfolio_type` gained the previously hidden synonyms **"origination
type"** and **"book type"**, matching "new **origination**" and "back **book**".

**No cap can separate the two cases.** `exposure` is index 4 on balance;
`origination type` is index 3 on `source_portfolio_type`. Any cut admitting the
first admits the second. The catalogue change was reverted whole.

### Attempt 2 — prompt rule 11 only (SHIPPED)

6 of 71 forced parses changed:

| Question | Old | New | Intended |
|---|---|---|---|
| **P1E-02** total exposure + 3 measures | refused: EAD unavailable | **answered, 4 measures** | ✅ the objective |
| A1 average LTV / age / borrower type in London | refused: weight_field error | refused: Borrower Type unavailable | ✅ same verdict, truer reason |
| A5 balance by borrower type by product | refused: Product Type unavailable | refused: **both** fields named | ✅ same verdict, fuller reason |
| B13 which product type has highest ticket size | refused: ranking not applied | refused: Product Type unavailable | ✅ same verdict, truer reason |
| B14 is the acquired book converging on LTV | refused: Reporting Date unavailable | refused: dimension neither applied nor rejected | ✅ same verdict |
| **B04** credit quality of new origination vs back book | refused: Vintage unavailable | **answered, grouped by sourcing channel** | ❌ **not intended — §9** |

Five of six are the objective or strictly better refusals. B04 is a wrong answer.

---

## 9. Safety

| Measure | Deterministic | Genuine LLM |
|---|---|---|
| Incorrect successful | **0** | **1 — B04** |
| Silent semantic errors | **0** | **1 — B04** |
| Hard failures | **0** | **0** (one found and fixed, below) |
| Exposure substitutions | 0 | 0 |
| Lost-share failures | 0 | 0 |
| Grain failures | 0 | 0 |
| Receipt coverage | every answer | every answer |

### B04 — the gate failure, characterised honestly

B04 is **not created by P1F. It is pre-existing, stochastic, and P1F makes it
consistent.** Repeated runs of the identical question, forced LLM:

| Code state | Result |
|---|---|
| Pre-P1F (`1905b54`) | **4/5 refused**, 1/5 answered by sourcing channel |
| P1F (rule 11) | **5/5 answered** by sourcing channel |

The committed pre-launch LLM baseline records B04 refusing — for a *third*
reason again ("PD is not available"), which is itself evidence of how unstable
this question was. The gate was previously met on this question by sampling
luck, not by a guard.

It is wrong in **production** configuration too, not only in the forced
diagnostic mode.

**Why P0 does not catch it.** The question raises a `cohort_comparison` facet —
"a comparison between two books" — and the answer genuinely produced two
cohorts, so the facet is marked applied. The ledger records *that* two books
were compared, never *which two*. A wrong pair satisfies it as readily as the
right one.

**Why it is not fixed here.** Making the cohort facet verify which cohorts were
compared is P0 safety architecture, outside the two objectives this phase was
scoped to. Widening scope unilaterally at the end of a phase is how the
regressions in this programme have historically been introduced. It is §11.

### A hard failure found and fixed

The first LLM validation run died on "Show me balance by region by borrower
type" with `TypeError: unhashable type: 'list'` — the model returned
`dimension=["collateral_geography", "borrower_type"]`, and the validator looks a
field key up with `fields.get(key)`.

Same defect class this file had already been bitten by: list-shaped `filters`
once raised `TypeError: unhashable type: 'dict'` and were folded, but the
reasoning was never extended to the scalar slots. They are folded now on the
same terms — nothing discarded, and `dimension`'s extra names carry into
`dimensions`. P1F did not create this; it provoked it.

---

## 10. Known limitations

1. **B04 — §9.** Pre-existing, now consistent. The reason this report is a FAIL.
2. **The catalogue still hides 99 governed synonyms** across 51 core fields.
   Lifting it is the right fix and is *demonstrably unsafe until the cohort
   guard exists* — the evidence is in §8.
3. **`£842k` rounds £841,638.96.** House money formatting is 3 significant
   figures at thousands scale. Immaterial (0.08%) and product-wide, unlike the
   share, which rounded to literal zero and was fixed.
4. **No reporting date in the B21 receipt.** The tape carries no
   `reporting_date` column, so the receipt omits it rather than inventing one.
5. **`MAX_MEASURES` is not enforced on the LLM path.** B04's spec executed six
   measures. Parser-side only; the executor does not re-check.
6. **`median` → `sum`** remains a strict xfail, untouched as instructed.

---

## 11. Recommended next breadth increment

**Guard the cohort comparison, then lift the catalogue truncation.**

A `cohort_comparison` facet should record *which* cohorts the question named and
require the executed grouping to express them. Today "new origination vs back
book", "direct vs acquired" and "London vs the rest" are indistinguishable to
the ledger — any two-group answer satisfies any two-cohort question.

That fixes B04 on its merits, and it is the precondition for restoring the 99
hidden synonyms, which is the real fix for the parser disagreement this phase
worked around. Doing them in that order means the wider catalogue lands behind a
guard rather than in front of one.

Recommended only. Not implemented.

---

## 12. Git

**Branch** `claude/mi-query-agent-review-n8d33r` — **not pushed, not merged.**

| SHA | |
|---|---|
| `03c8c6b` | P1F: exposure semantics and B21 answering both halves |
| `cc01a8b` | Fold a list given in a scalar slot instead of crashing on it |
| `7b0654b` | Carry the exposure convention in the instruction, not by widening the catalogue |

The wrong turn is deliberately left legible: `03c8c6b` widened the catalogue,
`7b0654b` reverts that part and records what it broke.

| File | Change |
|---|---|
| `mi_agent/llm_query_parser.py` | prompt rule 11; catalogue docstring recording the reverted attempt |
| `mi_workflows/concentration_analysis.py` | superlative/qualifier patterns from one shared source; magnitude-aware `_share_pct`; loan-kind answer states amount and share |
| `mi_agent_api/chat_routing.py` | `_single_name_evidence` declared on the envelope |
| `mi_agent/execution_receipt.py` | `concentration_evidence`, `_single_loan_share_proven`, `_single_name_measure`; share facet proven from execution |
| `mi_agent/mi_query_spec.py` | scalar slots fold a list instead of crashing |
| `tests/test_p1f_exposure_semantics.py` | **new** — 44 tests |

---

## Verdict

Both stated objectives are met, on both parser paths, reconciled independently:
generic exposure resolves to the governed current-exposure measure; explicit EAD
stays distinct and refuses rather than substituting; B21 calculates both
requested outputs at loan grain with a consistent denominator and an
execution-proven share.

The mandatory safety gate is not met. `INCORRECT_SUCCESSFUL = 1` and
`SILENT_SEMANTIC_ERROR = 1` on the genuine-LLM path, because B04 answers a
vintage question with a sourcing-channel breakdown. That defect pre-dates this
phase and was previously intermittent; P1F makes it consistent, which is not a
defence.

The gate is the gate. It has not been redefined to let this pass.

`P1F EXPOSURE SEMANTICS: FAIL`
