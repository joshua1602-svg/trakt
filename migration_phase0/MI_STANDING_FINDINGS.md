# Standing findings — MI Query Agent interpretation programme

Findings that outlived the stage that produced them. Each is here because it
changed how the next piece of work was scoped, and each is stated so that it can
be checked rather than believed.

---

## F1 · A vocabulary gap does not produce silence. It produces the nearest expressible thing.

**Measured, Stage 3.** The concept vocabulary offered to the model was
constrained hard, and both constraints held exactly as designed:

- **zero** raw governed field keys proposable;
- **zero** fields this book does not carry proposable.

And the vocabulary still produced a three-way narrowing error.

There is no concept kind for a numeric threshold, so *"borrowers over 75"*
cannot be said in registered concepts at all. The model did not fall silent. It
reached for the nearest expressible things: the FIELD as a `measure` — declined
32 times — and the BUCKET VALUES as category values, `75-80`, `80-85`, `85+`,
the first of which filled an empty slot. *"Balance for borrowers over 75"*
became *"balance for `age_bucket` == 75-80"*, on a question that is EXACT today.
8 of 157 correct-today questions would have taken that fill.

**The rule.** A constraint that cannot express a concept does not stop the model
reaching for it. When designing a constrained vocabulary, the question is not
"can it say anything wrong?" — it is **"can it say everything the sentence
says?"** Where it cannot, enumerate what the nearest expressible substitutes are
and measure whether they are reached, because the substitution is silent by
construction: every proposal is in-vocabulary, every binding is registry-owned,
and every guard reports green.

This generalises past this programme. It is the same shape as a validator that
accepts only known enum members and a producer that has no member for the state
it is in.

---

## F2 · An API key in the environment switches on the shipped free-form parser arm, so any "deterministic before" captured that way is void.

**Measured, Stage 3.** `mi_agent_api.datasets._mi_llm_config` runs `auto` by
default and sets `enabled = has_key`. With `ANTHROPIC_API_KEY` present, every
`/mi/query` call is parsed by the free-form LLM arm — the one that emits a whole
`MIQuerySpec` — which is the arrangement the concept-proposal split exists to
replace.

The first full Stage 3 run was captured that way and had to be discarded. It was
caught only because two must-refuse questions answered while the merge had
filled nothing on either, and it took a stashed working tree to establish that
the merge was not the cause.

**The guard, and it stays.** Any harness measuring a deterministic baseline
while holding a key for its own model calls must refuse to run unless
`_mi_llm_config()` reports `enabled=False available=False`. `MI_AGENT_LLM_PARSER=off`
forces that while leaving the key available for direct calls. The check belongs
at the top of the harness, as a hard exit, not in its documentation:
`migration_phase0/must_refuse_both_arms.py` and the Stage 3 harness both carry
it.

---

## F3 · An instrument that cannot be measured must report NOT MEASURED, never clean.

**Stage 3.** `question_interpretation.mi_recognition_diagnosis` refused to run —
`TRAKT_RUNTIME_MODE` resolved to `production`, so `trakt_core.policy` would
refuse both books as synthetic fixtures and every shape would rate ABSENT. A
clean-looking zero would have been indistinguishable from a passing surface.

Reported as **not measured**. That stays the convention: the absence of a
finding is evidence about the instrument before it is evidence about the
product.

---

## OPEN · A live wrong answer on the shipped path, independent of this split

**Not caused by this programme, and not fixed by it.** Logged here so it is not
absorbed into the interpretation work.

With an `ANTHROPIC_API_KEY` present and `MI_AGENT_LLM_PARSER` unset:

| question | outcome | repeats |
|---|---|---|
| `What changed?` | refused | 5 / 5 |
| `Show me the trend.` | **ANSWERED** | 5 / 5 |
| `Compare us with the market.` | **ANSWERED** | 5 / 5 |

*"Compare us with the market."* is answered with `640 loans · Current
Outstanding Balance: £172.1MM`. **There is no market data in this platform.** A
whole-book figure is returned to a question asking for a comparison against
something the estate has never held, and nothing in the answer says so.

All three refuse on the deterministic arm and the frozen CFO bank has expected
`REFUSE` for all three since it was written. The mechanism is that the free-form
arm supplies the missing element itself, so no governed default is ever
recorded and the guards that fire on a recorded default never see one — the same
mechanism the Opus acceptance run walked through.

**Owner: separate work.** Recorded every run by
`migration_phase0/must_refuse_both_arms.py`, which exits non-zero only if the
DETERMINISTIC arm stops refusing and prints the LLM arm in full regardless.
Failing the instrument on a known-open finding would only teach the estate to
stop running it.

---

## OPEN · The receipt stamps a threshold applied/lost the wrong way round

**Found while measuring the threshold kind. Not caused by it, not fixed by it.**

*"How much outstanding balance do we have where borrower age exceeds 75 and LTV
is over 40%?"* (Q02B) publishes:

```
served facets : ('threshold', 'LTV over 75', 'applied')
                ('threshold', 'LTV over 40', 'lost')
spec filters  : ('current_loan_to_value',)
```

Both labels are wrong — `execution_receipt._detect_thresholds` does not resolve
the field, so the borrower-age threshold is labelled `LTV` — and the two
statuses are **inverted against the contract**: the facet stamped *applied*
names a predicate the spec does not carry, and the facet stamped *lost* names
one it does.

The consequence for anything reading the receipt is that a threshold the
contract holds reads as lost and one it does not hold reads as satisfied. It is
why Q02B was classified as a threshold loss and is not one.

**Owner: separate work.** Recorded in
`migration_phase0/MI_THRESHOLD_KIND_RESULT.json` under
`prediction_B.why_the_third_did_not_land`.

---

## OPEN · A registry gap the estate reports to the reader as a data gap

**Found in Stage 4. Three questions, one registry entry.**

*"Show a table of balance by LTV bucket and interest-rate bucket."* (Q13A, and
its two siblings Q13B and Q13C) is answered with:

> 'interest rate bucket' is not available in this dataset. This book does not
> report it, so the question cannot be answered from the current data (no value
> was fabricated).

**The book reports it.** `interest_rate_bucket` is a fully populated column on
the acceptance tape — 640 of 640 rows, five bands: `4-5%`, `5-6%`, `6-7%`,
`7-8%`, `>=8%`. What is missing is the **registry declaration**: the field is
not in `semantics["fields"]`, so `requested_dimension_terms` resolves nothing
for any spelling of the term, no owner claims it, and the concept vocabulary
cannot offer it either.

The refusal is honest about having fabricated nothing and **false about the
data**. A reader is told the book does not hold something it holds.

This was classified as a capability gap in the frozen 75-bank grades. It is
not. Every other bucket axis on this tape — `ltv_bucket`, `age_bucket`,
`ticket_bucket` — is declared and works.

**Owner: separate work, and the cheapest remedy measured in Stage 4** — three
questions for one registry entry and no code, against the threshold kind's
three questions for a new concept kind.
