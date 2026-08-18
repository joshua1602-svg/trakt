# TRAKT MI Query Agent — P1H Safe Intent Recovery

**Branch** `claude/mi-query-agent-review-n8d33r` · **Base** `7fa9ddf` (P1G, pushed)
**Fixture** `demo_platform` / alderbridge — 11,035 loans, as at 30 June 2026
**Production code changed: none.** This phase is a measurement and a finding.

---

## 1. Executive result

**P1H recovered no answerability, because the phenomenon it targets does not
occur at a measurable rate.**

The phase was scoped to recover questions where *"the deterministic semantic
parser confidently identifies a requested concept that the LLM parser omits"*.
Measured against genuine live-model parses, over **80 question runs** spanning
the 40-question bank, the P1E bank and the curated P1H bank:

| Slot | OMISSION | AGREEMENT | CONFLICT | not requested |
|---|---:|---:|---:|---:|
| measure | **2** | 56 | 7 | 15 |
| dimension | **1** | 13 | 3 | 62 |
| filter | **0** | 14 | 0 | 65 |

And on the **curated P1H bank specifically** — the 14 questions chosen because
the LLM "historically omits or partially emits intent":

```
measure omissions:   0
dimension omissions: 0
filter omissions:    0
```

**Of the 30 refusals in the 40-question bank, `PARSER_OMISSION` accounts for
zero.**

A carry-forward mechanism built to this specification would ship having never
usefully fired. It was not built. The evidence is below.

---

## 2. Root cause: the LLM does not omit, it substitutes

The premise was that structured output is *incomplete*. It is not. The model
reliably emits a measure — it sometimes emits the **wrong** one, and it
sometimes **adds** one nobody asked for.

### The two omissions that did occur are both unrecoverable

**"Show me balance by region by borrower type"** — the model omitted
`collateral_geography`. Carrying it forward changes nothing: the question
refuses because **borrower type is not on the tape**, and it would still refuse.

**"Are older borrowers taking bigger loans relative to their property value?"**
— the model omitted `current_valuation_amount`. This question **already answers
correctly** (a scatter of age against LTV, which *is* loan relative to property
value). Carrying the deterministic measure forward would have changed a correct
answer into a different one.

So: 2 omissions in 80, one inert and one actively harmful to recover.

### Where the deterministic parser is the wrong authority

> **"What is the balance below 75% LTV?"**
> deterministic: `current_loan_to_value` · LLM: `current_outstanding_balance`

The deterministic parser reads the *filter subject* as the measure. The **LLM is
correct**, and the question answers correctly today. Under a carry-forward
regime that treated deterministic intent as recoverable, this would have been
**broken by P1H**.

This is the strongest argument against the mechanism, and it is independent of
the B25 ruling: deterministic intent is not uniformly higher quality, so
"preserve it when the LLM disagrees" cannot be assumed safe.

### The real blocker in this bank is OVER-emission

> **"What is the average borrower age in the funded portfolio?"**
> deterministic: `youngest_borrower_age`, filters `{}`
> LLM: `youngest_borrower_age`, filters **`{"funded_status": "Funded"}`**
> → *"'Funded Status' is not available in this dataset"*

Both parsers **agree** on the measure. The refusal is caused by a filter the
model **invented**: it reads "the funded portfolio" as a predicate, when the
entire governed tape *is* the funded book. Same for *"the weighted-average LTV
of the funded book"*.

This is the mirror image of the P1F defect where "the funded book" was resolved
as a *region*. Same phrase, different parser, different wrong resolution — and
carry-forward cannot address it, because nothing was omitted.

---

## 3. Merge architecture, as traced

The seam P1H would extend already exists and already encodes the omission-only
rule. No parallel combination layer is needed or was added.

```
question
  → _deterministic_parse            (always computed, free)
  → validate_mi_query(det_spec)     (always validated)
  → [zero_cost_first short-circuit]
  → LLM parse + repair loop
  → carry_specialist_intent         ← THE SEAM
      + reconcile_threshold_operators
      + carry_measure_set           (P1E)
      + reconcile_measure_aggregations
  → meta["specialist_intent_carried"]   ← provenance channel
  → routing
  → P1G semantic identity guard
  → execution
```

`carry_specialist_intent` already states the rule P1H was asked to generalise:

> *"Only fields the LLM left unset are filled, so a spec that genuinely
> expresses one of these keeps its own value."*

and already documents why analytical fields are excluded:

> *"These are INTENT markers, never data. Carrying them forward cannot change
> which rows or measures the LLM asked for; it only preserves the routing
> decision the deterministic parser already made."*

That exclusion was a deliberate design decision, and the measurement above
vindicates it.

### Two facts about the confidence signal

**`parser_confidence` is the wrong gate.** It measures *dimension* confidence,
and reads **`low`** for precisely the questions P1H targets:

| Question | `parser_confidence` | det measure |
|---|---|---|
| borrower age, Direct vs Acquired (B25) | **low** | `youngest_borrower_age` |
| average borrower age | **low** | `youngest_borrower_age` |
| weighted-average LTV | **low** | `current_loan_to_value` |
| current total exposure | **low** | `current_outstanding_balance` |
| show balance by region | high | — |

Gating on `confidence == "high"`, as the brief's confidence requirement implies,
would have recovered **nothing**.

**The real explicitness signal exists but is discarded.** `_detect_metric`
returns `matched_terms` — the registry-synonym and curated-grammar spans that
produced the measure, which is exactly the brief's criterion ("exact registry
synonym matches; explicit named fields"). It is computed at nine call sites and
recorded in `det_meta` at none of them. Any future carry-forward work needs this
plumbed first; that is a prerequisite, not the feature.

---

## 4. B25, under your ruling

**Ruling applied: LLM-emitted balance/count is a conflict, not a recoverable
omission.**

The measurement confirms the classification is correct on the facts:

| | |
|---|---|
| deterministic | `youngest_borrower_age` |
| LLM (every observed run) | `current_outstanding_balance` [, `loan_count`] |
| classification | **CONFLICT** |

Across the 20 live runs recorded in P1G and the classification runs here, the
model emitted a measure **every time**. It never omitted. So carry-forward would
not fire on B25 under any rule that respects the conflict boundary, and B25
retains its P1G safe refusal (10/10).

Independent truth recalculated from the fixture, matching your figures exactly:

```
Direct     72.187763
Acquired   69.957022
Difference  2.230741
```

The governed comparison exists and reconciles. It is reachable deterministically
and unreachable through the model's own parse. That gap is real — it is simply
not an *omission* gap, so P1H's mechanism is the wrong instrument for it.

---

## 5. Remaining refusal classification

All 30 refusals in the unchanged 40-question bank (production LLM path):

| Class | Count | Questions |
|---|---:|---|
| `MISSING_GOVERNED_CONCEPT` | **15** | A1, A2b, A2c, A2d, A3, A4, A5, A9, B03, B09, B13, B16, B20, B24, B28 |
| `MISSING_ANALYTIC_CAPABILITY` | **7** | A7, B05, B07, B14, B15, B19, B26 |
| `MISSING_DATA` | **5** | B02, B04, B12, B18, B27 |
| `SEMANTIC_CONFLICT` | **3** | B10, B17, B25 |
| `PARSER_OMISSION` | **0** | — |
| `AMBIGUOUS_PRODUCT_SEMANTIC` | **0** | — |

Two spot-checks confirming the classification rather than trusting the keyword
sort:

* **B07** *"How much headroom before the London concentration limit binds?"* —
  the parser identifies London correctly; the risk-limits route **enumerates
  every limit and cannot scope to a region**. Capability, not parser.
* **B02** *"Which segments are driving balance growth this quarter?"* — the tape
  carries three monthly snapshots, so a quarter comparison spans 2 months. Data,
  not parser.

**Half the remaining gap is fields this book does not carry** — borrower type,
product type, broker, vintage, NNEG, equity. No parser work reaches those.

---

## 6. Safety

No production code was changed, so no safety property moved. The P1G posture
stands as pushed at `7fa9ddf`:

```
INCORRECT_SUCCESSFUL   = 0
SILENT_SEMANTIC_ERROR  = 0
HARD_FAILURE           = 0
carry-forward count    = 0   (mechanism not built)
conflicts observed     = 10  (7 measure, 3 dimension)
conflicts safely handled = 10
```

Of the 10 observed conflicts, 8 refuse safely, 1 answers correctly because the
**LLM** was right (`balance below 75% LTV`), and 1 answers correctly on a route
where the disagreement was immaterial. None produced a wrong answer.

---

## 7. Why no code was written

Building the mechanism as specified would have meant shipping:

* a carry-forward path that fires on 2 of 80 parses, both of which must be
  suppressed anyway;
* tests asserting recovery of omissions that have to be **injected**, because
  the model does not produce them;
* a new deterministic-precedence rule that the data shows would break at least
  one currently-correct question.

That is dead code with a test suite that proves only that the test suite works.
The finding — measured, at scale, on the genuine model — is the deliverable.

---

## 8. Recommended next increment

**Governed scope-phrase neutralisation.**

The single recoverable class this phase actually found is **over-emission of a
filter on a phrase that describes the dataset rather than a subset of it**:
"the funded portfolio", "the funded book". The model emits
`funded_status: "Funded"`; the column does not exist; the question refuses.

The deterministic parser already has the concept — P1F added funding-state words
to `_NON_PLACE_TERMS` so "the funded book" is not read as a region. The
equivalent governed statement is missing on the filter axis: *a phrase naming
the dataset's own scope is not a predicate over it*.

Narrow, testable, and it addresses observed refusals rather than hypothesised
ones. It needs care — dropping a filter silently broadens a population, which is
the P0 cardinal sin — so the safe form is a governed **scope-phrase registry**,
not a filter-dropping heuristic.

Recommended only. Not implemented.

---

## 9. Also observed, not fixed (per scope)

* `median` → `sum` (strict xfail, untouched)
* `MAX_MEASURES` wording, B23 receipt, B08 completeness — reported in P1F/P1G,
  untouched
* The catalogue still hides 99 governed synonyms; P1H explicitly does not
  expand vocabulary

---

## 10. Git

**Branch** `claude/mi-query-agent-review-n8d33r` · base `7fa9ddf`

| File | Change |
|---|---|
| `due_diligence/MI_QUERY_AGENT_P1H_SAFE_INTENT_RECOVERY.md` | **new** — this report |

**No production code, no test code, no configuration changed.**

---

## Verdict

P1H's success criteria include *"B25 correctness improves materially under
repeated live runs"* (6) and *"answerability improves where parser omission was
genuinely the cause"* (13). Under your conflict ruling B25 cannot improve
through this mechanism, and parser omission is never the cause: it accounts for
0 of 30 refusals, and 0 of 14 on the bank curated to expose it.

The objective is not achievable as scoped, because the defect class it targets
does not exist at a material rate. Reporting that is more useful than shipping a
mechanism that never fires — and the phase did establish where the remaining
breadth actually lives, which is §5 and §8.

Nothing regressed. Nothing was weakened. Nothing was recovered.

`P1H SAFE INTENT RECOVERY: FAIL`
