# TRAKT MI Query Agent — P1G Semantic Identity (Cohort + Measure)

**Branch** `claude/mi-query-agent-review-n8d33r` · **Base** `1905b54` (P1E, accepted)
**Fixture** `demo_platform` / alderbridge — 11,035 loans, £1,964,886,258.21, as at 30 June 2026
**40-question bank** sha256 `e0fc0b61…3194` — **unmodified**

---

## 1. What P1G is

P1F delivered exposure semantics and B21 but could not pass the safety gate: a
vintage question was being answered by sourcing channel. That was not an
exposure defect. It was a hole in the semantic guard, and it had a twin.

P0 already held this chain for measures-in-a-set, filters and dimensions:

```
requested concept -> resolved governed field -> executed calculation -> receipt
```

It did **not** hold it for **cohorts** or for a **singly-named measure**. In both
cases the guard proved that *something of the right kind* had executed, never
that it was *the thing asked for*.

| | The guard proved | It did not prove |
|---|---|---|
| **Cohort** | two books were compared | *which two* |
| **Measure** | the spec's measures were compared | the *question's* measure reached the spec |

Two questions in the bank sat in those holes. Both were **pre-existing**, both
**stochastic on the LLM path**, and both had been recorded as passing by
baselines that happened to sample the lucky branch.

---

## 2. Cohort identity

### The defect

> **B04 — "Is the credit quality of new origination better or worse than the
> back book?"** → answered, grouped by **Source Portfolio Type**.

That is a question about *seasoning*. Direct-versus-acquired is how loans were
*sourced*. The arithmetic was right and the answer was to a different question.

The reconciliation was:

```python
if any(k in group_keys for k in _COHORT_DIMENSION_KEYS):
    facet.status = APPLIED
```

Any sourcing key satisfied any cohort facet, and the facet itself recorded only
"a comparison between two books" — no concept, nothing to check against.

### The invariant

`_COHORT_CONCEPTS` curates the concepts a question can name and the governed
fields that express each:

| Concept | Language | Governed fields |
|---|---|---|
| **sourcing** | direct, acquired, purchased, organic, source portfolio | `source_portfolio_type`, `source_portfolio_id`, `source_portfolio_label`, `portfolio_cohort` |
| **vintage** | new origination, back book, front book, new lending, new business, seasoned, seasoning, vintage | `vintage_year`, `origination_year`, `vintage`, `vintage_bucket`, `seasoning_bucket`, `origination_vintage` |

The facet now carries the concept; reconciliation requires the executed grouping
to express **that** concept; a concept whose fields the dataset lacks is
**UNAVAILABLE and named**, never satisfied by a different split.

Concept language is curated rather than read from field synonyms because the
registry describes *fields*, and "back book" is not a synonym of any field — it
is a way of asking about seasoning, which `vintage_year` expresses.

### Three things the build surfaced

**Detection was too narrow to matter.** *"How does new lending compare with the
seasoned book on LTV?"* raised no facet at all, so the identity check never ran
and the question was still answered by direct-versus-acquired. The facet is now
raised by *concept + comparison framing*, not only by the original phrasings.

**The routed path needed the same treatment.** The comparison route now
*declares* `cohortConcept: "sourcing"`, and a cohort route that declares nothing
is treated as proving nothing.

**A raw origination date is not a vintage dimension.** This tape carries
`origination_date`. Listing it as expressing the vintage concept would have made
the guard report "we could have and didn't" when the truth is that no governable
vintage dimension exists. The refusal would have been right for the wrong
reason. Only groupable vintage dimensions are listed, and a test pins that.

---

## 3. Measure identity

### The defect

> **B25 — "How does the direct book compare with the acquired book on **borrower
> age**?"** → *"Direct has higher observed Current Outstanding Balance than
> Acquired. Direct has higher observed Loan Count than Acquired."*

Age is never mentioned. Both figures are correct; neither is the answer.

### The hole, exactly

```python
substitution = detect_measure_substitution(
    question, route=route, metric_key=spec.get("metric"))
```

It read the spec's **singular** `metric`. A P1E measure-set spec carries
`metric=None`, so `executed` came back empty, the function returned early, and
nothing was compared against anything.

**That asymmetry is why the defect was intermittent.** When the model emitted a
singular metric the guard fired and the question refused; when it emitted a
measure set the guard was silent and the question answered wrongly. A single
sampled run could land either way — which is exactly how it passed unnoticed
through P1E's acceptance and the recorded baseline.

### The invariant

The check now reads the measure set that **actually ran**:

* routed comparisons — `comparison_measure_concepts()` over the route's own
  declared `measuresCompared`;
* the generic path — `executed_measure_concepts()` over the executor's declared
  set.

The question's named measure is reconciled against everything that executed,
rather than against one slot a modern spec no longer fills. The singular
`metric_key` still works, so pre-P1E specs are unaffected.

Nothing in the implementation knows about B25.

---

## 4. Adversarial tests

`tests/test_p0_cohort_identity.py` (30) · `tests/test_p1g_measure_identity.py` (20)

| Case | Outcome |
|---|---|
| age requested, balance + count executed | **rejected** |
| LTV requested, balance executed | **rejected** |
| neighbouring measure (rate for age) | **rejected** |
| *no* other measure set can stand in for age (parametrised sweep) | **rejected** |
| age requested, age executed | accepted |
| age among a larger executed set | accepted |
| multi-measure Direct vs Acquired | every requested measure accounted for |
| unavailable measure | refuses, names it, offers no substitute |
| question naming no measure | never refused on this basis |
| singular `metric` slot (pre-P1E spec) | still works |
| sourcing grouping vs vintage facet | **rejected** |
| geography grouping vs cohort facet | **rejected** |
| correct grouping vs its own facet | accepted |
| unavailable cohort concept | UNAVAILABLE, named |
| routed sourcing comparison vs vintage facet | **rejected** |
| routed comparison declaring no cohort | proves nothing |
| cohort concept without comparison framing | no facet raised |

The parametrised sweep matters: it proves the invariant cannot be satisfied by
patching one pairing.

---

## 5. Genuine-LLM acceptance

Live model. **174 question runs, 0 exceptions.**

### The gate probes, repeated

| Probe | Runs | Result |
|---|---|---|
| **B25** (measure identity) | **10** | **10/10 safe refusal — 0 incorrect successful** |
| B04 (cohort identity) | 5 | 5/5 safe refusal |
| "new lending vs seasoned book" | 5 | 5/5 safe refusal |
| "direct vs acquired LTV" (control) | 5 | **5/5 answered correctly** |

Before P1G, B04 was 5/5 *wrong* and B25 was wrong in roughly 1-2 runs in 5.

### Banks

| Bank | Result |
|---|---|
| Five CFO questions | **5/5** |
| P1E 26-question bank | **26/26** |
| Exposure bank (generic) | **5/5** → `current_outstanding_balance` |
| Exposure bank (explicit EAD) | **3/3** refuse, no substitution |
| B21 bank, all phrasings | **correct**, EAD variant refuses |
| 40-question bank (production) | **10/40** vs LLM baseline **9/40** |

### Adjudication of every successful answer

All ten `ok=True` answers in the production 40-bank were read individually:
A2a, A6, A8, B01, B06, B08, B11, B21, B22, B23. None reports a measure the
question did not ask for; none presents a partial as complete (B22 states
*"Not applied: postcode"* explicitly).

```
INCORRECT_SUCCESSFUL   = 0
SILENT_SEMANTIC_ERROR  = 0
HARD_FAILURE           = 0
```

---

## 6. Where B25 lands, stated precisely

The gate allows **correct or safe refusal**; the target was correct.

* **Deterministic path: CORRECT.** *"Direct has higher observed Youngest
  Borrower Age than Acquired"* — the governed comparison runs and reconciles.
* **Genuine-LLM path: SAFE REFUSAL, 10/10.** The model drops "borrower age" from
  the spec and emits balance and loan count. The guard catches it every time.

So the remaining gap is **LLM parse quality, not guard coverage**. Making the
LLM path correct needs a new precedence rule — carry a measure the deterministic
parser positively resolved when the model's spec omits it — which changes
**every** LLM parse and needs its own blast-radius analysis. Adding a global
precedence rule at the end of a phase is precisely what produced the B04
regression earlier in P1F. It is §9.

---

## 7. Regression

| | Before | After |
|---|---|---|
| Exposure bank | 5/5, 3/3 | **unchanged** |
| B21 all phrasings | correct | **unchanged** |
| Cohort identity | 10/10 refuse | **unchanged** |
| CFO questions | 5/5 | **5/5** |
| P1E bank | 26/26 | **26/26** |
| Deterministic 40-bank | 11/40 | **11/40** |
| Targeted suites | — | **2,687 passed** |

The exposure/B21 and cohort-identity implementations were not touched.

The production 40-bank moves 11 → 10 because B25 stopped answering wrongly. An
incorrect successful answer became a safe refusal: the count went down and the
book got safer.

---

## 8. Known limitations

1. **B25 is a safe refusal, not correct, on the LLM path** — §6.
2. **B23's receipt under-describes the calculation.** *"Are older borrowers
   taking bigger loans relative to their property value?"* renders the right
   scatter (age against LTV — loan relative to property value *is* LTV) but the
   receipt reads *"Calculated: Count of"*. The answer is right; the receipt is
   weak. Pre-existing.
3. **B08 answers half its question.** *"What is the run rate of new lending and
   is it accelerating?"* gives the run-rate and does not address acceleration.
   Baseline-accepted, unchanged here, and the same shape as B21 before P1F.
4. **The catalogue still hides 99 governed synonyms** across 51 core fields
   (P1F §10.2). Lifting it needs the cohort guard — which now exists — plus its
   own blast-radius run.
5. **`median` → `sum`** remains a strict xfail, untouched as instructed.
6. **`MAX_MEASURES` is not enforced on the LLM path.**

---

## 9. Recommended next increment

**Carry a measure the deterministic parser resolved when the model's spec omits
it.** B25 is the worked example: "borrower age" resolves deterministically to
`youngest_borrower_age`, the model returns balance and loan count, and the guard
turns a wrong answer into a refusal. The answer exists and reconciles exactly —
it is being discarded.

The precedence rule already exists for specialist routing intent
(`carry_specialist_intent`) and for measure sets (`carry_measure_set`). Extending
it to a *named but omitted* measure is the same shape. It changes every LLM
parse, so it wants its own phase and its own before/after run — the discipline
P1F's catalogue attempt proved is necessary.

Recommended only. Not implemented.

---

## 10. Git

**Branch** `claude/mi-query-agent-review-n8d33r`

| SHA | |
|---|---|
| `03c8c6b` | P1F: exposure semantics and B21 answering both halves |
| `cc01a8b` | Fold a list given in a scalar slot instead of crashing on it |
| `7b0654b` | Carry the exposure convention in the instruction, not by widening the catalogue |
| `1dab1c8` | Record the P1F findings and the FAIL verdict |
| `8152c46` | Prove which cohorts were compared, not merely that two were |
| `cace275` | Prove the measure asked for is the measure answered |

| File | Change |
|---|---|
| `mi_agent/execution_receipt.py` | `_COHORT_CONCEPTS`, `cohort_concepts_named`, cohort identity on both reconciliation paths; `comparison_measure_concepts`; `detect_measure_substitution` reads the executed set |
| `mi_agent_api/chat_routing.py` | comparison route declares `cohortConcept` |
| `mi_agent_api/mi_service.py` | routed substitution check reads the compared measure set |
| `mi_agent/mi_agent_workflow.py` | generic substitution check reads the executed measure set |
| `tests/test_p0_cohort_identity.py` | **new** — 30 tests |
| `tests/test_p1g_measure_identity.py` | **new** — 20 tests |

---

## Verdict

Both halves of semantic identity hold. A cohort comparison must compare the
cohorts asked for; a measure answer must report the measure asked for. Neither
can be satisfied by a neighbouring concept that happens to have executed
correctly.

Two pre-existing silent semantic errors are closed — both of which prior
acceptance runs had recorded as passing because a single sample landed on the
safe branch. Repetition, not sampling, is what established these results.

P1F's exposure semantics and B21 completion are intact and unmodified.

```
INCORRECT_SUCCESSFUL   = 0
SILENT_SEMANTIC_ERROR  = 0
HARD_FAILURE           = 0
```

`P1G SEMANTIC IDENTITY: PASS`
