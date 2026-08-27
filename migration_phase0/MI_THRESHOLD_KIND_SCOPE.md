# The threshold kind — scope, and the pre-registration for measuring it

Base `aeeaed5` (Stage 3), tree clean. Written **before** the threshold kind
exists and **before** any bank was re-run.

---

## The scoping question, answered: NO NEW BINDER IS NEEDED

A numeric threshold on a governed field can be expressed as a proposal that
**existing owners bind**. Measured, not argued.

### The four owners, all of which already ship

| step | owner | what it already does |
|---|---|---|
| comparator phrase → operator | `question_interpretation.lexical.COMPARATOR_PHRASES` | 49 phrases, longest-first, **THE** list. Its own docstring: *"one fact about English, two consumers"* |
| field term → governed field key | `llm_query_parser._detect_metric`, `_explicit_dimensions` | the same term→field owners Stage 2's binder already uses |
| (field, op, value) → `Predicate` | `mi_agent.population.predicate_of` | a pure normaliser over the three shapes `spec.filters` uses; reads no question |
| `Predicate` → rows | `population.apply_population` → `mi_query_executor.governed_predicate_mask` | **the** one governed meaning of a predicate — percent rescaling, operator aliases, value-domain and field resolution |

### The measurement

Six thresholds bound from `(term, comparator, value)` through those owners only,
executed against the acceptance tape, and compared against what the
deterministic parser produces for the equivalent question:

```
youngest_borrower_age gt 75      proposal 297 rows   deterministic 297 rows   MATCH
current_loan_to_value gt 50      proposal 144 rows   deterministic 144 rows   MATCH
current_loan_to_value gt 40      proposal 272 rows   deterministic 272 rows   MATCH
youngest_borrower_age gt 55      proposal 640 rows   deterministic 640 rows   MATCH
current_interest_rate gt 7       proposal 203 rows   deterministic 203 rows   MATCH
current_outstanding_balance gt 300000  proposal 262 rows  deterministic 262 rows  MATCH
```

**6 of 6, exactly.** Scale resolution is in `governed_predicate_mask`, at
execution, so a proposal carrying `50` for LTV is executed correctly whether the
tape stores a ratio or a percentage — the defect that made a route narrow every
snapshot to nothing when it was handed the wrong semantics dict.

### What is new, and it is not a binder

Only the **proposal shape**: three parts (`field_term`, `comparator`, `value`)
instead of one term. The module owns the assembly and nothing else, which is the
same rule Stage 2 established — *the module owns the comparison, not the
reading*. **Nothing here decides what a phrase means.**

### One caveat, stated rather than discovered later

`loan size` is on the cross-kind collision list (a `dimension` term and a
`measure` term). For a threshold, the measure reading is the one wanted, and the
threshold binder therefore asks the measure owner first. That is a precedence
decision this proposal shape makes, and it is recorded here so it is visible
rather than buried in a call order.

---

## A blocker found while scoping, and it is in the instrument

**The completeness check cannot register a model-filled threshold.** Measured:

```
stated: facet:threshold "LTV over 40";  contract carries no filter
   before merge  -> LOST
   merge fills   -> row_predicates[current_loan_to_value] = 40.0
   after merge   -> STILL LOST
```

The check decides a stated facet is carried with
`contract.facet_applied(kind, label)` — it matches the **served facet list** by
`(kind, label)` — and `merged_contract` changes only `filters`. So reach on
threshold losses is pinned at zero **whatever the model proposes**, and
measuring it without fixing this would measure the check's plumbing.

### The fix, and why this shape

The stated threshold facet carries a **number and nothing else reliable**:
`_detect_thresholds` does not resolve the field, which is why the facet for
*"borrowers over 55"* is labelled `"LTV over 55"`. So the only fact both sides
hold is the bound.

`merged_contract` will therefore mark a stated threshold applied **iff the merge
filled a predicate whose value equals the number in that facet's label**. It
fails closed: no match, no mark. It requires the stated concepts to be passed
in, which is one more argument and no new reader.

**Acceptance for this change on its own:** the frozen calibration must hold
**unchanged** — 0 false positives across all 157 delivering questions, type-(c)
19/20 on deterministic contracts. If it moves, the change is wrong and I report
that instead.

---

## Pre-registered predictions

### A · the 8 bucket-predicate fills

Today, 8 correct-today questions receive a bucket predicate the reader never
asked for: **Q02A, Q02C, Q04B, CFO39, CFO40, CFO41, CFO42, CFO45.**

**Predicted: at least 6 of the 8 stop receiving one.**

Stated risk, before the run: the bucket values are **genuine governed values** —
`ltv_bucket` really does carry `40-50%` — so they stay in the `category_value`
vocabulary. The threshold kind **competes** with them; it does not remove them.
If the model still prefers buckets, prediction A fails, and that is a finding
about the prompt. **It would not be a licence to delete the buckets from the
vocabulary** — they are legitimate concepts and some questions genuinely name
them.

### B · reach

Reach today is **4 of 20** (Q03A, Q05C, Q07B, Q16B). From Stage 1's own lost
lists, the type-(c) failures whose lost concept is a **threshold** are
**Q01C, Q02B, Q03C** — three.

**Predicted: reach rises from 4 to 7. Stated range 5–8.**

Anything above 8 means a threshold proposal is recovering a question whose loss
was not a threshold, which I would need to explain rather than bank.

### C · nothing else moves

- the frozen calibration holds unchanged;
- 278-module regression: 85 failing names, identical;
- the three must-refuse questions still refuse on the deterministic arm, and the
  merge fills nothing on any of them;
- **the merge is still NOT wired into serving.** No answer can move, and any
  harm predicted from the merged claim set is a prediction, exactly as in
  Stage 3.

### D · what would make me stop

- the threshold kind cannot be bound by an existing owner — **already answered:
  it can, 6 of 6**;
- the calibration moves when the `merged_contract` threshold rule is added;
- reach falls;
- a bucket fill appears on a question that has none today.
