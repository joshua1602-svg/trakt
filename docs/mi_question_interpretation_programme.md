# Question interpretation — programme brief (living)

The standing conditions and backlog for the question-interpretation contract,
kept in the repository so an amendment has somewhere to live. Base `4e051f3`;
release candidate `28ece25`, unchanged and shippable throughout.

## Standing conditions

1. **Every measurement runs both surfaces, deterministic arm, at every stage.**
   **AMENDED — see below.**
2. One change per commit, each with its own before/after. Stop at the first
   unattributable movement.
3. Confirm the base commit before reporting anything.
4. No instrument ships without a test proving it can fail.
5. Report a flat result as flat. Do not re-author an instrument after seeing
   its result.
6. Do not weaken a rule to make a change fit.

---

## Amendment 1 — standing condition 1, the measurement arm

**Adopted.** Standing condition 1 previously required the calibration bank and
the 44-variation robustness bank without naming an arm. It now requires the
**deterministic arm of both surfaces at every stage.**

### The reason

The LLM arm **self-disagrees on 6–10% of cells** — larger than any change this
programme will make. An instrument whose own noise floor exceeds the effect
size cannot attribute a movement to a cause, which is the entire purpose of
running it at each stage. The deterministic arm has always been the
attribution-grade instrument.

This is not a workaround for the missing API key. It is the correct instrument
for per-stage attribution, and it happens also to be the one that runs here.

### What this changes in practice

| | Before | After |
|---|---|---|
| Per-stage gate | ambiguous | deterministic arm, both surfaces, both books |
| Robustness bank | 752 runs, LLM on | 88 runs, LLM off, reproducible run-for-run |
| Repeat variance | measured | not applicable — a deterministic arm cannot vary |

### What is reserved, not discarded

**One LLM-arm run is reserved for the final merge decision**, if a key becomes
available. Merging is a separate decision taken on the full result, and that is
the point at which the noisier, more realistic arm is the right instrument.

### What this amendment does not claim

A regression that manifests **only** through the LLM parser will not be caught
by per-stage measurement. That is accepted, deliberately, in exchange for
attribution. The reserved merge-decision run is the control for it.

### Numbers that must not be compared across arms

The recorded **91.0% correct/disclosed** and the **160-run regression** are
LLM-arm measurements. The deterministic arm's **34 of 44 correct/disclosed** is
a different arm on a different denominator. Neither figure should ever be
quoted against the other.

### Deterministic baseline, as at `1863b1b`

| | alderbridge | kestrelmoor |
|---|---:|---:|
| `CORRECT` | 32 | 32 |
| `SAFE_REFUSAL` | 10 | 10 |
| `CORRECT_WITH_DISCLOSED_LIMITATION` | 2 | 2 |
| unsafe outcomes | 0 | 0 |

Same verdict on **44 / 44** variations across both books. Seasoning family
(Q1 4, Q7 4, Q8 12) — **20 / 20 `CORRECT`**, per book, by name.

Reproduce: `python -m question_interpretation.run_robustness_deterministic --all-books`

---

## Backlog

### B1 — Route the categorical filter regex through the profiled allowlist

**Scoped, not scheduled. Not part of this programme.**

`llm_query_parser._CATEGORICAL_FILTER_RE` (`llm_query_parser.py:1820`) validates
an extracted geography value against two **denylists**,
`_CATEGORICAL_STOPWORDS` and `_NON_PLACE_TERMS`, accepting anything not listed.
`execution_receipt.geographic_values` already holds the **allowlist** — the
values the loaded book actually contains, 11 tokens on the alderbridge tape.

**The change:** route the regex's operand through the profiled allowlist instead
of the denylists.

**Why it is worth doing once rather than patching again:** the denylist has
already been patched twice, each time after a defect — *"when is it expected to
complete"* binding a geography called **Complete**, and *"for joint borrowers"*
binding a borrower predicate to the geography field. A denylist cannot be
completed. Routing through the allowlist retires the fabricated-geography class
permanently rather than removing its third instance.

**Why it is not urgent:** the surviving cases **fail closed**. *how much is in
the good book* binds `geographic_region_obligor='Good'`, matches zero rows, and
returns *"No loans in this book match that filter … I have not returned a
whole-book figure in its place."* Wrong reason, correct refusal, no wrong
number.

Fuller analysis, including what tape normalisation would additionally require:
`docs/mi_value_domain_prerequisite.md`.

### B2 — `answer_type.asked` disagrees with the parser on 46 questions

**Recorded, not to be fixed in Stage 2.** Analysis and user-visibility finding:
`docs/mi_question_interpretation_stage2_readiness.md`.
