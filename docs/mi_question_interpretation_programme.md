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
7. **Every stage diffs answer TEXT, not only grades.** See below.

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

## Standing condition 7 — diff the answer text, not only the grades

**Adopted, and it is what covers Finding 1 for the duration of the programme.**

Every stage compares the **answer text byte for byte**, on both surfaces,
deterministic arm — not merely whether each case still grades as passing.

### Why this is the safe form

The bounded pre-Stage-2 check found four cases (`kpi_028`–`kpi_031`) whose
`expected_answer_type: mixed` is satisfied by an observed type of `count`, so a
portfolio summary that silently dropped its balance measure would **pass the
bank**. The grader cannot see that regression.

The answer diff can. A summary that lost a measure produces different text.

> **Byte-identical answer text is strictly stricter than the grader.**
> A change that passes the bank and fails the diff is a real regression the bank
> could not see. A change that fails the diff and passes the bank is stopped.

That is why option (a) — proceed, record, and fix nothing in the grading path —
is safe rather than merely convenient. It does not rely on the grader being
right about `mixed`; it relies on the text not moving.

### Why the grading path is not touched instead

`of_measure` and `_SATISFIES` are **graders**. Changing a grader mid-programme
invalidates every measurement taken before the change, because the before and
the after are no longer scored by the same instrument. A moved grader is worse
than a weak one: a weak grader has a known blind spot, a moved grader has no
comparable history.

---

## Amendment 2 — join sequencing, Stage 2 vs Stage 3

**Adopted.** The filter join is built in halves.

**Stage 2 emits the facet half.** `_detect_thresholds` and
`_detect_geographic_scope` already compute `match.start()` / `match.end()` and
discard them; recording those into the object is **additive** and cannot move an
answer. Parser-side claims are recorded as **spanless**, and the join is reported
as **half-built, with the missing half named**.

**Stage 3 supplies the parser half**, when `_parse_filters` is converted as a
consumer. `_parse_filters` rewrites the question as it consumes clauses
(`work_q = work_q[:bm.start()] + " " + work_q[bm.end():]`), so sound spans need
either removing the rewrite or maintaining an offset map through it. **That
choice is made then, on measurement rather than preference.**

**Why not in Stage 2:** Stage 2's guarantee is byte-identical answers. Changing
how `_parse_filters` rewrites the question forfeits that guarantee, and a stage
whose acceptance is "nothing moved" cannot also be the stage that moves
something.

---

## Amendment 3 — the principle behind removing `coverage` and keeping `CONFIGURED`

**Adopted.** *Remove things that can be wrong; keep empty things that can only be
unused.*

An unused **operation type** can still misclassify later — it is a value
something may be assigned. An unfilled **slot state** is inert: it describes a
condition nothing currently reports. That distinction, not the presence of a
rationale, is what justified removing `coverage` and keeping `CONFIGURED` marked
unsupplied.

**The contract's rationale for the configured-target sense is not evidence.**
The wording (*on target*, *versus plan*, *versus budget*) appears in **0 of 690**
real-surface questions, which contradicts it. The slot is retained because it is
inert, not because the corpus supports it.

**Review rule:** if the configured-target wording does not appear in the
client's real questions **within the first month**, the slot is removed.

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

**Recorded, not to be fixed in Stage 2.** No user-visible difference: `asked()`
is on no production path. Analysis:
`docs/mi_question_interpretation_stage2_readiness.md`.

### B3 — `of_measure` cannot distinguish one measure from several

**Open, blocking nothing, and it weakens `mixed` as an acceptance type.**
`of_measure` types an answer from a single `metric` + `aggregation`. A portfolio
summary carries `metric=None, aggregation='count'` and types as `count`, which
`_SATISFIES[MIXED]` accepts — so four calibration cases declaring `mixed` would
pass identically if the answer lost every measure but the count.

Found by the bounded pre-Stage-2 check:
`docs/mi_answer_type_expectation_check.md`.

**The right shape of the fix, recorded while it is fresh.**

The defect is *four cases asserting a property nothing verifies*. So the fix is
to **verify the property**: assert the required measures on `kpi_028`,
`kpi_029`, `kpi_030` and `kpi_031` directly — that the result carries both
`loan_count` and a balance measure — rather than inferring it from a type.

That is a **test-side addition**. No production change, no effect on the
baseline, and it makes the four cases detect the regression they describe.

**Two fixes that look adjacent and are wrong:**

* **Changing `of_measure`** so it reads the result's measure set rather than a
  single spec slot. It is a grader. Changing it mid-programme means before and
  after are scored by different instruments, and every measurement taken
  earlier stops being comparable.
* **Narrowing `_SATISFIES[MIXED]`** so `count` no longer satisfies `mixed`. Same
  objection, and it would likely fail those four cases immediately — which is
  Finding 1 becoming visible rather than being fixed. The property would still
  be unverified; only the symptom would move.

The distinction is that asserting the measures **adds a check**, while both
alternatives **move an existing one**.

### Recorded as working — the derivation cross-check

56 of 252 stored `expected_answer_type` values differ from `answer_type.asked()`
today, and **none of them is drift**. 33 of the 35 `currency`-versus-`any` cases
carry an `expected_metric` that justifies `currency` — 27
`current_outstanding_balance`, 5 `current_valuation_amount`, 1
`original_principal_balance` — and the 21 `none` cases are authored from
`expected_status` rather than from the wording.

That is `derive_answer_type.py`'s documented cross-check behaving exactly as
designed: *"the question's own wording decides, cross-checked against the
declared expected_metric"*. Recorded as a control that works, not dropped
quietly — a mechanism only known to be sound if someone has checked it and said
so.

### Standing rule — do not regenerate the calibration bank

All 252 `expected_answer_type` values were derived from `answer_type.asked()`.
Regenerating the bank during this programme would rewrite those expectations
from a classifier the programme is changing, and a bank that moves with the code
it grades has stopped being a control.
