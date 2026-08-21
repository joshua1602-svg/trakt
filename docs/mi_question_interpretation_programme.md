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

## Note 1 — a governed-config dependency here is correct

**For the record, to prevent a false conflict later.**

`_analytical_population_satisfies` now reads the governed seasoning
configuration to resolve a lending window before comparing. That does **not**
contradict the reason conversion 4 was left partial.

| | |
|---|---|
| `question_interpretation.lexical` | the LEXICAL owner. Reads question text and nothing else — no registry, no frame, no spec, no config. `requested_span` stayed out of it because resolving vague recency against `lending_windows.recent_max_months` is a config decision. |
| `execution_receipt` | the SEMANTIC layer, downstream. Resolving a governed concept is exactly its job. |

The principle protects the lexical owner's domain-blindness. It is not a general
prohibition on configuration, and applying it downstream would be a
misreading — a semantic layer that cannot consult the governed model cannot do
its job at all.

## Note 2 — generate the coverage where the corpus cannot exercise a construct

**Adopted as the standing pattern.**

Removing the `_parse_filters` rewrite could not be proved by the corpus: exactly
**one** of 690 real-surface questions contains `between`, the only construct the
rewrite existed for. So the old algorithm was reproduced verbatim against the
same helpers and compared across a generated set built to hit every shape the
construct appears in — **11,474 questions, 22,948 comparisons, 0 mismatches**.

The rule this sets:

> Where the corpus cannot exercise a construct, **generate the coverage rather
> than declare it untested.**

"The corpus does not cover this" is a statement about the corpus, not evidence
about the change. A construct rare in the corpus is not rare in the field, and
the one case that exists cannot distinguish a correct rewrite from a lucky one.
This applies to any conversion touching a path the banks barely reach.

---

## Standing rule — an instrument tends to carry the defect it was built to find

Not an incident. A pattern, recorded because it has now happened often enough
that it must be designed against rather than caught by luck. Every instance was
found by chance or by a late cross-check, and each one would have shipped a
false clean result.

1. The Phase 1 measurements were taken 136 commits off the intended base. Every
   score was internally consistent and every one was void.
2. The calibration bank graded against `build_fixture` rather than the real tape
   — the surface built to prove the tape's answers, not reading the tape.
3. `answer_diff` keyed on `(intent, variation)` and silently dropped 16 of 88
   robustness answers, all of them the seasoning family: the differ built to
   catch seasoning movement could not see the seasoning questions.
4. Two of the 14 mutations did not reproduce the defect they named, so the
   mutation suite that proves the instruments can fail contained instruments
   that could not fail.
5. My own role-split test asserted the Stage-1 role value rather than the value
   production gives.
6. The source-check test for the removed rewrite matched a comment describing
   the rewrite rather than executable code.
7. The B5 scanner watched detection-time facets while the split it guards
   happens at reconcile — it would have missed precisely the facets it exists to
   guard.
8. `run_robustness_deterministic --all-books` re-invokes itself per book and
   forwarded only `--book`, so a variant run measured the default twice and
   reported it as the variant.
9. `answer_diff` had the same defect on `--only-book`, so a variant run moved
   the 252 in-process calibration records and left the 88 subprocess robustness
   records on the default — which reads as "the variant only affects the
   calibration bank", a conclusion about the product drawn from a defect in the
   instrument.

Three properties separate the instances that were caught from the ones that
nearly were not, and they are the rule:

* **An instrument must be able to produce the failure it rules out.** Every
  instrument ships with a case proving it fails. Where the corpus cannot
  exercise the construct, generate it (Note 2).
* **An instrument must be read at the point the code it measures runs.** Both
  the B5 scanner and the two forwarding defects were instruments reading a
  different moment, or a different process, from the one under test.
* **Every argument that changes what is measured must reach every process that
  measures.** A runner that fans out to subprocesses and forwards a subset
  cannot report that it did not measure what it was asked to.

Corollary, from instance 9: when a measurement splits cleanly along the seam of
the instrument's own plumbing — one surface moves, the other does not, and the
seam is a process boundary or a call site rather than anything in the product —
suspect the instrument first.

Instances 10 and 11, both from the stamping coverage inventory and both caught
by its own `--self-test` rather than by reading its output: a one-size evidence
bundle reported eleven false holes, and a malformed analytical envelope reported
a false hole on the very cell that distinguishes the two reconcilers. The rule
they add: **an instrument that classifies must be able to produce every class it
reports.** A hole-finder that can only produce holes has not found any.

**Companion rule, from instances 10 and 11: an instrument that CLASSIFIES must
be able to produce every class it reports.** A hole-finder that can only produce
holes has found none. The coverage matrix reports four cell values and its
`--self-test` asserts each is producible, including the one it must not confuse
with a defect: a route that correctly did not do the thing asked of it.

**And its converse, from closing the hole: an instrument must not be anchored to
the defect it was built to find.** The first self-test asserted
`(point-in-time)/row_population` reads as a hole. That was true when written and
started failing the moment the hole was closed — the right outcome from the
wrong assertion, because a test tied to a bug stops asserting anything once the
bug is fixed and silently becomes a test of nothing. Re-anchored on a DESIGNED
hole, which will still be there, plus a separate assertion that the fixed one
stays fixed.

Instance 12 is of a different and worse kind, from the same inventory: the two
standing measurement surfaces were BOTH clean throughout a live shipped
regression — three ordinary questions about the front and back book refusing on
the shipped tape. Neither surface was defective; between them they simply do not
exercise the point-in-time population path. Recorded as backlog B6. The rule:
a clean surface is evidence about the surface's coverage before it is evidence
about the product.

## Recorded as implemented and inert — the unresolved-role clarification

Stated plainly so nobody later quotes the three-variant measurement as evidence
of a behavioural gain, because it is not.

**Implemented** (`1f8078d`). A dimension no source gave a role to becomes a
question rather than an answer over a set the reader may not have asked for. The
principle is right: a refusal and a clarification both decline to answer, and
only one hands the reader the next move.

**Currently inert on these corpora.** 343 of 343 answers identical; robustness
44/44 on both books; zero new test failures. It classifies 10 facets across 9
questions on the real tape and the clarification wins on five of them through
the point-in-time workflow — but on the shipped service path all five are
claimed by `risk_limits` or an evolution route and never reach
`reconcile_facets`.

**The measurement that chose it no longer applies.** §2.2 of
`docs/mi_stage4_unresolved_role_variants.md` recorded clarify converting three
refusals into three questions. Two rules added while applying it — a
clarification is only worth asking when answering it changes something, and a
field the book cannot express has no role worth settling — remove all three,
because all three are `borrower_type` on a tape that does not carry it.

Anyone citing this work should cite it as: correct, shipped, and not yet
observable on any available book. Not as an improvement measured on the
surfaces.

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

### B0 — wire the parser's filter spans onto the spec, completing the join

**Ready, not started.** `_parse_filters` now returns `{field_key: (start, end)}`
through an optional `spans` sink, but nothing carries it onto `MIQuerySpec`, so
the object still reports the join as half-built.

The obstacle I expected is not there. `_mask_spans` — the other place the parser
appears to rewrite the question — **blanks characters in place rather than
deleting them**, and its docstring says so: *"Blanking rather than deleting keeps
every other offset valid"*. So offsets taken from the masked remainder are valid
offsets into the original question, and all four `_parse_filters` call sites can
supply sound spans.

What remains is mechanical rather than risky: an additive `filter_spans` field
on `MIQuerySpec`, excluded from `referenced_fields()` and validation as the
other non-semantic fields are, and a sink threaded at four call sites. It is its
own commit with its own before/after.

### B4 — `mi_agent/interpreter/deterministic.interpret` duplicates a serving concern

**Recorded, not now.** The package is a development smoke tool — imported only
by `scripts/mi_nlq_dev_smoke.py`, `scripts/phase8e_live_anthropic_smoke.py` and
its own modules — but it carries a second whole parser for a concern the serving
path also implements.

Dev-only code that duplicates a serving-path concern will drift, and the drift
is invisible because nothing measures it. A future reader finding two parsers
has no way to know which one ships. Either it consumes the same owner as the
serving path, or it is deleted, or it carries a header saying plainly that it is
not the parser and must not be read as one.

### B5 — the literal population comparison is permissive when a label omits its field

**Found during Stage 4, pre-existing, recorded rather than fixed.**

`_analytical_population_satisfies` derives the value it wants by splitting the
facet's label on its field name. Where the label does not contain the field
name, the value comes out empty and the check accepts **any** predicate naming
that field — including the wrong population. A facet for *front book* is
accepted against a declared `seasoning_segment = Back Book`.

Verified present before the Stage 4 change by stashing it. Not reachable with
the labels the receipt builds today, which embed the field and the value, and
the governed comparison added in Stage 4 is stricter rather than looser.

Not fixed here because it would change acceptance for populations outside the
seasoning family, and the pre-registered prediction for Stage 4 says to report
such a movement rather than absorb it.

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
