# Stage 2 readiness — corrections applied, filter join analysed

| | |
|---|---|
| Base | `4e051f3`; merge-base with `claude/mi-analytical-capability-layer-vlkjfw` is `4e051f3` exactly |
| Ancestry | `4e051f3` ✓ · `28ece25` ✓ |
| Production code changed | **none** — `git diff 4e051f3..HEAD -- mi_agent mi_agent_api due_diligence` is empty |
| Schema tests | 50 passed |
| Mutation check | **14 mutations, 14 caught, 0 undetected** |
| Calibration bank | 260 passed |
| Robustness, both books | unmoved: 32 / 10 / 2, 44 of 44 agreeing, seasoning 20 / 20 |

**Counts below are from the three real surfaces only** — calibration bank (252),
ERE golden library (350), 44-variation robustness bank (88) = **690 questions**.
The generated harness is excluded from every justification: its phrasings are
machine-generated from registry names, which makes it a valid invariant check
and no evidence at all about client wording. Where a Stage 1 number differs from
one here, that is why.

---

## The five corrections, and the finding that drove each

### 1. A filter claim declares which half of the clause it carries

**Driven by:** 76 of 690 questions where **one** filter clause is read twice —
the facet layer supplying the wording with no field, the parser supplying the
field and bound with no wording.

`FilterClaim` gains `provides` (`wording` / `bound` / `field`) and `clause_id`.
Before this, a half-claim was inferred from which attributes happened to be
`None`, which is indistinguishable from *"this interpreter looked and found
nothing"*. The default is empty: a claim that has declared nothing does not read
as complete.

`clause_id` stays `None`. Stage 1 established that no interpreter emits anything
that makes the join sound, so an unjoined pair is **reported as unjoined**,
never guessed.

### 2. Span absence is explicit, and position capture is not added

**Driven by:** 170 dimension and filter claims across 690 questions have no
recoverable span, because the interpreter emitted a field key or a rendered
label rather than the words.

`Slot.has_span` makes the absence reportable. **No position capture was added** —
making an interpreter emit offsets changes that interpreter, which is Stage 3,
one consumer at a time.

### 3. `coverage` removed from the operation vocabulary

**Driven by:** produced **0 times in 690 questions**, and no interpreter supplies
it. An unsupplied member invites population by intuition. Re-add it when a
question demands it, with the question.

### 4. The configured target source is marked unsupplied, and the projection
stops inventing it

**Driven by:** the wording (*on target*, *versus plan*) appears in **0 of 690
questions**, and the only thing producing the slot was a regex the projection
owned — a reading the projection invented rather than observed. That regex is
removed.

`CONFIGURED` stays in the vocabulary because it is a stated requirement of the
contract, not because the corpus earned it, and it is listed in
`UNSUPPLIED_TARGET_SOURCES` so the difference is recorded rather than implied.

**This is the one place two corrections were treated differently on the same
evidence.** Both `coverage` and `configured` have zero corpus support. `coverage`
was removed and `configured` kept, on the grounds that the contract gives
`configured` an explicit rationale and `coverage` none. Flagging it because the
distinction is a judgement, not a measurement.

### 5. An unresolved role must say why — and the second role value was *not*
invented

**Driven by:** 55 of 690 questions name a dimension no source assigns a role to.

Stage 1 proposed splitting `unresolved` into *"no source has an opinion"* and
*"the sources disagree"*. **The corpus supports only the first.** Zero of 690
questions show a dimension that two sources put in different roles, so a
`conflicted` value would have been invented from intuition. It is deliberately
absent, and a test exists to stop it reappearing.

What was applied instead: an unresolved role carries `ROLE_UNATTRIBUTED` in
`Slot.reason`, so the distinction can be made from evidence if a conflicting
case ever appears.

---

## Stage 2's primary deliverable — how the object bridges the two halves

76 questions, one clause, two claims, no link. What the join could be:

| Arity | Questions | |
|---|---:|---|
| 1 facet half : 1 parser half | 72 | pairs trivially — but only by arity coincidence |
| 1 : 2 | 2 | *How many loans have borrower age 70+ and LTV above 50%* — the facet layer found one threshold, the parser two |
| 2 : 1 | 2 | *How many South East loans have LTV above 50%* — the facet found the geography, the parser did not |

**Arity is not a join.** It works on 72 and is wrong in principle on all 76: it
pairs whatever happens to be present rather than what refers to the same words.

### A value-based join, measured

Matching the numbers in the facet's label against the parser's bound:

| | Questions |
|---|---:|
| every facet half joined **uniquely** | **61** |
| at least one facet half joined to **nothing** | **15** |
| ambiguous — matched more than one parser half | 0 |

The 15 failures share one cause, and it is instructive:

> *how many loans have a balance above **£250k*** — the facet label is
> `"balance over 250"`; the parser holds `250000.0`.

`_detect_thresholds` builds its label as `f"{subject} {word} {number}"` where
`number = match.group(1)` — the raw digit group, **losing the multiplier
suffix**. The two interpreters hold different magnitudes of the same number, so
a value join fails on precisely the scaled-unit cases and cannot be made sound
by tightening the matcher.

### Offsets are the sound key, and they exist already

`execution_receipt._detect_thresholds` iterates
`re.finditer(pattern, q, re.I)`. It uses `match.start()` and `match.end()` to
slice a window for the subject word — and then **discards them**, constructing
`RequestedFacet(kind=..., label=...)` with no position. The information the join
needs is in hand and thrown away one line later.
`_detect_geographic_scope` is the same shape: `re.search`, match discarded.

**Two claims whose spans overlap are two halves of one clause.** That is sound
where a value match is not: it survives unit scaling, it disambiguates the 1:2
and 2:1 cases correctly, and it leaves a genuinely unmatched half unmatched
rather than pairing it with whatever remains.

### What each interpreter would need to emit

| Interpreter | What it must emit | Difficulty |
|---|---|---|
| `execution_receipt._detect_thresholds` | `(match.start(), match.end())` on the facet | **one line** — already computed and discarded |
| `execution_receipt._detect_geographic_scope` | the same, from its `re.search` match | **one line** |
| `llm_query_parser._parse_filters` | the span of each clause it consumed | **not one line** — see below |

The parser side is the harder half and should not be assumed cheap.
`_parse_filters` sets `work_q = q` and then **rewrites it as it consumes
clauses** (`work_q = work_q[:bm.start()] + " " + work_q[bm.end():]`, line 2106)
before splitting the remainder on `_CLAUSE_SPLIT_RE`. After that rewrite the
clause strings carry no offsets, and any offsets they did carry would index a
mutated string rather than the question. Emitting sound spans there requires
either not rewriting, or carrying an offset map through the rewrite.

**Consequence for sequencing:** the facet side of the join can be supplied
cheaply; the parser side is a real change to a consumer, which is Stage 3 work
by the contract's own rule. Stage 2 can therefore *record* both halves faithfully
and report them as unjoined, but it **cannot complete the join without a Stage 3
change**. That should be settled before Stage 2 rather than discovered in it.

---

## Recorded, not acted on

### The 46 operation disagreements — no user-visible difference today

**Answered: none of the 46 reaches a user-visible difference.**

`answer_type.asked()` is called from **no production path**. The only non-test
importer of `mi_agent.answer_type` is `mi_agent/mi_calibration.py` — the bank
evaluator — and it uses `of_measure()` and `satisfies()`, **not `asked()`**.
Every other caller is an offline evidence script under `due_diligence/evidence/`.
The module's own docstring says as much: *"One classifier, used by both the
bank's evaluator and the type-conformance sweep."*

So the disagreement is real and live in shipping code, and it changes no served
answer.

**But it is not inert, and there is a hazard for Stage 3.** All **252**
calibration cases carry `expected_answer_type`, and those values were *derived*
from `asked()` by
`due_diligence/evidence/forecast_composition_hardening/derive_answer_type.py`.
`asked()`'s readings — including the 46 disagreements — are **frozen into the
bank's expectations**.

Two consequences:

1. Converting `answer_type.subject_side` first, as the contract's Stage 3 order
   specifies, is confirmed as the **safest possible first conversion**: it
   cannot move a served answer, because nothing serving calls it.
2. **The calibration bank must not be regenerated during this programme.**
   Regenerating it would rewrite 252 expectations from a changed classifier,
   and a bank that moves with the code it grades has stopped being a control.

### The value-domain finding

Logged as backlog item **B1** in `docs/mi_question_interpretation_programme.md`,
with the mechanism scoped: route `_CATEGORICAL_FILTER_RE` through the
book-profiled allowlist rather than its two denylists. Not now — the surviving
cases fail closed.

---

## Stage 2 is not started

Awaiting approval, and a decision on the join sequencing question above.
