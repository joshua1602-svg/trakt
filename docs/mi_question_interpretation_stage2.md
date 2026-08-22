# Stage 2 — the object is built, carried, and ignored

| | |
|---|---|
| Base | merge-base with `claude/mi-analytical-capability-layer-vlkjfw` is `4e051f3` exactly; `4e051f3` ✓ and `28ece25` ✓ ancestors of HEAD |
| Release candidate | `28ece25`, unchanged and shippable |
| Commits | `49cde79` facet spans · `8d3dcea` build and carry |
| Production change | 2 files, **+46 lines, 0 removed** |

## Acceptance — met on both surfaces, deterministic arm

| Gate | Result |
|---|---|
| **Answer text, byte-identical** (standing condition 7) | **340 / 340 identical, 0 moved** |
| Calibration bank | 260 passed |
| Robustness, alderbridge | 32 `CORRECT` · 10 `SAFE_REFUSAL` · 2 disclosed · **0 unsafe** |
| Robustness, kestrelmoor | identical, and 44 of 44 agreeing across books |
| **Seasoning family (Q1, Q7, Q8)** | **20 / 20 `CORRECT`, per book, by name** |
| `question_interpretation` tests | 72 passed |
| Schema mutation check | 14 mutations, **14 caught** |

Recorded for attribution: three tests in
`mi_agent/tests/test_p0_execution_receipt.py` fail in this environment. The
failing set is **identical before and after both commits**, verified by
stashing. They require `borrower_type`, which the real alderbridge tape does not
carry. Pre-existing, not caused here, not fixed here.

---

## Standing condition 7 earned its place on first use

The facet span was initially added to `RequestedFacet.to_dict()` as well as the
dataclass. That is one line, obviously additive, and:

> **the calibration bank passed all 260 — and the answer-text diff caught 32 of
> 340 answers moving.**

`executionSummary` carries facets and is user-visible payload, so serialising a
new key changed what a reader receives. The span now lives in memory only,
which is where the join needs it.

This is exactly the case that justified option (a): proceed without touching the
grading path, because the answer diff is stricter than the grader. It was not a
hypothetical.

---

## The primary deliverable — how the object bridges the two halves

### What changed

`_detect_thresholds` computed `match.start()` and `match.end()`, used them to
slice a window for the subject word, and discarded them one line later.
`_detect_geographic_scope` had its match in hand and dropped it. Both now keep
them, on `RequestedFacet.span`.

### The result, on the three real surfaces

| | |
|---|---:|
| questions carrying a **wording-only** claim and a **binding-only** claim | **71** |
| of those, wording half **located by span** | **71 / 71** |
| binding halves located | **0 / 71** |
| `clause_id` set | **0** |
| claims with no recoverable span, whole 939-question corpus | **215 → 120** |

**Stage 1 measured 76 by the equivalent test; this measures 71.** The five are
`geographic_scope` facets, which carry a `field_key` and were therefore never
really half-claims — `provides` now classifies them correctly. A refinement,
stated rather than glossed.

### Why the join is not completed

`clause_id` stays `None` on every claim, and the object records why:

> `filter join HALF-BUILT: 1 wording claim(s), 1 of them located by span;
> 1 binding claim(s), none located — the parser half supplies no offsets, so
> clause_id stays None`

The missing half is named. `_parse_filters` sets `work_q = q` and then rewrites
it as it consumes clauses
(`work_q = work_q[:bm.start()] + " " + work_q[bm.end():]`), splitting the
remainder afterwards. Offsets taken after that rewrite index a mutated string,
not the question. Supplying them means either removing the rewrite or
maintaining an offset map through it — a change to a consumer, which forfeits
this stage's byte-identical guarantee.

**That is Stage 3, decided on measurement rather than preference**, when
`_parse_filters` is converted.

### Why offsets and not the alternatives, restated with the measurement

* **Arity** pairs 72 of 76 by coincidence — 2 cases are 1:2 and 2 are 2:1.
* **Value matching** reaches 61 of 76 and fails on exactly the scaled-unit
  cases: the label is built from the raw digit group, so *£250k* renders as
  `balance over 250` while the parser holds `250000.0`.
* **Offsets** are exact, survive unit scaling, disambiguate the 1:2 and 2:1
  cases, and leave a genuinely unmatched half unmatched.

---

## Faithful population, including what the interpreters get wrong

No role was corrected while populating. The 663-to-15 grouping/filter imbalance
is recorded as it stands and is Stage 4's problem.

### A finding: role attribution is book-dependent

*balance by region for joint borrowers*, on the alderbridge tape, records
`borrower_type` as **`unresolved`**, reason *"no source supplies a role"* — not
as `filter`, which is what the Stage 1 read-only projection recorded.

Both are correct, and the difference is the book:

| | `spec.filters` |
|---|---|
| parsed without `available_columns` | `{'borrower_type': 'Joint'}` |
| parsed **with the real tape's columns** | `{}` |
| `borrower_type` in the tape | **no** |

The parser correctly declines to bind a filter on a field the book lacks. The
facet layer still names it, because `requested_dimension_terms` resolves
*without* availability filtering so the omission can be disclosed rather than
vanishing — and the query then refuses with *"I understood that you asked for
joint borrower, but that could not be applied"*.

**A role is therefore not a property of the question alone.** Stage 4's split
inherits this: the same sentence yields a different role on a book that carries
the field. It is asserted by a test so it cannot drift unnoticed.

---

## The object, on 939 questions

Unchanged from Stage 1 except where the spans moved, because population is
observation:

| Slot | Filled |
|---|---:|
| operation | 934 / 939 |
| subject | 898 / 939 |
| any dimension | 575 / 939 |
| any filter | 123 / 939 |
| time grain | 64 / 939 |
| target | 12 / 939 |

Dimension roles across the corpus: **663 grouping, 15 filter, 55 unresolved**.

Per-question record: `question_interpretation/stage2_corpus.json`.

---

## What Stage 2 did not do

* No consumer reads the object. A test asserts it, and a second asserts that a
  builder failure cannot cost an answer.
* No role corrected, no span capture added to the parser, no grader touched.
* The two known-wrong bank expectations (`pipe_183`, `pipe_194`) are untouched.

Stage 3 is not started.
