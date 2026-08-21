# Stage 3 — four consumers converted, nothing moved

| | |
|---|---|
| Base | merge-base with `claude/mi-analytical-capability-layer-vlkjfw` is `4e051f3` exactly; `4e051f3` ✓ and `28ece25` ✓ ancestors of HEAD |
| Release candidate | `28ece25`, unchanged and shippable |
| Commits | `699ff53` instrument · `3e4e53e` · `1213f5f` · `c075250` · `cbd9f46` |
| Production change | 4 files, **+52 / −84 lines** — a net removal |

**Every conversion is its own commit with its own before/after on both surfaces.
Nothing moved at any of them.**

## Acceptance, per conversion

| | lexical decisions | answer text | calibration bank | robustness, both books |
|---|---|---|---|---|
| 1 `answer_type.subject_side` | 690/690 | 340/340 | 260 | 32/10/2 · seasoning 20/20 |
| 2 `execution_receipt._is_filter_subject` | 690/690 | 340/340 | 260 | 32/10/2 · seasoning 20/20 |
| 3 `llm_query_parser._metric_slot` | 690/690 | 340/340 | 260 | 32/10/2 · seasoning 20/20 |
| 4 `period_request.requested_unit` | 690/690 | 340/340 | 260 | 32/10/2 · seasoning 20/20 |

44 of 44 variations agreeing across books at every step. Unsafe outcomes zero
throughout. 133 `question_interpretation` tests; 14 of 14 schema mutations
caught.

### A finer gate than the answer diff

The answer diff proves nothing a reader sees moved. It cannot prove a
consumer's own decision did not move — a conversion that changed
`subject_side` but happened not to reach the answer would pass it. So
`question_interpretation.lexical_decisions` snapshots what each consumer
decides, verbatim, on all 690 real-surface questions, and every conversion was
gated on both. Seven tests prove that instrument can fail, including
attributing two different consumers moving to each by name.

---

## What each conversion did

### 1 · `answer_type.subject_side`

**Used to decide** the subject-side span from its own copy of thirteen condition
openers. **Now** asks `question_interpretation.lexical`, which declares them
once. Equivalence proved on 690 corpus questions **and 18 edge cases the corpus
does not reach** — empty, `None`, a bare opener, a cut at position zero, an
uppercase `BY` — because the original returned the whole head where the prefix
was empty and that branch has no corpus coverage.

### 2 · `execution_receipt._is_filter_subject`

**Used to decide** whether a measure word at a given position is the subject of
a predicate, from two window regexes and a comparator list. **Now** asks the
owner. This one is on the parser path — `llm_query_parser._measure_hits`
imports it — so equivalence was proved **exhaustively at 400,097 positions**:
every offset pair across the corpus, not only the ones the receipt layer probes.

**The two vocabularies are kept distinct, deliberately.** This test wants
comparators, so it carries `between`, `exceeding`, `in excess of` and the
operators and not `where`, `with`, `for`. The subject-side split wants clause
openers and carries the opposite. The inventory's finding was that the three
splits were separately **declared**, not that they should be identical —
merging them would have been the wrong fix, and a test records why.

### 3 · `llm_query_parser._metric_slot` — the acceptance criterion

**Used to decide** the metric-naming span from thirteen openers declared here
**and, byte for byte, in `answer_type`**. **Now** asks the owner.

> **The subject-side clause split exists once, not three times.**

Asserted directly, not inferred: a test searches the three converted production
sources for a second declaration of the opener trio, and a companion proves that
search is live by matching the owner's own declaration. A re-introduced copy
that nothing calls yet would pass every behavioural test and fail this one.

`_metric_slot` and `subject_side` share the vocabulary and differ in
composition — `_metric_slot` must **not** cut at a grouping clause, because the
parser splits grouping upstream in `_grouping_segments`. That is why the owner
exposes `condition_cut` and `grouping_cut` separately: one fused function could
serve only one consumer.

### 4 · `period_request.requested_unit` — Stage 5's input

**Used to decide** the finest time unit named, from its own unit vocabulary.
**Now** asks the owner.

A test now asserts the Stage 5 claim rather than quoting it: **the grain is
already correct on every time-series probe that names one**, including *by
quarter* and *each month*, which `_deterministic_parse` does not recognise as a
time axis at all. What is missing is carriage, not comprehension.

**Deliberately partial.** `requested_span` did **not** move: it resolves vague
recency (*"recently"*, *"a few months ago"*) against the governed seasoning
configuration's `lending_windows.recent_max_months` rather than from the
wording. That is a config decision wearing a lexical coat, and the owner reads
question text and nothing else — no registry, no frame, no spec, no config.
The value of the owner is precisely that a consumer converting onto it gains no
knowledge it did not have.

---

## The remainder — convert, or documented as independent

The acceptance criterion requires every remaining interpreter to consume the
object or be documented as legitimately independent, with the reason stated.
This is the classification, for decision.

| Interpreter | Classification | Reason |
|---|---|---|
| `llm_query_parser._deterministic_parse` | **partially converted** | `_metric_slot` now reads the owner. `_parse_filters` remains — converting it means resolving how it rewrites the question, which is the open item below |
| `execution_receipt.detect_requested_facets` | **feeds the object** | supplies the wording half and its offsets, since Stage 2 |
| `answer_type.asked` | **converted** | via `subject_side` |
| `period_request` | **partially converted** | `requested_unit` and `finer_than` converted; `requested_span` independent, above |
| `population.fabricated_bounds` / `fabricated_concepts` | **legitimately independent — safety mechanism** | it exists to catch the spec disagreeing with the question. It must derive from the question *independently*, or it cannot disagree. Converting it onto a shared object would defeat its purpose exactly |
| `mi_agent_workflow._detect_unsupported_concept` | **legitimately independent — safety mechanism** | same shape: a guard that shares its input with the thing it guards is not a guard |
| `portfolio_lens` | **legitimately independent — semantic resolver** | resolves scope against the portfolio registry, not from wording alone |
| `chat_routing` — 15 functions | **legitimately independent — intent classifiers** | they decide *which capability answers*, not what the words mean. Candidates for a later pass, not for this programme |
| `concentration_query.detect_intent` | **legitimately independent — intent classifier** | as above |
| `period_change.recognise` | **legitimately independent — intent classifier** | as above |
| `concentration_tests/matching.extract_from_text` | **legitimately independent — extractor** | reads covenant documents, not user questions |

### One correction to the inventory

The inventory listed `mi_agent/interpreter/deterministic.interpret` as "a second
whole parser". **It is not on the serving path.** The `mi_agent/interpreter`
package is imported only by `scripts/mi_nlq_dev_smoke.py` and
`scripts/phase8e_live_anthropic_smoke.py`, and by its own modules. It is a
development smoke tool. The inventory's count of eleven live entry points should
read ten, and this one needs no conversion — though whether a dev tool should
carry a parallel parser at all is a separate question.

---

## Open, for decision before the remainder

**`_parse_filters` is the binding half of the filter join**, and converting it
is the one conversion that cannot be proved by equivalence alone. It sets
`work_q = q` and rewrites it as it consumes clauses
(`work_q = work_q[:bm.start()] + " " + work_q[bm.end():]`), then splits the
remainder. Offsets taken after that rewrite index a mutated string.

Two routes, and the contract says to decide on measurement rather than
preference:

1. **Remove the rewrite** — parse from the original string with consumed ranges
   tracked separately. Cleaner, and a larger behavioural surface.
2. **Maintain an offset map** through the rewrite. Smaller, and it leaves the
   rewrite in place to be reasoned about again later.

Both change a consumer that is on the answer path, so both need their own
before/after. Neither is started.

---

## Pre-existing failures, unchanged throughout

Four, all one root cause: the real alderbridge tape does not carry
`borrower_type`.

* `test_p0_execution_receipt.py` — three tests
* `test_mi_predicate_extraction.py::test_complex_query_executes_all_filters`

Verified identical before and after every conversion by stashing. Not caused
here, not fixed here.

Stage 4 is not started.
