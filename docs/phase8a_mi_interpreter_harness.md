# Phase 8A — MI Question Interpretation Harness, Deterministic Baseline & Golden Examples

**Status:** Implemented (deterministic baseline interpreter + golden dataset +
evaluator + tests + docs). **No external LLM calls.** No Azure, no onboarding
orchestration, no Streamlit migration, no M&A runtime, no Annex 2/regulatory
changes, no new chart types. Generated specs never bypass validation; the
interpreter computes no analytics.

**Dependency order:** Phases 0–7 → **Phase 8A**.

**Date:** 2026-06-18

Phase 8A builds the **testing & evaluation harness** for translating business
questions into MIQuerySpec v2. It deliberately uses a deterministic, rule-based
baseline (not an LLM) so the grading harness exists *before* any model is wired
in.

---

## 1. What was built

`mi_agent/interpreter/`:

| File | Responsibility |
|---|---|
| `models.py` | `InterpreterContext` (deterministic client/date anchors) and `InterpretationResult` (raw_question, candidate_spec, normalized_spec, validation_result, confidence, issues, clarification_required/question, interpretation_method). |
| `deterministic.py` | `interpret(question, context)` — a keyword/rule baseline mapping a controlled question set onto MIQuerySpec v2 dicts, then `normalized()` + `validate_query_spec()`; returns a clarification when ambiguous. |
| `evaluator.py` | `evaluate_interpretation(result, expected_spec=…, expected_valid=…, expected_issue_codes=…, expected_clarification_required=…)` → `EvalReport`. Reused later to grade an LLM. |
| `examples.py` | Golden loader + the supported question-family list. |
| `__init__.py` | Public API. |

Golden dataset: `tests/fixtures/mi_interpreter/golden_questions.yaml` — **30
examples (20 valid, 10 invalid/ambiguous)** aligned with
`docs/mi_query_spec_v2_interpretation_contract.md`.

### Supported question families (baseline)
Current state (funded / pipeline / forecast-funded); breakdown by portfolio /
region / stage; temporal trend (last three months); temporal compare (vs last
month / what changed); risk grade / IFRS 9 / PD migration; risk deterioration
flags; concentration by region / broker; quantile buckets (balance / interest
rate / time-on-book) and configured LTV bands; and ambiguous →
clarification (bare *stage* / *portfolio* / *risk* / *changes* / *rate*).

---

## 2. Why this is NOT the final LLM interpreter

The baseline is a small set of deterministic rules over keywords — it does not
generalise, does not understand paraphrase robustly, and is not meant to ship as
the product interpreter. Its job is to (a) prove the end-to-end translation path
(question → spec → normalise → validate) works, and (b) provide a **fixed,
deterministic dataset and evaluator** so a future LLM interpreter can be graded
objectively and regression-tested without an LLM in CI.

---

## 3. How deterministic examples create a grading harness

`golden_questions.yaml` is the single source of truth. Each example declares the
question, the expected (normalised) spec fields, expected validity, expected
issue codes, and whether a clarification is required. Two grading modes:

- **Interpreter-graded** (`interpreter_supported: true`): run `interpret(...)` and
  compare with `evaluate_interpretation` (spec fields / clarification / validity).
- **Spec-graded** (`interpreter_supported: false`): build the expected spec and
  assert `validate_query_spec` produces the expected validity + issue codes —
  this lets the dataset include route/temporal/risk validation-failure cases the
  baseline doesn't emit.

Swapping in an LLM later means pointing the same evaluator at the LLM's
`InterpretationResult`; the dataset and pass/fail criteria are unchanged.

---

## 4. How future LLM output will be constrained to MIQuerySpec v2

The LLM (Phase 8B) will be required to emit **only** MIQuerySpec-v2 JSON (per
`docs/mi_query_spec_v2_interpretation_contract.md`) — never code, SQL, chart
specs, computed numbers, or unlisted fields. Its output will be fed through the
exact same pipeline the deterministic baseline uses:

```
llm_json → MIQuerySpec.from_dict → .normalized() → validate_query_spec → (run_mi_query)
```

`MIQuerySpec.from_dict` drops unknown keys, `normalized()` maps convenience
fields onto canonical runtime fields, and `validate_query_spec` rejects invalid
combinations — so a misbehaving LLM cannot bypass validation or compute
analytics. All analytics remain in `run_mi_query` and the governed engines.

---

## 5. How ambiguity / clarification works

When a question is under-specified or uses a bare ambiguous concept, the
interpreter returns `clarification_required=True` with a concrete
`clarification_question` and **no spec** — it never guesses. Triggers:

- bare *stage* outside a pipeline context (which stage taxonomy?);
- *portfolio* with no Trakt portfolio reference config available;
- *risk* with no chosen dimension/mode (migration vs flags vs concentration);
- *changes* with no period;
- *rate* where metric vs buckets is unclear.

A clarification is explicitly **not** a valid answer (`result.ok` is False), so a
clarifying interpretation can never be mistaken for a runnable query.

---

## 6. Validation integration

Every interpreted spec is passed through `MIQuerySpec.normalized()` and
`validate_query_spec()`; `InterpretationResult.validation_result` is always
populated for a produced spec, and `result.ok` requires `validation_result.ok`.
Raw interpreter output is never treated as valid on its own.

---

## 7. Files changed

- `mi_agent/interpreter/{__init__,models,deterministic,evaluator,examples}.py` (new)
- `tests/fixtures/mi_interpreter/golden_questions.yaml` (new, 30 examples)
- `tests/test_phase8a_mi_interpreter_harness.py` (new, 67 tests)
- `docs/phase8a_mi_interpreter_harness.md` (this file)

## 8. Tests run

`tests/test_phase8a_mi_interpreter_harness.py` — **67 passed**. MI Agent +
Phase 6 + Phase 7 + Phase 8A — **291 passed** (no regression).

## 9. What remains deferred to Phase 8B

- **The LLM interpreter itself** — prompt, an `llm_stub`/adapter behind the same
  `InterpretationResult` contract, and grading the LLM against this dataset
  (still with no external LLM call in CI — use recorded fixtures / a stub).
- **Broader natural-language coverage** — paraphrase robustness, multi-intent
  questions, follow-up/clarification dialogue, and resolving relative dates from
  a real reporting calendar rather than the fixed harness anchors.
- **Wiring the interpreter into `run_mi_query`** end-to-end (interpret → execute)
  as a single call, once the LLM path is graded.
