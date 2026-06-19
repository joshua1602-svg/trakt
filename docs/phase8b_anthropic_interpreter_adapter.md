# Phase 8B — Anthropic-first LLM interpreter adapter for MIQuerySpec v2

## Purpose

Phase 8A built a deterministic, rule-based interpreter that maps a controlled
set of MI questions onto a governed `MIQuerySpec` v2, plus a golden dataset and
an evaluator. Phase 8B adds the **LLM path**: an Anthropic/Claude-first adapter
that turns an arbitrary natural-language MI question into the *same* governed
spec contract.

The central product rule is unchanged and strictly enforced:

> **The LLM only interprets. It never computes analytics.**

Claude is asked to return a `MIQuerySpec`-v2-compatible JSON object (or a
clarification). Every candidate spec is then run through the deterministic gate —
`MIQuerySpec.from_dict(...).normalized()` then `validate_query_spec(...)` — before
it is ever considered runnable. All numbers, aggregates, migrations and trends
continue to be produced exclusively by the deterministic runtime
(`run_mi_query`). The model contributes *intent*, not *results*.

## Why Anthropic-first (and not a provider framework)

For the foreseeable future the intended provider is Anthropic/Claude. Rather than
build a broad provider-agnostic abstraction, Phase 8B keeps the Anthropic-specific
surface behind a single thin, mockable boundary and invests the effort in the
governance/validation layer instead. Swapping or adding a provider later means
implementing one small method — not rewriting the adapter.

## Components

| File | Role |
| --- | --- |
| `mi_agent/interpreter/prompt.py` | Builds the constrained Claude prompt from code constants (allowed fields, enums, route ids, interpretation rules, context anchors). |
| `mi_agent/interpreter/anthropic.py` | The adapter: client boundary, safe parsing, governance checks, normalise + validate, result assembly. |
| `tests/test_phase8b_anthropic_interpreter_adapter.py` | Fake-client tests; reuses the Phase 8A golden dataset + evaluator. |

### The client boundary

```python
class AnthropicMIInterpreterClient(Protocol):
    def complete_mi_spec_json(self, prompt: str) -> str | dict: ...
```

This is the *only* contract the adapter depends on: given a built prompt, return
the model's raw completion. All transport, auth, model selection and retries live
behind it.

- `AnthropicClient` is the real implementation. It imports the `anthropic` SDK
  **lazily** (via `importlib`, inside the method) so the dependency is optional
  and never touched at import time.
- Tests use **fake clients only** — no API keys, no network. The adapter module
  imports no SDK at module load.

### Entry points

```python
interpret_with_anthropic(question, context, client, *, semantics=None) -> InterpretationResult
interpret_from_llm_output(question, raw, context, *, semantics=None) -> InterpretationResult
```

- `interpret_with_anthropic` builds the prompt, calls the client, and delegates.
  Client exceptions are captured as a structured `llm_client_error` clarification
  — transport failures never propagate.
- `interpret_from_llm_output` is the directly testable core: feed it a canned
  string/dict and it parses, governs, normalises and validates with no client at
  all.

## The constrained prompt

`build_mi_spec_prompt` assembles a single instruction from code constants (no
markdown read at runtime). It tells Claude to:

1. Return **JSON only** — no prose, no markdown fences.
2. Return **either** one `MIQuerySpec`-v2 object **or** a clarification object
   `{"clarification_required": true, "clarification_question": "..."}`.
3. Never return Python, pandas, or SQL.
4. Never calculate or return data values/results.
5. Never invent field names — use only the allowed fields and enum values.
6. Never invent dates — use only the supplied context anchors.
7. Never use chart types outside the governed chart list.
8. Ask for clarification when the question is ambiguous rather than guessing.

It embeds the allowed top-level spec fields, the allowed enum values (route ids,
execution modes, states, temporal/risk modes, bucket strategies, trend grains,
forecast probability sources, output types, chart types, segments, aggregations),
the known semantic field keys, the natural-language interpretation rules
(portfolio → `portfolio_id`, region → `geographic_region_obligor`, balance/rate/
time-on-book → quantile buckets, vague "risk"/"changes" → clarify, etc.), and the
deterministic context anchors (`as_of`, `prev_period`, `range_start`, client id,
route, portfolio-config availability).

## Safe-by-construction behaviour

The adapter treats the model output as untrusted and degrades to a structured
**clarification** rather than executing anything unsafe. Outcomes:

| Model output | Handling | Result |
| --- | --- | --- |
| Valid JSON spec that validates | normalise + validate | `ok=True` |
| Clarification object | passed through | clarification |
| Markdown fences / prose around JSON | fences stripped, first balanced `{...}` extracted | parsed |
| JSON list / scalar | rejected | `llm_output_not_object` → clarify |
| Malformed JSON | rejected | `llm_malformed_json` → clarify |
| Code (import/def/`pd.`/SELECT/…) | detected | `llm_output_contains_code` → clarify |
| Empty / `None` | rejected | `llm_empty_output` → clarify |
| Unknown field | dropped by `from_dict`, warned | `llm_unknown_field` (WARNING), still validates |
| Unsupported `chart_type`/`chart_preference` | error | `llm_unsupported_chart_type` → clarify |
| Hallucinated dimension (when `semantics` supplied) | error | `llm_hallucinated_field` → clarify |
| Spec that fails `validate_query_spec` | not run | validation issue codes → clarify |
| Client/transport exception | captured | `llm_client_error` → clarify |

The governing rule: `result.ok` is `True` only when the spec validates **and**
there is no adapter-level ERROR. Otherwise `clarification_required=True`, the spec
is never executed, and the issues explain why. `interpretation_method` is
`llm_stub` for this path (distinct from `deterministic`).

### Adapter issue codes

`llm_malformed_json`, `llm_output_not_object`, `llm_output_contains_code`,
`llm_unknown_field`, `llm_unsupported_chart_type`, `llm_hallucinated_field`,
`llm_client_error`, `llm_empty_output`. These sit alongside (never replace) the
spec-validation codes from `mi_spec_validation`.

## Grading / reuse of Phase 8A

The Phase 8A golden dataset (`tests/fixtures/mi_interpreter/golden_questions.yaml`)
and `evaluate_interpretation` are reused: a fake Anthropic client returns a
complete, valid spec for each valid golden question, and the adapter result is
graded against the same golden expectation on the same governed contract. This
proves the LLM path and the deterministic path converge on identical, validated
specs when the model behaves — and the negative tests prove every misbehaviour
degrades safely.

## Explicit non-goals (honoured)

- No external Anthropic calls in tests; no API keys; no network.
- No analytics computed by the LLM.
- No broad multi-provider framework; no Azure.
- No onboarding orchestration, Streamlit migration, or M&A runtime changes.
- No Annex 2 / regulatory changes; no new chart types.

## How to run

```bash
python -m pytest tests/test_phase8b_anthropic_interpreter_adapter.py -q
```
