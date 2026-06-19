# Phase 8C — end-to-end NLQ → runtime MI smoke harness

## What was built

Phase 8C connects the two halves built earlier into one governed pipeline and
proves it works end to end:

* **Phase 8A** gave a deterministic interpreter + golden questions.
* **Phase 8B** gave the Anthropic-first interpreter adapter (NL → `MIQuerySpec`
  v2 JSON), with safe parsing, normalisation, validation and clarification.
* **Phase 6B/6C** proved `run_mi_query` over canonical / multi-artefact
  snapshots.

Phase 8C adds a thin orchestration bridge and an end-to-end smoke suite:

| File | Role |
| --- | --- |
| `mi_agent/interpreter/runtime_bridge.py` | `interpret_and_run_mi_query(...)` — interpret a question, then (only if valid & unambiguous) execute it via `run_mi_query`. Returns a combined `BridgeResult`. |
| `tests/test_phase8c_nlq_to_runtime_smoke.py` | Fake-client end-to-end tests over the Phase 6B synthetic snapshots + invalid/ambiguous execution guards. |
| `scripts/mi_nlq_dev_smoke.py` | **Dev-only** manual script for a real Anthropic client (not part of CI). |

### The bridge

```python
interpret_and_run_mi_query(
    question,
    context,
    llm_client_or_interpreter,   # Anthropic-style client OR interpreter callable
    store,
    *, data=None, semantics=None, risk_config=None, build_chart=False, ...
) -> BridgeResult
```

`BridgeResult` carries: `raw_question`, `interpretation` (the
`InterpretationResult`), `normalized_spec`, `runtime_result`, combined `issues`,
and `executed: bool`. `executed` is `True` only when a valid, unambiguous spec
was actually run through `run_mi_query`.

The bridge accepts either an Anthropic-style client (anything with
`complete_mi_spec_json`, routed through the Phase 8B adapter) or a plain
interpreter callable `f(question, context) -> InterpretationResult` (e.g. the
Phase 8A deterministic baseline). This keeps it usable in tests and from a future
UI without coupling to a provider.

## What "end-to-end" means here

A single call takes a **natural-language question** and returns **computed MI
output** — but through the governed contract, not by letting the model compute:

```
question
  → interpreter (LLM adapter or deterministic)   # proposes MIQuerySpec v2 JSON
  → MIQuerySpec.normalized()                      # canonical runtime fields
  → validate_query_spec()                         # deterministic gate
  → run_mi_query()                                # the ONLY execution engine
  → BridgeResult (rows / metrics / chart_instruction)
```

The LLM contributes **intent only**. Every number, aggregate, migration and trend
is produced by `run_mi_query` over the snapshot layer.

## Why this is still synthetic / local-only

* Data is small, deterministic, in-memory snapshot frames registered in a
  `LocalFsSnapshotStore` (the Phase 6B setup) — no real client data.
* CI/tests use **fake Anthropic clients only**: canned spec JSON, no network, no
  API key, no `anthropic` SDK dependency for test collection.
* The only path that talks to a real model is the **dev-only** script, which is
  excluded from the suite and refuses to run without `ANTHROPIC_API_KEY`.

## How interpretation and execution are separated

* The interpreter (LLM or deterministic) **never** computes analytics — it emits
  a spec or a clarification.
* The bridge holds two gates before any execution:
  1. **Clarification gate** — a `clarification_required` interpretation is never
     executed (`not_executed_clarification_required`).
  2. **Validity gate** — an interpretation that did not produce a valid,
     validated spec is never executed (`not_executed_invalid_spec`).
* Only a validated, normalised `MIQuerySpec` reaches `run_mi_query`.

## How invalid / ambiguous questions are blocked

The guard tests prove none of the following execute (each yields
`executed=False`, `runtime_result is None`):

| Situation | Why blocked |
| --- | --- |
| Ambiguous question (e.g. "show risk") | interpreter returns a clarification |
| Model clarification object | clarification gate |
| Malformed LLM output | adapter → clarification (not parseable) |
| Invalid enum (`state: not_a_state`) | adapter validation error → clarification |
| Hallucinated field (with semantics) | adapter `llm_hallucinated_field` → clarification |
| Missing temporal dates (compare/trend) | `temporal_selector_incomplete` |
| Regulatory route + MI state | `invalid_route_for_state` |
| M&A route + pipeline/forecast state | `invalid_route_for_state` |

## Supported questions and expected MI outputs (synthetic snapshots)

| Question | Mode | Expected output |
| --- | --- | --- |
| show total funded | state | 3 funded loans, balance 620 |
| show total pipeline | state | 1 pipeline loan, balance 50 |
| show forecast funded | state | `forecast_funded_total` 645 |
| trend funded balance over the last three months | temporal | balances [300, 620, 620], `chart_instruction={"chart_type":"line"}` |
| compare funded balance to last month | temporal | baseline 620, current 620, change 0 |
| show funded balance by portfolio | risk/concentration | PF_001 400, PF_002 220 |
| show funded balance by region | risk/concentration | North 400, South 220 |
| show pipeline by stage | risk/concentration | OFFER 50, APPLICATION 40 |
| show concentration by region | risk/concentration | North 400 (+ status column) |
| show risk grade migration | risk/migration | B→C `deteriorated` |
| show IFRS stage migration | risk/migration | Stage 1→Stage 2 `deteriorated` |
| show PD bucket migration | risk/migration | a `deteriorated` movement present |

## How a future UI could call this bridge

```python
from mi_agent.interpreter import (
    InterpreterContext, interpret_and_run_mi_query)
from mi_agent.interpreter.anthropic import AnthropicClient

result = interpret_and_run_mi_query(
    user_question,
    InterpreterContext(snapshot_client_id=client_id, route_id="mi", ...),
    AnthropicClient(),            # or a deterministic interpreter callable
    store,                        # the client's SnapshotStore
    semantics=semantics, risk_config=risk_cfg)

if result.clarification_required:
    show_clarification(result.interpretation.clarification_question)
elif not result.executed:
    show_blocked(result.issue_codes())
else:
    render(result.data, result.chart_instruction)
```

The UI never touches analytics or validation logic directly — it submits a
question and renders a `BridgeResult`.

## Dev-only live Anthropic smoke

`scripts/mi_nlq_dev_smoke.py` runs a real Anthropic client against synthetic
local snapshots. It is **not** collected by pytest, requires `ANTHROPIC_API_KEY`,
imports the `anthropic` SDK lazily, labels its output `DEV SMOKE ONLY`, and still
passes everything through `MIQuerySpec.normalized()`, `validate_query_spec()` and
`run_mi_query`. The repo does **not** depend on the Anthropic SDK for normal test
collection.

```bash
ANTHROPIC_API_KEY=sk-... python scripts/mi_nlq_dev_smoke.py "show total funded"
```

## What remains deferred

* No Azure adapter, onboarding orchestration, Streamlit migration, M&A runtime,
  Annex 2 / regulatory changes, new chart types, or production UI.
* No live LLM in CI; the deterministic interpreter and fake clients are the test
  oracles.
* Real client data, multi-tenant stores, auth, and a production NLQ surface are
  future work that can build on this bridge unchanged.

## How to run

```bash
python -m pytest tests/test_phase8c_nlq_to_runtime_smoke.py -q
```
