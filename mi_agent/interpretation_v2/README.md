# interpretation_v2 — the natural-language control plane

```
question
   │
   ▼
OpusInterpreter          the model owns LANGUAGE
   │
   ▼
CandidateIntent          meaning — never a binding
   │
   ▼
DeterministicCompiler    the compiler owns BINDING
   │
   ├──────────────► GovernedQueryPlan
   └──────────────► Refuse / Clarify
```

**Shadow only.** Nothing here is wired into `/mi/query`, the API, React, Teams,
the dashboard or any deterministic engine, and nothing here executes a plan. The
sprint ends at the plan. `tests/interpretation_v2/test_bank_and_shadow_boundary.py`
asserts that no module outside this package imports it.

## Why the middle was missing

The free-form LLM parser let the model write a complete `MIQuerySpec` — an
executable contract. The guards then validated a contract the model itself had
authored, and the repair loop negotiated around a refusal until one passed.

The replacement went the other way: a deterministic parser interpreted the
question first, and the model was only allowed to fill empty slots. That left
Python doing natural-language understanding, which is the thing it is worst at.

This package is the middle. **The model may interpret meaning. The model may not
create the executable contract.**

## The ownership split

| Opus owns | The compiler owns |
|---|---|
| capability intent | whether a concept exists |
| operation intent | concept → governed registry binding |
| semantic measures | canonical field selection |
| aggregation / statistic | portfolio and capability availability |
| population, filters, dimensions | allowed operation/measure combinations |
| geography request (basis, level) | geography field/basis binding |
| time request (semantic form) | period/snapshot binding |
| comparison, output structure | ambiguity and unsupported-composition detection |
| what it could not decide | the refusal / clarification decision |

Execution — filters, grouping, arithmetic, borrowing-base methodology, bridge
methodology, output values — is owned by the deterministic engines and is out of
scope here.

## The safety boundary

`CandidateIntent` and `GovernedQueryPlan` are different objects, and the
difference is enforced rather than documented.

The model may say:

```json
{"measures": [{"concept": "loan_to_value", "statistic": "weighted_average",
               "weight": "balance"}],
 "geography": {"basis": "collateral", "level": "itl3"},
 "time": {"form": "previous_reporting_period"}}
```

The model cannot say `field: current_ltv`, `snapshot: 2026-06-30`, a dataframe
expression, SQL, or pandas — **there is no slot for it**. Three layers:

1. **No slot exists.** Walked and asserted over the dataclass tree.
2. **The generated schema is closed.** `additionalProperties: false` everywhere,
   with `tool_choice` pinned, so a compliant generation cannot emit one.
3. **The parser fails closed.** An unknown key, an out-of-vocabulary enum, a code
   marker, an ISO date or a snapshot-shaped token rejects the whole intent.
   Nothing is stripped and salvaged, and there is no repair loop.

## Modules

| file | what it owns |
|---|---|
| `intent.py` | the `CandidateIntent` contract, its JSON schema, fail-closed parsing |
| `vocabulary.py` | the semantic boundary in front of the physical registries |
| `opus_interpreter.py` | the one place a model is consulted, and the only place the prompt is built |
| `compiler.py` | validate → bind → authorise → produce |
| `plan.py` | the immutable, versioned `GovernedQueryPlan` |
| `outcomes.py` | `PLAN` / `CLARIFY` / `REFUSE` and the governed reason codes |
| `equivalence.py` | paraphrase invariance and per-dimension scoring |
| `benchmark.py` | the 135-question interpretation benchmark |
| `banks/` | the frozen bank and the human-reviewable expected intents |

## The rule the compiler obeys

**It never re-reads the question.** No regex over the original sentence, no
recogniser cascade, no second opinion from wording. `compiler.py` does not import
`re`, and the same intent under two completely different question strings
compiles to a byte-identical plan. Source-span evidence is copied into
provenance and read by nothing.

## Running it

```bash
# unit tests — no network
python -m pytest tests/interpretation_v2 -q

# the 135-question benchmark against the configured model
ANTHROPIC_API_KEY=... python -m mi_agent.interpretation_v2.benchmark \
    --live --out run.json

# replay a previous run's model payloads, offline and deterministically
python -m mi_agent.interpretation_v2.benchmark --replay run.json --out replay.json

# rebuild the frozen bank from its sources (must reproduce byte for byte)
python -m mi_agent.interpretation_v2.banks.build_bank_135
```

`MI_BEARER` is not used and is not required: nothing here calls the Trakt MI
API. `ANTHROPIC_API_KEY` is the model credential and the only environment
variable this package reads. Without one, the compiler is fully testable against
replayed payloads and the interpretation layer is reported as **NOT PROVEN**
rather than assumed.

## The benchmark

135 questions = 45 canonical × 3 variants, every string copied verbatim from a
bank already frozen in this repository (`banks/build_bank_135.py` names the four
sources and the rule by which each triplet was taken). It is an INTERPRETATION
benchmark: it scores what a question means and what Trakt would be authorised to
do, never what the answer is.

Two scoring rules:

* a dimension the fixture does not state is **unscoreable, not correct**;
* a governed refusal is **not a failure** — it is reported with its reason code
  beside the plans.

Expected intents in `banks/expected_intents.yaml` are derived from the question
text and from independent semantic truth that predates this sprint. They are
**not** derived from the legacy parser: scoring a new interpreter by agreement
with a known-flawed one measures agreement, not correctness.
