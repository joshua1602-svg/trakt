# Dataset ownership remediation — pre-registered conditions

**Committed before any production change.** Base: `d927963`. Tree clean.

---

## The product decision this implements

> Natural-language MI is self-contained. **The user's question determines the
> analytical dataset.** The active React/workspace tab must not silently change
> dataset semantics.

Funded / Pipeline / Forecast are **dataset** semantics.
Direct / Acquired / named SPV are **population** semantics.
They are not the same axis and must not be conflated.

## Target state

```
question
  → one authoritative dataset resolver   (no tab input)
  → FUNDED | PIPELINE | FORECAST
  → interpretation contract
  → every downstream consumer
```

**One semantic owner. Not two that agree.**

## What is being reproduced first

Measured before any change, `migration_phase0.dataset_ownership_disagreement`,
14 cases × 4 tab values = **56 executions**: **6 cases are tab-sensitive** —
the served dataset changes with the tab for the same question. Among them:

* `"How many cases / applications / KFIs / offers are there?"` — served from
  **funded** on the funded tab, **pipeline** on the pipeline tab, **forecast**
  on the forecast tab, for one unchanged question.
* `"What is the balance by seasoning segment excluding pipeline cases?"` —
  served from **pipeline** on the pipeline tab. The question rules the pipeline
  out in words and the tab puts it back.
* `"What is the total balance?"` — no dataset vocabulary at all; the tab is the
  entire decision.

## Authorised semantic movement

Exactly three classes. Everything else is a blast-radius failure.

| class | what moves | why it is authorised |
|---|---|---|
| **M1 — tab influence removed** | a question whose dataset varied by tab settles on its question-determined value | the product decision; this is the whole point of the task |
| **M2 — one owner, not two** | the tape vocabulary `pipeline \| case \| kfi \| application \| offer`, reachable today **only** from `_route_compare` and `_route_evolution` via `_dataset_for`, now applies at the single owner | §4: duplicated ownership is not preserved for backward compatibility. Without this, "How many applications are there?" cannot resolve PIPELINE, which the target state requires |
| **M3 — forecast precedence restored** | a question naming forecast resolves FORECAST even when it also names a pipeline artefact | `_dataset_for` tests the tape vocabulary **before** any forecast reading, so "Forecast application volumes next quarter" is `pipeline` to it. Forecast must win |

## Required invariants

* `dataset(question, funded tab) == dataset(question, pipeline tab)` for
  ordinary natural-language MI — **the tab-independence invariant**.
* Forecast precedence remains correct; the three corpus forecast questions
  carrying pipeline vocabulary stay **FORECAST**.
* Pipeline concepts resolve **PIPELINE**; funded concepts resolve **FUNDED**.
* **Population scope stays independent**: `acquired`/`direct`/named SPV never
  participate in dataset resolution, and dataset never participates in
  population resolution.
* A **disclaimed** tape word does not select (B21 preserved).
* No route economics change **except** where execution was previously using the
  wrong dataset — and every such case is reported by name, not hidden.
* Silent drops remain **0**.

## STOP conditions

| # | condition | label |
|---|---|---|
| B1 | the fix requires route-specific dataset rules | STOP — SEMANTIC OWNER NOT CONSOLIDATABLE |
| B2 | a planner or route must reread raw question text independently | STOP — SEMANTIC OWNER NOT CONSOLIDATABLE |
| B3 | a forecast case regresses to pipeline | STOP — BLAST RADIUS |
| B4 | population scope is used to determine dataset | STOP — BLAST RADIUS |
| B5 | interpretation semantics move on any axis other than `dataset` | STOP — BLAST RADIUS |
| B6 | a new second dataset owner is introduced | STOP — SEMANTIC OWNER NOT CONSOLIDATABLE |
| B7 | any 882-corpus movement falls outside M1–M3 | STOP — BLAST RADIUS |
| B8 | executable evidence does not support one deterministic rule | STOP — PRODUCTION SEMANTICS AMBIGUOUS |

## How the resolver's vocabulary will be chosen

From **existing production semantics as evidence**, not invented. The candidate
rules will be censused over the 882 distinct Stage 1 + Stage 2 corpus questions
**before** one is chosen, and the choice justified against the measured
movement. A rule that satisfies the target state's worked examples with the
**narrowest** movement wins; breadth is not a virtue here, because every extra
term is an unauthorised movement waiting to happen.

Specifically to be measured and reported, not assumed: whether
`mi_workflows.analytical.intent`'s governed `REQ_PIPELINE_DATASET` /
`REQ_FORECAST` requirements — which already express these concepts structurally
— can serve as the resolver, or whether their breadth (they exist to decide
*refusability*, and are checked **against** a dataset rather than selecting one)
makes them unsuitable.

## Out of scope

Do not resume C5. Do not fix the `comparison_period` structural gap. Do not add
subject/operation accessors. Do not change Direct/Acquired semantics or the
portfolio registry. Do not modify economic calculations. Do not enable T3–T7.
Do not add LLM logic. Do not redesign UI tabs — **this task changes semantic
ownership, not UI behaviour.**

**The user should not need to know which tab they are on to ask a correct MI
question. The question determines the dataset; the UI displays context.**
