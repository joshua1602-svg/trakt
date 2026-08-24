# Defect A — `funded_bridge` grouping declaration — zero-blast conditions

**Committed before any production change.**

Base: `a290f30`. Working tree clean. Defect B (`a126e45`) and the contract-role
fix (`a290f30`) both present.

## The defect

`funded_bridge` computes a bridge by a requested dimension correctly, but never
publishes `metadata.groupedBy`. `declared_group_fields` reads declarations only —
*"a route that declares nothing gets no certification"* — so `grouping_proven`
returns False, the requested grouping facet is marked **LOST**, and the route
refuses an answer it computed correctly.

## Target state

> Execution must declare what it **actually grouped by**, and governance must
> prove the requested grouping from that declaration — not from route identity
> and not from rereading the question.

**Declare what was EXECUTED, not what was REQUESTED.** The distinction is the
whole safety property: if the question names a dimension the bridge did not use,
declaring the executed one leaves the request correctly unproven and the answer
correctly refused.

## Required outcome

* Only valid `funded_bridge` cases with an explicitly requested, **available**
  bridge dimension may move from refusal to delivery.
* The declaration must come from **executed** bridge semantics
  (`evolution.funded_bridge`'s own `dimensionCol`), never from raw question text.
* `metadata.groupedBy` must contain the canonical governed dimension **actually
  used**.
* Missing-dimension cases remain unavailable / refused (Defect B's guarantee).
* Non-dimension bridge cases remain unchanged.
* All non-`funded_bridge` routes remain unchanged.
* Economics remain identical — this exposes an already-correct calculation, it
  does not replace one.
* Silent drops remain 0.

## STOP conditions

| stop | when |
|---|---|
| **STOP — BLAST RADIUS** | any unrelated answer/refusal changes; any missing-dimension case becomes deliverable; any economic result moves |
| **STOP — GOVERNANCE MODEL INSUFFICIENT** | `metadata.groupedBy` cannot express truthful execution evidence without broader receipt changes |
| **STOP — CALCULATION/DECLARATION DISAGREEMENT** | the dimension execution used cannot be mapped deterministically to the governed declaration |

The fix must NOT:

* infer grouping from raw text;
* require route-name allowlisting (`ROUTE_DECLARED_AXES`) to prove grouping;
* declare a dimension execution did not actually use;
* change `grouping_proven`;
* add a `funded_bridge` exception inside receipt reconciliation.

The existing generic proof mechanism should start succeeding **because execution
now supplies the evidence it was designed to consume**.

## The precedent this follows

`risk_limits` already publishes into the same channel from executed evidence:

```python
fields_tested = risk_mod.tested_fields(tests)
if fields_tested:
    envelope.setdefault("metadata", {})["groupedBy"] = fields_tested
```

with the comment *"Derived from the tests that actually computed, so a limit
reported unavailable certifies nothing."* This task applies the identical
discipline to `funded_bridge`, using the same channel and the same idiom — so no
new metadata channel is introduced.

## No over-declaration

No declaration may be published when:

* the bridge returns **unavailable** for a missing dimension (Defect B);
* the bridge returns **unavailable** for any other reason (e.g. fewer than two
  governed reporting periods);

i.e. the declaration is published only on a **successful** grouped bridge, and
its value is execution's own `dimensionCol`. A requested dimension alone never
produces a declaration.
