# Contract-role fix — `funded_bridge` dimension role — zero-blast conditions

**Committed before any production change.**

Base: `a126e45` (Defect B fixed). Working tree clean.

## The prerequisite

Discovered during Conversion 4: on a bridge question the parser already knows the
attribution dimension (`spec.bridge_dimension`), but the interpretation contract
projects that dimension's claim with `role='unresolved'`. This task carries the
already-known governed fact into the contract, and nothing else.

**This is a projection fix, not a parser redesign. The system learns nothing new
about bridge language.** `projection.py` should project the parser's existing
decision, not reinterpret the question.

## Required outcome

* Only bridge questions with an already-populated `spec.bridge_dimension` may
  change interpretation output.
* Only the dimension claim that matches `spec.bridge_dimension` by governed field
  identity may change role.
* All unrelated claims remain equivalent.
* Non-bridge questions remain unchanged.
* Routing outcomes remain unchanged.
* Answer / refusal outcomes remain unchanged.
* Economics remain unchanged.
* Silent drops remain 0.

## STOP conditions

| stop | when |
|---|---|
| **STOP — BLAST RADIUS** | any non-bridge interpretation changes; more than one dimension role changes where only one `bridge_dimension` exists; any unrelated claim moves |
| **STOP — CONTRACT MODEL INSUFFICIENT** | the existing contract has no suitable role value for a bridge attribution axis (i.e. `GROUPING` is not the right existing role) |
| **STOP — OWNERSHIP AMBIGUOUS** | `spec.bridge_dimension` cannot be mapped deterministically to exactly one governed claim |
| **STOP — UNEXPECTED BEHAVIOUR MOVEMENT** | this fix alone makes a currently-refusing bridge answer deliver, or moves any economics or receipt verdict |

The fix must NOT:

* reread raw question text;
* infer a role rather than derive it from an existing parser field;
* add regexes, phrase lists, raw-question checks, route-name checks, or
  `funded_bridge`-specific parsing;
* introduce a new semantic owner.

## Why the blast radius is structurally zero on production behaviour

Traced before implementing:

* The projected `QuestionInterpretation` is consumed in production **only** by the
  three converted routes (`portfolio_summary`, `period_movement`,
  `geo_exposure`) via `RouteRequest.resolve_interpretation()`. **None reads
  `qi.dimensions[].role`** — they read `source_scope` and `time`.
* `funded_bridge` is **not** converted; `_route_bridge` reads `spec` directly, not
  the projected contract. So the bridge route never sees this field.
* Inside `projection.py`, the only reader of `DimensionClaim.role` is `_filters`,
  which reads `role == FILTER` to build `claimed_fields`. This fix promotes an
  `unresolved` claim to `GROUPING` and never creates or removes a `FILTER` claim,
  so `claimed_fields` is unchanged.

The consequence: this change makes the contract faithful for a future C4 to
consume, and cannot move any current client-visible outcome. The census (§8) and
the production-equivalence check (§9) prove it empirically rather than by
assertion.

## Defect A stays unfixed by design

`metadata.groupedBy` is deliberately not published in this task. Therefore the
dimension-naming bridge cases that refuse today (via the receipt's grouping
proof, which reads route declarations — not this contract field) must **still
refuse** after this fix. A bridge answer becoming newly deliverable is
`STOP — UNEXPECTED BEHAVIOUR MOVEMENT`.
