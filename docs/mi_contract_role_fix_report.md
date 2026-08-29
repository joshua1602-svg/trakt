# Contract-role fix — `funded_bridge` dimension role — report

## Status

# CONTRACT ROLE FIXED — ZERO BLAST

The interpretation contract now projects the parser's already-known bridge
attribution dimension (`spec.bridge_dimension`) with the `grouping` role instead
of `unresolved`. Nothing else moved: not another claim's role, not a non-bridge
question, not a route, an answer, a refusal, or an economic figure.

**Projection, not reinterpretation.** No raw question text is read, no phrase
list or route-name check is added, and the trigger is an existing parser field.

---

## 1. Base and HEAD

| | |
|---|---|
| branch | `claude/clause-splitting-scoping-38ahbz` |
| base | `a126e45` — Defect B fixed, working tree clean |
| Defect B commit present | yes (`a126e45`) |
| C1/C2/C3 live through composition | confirmed |
| C4 production conversion | none |
| zero-blast conditions committed **before** production change | `251ca88` |

## 2. The defect, reproduced

On a bridge question the parser resolves the attribution axis into
`spec.bridge_dimension`, but the projection emitted the matching claim as
`unresolved`:

```
"Funded balance bridge by region"
    spec.bridge_dimension = 'collateral_geography'
    claim collateral_geography  role='unresolved'  source='facet.grouping_dimension(role not supplied)'

"Bridge the funded balance by region for joint borrowers"
    spec.bridge_dimension = 'collateral_geography'
    claim collateral_geography  role='unresolved'
    claim borrower_type         role='unresolved'
```

The parser already knew `collateral_geography` was the bridge dimension; the
projection dropped that fact. And `spec.bridge_dimension` is populated **only**
on bridge questions — it is `None` on "Show funded balance by region", "Show
funded balance over time by region", and "Balance by region and ticket size" —
so its presence is the parser's own bridge signal, not an inference.

## 3. Semantic-owner trace

| | |
|---|---|
| where `spec.bridge_dimension` is populated | the parser (`mi_agent.llm_query_parser`), for bridge intent only |
| where dimension claims are projected | `question_interpretation/projection.py::_dimensions` (the role split) |
| where the role split emitted `unresolved` | the `else` branch — the split read `spec.dimension(s)` and `spec.filters`, never `spec.bridge_dimension` |
| consumers of `DimensionClaim.role` | `projection._filters` (reads `role == FILTER`); `schema.dimensions_with_role` (read-only view); **no production routing/receipt/answer path reads it for bridge questions** |

The projected `QuestionInterpretation` is consumed in production only by the
three converted routes via `resolve_interpretation()`, none of which reads
dimension role; `funded_bridge` is unconverted and reads `spec` directly. So the
change is structurally incapable of moving a live bridge outcome — proven
empirically in §8 and §9.

## 4. The change — narrowest possible rule

`question_interpretation/projection.py`, **+10 / −0**, inside `_dimensions`:

```python
bridge_dim = getattr(spec, "bridge_dimension", None)
...
    if key in parser_groups:
        role, src = GROUPING, "parser.dimension"
    elif bridge_dim is not None and key == bridge_dim:
        role, src = GROUPING, "parser.bridge_dimension"
    elif key in parser_filters:
        role, src = FILTER, "parser.filters"
    else:
        role, src = UNRESOLVED_ROLE, "facet.grouping_dimension(role not supplied)"
```

* The trigger is the existing parser field `spec.bridge_dimension`.
* The match is governed-key to governed-key (`key == bridge_dim`), never a
  wording-string comparison.
* It fires for exactly the one claim whose key equals `spec.bridge_dimension`.
* The role assigned is the **existing** `GROUPING` value — the bridge waterfall
  *is* a grouping/attribution — so no schema change and no
  `STOP — CONTRACT MODEL INSUFFICIENT`.
* A distinct source `parser.bridge_dimension` records the provenance faithfully
  (Correction 5's discipline: a role must say where it came from).

## 5. Ownership is deterministic

`spec.bridge_dimension` is a single governed field key, and it matched exactly
one projected claim in every case (`collateral_geography`, `erm_product_type`,
`ltv_bucket`). No alias ambiguity arose, so no `STOP — OWNERSHIP AMBIGUOUS`.

## 6. Positive tests

`question_interpretation/tests/test_bridge_dimension_role.py` — 8 tests, **8
passed**.

| test | asserts |
|---|---|
| bridge by region / product / LTV band | the bridge dimension carries `(grouping, parser.bridge_dimension)` |
| only the bridge dimension moves | "by region for joint borrowers" → region grouping, **borrower_type stays unresolved** |

## 7. Negative / zero-blast tests

| test | asserts |
|---|---|
| ordinary grouping ("Show funded balance by region") | unchanged — `(grouping, parser.dimension)` |
| trend + grouping ("…over time by region") | stays `unresolved` — not promoted merely for containing a dimension |
| two dimensions ("by region and ticket size") | both unchanged; no role guessed from ordering |
| bridge with no resolved dimension ("funded balance bridge") | nothing promoted — absence is not turned into inference |

## 8. Before/after interpretation census

`migration_phase0/contract_role_census.py` — projects every question in the
calibration + question corpora (plus five explicit bridge probes) at base and at
HEAD, and diffs the projected dimension roles.

```
questions compared          : 645
questions with any delta    : 5
ILLEGAL deltas (blast)      : 0
```

Every one of the 5 deltas is a bridge question with a populated
`spec.bridge_dimension` whose matching claim moved `unresolved → grouping`:

| question | bridgeDimension | roles moved |
|---|---|---|
| Funded balance bridge by region | collateral_geography | collateral_geography: unresolved → grouping |
| Bridge the funded balance by product | erm_product_type | erm_product_type: unresolved → grouping |
| balance bridge by LTV band | ltv_bucket | ltv_bucket: unresolved → grouping |
| Bridge …by region for joint borrowers | collateral_geography | collateral_geography: unresolved → grouping (borrower_type unchanged) |
| Bridge …by region since March 2026 | collateral_geography | collateral_geography: unresolved → grouping |

**Nothing outside that set moved** — no filter, no `bridgeDimension` field, no
non-bridge dimension role, across 640 corpus questions.

## 9. Production-behaviour equivalence

`migration_phase0/route_ownership_funded_bridge.py`, re-run end-to-end:

| | |
|---|---|
| owned cases | 12 (unchanged) |
| deliver | **6** (unchanged) |
| refuse under every scope | **6** (unchanged) |
| route/ok differences vs the C4 capture, across 15 cases × 3 scopes | **0** |

The dimension-naming bridge cases that refuse today (via Defect A — the receipt's
grouping proof reads route declarations, not this contract field) **still
refuse**. No answer became newly deliverable. No `STOP — UNEXPECTED BEHAVIOUR
MOVEMENT`.

## 10. Regression, by name

* `question_interpretation/tests/` (whole suite): **589 passed, 1 failed** — the
  one failure is `test_the_wording_that_asked_is_returned[balance by each month]`,
  a pre-existing time-axis baseline failure, verified failing identically with
  `projection.py` reverted to base and present in the C3-era baseline. Unrelated
  to dimensions.
* Receipt role suites (`test_stage4_role_split`, `test_dimension_role_owner`,
  `test_d2_routed_role_carriage`), `mi_agent/tests/`, and the C1/C2/C3 conversion
  guards: run and confirmed.
* The five A5 surfaces (robustness, shipped shapes, routed surface, recognition,
  time-series): **byte-identical** to the C3 capture — **silent drops 0**.

**Introduced failing names: 0. Silent drops: 0. No route/answer/refusal/economic
movement.**

## 11. Cost

| | |
|---|---|
| production lines changed | **+10 / −0** in `question_interpretation/projection.py` (one owner) |
| production modules changed | **1** |
| new semantic owner | **none** — `spec.bridge_dimension` remains the owner; the projection projects it |
| new primitives / bridges / schema changes | 0 |
| test files added | 1 (`test_bridge_dimension_role.py`, 8 tests) |
| instruments added | 1 (`contract_role_census.py`) |

## 12. Whether Defect A can proceed next

**Yes.** The contract now faithfully carries the bridge grouping role, which is
the input a fixed Defect A (publishing `metadata.groupedBy` / the plan's declared
grouping) needs to prove the bridge grouped by that dimension. With Defect B
already making the underlying calculation truthful and this fix making the
contract truthful, Defect A is the remaining step before the dimension-naming
bridge cases can deliver — and it can be undertaken as its own change with its
own before/after, since it is the one that will move six owned cases from
refusing to answering.
