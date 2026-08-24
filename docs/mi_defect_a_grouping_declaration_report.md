# Defect A — `funded_bridge` grouping declaration — report

## Status

# DEFECT A FIXED — ZERO BLAST

`funded_bridge` now declares the axis it **actually attributed by**, and the
existing generic `grouping_proven` certifies the requested grouping from that
declaration. Nine renders across three cases moved refusal → delivery, all
inside the pre-registered authorised class, with economics identical to the
already-correct underlying calculation.

**Only the valid, available, explicitly-requested dimension bridges moved.**
Missing-dimension cases still refuse. Scope refusals still refuse. Silent drops
still 0.

---

## 1. Base and HEAD

| | |
|---|---|
| branch | `claude/clause-splitting-scoping-38ahbz` |
| base | `a290f30`, working tree clean |
| Defect B (`a126e45`) | present |
| contract-role fix (`a290f30`) | present |
| C1/C2/C3 live through composition | confirmed |
| C4 production conversion | none |
| zero-blast conditions committed **before** production change | `cd102ae` |

## 2. Defect A, reproduced

`"Bridge the funded balance by region"`:

```
spec.bridge_dimension            = 'collateral_geography'
projected claim                  = collateral_geography role='grouping'
                                   source='parser.bridge_dimension'
_bridge_dimension resolves       = key='collateral_geography'
                                   candidates=[collateral_geography,
                                               geographic_region_collateral,
                                               geographic_region_obligor]

CALCULATION (evolution.funded_bridge)
    available     = True
    dimensionCol  = 'collateral_geography'      <- what execution ACTUALLY grouped by
    open -> close = 1,932,310,991.20 -> 1,964,886,258.21
    netChange     = 32,575,267.01
    contributions = 9

ROUTE ENVELOPE
    ok = True    metadata.groupedBy = None      <- MISSING
    declared_group_fields(...) = {'category','end','start','label','type'}
        (artifact column names — they match no registry field and prove nothing)

END TO END
    ok = False   facets = [('grouping_dimension','region','lost')]
    "…region could not be applied…neither narrowed to nor broken down by region"
```

**The calculation was correct and the refusal was caused purely by the missing
execution declaration.** The waterfall had attributed by exactly the dimension
the answer claimed it had lost.

## 3. Declaration-owner trace

`declared_group_fields` reads, in order: the concentration workflow's own
results, **`metadata.groupedBy`**, `rankedMovement.canonicalField`, and
`ROUTE_DECLARED_AXES`. Its contract is explicit — *"An empty set means the answer
proves no breakdown… a route that declares nothing gets no certification."*

The narrowest point where execution knows **both** the resolved canonical column
**and** that the grouped bridge succeeded is `_route_bridge`, immediately after
the `br.get("available")` guard: `evolution.funded_bridge` already returns
`dimensionCol` — its own report of the candidate it found present in the data.
The calculation was already declaring it; the route simply never forwarded it.

**Precedent followed exactly.** `risk_limits` publishes into the same channel
from executed evidence:

```python
fields_tested = risk_mod.tested_fields(tests)
if fields_tested:
    envelope.setdefault("metadata", {})["groupedBy"] = fields_tested
```

*"Derived from the tests that actually computed, so a limit reported unavailable
certifies nothing."* No new metadata channel; `grouping_proven` untouched; no
`funded_bridge` exception in receipt reconciliation; `funded_bridge` remains
**absent** from `ROUTE_DECLARED_AXES` (asserted by test), so the certification
stands on the declaration and not on route identity.

## 4. The production change

`mi_agent_api/chat_routing.py`, **+22 / −3** (of which ~4 are logic, the rest the
explanatory comment), in `_route_bridge`:

```python
executed_dim = br.get("dimensionCol")
if executed_dim:
    envelope.setdefault("metadata", {})["groupedBy"] = [str(executed_dim)]
```

**Declare what was executed, not what was requested.** `dimensionCol` is the
candidate execution found present — not the list it was offered, and not the
dimension the question asked for. That is the safety property: a question naming
a dimension the bridge could not use leaves its request correctly unproven and
correctly refused.

## 5. Positive tests

`tests/test_funded_bridge_grouping_declaration.py` — **11 passed**.

| test | asserts |
|---|---|
| bridge by region | `groupedBy == ['collateral_geography']`, facet **applied**, delivers, £1.93bn→£1.96bn |
| bridge by LTV band | `groupedBy == ['ltv_bucket']` — the **exact executed** axis, not a fixed route axis |
| scoped bridge | same axis declared, £568.3m acquired figures |
| the generic proof certifies it | `funded_bridge` absent from `ROUTE_DECLARED_AXES`; `declared_group_fields` contains the executed field |
| grouping/filter roles do not collapse | "by region for joint borrowers" → region applied; borrower type judged separately, never swept into the declaration |

## 6. No-over-declaration tests

| test | asserts |
|---|---|
| missing dimension | `groupedBy is None`, still refuses, no `£0` in the answer — **Defect B's gate** |
| a requested dimension alone never certifies | `erm_product_type` absent from `declared_group_fields` |
| an unavailable bridge declares nothing | below the route: a one-period bridge returns `available=False` with **no `dimensionCol`** |
| whole-book bridges unchanged | both default-dimension cases still deliver, same figures |
| unheld portfolio name | still refuses — a scope refusal is not a grouping refusal |

## 7. Refusal → delivery census

Full owned surface, 15 cases × 3 scopes:

| | |
|---|---|
| renders compared | **45** |
| verdict changes | **9** — all refusal → delivery |
| cases affected | **D1, D3, D4** |
| ROUTE changes | **0** |
| answer-text moves at unchanged verdict | **0** |

| case | question | before | after |
|---|---|---|---|
| D1 ×3 scopes | Funded balance bridge by region | LOST → refuse | `['collateral_geography']`, applied, delivers |
| D3 ×3 scopes | balance bridge by LTV band | LOST → refuse | `['ltv_bucket']`, applied, delivers |
| D4 ×3 scopes | Bridge …by region since March 2026 | LOST → refuse | `['collateral_geography']`, applied, delivers |
| **D2 ×3** | Bridge the funded balance by product | refuse | **still refuses**, `groupedBy=None` |
| **S3, R1** | unheld / unresolved portfolio names | refuse | **still refuse** (scope, not grouping) |

### One further movement, found outside the ownership surface and verified

The A5 surfaces moved, so the **entire 29-question time-series corpus** was run
before and after:

```
questions compared : 29
verdict/route moves: 1
  'period on period movement by LTV band'
     before: route=funded_bridge ok=False
     after : route=funded_bridge ok=True groupedBy=['ltv_bucket']
```

That case is `funded_bridge`, names LTV band explicitly, and `ltv_bucket` is on
the tape — **inside the authorised class**, with both facets applied
(`comparison_period` and `grouping_dimension`). It was simply not among the 15
phrasings the ownership instrument declares. It is recorded here rather than
absorbed silently.

Consequent A5 shifts, all attributable to that one question:

| surface | before | after |
|---|---|---|
| recognition — DELIVERED | 15 (25%) | **17 (28%)** — 2 rows, one question run twice |
| recognition — T6 delivered | 0 | **2** |
| time-series — honest refusals | 20 of 29 | **19 of 29** |
| time-series — T6 rating | ABSENT | **ABSENT (unchanged)** |
| **SILENT DROPS** | **0** | **0** |
| robustness / shipped shapes / routed surface | — | **byte-identical** |

## 8. Economic equivalence for the newly delivered answers

Each newly delivered render compared against the underlying calculation, run
independently:

```
OK  Funded balance bridge by region      None      collateral_geography  £1.93bn->£1.96bn net £32.6m  contribs=9
OK  Funded balance bridge by region      direct    collateral_geography  £1.36bn->£1.39bn net £21.5m  contribs=9
OK  Funded balance bridge by region      acquired  collateral_geography  £568.3m->£579.4m net £11.1m  contribs=9
OK  balance bridge by LTV band           None      ltv_bucket            £1.93bn->£1.96bn net £32.6m  contribs=9
OK  balance bridge by LTV band           direct    ltv_bucket            £1.36bn->£1.39bn net £21.5m  contribs=8
OK  balance bridge by LTV band           acquired  ltv_bucket            £568.3m->£579.4m net £11.1m  contribs=9
OK  Bridge …by region since March 2026   None      collateral_geography  £1.93bn->£1.96bn net £32.6m  contribs=9
OK  Bridge …by region since March 2026   direct    collateral_geography  £1.36bn->£1.39bn net £21.5m  contribs=9
OK  Bridge …by region since March 2026   acquired  collateral_geography  £568.3m->£579.4m net £11.1m  contribs=9

mismatches: 0 of 9
```

Opening, closing, net movement, contribution count, selected population and
grouping dimension all identical. **No calculation was introduced or replaced —
this exposes an already-correct one.**

## 9. Defect B safety gate

Re-run: `"Bridge the funded balance by product"` (the tape carries no product
type) — under every scope:

* `available = False` at the owner;
* `metadata.groupedBy = None` — **no declaration published**;
* `grouping_proven` false — no delivery;
* the answer refuses, and contains **no `£0`** bridge.

The two fixes compose as intended: Defect B keeps the calculation truthful, and
this declaration cannot route around it because the declaration is execution
evidence.

## 10. Regression, by name

* `tests/test_funded_bridge_grouping_declaration.py` — 11 passed (new).
* `mi_agent/tests/`, `mi_agent_api/`, `question_interpretation/tests/`, and the
  C1/C2/C3 conversion guards — no introduced failing names; the residual
  failures are the pre-existing baseline set (live-fixture cases, the time-axis
  wording case, and the `mi_agent_api` set carried since C3).
* A5 surfaces: robustness, shipped shapes and routed surface **byte-identical**;
  recognition and time-series moved only by the single authorised question in §7;
  **silent drops 0**.

**Introduced failing names: 0. Silent drops: 0. Route movement: 0. Economic
movement outside the authorised refusal→delivery set: 0.**

## 11. One pre-existing inconsistency, observed and left alone

On the **default-dimension** bridge (no dimension named), the prose label and the
executed column come from different sources:

```
'funded balance bridge'
   _bridge_dimension key = 'geographic_region_obligor'  -> prose "Obligor Region (NUTS3)"
   execution grouped by  = 'collateral_geography'
```

`_bridge_dimension` derives the label from the registry key while execution picks
the first present candidate. This is **pre-existing and independent of this
change** (verified by running both sides), it is deterministic, and the
declaration reports the truth of what executed. Fixing the prose label is
unrelated semantics and out of scope — recorded here so it is not lost.

## 12. Cost

| | |
|---|---|
| production lines changed | **+22 / −3** in `mi_agent_api/chat_routing.py` (~4 logic) |
| production modules changed | **1** |
| new metadata channel | **none** — reuses `metadata.groupedBy` |
| `grouping_proven` changed | **no** |
| route allowlist entries added | **none** |
| test files added | 1 (11 tests) |

## 13. Can C4 now be retried?

**Yes.** The three prerequisites the C4 STOP named are now closed:

| | |
|---|---|
| Defect B — the calculation | fixed (`a126e45`): an absent dimension fails closed |
| the contract role | fixed (`a290f30`): the bridge dimension carries `grouping` |
| Defect A — the declaration | **fixed here**: execution declares the axis it used |

The condition that forced the STOP — that converting `funded_bridge` would
remove a false refusal and thereby expose a £0 wrong number — no longer holds.
The route now delivers its valid dimension bridges and refuses the rest honestly,
so a conversion can be measured against a stable, truthful baseline.

**Two things to carry into a retried C4**, both unchanged from the STOP report:
`resolve_lens_with_default` is still a third population entry point that diverges
from the governed path on 4 of 5 probes (held closed by the Phase 1E guard), and
the `dimensions` axis is still unbridged into the plan layer — which remains the
one new bridge C4 was pre-registered to measure, at shared ≤ 45.
