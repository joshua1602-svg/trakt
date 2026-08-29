# Product correctness fix — `funded_bridge` missing-dimension safety (Defect B)

## Status

# DEFECT B FIXED

A bridge requested on a dimension the funded tape does not carry now returns
**unavailable**, instead of a valid-looking **£0 → £0 (net £0)** result for a
book that moved materially.

**Scope: Defect B only.** No C4 conversion, no dimension-role projection change,
no `metadata.groupedBy`, no `resolve_lens_with_default` change. One calculation
owner, two lines of real logic.

---

## 1. Base and HEAD

| | |
|---|---|
| branch | `claude/clause-splitting-scoping-38ahbz` |
| base | `de0edd0` — the C4 STOP report, working tree clean |
| C1/C2/C3 live through composition | confirmed — 27/27, 21/21, 12/12 plans built, 0 deferrals |
| C4 production changes | **none** (`git diff 5daf451 de0edd0` over the production tree is empty) |

## 2. The defect, reproduced at its owner

`evolution.funded_bridge`, called directly — below the routing and receipt
layers that currently mask it:

```
requested dimension : 'erm_product_type'
present on tape?    : False
returned available  : True          <- WRONG
returned reason     : None
dimensionCol chosen : 'erm_product_type'   <- a column that is not in the data
opening total       : 0
closing total       : 0
net movement        : 0
```

Proof the zero is not a legitimate zero-change result — the same book, bridged
on a **present** dimension over the same periods:

```
whole-book movement (via ltv_bucket):
  open  = 1,932,310,991.20
  close = 1,964,886,258.21
  net   =    32,575,267.01   (materially non-zero)
```

The book moved **+£32.6m**; the product bridge reported **£0 → £0, net £0**.

## 3. Root cause

```python
col = next((c for c in candidates if c in present_cols), candidates[0] if candidates else None)
```

The fallback `candidates[0]` selected the **first candidate even when no
candidate is present in the data**. `_group_balance(df, col)` returns `{}` when
`col not in df.columns`, so both sides of the bridge grouped nothing:
`open_total = sum({}.values()) = 0.0`, `close_total = 0.0`, `netChange = 0.0` —
and the function returned `available=True`.

The fallback defeated the function's own documented contract ("the first one
actually present in the data is used").

## 4. The fix — smallest owner-side change

`mi_agent_api/evolution.py`, **13 insertions / 2 deletions**, all inside
`funded_bridge`:

```python
col = next((c for c in candidates if c in present_cols), None)
if not col:
    if candidates:
        return {"available": False, "lens": lens_label,
                "reason": ("the requested attribution dimension is not "
                           "available in the funded data"),
                "requestedDimension": [str(c) for c in candidates]}
    return {"available": False, "lens": lens_label,
            "reason": "no attribution dimension is available in the funded data"}
```

* The `candidates[0]` fallback is removed, so an absent dimension routes into the
  existing **unavailable** result semantics rather than a zero-valued one.
* A distinct, machine-readable reason and `requestedDimension` list separate
  "the requested dimension is not on the tape" from the pre-existing "no
  dimension was requested" case — absence of a grouping request is not the same
  error as a grouping request that cannot be honoured.
* Fixed at the calculation owner only. No routing, receipt, interpretation,
  planning or narration code was touched.

## 5. Valid dimension — before/after **identical**

```
ltv_bucket (present):   available=True   dimensionCol='ltv_bucket'
  open=1,932,310,991.20  close=1,964,886,258.21  net=32,575,267.01  contributions=9
region family (list):   available=True   dimensionCol='collateral_geography'  net=32,575,267.01
```

Unchanged. The existing `test_funded_bridge_reconciles_to_net_change` and
`test_funded_bridge_picks_present_candidate_column` pass without modification.

## 6. Missing dimension — before/after

| | before | after |
|---|---|---|
| `available` | **True** | **False** |
| opening / closing / net | 0 / 0 / 0 | absent (no economic result) |
| `reason` | `None` | "the requested attribution dimension is not available in the funded data" |
| `requestedDimension` | — | `['erm_product_type']` |

## 7. Whole-book bridge — unchanged

An ungrouped bridge (no dimension named) still resolves the default dimension and
delivers. An **empty** dimension request keeps its own pre-existing message
(`"no attribution dimension is available in the funded data"`, no
`requestedDimension`), verified by `test_funded_bridge_no_dimension_requested_keeps_its_own_message`.
Absence of a grouping request is not turned into an error.

## 8. Caller inventory — no downstream fix required

Three production callers of `evolution.funded_bridge`:

| caller | how it takes the result | effect of the new unavailable |
|---|---|---|
| `chat_routing._route_bridge` | `if not br.get("available"): return _envelope(... reason ...)` | **converted to an honest refusal.** End-to-end, "by product" moves from a misleading LOST refusal ("this answer covers the whole population") to an accurate UNAVAILABLE refusal ("field is unavailable in this dataset") — still a refusal, no delivering case moves |
| `movement_summary.py` (period_movement regional attribution) | passes the region family (present); `if bridge.get("available")` else `contributions=[]` | **unchanged** — region is present, and it already degrades gracefully |
| `mi_agent_pptx/movement.py::_adapt` | `if not payload or not payload.get("available"): return MovementBridge(available=False, reason=payload reason)` | **already defended against this exact defect** with its own `0 → 0` workaround; now the owner supplies the honest reason too. Its workaround branch is left in place (out of scope for this fix) |

**No caller becomes unsafe.** No `STOP — DOWNSTREAM DEPENDENCY`.

## 9. Regression, by name

New: `mi_agent_api/tests/test_evolution.py` — 3 tests
(`test_funded_bridge_missing_dimension_fails_closed`,
`…_from_a_candidate_list_fails_closed`,
`…_no_dimension_requested_keeps_its_own_message`). **22 passed** in the file.

The missing-dimension test is **non-vacuous** by construction: it asserts the
dimension is genuinely absent from the frames, that a bridge on a present
dimension over the same frames moves **£600k → £800k**, and that the requested
absent dimension therefore returns unavailable — so the old £0 would have been a
wrong answer for a £200k move.

Attribution over `mi_agent_api` and the affected e2e/conversion suites, **by
exact name, base vs change** (same execution environment on both sides):

* **Introduced failing names: 0.**
* **Newly passing names: 0.**

The suite carries pre-existing `live`-fixture failures (they need the
development dataset, absent in plain `test` mode) and the pre-existing **Defect
A** refusals (a bridge that names a dimension is refused because the route does
not publish `metadata.groupedBy` — explicitly out of scope). Both sets are
identical at base `de0edd0` and after the fix. In particular
`test_funded_balance_bridge_returns_reconciling_waterfall` fails on Defect A, not
Defect B, at base and after.

## 10. Cost

| | |
|---|---|
| production lines changed | **13 / −2** in `mi_agent_api/evolution.py` (one owner) |
| — of which real logic | ~5 (the fallback change + the requested-but-absent branch); the rest is the explanatory comment |
| test lines added | 73 in `mi_agent_api/tests/test_evolution.py` |
| production modules changed | **1** |
| new primitives | 0 |

## 11. Whether the contract-role fix can proceed next

**Yes.** Defect B was the wrong number, and it was the one held closed by Defect
A's refusal — fixing it first means that when the contract-role fix and Defect A
are later addressed and the six dimension-naming bridge cases begin to deliver,
the underlying calculation is already truthful: a bridge on an absent dimension
will refuse honestly rather than deliver £0. The C4 sequencing in
`docs/mi_conversion4_stop_contract_prerequisite.md` §8 (fix B → close the
contract role prerequisite → fix A → then convert) can proceed from step 2.
