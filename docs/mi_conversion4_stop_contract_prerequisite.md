# Conversion 4 — `funded_bridge`

# STOP — CONTRACT PREREQUISITE

**No production code was changed.** `git diff 5daf451 HEAD` over
`mi_agent/ mi_agent_api/ question_interpretation/ trakt_core/ mi_workflows/` is
empty. Everything below is measurement.

Conversion 4 stopped at **§4 — dimension role preservation**. The interpretation
contract cannot today distinguish the bridge's *attribution* dimension from any
other dimension named in the same sentence, and §4 is explicit that this belongs
in the contract, not in route-specific logic.

Two live defects were found on the way, and **they cancel each other**. That is
why neither has been visible, and it is the reason this conversion must not
proceed: **converting the route removes the one that is hiding the other.**

---

## 1. Base and HEAD

| | |
|---|---|
| branch | `claude/clause-splitting-scoping-38ahbz` |
| HEAD | `5daf451` — Conversion 3 report, working tree clean |
| production changes made | **none** |

### All three prior conversions confirmed live through composition

Executed, not assumed — the plan builders were wrapped with counters and each
route's owned surface was run:

```
portfolio_summary    owned renders=27   other route=0   plans built=27
period_movement      owned renders=21   other route=0   plans built=21
geo_exposure         owned renders=12   other route=0   plans built=12
```

### Regression baseline

Unchanged from Conversion 3's close: shard A 7 failed / 1811 passed, shard B
12 failed / 1231 passed.

## 2. The candidate assumption held — §2 did **not** fire

`funded_bridge` needs **exactly one** currently-unbridged contract axis.

| what `_route_bridge` reads | contract axis | bridged today? |
|---|---|---|
| `_bridge_dimension(spec, semantics)` | **`dimensions`** | **no — this is the one** |
| `resolve_lens_with_default(question, default)` | `source_scope` | yes (C1/C2/C3) |
| `spec.compare_periods[0]` | `time.comparison_period` | the **axis** is bridged; **this field is not** |

The third row is worth stating plainly rather than glossing. `time` is a bridged
axis, but Conversion 2 bridged only `window_periods`. The bridge's *named start
period* lives in `time.comparison_period`, and the plan layer has never read it.
That is **additional shared work inside an already-bridged axis**, not a second
axis — so the registered thresholds remain the right ones, and the extra work
must be counted as shared cost when the conversion runs.

Verified that the contract does carry the fact:

```
"Bridge the funded balance by region since March 2026"
    spec.compare_periods            = ['March']
    contract time.comparison_period = state='filled' raw='March'
```

Also confirmed: **no new primitive** is required. `select_population`,
`resolve_measure`, `group` and `compare` all exist.

## 3. The one new bridge, named precisely

| | |
|---|---|
| contract axis | `dimensions` — `DimensionClaim(candidate_concept, role, raw_text)` |
| where the meaning is already owned | `mi_agent.llm_query_parser` → `spec.bridge_dimension`; the registry supplies the canonical column and business label |
| how the shipped route obtains it | `chat_routing._bridge_dimension(spec, semantics)`, which reads `spec.bridge_dimension` and falls back to `_BRIDGE_DEFAULT_DIMS` |
| where the compositional layer loses it | the plan layer reads **2 of the contract's 9 claim axes** — `source_scope` and `time`. `dimensions` reaches the contract and stops there |
| minimum generic bridge | carry `candidate_concept` **and its role** into a `GROUP` step, so the plan declares the axis it is attributed by |

The contract already carries the concept correctly:

```
"Funded balance bridge by region"      spec.bridge_dimension='collateral_geography'
                                       contract candidate_concept='collateral_geography'
"…by product"                          'erm_product_type'   /  'erm_product_type'
"…by LTV band"                         'ltv_bucket'         /  'ltv_bucket'
```

**The concept is carried. The role is not. That is the whole blocker.**

## 4. Dimension role preservation — **THIS IS THE STOP**

Every dimension on a bridge question arrives with `role='unresolved'`:

```
"Funded balance bridge by region"
    candidate_concept='collateral_geography'  role='unresolved'
    source='facet.grouping_dimension(role not supplied)'
```

That is the projection behaving **correctly**. Its role split is deliberately
conservative — `parser.dimension` → `grouping`, `parser.filters` → `filter`,
and where no source supplies a role it records `unresolved` **rather than
guessing**. `spec.bridge_dimension` is a third parser field that no role source
reads.

The consequence is not theoretical:

```
"Bridge the funded balance by region for joint borrowers"
    dim 1  candidate_concept='collateral_geography'  role='unresolved'
    dim 2  candidate_concept='borrower_type'         role='unresolved'
```

**Two dimensions, both unresolved, and nothing in the contract says which one
the waterfall is attributed by.** A plan that took "the dimension" — or the
first one — could attribute the bridge by *borrower type* while the user asked
for region, and declare that as its grouping. That is precisely the silent role
collapse §4 exists to prevent.

### Why this cannot be fixed in the route

The fix is to attribute `GROUPING` when a dimension came from the
`parser.bridge_dimension` source. That is a **contract-layer** change, in
`question_interpretation/projection.py`, one line of provenance in the role
split. §4 is explicit: *"Do not patch it inside route-specific logic."* Reading
`candidate_concept` while ignoring `role` inside the bridge's plan builder is
exactly that patch, and it would ship a plan asserting a role the contract
declined to resolve.

**STOP — CONTRACT PREREQUISITE.**

## 5. Two live defects, and they cancel

This is the part that makes proceeding unsafe rather than merely premature.

### Defect A — the route refuses its own reason for existing

`funded_bridge` never publishes `metadata.groupedBy`. The receipt's
`declared_group_fields` reads declarations only — *"a route that declares
nothing proves nothing"* — so `grouping_proven` fails and the named dimension is
marked **LOST**:

```
"Funded balance bridge by region"   ok=False
   facets: [('grouping_dimension', 'region', 'lost')]
   "I understood that you asked for region, but that could not be applied…
    this answer covers the whole population; it is neither narrowed to nor
    broken down by region"
```

…while the handler behind that refusal computed the bridge **by region**,
correctly. Measured over the frozen surface: **6 of 12 owned cases deliver, 6
refuse under every scope.** Every case that names a dimension refuses.

**Pre-existing.** Reproduced identically with production reverted to `42cef00`,
before any migration work. This programme did not cause it.

### Defect B — a wrong number, hidden behind Defect A

`evolution.funded_bridge` on a dimension the funded tape does not carry returns
`available=True` with zeros rather than declaring itself unavailable:

```
funded tape columns containing 'product' : NONE
funded_bridge(dim='erm_product_type')
    available : True        reason: None
    start     : 0
    end       : 0
    netChange : 0
    contributions: 0
```

The funded book moved **£1.93bn → £1.96bn**. The handler's answer is:

> "Product Type bridge (Total): funded balance moved from **£0** in 2026-04 to
> **£0** at 2026-06 (latest) — a net change of **+£0 (flat)**."

**This is not reachable today**, because Defect A refuses the question first.

### Why that combination forbids the conversion

A compositional plan **declares what it grouped by** — that is what `Plan.
declares_grouped_by` is for, and both converted plans publish it. So converting
`funded_bridge` naturally supplies the declaration that `grouping_proven` is
looking for.

```
convert the route  →  the plan declares its grouping
                   →  grouping_proven succeeds
                   →  Defect A's refusal disappears
                   →  Defect B's £0 becomes a delivered answer, ok=True
```

**The conversion would turn a false refusal into a confident wrong number.** It
would also violate the conversion's own terms — §7's *"expected economic
differences: 0"* and §11's *"no unexplained refusal movement"* — because six of
twelve owned cases would move from refusing to answering.

There is no honest way to convert around this. Building the bridge and
deliberately **not** declaring the grouping would keep equivalence, but it would
mean shipping a plan that knows its attribution axis while the answer continues
to tell the user that axis was lost — the machinery built and then hidden, and
§4's role preservation unprovable at execution.

## 6. `resolve_lens_with_default` — a third, weaker population owner

| | |
|---|---|
| what it owns | **precedence**: a scope named in the question beats the caller's dropdown, else the default, else Total |
| does the contract carry it | **yes** — `source_scope.provenance` (`explicit_user` / `caller_context` / `default` / `unresolved`), Phase 1G |
| production callers | `chat_routing._resolve_lens` (383), `_route_bridge` (1692), `_route_cohort_progression` (1794), `mi_agent_workflow` (669), `mi_service` (410) |

**`_route_bridge` calls it without the registry**, where `_resolve_lens` passes
`registry=` and then resolves the governed context. Measured divergence, 4 of 5
probes:

| question | governed path (`_resolve_lens`) | the bridge's path |
|---|---|---|
| "…for the acquired book" | `{'source_portfolio_id': ['alp_acquired']}` | `{'source_portfolio_type': 'acquired'}` |
| "…for the direct book" | `{'source_portfolio_id': ['alp_origination']}` | `{'source_portfolio_type': 'direct'}` |
| "…for the ALP Origination Book" | `cohort` → `['alp_origination']` | **`total` → `{}`** |
| "…for the Highgate Mortgages Book" | `unresolved` → refuse | **`total` → `{}`** |
| "funded balance bridge" | `total` | `total` |

The first two are the raw-type-vs-governed-id divergence Phase 1C measured at
£300 against £1,200 on a book holding two portfolios of one type. They select
the same rows on *this* book.

The last two are worse in principle — a named portfolio silently widening to the
whole book — but **the Phase 1E guard catches both**, because it is
route-independent and reads the question rather than the route's lens:

```
"Funded balance bridge for the ALP Origination Book"   ok=False
   "…this narrowing was not applied, so the figure covers…"
```

**So: a real second owner, currently held closed by a guard the route does not
own.** No live wrong number from it. Converting `funded_bridge` should make it
unreachable from this route and pin that with a test — but it must **not** be
retired globally, because `mi_agent_workflow` and `_route_cohort_progression`
still depend on it.

## 7. The frozen surface

`migration_phase0/route_ownership_funded_bridge.py` — 15 candidates × 3 caller
scopes, ownership from executed routing, plus what the handler itself computed
before the guard saw it.

| | |
|---|---|
| claimed by `funded_bridge` | **12** |
| of which deliver | **6** |
| of which refuse under every scope | **6** |
| deliberately not claimed, and verified | 3 |
| disagreeing with the declared expectation | **0** |

## 8. What must happen before Conversion 4 runs

Three pieces, in this order. **They must not be bundled into the conversion** —
each changes behaviour, and a migration that also changes answers cannot prove
equivalence.

1. **Fix Defect B first, on its own.** `evolution.funded_bridge` must return
   `available=False` with a reason when the requested dimension is not on the
   tape, instead of £0. This must land **before** anything removes the refusal
   that is currently hiding it. Its own regression: the six delivering cases
   must be unchanged, and "by product" must refuse *honestly* rather than
   silently.
2. **Close the contract prerequisite.** In `projection.py`'s role split,
   attribute `GROUPING` when the dimension came from `parser.bridge_dimension`.
   Check the blast radius first: confirm no question that routes elsewhere sets
   `spec.bridge_dimension`, because the projection is shared and a role change
   moves facets on every route that reads them.
3. **Then fix Defect A** — publish the declaration so `grouping_proven` can see
   it — as product work with its own before/after, since six of twelve owned
   cases move from refusing to answering.

Only then does Conversion 4 become the clean bridge-cost measurement it was
designed to be, against the thresholds already committed in
`docs/mi_conversion4_stop_conditions.md`, unchanged.

### Should C5 proceed instead?

**No.** Not because `funded_bridge` is a bad candidate — the §2 axis count held,
and it remains the right one — but because every remaining candidate needs a
bridge, and this stop is about the *contract's* ability to carry a role, which
is upstream of whichever route goes next. `evolution` and `temporal_compare`
need the `dataset` axis and would hit their own version of the same question:
does the contract carry the fact **with enough precision to act on**, or only
the fact?

## 9. Where the migration stands

```
compositional   3 of 15   portfolio_summary, period_movement, geo_exposure
specialist     12 of 15

contract axes bridged BEFORE C4 : 2 of 9   (source_scope, time.window_periods)
contract axes bridged AFTER  C4 : 2 of 9   — unchanged, C4 did not run
remaining unbridged             : operation, subject, dimensions, filters,
                                  target, population, dataset
                                  (+ time.comparison_period, an unbridged FIELD
                                   on a bridged axis)
```

The four-conversion cost sequence is **unchanged at three observations**:

| | shared | route-specific | hardening | cleanup | total | production commits |
|---|---|---|---|---|---|---|
| C1 `portfolio_summary` | 200 | 176 | 0 | 7 | 383 | 2 |
| C2 `period_movement` | 138 | 144 | 0 | 0 | 282 | 1 |
| C3 `geo_exposure` | 21 | 129 | 0 | 1 | 151 | 1 |
| **C4 `funded_bridge`** | — | — | — | — | **not run** | **0** |

**Nothing about bridge cost is claimed.** The only measured bridge in this
programme remains `span_from_claim` at 24 lines, and one observation is not a
rate.

## 10. Recommended next step

> **Fix Defect B alone — make `evolution.funded_bridge` declare itself
> unavailable for a dimension the tape does not carry — and ship it as product
> correctness with its own before/after, not as part of a conversion.**

It is first because it is the only one of the three that is a **wrong number**,
because it is currently held closed by a refusal that steps 2 and 3 will remove,
and because it is the smallest and most testable of the three.

Do not start it by converting anything.
