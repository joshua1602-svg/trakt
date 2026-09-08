# Region: target state for scalability, lineage and capability

Asked 2026-09-04 against four goals: Risk Limits by Region; MI Query by Region;
Region stratifications, single and multi-variate; and an ITL/NUTS3 map where the
raw data carries postcodes.

**The headline: most of the target state already exists, in ingestion, and no
consumer reads it.** This is a wiring problem, not a modelling one — which is
good news for cost and bad news for how long it has been silently true.

## What already exists (and is unused)

`engine/region_taxonomy` is wired into the funded prep path
(`funded_prep.py:477`) and produces, per row:

| column | meaning |
|---|---|
| `canonical_region_detail` | the most granular governed value the source supports |
| `canonical_region_reporting` | the client-level consolidated value, equal to detail unless an approved taxonomy consolidates |
| `region_source_value` | the raw value as the source system wrote it |
| `region_mapping_method` | `exact` / `synonym` / `unresolved` / `absent` |
| `region_mapping_rule` | which governed rule mapped it |

…plus a report carrying `rows_resolved`, `rows_unresolved`, `unresolved_values`
and per-method counts. Mapping is deterministic (clean → exact → approved
synonym); the LLM is used only to PROPOSE mappings at onboarding, reviewed and
persisted into the synonym table, never at answer time.

That is a correct two-level model with full lineage and a coverage measure.

**Measured consumers of the three provenance columns: zero.** Nothing outside
the module that writes them reads `region_source_value`,
`region_mapping_method` or `region_mapping_rule`. Coverage is likewise unread.
Meanwhile five surfaces each choose a region column from their own hard-coded
list (see `mi_query_region_end_to_end_audit.md`).

## Target state

### 1. One concept, two governed levels, nothing else called Region

* `canonical_region_reporting` — **the** Region dimension. Risk Limits, MI
  queries and stratifications all mean this.
* `canonical_region_detail` — Region Detail, the drill level beneath it.
  Retained precisely so a consolidation is reversible.
* `collateral_geography` — the raw source column. Keeps its role as INPUT to the
  taxonomy; it should stop being a queryable "Region" in its own right once
  coverage is high enough to rely on the canonical pair.
* NUTS3 and ITL3 — a **different concept**, not another spelling of Region.
  They should carry their own `value_domain` (e.g. `uk_itl3`) so the alias
  binder cannot pool them with Region, and should be addressable only when a
  reader names them explicitly.

This is what makes goal 4 safe: the map keeps its fields, and no region question
can ever drift onto them.

### 2. One accessor, asked by every surface

Today each surface holds a list. Target: one governed accessor that answers,
for a given frame and purpose:

    (field, level, coverage, provenance_summary)

* MI axis + filter → reporting, detail on drill
* Stratifications (single and multi-variate) → the same accessor
* Funded bridge → the same accessor, and `_REGION_FAMILY` disappears
* Risk Limits → the same accessor, applying the owner's rule: **NUTS3 where
  universally available, `collateral_geography`/reporting where it is not** —
  with "universally available" answered from the COVERAGE REPORT rather than
  assumed
* Map → the ITL3 fields, gated on postcode coverage

Adding a region field then requires classifying it once, not editing five lists.
`tests/test_region_topology.py` already fails an unclassified field.

### 3. Coverage and provenance become part of the answer

The measured coverage on this estate is roughly 82.6% (acquired) and 90%
(direct). An answer computed over 82.6%-resolved geography is a **good** answer
if it says so, and an unreliable one if it does not. The receipt already names
the field it grouped on; it should also name the level and the resolved share,
and a limit evaluated on partial geography should say which basis it used.

That single change converts the residual risk from "wrong number" to "disclosed
number", which is the difference that matters commercially.

## Sequencing

| step | effect | blast |
|---|---|---|
| 1. Split the ITL3/NUTS3 `value_domain` off `uk_region` | goal 4 made safe; F2 closed | small, registry + one test |
| 2. Publish level + coverage on region answers | residual risk disclosed | small, receipt only |
| 3. One accessor; repoint MI, bridge, stratifications | F3 closed, five lists become one | medium |
| 4. Repoint Risk Limits through the accessor with the fallback rule | goal 1 correct on partial-NUTS3 books | **governed** — changes what limits measure; needs the approved-configuration owner |
| 5. Retire `collateral_geography` as a queryable Region once coverage allows | one field named Region | medium, gated on coverage |

Steps 1-3 are code and can be measured against the banks and the replay. Step 4
is a decision. Step 5 is a consequence of coverage, not of code.

## Is Region what stands between here and ~90%?

**No, on the measured evidence — and this matters for sequencing.**

The 115-question replay after PR #398 answered 87 (75.7%). Classifying the 27
failures by root cause:

| cluster | count | region-related? |
|---|---|---|
| A field absent from the executed frame | 4 | 3 were region; all three now answer |
| B `period_change` claims it, then refuses (no subject/period) | 5 | no |
| C unmapped — no governed route owns the phrasing | 4 | no |
| D vocabulary (`funded`, `withdrawals`, `amount`, …) | 6 | no — **shipped in #400** |
| E narrowing stated but not applied | 4 | no |
| F comparison period not applied | 2 | no |
| G forward projection genuinely absent | 1 | no (correct refusal) |
| Z pipeline movement needs two governed periods | 1 | no |

By keyword the failures are pipeline/stage **18**, funded 6, region 6. The
dominant blocker is stage-movement recognition and period-change routing, not
geography.

Arithmetically: 87 today, +6 from cluster D already deployed → ~93 (81%).
Reaching ~104/115 (90%) needs roughly 11 more, and the available pool is
B (5) + C (4) + E (4) + F (2) = 15 — all stage/movement and period routing.

So the region work is worth doing for **lineage, consistency and go-live
defensibility** — a limit and an MI answer disagreeing about "Region" is a
credibility failure whatever the hit rate — but it is not the path to 90%. The
path to 90% runs through cluster B/C: questions the governed capability can
already compute and no recogniser claims.
