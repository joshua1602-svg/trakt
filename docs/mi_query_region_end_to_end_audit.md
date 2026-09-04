# "Region", end to end — what each surface means by it

Audit requested 2026-09-04, after three region questions that refused in the
115-question replay were found answering on the deployed build. The concern is
right: the symptom is gone and the **journey is not remediated**, so it can come
back through any of five owners.

Everything below is read from the code and the registry, not recalled. The
topology is pinned by `tests/test_region_topology.py`, which asserts no
behaviour — it records these couplings so a change to any of them fails loudly.

## Three families, one word

| family | fields | business name |
|---|---|---|
| **reporting** | `canonical_region_reporting`, `canonical_region_detail`, `collateral_geography` | "Region" / "Region Detail" |
| **NUTS3** | `geographic_region_obligor`, `geographic_region_collateral` | "Obligor/Collateral Region (NUTS3)" |
| **ITL3** | `geographic_region_obligor_itl3`, `geographic_region_collateral_itl3` | "Obligor/Collateral ITL3" |

All seven declare `value_domain: uk_region`.

## Five owners, each with its own answer

| # | surface | what it treats as "region" | where |
|---|---|---|---|
| 1 | Ingestion | writes `canonical_region_detail` + `canonical_region_reporting` **from `collateral_geography`** | `engine/region_taxonomy.py` |
| 2 | MI axis + filter | `_REGION_PREFERENCE`: reporting → detail → raw → collateral NUTS3 → obligor NUTS3, **data-aware**; `_REGION_DEFAULT = collateral_geography` when columns are unknown | `mi_agent/llm_query_parser.py` |
| 3 | Funded bridge | `_REGION_FAMILY`, its own list | `mi_agent_api/chat_routing.py` |
| 4 | **Risk Limits (Schedule 8)** | **`geographic_region_obligor`** — the NUTS3 field, hard-coded in the category rules | `mi_agent/risk_monitor/schedule8_extractor.py:46` |
| 5 | Exposure map | `_ITL3_FIELDS` — the ITL3 pair, plus `uk_itl_master_lookup_v2.csv` postcode → ITL3 | `mi_agent_api/geo.py:31` |

## Findings, ranked

### F1 — Risk Limits and MI answer "region" on different fields (material)

The limit monitor evaluates geographic concentration on **NUTS3**; MI answers on
the **reporting** family. Same word on two dashboard surfaces, two different
groupings of the same book. A breach reported against one and an MI answer given
on the other can disagree, and a reader has no way to see why — the MI receipt
names its field in `executionSummary.facets`, the limit test names a test label.

Nothing reconciles them. This is the end-to-end gap the audit was asked for.

### F2 — `value_domain` now carries two meanings (latent)

Since the aliasing fix, `value_domain: uk_region` says both *"this vocabulary
resolves the value"* (`region_resolution`) and *"these fields are the same
concept"* for filter binding. The second is true **within** a family and false
across them — `Region` and `Collateral ITL3` are not the same concept.

Measured today, the pooling is safe, and only by accident of ordering:

    reporting + obligor NUTS3  -> canonical_region_reporting
    reporting + obligor ITL3   -> canonical_region_reporting
    NUTS3 pair only            -> geographic_region_collateral
    ITL3 pair only             -> None   (disclosed, not bound)

The last line is the safety, and it holds **only because ITL3 is absent from
`_REGION_PREFERENCE`**. Nothing declares that; editing the preference order
would silently make an ITL3 field bindable as "region". Now asserted.

### F3 — Five lists, no single owner (recurring)

Adding a region field means updating an unknown number of the five. This has
already bitten once: the harmonised columns were registered on 2026-09-03 and
the funded bridge kept its own idea of the family, so "Show movement by region."
answered before the change and refused after. F1 is the same shape, unfixed.

### F4 — `_REGION_DEFAULT` can name a column the book lacks (low)

With no column context the parser returns `collateral_geography`. That is the
right instinct — never reach for a harmonised column that may not exist — but on
a book that carries only the canonical pair it names a field the frame has not
got. Deliberate and documented ("A preference is not a default"); recorded here
because it is a live path into "'Region' is not available in this dataset".

### F5 — Two geographies, unreconciled (known)

The map aggregates ITL3 areas derived from postcodes; MI queries answer on ITL1
reporting regions. Already in the handover; restated because F1 makes it three
systems, not two.

## What is now guarded

`tests/test_region_topology.py` pins: every `uk_region` field is classified into
exactly one family; `_REGION_PREFERENCE` contains no ITL3 field; an ITL3-only
claim resolves to `None`; Risk Limits reads the NUTS3 field; the map reads only
ITL3; the funded bridge knows the reporting family.

A new region field, or a repointed surface, fails that file.

## Remediation options for F1

1. **Repoint Risk Limits to the reporting family** — one line in the category
   rules. Cheapest, and changes what a limit test measures, so it needs the
   approved-configuration owner's sign-off and a re-evaluation of every
   geographic limit against the new grouping. Blast: every geographic limit.
2. **Declare the field on both surfaces** — leave the calculation alone and make
   each answer name the field it grouped on, so the disagreement is visible
   rather than silent. Cheap, no behaviour change, does not fix the divergence.
3. **One region owner** — a single governed accessor every surface asks, with
   the family as a parameter. Correct, largest blast, and it subsumes F2-F4.

Recommended: 2 now (it costs nothing and stops the silent case), then 1 or 3 as
a governed decision. Not taken here — repointing a limit is not a call to make
from a code reading.

## Not audited

The React geography view's own aggregation, and whether the approved
concentration-test configuration (`SOURCE_APPROVED`) carries its own field
choice — it is operator-approved YAML outside this repo's fixtures, so what it
names could not be read here.
