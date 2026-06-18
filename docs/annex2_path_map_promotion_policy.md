# Annex 2 path-map promotion policy

How a field-to-XSD path mapping moves from "evidence we have" to "path we will
trust in the production XML builder" — **without** generating production XML or
weakening any production gate.

> **Keep the map. Retire the runtime.** The ESMA mapping workbook (read by the
> legacy Gate 5 builder) gave us XSD-validated paths worth keeping. The legacy
> builder's *runtime behaviour* (silent ND5 injection, value fabrication, wide
> one-row-per-loan, flattened collateral) stays retired. See
> `docs/legacy_gate5_annex2_xml_builder_review.md`.

## Two independent axes

Production XML for a field needs **both**:

1. **Path readiness** — do we know, and trust, *where* the value goes in the XSD
   tree? (this policy)
2. **Data readiness** — do we actually *have* the real client/operator/config/
   source value? (the Delivery/XML Agent; not certified here)

> A field can have a production-eligible XML **path** and still be **blocked by
> missing data**. `production_ready` is therefore `false` for **all 107 fields**
> today: no data is certified, the workbook paths are not yet formally accepted,
> and the vendored XSD is still the DRAFT.

## Promotion-status vocabulary (the path axis)

| `promotion_status` | meaning | path production-eligible? | builder-eligible behind a structure gate? |
|---|---|---|---|
| `confirmed_by_xsd_sample` | XSD **and** the sample message agree | **yes** (path only) | yes |
| `workbook_xsd_validated` | ESMA workbook path, re-validated against the XSD tree | **no** — needs formal acceptance | yes |
| `manual_review_required` | workbook touched the field but no clean XSD path | no | no |
| `unresolved` | no XSD/workbook/sample evidence | no | no |
| `conflict` | collision / multi-code-cell pollution | no | no |

`workbook_xsd_validated` is deliberately **not** the same as
`confirmed_by_xsd_sample`. The workbook is strong corroboration (and every such
path validates against the XSD), but it is not, on its own, treated as
production proof — it must pass the formal acceptance step below first.

## Promotion rules

- **`confirmed_by_xsd_sample`** → path is production-eligible *if and when data is
  ready*. Still gated by data readiness and the production gates.
- **`workbook_xsd_validated`** → **not** production-eligible until accepted by this
  policy; **may** be implemented in the future builder **behind a structure gate**
  (i.e. the builder may wire the path but must refuse to emit until acceptance +
  data readiness + final XSD).
- **`unresolved`** → production-blocking.
- **`conflict`** → production-blocking.
- **Legacy polluted multi-code paths** (e.g. `RREC1 → ScrtstnIdr`,
  `RREC2 → NewUndrlygXpsrIdr`) → **do not promote**; the XSD wins.
- **RREC fields** → must remain nested under `Coll` (never flattened).
- **NoDataOptn fields** → must carry `value_mode = value_or_nodata` and an
  `nd_wrapper_path`; promotion does not change ND semantics.

## What "formal acceptance" requires (to promote `workbook_xsd_validated` → confirmed)

1. Confirm the **final, non-DRAFT** auth.099.001.04 XSD/version.
2. Human sign-off on the ESMA-code ↔ XSD-element crosswalk for the field.
3. Reconcile cardinality / `minOccurs` per asset class (esp. repeating `Coll`).
4. Confirm value↔`NoDataOptn` wrapper handling for the field.
5. A full-builder XSD validation pass on representative records.

Only after 1–5 may a field's `promotion_status` become `confirmed_by_xsd_sample`-
grade for the path axis — and even then production XML also requires data
readiness and the production gates being opened (out of scope here).

## Current state

```
promotion_status (path axis):
  confirmed_by_xsd_sample : 11
  workbook_xsd_validated  : 89
  manual_review_required  : 2     (RREL67, RREL83 — in workbook, no clean path)
  unresolved              : 3     (RREC22, RREL18, RREL28 — not in workbook)
  conflict                : 2     (RREC1, RREC2 — multi-code-cell pollution)

path_production_eligible  : 11
production_ready          : 0     (all fields — data readiness pending; gates closed)

PATH blocks production before review : 96
PATH blocks production after review  : 7     (only manual/unresolved/conflict)
```

The before→after delta (96 → 7) is the **path axis only**: it shows what would
remain path-blocked if the review policy accepts the 89 workbook paths. It does
**not** make any field production-ready — data readiness and the production gates
remain.

## Artefacts

- `config/delivery/annex2_field_xsd_path_map.yaml` — the source of truth (now
  carries `promotion_status`, `path_production_eligible`,
  `builder_eligible_behind_structure_gate`, `data_readiness`, `production_ready`).
- `output/config_review/annex2_path_map_promotion_checklist.csv` — per-field
  promotion checklist (this policy applied).
- `scripts/build_annex2_path_map_promotion_checklist.py` — reproducible generator.

## Guardrails (unchanged)

```
xml_generation_allowed = false
xml_generated          = false
ready_for_xml_delivery = false
```

No production XML is generated. The legacy Gate 5 runtime is not wired in.
