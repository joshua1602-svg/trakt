# Annex 2 field-to-XSD path mapping layer

**Purpose.** Tell the (future) production XML builder *where each Annex 2 field
goes in the real ESMA XML tree* — and how certain we are. This is the bridge
between the 107 flat ESMA codes (RREL*/RREC*) and the deeply nested
`auth.099.001.04` schema. It does **not** generate production XML and does
**not** change any production gate.

Artefacts:

- `config/delivery/annex2_field_xsd_path_map.yaml` — the production source of truth.
- `output/config_review/annex2_field_xsd_path_map.csv` — the same in tabular form.
- `scripts/build_annex2_field_xsd_path_map.py` — the reproducible generator.

## Why this layer is needed

The structure contract proved the real XML tree is **nested, not flat**
(collateral repeats inside the loan). But knowing the *shape* is not enough to
build XML: the builder needs, **per field**, the exact element path, the level
(header / loan / collateral), the cardinality, whether the value is wrapped in a
`NoDataOptn` choice, and — critically — **how confident we are** in that mapping.
Guessing here would silently produce invalid or wrong-meaning XML, so every row
carries an explicit `mapping_status` and `evidence_source`.

## How the mapping is derived (evidence, not guesswork)

1. The generator recursively walks the **residential-real-estate / performing**
   branch of the vendored XSD
   (`ResdtlRealEsttLn/PrfrmgLn → SecuritisationLoanData2 → ExposureData1 +
   CollateralData22`), plus the report header (`ScrtstnRpt`) and the
   exposure-identification block (`UndrlygXpsrId`). This yields the real element
   path, XSD type, sequence order, and whether the element is a
   value-or-`NoDataOptn` choice.
2. It cross-checks the vendored **sample message**.
3. It reads the delivery-rules `workbook_semantic` leaf tokens — but treats them
   as **inference, not proof**, because they use a workbook naming convention
   that often differs from the XSD element names (e.g. `AmrtstnType` vs the real
   XSD `AmtstnTp`, `Prps` vs `Purp`).

### Status meanings

| status | meaning |
|---|---|
| `confirmed` | `workbook_semantic` token **and** the sample message agree with the XSD path |
| `inferred_high_confidence` | element present in the sample at a determinable path; field-label match strong but not token-proven |
| `inferred_low_confidence` | a single/closest XSD candidate exists but needs manual confirmation (incl. fuzzy token matches and ambiguous multi-location leaves) |
| `unresolved` | no usable evidence (workbook token is TBC/mismapped and no close XSD element) |
| `conflict` | two codes resolve to the **same** element (e.g. origination vs current LTV) — at most one can map there |

## What is confirmed

11 fields are confirmed by exact token + sample agreement, e.g.:

- `RREL1` → `ScrtstnRpt/ScrtstnIdr` — **report-level securitisation identifier**
  (XSD definition + 28-char pattern match RREL1's content). Note: this is a
  report/header identifier, *not* an exposure identifier.
- `RREL6` → `ScrtstnRpt/CutOffDt` — report cut-off date.
- `RREL2` → `UndrlygXpsrId/OrgnlUndrlygXpsrIdr`.
- `RREC9` → `Coll/CollCmonData/Dtls/PrprtyTp` (collateral, value-or-`NoDataOptn`).
- plus `RREL22, RREL25, RREL31, RREL32, RREL40, RREL41, RREL69`.

## What is inferred

- **High confidence (5):** `RREL3/4/5` (exposure identifiers under
  `UndrlygXpsrId`) and `RREC3/4` (collateral `CollIdr/OrgnlIdr` / `NewIdr`).
  These appear in the sample and the field labels disambiguate them, but they
  are **not** delivery-rules-proven — so they remain inferred (a deliberate
  downgrade from the earlier "confirmed" claim in the structure contract).
- **Low confidence (18):** fuzzy/ambiguous XSD candidates that need manual
  confirmation (e.g. `RREL35` → candidate `AmtstnTp`).

## What is unresolved

72 fields are unresolved: the `workbook_semantic` is TBC/mismapped (or uses a
naming the XSD does not share) and no close XSD element was found automatically.
These need a manual ESMA-code ↔ XSD-element crosswalk.

## Why production XML still cannot be generated

- Only **11 / 107** fields are confirmed; **96** carry a production-blocking
  mapping gap.
- The vendored XSD is the ESMA **DRAFT** (`DRAFT1auth…`); the final schema must
  be confirmed.
- `NoDataOptn` wrapper handling, asset-class/performing choice selection, and
  per-container ordering are not yet wired.
- Preview placeholders must never be used to fill a production mapping gap.

Production gates remain unchanged:

```
xml_generation_allowed = false
xml_generated          = false
ready_for_xml_delivery = false
```

## How this feeds the future production XML builder

The YAML is the intended **single source of truth for field placement**. A
future builder will, for each field: look up `xml_path`, place the value at that
path under the correct record (`UndrlygXpsrRcrd` / nested `Coll`), wrap it in
`NoDataOptn` when `value_mode = value_or_nodata` and an ND is selected, and order
elements by the XSD sequence. The builder must **refuse** any field whose
`mapping_status` is not `confirmed` (and `blocks_production_xml = true`) until it
is resolved — so the map drives a hard gate, not a guess.

## Summary

```
Total fields:                     107
Confirmed mappings:               11
High-confidence inferred:         5
Low-confidence inferred:          18
Unresolved:                       72
Conflicts:                        1
Production-blocking mapping gaps: 96
```

## Regenerating

```
python scripts/build_annex2_field_xsd_path_map.py
```

Read-only over config + the vendored XSD/sample. Generates no XML and changes no
production gate.
