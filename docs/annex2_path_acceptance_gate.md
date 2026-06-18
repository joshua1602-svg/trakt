# Annex 2 path acceptance gate

A formal, evidence-based gate that decides **which field-to-XSD paths are good
enough for the new XML builder to wire** — not whether anything is production
XML. It reviews the 89 `workbook_xsd_validated` paths (plus the rest) against the
three ESMA artefacts and records a decision per field.

> **This gate is about path quality, not production.** Accepting a path for the
> builder does **not** make a field production-ready. `production_ready` stays
> `false` for all 107 fields; the production gates stay closed; no production XML
> is generated; and the legacy Gate 5 runtime (silent ND5 defaults, value
> fabrication, wide one-row-per-loan, flattened collateral) is **not** imported.

## Evidence used (you don't need to be an XSD expert)

1. **The ESMA workbook** (`DRAFT1auth.099.001.04_...Version_1.3.1.xlsx`) — its
   `PATH` column gives the full XML path per RTS field code.
2. **The XSD** (`DRAFT1auth.099.001.04_1.3.0.xsd`) — every candidate path is
   **re-walked and re-validated** against the actual schema tree by the gate.
3. **The sample message** — used as *stronger* confirmation **where present**;
   it is **not** required to contain every field.

## Acceptance criteria

A path is **`accepted_for_builder`** only if all of these hold:

- it comes from the ESMA workbook `PATH` column (`from_workbook_path`);
- it **re-validates against the XSD** (`xsd_validated`);
- the field code/leaf is consistent with the workbook element
  (`code_label_type_consistent`);
- it is **not** a polluted multi-code row (`not_polluted_multicode`);
- it respects the RREL/RREC hierarchy (`respects_rrel_rrec_hierarchy`);
- RREC collateral fields stay **nested under `Coll`** (`rrec_nested_under_coll`);
- NoDataOptn handling is **understood or explicitly flagged** (`nodataoptn_handling`
  ∈ `handled_value_or_nodata` / `not_applicable` / `flagged_needs_review`). A flag
  is recorded but does not by itself block acceptance.

## Decision vocabulary (`builder_acceptance_status`)

| status | meaning | builder may wire? | production-ready? |
|---|---|---|---|
| `sample_confirmed` | XSD **and** the sample message agree (strongest) | yes (behind structure gate) | no |
| `accepted_for_builder` | workbook PATH, XSD-validated, all criteria pass | yes (behind structure gate) | no |
| `needs_manual_review` | no clean XSD-validated workbook path | no | no |
| `rejected` | polluted multi-code / path collision | no | no |

## Decisions

```
sample_confirmed     : 11   (XSD + sample agree)
accepted_for_builder : 89   (workbook PATH, XSD-validated, criteria pass)
needs_manual_review  : 5    (RREL67, RREL83 — in workbook, no clean element path;
                             RREC22, RREL18, RREL28 — not in the workbook branch)
rejected             : 2    (RREC1, RREC2 — multi-code-cell pollution; XSD wins)
production_ready     : 0    (ALL fields — data readiness + final schema pending)
```

Per-field evidence and reasons are in
`output/config_review/annex2_path_acceptance_decisions.csv`.

### Why RREC1 / RREC2 are rejected

In the workbook, RREC1 shares a multi-code `RTS Field code` cell that maps to the
report-level `ScrtstnIdr`, and RREC2 to the loan-level `NewUndrlygXpsrIdr`. Both
would place a **collateral** field outside `Coll`. The XSD wins: collateral must
stay nested under `.../PrfrmgLn/Coll`, so these paths are rejected, not promoted.

### NoDataOptn handling

Where a field is `value_or_nodata`, the gate confirms the `nd_wrapper_path`
(`.../NoDataOptn/NoData`) also re-validates against the XSD and records
`handled_value_or_nodata`. Plain values record `not_applicable`. Anything the
gate cannot confirm is recorded as `flagged_needs_review` (acceptable to flag,
per the criteria, but surfaced for the builder).

## Path readiness vs data readiness (kept separate)

`accepted_for_builder` is a **path-axis** decision. Producing real XML for a
field additionally requires **data readiness** (the real client/operator/config/
source value — certified by the Delivery/XML Agent, **pending for all fields**)
and the production gates being opened. The CSV carries both: the acceptance
decision and `production_ready = false`.

## What the builder may do with an accepted path

A future XML builder may wire an `accepted_for_builder` / `sample_confirmed` path
**behind a structure gate**: build the element at that path under the correct
record (`UndrlygXpsrRcrd` / nested `Coll`), apply the `value_or_nodata` wrapper
when an ND is genuinely selected, and order by the XSD sequence — but it must
**refuse to emit** until (a) the path is formally accepted under
`docs/annex2_path_map_promotion_policy.md`, (b) data readiness is certified, and
(c) the final (non-DRAFT) XSD validates the output. It must **never** inject ND5
defaults or fabricate values.

## Guardrails (unchanged)

```
xml_generation_allowed = false
xml_generated          = false
ready_for_xml_delivery = false
production_ready       = false (all 107 fields)
```

## Regenerating

```
python scripts/build_annex2_field_xsd_path_map.py            # stamps builder_acceptance_status
python scripts/build_annex2_path_acceptance_decisions.py     # the decisions CSV
```

Read-only over config + the vendored XSD/workbook/sample. Generates no XML and
changes no production gate.
