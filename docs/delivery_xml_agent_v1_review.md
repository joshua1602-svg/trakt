# Delivery / XML Agent v1 — Targeted Review & Implementation Plan

This document is the **required first step** for Delivery/XML Agent v1: a review of
the existing frozen Gate 4b delivery normaliser and Gate 5 Annex 2 XML builder,
and a conservative plan for the new agent that consumes the **Projection** package
and either (1) produces delivery-normalised output for delivery-ready controlled
data, or (2) **refuses** XML generation with clear delivery issues.

```text
Onboarding → Transformation → Validation → Projection → Delivery / XML Agent v1
                                                          ^^^^^^^^^^^^^^^^^^^^^^^  (this PR — v1, review + skeleton)
```

**No production XML is generated in this PR.** XML preview is hard-gated behind
delivery readiness and a `--allow-xml-preview` flag, and the current run does not
pass the readiness gates.

## Governing principle

The Delivery/XML Agent **must not** solve upstream onboarding, operator, client,
config or projection blockers. It consumes the projected Annex 2 target frame and
the controlled blockers, and it must:

* never silently fill missing values;
* never let the XML builder override Onboarding / Transformation / Validation /
  Projection decisions;
* never claim XML readiness unless **all** required delivery validations pass.

---

## Components reviewed

| Component | File | Role today |
| --- | --- | --- |
| Gate 4b — Annex 2 delivery normaliser | `engine/gate_4b_delivery/annex2_delivery_normalizer.py` | wide projected CSV → schema-ready "delivery" CSV (precision / regex / boolean / ND preflight) |
| Gate 5 — Annex 2 XML builder | `engine/gate_5_delivery/xml_builder_annex2.py` | wide delivery CSV → ESMA `auth.099` XML tree (lxml) |
| Gate 4 — regime projector | `engine/gate_4_projection/regime_projector.py` | legacy canonical CSV → wide ESMA-coded projected CSV (predecessor of the Projection Agent) |
| Authoritative regime contract | `config/regime/annex2_delivery_rules.yaml` | per-code `field_rules` (mandatory, `nd_allowed`, `transform`, `validators`, `precision`) |
| Field universe | `config/regime/annex2_field_universe.yaml` | 107 workbook-derived codes, ND allowances, `format` token |
| Code order | `config/system/esma_code_order.yaml` | `Record:` list (77 codes) used for XML ordering |
| Reference artefacts | `DRAFT1auth.099.001.04_*.xsd / .xml / .xlsx` | ESMA Annex 2 XSD, sample XML, mapping workbook |

---

## Review questions

### 1. What does the existing Gate 4b normalizer do?

`annex2_delivery_normalizer.py` reads a **wide** `*_ESMA_Annex2_projected.csv`
(one column per ESMA code, one row per exposure) and, per `field_rules` entry in
`annex2_delivery_rules.yaml`, normalises each cell into a schema-ready delivery
value, emitting `*_delivery_ready.csv`, a `*_delivery_report.json` and a
`*_delivery_issues.csv`. Per field it:

* `derive`s values (`first_non_blank_from_fields`, `months_between_dates`);
* fills `default_value` where `default_allowed`;
* generates a `securitisation_id` (`ScrtstnIdr`) from LEI + year + sequence;
* enforces **ND restriction** (`ND value 'NDx' not allowed for field`);
* normalises booleans to `xsd_lowercase_true_false`;
* applies `transform.enum_map` (strict) and `transform.geography_map` (best-effort);
* validates LEI and `validators.regex`;
* applies numeric **precision** (`total_digits` / `fraction_digits`, half-up).

It ends with a **hard-gate preflight**: any `error`-severity issue → non-zero exit.

### 2. What does the existing Gate 5 XML builder do?

`xml_builder_annex2.py` builds the ESMA `auth.099` XML tree with `lxml`:

* loads **mapping specs** from the Excel workbook (`RTS Field code`, `XML TAG`,
  `PATH`, `MULTIPLICITY`, `Template`, `Performing/Non Performing`), filtered to
  templates `ALL`/`RRE` and the selected performance mode (`PRF`/`NPRF`);
* derives the `targetNamespace` from the XSD (default
  `urn:esma:xsd:DRAFT1auth.099.001.04`);
* builds a child-ordering index from workbook PATH sequences;
* applies **header/singleton codes once** (with a cross-row "header field varies"
  consistency check), then emits **one fresh `UndrlygXpsrRcrd` record per CSV row**;
* selects choice branches by value shape (ND → `NoDataOptn/NoData`; dates → `Dt`;
  amounts → `Amt` with a `Ccy` attribute);
* injects required NoData defaults (`ScndryOblgrIncm`, NPE `HstrclColltn` 36-month
  series) where branches are mandatory but absent;
* optionally validates the result against the XSD and exits non-zero on failure.

### 3. Which parts are safe to reuse immediately?

| Reusable now | Why it is safe |
| --- | --- |
| `annex2_delivery_rules.yaml::field_rules` metadata (mandatory, `nd_allowed`, `transform.enum_map`/`geography_map`, `validators.regex`, `precision`) | already the authoritative regime contract shared by Gate 4b; v1 reads it the same way |
| Gate 4b **pure value predicates** — `_is_nd`/`ND_PATTERN`, `normalize_boolean`, `validate_lei`, `apply_precision`, `issue_category` | pure, non-mutating, no I/O, no hard exit; v1 reuses them **read-only** for `format_valid` / `enum_valid` / `is_nd_value` checks (never to rewrite a cell) |
| Gate 5 **shape predicates** — `_is_nd`, `_is_date`, `_is_iso_year`, `_parse_multiplicity`, `_split_path`, `record_group_for_code` concept, `load_code_order` | pure path/value helpers, no tree mutation; v1 surfaces them through `gate5_adapter` for `xml_path` / `xml_record_group` / `xsd_type` annotation |
| `annex2_field_universe.yaml` (107 codes, ND allowances, `format`) + `esma_code_order.yaml::Record` (77 codes) | static reference for the `xsd_type`, `is_mandatory`, `nd_allowed` columns and the template-order completeness gate |

### 4. Which parts are unsafe because they assume the old wide-frame / Gate 4 world?

* **Gate 4b cell normalisation as a mutation pass.** `normalize_delivery` iterates
  a **wide** DataFrame (`df.at[row_idx, field]`) and **rewrites** cells:
  `default_value` fill, `securitisation_id` generation, `derive`, boolean casing,
  enum/geo mapping. Run against the new **long** frame it would (a) not find the
  columns, and (b) silently fill / overwrite values that Projection deliberately
  left blocked or ND. v1 must **not** run it as a mutator.
* **Gate 4b hard-gate `sys.exit(2)`.** A library-grade agent must not exit the
  process; v1 carries issues, it does not abort.
* **Gate 5 wide-frame assumption.** `build_annex2_tree` expects a wide CSV
  (`code in df.columns`, `df[code]`) and emits one flat record per row. The long
  target frame has columns `esma_code` / `projected_value`, not one column per
  code — calling Gate 5 on it directly would find **no** mapped codes and raise
  *"No Annex2 mapping specs found"* or build an empty tree.
* **Gate 5 NoData backfill** (`_ensure_scndry_oblgr_incm_defaults`,
  `_ensure_hstrcl_colltn_nd_defaults`, `_ensure_nprf_nonprfrmgdata_defaults`)
  **invents `ND5`** into the tree. That is exactly the silent-fill behaviour v1
  forbids — it would override Projection's blocked decisions.
* **Gate 5 header-constant forcing** (RREL1/RREL6 collapsed to a single pool value)
  encodes a delivery/XML shape decision that must be governed, not assumed.
* **Gate 4 `regime_projector` enum_agent path** raises on unreviewed enums — not
  used by v1 at all.

### 5. What should Delivery/XML Agent v1 consume as input?

The **Projection manifest** plus the artefacts it discovers next to it:

```
--projection-manifest output/projection/50_projection_manifest.json
  → output/projection/51_projected_annex2_target_frame.csv   (long target frame)
  → output/projection/52_projection_field_contract.csv
  → output/projection/55_projection_issues.csv
  → output/projection/56_projection_blocker_resolution.csv
config/regime/annex2_delivery_rules.yaml
config/regime/annex2_field_universe.yaml
config/system/esma_code_order.yaml
config/system/fields_registry.yaml
```

It validates the manifest is from `projection_agent`, did not perform XML
delivery, and exposes a target frame. It **never** re-runs or mutates
Onboarding / Transformation / Validation / Projection artefacts.

### 6. What should Delivery/XML Agent v1 output?

Under `output/delivery_xml/`:

| Artefact | Content |
| --- | --- |
| `60_delivery_manifest.json` / `.yaml` | governance flags, input/output links, counts, readiness booleans, `next_agent` |
| `61_delivery_readiness.json` / `.md` | the named delivery-readiness gates and their pass/fail with reasons |
| `62_delivery_normalised_frame.csv` / `.json` | delivery-facing view of `51_*` with delivery columns (see below) |
| `63_delivery_issues.csv` / `.json` | delivery-blocking issues by category |
| `64_delivery_lineage.json` | extends projection/validation/transformation lineage |
| `65_xml_preview.xml` *(guarded)* | only if all readiness gates pass **and** `--allow-xml-preview` |
| `66_xml_validation_report.json` *(guarded)* | only alongside `65_*` |

### 7. What delivery readiness checks are required before XML generation?

v1 **refuses** XML if **any** of these is true (each is a named gate in `61_*`):

1. `projection_complete` is false;
2. `ready_for_delivery_normalisation` is false;
3. `ready_for_xml_delivery` is false;
4. any delivery-blocking projection issue remains (from `55_*`);
5. any target-frame row has a `blocked_*` projection status;
6. any mandatory field is blank without an allowed/selected ND treatment;
7. any value violates delivery rules / XSD-ish format (`format_valid` / `enum_valid`);
8. any required XML header/report metadata is missing;
9. record grouping cannot be determined for a row;
10. template/code order is incomplete for required XML fields.

`xml_generation_allowed = all gates pass`. Even when XML is refused, v1 still
writes the readiness report and the normalised preview frame showing what *would*
be deliverable and what is blocked.

### 8. How should blocked fields from Projection be represented?

Blocked rows are **carried, never rewritten**. In `62_*` each row carries:

* `delivery_status = blocked` (deliverable rows are `deliverable`; non-mandatory
  blanks are `not_required_blank`; present-but-invalid values are
  `delivery_invalid`);
* `delivery_blocker_type` derived from the projection status / issue type;
* `delivery_value` left **blank** for blocked rows (the projected value, if any,
  is preserved in the carried `projected_value` column — v1 does not promote it);
* a `delivery_issue_id` linking to the matching `63_*` row.

These projection statuses are treated as **blockers**:
`blocked_client_onboarding_dependency`, `blocked_operator_or_config_dependency`,
`unresolved_not_materialised`, `unresolved_source_mapping`, `invalid_nd_for_field`,
and `not_projected_blank` **where the field is mandatory and no allowed/selected
ND/default exists**.

These are treated as **candidate deliverable** values (subject to delivery
validation): `projected_from_transformed`, `projected_nd_default`,
`projected_asset_default`.

> **Note — `delivery_invalid` (enum/format) is a config issue, not an XML builder
> issue.** A candidate-deliverable value that fails the regime `enum_map`/regex is
> flagged `delivery_invalid`; the agent does **not** rewrite it. The first such
> case was **RREL35 amortisation_type** (`Bullet`, 1,526 rows): the regime
> `enum_map` had been narrowed to ERM `OTHR` synonyms, dropping the authoritative
> `FRXX/DEXX/FIXE/BLLT/OTHR` list. The fix was **config only** — restore the
> authoritative regime code list and add a config-driven ERM asset-policy override
> (`Bullet → OTHR`) — applied at **Projection** time, never in the XML builder and
> never hard-coded in Python. See `docs/rrel35_amortisation_type_remediation.md`.

### 9. How should target-frame long format be converted into XML-ready shape?

The frame is **long** (one row per *loan × ESMA field*). The conversion is
deferred to a later XML build step, but v1 records everything needed for it:

* `row_id` / `loan_identifier` identify the exposure (the XML record key);
* `esma_code` + `xml_record_group` + `xml_path` + `xsd_type` annotate each value's
  destination in the `auth.099` tree;
* delivery validation runs **per cell** on the long frame (no pivot required).

A real pivot to a wide, per-record shape is **deferred to v2** (it is a structural
decision — see Q12 / Q16). v1 deliberately does not freeze the wide shape.

### 10. How should record groups be handled?

v1 tags, it does not nest:

* **RREL** (`RREL*`) → `xml_record_group = underlying_exposure` (loan/exposure record);
* **RREC** (`RREC*`) → `xml_record_group = collateral` (collateral/property record);
* everything else → `xml_record_group = header_pool_report` (header/pool/report
  metadata, emitted once).

The record group is preserved from `51_*`'s `record_group` column into `62_*`'s
`xml_record_group`. Whether RREC collateral nests under RREL exposures, sits in a
separate section, or repeats as related records is **identified, not solved** here.

### 11. Does existing Gate 5 assume one flat row per XML exposure?

**Yes.** `build_annex2_tree` iterates `df.iterrows()` and calls
`create_new_record_node` once per row, putting all RREL **and** RREC fields on the
**same** flat `UndrlygXpsrRcrd`. Header fields (RREL1/RREL6) are forced to a single
pool value via `apply_header_code`'s "header field varies" check. This is the
legacy flat-record shape.

### 12. Does Annex 2 require nested collateral records under exposure, separate collateral sections, or repeated related records?

ESMA `auth.099` genuinely models **collateral as repeatable related records** —
one underlying exposure can carry multiple collateral items, so RREC fields are
**not** simply extra columns on the exposure row. The legacy Gate 5 flat-row
shape papers over this by assuming exactly one collateral per exposure. Resolving
the true cardinality (nested vs separate-section vs repeated `Coll` records) is a
structural decision **deferred to v2**; v1 only records `xml_record_group` so the
question is explicit and traceable.

### 13. What configuration is missing before XML can be trusted?

* **Per-code XML PATH / XSD type** not yet in a runtime config — today they live
  only in the Excel workbook (read by Gate 5 via `pandas.read_excel`). v1 annotates
  `xml_path` from the regime `workbook_semantic` where present and leaves it blank
  (marked deferred) otherwise; a generated `annex2_xml_paths.yaml` is needed.
* **Record-group cardinality / nesting rules** (RREL↔RREC) — no config expresses
  whether collateral nests or repeats.
* **Header/pool/report metadata contract** — which codes are report-level
  singletons and their required values (`ScrtstnIdr`, cut-off date, currency).
* **ND-default policy completeness** — several mandatory codes (RREL1/RREL2)
  permit **no** ND and have unresolved client/operator dependencies, so no
  delivery default is configurable yet (correctly).
* **Currency / amount `Ccy`** source for `Amt` leaves (Gate 5 hard-codes `GBP`).

### 14. What fields are still missing from esma_code_order.yaml, and does this matter for XML ordering?

`esma_code_order.yaml::Record` lists **77** codes; the authoritative
`annex2_field_universe.yaml` carries **107**. The **30** codes present in the
universe but absent from the `Record` order are exactly the Projection
"30 fields not in template order" warning. **It matters**: XSD requires strict
`<xs:sequence>` order, and any *required* delivery field missing from the order
cannot be placed deterministically. v1 raises a `template_order_incomplete` gate
failure listing the missing required codes; it does not guess an order.

### 15. What XSD/schema validation capability exists today?

Gate 5 can validate against the XSD via `lxml` (`etree.XMLSchema(...).validate`)
and the repo ships `config/system/DRAFT1auth.099.001.04_1.3.0.xsd` plus a sample
XML. There is **no** standalone, frame-level XSD validator — XSD validation today
only runs **after** a full tree is built. v1 performs **XSD-ish** value checks
(ND allowance, `validators.regex`, `format` token, enum membership) on the frame
*without* building a tree; true XSD validation remains a post-build step deferred
with XML generation.

### 16. What should be deferred to Delivery/XML Agent v2?

* Actual production XML generation (full tree build) and XSD validation report.
* The long→wide pivot and the **RREL↔RREC nesting / collateral cardinality** decision.
* A runtime `annex2_xml_paths.yaml` (PATH + XSD type per code) replacing the workbook read.
* Header/pool/report metadata contract + currency source.
* Re-using Gate 4b's value transforms as an **authorised** delivery mutation pass
  (precision/boolean/enum/derive) once Projection is complete — strictly governed,
  never silent.
* Completing `esma_code_order.yaml` to all 107 codes.

---

## Gate 4b review (`annex2_delivery_normalizer.py`)

* **Reusable normalisation rules:** the pure value predicates — `ND_PATTERN`/`_is_nd`,
  `normalize_boolean`, `validate_lei`, `apply_precision`, `issue_category`. v1 reuses
  them **read-only** to compute `format_valid` / `enum_valid` / `is_nd_value`, never
  to rewrite a cell.
* **Reusable preflight checks:** the *concepts* of ND-restriction, regex/LEI and
  precision validation map directly onto the v1 readiness gates (Q7.4/Q7.6/Q7.7).
* **Stale assumptions:** wide-frame cell addressing (`df.at[row_idx, field]`,
  `field in out_df.columns`), in-place mutation (`default_value` fill,
  `securitisation_id` / `derive` generation), and the hard-gate `sys.exit(2)`.
* **How to adapt to the long frame:** iterate `51_*` rows; for each, look up the
  regime rule by `esma_code`; run the predicates against `projected_value` to set
  `format_valid` / `enum_valid` / `nd_allowed`; **emit a delivery column / issue**
  rather than overwriting the value. Defaulting / derivation / ScrtstnIdr generation
  stay **off** in v1 (they are silent fills).

## Gate 5 review (`xml_builder_annex2.py`)

* **Input shape expected:** a **wide** delivery-ready CSV — one column per ESMA
  code, one row per exposure (`code in df.columns`, `df[code]`).
* **Can it consume long-frame target data?** **No.** The long frame exposes
  `esma_code` / `projected_value` columns, so `load`/`apply_*` find no mapped code
  columns and either raise *"No Annex2 mapping specs found"* or build an empty tree.
* **Does it assume flat exposure rows?** **Yes** — one fresh `UndrlygXpsrRcrd` per
  row, RREL and RREC on the same record, header codes collapsed to one pool value.
* **ND tags:** `select_specs_for_value` routes `ND[1-5]` to a `NoDataOptn/NoData`
  branch; `_ensure_*` helpers **inject `ND5`** where mandatory NoData branches are
  absent (silent fill — unsafe for v1).
* **Namespaces:** `targetNamespace` from the XSD (default
  `urn:esma:xsd:DRAFT1auth.099.001.04`), with `xsi:schemaLocation` set when an XSD
  is supplied.
* **Header/report/pool metadata:** `apply_header_code` emits non-record codes once
  and **raises** if a header field varies across rows.
* **RREL/RREC grouping:** distinguished only by workbook PATH (presence of the
  `UndrlygXpsrRcrd` anchor) — there is no first-class record-group model; RREC sits
  inside the same flat record.
* **What would break if called directly on the new target frame:** no code columns →
  empty/raised tree; and even after a pivot, the NoData backfill and header-forcing
  would **override** Projection's blocked/ND decisions — which v1 forbids.

---

## v1 hand-off

`62_*` shows what is deliverable vs blocked; `63_*` enumerates the delivery
issues by category; `61_*`/`60_*` carry the readiness verdict. Because the current
run's Projection is not complete and blockers remain, v1 reports:

```
delivery_xml_ran               = true
delivery_normalisation_complete = false
xml_generation_allowed         = false
xml_generated                  = false
ready_for_xml_delivery         = false
next_agent                     = operator_config_projection_remediation
```

with issue categories `client_onboarding_dependency`,
`operator_or_config_dependency`, `config_dependency`, `source_mapping_unresolved`,
`nd_default_rule_missing`, `delivery_structure_deferred`,
`template_order_incomplete`. **No production XML is generated.**
