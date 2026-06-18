# Production Annex 2 XML structure contract (auth.099.001.04)

**Purpose.** Scope *exactly* what is required to produce true, ESMA/XSD-valid
production Annex 2 XML — without generating it. This is a structure contract and
gap analysis, not an implementation.

**Status of the boundary (unchanged):**

```
xml_generation_allowed = false
xml_generated          = false
ready_for_xml_delivery = false
production_xsd_mapping_configured = false
```

> The current non-production artefacts — Client Preview XML and Synthetic
> Full-Coverage Schema Test XML — use an **internal flat preview namespace**
> `urn:trakt:nonproduction:preview`. **They are NOT production-XSD-valid.** In
> particular the preview emits collateral as a *flat sibling* of the loan, which
> is structurally wrong (see Q4). Nothing in this document changes that, and no
> preview placeholder may be used to satisfy production structure.

## Authoritative sources inspected

| Source | What it gives | In repo? |
|---|---|---|
| `DRAFT1auth.099.001.04_1.3.0.xsd` (also `config/system/…`) | element tree, types, `minOccurs`/`maxOccurs`, `xs:sequence` order | ✅ yes (but **DRAFT**) |
| `DRAFT1auth.099.001.04_non-ABCP Underlying Exposure Report.xml` | a real residential-performing instance showing the full nesting | ✅ yes |
| `config/regime/annex2_field_universe.yaml` | 107 Annex 2 codes, format token, ND allowances | ✅ yes |
| `config/regime/annex2_delivery_rules.yaml` | `workbook_semantic` **leaf** tokens, mandatory, source field | ✅ yes |
| `config/system/esma_code_order.yaml` | a **flat** code order ("starter template — verify before prod") | ✅ yes |
| `docs/delivery_xml_agent_v1_review.md`, `docs/xml_preview_policy_spec.md` | prior gating / preview rationale | ✅ yes |

**What is present vs missing at the schema level:** the XSD, namespace, root and
report/record structure are **present and resolved**. What is *missing* is (a) a
confirmed **final, non-DRAFT** XSD, and (b) a complete, confirmed **per-ESMA-code
→ full XML path** mapping with cardinality, ND-wrapper and ordering rules. Today
only **8 of 107** codes have a path confirmed against the XSD + sample.

---

## Q1 — Expected production root / header structure

Resolved from the XSD and sample:

```
Document  (xmlns="urn:esma:xsd:DRAFT1auth.099.001.04")
└─ ScrtstnNonAsstBckdComrclPprUndrlygXpsrRpt        (report message V01)
   └─ NewCrrctn            (choice: NewCrrctn submit/append/update | Cxl cancel)
      └─ ScrtstnRpt        (Securitisation1)   ← report/header level
         ├─ ScrtstnIdr     = RREL1   (SecuritisationIdentifier)   one_per_report
         ├─ CutOffDt       = RREL6   (ISODate)                    one_per_report
         └─ UndrlygXpsrRcrd  (UnderlyingExposureReport1)  minOccurs=1 maxOccurs=unbounded
```

Namespace: `urn:esma:xsd:DRAFT1auth.099.001.04`; `elementFormDefault="qualified"`;
`schemaLocation` pairs the namespace with the XSD file.

## Q2 — Where RREL (exposure) fields sit

RREL splits across **three** XML levels:

- **Report header** — `RREL1` → `ScrtstnRpt/ScrtstnIdr`, `RREL6` → `ScrtstnRpt/CutOffDt`.
- **Exposure identification** — `UndrlygXpsrRcrd/UndrlygXpsrId`:
  `RREL2`→`OrgnlUndrlygXpsrIdr`, `RREL3`→`NewUndrlygXpsrIdr`,
  `RREL4`→`OrgnlOblgrIdr`, `RREL5`→`NewOblgrIdr`.
- **Exposure common data** — `UndrlygXpsrRcrd/UndrlygXpsrData/<assetClass>/<perf>/UndrlygXpsrCmonData`
  holds the bulk of RREL detail in sub-groups: `ActvtyDtDtls`, `OblgrDtls`,
  `UndrlygXpsrDtls`, `BalDtls`, `RpmtDtls`, `IntrstRateDtls`, `RskDtls`,
  `PrfrmncDtls`, `InstnlDtls`.

## Q3 — Where RREC (collateral) fields sit

Collateral lives under `Coll` (e.g. residential-performing `CollateralData22`),
nested **inside the loan**: `…/UndrlygXpsrData/ResdtlRealEsttLn/PrfrmgLn/Coll`,
with children `CollIdr` (`OrgnlIdr`=RREC3, `NewIdr`=RREC4), `CollCmonData`
(`Dtls`, `Valtn`, `Nfrcmnt`).

## Q4 — Loan ↔ collateral relationship (critical)

**One exposure record has one-or-more collateral records, nested within it.**

```
UndrlygXpsrRcrd (1..*)
└─ UndrlygXpsrData → ResdtlRealEsttLn → PrfrmgLn        (asset-class / perf choice)
   ├─ UndrlygXpsrCmonData            (1)   ← loan-level RREL data
   └─ Coll                          (1..*) ← collateral, REPEATING, NESTED
```

`Coll` is `maxOccurs="unbounded"` (`minOccurs` 0 or 1 depending on asset class).
This is the single biggest structural correction over the current preview, which
emits `Collateral` as a *sibling* of the loan rather than a nested, repeating
child.

## Q5 — Report/header vs loan-level vs collateral-level

| Level | XML container | ESMA codes |
|---|---|---|
| Report / header | `ScrtstnRpt` | `RREL1`, `RREL6` |
| Exposure identification | `UndrlygXpsrId` | `RREL2`–`RREL5` |
| Loan level | `UndrlygXpsrCmonData` | most other `RREL*` |
| Collateral level | `Coll` | `RREC*` |

The full per-code split is in the gap matrix (`proposed_xml_level`).

## Q6 — Fields required for structure even if not reportable yet

Structurally required (the XML cannot validate without them) regardless of data
readiness: `RREL1` (ScrtstnIdr), `RREL6` (CutOffDt), and the exposure identity
keys `RREL2`–`RREL5`. At least one `UndrlygXpsrRcrd` and (for residential
performing) at least one `Coll` per loan are also required by `minOccurs`. These
carry `required_for_structure: true` in the contract — and must be satisfied with
**real** values, never preview placeholders.

## Q7 — Which fields can repeat

- `UndrlygXpsrRcrd` — once per loan, **many per report**.
- `Coll` — **many per loan** (nested).
- Within the loan, fixed multi-revision groups exist (e.g. `FrstRvsnData` /
  `ScndRvsnData` / `ThrdRvsnData` under `IntrstRateDtls`) — bounded repetition
  baked into the schema rather than free `maxOccurs`.

Individual leaf fields are generally one-per-container; repetition is at the
record (`UndrlygXpsrRcrd`, `Coll`) level.

## Q8 — one-per-report / loan / collateral / conditional

- **one_per_report:** `RREL1`, `RREL6`.
- **one_per_loan:** exposure-identification + loan-level RREL fields.
- **one_per_collateral:** RREC fields (one set per `Coll`).
- **conditional:** the asset-class choice (`ResdtlRealEsttLn` | `ComrclRealEsttLn`
  | `CorpLn` | `AutomblLn` | `CsmrLn` | `CdtCardLn` | `LeasgLn` | `EstrcLn`) and
  the performing/non-performing choice (`PrfrmgLn` | `NonPrfrmgLn`) select which
  fields/cardinalities apply. ERM = `ResdtlRealEsttLn/PrfrmgLn`.

## Q9 — XML tag/path per ESMA code

Confirmed against the XSD + sample for **8** codes (RREL1, RREL6, RREL2–5,
RREC3, RREC4) — see `field_mappings` in the YAML and `xml_path_known=true` rows
in the gap matrix. For the remaining **99** codes:

- `annex2_delivery_rules.yaml` provides only a **leaf token** (`workbook_semantic`,
  e.g. `PrprtyTp/Cd`, `AmrtstnType/Cd`) — **not** a full path — and several are
  explicitly `TBC`/mismapped (e.g. RREL10, RREL13, RREL14, RREL26, RREL75).
- The full path requires walking the XSD `xs:sequence` nesting (intermediate
  wrappers such as `Val/Amt`, `…/Cd`, `NoDataOptn/NoData`).

This per-code path completion is the principal remaining structure gap.

## Q10 — Namespaces / schema files required

- Target namespace: `urn:esma:xsd:DRAFT1auth.099.001.04`.
- Schema file: `DRAFT1auth.099.001.04_1.3.0.xsd` (vendored at repo root and
  `config/system/`). The companion `auth.098.001.04` XSD (`config/system/`) is
  the securitisation/registration message, not the underlying-exposure report.
- **Missing:** confirmation that this DRAFT XSD equals the **final published**
  auth.099.001.04 the repository must submit against.

## Q11 — Ordering rules

Element order is fixed by the **nested `xs:sequence`** blocks in the XSD (order
within each complexType). `config/system/esma_code_order.yaml` is a **flat**
ESMA-code list and self-describes as a "starter template — verify against
official ESMA XSD before production use"; it cannot drive production element
ordering on its own. The production ordering rule must be **derived from the XSD
sequence**, per container.

## Q12 — Validation steps before production XML can be claimed

1. Confirm the **final, non-DRAFT** auth.099.001.04 XSD and version.
2. Complete the **per-field XML path** mapping for all 107 codes (resolve every
   `workbook_semantic` leaf/TBC into a full XSD path).
3. Model the **asset-class** + **performing/non-performing** choice selection.
4. Model the **value ↔ `NoDataOptn`** wrapper choice per field (ND handling).
5. Derive **nested element order** from the XSD `xs:sequence`.
6. **Validate** emitted XML against the XSD (`xmllint --schema` / `lxml`).
7. Resolve all **data dependencies** (client/operator/config/source blockers).
8. Reconcile **cardinality / `minOccurs`** per asset class (esp. `Coll`).

Only when 1–8 pass may the production gate be revisited — and never via the
preview path.

---

## Gap classification summary

From `output/config_review/annex2_xml_structure_gap_matrix.csv` (107 codes):

| gap_class | count |
|---|---|
| `missing_field_xml_path` | 47 |
| `missing_exposure_record_mapping` | 44 |
| `data_dependency` (path resolved, data pending) | 8 |
| `missing_collateral_record_mapping` | 7 |
| `resolved_for_data_not_structure` (RREL35) | 1 |

The fuller taxonomy used across the contract and matrix:
`missing_xsd_reference` (RESOLVED — XSD vendored), `missing_namespace`
(RESOLVED), `missing_root_mapping` (RESOLVED), `missing_header_mapping`
(RESOLVED), `missing_exposure_record_mapping`, `missing_collateral_record_mapping`,
`missing_field_xml_path`, `missing_cardinality_rule`, `missing_order_rule`,
`data_dependency`, `resolved_for_data_not_structure`.

## Legacy Gate 5: map reused, runtime retired

Since this contract was written, the per-field XML paths have advanced via the
ESMA mapping workbook (the crosswalk the legacy Gate 5 builder read at runtime):

- **Map reused.** Every RREL/RREC workbook `PATH` re-validates against this XSD,
  so the field path map now carries 89 `workbook_xsd_validated` paths in addition
  to the 11 `confirmed_by_xsd_sample` ones. See
  `config/delivery/annex2_field_xsd_path_map.yaml`.
- **Runtime retired.** The legacy builder's silent ND5 injection, value
  fabrication (`RREL12 → "2026"`), wide one-row-per-loan shape, singleton/
  flattened `Coll`, and workbook-order sequencing are **not** adopted. RREC stays
  nested under `Coll`.
- **Path ≠ data ≠ production.** `workbook_xsd_validated` paths are **not**
  production-eligible until formally accepted (see
  `docs/annex2_path_map_promotion_policy.md`), and even a production-eligible path
  is still gated by **data readiness**. `production_ready` is `false` for all 107
  fields. The validation steps in Q12 above remain outstanding, so production XML
  stays blocked.

## Regenerating the gap matrix

```
python scripts/build_annex2_xml_structure_gap_matrix.py
```

Read-only over the configs + vendored XSD/sample. It generates no XML and touches
no production gate.
