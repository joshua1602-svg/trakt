# XML Preview Policy Spec — two non-production artefact modes

This spec governs the **two separate, non-production** XML artefact modes built
by `engine/delivery_xml_agent/preview_readiness.py`. Neither mode is production.
Neither mode may flip a production XML gate. Both modes are **disabled by
default**.

| | Client Preview XML | Synthetic Full-Coverage Schema Test XML |
|---|---|---|
| mode key | `client_safe_preview` | `synthetic_full_coverage_schema_test` |
| audience | client demo | engineering / schema test |
| data | real delivery-valid values + approved `PREVIEW_ONLY_*` placeholders + explicit exclusions | dummy values for every Annex 2 field (real reused where available) |
| fabricate economic values? | **no** | yes (clearly labelled synthetic) |
| client-facing? | yes (watermarked) | **no** (engineering only) |
| reportable? | no | no |
| output dir | `output/delivery_xml/preview/client_preview/` | `output/delivery_xml/preview/synthetic_schema_test/` |

## Production boundary (unchanged)

The production Delivery/XML Agent still refuses XML:

```
xml_generation_allowed = false
xml_generated          = false
ready_for_xml_delivery = false
```

The preview evaluator **reads these flags read-only** and echoes them in every
readiness artefact (`production_flags_unchanged`). It never writes or sets them.
The new, separate flags it introduces are:

```
xml_preview_allowed / ready_for_xml_preview / xml_preview_generated
synthetic_schema_test_allowed / ready_for_synthetic_schema_test / synthetic_schema_test_generated
```

## Source of truth

Field sets are defined in **`config/delivery/xml_preview_policy.yaml`**, not in
Python:

- `client_preview_field_policy.placeholder_fields` — identifier/reference fields
  that may carry a `PREVIEW_ONLY_<code>` placeholder
  (`RREL1, RREL2, RREL3, RREL4, RREL5, RREL82, RREC1, RREC2, RREC3, RREC4`).
- `exclusion_blocker_types` — operator-ambiguous (`operator_or_config_dependency`)
  and optional/deferred (`delivery_structure_deferred`, `template_order_incomplete`)
  fields are **excluded** from the client preview.
- `must_resolve_before_preview_*` — config mappings (`config_dependency`),
  source/projection gaps (`source_mapping_unresolved`, `nd_default_rule_missing`),
  format-invalid values, plus `RREL69` and `RREL6` (header/data-cut-off ordering).
  While any of these remain, the **client preview verdict is not allowed**.
- `never_fabricate_fields` / `never_fabricate_format_tokens` — valuation / rate /
  monetary / percentage fields are **never** fabricated or placeholdered in the
  client preview; if blocked they are excluded or flagged must-resolve.
- `resolved_fields` — e.g. `RREL35` (source Bullet → ERM asset policy → `OTHR`):
  delivery-valid and therefore never placeholder/exclusion/synthetic.

## Verdict rules

**Client preview** is *allowed* only when every blocked field is covered by an
approved placeholder or an explicit exclusion — i.e. there are **no
must-resolve** fields left. Otherwise it reports the blocking codes and refuses
to emit, even when the mode is enabled.

**Synthetic full-coverage** is *allowed* whenever the Annex 2 field universe
loads. It plans a value for all 107 fields, reusing real delivery-valid values
where present (labelled `real_delivery_valid`) and a labelled
`synthetic_schema_test` dummy everywhere else. Dummy values are chosen per format
token (and prefer the first authoritative `enum_map` code) so they pass
type/enum validation where possible.

## Artefacts

Readiness (always written under `preview/`):

```
70_xml_preview_readiness.json        72_xml_preview_policy_application.csv
71_xml_preview_readiness.md          73_xml_preview_assumptions.csv
                                     74_xml_preview_blockers.csv
75_synthetic_schema_test_readiness.json   77_synthetic_schema_field_plan.csv
76_synthetic_schema_test_readiness.md
```

Client preview (only when `client_safe_preview.enabled` **and** allowed):

```
80_client_preview_frame.csv   83_client_preview_exclusions.csv   85_client_preview.xml
81_client_preview_lineage.json 84_client_preview_watermark.txt   86_client_preview_summary.md
82_client_preview_assumptions.csv
```

Synthetic schema test (only when `synthetic_full_coverage_schema_test.enabled`
**and** allowed):

```
90_synthetic_schema_frame.csv   93_synthetic_schema_watermark.txt   94_synthetic_schema_test.xml
91_synthetic_schema_lineage.json 92_synthetic_values_catalog.csv    95_synthetic_schema_summary.md
```

## XML structure — what is and is NOT valid

> **Production XSD mapping remains a blocker.** The production Annex 2 XSD
> (`auth.099.001.04`) requires an XML path / cardinality / element-nesting
> mapping that is **not yet configured** (`xml_emission.production_xsd_mapping_configured: false`).

Therefore both preview XML files are emitted as a **flat, internally-consistent**
structure under an **internal preview namespace** `urn:trakt:nonproduction:preview`,
with `UnderlyingExposure` (RREL) and `Collateral` (RREC) record groups and a
`<Field code= name= source=>` element per field. They are well-formed XML and
clearly watermarked, but they are **not production-XSD-valid** and cannot be
mistaken for a submission.

- `client_preview.xml` = structurally illustrative / internally consistent preview.
- `synthetic_schema_test.xml` = full-field coverage test (every Annex 2 field).
- production XML structure = **still deferred** until the XSD path/cardinality
  mapping is configured.

### Work remaining before production XML can be claimed

1. Configure the Annex 2 XML path / cardinality / element nesting (RREL↔RREC
   nesting, collateral cardinality) against `DRAFT1auth.099.001.04_1.3.0.xsd`.
2. Resolve the production blockers: `client_onboarding_dependency`,
   `operator_or_config_dependency`, `config_dependency`,
   `source_mapping_unresolved`, `delivery_structure_deferred`,
   `template_order_incomplete`.
3. Only then may the production gate (`xml_generation_allowed`,
   `ready_for_xml_delivery`) be revisited — never via the preview path.

## Third mode: `xsd_structured_preview`

A later, non-production mode that places values **inside the real ESMA XSD
hierarchy** (`Document → ScrtstnRpt → UndrlygXpsrRcrd → ResdtlRealEsttLn →
PrfrmgLn → UndrlygXpsrCmonData` + nested `Coll`), using **only builder-accepted**
field-to-XSD paths (see `docs/annex2_path_acceptance_gate.md`). It proves nested
ESMA-path construction — the opposite of the flat preview above — and:

- uses real deliverable values + approved placeholders only; never fabricates
  valuation/rate/economic values; excludes non-accepted (`rejected` /
  `needs_manual_review` / `unresolved` / `conflict`) paths;
- keeps RREC/collateral nested under `Coll`; emits NoDataOptn wrappers only where
  the path map says `value_or_nodata` and the value is a genuine ND sentinel;
- emits children in **XSD sequence order**, and always emits the mandatory
  **report header** (`ScrtstnIdr`, `CutOffDt` — `mandatory_report_header`) and the
  mandatory **leading record siblings** (`NewUndrlygXpsrIdr` before
  `OrgnlUndrlygXpsrIdr`; `ActvtyDtDtls`/`PoolAddtnDt`,`RpDt` before
  `UndrlygXpsrDtls`; `CollIdr`/`OrgnlIdr`,`NewIdr` before `CollCmonData` —
  `structural_mandatory_codes`). It also fills the next required sub-containers:
  the obligor identifiers (`NewOblgrIdr`,`OrgnlOblgrIdr`), `OblgrDtls` (via
  `Resdt`), collateral `Dtls/CmonData` (via `TrtrlUnitClssfctn`,`LienVal`) and
  `Valtn` (via `InfAtOrgtn`/`CurInf` `LnToVal`). When absent from the data these
  use accepted path-map codes filled with **preview-only placeholders**, recorded
  in `101_..._lineage.json` and `102_..._assumptions.csv`. Two kinds:
  - identifier fields → `type/pattern-valid` text placeholders
    (`assumption_kind = mandatory_structural_sibling_placeholder`);
  - non-identifier `value_or_nodata` fields → a **`NoDataOptn` (ND)** placeholder
    valid for that field's ND subset (the builder picks an allowed code per
    field; `assumption_kind = mandatory_structural_nodata_placeholder`). NoData
    asserts **no data** — it never invents an amount, date or enum.
  None of these are production values.
- **Economic fields are never fabricated.** Valuation amounts (`ValtnAmt`),
  income values (`IncmVal`/`PmryOblgrIncm`) and similar are NOT emitted with
  invented numbers; only real deliverable values would be used. Their absence
  leaves honest deeper "missing ValtnAmt" validation errors, reported as such;
- defaults to a small sample (`max_records: 5`) — structure proof, not volume;
- attempts XSD validation and records the result **honestly** in
  `107_xsd_structured_preview_xsd_validation.json` (it is expected to FAIL today —
  incomplete mandatory content, shallow leaf typing, approximate ordering, DRAFT
  schema — all listed under `known_limitations`);
- writes only under `output/delivery_xml/preview/xsd_structured_preview/`
  (artefacts `100..107`, readiness `78..79`); `production_ready` stays false and
  no production gate is changed.

## Fourth mode: `xsd_structured_synthetic_schema_test`

A **separate, engineering-only** mode (disabled by default) that proves the real
ESMA tree can pass full DRAFT-XSD validation. It is deliberately distinct from
the client-safe `xsd_structured_preview` — the preview stays honest and never
fabricates economic values; this mode fabricates dummy values for **everything**
so the schema itself can be exercised.

- recursively builds the **full mandatory** residential-performing tree directly
  from the vendored XSD (`Document → … → ScrtstnRpt → UndrlygXpsrRcrd → … →
  PrfrmgLn → UndrlygXpsrCmonData` + nested `Coll`), choosing residential /
  performing branches and the value branch of value-or-NoData choices;
- fills every mandatory leaf with a **type-valid dummy** (patterns, enums, dates,
  decimals, amounts with required `Ccy`, booleans, integers) via a controlled
  generator; **every value is labelled `synthetic_schema_test`** with a
  `value_reason`/`source_reason` in `112_..._values_catalog.csv`;
- attempts XSD validation and reports it honestly in
  `116_..._xsd_validation.json` (`xsd_validation_attempted`,
  `xsd_validation_passed`, `error_count`, `records_generated`,
  `fields_generated`, `synthetic_values_count`, `validation_errors`,
  `known_limitations`). Against the vendored DRAFT XSD it currently **passes**;
- heavily watermarked, engineering-only, never client-facing, never reportable;
- writes only under `output/delivery_xml/preview/xsd_structured_synthetic_schema_test/`
  (artefacts `110..116`); `production_ready` stays false; no production gate
  changes; no production XML.

These dummy values are NEVER used in the client preview or production, and are
never called real.

## Inspecting

```
python scripts/inspect_delivery_xml_readiness.py <delivery_xml_dir> \
  --preview --synthetic-schema-test --xsd-structured-preview --xsd-structured-synthetic
```

prints production XML readiness, client-preview readiness, synthetic
schema-test readiness, whether any preview/production XML exists, placeholder /
exclusion / synthetic value counts, and the remaining production blockers.
