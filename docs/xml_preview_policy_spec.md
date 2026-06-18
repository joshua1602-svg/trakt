# XML Preview Policy — Specification (non-production)

This specifies an **explicit, non-production XML preview layer** for the
Delivery/XML Agent. It is a **design + config** deliverable: **no XML is
generated**, and the **production delivery gates are unchanged**.

- Config: `config/delivery/xml_preview_policy.yaml` (disabled by default).
- Field-level source of truth: `output/config_review/minimum_xml_preview_remediation_matrix.csv`
  (regenerate via `python scripts/build_minimum_xml_preview_matrix.py`).
- Companion: `docs/minimum_xml_preview_remediation_plan.md`.

## Gate model — preview is SEPARATE from production

| Concern | Production flags (UNCHANGED) | Preview flags (NEW, separate) |
| --- | --- | --- |
| May generate reportable XML | `xml_generation_allowed` | — |
| Reportable XML written | `xml_generated` | — |
| Ready for regulatory submission | `ready_for_xml_delivery` | — |
| May render a watermarked preview | — | `xml_preview_allowed` |
| Preview rendered | — | `xml_preview_generated` |
| Preview readiness reached | — | `ready_for_xml_preview` |

The preview evaluator reads/writes **only** the preview flags. It must **never**
set or relax a production flag. `xml_preview_allowed = true` does **not** imply
`xml_generation_allowed`; the two are computed independently, and production
readiness keeps its existing, stricter gate logic in
`engine/delivery_xml_agent/delivery_readiness.py` (not modified by this layer).

---

## 1. Purpose of preview XML

To let stakeholders **see the shape** of the Annex 2 `auth.099` XML — element
nesting, ordering, record grouping, namespaces — built from the **deliverable
subset** of the current run, **before** all production blockers are resolved. It
is a demonstration/QA aid, not a submission.

## 2. Difference between preview XML and production XML

| | Preview XML | Production XML |
| --- | --- | --- |
| Gate | `xml_preview_allowed` | `xml_generation_allowed` |
| Reportable | **No** — watermarked, demo-only | Yes |
| Missing identifiers | clearly-labelled placeholders | must be real (client onboarding) |
| Ambiguous valuations/rates | **excluded** (never fabricated) | operator-confirmed source |
| ND | only where explicitly allowed **and** policy-selected | same |
| Output location | `output/delivery_xml/preview/` (separate) | `output/delivery_xml/` |
| Effect on production flags | none | sets production flags |

## 3. Which production blockers can be handled with preview assumptions

Only via the two explicit, audited mechanisms in the policy (derived from the
matrix), never by silent fills:

- **Synthetic placeholders** (`synthetic_placeholder_for_demo_only`) for the
  **no-ND identifier / static-reference** fields that must carry a value for the
  tree to anchor/validate: **RREL1, RREL2, RREC1, RREC2, RREC3, RREC4, RREL3,
  RREL4, RREL5, RREL82**. Values are prefixed `PREVIEW_ONLY_`, watermarked,
  non-reportable, and never promoted to production.
- **Preview exclusions** (`preview_exclusion`) for **optional / ND-eligible /
  deferred** fields and **operator-ambiguous** valuation/rate/property fields:
  **RREC9, RREC13, RREC17, RREL9, RREL43** (operator) and **RREC21, RREC23,
  RREL62–66, RREL70, RREL72, RREL76, RREL78–81** (optional/deferred). These are
  omitted from the minimal preview — never fabricated.

## 4. Which production blockers still block preview

`must_resolve_before_preview` — no placeholder/exclusion is permitted:

- **Deterministic config mappings** (the RREL35 pattern): RREC7, RREC14, RREC16,
  RREL10, RREL11, RREL14, RREL26, RREL27, RREL44, RREL45, RREL75.
- **Source/projection mappings** (no fabrication): RREC5, RREL67, RREL68, RREL84.
- **Mandatory no-ND value**: RREL69 (account status).
- **Delivery structure**: a minimal flat preview record shape must be designed.
- **Required header ordering**: RREL6 (data cut-off date) must be in
  `esma_code_order::Record`.

## 5. How synthetic placeholders are labelled

- Value form: `PREVIEW_ONLY_<rowkey>` (config `placeholder_policy.prefix`).
- Marked `non_reportable: true`; each carries `business_group`, `owner`, `reason`.
- Recorded as a `preview_assumption` row in the preview issues artefact and the
  preview lineage, with the original blocked status preserved.
- **Never** copied into `output/delivery_xml/` or any production artefact.

## 6. How preview exclusions are recorded

- Each excluded field is logged as a `preview_exclusion` entry (code, canonical
  field, business group, reason) in the preview manifest + preview issues.
- Exclusion respects XML multiplicity: a field is only excludable where its
  minimum occurrence is 0 (or it is rendered absent under an allowed choice
  branch). A mandatory element is never silently dropped — it is `must_resolve`.

## 7. How preview artefacts are watermarked

- Every preview XML carries an XML comment header with
  `watermark: "NON-PRODUCTION PREVIEW - NOT FOR REGULATORY SUBMISSION"`.
- The preview manifest sets `mode: non_production_preview`, `reportable: false`.
- Preview files use a distinct name/location
  (`output/delivery_xml/preview/65_xml_preview.xml`, `66_xml_preview_validation.json`)
  so they cannot be mistaken for a production submission.

## 8. How preview never changes production readiness flags

- The preview evaluator is a **separate** function that returns only the preview
  verdict; it does not import or mutate the production readiness result.
- Policy guardrails (enforced in code + asserted in tests):
  `never_set_xml_generation_allowed`, `never_set_ready_for_xml_delivery`,
  `never_set_xml_generated`, `preview_output_must_be_separate`,
  `do_not_promote_placeholders_to_production`.
- With `preview_policy.enabled: false` (the default), the layer is inert and
  nothing is produced.

## 9. Required audit lineage

Each preview run must record, per affected field:

```text
esma_code, canonical_field, business_group, owner,
preview_treatment (synthetic_placeholder_for_demo_only | preview_exclusion),
original_projection_status, original_delivery_status,
placeholder_value (if any, prefixed/watermarked), reason,
matrix_source_row, policy_version, watermark, reportable=false
```

plus run-level lineage chaining `51 → 62 → preview` and an explicit statement
that production flags were not modified. Lineage lives under the separate preview
output dir and references (does not overwrite) `64_delivery_lineage.json`.

## 10. Required test coverage before implementation

Proposed (some implemented now — see "Tests" below):

1. preview policy is **disabled by default** (`enabled: false`).
2. policy field lists **match the matrix** (placeholder/exclusion/must_resolve).
3. RREL82 is `onboarding_static_reference` (not nd_default), placeholder demo-only.
4. RREL35 is **absent** from all preview placeholder/exclusion/must_resolve lists.
5. placeholder values carry the `PREVIEW_ONLY_` prefix and `reportable=false`.
6. enabling preview **never** sets any production flag (guardrail assertions).
7. preview output is written only under the separate preview dir.
8. ND is never used unless explicitly allowed **and** policy-selected.
9. valuation/property/source fields are never fabricated (they are excluded).
10. with blockers present, `ready_for_xml_preview` may be true while
    `ready_for_xml_delivery` stays false (gates do not collapse).

---

## Implementation plan (phased — not built yet)

```text
Phase 1 — Preview readiness evaluator ONLY
  * load config/delivery/xml_preview_policy.yaml (default disabled).
  * compute xml_preview_allowed / ready_for_xml_preview from: deliverable subset
    present + every remaining blocker covered by a placeholder/exclusion + no
    must_resolve_before_preview item outstanding.
  * emit a preview readiness report; write NO XML; touch NO production flag.

Phase 2 — Preview frame builder
  * produce a preview-only frame: deliverable rows as-is + placeholder rows
    (prefixed/watermarked) + excluded fields dropped, with full audit lineage.
  * write under output/delivery_xml/preview/ only.

Phase 3 — XML preview emitter
  * render a watermarked auth.099 tree from the preview frame, behind
    --allow-xml-preview AND preview_policy.enabled AND xml_preview_allowed.
  * never reuse the production builder path to set production flags.

Phase 4 — XSD/structure validation for preview
  * validate the preview tree against the XSD; report shape/order issues; clearly
    label results as preview-only (a pass does NOT imply production readiness).

Phase 5 — Production XML design
  * separate workstream: real identifiers, operator-confirmed sources, full
    RREL/RREC nesting & cardinality, complete code order — gated by the unchanged
    production flags.
```

## Guarantees (acceptance)

- Production gates unchanged; preview gate separate; **disabled by default**.
- Placeholder/exclusion logic is explicit, matrix-derived and auditable.
- **RREL82** = onboarding static reference (no ND; production must_resolve).
- **RREL35** = resolved (Bullet → ERM OTHR), **not** in preview placeholder logic.
- No XML is generated by this spec/config; no ND without policy; no fabrication.
