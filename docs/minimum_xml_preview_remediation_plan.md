# Minimum XML-Preview Remediation Plan (Annex 2)

A practical, prioritised plan to reach a **controlled, non-production XML
preview** from the current Delivery/XML readiness output. **No XML is generated
yet**, and this plan **does not weaken the production delivery gates**.

> Companion artefact: `output/config_review/minimum_xml_preview_remediation_matrix.csv`
> (field-level tracking; regenerate with `python scripts/build_minimum_xml_preview_matrix.py`).
> RREL35 is included as a **resolved example** of correct remediation.

## Current state (real run after RREL35 fix)

```text
projection_complete = false ; ready_for_delivery_normalisation = false
delivery_xml_ran = true ; delivery_normalisation_complete = false
xml_generation_allowed = false ; xml_generated = false
delivery frame rows = 163,282
  deliverable        = 85,456   (incl. RREL35 1,526 now valid)
  blocked            = 47,306   (31 codes x 1,526)
  not_required_blank = 30,520
  delivery_invalid   = 0        (RREL35 resolved)
delivery issues = 33
next_agent = operator_config_projection_remediation
```

Issue mix: client_onboarding 2 · operator_or_config 7 · config 11 ·
source_mapping 10 · nd_default 1 · delivery_structure 1 · template_order 1.

> The single `nd_default_rule_missing` issue (**RREL82 originator_name**) is a
> **technical gate label only**. Its business remediation group is
> **onboarding_static_reference** (originator name captured at onboarding), **not**
> ND/default — no ND is permitted for RREL82. There are therefore **no genuine
> ND/default-rule gaps** in this run.

## Guard-rails (non-negotiable)

- **No silent fills.** A blank is never quietly populated.
- **No ND unless explicitly allowed AND policy-selected.** ND is not a preview shortcut.
- **No fake values for production.** Synthetic values are **preview/demo-only and
  non-reportable**, and must be clearly labelled as such.
- **Client identifiers (RREL1/RREL2) are never guessed.** They remain client
  onboarding dependencies; a preview may use a clearly non-production placeholder
  strategy only if that strategy is explicitly marked non-production.
- **Valuation/rate/property/status ambiguity (RREC17/RREC13/RREC9/RREL43, RREC1,
  RREL69, RREL9) stays an operator decision** — never fabricated.
- **XML structure/cardinality/header design is separated from data-content
  blockers** (see groups 6 vs 1–5).
- **Template/order is assessed separately** (group 7): it gates *ordering*, not data.
- **XML generation stays disabled** until a preview policy is explicitly implemented.

---

## The ten questions

### 1. Which blockers must be fixed before any XML preview?
Anything a preview cannot honestly show or structurally omit:
- **Delivery structure (group 6)** — a minimal preview record shape must be designed.
- **Template/order for included codes (group 7)** — required codes (notably
  **RREL6** data cut-off) must be ordered.
- **Config-mapping codes that are structurally part of the preview subset (group 3)** —
  these are deterministic config (the RREL35 pattern) and should simply be resolved.
- **Mandatory, no-ND fields with no value** that cannot be excluded
  (**RREL69** account status; the identifier family — see Q2/Q4).

### 2. Which blockers can be handled by explicit preview assumptions?
Handled by a **declared, labelled** assumption (not a silent fill):
- **Identifiers with no ND** (RREL1, RREL2, RREC1, RREC2, RREC3, RREC4, RREL3,
  RREL4, RREL5): `synthetic_placeholder_for_demo_only` — a clearly non-production,
  non-reportable placeholder (e.g. `PREVIEW-NONPROD-<rowkey>`), used **only** to
  let the tree anchor/validate for demonstration.
- Optional/ND-eligible operator fields: handled by **`preview_exclusion`** (omit),
  not by ND.

### 3. Which blockers can be deferred until production XML?
- **Full RREL↔RREC nesting & collateral cardinality and header/pool fidelity**
  (group 6 production half).
- **Optional performance/restructuring/lender fields** (group 7 `preview_exclusion`
  codes: RREC21, RREC23, RREL62–66, RREL70, RREL72, RREL76, RREL78–81) — excluded
  from a minimal preview, completed for production.

### 4. Which blockers require client/lender input?
- **RREL1, RREL2** (formal identifiers; no ND).
- **RREL82 originator_name** — **onboarding static-reference data**: captured
  during the Onboarding Agent step / static client configuration. It carries the
  technical gate label `nd_default_rule_missing`, but it is **not** an ND/default
  item — **no ND is allowed** for RREL82, so it must never be solved with an ND or
  a silent fill. It must be supplied from onboarding/config, not invented.
- **RREL84** (originator establishment country) — likewise known to the
  client/lender and supplied via config, not guessed.

### 5. Which blockers require operator review?
- **RREC1, RREC9, RREC13, RREC17, RREL9, RREL43, RREL69** — source ambiguity on
  collateral id, property type, valuations, redemption date, interest rate and
  account status. These remain operator decisions for production.

### 6. Which blockers require config/policy decisions?
- **All 11 config-mapping codes** (RREC7, RREC14, RREC16, RREL10, RREL11, RREL14,
  RREL26, RREL27, RREL44, RREL45, RREL75) — deterministic enum/boolean/NUTS
  mappings, resolved exactly like RREL35 (regime `enum_map` + asset policy where needed).

### 7. Which blockers require source/projection mapping?
- **RREC2, RREC3, RREC4, RREC5, RREL3, RREL4, RREL5, RREL67, RREL68, RREL84** —
  identifier chains, collateral type, arrears balance/days, originator country.
  Need an explicit projection/source rule (never guessed).

### 8. Which blockers require delivery XML structural design?
- **Group 6 only.** RREL/RREC hierarchy, collateral cardinality, header/pool
  metadata. A **minimal flat preview shape** (one `UndrlygXpsrRcrd` per loan,
  collateral inline, header once) is the preview prerequisite; full nesting is
  production.

### 9. Which blockers are merely template ordering completeness issues?
- **Group 7** — 20 codes missing from `esma_code_order::Record`. **RREL6** (header
  cut-off date) blocks the preview; the other 19 are order-completeness only and
  can be `preview_exclusion`ed and completed for production. Ordering matters
  **only for codes included** in the preview subset.

### 10. What is the smallest safe path to a preview XML?

```text
A. DESIGN a minimal flat preview record shape (group 6).            [delivery_xml]
B. COMPLETE esma_code_order for the codes the preview INCLUDES       [delivery_xml]
   (at minimum RREL6 + the deliverable backbone).
C. BUILD the preview from the 85,456 deliverable rows only.         [delivery_xml]
D. For mandatory no-ND IDENTIFIERS needed to anchor records
   (RREL1/RREL2/RREC1/RREC2/RREC3/RREC4/RREL3/RREL4/RREL5):
   inject CLEARLY-LABELLED non-production placeholders.              [preview policy]
E. EXCLUDE every optional operator/config/source field not yet
   resolved (preview_exclusion) — never fabricate valuations/rates.  [preview policy]
F. RESOLVE the deterministic config-mapping codes (group 3) and
   RREL69 if they fall inside the preview subset.                   [config_policy/operator]
G. EMIT the preview ONLY behind an explicit, separate preview policy
   (a preview_policy.yaml + the existing --allow-xml-preview), which
   satisfies a PREVIEW readiness — NOT production readiness. The
   production gates and xml_generation_allowed stay unchanged.       [delivery_xml]
```

The preview is a **demonstration of shape**, watermarked non-reportable. It is
emitted only when steps A–G are implemented; until then XML generation remains
disabled.

---

## Preview vs production readiness (how this stays safe)

The production gates in `engine/delivery_xml_agent/delivery_readiness.py` are
**not modified**. Instead, a future, explicit *preview policy* introduces a
**separate** readiness verdict (`xml_preview_allowed`) that:

1. requires the deliverable backbone + a designed preview shape (A–C);
2. accepts **labelled** placeholders/exclusions (D–E) as *preview* satisfaction only;
3. records every placeholder/exclusion in `63_delivery_issues` and the preview
   manifest as `preview_assumption` / `preview_exclusion`, so the preview is fully
   auditable and obviously non-production;
4. **never** sets `ready_for_xml_delivery` or `xml_generation_allowed` (production)
   to true.

`--allow-xml-preview` continues to write nothing unless that preview policy is
present and satisfied — it does not bypass the production gates today.

## Preview-treatment legend

| Treatment | Meaning |
| --- | --- |
| `must_resolve` | Fix before preview (deterministic config, structure, ordering, or mandatory no-ND value). |
| `explicit_preview_assumption` | Declared, labelled assumption (e.g. policy-selected ND) — not a silent fill. |
| `preview_exclusion` | Omit from a minimal preview (optional / ND-eligible / deferred). |
| `synthetic_placeholder_for_demo_only` | Clearly non-production, non-reportable placeholder to anchor/validate the tree. |
| `defer_until_production` | Not needed for preview; required for production. |
| `not_required_for_preview` | Already resolved or out of scope (e.g. RREL35). |

## Field-level matrix (summary)

| Group | Codes | Preview treatment | Owner |
| --- | --- | --- | --- |
| client_onboarding | RREL1, RREL2 | synthetic_placeholder_for_demo_only | client_onboarding |
| operator_review | RREC1 (placeholder); RREC13, RREC17, RREC9, RREL43, RREL9 (exclude); RREL69 (must_resolve) | mixed | operator |
| config_mapping | RREC7, RREC14, RREC16, RREL10, RREL11, RREL14, RREL26, RREL27, RREL44, RREL45, RREL75 | must_resolve (config) | config_policy |
| source_projection | RREC2/3/4, RREL3/4/5 (placeholder); RREC5, RREL67, RREL68, RREL84 (must_resolve) | mixed | projection / config_policy |
| onboarding_static_reference | RREL82 originator_name (no ND → captured at onboarding; preview placeholder is demo-only) | synthetic_placeholder_for_demo_only (preview) / must_resolve (production) | client_onboarding / onboarding_agent |
| delivery_structure | record hierarchy / header / cardinality | must_resolve (minimal preview shape) | delivery_xml |
| template_order | RREL6 (must_resolve); 19 others (preview_exclusion) | mixed | delivery_xml |
| resolved_example | RREL35 | not_required_for_preview | resolved |

The full per-code matrix (with `affected_rows`, `risk_level`, production
treatment and reasons) is in
`output/config_review/minimum_xml_preview_remediation_matrix.csv`.

## RREL35 — worked example of correct remediation

`RREL35 amortisation_type` was the prior `delivery_format_invalid` code. It was
**not** patched with a silent fill or an unapproved ND:

```text
source/internal "Bullet"
  -> ERM asset policy (config/asset/product_defaults_ERM.yaml: enum_overrides) "OTHR"
  -> regime enum_map recognises OTHR (authoritative RREL35 code list restored)
  -> delivery-valid (1,526 rows: delivery_invalid -> deliverable)
```

This is the template for the config-mapping group (3): restore the authoritative
regime code list and add an explicit, config-driven asset/client policy where the
asset class genuinely differs. See `docs/rrel35_amortisation_type_remediation.md`.
