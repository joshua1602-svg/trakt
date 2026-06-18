#!/usr/bin/env python3
"""
build_minimum_xml_preview_matrix.py
===================================

Generate the field-level minimum XML-preview remediation matrix
(``output/config_review/minimum_xml_preview_remediation_matrix.csv``) from the
current Delivery/XML readiness blocker groups.

This encodes the per-code remediation *classification* (the judgement) as data,
reads the authoritative field labels from ``annex2_field_universe.yaml`` and the
canonical names from ``annex2_delivery_rules.yaml``, and writes a deterministic
CSV. It does NOT generate XML and does NOT mutate any pipeline artefact.

Run:
    python scripts/build_minimum_xml_preview_matrix.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_UNIVERSE = _REPO_ROOT / "config" / "regime" / "annex2_field_universe.yaml"
_REGIME = _REPO_ROOT / "config" / "regime" / "annex2_delivery_rules.yaml"
_OUT = _REPO_ROOT / "output" / "config_review" / "minimum_xml_preview_remediation_matrix.csv"

COLUMNS = [
    "esma_code", "canonical_field", "current_blocker_type", "issue_group",
    "affected_rows", "preview_required", "production_required",
    "recommended_preview_treatment", "recommended_production_treatment",
    "owner", "risk_level", "reason",
]

ROWS_PER_CODE = 1526  # one row per loan; 31 blocked codes x 1526 = 47,306 blocked.

# Per-code classification. Tuple:
#   (code, blocker_type, group, affected_rows, preview_required, production_required,
#    preview_treatment, production_treatment, owner, risk, reason)
# affected_rows: int or "ordering" (order-completeness only, no blocked data rows).
_CLASS = [
    # --- 1. client onboarding (formal regulatory identifiers; NO ND allowed) ---
    ("RREL1", "client_onboarding_dependency", "client_onboarding", ROWS_PER_CODE,
     "yes", "yes", "synthetic_placeholder_for_demo_only", "must_resolve",
     "client_onboarding", "high",
     "Securitisation identifier (ScrtstnIdr) + report anchor; no ND allowed. Must "
     "NOT be guessed. Preview only: a clearly-labelled non-production placeholder; "
     "production requires the formal client identifier policy."),
    ("RREL2", "client_onboarding_dependency", "client_onboarding", ROWS_PER_CODE,
     "yes", "yes", "synthetic_placeholder_for_demo_only", "must_resolve",
     "client_onboarding", "high",
     "Original underlying-exposure identifier; no ND allowed. Same placeholder "
     "rule as RREL1 — non-reportable preview placeholder only."),

    # --- 2. operator review (source ambiguity — must stay operator decisions) ---
    ("RREC1", "operator_or_config_dependency", "operator_review", ROWS_PER_CODE,
     "yes", "yes", "synthetic_placeholder_for_demo_only", "must_resolve",
     "operator", "high",
     "Collateral record unique identifier (no ND); structurally needed to anchor a "
     "collateral record. Preview placeholder only; operator confirms the real id."),
    ("RREC13", "operator_or_config_dependency", "operator_review", ROWS_PER_CODE,
     "no", "yes", "preview_exclusion", "must_resolve", "operator", "medium",
     "Current valuation amount; source ambiguous. Do not fabricate a valuation. "
     "Omit from preview (optional/ND-eligible); operator confirms source for production."),
    ("RREC17", "operator_or_config_dependency", "operator_review", ROWS_PER_CODE,
     "no", "yes", "preview_exclusion", "must_resolve", "operator", "high",
     "Original valuation amount; source ambiguous and NOT ND5-eligible. Never "
     "fabricate. Omit from preview; operator confirms source for production."),
    ("RREC9", "operator_or_config_dependency", "operator_review", ROWS_PER_CODE,
     "no", "yes", "preview_exclusion", "must_resolve", "operator", "medium",
     "Property type; source ambiguity. ND5-eligible but not policy-selected, so do "
     "not ND it. Omit from preview; operator confirms for production."),
    ("RREL43", "operator_or_config_dependency", "operator_review", ROWS_PER_CODE,
     "no", "yes", "preview_exclusion", "must_resolve", "operator", "medium",
     "Current interest rate; source ambiguity. Omit from preview; operator "
     "confirms the rate source for production."),
    ("RREL69", "operator_or_config_dependency", "operator_review", ROWS_PER_CODE,
     "yes", "yes", "must_resolve", "must_resolve", "operator", "high",
     "Account status (LIST, no ND). Cannot ND or silently fill; operator/source "
     "must supply. Required for both preview and production."),
    ("RREL9", "operator_or_config_dependency", "operator_review", ROWS_PER_CODE,
     "no", "yes", "preview_exclusion", "must_resolve", "operator", "low",
     "Redemption date; optional/ND5-eligible. Omit from preview; operator confirms "
     "for production."),

    # --- 3. config mapping (deterministic enum/config — same pattern as RREL35) ---
    ("RREC14", "config_dependency", "config_mapping", ROWS_PER_CODE,
     "yes", "yes", "must_resolve", "must_resolve", "config_policy", "medium",
     "Current valuation method (LIST). Resolve by adding the enum mapping in the "
     "regime contract (deterministic config, like RREL35)."),
    ("RREC16", "config_dependency", "config_mapping", ROWS_PER_CODE,
     "yes", "yes", "must_resolve", "must_resolve", "config_policy", "medium",
     "Original loan-to-value (PERCENTAGE). Resolve the projection/format config."),
    ("RREC7", "config_dependency", "config_mapping", ROWS_PER_CODE,
     "yes", "yes", "must_resolve", "must_resolve", "config_policy", "medium",
     "Occupancy type (LIST). Add the enum mapping in config."),
    ("RREL10", "config_dependency", "config_mapping", ROWS_PER_CODE,
     "yes", "yes", "must_resolve", "must_resolve", "config_policy", "low",
     "Resident (Y/N). Add the boolean/enum mapping in config."),
    ("RREL11", "config_dependency", "config_mapping", ROWS_PER_CODE,
     "yes", "yes", "must_resolve", "must_resolve", "config_policy", "medium",
     "Geographic region - obligor (NUTS). Map source region to NUTS via config."),
    ("RREL14", "config_dependency", "config_mapping", ROWS_PER_CODE,
     "yes", "yes", "must_resolve", "must_resolve", "config_policy", "low",
     "Credit impaired obligor (Y/N). Add the boolean mapping in config."),
    ("RREL26", "config_dependency", "config_mapping", ROWS_PER_CODE,
     "yes", "yes", "must_resolve", "must_resolve", "config_policy", "medium",
     "Origination channel (LIST). Add the enum mapping in config."),
    ("RREL27", "config_dependency", "config_mapping", ROWS_PER_CODE,
     "yes", "yes", "must_resolve", "must_resolve", "config_policy", "medium",
     "Purpose (LIST). Add the purpose enum mapping in config (like RREL35)."),
    ("RREL44", "config_dependency", "config_mapping", ROWS_PER_CODE,
     "yes", "yes", "must_resolve", "must_resolve", "config_policy", "medium",
     "Current interest rate index (LIST). Add the index enum mapping in config."),
    ("RREL45", "config_dependency", "config_mapping", ROWS_PER_CODE,
     "yes", "yes", "must_resolve", "must_resolve", "config_policy", "medium",
     "Current interest rate index tenor (LIST). Add the tenor enum mapping in config."),
    ("RREL75", "config_dependency", "config_mapping", ROWS_PER_CODE,
     "yes", "yes", "must_resolve", "must_resolve", "config_policy", "low",
     "Litigation (Y/N). Add the boolean mapping in config."),

    # --- 4. source / projection mapping ---
    ("RREC2", "source_mapping_unresolved", "source_projection", ROWS_PER_CODE,
     "yes", "yes", "synthetic_placeholder_for_demo_only", "must_resolve",
     "projection", "high",
     "Underlying-exposure identifier on the collateral record (no ND). Preview "
     "placeholder derived from the loan key (demo-only); production needs the "
     "confirmed projection/source mapping."),
    ("RREC3", "source_mapping_unresolved", "source_projection", ROWS_PER_CODE,
     "yes", "yes", "synthetic_placeholder_for_demo_only", "must_resolve",
     "projection", "high",
     "Original collateral identifier (no ND). Preview placeholder only."),
    ("RREC4", "source_mapping_unresolved", "source_projection", ROWS_PER_CODE,
     "yes", "yes", "synthetic_placeholder_for_demo_only", "must_resolve",
     "projection", "high",
     "New collateral identifier (no ND). Preview placeholder only."),
    ("RREC5", "source_mapping_unresolved", "source_projection", ROWS_PER_CODE,
     "yes", "yes", "must_resolve", "must_resolve", "projection", "medium",
     "Collateral type (LIST, no ND). Add the enum mapping; ALSO missing from "
     "esma_code_order (see template_order)."),
    ("RREL3", "source_mapping_unresolved", "source_projection", ROWS_PER_CODE,
     "yes", "yes", "synthetic_placeholder_for_demo_only", "must_resolve",
     "projection", "high",
     "New underlying-exposure identifier (no ND). Preview placeholder only."),
    ("RREL4", "source_mapping_unresolved", "source_projection", ROWS_PER_CODE,
     "yes", "yes", "synthetic_placeholder_for_demo_only", "must_resolve",
     "projection", "high",
     "Original obligor identifier (no ND). Preview placeholder only; ALSO missing "
     "from esma_code_order (see template_order)."),
    ("RREL5", "source_mapping_unresolved", "source_projection", ROWS_PER_CODE,
     "yes", "yes", "synthetic_placeholder_for_demo_only", "must_resolve",
     "projection", "high",
     "New obligor identifier (no ND). Preview placeholder only."),
    ("RREL67", "source_mapping_unresolved", "source_projection", ROWS_PER_CODE,
     "yes", "yes", "must_resolve", "must_resolve", "projection", "medium",
     "Arrears balance (MONETARY, no ND). Confirm the source/derivation; do not "
     "fabricate. ALSO missing from esma_code_order."),
    ("RREL68", "source_mapping_unresolved", "source_projection", ROWS_PER_CODE,
     "yes", "yes", "must_resolve", "must_resolve", "projection", "medium",
     "Number of days in arrears (INTEGER, no ND). Confirm the source/derivation. "
     "ALSO missing from esma_code_order."),
    ("RREL84", "source_mapping_unresolved", "source_projection", ROWS_PER_CODE,
     "yes", "yes", "must_resolve", "must_resolve", "config_policy", "low",
     "Originator establishment country (COUNTRYCODE_2, no ND). Likely known (GB) "
     "via client/config; supply explicitly, do not guess."),

    # --- 5. ND/default policy gap (mislabelled — RREL82 has NO ND allowed) ---
    ("RREL82", "nd_default_rule_missing", "nd_default", ROWS_PER_CODE,
     "yes", "yes", "must_resolve", "must_resolve", "config_policy", "medium",
     "Originator name (ALPHANUM-100, no ND). 'nd_default_rule_missing' is a "
     "misnomer: no ND is permitted, so the fix is to SUPPLY the originator name "
     "from client/config — never an ND or silent fill. ALSO missing from order."),

    # --- 6. delivery structure (separate from data-content blockers) ---
    ("(structural)", "delivery_structure_deferred", "delivery_structure", 163282,
     "yes", "yes", "must_resolve", "must_resolve", "delivery_xml", "high",
     "RREL/RREC record hierarchy, collateral cardinality and header/pool metadata "
     "are still deferred. A MINIMAL flat preview shape (one exposure record per "
     "loan, collateral inline, header once) must be DESIGNED before any preview; "
     "full nesting/cardinality fidelity is a production task."),

    # --- 7. template / order completeness (order-only; no blocked data rows) ---
    ("RREL6", "template_order_incomplete", "template_order", "ordering",
     "yes", "yes", "must_resolve", "must_resolve", "delivery_xml", "low",
     "Data cut-off date — required header metadata, present but missing from "
     "esma_code_order::Record. Must be ordered for the preview."),
    ("RREC21", "template_order_incomplete", "template_order", "ordering",
     "if_included", "yes", "preview_exclusion", "must_resolve", "delivery_xml", "low",
     "Sale price — ordering completeness only; exclude from a minimal preview."),
    ("RREC23", "template_order_incomplete", "template_order", "ordering",
     "if_included", "yes", "preview_exclusion", "must_resolve", "delivery_xml", "low",
     "Guarantor type — ordering completeness only; exclude from a minimal preview."),
    ("RREL62", "template_order_incomplete", "template_order", "ordering",
     "if_included", "yes", "preview_exclusion", "must_resolve", "delivery_xml", "low",
     "Prepayment fee end date — ordering only; exclude from a minimal preview."),
    ("RREL63", "template_order_incomplete", "template_order", "ordering",
     "if_included", "yes", "preview_exclusion", "must_resolve", "delivery_xml", "low",
     "Prepayment date — ordering only; exclude from a minimal preview."),
    ("RREL64", "template_order_incomplete", "template_order", "ordering",
     "if_included", "yes", "preview_exclusion", "must_resolve", "delivery_xml", "low",
     "Cumulative prepayments — ordering only; exclude from a minimal preview."),
    ("RREL65", "template_order_incomplete", "template_order", "ordering",
     "if_included", "yes", "preview_exclusion", "must_resolve", "delivery_xml", "low",
     "Date of restructuring — ordering only; exclude from a minimal preview."),
    ("RREL66", "template_order_incomplete", "template_order", "ordering",
     "if_included", "yes", "preview_exclusion", "must_resolve", "delivery_xml", "low",
     "Date last in arrears — ordering only; exclude from a minimal preview."),
    ("RREL70", "template_order_incomplete", "template_order", "ordering",
     "if_included", "yes", "preview_exclusion", "must_resolve", "delivery_xml", "low",
     "Reason for default/foreclosure — ordering only; exclude from a minimal preview."),
    ("RREL72", "template_order_incomplete", "template_order", "ordering",
     "if_included", "yes", "preview_exclusion", "must_resolve", "delivery_xml", "low",
     "Default date — ordering only; exclude from a minimal preview."),
    ("RREL76", "template_order_incomplete", "template_order", "ordering",
     "if_included", "yes", "preview_exclusion", "must_resolve", "delivery_xml", "low",
     "Recourse — ordering only; ERM recourse is configured (N) but the code must "
     "be ordered if included."),
    ("RREL78", "template_order_incomplete", "template_order", "ordering",
     "if_included", "yes", "preview_exclusion", "must_resolve", "delivery_xml", "low",
     "Insurance/investment provider — ordering only; exclude from a minimal preview."),
    ("RREL79", "template_order_incomplete", "template_order", "ordering",
     "if_included", "yes", "preview_exclusion", "must_resolve", "delivery_xml", "low",
     "Original lender name — ordering only; exclude from a minimal preview."),
    ("RREL80", "template_order_incomplete", "template_order", "ordering",
     "if_included", "yes", "preview_exclusion", "must_resolve", "delivery_xml", "low",
     "Original lender LEI — ordering only; exclude from a minimal preview."),
    ("RREL81", "template_order_incomplete", "template_order", "ordering",
     "if_included", "yes", "preview_exclusion", "must_resolve", "delivery_xml", "low",
     "Original lender establishment country — ordering only; exclude from preview."),

    # --- RESOLVED example (documented as correct remediation) ---
    ("RREL35", "resolved", "resolved_example", ROWS_PER_CODE,
     "no", "no", "not_required_for_preview", "not_required_for_preview",
     "resolved", "none",
     "RESOLVED EXAMPLE: source/internal 'Bullet' -> ERM asset policy 'OTHR' -> "
     "delivery-valid. Authoritative regime code list restored + config-driven ERM "
     "override. 1,526 rows moved from delivery_invalid to deliverable."),
]


def _canonical_lookup():
    regime = yaml.safe_load(_REGIME.read_text(encoding="utf-8")) or {}
    rules = regime.get("field_rules", {}) or {}
    universe = (yaml.safe_load(_UNIVERSE.read_text(encoding="utf-8")) or {}).get("fields", {})
    def canonical(code: str) -> str:
        r = rules.get(code) or {}
        if r.get("projected_source_field"):
            return str(r["projected_source_field"])
        fn = (universe.get(code) or {}).get("field_name", "")
        return str(fn).strip().lower().replace(" ", "_").replace("-", "_") if fn else ""
    return canonical


def build_rows():
    canonical = _canonical_lookup()
    out = []
    for (code, blocker, group, rows, prev_req, prod_req, prev_t, prod_t,
         owner, risk, reason) in _CLASS:
        out.append({
            "esma_code": code,
            "canonical_field": canonical(code) if code != "(structural)" else "record_structure",
            "current_blocker_type": blocker,
            "issue_group": group,
            "affected_rows": rows,
            "preview_required": prev_req,
            "production_required": prod_req,
            "recommended_preview_treatment": prev_t,
            "recommended_production_treatment": prod_t,
            "owner": owner,
            "risk_level": risk,
            "reason": reason,
        })
    return out


def main() -> int:
    rows = build_rows()
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
