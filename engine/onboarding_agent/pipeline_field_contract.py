"""
pipeline_field_contract.py
==========================

PART 2 — formalise the *existing, working* Pipeline MI field contract.

The Pipeline MI analytics (``analytics/pipeline_prep.normalize_pipeline_snapshot``)
already consumes a fixed set of pipeline/KFI/origination columns and derives a
canonical pipeline representation from them. This module surfaces that contract
as data so the Onboarding Agent can *bridge* lender KFI/pipeline columns to the
fields the working Pipeline MI already expects — without inventing new fields or
polluting the regulatory/funded-loan registry.

The contract here mirrors the ``rename_map`` + derived columns in
``pipeline_prep`` (a test cross-checks it against the live module). Pipeline
fields are deliberately kept *separate* from the regulatory funded-loan registry
(``fields_registry.yaml``): they feed the central PIPELINE tape, never the
central lender tape.

Artefacts:
    28_existing_pipeline_field_contract.csv / .json / _summary.md
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Raw lender KFI/pipeline header -> pipeline-contract field name (mirrors
# analytics/pipeline_prep.normalize_pipeline_snapshot rename_map).
RAW_TO_PIPELINE_FIELD: Dict[str, str] = {
    "Snapshot Date": "snapshot_date",
    "Company": "company",
    "Pool": "pool",
    "Account Number": "account_number",
    "KFI Number": "kfi_number",
    "Broker": "broker",
    "KFI Submitted Date": "kfi_submitted_date",
    "Loan Amount": "loan_amount",
    "Estimated Value": "estimated_value",
    "Product": "product",
    "Product Rate": "product_rate",
    "Loan Plan": "loan_plan",
    "Facility": "facility",
    "Max Facility": "max_facility",
    "Max Entitlement": "max_entitlement",
    "Property Region": "property_region",
    "PEG Percentage": "peg_percentage",
    "Fees Added": "fees_added",
    "Property Value": "property_value",
    "Loan Purpose": "loan_purpose",
    "Loan Purpose Detail": "loan_purpose_detail",
    "Status": "status_raw",
    "DPR Status": "dpr_status_raw",
    "Application Submitted Date": "application_submitted_date",
    "Offer Date": "offer_date",
    "Date Funds Released": "date_funds_released",
    "Rejection Reason A": "rejection_reason_a",
    "Rejection Reason B": "rejection_reason_b",
    "KFI Used For App": "kfi_used_for_app",
    "Contracted Payment Period": "contracted_payment_period",
    "Interest Payment Percentage": "interest_payment_percentage",
    "Borrower Age": "borrower_age",
    "Youngest Borrower Age": "youngest_borrower_age",
}

# field -> (type, purpose, expected_values_or_range, required, derived)
_FIELD_META: Dict[str, Dict[str, Any]] = {
    "snapshot_date": {"type": "date", "purpose": "weekly pipeline snapshot date", "required": True},
    "company": {"type": "string", "purpose": "originating company / lender"},
    "pool": {"type": "string", "purpose": "pool / programme identifier"},
    "account_number": {"type": "identifier", "purpose": "lender account / case number"},
    "kfi_number": {"type": "identifier", "purpose": "Key Facts Illustration reference"},
    "broker": {"type": "string", "purpose": "introducing broker / intermediary"},
    "kfi_submitted_date": {"type": "date", "purpose": "date the KFI was submitted",
                            "range": "<= application_submitted_date"},
    "application_submitted_date": {"type": "date", "purpose": "application submitted milestone",
                                    "range": ">= kfi_submitted_date; <= offer_date"},
    "offer_date": {"type": "date", "purpose": "offer issued milestone",
                    "range": ">= application_submitted_date; <= date_funds_released"},
    "date_funds_released": {"type": "date", "purpose": "completion / funds released milestone",
                             "range": ">= offer_date"},
    "loan_amount": {"type": "numeric", "purpose": "requested / pipeline loan amount",
                     "range": "<= max_facility <= max_entitlement", "required": True},
    "estimated_value": {"type": "numeric", "purpose": "estimated property value",
                         "range": ">= loan_amount"},
    "property_value": {"type": "numeric", "purpose": "property value"},
    "product": {"type": "string", "purpose": "product name"},
    "product_rate": {"type": "rate", "purpose": "headline product interest rate",
                      "range": "~0-15 (percent)"},
    "interest_payment_percentage": {"type": "percentage",
                                     "purpose": "interest-servicing percentage (0/50/100)",
                                     "range": "0-100"},
    "loan_plan": {"type": "string", "purpose": "product plan variant"},
    "facility": {"type": "numeric", "purpose": "current facility amount"},
    "max_facility": {"type": "numeric", "purpose": "maximum facility amount"},
    "max_entitlement": {"type": "numeric", "purpose": "maximum entitlement amount"},
    "property_region": {"type": "string", "purpose": "property region (display geography)"},
    "property_region_code": {"type": "string", "purpose": "derived region code", "derived": True},
    "peg_percentage": {"type": "percentage", "purpose": "PEG percentage"},
    "fees_added": {"type": "numeric", "purpose": "fees added to the loan"},
    "loan_purpose": {"type": "string", "purpose": "loan purpose"},
    "loan_purpose_detail": {"type": "string", "purpose": "loan purpose detail (free text)"},
    "status_raw": {"type": "enum", "purpose": "raw pipeline status",
                    "values": "KFI/APPLICATION/OFFER/COMPLETED/WITHDRAWN/...", "required": True},
    "dpr_status_raw": {"type": "enum", "purpose": "DPR (decision in principle) status"},
    "rejection_reason_a": {"type": "string", "purpose": "primary rejection reason"},
    "rejection_reason_b": {"type": "string", "purpose": "secondary rejection reason"},
    "kfi_used_for_app": {"type": "boolean", "purpose": "whether the KFI was used for application"},
    "contracted_payment_period": {"type": "numeric", "purpose": "contracted payment period (months)"},
    "borrower_age": {"type": "numeric", "purpose": "borrower age"},
    "youngest_borrower_age": {"type": "numeric", "purpose": "youngest borrower age"},
    # Derived by pipeline_prep (not mapped from source).
    "stage": {"type": "enum", "purpose": "canonical pipeline stage", "derived": True,
               "values": "KFI/APPLICATION/OFFER/COMPLETED/WITHDRAWN/OTHER"},
    "pipeline_stage": {"type": "enum", "purpose": "alias of stage", "derived": True},
    "is_funded_stage": {"type": "boolean", "purpose": "stage == COMPLETED", "derived": True},
    "is_terminal_stage": {"type": "boolean", "purpose": "completed/withdrawn", "derived": True},
    "is_live_stage": {"type": "boolean", "purpose": "still live", "derived": True},
    "termination_reason_class": {"type": "enum", "purpose": "withdrawn/rejected class", "derived": True},
    "pipeline_opportunity_id": {"type": "identifier", "purpose": "synthetic opportunity key",
                                 "derived": True},
    "stage_date": {"type": "date", "purpose": "date of current stage", "derived": True},
    "days_in_stage": {"type": "numeric", "purpose": "days since stage_date", "derived": True},
    "current_ltv": {"type": "numeric", "purpose": "loan_amount / estimated_value", "derived": True},
}

# Fields the Pipeline MI tab/charts treat as core inputs (used_in references).
_USED_IN = {
    "snapshot_date": "pipeline tab (snapshot selector)",
    "stage": "stage funnel chart; forward exposure",
    "pipeline_stage": "pipeline tab",
    "loan_amount": "expected funding; forward exposure",
    "estimated_value": "current_ltv derivation",
    "product_rate": "weighted rate MI",
    "property_region": "regional pipeline MI",
    "broker": "broker MI",
    "product": "product MI",
    "days_in_stage": "ageing MI",
    "date_funds_released": "completion / funded MI",
    "offer_date": "offer pipeline MI",
    "application_submitted_date": "application pipeline MI",
    "kfi_submitted_date": "KFI pipeline MI",
    "pipeline_opportunity_id": "opportunity keying / reconciliation",
}


def _load_registry_field_names(registry_path: str | Path) -> set:
    data = yaml.safe_load(Path(registry_path).read_text(encoding="utf-8")) or {}
    return set((data.get("fields", {}) or {}).keys())


def build_pipeline_field_contract(
    registry_path: str | Path = "config/system/fields_registry.yaml",
    pipeline_registry_path: str | Path = "config/system/fields_registry_pipeline.yaml",
) -> List[Dict[str, Any]]:
    """Build the existing Pipeline MI field contract as a list of records."""
    reg_fields = _load_registry_field_names(registry_path)
    pipe_fields: set = set()
    p = Path(pipeline_registry_path)
    if p.exists():
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        pipe_fields = set((data.get("fields", {}) or {}).keys())

    raw_by_field: Dict[str, str] = {}
    for raw, fld in RAW_TO_PIPELINE_FIELD.items():
        raw_by_field.setdefault(fld, raw)

    rows: List[Dict[str, Any]] = []
    for field, meta in _FIELD_META.items():
        derived = bool(meta.get("derived"))
        in_reg = field in reg_fields
        in_pipe_reg = field in pipe_fields
        if in_reg:
            status = "in_funded_registry"
            action = "none"
        elif in_pipe_reg:
            status = "in_pipeline_registry_extension"
            action = "none"
        else:
            status = "missing_from_registry"
            action = ("derived_no_action" if derived
                      else "add_to_pipeline_registry_extension")
        rows.append({
            "field_name": field,
            "raw_source_header": raw_by_field.get(field, ""),
            "source_module": "analytics/pipeline_prep.py",
            "used_in_tab_or_chart": _USED_IN.get(field, "pipeline normalization"),
            "required_or_optional": "required" if meta.get("required") else "optional",
            "field_purpose": meta.get("purpose", ""),
            "expected_type": meta.get("type", "string"),
            "expected_values_or_range": meta.get("values") or meta.get("range", ""),
            "is_pipeline_field": True,
            "is_funded_field": in_reg and not derived,
            "is_derived": derived,
            "current_registry_status": status,
            "recommended_registry_action": action,
        })
    return rows


_CONTRACT_COLUMNS = [
    "field_name", "raw_source_header", "source_module", "used_in_tab_or_chart",
    "required_or_optional", "field_purpose", "expected_type",
    "expected_values_or_range", "is_pipeline_field", "is_funded_field",
    "is_derived", "current_registry_status", "recommended_registry_action",
]


def write_contract_artifacts(
    rows: List[Dict[str, Any]], output_dir: str | Path
) -> Dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "28_existing_pipeline_field_contract.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_CONTRACT_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in _CONTRACT_COLUMNS})
    json_path = out_dir / "28_existing_pipeline_field_contract.json"
    json_path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    md_path = out_dir / "28_existing_pipeline_field_contract_summary.md"
    md_path.write_text(_render_md(rows), encoding="utf-8")
    return {"csv": str(csv_path), "json": str(json_path), "summary_md": str(md_path)}


def _render_md(rows: List[Dict[str, Any]]) -> str:
    missing = [r for r in rows if r["current_registry_status"] == "missing_from_registry"
               and not r["is_derived"]]
    derived = [r for r in rows if r["is_derived"]]
    in_reg = [r for r in rows if r["is_funded_field"]]
    lines = ["# Existing Pipeline MI field contract", ""]
    lines.append("Source of truth: `analytics/pipeline_prep.normalize_pipeline_snapshot` "
                 "(the working Pipeline MI already consumes these columns).")
    lines.append("")
    lines.append(f"- {len(rows)} pipeline-contract fields in total.")
    lines.append(f"- {len(in_reg)} already exist in the funded-loan registry.")
    lines.append(f"- {len(derived)} are derived by pipeline_prep (no source mapping needed).")
    lines.append(f"- {len(missing)} are pipeline-specific and NOT in any registry yet "
                 f"(recommend a pipeline registry extension).")
    lines.append("")
    lines.append("## Pipeline-specific fields needing a canonical home")
    lines.append("| field | raw header | type | purpose | action |")
    lines.append("| --- | --- | --- | --- | --- |")
    for r in missing:
        lines.append(f"| {r['field_name']} | {r['raw_source_header']} | {r['expected_type']} "
                     f"| {r['field_purpose']} | {r['recommended_registry_action']} |")
    lines.append("")
    lines.append("> Pipeline fields feed the **central pipeline tape**, never the central "
                 "lender (funded-loan) tape. Regulatory/funded fields are never created "
                 "from pipeline inference.")
    return "\n".join(lines) + "\n"


def pipeline_contract_field_names() -> List[str]:
    """All pipeline-contract field names (mapped + derived)."""
    return list(_FIELD_META.keys())


def pipeline_target_for_raw(raw_header: str) -> str:
    """Return the pipeline-contract field for a raw header, or '' if none."""
    return RAW_TO_PIPELINE_FIELD.get(raw_header, "")
