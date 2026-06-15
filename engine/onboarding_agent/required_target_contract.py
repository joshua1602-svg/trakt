"""
required_target_contract.py
===========================

v1 required target-data-contract builder.

Given the detected onboarding context (asset class / regime / use cases), build
the FIXED required dataset the source files must be mapped onto. The LLM resolver
maps source columns to THIS contract (not a generic field universe), and the
deterministic backstop validates against it.

For UK equity-release MI the contract spans funded-loan, borrower, collateral/
property, cashflow/ledger and pipeline domains — MI is NOT pipeline-only.

Artefacts:
    28_required_target_contract.csv / .json / _summary.md
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

# required_level: mandatory | conditional | optional
# Each entry: (target_field, domain, required_level, expected_type, synonyms, description)
_EQUITY_RELEASE_MI: List[tuple] = [
    # --- funded loan ---
    ("loan_identifier", "funded_loan", "mandatory", "identifier",
     ["loan id", "account number", "loan policy number", "policy number", "kfi number"],
     "Unique loan / account / policy identifier."),
    ("original_underlying_exposure_identifier", "funded_loan", "conditional", "identifier",
     ["loan policy number", "policy number", "exposure id"],
     "Stable underlying exposure / policy identifier."),
    ("pool", "funded_loan", "optional", "string", ["pool", "programme", "tranche"],
     "Pool / programme grouping."),
    ("company", "funded_loan", "optional", "string", ["company", "funder", "originator", "lender"],
     "Originating company / funder."),
    ("product", "funded_loan", "optional", "string", ["product", "product name"], "Product name."),
    ("product_type", "funded_loan", "optional", "string",
     ["product type", "loan type", "plan"], "Product type / subtype / plan."),
    ("current_interest_rate", "funded_loan", "mandatory", "rate",
     ["interest rate", "product rate", "current rate", "coupon"], "Current interest rate."),
    ("original_principal_balance", "funded_loan", "mandatory", "amount",
     ["original principal", "original balance", "original advance", "initial advance"],
     "Original principal / advance amount."),
    ("current_principal_balance", "funded_loan", "mandatory", "amount",
     ["current balance", "principal outstanding", "outstanding balance"],
     "Current principal balance."),
    ("origination_date", "funded_loan", "mandatory", "date",
     ["origination date", "completion date", "start date", "date funds released"],
     "Origination / completion date."),
    ("maturity_date", "funded_loan", "conditional", "date", ["maturity date", "end date"],
     "Maturity date."),
    # --- borrower ---
    ("date_of_birth", "borrower", "optional", "date", ["dob", "date of birth", "birth date"],
     "Borrower date of birth (where useful)."),
    ("gender", "borrower", "optional", "enum", ["gender", "sex"], "Borrower gender (where useful)."),
    ("employment_status", "borrower", "optional", "enum", ["employment status", "occupation"],
     "Borrower employment status."),
    # --- collateral / property ---
    ("current_valuation_amount", "collateral_property", "mandatory", "amount",
     ["current valuation", "latest property value", "current property value", "valuation amount"],
     "Latest property valuation."),
    ("original_valuation_amount", "collateral_property", "conditional", "amount",
     ["original property value", "original valuation", "valuation at origination"],
     "Property valuation at origination."),
    ("property_post_code", "collateral_property", "conditional", "postcode",
     ["post code", "postcode", "property post code"], "Property postcode."),
    ("collateral_geography", "collateral_property", "optional", "string",
     ["property region", "region", "collateral region"], "Property region (display geography)."),
    ("current_loan_to_value", "collateral_property", "conditional", "percentage",
     ["ltv", "current ltv", "loan to value"], "Current loan-to-value ratio."),
    ("property_type", "collateral_property", "optional", "enum",
     ["property type", "dwelling type"], "Property type."),
    # --- redemption ---
    ("redemption_date", "funded_loan", "conditional", "date",
     ["redemption date", "full redemption date", "repaid date"], "Loan redemption date."),
    ("account_status", "funded_loan", "conditional", "enum",
     ["status", "loan status", "account status", "dpr status"], "Account / loan status."),
    # --- cashflow / ledger ---
    ("cf_bf_current_balance", "cashflow_ledger", "conditional", "amount",
     ["b/f current balance", "opening balance", "brought forward balance"],
     "Brought-forward current balance."),
    ("cf_cf_current_balance", "cashflow_ledger", "conditional", "amount",
     ["c/f current balance", "closing balance", "carried forward balance"],
     "Carried-forward current balance."),
    ("cf_bf_principal_balance", "cashflow_ledger", "conditional", "amount",
     ["b/f principal balance", "opening principal"], "Brought-forward principal balance."),
    ("cf_cf_principal_balance", "cashflow_ledger", "conditional", "amount",
     ["c/f principal balance", "closing principal"], "Carried-forward principal balance."),
    ("cf_bf_interest_balance", "cashflow_ledger", "optional", "amount",
     ["b/f interest balance", "opening interest"], "Brought-forward interest balance."),
    ("cf_cf_interest_balance", "cashflow_ledger", "optional", "amount",
     ["c/f interest balance", "closing interest"], "Carried-forward interest balance."),
    ("cf_payment_allocation_principal", "cashflow_ledger", "optional", "amount",
     ["payment_allocation - principal", "principal allocated", "principal paid"],
     "Principal payment allocation in the period."),
    ("cf_payment_allocation_interest", "cashflow_ledger", "optional", "amount",
     ["payment_allocation - interest", "interest allocated", "interest paid"],
     "Interest payment allocation in the period."),
    ("cf_payment_allocation_fees", "cashflow_ledger", "optional", "amount",
     ["payment_allocation - fees", "fees allocated"], "Fees payment allocation in the period."),
    ("cf_redemptions_received", "cashflow_ledger", "optional", "amount",
     ["redemptions received in period", "redemptions", "recoveries"],
     "Redemptions / recoveries received in the period."),
    ("cf_cash_paid_out", "cashflow_ledger", "optional", "amount",
     ["cash paid out in the period", "cash out"], "Cash paid out in the period."),
    # --- pipeline / origination ---
    ("kfi_number", "pipeline", "optional", "identifier", ["kfi number"], "KFI reference."),
    ("application_submitted_date", "pipeline", "optional", "date",
     ["application submitted date", "application date"], "Application submitted date."),
    ("offer_date", "pipeline", "optional", "date", ["offer date"], "Offer date."),
    ("date_funds_released", "pipeline", "optional", "date",
     ["date funds released", "completion date", "funds released"], "Funds released / completion date."),
    ("broker", "pipeline", "optional", "string", ["broker", "intermediary"], "Introducing broker."),
    ("pipeline_stage", "pipeline", "optional", "enum", ["status", "stage", "dpr status"],
     "Pipeline stage / status."),
    # --- concentration limits ---
    ("concentration_limit", "concentration_limits", "optional", "string",
     ["concentration limit", "covenant", "limit"], "Concentration / covenant limit."),
]


def build_required_contract(context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build the required target contract for the detected context."""
    asset = context.get("asset_class", "equity_release_mortgage")
    regime = context.get("reporting_regime", "mi_only")
    required_domains = set(context.get("required_domains", []) or [])
    use_cases = context.get("use_cases", []) or []

    rows: List[Dict[str, Any]] = []
    for (field, domain, level, etype, synonyms, desc) in _EQUITY_RELEASE_MI:
        # Mandatory only stays mandatory if its domain is required for this run.
        eff_level = level
        if level == "mandatory" and required_domains and domain not in required_domains:
            eff_level = "conditional"
        rows.append({
            "target_field": field,
            "domain": domain,
            "required_level": eff_level,
            "use_case": "; ".join(use_cases),
            "asset_class_applicability": asset,
            "regime_applicability": regime,
            "expected_type": etype,
            "accepted_value_profile": etype,
            "synonyms": "; ".join(synonyms),
            "description": desc,
        })
    return rows


def contract_field_set(contract: List[Dict[str, Any]]) -> set:
    return {r["target_field"] for r in contract}


def contract_domain_map(contract: List[Dict[str, Any]]) -> Dict[str, str]:
    return {r["target_field"]: r["domain"] for r in contract}


def mandatory_fields(contract: List[Dict[str, Any]]) -> set:
    return {r["target_field"] for r in contract if r["required_level"] == "mandatory"}


def synonym_index(contract: List[Dict[str, Any]]) -> Dict[str, str]:
    """normalized synonym -> target field (for deterministic contract matching)."""
    import re
    idx: Dict[str, str] = {}
    for r in contract:
        for syn in (r["synonyms"].split("; ") if r["synonyms"] else []):
            key = re.sub(r"[^a-z0-9]+", " ", syn.lower()).strip()
            if key and key not in idx:
                idx[key] = r["target_field"]
    return idx


_CONTRACT_COLUMNS = [
    "target_field", "domain", "required_level", "use_case", "asset_class_applicability",
    "regime_applicability", "expected_type", "accepted_value_profile", "synonyms",
    "description",
]


def write_contract_artifacts(rows: List[Dict[str, Any]], output_dir: str | Path) -> Dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "28_required_target_contract.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_CONTRACT_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in _CONTRACT_COLUMNS})
    json_path = out_dir / "28_required_target_contract.json"
    json_path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    md = ["# Required target data contract", ""]
    by_domain: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_domain.setdefault(r["domain"], []).append(r)
    md.append(f"{len(rows)} target fields across {len(by_domain)} domains. "
              f"{len(mandatory_fields(rows))} mandatory.")
    md.append("")
    for dom, frs in by_domain.items():
        md.append(f"## {dom} ({len(frs)})")
        for r in frs:
            md.append(f"- `{r['target_field']}` · {r['required_level']} · {r['expected_type']} "
                      f"— {r['description']}")
        md.append("")
    md_path = out_dir / "28_required_target_contract_summary.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return {"csv": str(csv_path), "json": str(json_path), "summary_md": str(md_path)}
