#!/usr/bin/env python3
"""
build_mi_semantics_registry.py

Generate the *curated* MI semantic field registry from the canonical field
registry.

This is an ADDITIVE tool.  It only READS the canonical registry
(config/system/fields_registry.yaml) and WRITES a new, separate file:

    mi_agent/mi_semantics_field_registry.yaml

DESIGN (v0.2 — curated, MI-first)
---------------------------------
v0.1 selected ~235 fields via a broad rule (core_canonical OR
layer in {performance,product} OR category == analytics).  That re-projected
half of the regulatory registry and produced 24 unclassifiable fields and many
duplicate concepts (see mi_agent/reports/mi_semantics_review.md).

v0.2 replaces that with an EXPLICIT CURATED ALLOWLIST (see ``CURATION`` below):
a small (~40-80) MI vocabulary chosen for portfolio management reporting.  Each
curated field carries MI business metadata (business_name, business_description,
synonyms) and an ``mi_tier`` (core | extended).  Identifiers, LEIs, industry /
tax codes, rating-agency equivalents, waterfall / swap fields, balance-period
buckets and duplicate borrower-2 / guarantor fields are deliberately excluded.

The generated file does NOT duplicate the canonical registry: every entry
REFERENCES a canonical field by name and adds analytics + business metadata.
Format / role / aggregation / chart-role inference is heuristic; per-field
``overrides`` in CURATION pin the cases the heuristic gets wrong.

Usage:
    python -m mi_agent.build_mi_semantics_registry
    python -m mi_agent.build_mi_semantics_registry \
        --source config/system/fields_registry.yaml \
        --output mi_agent/mi_semantics_field_registry.yaml
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "config" / "system" / "fields_registry.yaml"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "mi_semantics_field_registry.yaml"

VERSION = "0.2.1"

CLEANUP_NOTES = [
    "numeric axis roles enabled",
    "weighted_avg defaults for rate/LTV fields",
    "monetary performance fields normalised",
    "YAML aliases disabled",
    "core-tier preference added to parser",
]


class _NoAliasDumper(yaml.SafeDumper):
    """yaml.safe_dump emits anchors when the same list/dict object is referenced
    twice (e.g. shared ``["curated"]`` source_criteria). For human / downstream
    readability we disable anchors entirely."""

    def ignore_aliases(self, data):  # pragma: no cover - trivial
        return True

# Acronyms that should be upper-cased in display names.
_ACRONYMS = {
    "ltv": "LTV", "lei": "LEI", "id": "ID", "lgd": "LGD", "pd": "PD",
    "isin": "ISIN", "nuts": "NUTS", "abcp": "ABCP", "esma": "ESMA",
    "boe": "BoE", "erm": "ERM", "iban": "IBAN", "yn": "Y/N",
    "dscr": "DSCR", "dti": "DTI", "noi": "NOI", "dob": "DOB",
}

# --------------------------------------------------------------------------- #
# CURATED MI VOCABULARY
# --------------------------------------------------------------------------- #
# Each entry maps a CANONICAL field name to MI metadata:
#   tier                  : "core" | "extended"
#   business_name         : short label an analyst recognises
#   business_description  : one-line plain-English meaning
#   synonyms              : NL phrases that resolve to this field
#   overrides (optional)  : pins inference results (role/format/aggregations/
#                           default_aggregation/allowed_chart_roles/
#                           default_chart_role/bucket_field/weight_field/
#                           chartable)
#
# Fields not present in the canonical registry are skipped with a warning, so
# this list stays robust across registry versions.

_DAYS_AGG = {
    "allowed_aggregations": ["avg", "median", "distribution"],
    "default_aggregation": "avg",
}
_RATIO_AGG = {
    "allowed_aggregations": ["avg", "median", "distribution"],
    "default_aggregation": "avg",
}

CURATION: Dict[str, dict] = {
    # ---------------- CORE — portfolio balances / amounts ----------------
    "current_outstanding_balance": {
        "tier": "core", "business_name": "Balance",
        "business_description": "Current outstanding loan balance (exposure).",
        "synonyms": ["balance", "outstanding balance", "current balance",
                     "loan balance", "exposure", "current outstanding"],
        "overrides": {"bucket_field": "ticket_bucket"},
    },
    "current_principal_balance": {
        "tier": "core", "business_name": "Principal Balance",
        "business_description": "Current outstanding principal balance.",
        "synonyms": ["principal balance", "current principal",
                     "outstanding principal"],
        "overrides": {"bucket_field": "ticket_bucket"},
    },
    "original_principal_balance": {
        "tier": "core", "business_name": "Original Balance",
        "business_description": "Principal balance at origination.",
        "synonyms": ["original balance", "original principal",
                     "balance at origination", "advance amount"],
        "overrides": {"bucket_field": "ticket_bucket"},
    },
    "arrears_balance": {
        "tier": "core", "business_name": "Arrears Balance",
        "business_description": "Total amount in arrears.",
        "synonyms": ["arrears", "arrears balance", "amount in arrears",
                     "total arrears"],
    },
    "interest_arrears_amount": {
        "tier": "core", "business_name": "Interest Arrears",
        "business_description": "Interest amount in arrears.",
        "synonyms": ["interest arrears", "interest in arrears amount"],
    },
    "principal_arrears_amount": {
        "tier": "core", "business_name": "Principal Arrears",
        "business_description": "Principal amount in arrears.",
        "synonyms": ["principal arrears", "principal in arrears amount"],
    },
    "allocated_losses": {
        "tier": "core", "business_name": "Losses",
        "business_description": "Losses allocated to the loan.",
        "synonyms": ["losses", "allocated losses", "loss amount",
                     "credit losses"],
    },
    "recoveries_in_period": {
        "tier": "core", "business_name": "Recoveries",
        "business_description": "Recoveries received in the period.",
        "synonyms": ["recoveries", "recovery amount", "recoveries in period"],
    },
    "redemptions_received_in_period": {
        "tier": "core", "business_name": "Redemptions",
        "business_description": "Redemption proceeds received in the period.",
        "synonyms": ["redemptions", "redemption amount", "repayments received",
                     "redemptions received"],
    },
    "default_amount": {
        "tier": "core", "business_name": "Default Amount",
        "business_description": "Balance at the point of default.",
        "synonyms": ["default amount", "defaulted balance", "default balance"],
    },
    "current_valuation_amount": {
        "tier": "core", "business_name": "Valuation",
        "business_description": "Most recent property/collateral valuation.",
        "synonyms": ["valuation", "current valuation", "property value",
                     "collateral value", "valuation amount"],
    },
    "indexed_value": {
        "tier": "core", "business_name": "Indexed Valuation",
        "business_description": "Valuation indexed to the current period.",
        "synonyms": ["indexed value", "indexed valuation",
                     "indexed property value"],
        "overrides": {"format": "currency"},
    },
    # ---------------- CORE — rates / ratios ----------------
    "current_interest_rate": {
        "tier": "core", "business_name": "Interest Rate",
        "business_description": "Current interest rate on the loan.",
        "synonyms": ["interest rate", "rate", "coupon", "current rate"],
    },
    "current_loan_to_value": {
        "tier": "core", "business_name": "Current LTV",
        "business_description": "Current loan-to-value ratio.",
        "synonyms": ["ltv", "current ltv", "loan to value", "cltv"],
    },
    "indexed_loan_to_value": {
        "tier": "core", "business_name": "Indexed LTV",
        "business_description": "Loan-to-value using the indexed valuation.",
        "synonyms": ["indexed ltv", "iltv", "indexed loan to value"],
    },
    "original_loan_to_value": {
        "tier": "core", "business_name": "Original LTV",
        "business_description": "Loan-to-value at origination.",
        "synonyms": ["original ltv", "oltv", "ltv at origination"],
        "overrides": {"bucket_field": "original_ltv_bucket"},
    },
    # ---------------- CORE — counts / ages ----------------
    "number_of_days_in_arrears": {
        "tier": "core", "business_name": "Days in Arrears",
        "business_description": "Number of days the loan has been in arrears.",
        "synonyms": ["days in arrears", "arrears days", "dpd",
                     "days past due", "delinquency days"],
        "overrides": {"role": "metric", "format": "integer",
                      "bucket_field": "arrears_bucket", **_DAYS_AGG},
    },
    "youngest_borrower_age": {
        "tier": "core", "business_name": "Borrower Age",
        "business_description": "Age of the youngest borrower on the loan.",
        "synonyms": ["age", "borrower age", "youngest borrower age",
                     "applicant age"],
        "overrides": {"bucket_field": "age_bucket"},
    },
    # ---------------- CORE — portfolio dimensions ----------------
    "origination_date": {
        "tier": "core", "business_name": "Origination Date",
        "business_description": "Date the loan was originated.",
        "synonyms": ["origination date", "vintage", "origination",
                     "completion date", "drawdown date"],
    },
    "maturity_date": {
        "tier": "core", "business_name": "Maturity Date",
        "business_description": "Scheduled maturity date of the loan.",
        "synonyms": ["maturity date", "maturity", "end date"],
        "overrides": {"chartable": True, "bucket_field": "maturity_year"},
    },
    "originator_name": {
        "tier": "core", "business_name": "Originator",
        "business_description": "Name of the loan originator.",
        "synonyms": ["originator", "originator name", "lender"],
    },
    "broker_channel": {
        "tier": "core", "business_name": "Broker",
        "business_description": "Broker / origination channel.",
        "synonyms": ["broker", "broker channel", "channel",
                     "intermediary", "distribution channel"],
    },
    "geographic_region_obligor": {
        "tier": "core", "business_name": "Region",
        "business_description": "Geographic region of the obligor.",
        "synonyms": ["region", "geography", "obligor region",
                     "borrower region", "area"],
    },
    "interest_rate_type": {
        "tier": "core", "business_name": "Rate Type",
        "business_description": "Interest rate type (fixed / floating, etc.).",
        "synonyms": ["rate type", "interest rate type", "fixed or floating",
                     "product rate type"],
    },
    "erm_product_type": {
        "tier": "core", "business_name": "Product Type",
        "business_description": "Product type.",
        "synonyms": ["product", "product type", "loan type"],
    },
    "erm_sub_product_type": {
        "tier": "core", "business_name": "Sub Product Type",
        "business_description": "Sub-product type / product variant.",
        "synonyms": ["sub product", "sub product type", "product variant"],
    },
    "account_status": {
        "tier": "core", "business_name": "Account Status",
        "business_description": "Current account/loan status.",
        "synonyms": ["status", "account status", "loan status"],
    },
    "valuation_type": {
        "tier": "core", "business_name": "Valuation Type",
        "business_description": "Type/basis of the property valuation.",
        "synonyms": ["valuation type", "valuation basis"],
    },
    "tenure": {
        "tier": "core", "business_name": "Tenure",
        "business_description": "Property tenure (freehold / leasehold, etc.).",
        "synonyms": ["tenure", "freehold leasehold"],
    },
    "occupancy_type": {
        "tier": "core", "business_name": "Occupancy Type",
        "business_description": "Property occupancy type.",
        "synonyms": ["occupancy", "occupancy type", "owner occupied"],
    },
    # ---------------- CORE — borrower dimensions ----------------
    "borrower_jurisdiction": {
        "tier": "core", "business_name": "Borrower Jurisdiction",
        "business_description": "Legal jurisdiction of the borrower.",
        "synonyms": ["jurisdiction", "borrower jurisdiction", "country"],
    },
    "borrower_1_gender": {
        "tier": "core", "business_name": "Borrower Gender",
        "business_description": "Gender of the primary borrower.",
        "synonyms": ["gender", "borrower gender", "sex"],
    },
    # ---------------- CORE — flags ----------------
    "interest_in_arrears": {
        "tier": "core", "business_name": "In Arrears",
        "business_description": "Whether the loan is in arrears.",
        "synonyms": ["in arrears", "arrears flag", "delinquent",
                     "is in arrears"],
    },
    "loan_redemption_flag": {
        "tier": "core", "business_name": "Redeemed",
        "business_description": "Whether the loan has redeemed.",
        "synonyms": ["redeemed", "redemption flag", "closed", "repaid"],
    },
    "further_advance_flag": {
        "tier": "core", "business_name": "Further Advance",
        "business_description": "Whether a further advance has been taken.",
        "synonyms": ["further advance", "further advance flag", "additional borrowing"],
    },
    "protected_equity_flag": {
        "tier": "core", "business_name": "Protected Equity",
        "business_description": "Whether the loan carries protected equity.",
        "synonyms": ["protected equity", "protected equity flag",
                     "equity protection"],
    },
    "negative_equity_guarantee": {
        "tier": "core", "business_name": "Negative Equity Guarantee",
        "business_description": "Whether a no-negative-equity guarantee applies.",
        "synonyms": ["negative equity guarantee", "nneg", "no negative equity"],
    },

    # ================= EXTENDED =================
    "original_valuation_amount": {
        "tier": "extended", "business_name": "Original Valuation",
        "business_description": "Property valuation at origination.",
        "synonyms": ["original valuation", "valuation at origination"],
    },
    "current_interest_rate_margin": {
        "tier": "extended", "business_name": "Interest Margin",
        "business_description": "Current interest rate margin / spread.",
        "synonyms": ["margin", "interest margin", "spread", "rate margin"],
    },
    "current_debt_service_coverage_ratio": {
        "tier": "extended", "business_name": "DSCR",
        "business_description": "Current debt service coverage ratio.",
        "synonyms": ["dscr", "debt service coverage", "coverage ratio"],
        "overrides": {"format": "decimal", **_RATIO_AGG},
    },
    "debt_to_income_ratio": {
        "tier": "extended", "business_name": "DTI",
        "business_description": "Debt-to-income ratio.",
        "synonyms": ["dti", "debt to income", "income multiple"],
        "overrides": {"format": "decimal", **_RATIO_AGG},
    },
    "number_of_days_in_interest_arrears": {
        "tier": "extended", "business_name": "Days in Interest Arrears",
        "business_description": "Number of days in interest arrears.",
        "synonyms": ["days in interest arrears", "interest arrears days"],
        "overrides": {"role": "metric", "format": "integer", **_DAYS_AGG},
    },
    "number_of_days_in_principal_arrears": {
        "tier": "extended", "business_name": "Days in Principal Arrears",
        "business_description": "Number of days in principal arrears.",
        "synonyms": ["days in principal arrears", "principal arrears days"],
        "overrides": {"role": "metric", "format": "integer", **_DAYS_AGG},
    },
    "cumulative_prepayments": {
        "tier": "extended", "business_name": "Cumulative Prepayments",
        "business_description": "Cumulative unscheduled prepayments to date.",
        "synonyms": ["prepayments", "cumulative prepayments", "overpayments"],
        "overrides": {"format": "currency"},
    },
    "contractual_annual_rental_income": {
        "tier": "extended", "business_name": "Rental Income",
        "business_description": "Contractual annual rental income.",
        "synonyms": ["rental income", "rent", "annual rent"],
    },
    "net_operating_income_at_securitisation": {
        "tier": "extended", "business_name": "Net Operating Income",
        "business_description": "Net operating income at securitisation.",
        "synonyms": ["noi", "net operating income", "operating income"],
    },
    "equity": {
        "tier": "extended", "business_name": "Equity",
        "business_description": "Borrower equity in the property.",
        "synonyms": ["equity", "borrower equity"],
        "overrides": {"format": "currency"},
    },
    "original_term": {
        "tier": "extended", "business_name": "Original Term",
        "business_description": "Original loan term.",
        "synonyms": ["term", "original term", "loan term"],
        "overrides": {"bucket_field": "term_bucket"},
    },
    "number_of_bedrooms": {
        "tier": "extended", "business_name": "Bedrooms",
        "business_description": "Number of bedrooms in the property.",
        "synonyms": ["bedrooms", "number of bedrooms"],
        "overrides": {"role": "dimension",
                      "allowed_aggregations": ["count", "balance_sum"],
                      "default_aggregation": "count",
                      "allowed_chart_roles": ["x", "group", "filter", "color"],
                      "default_chart_role": "group", "chartable": True},
    },
    "lien": {
        "tier": "extended", "business_name": "Lien Position",
        "business_description": "Lien / charge ranking on the collateral.",
        "synonyms": ["lien", "lien position", "charge", "ranking"],
        "overrides": {"role": "dimension",
                      "allowed_aggregations": ["count", "balance_sum"],
                      "default_aggregation": "count",
                      "allowed_chart_roles": ["x", "group", "filter", "color"],
                      "default_chart_role": "group", "chartable": True},
    },
    "postcode": {
        "tier": "extended", "business_name": "Postcode",
        "business_description": "Property postcode.",
        "synonyms": ["postcode", "post code", "zip", "zip code"],
        "overrides": {"role": "dimension"},
    },
    "employment_status": {
        "tier": "extended", "business_name": "Employment Status",
        "business_description": "Borrower employment status.",
        "synonyms": ["employment status", "employment", "employed"],
    },
    "origination_channel": {
        "tier": "extended", "business_name": "Origination Channel",
        "business_description": "Channel through which the loan was originated.",
        "synonyms": ["origination channel", "sales channel"],
    },
    "geographic_region_collateral": {
        "tier": "extended", "business_name": "Collateral Region",
        "business_description": "Geographic region of the collateral.",
        "synonyms": ["collateral region", "property region", "asset region"],
    },
    "default_date": {
        "tier": "extended", "business_name": "Default Date",
        "business_description": "Date the loan defaulted.",
        "synonyms": ["default date", "date of default"],
        "overrides": {"chartable": True},
    },
    "current_valuation_date": {
        "tier": "extended", "business_name": "Valuation Date",
        "business_description": "Date of the most recent valuation.",
        "synonyms": ["valuation date", "date of valuation"],
    },
    "date_last_in_arrears": {
        "tier": "extended", "business_name": "Date Last in Arrears",
        "business_description": "Date the loan was last in arrears.",
        "synonyms": ["date last in arrears", "last arrears date"],
    },
    "date_of_restructuring": {
        "tier": "extended", "business_name": "Restructuring Date",
        "business_description": "Date the loan was restructured.",
        "synonyms": ["restructuring date", "restructure date", "modification date"],
    },
    "early_repayment_charge": {
        "tier": "extended", "business_name": "Early Repayment Charge",
        "business_description": "Whether an early repayment charge applies.",
        "synonyms": ["erc", "early repayment charge", "prepayment penalty"],
    },
    "originator_affiliate": {
        "tier": "extended", "business_name": "Originator Affiliate",
        "business_description": "Whether the originator is an affiliate.",
        "synonyms": ["originator affiliate", "affiliate"],
    },
    "number_of_properties_at_data_cut_off_date": {
        "tier": "extended", "business_name": "Number of Properties",
        "business_description": "Number of properties securing the loan.",
        "synonyms": ["number of properties", "properties", "property count"],
    },
}


# --------------------------------------------------------------------------- #
# Defensive accessors (the canonical registry casing/nesting may vary)
# --------------------------------------------------------------------------- #


def _get(d: dict, *keys, default=None):
    """Case-insensitive first-match lookup across candidate keys."""
    if not isinstance(d, dict):
        return default
    lowered = {str(k).lower(): v for k, v in d.items()}
    for k in keys:
        if k in d:
            return d[k]
        if str(k).lower() in lowered:
            return lowered[str(k).lower()]
    return default


def load_canonical_registry(path: Path) -> Dict[str, dict]:
    """Return the {field_name: meta} mapping from the canonical registry."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    fields = _get(data, "fields", default=None)
    if not isinstance(fields, dict):
        fields = {
            k: v
            for k, v in data.items()
            if isinstance(v, dict) and ("layer" in v or "category" in v)
        }
    if not fields:
        raise ValueError(f"No 'fields' mapping found in registry: {path}")
    return fields


# --------------------------------------------------------------------------- #
# Traceability: which canonical attributes a curated field satisfies
# --------------------------------------------------------------------------- #

_PERF_PRODUCT_LAYERS = {"performance", "product"}


def selection_criteria(meta: dict) -> List[str]:
    """Canonical attributes that justify MI relevance (for traceability only)."""
    crit: List[str] = []
    if _get(meta, "core_canonical") is True:
        crit.append("core_canonical")
    layer = str(_get(meta, "layer", default="") or "").lower()
    if layer in _PERF_PRODUCT_LAYERS:
        crit.append(f"layer:{layer}")
    if not crit:
        # curated even though it is not in the broad rule set
        crit.append("curated")
    return crit


# --------------------------------------------------------------------------- #
# Inference helpers
# --------------------------------------------------------------------------- #


def _tokens(name: str) -> List[str]:
    return [t for t in str(name).lower().split("_") if t]


def display_name(name: str) -> str:
    parts = []
    for tok in _tokens(name):
        parts.append(_ACRONYMS.get(tok, tok.capitalize()))
    return " ".join(parts) if parts else name


def _contains(name: str, needles) -> bool:
    low = name.lower()
    return any(n in low for n in needles)


def infer_format(name: str, reg_format: Optional[str], allowed_values) -> str:
    """Map to one of: currency|percent|integer|decimal|date|string|boolean."""
    nf = str(reg_format).strip().lower() if reg_format else ""
    av = str(allowed_values).strip().lower() if allowed_values else ""
    low = name.lower()
    toks = _tokens(name)

    # 1. Count-like fields take precedence over name-based date detection
    #    (e.g. ``number_of_properties_at_data_cut_off_date`` is a COUNT, not a date).
    if low.startswith("number_of_") or "count" in toks:
        return "integer"
    # 2. Date
    if nf == "date" or "date" in low:
        return "date"
    if nf in ("y/n", "boolean", "bool") or av in ("yes_no", "y/n"):
        return "boolean"
    if nf in ("list", "string", "currency_code", "enum", "category"):
        return "string"

    if _contains(name, ("ltv", "loan_to_value", "percentage", "pct", "margin", "spread")) or \
            "rate" in toks:
        return "percent"
    # Currency keywords use stems so plurals/variants are caught
    # (``recover`` -> recovery/recoveries, ``redemption`` -> redemptions,
    #  ``prepay`` -> prepayments, ``advance`` -> advances).
    if _contains(name, ("balance", "amount", "valuation", "value", "income",
                         "loss", "recover", "arrears", "price", "proceeds",
                         "redemption", "prepay", "advance")):
        return "currency"
    if _contains(name, ("term_months", "months")) or "age" in toks:
        return "integer"
    if nf == "integer":
        return "integer"
    if nf == "decimal":
        return "decimal"
    return "string"


_ID_TOKENS = {"id", "identifier", "lei", "code", "reference", "ref"}
_ID_PHRASES = ("account_number", "loan_number", "_identifier")


def _is_identifier(name: str) -> bool:
    toks = set(_tokens(name))
    if toks & _ID_TOKENS:
        return True
    return _contains(name, _ID_PHRASES)


def infer_role(name: str, fmt: str, reg_format: Optional[str]) -> str:
    """metric | dimension | date | identifier | flag | unknown."""
    if _is_identifier(name):
        return "identifier"
    if fmt == "date":
        return "date"
    if fmt == "boolean":
        return "flag"
    if fmt in ("currency", "percent", "integer", "decimal"):
        return "metric"
    nf = str(reg_format).strip().lower() if reg_format else ""
    if nf in ("list", "string", "currency_code", "enum", "category"):
        return "dimension"
    return "unknown"


def _is_count_metric(name: str) -> bool:
    return _contains(name, ("count", "number_of")) and "age" not in _tokens(name)


def infer_aggregations(role: str, fmt: str, name: str,
                       has_weight: bool = False) -> Tuple[List[str], str]:
    if role == "identifier":
        return ["count_distinct"], "count_distinct"
    if role == "date":
        return ["count"], "count"
    if role == "flag":
        return ["count", "balance_sum"], "count"
    if role == "dimension":
        return ["count", "balance_sum"], "count"
    if role == "metric":
        if fmt == "currency":
            return ["sum", "avg", "median"], "sum"
        if fmt == "percent":
            # Rate/LTV/percentage fields prefer weighted_avg over a simple
            # arithmetic average for portfolio MI (when a balance weight exists).
            default = "weighted_avg" if has_weight else "avg"
            return ["avg", "weighted_avg", "distribution"], default
        if fmt == "integer":
            if _is_count_metric(name):
                # ``number_of_*`` style counts: useful as sum/avg/median/distribution.
                return ["sum", "avg", "median", "distribution"], "sum"
            # Ages / days / terms: not sensible to sum across loans.
            return ["avg", "median", "distribution"], "avg"
        return ["sum", "avg", "median"], "avg"
    return [], ""


# Numeric metrics that read naturally on the x axis of a chart (age, days,
# terms, ratios). Currency / percent default to y but can also be x'd.
_AXIS_NUMERIC_FORMATS = {"integer", "decimal"}


def infer_chart_roles(role: str, fmt: Optional[str] = None
                      ) -> Tuple[List[str], Optional[str]]:
    if role == "identifier":
        return ["filter"], None
    if role == "date":
        return ["x", "filter", "cohort"], "x"
    if role == "flag":
        return ["filter", "color", "group"], "filter"
    if role == "dimension":
        return ["x", "group", "filter", "color"], "x"
    if role == "metric":
        if fmt in _AXIS_NUMERIC_FORMATS:
            # Axis-like numeric metric: ages, days_in_arrears, terms, DSCR, DTI.
            return ["x", "y", "bucket", "filter", "color"], "x"
        if fmt == "percent":
            # Rates / LTV / percentages: usually y, but legitimately bucketable
            # on x (e.g. histogram of LTV).
            return ["y", "x", "bucket", "filter", "color"], "y"
        if fmt == "currency":
            # Balances / amounts: y for bar/line, size for bubble, bucketable
            # for ticket-size analysis, filterable for thresholds.
            return ["y", "size", "x", "bucket", "filter", "color"], "y"
        return ["y", "size", "color"], "y"
    return [], None


def infer_chartable(role: str, name: str) -> bool:
    if role in ("metric", "dimension", "flag"):
        return True
    if role == "date":
        return "origination" in name.lower()
    return False


def pick_weight_field(selected_names: set) -> Optional[str]:
    for candidate in ("total_balance", "current_outstanding_balance",
                       "current_principal_balance", "original_principal_balance"):
        if candidate in selected_names:
            return candidate
    return None


def infer_weight_field(role: str, fmt: str, weight_target: Optional[str]) -> Optional[str]:
    if role == "metric" and fmt == "percent":
        return weight_target
    return None


def infer_bucket_field(name: str, role: str, fmt: str) -> Optional[str]:
    low = name.lower()
    if fmt == "date" and "origination" in low:
        return "vintage_year"
    if "ltv" in low or "loan_to_value" in low:
        return "original_ltv_bucket" if "original" in low else "ltv_bucket"
    if "age" in _tokens(name):
        return "age_bucket"
    if "balance" in low and role == "metric":
        return "ticket_bucket"
    return None


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #


def build_entry(name: str, meta: dict, curated: dict,
                weight_target: Optional[str]) -> dict:
    reg_format = _get(meta, "format")
    allowed_values = _get(meta, "allowed_values")
    overrides = curated.get("overrides", {}) or {}

    # 1. Inferred role / format, with overrides applied immediately so all
    #    downstream derivations see the final values.
    fmt = overrides.get("format") or infer_format(name, reg_format, allowed_values)
    role = overrides.get("role") or infer_role(name, fmt, reg_format)

    # 2. Weight field: override > heuristic.  Computed before aggregations so
    #    weighted_avg defaults only apply when a weight is actually present.
    if "weight_field" in overrides:
        weight_field = overrides["weight_field"]
    else:
        weight_field = infer_weight_field(role, fmt, weight_target)

    # 3. Derived defaults (each can be pinned by an override).
    inf_aggs, inf_default_agg = infer_aggregations(
        role, fmt, name, has_weight=bool(weight_field)
    )
    inf_chart_roles, inf_default_chart_role = infer_chart_roles(role, fmt=fmt)
    inf_chartable = infer_chartable(role, name)
    inf_bucket_field = infer_bucket_field(name, role, fmt)

    entry = {
        "canonical_field": name,
        "mi_tier": curated.get("tier", "extended"),
        "business_name": curated.get("business_name") or display_name(name),
        "display_name": display_name(name),
        "business_description": curated.get("business_description", ""),
        "description": "",
        "synonyms": list(curated.get("synonyms", []) or []),
        "source_criteria": list(selection_criteria(meta)),
        "role": role,
        "format": fmt,
        "chartable": overrides.get("chartable", inf_chartable),
        "allowed_aggregations": list(overrides.get("allowed_aggregations", inf_aggs)),
        "default_aggregation": overrides.get("default_aggregation", inf_default_agg),
        "allowed_chart_roles": list(overrides.get("allowed_chart_roles", inf_chart_roles)),
        "default_chart_role": overrides.get("default_chart_role", inf_default_chart_role),
        "weight_field": weight_field,
        "bucket_field": overrides.get("bucket_field", inf_bucket_field),
        "notes": "",
    }

    if entry["role"] == "unknown" and not entry["notes"]:
        entry["notes"] = "requires manual analytics classification"

    return entry


def build_registry(source: Path) -> dict:
    fields = load_canonical_registry(source)

    present = [name for name in CURATION if name in fields]
    missing = [name for name in CURATION if name not in fields]
    for name in missing:
        print(f"WARNING: curated field not found in canonical registry, "
              f"skipping: {name}", file=sys.stderr)

    selected_names = set(present)
    weight_target = pick_weight_field(selected_names)

    out_fields: Dict[str, dict] = {}
    for name in sorted(present):
        out_fields[name] = build_entry(
            name, fields[name], CURATION[name], weight_target
        )

    tier_counts: Dict[str, int] = {}
    for entry in out_fields.values():
        tier_counts[entry["mi_tier"]] = tier_counts.get(entry["mi_tier"], 0) + 1

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_registry": str(source),
        "selection_approach": "curated MI-first allowlist (see CURATION in build script)",
        "selection_rules": [
            "explicit curated allowlist of MI-relevant canonical fields",
            "excludes identifiers, LEIs, industry/tax codes, rating-agency "
            "fields, waterfall/swap fields, balance-period buckets and "
            "duplicate borrower-2/guarantor fields",
        ],
        "mi_tiers": ["core", "extended"],
        "field_count": len(out_fields),
        "core_field_count": tier_counts.get("core", 0),
        "extended_field_count": tier_counts.get("extended", 0),
        "missing_curated_fields": missing,
        "version": VERSION,
        "default_weight_field": weight_target,
        "cleanup_notes": list(CLEANUP_NOTES),
    }

    return {"metadata": metadata, "fields": out_fields}


def write_registry(registry: dict, output: Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# AUTO-GENERATED by mi_agent/build_mi_semantics_registry.py\n"
        "# Curated MI semantic layer over the canonical field registry.\n"
        "# Heuristic metadata + human curation — REVIEW BEFORE PRODUCTION USE.\n"
        "# Do NOT edit by hand without also updating the build script (CURATION),\n"
        "# or your changes will be overwritten on the next regeneration.\n"
    )
    with output.open("w", encoding="utf-8") as fh:
        fh.write(header)
        yaml.dump(registry, fh, Dumper=_NoAliasDumper,
                  sort_keys=False, allow_unicode=True, width=100,
                  default_flow_style=False)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build the curated MI semantic field registry.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="Path to canonical fields_registry.yaml")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Path to write the MI semantic registry YAML")
    args = parser.parse_args(argv)

    if not args.source.exists():
        print(f"ERROR: source registry not found: {args.source}", file=sys.stderr)
        return 2

    registry = build_registry(args.source)
    write_registry(registry, args.output)

    fields = registry["fields"]
    meta = registry["metadata"]
    role_counts: Dict[str, int] = {}
    for entry in fields.values():
        role_counts[entry["role"]] = role_counts.get(entry["role"], 0) + 1

    print(f"Wrote {len(fields)} fields -> {args.output}")
    print(f"  core={meta['core_field_count']}  extended={meta['extended_field_count']}")
    print("Roles inferred:")
    for role, count in sorted(role_counts.items()):
        print(f"  {role:11s} {count}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
