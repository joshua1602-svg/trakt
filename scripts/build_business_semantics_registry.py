#!/usr/bin/env python3
"""
build_business_semantics_registry.py

Generate the governed Business Semantics Registry from the canonical field
registry.

This is an ADDITIVE tool.  It only READS the canonical registry
(config/system/fields_registry.yaml) and WRITES a new, separate file:

    config/business_semantics_registry.yaml

DESIGN
------
The Business Semantics Registry identifies the subset of canonical registry
entries that represent meaningful analytical concepts or metrics (exposure,
payment performance, risk, leverage, valuation, cashflow, pricing, maturity,
concentration, ...) and enriches them with governed metadata for future
reasoning workflows (period-change analysis, portfolio risk comparison,
ranking, monitoring).

Selection is an EXPLICIT CURATED ALLOWLIST (``CURATION`` below), reviewed
field-by-field against the canonical registry, the curated MI semantics layer
(mi_agent/build_mi_semantics_registry.py) and the risk-monitor configuration
(config/mi/risk_monitor.yaml).  Identifiers, names, addresses, free text,
provenance/audit metadata, ingestion metadata, non-analytical dates, document
identifiers, static descriptive facts and technical control fields are
deliberately excluded.  Borderline fields are NOT forced into the registry:
they are tracked in ``UNCERTAIN`` (with a proposed classification and a
recommended human decision) and surfaced in
docs/business_semantics_registry_review.md.

All metadata uses the controlled taxonomy in ``TAXONOMY``.  No calculation
logic, materiality thresholds, covenant limits or risk weights are defined
here — only semantic classification supported by existing definitions.

GOVERNANCE
----------
The generated file is a reviewed artefact.  To protect human review decisions,
this script REFUSES to overwrite an existing output file unless ``--force`` is
passed.  ``--check`` regenerates in memory and verifies the committed file is
identical (used by the determinism test / CI).

Usage:
    python scripts/build_business_semantics_registry.py            # first write
    python scripts/build_business_semantics_registry.py --check    # verify
    python scripts/build_business_semantics_registry.py --force    # regenerate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "config" / "system" / "fields_registry.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "config" / "business_semantics_registry.yaml"

VERSION = "0.1.0"

# --------------------------------------------------------------------------- #
# Controlled taxonomy
# --------------------------------------------------------------------------- #

TAXONOMY: Dict[str, List[str]] = {
    # Primary analytical concept (exactly one per entry). Asset-agnostic.
    "analytical_concepts": [
        "cashflow",
        "collateral",
        "concentration",
        "coverage",
        "credit_quality",
        "data_quality",
        "eligibility",
        "exposure",
        "forecast",
        "leverage",
        "liquidity",
        "loss",
        "maturity",
        "operational_performance",
        "payment_performance",
        "pricing",
        "tail_risk",
        "valuation",
    ],
    # Secondary categories (one or more per entry).
    "categories": [
        "affordability",
        "cashflow",
        "collateral",
        "concentration",
        "coverage",
        "credit_quality",
        "data_quality",
        "delinquency",
        "eligibility",
        "exposure",
        "forecast",
        "geography",
        "income",
        "leverage",
        "liquidity",
        "loss",
        "maturity",
        "obligor_financials",
        "operational_performance",
        "payment_performance",
        "pricing",
        "product_mix",
        "recovery",
        "return",
        "risk",
        "seasoning",
        "tail_risk",
        "valuation",
    ],
    # Reasoning workflows this first version supports.
    "workflow_tags": [
        "monitoring",
        "period_change",
        "portfolio_comparison",
        "ranking",
    ],
    "aggregation_types": [
        "average",
        "distribution",
        "share",
        "sum",
        "weighted_average",
    ],
    "directionality": [
        "context_dependent",
        "higher_is_better",
        "higher_is_worse",
        "lower_is_better",
        "lower_is_worse",
        "neutral",
    ],
    "confidence": ["high", "medium", "low"],
    # Derived from the canonical registry's portfolio_type (see
    # ASSET_APPLICABILITY_MAP); 'cross_asset' mirrors portfolio_type 'common'.
    "asset_applicability": [
        "commercial_real_estate",
        "corporate",
        "cross_asset",
        "equipment_leasing",
        "equity_release",
        "residential_mortgage",
        "sme",
    ],
}

# portfolio_type (canonical registry, case-insensitive) -> asset_applicability
ASSET_APPLICABILITY_MAP = {
    "common": "cross_asset",
    "cre": "commercial_real_estate",
    "rre": "residential_mortgage",
    "sme": "sme",
    "equity_release": "equity_release",
    "equipment": "equipment_leasing",
    "corporate": "corporate",
}

# Acronyms upper-cased in display names (mirrors the MI semantics builder).
_ACRONYMS = {
    "ltv": "LTV", "lgd": "LGD", "pd": "PD", "ead": "EAD", "dti": "DTI",
    "dscr": "DSCR", "noi": "NOI", "erm": "ERM", "ebitda": "EBITDA",
    "ebit": "EBIT", "ifrs9": "IFRS 9", "epc": "EPC", "nace": "NACE",
    "sspe": "SSPE", "dbrs": "DBRS",
}

# Aggregations whose period-on-period movement is quantitatively assessable.
_MATERIALITY_AGGREGATIONS = {"sum", "average", "weighted_average", "share"}


def _display_name(name: str) -> str:
    parts = []
    for tok in str(name).lower().split("_"):
        if not tok:
            continue
        parts.append(_ACRONYMS.get(tok, tok.capitalize()))
    return " ".join(parts) if parts else name


def E(concept: str,
      categories: List[str],
      tags: List[str],
      directionality: str,
      aggregation: str,
      confidence: str,
      rationale: str,
      *,
      assets: Optional[List[str]] = None,
      display: Optional[str] = None,
      comparable: bool = True,
      materiality: Optional[bool] = None) -> Dict[str, Any]:
    """Compact curated-entry constructor (validated in build_registry)."""
    return {
        "concept": concept,
        "categories": list(categories),
        "tags": list(tags),
        "directionality": directionality,
        "aggregation": aggregation,
        "confidence": confidence,
        "rationale": rationale,
        "assets": list(assets) if assets else None,
        "display": display,
        "comparable": comparable,
        "materiality": materiality,
    }


# Workflow tag shorthands.
_ALL4 = ["period_change", "portfolio_comparison", "ranking", "monitoring"]
_PC_PCMP = ["period_change", "portfolio_comparison"]
_PC_PCMP_MON = ["period_change", "portfolio_comparison", "monitoring"]
_PCMP = ["portfolio_comparison"]
_PCMP_MON = ["portfolio_comparison", "monitoring"]

# --------------------------------------------------------------------------- #
# CURATED BUSINESS SEMANTICS VOCABULARY
# --------------------------------------------------------------------------- #
# Every key must exist in config/system/fields_registry.yaml.  Classification
# uses only meaning that is supported by the canonical registry metadata
# (category/format/layer/portfolio_type/regime mapping), the curated MI
# semantics layer or the risk-monitor configuration.

CURATION: Dict[str, Dict[str, Any]] = {

    # ================= EXPOSURE / BALANCES =================
    "current_outstanding_balance": E(
        "exposure", ["exposure", "risk"], _ALL4, "neutral", "sum", "high",
        "Primary current-exposure metric; the MI layer's default balance/"
        "weighting field for portfolio aggregation."),
    "current_principal_balance": E(
        "exposure", ["exposure"], _ALL4, "neutral", "sum", "high",
        "Current outstanding principal; core exposure measure used across "
        "regulatory annexes and MI."),
    "original_principal_balance": E(
        "exposure", ["exposure"], ["portfolio_comparison", "ranking"],
        "neutral", "sum", "high",
        "Principal at origination; static exposure baseline for comparison "
        "and vintage analysis (does not move period to period)."),
    "committed_undrawn_facility_underlying_exposure_balance": E(
        "exposure", ["exposure", "liquidity"], _PC_PCMP_MON,
        "context_dependent", "sum", "medium",
        "Committed undrawn facility balance; contingent exposure and funding "
        "commitment measure."),
    "undrawn_facility": E(
        "exposure", ["exposure", "liquidity"], _PC_PCMP_MON,
        "context_dependent", "sum", "high",
        "Undrawn facility amount; contingent exposure that can convert to "
        "drawn balance."),
    "drawdown_facility": E(
        "exposure", ["exposure"], _PC_PCMP, "neutral", "sum", "medium",
        "Facility drawdown amount; exposure origination measure."),
    "cumulative_drawn_amount": E(
        "exposure", ["exposure", "cashflow"], _PC_PCMP, "neutral", "sum",
        "medium",
        "Cumulative amount drawn on the facility; exposure build-up over "
        "time."),
    "total_credit_limit": E(
        "exposure", ["exposure"], _PCMP_MON, "neutral", "sum", "medium",
        "Total committed credit limit; upper bound of potential exposure."),
    "total_other_amounts_outstanding": E(
        "exposure", ["exposure"], _PC_PCMP, "context_dependent", "sum",
        "medium",
        "Other amounts outstanding (fees, charges) beyond principal and "
        "interest; part of total exposure."),
    "exposure_at_default": E(
        "exposure", ["exposure", "risk", "credit_quality"], _ALL4,
        "higher_is_worse", "sum", "high",
        "EAD from the risk model layer (Phase 0B); expected exposure amount "
        "at default, curated in the MI risk registry."),
    "ead_bucket": E(
        "exposure", ["exposure", "concentration"], _PC_PCMP_MON, "neutral",
        "distribution", "high",
        "Banded EAD stratification dimension defined at the analytics layer "
        "(config/mi/buckets.yaml)."),
    "prior_principal_balances": E(
        "exposure", ["exposure"], ["period_change"], "neutral", "sum",
        "medium",
        "Prior-period principal balance; the comparison basis for balance "
        "movement analysis."),
    "negative_amortisation": E(
        "exposure", ["exposure", "payment_performance"], _PC_PCMP_MON,
        "higher_is_worse", "sum", "medium",
        "Negative amortisation amount; balance growth from unpaid interest "
        "capitalisation."),
    "further_advance_amount": E(
        "exposure", ["exposure", "cashflow"], _PC_PCMP, "neutral", "sum",
        "medium",
        "Amount of further advances taken; incremental exposure on existing "
        "loans."),
    "further_advance_flag": E(
        "exposure", ["exposure"], _PC_PCMP, "neutral", "share", "medium",
        "Whether a further advance has been taken (MI-curated flag); share of "
        "book with incremental borrowing."),
    "pari_passu_underlying_exposures": E(
        "leverage", ["leverage", "exposure"], _PCMP, "higher_is_worse", "sum",
        "medium",
        "Equal-ranking debt amount alongside the exposure; dilutes effective "
        "collateral coverage."),
    "cumulative_accrued_interest": E(
        "exposure", ["exposure", "cashflow"], _PC_PCMP_MON,
        "context_dependent", "sum", "high",
        "Cumulative accrued (rolled-up) interest; drives balance compounding, "
        "central to equity-release roll-up exposure."),
    "accrued_interest_in_period": E(
        "cashflow", ["cashflow", "exposure"], ["period_change"], "neutral",
        "sum", "high",
        "Interest accrued in the reporting period; period income accrual and "
        "balance roll-up component."),
    "balloon_amount": E(
        "maturity", ["maturity", "cashflow", "exposure"], _PCMP_MON,
        "context_dependent", "sum", "medium",
        "Balloon repayment amount; repayment concentration at maturity "
        "(refinancing risk)."),

    # ================= PAYMENT PERFORMANCE / ARREARS =================
    "arrears_balance": E(
        "payment_performance", ["payment_performance", "delinquency", "risk"],
        _ALL4, "higher_is_worse", "sum", "high",
        "Total amount in arrears; core delinquency metric (MI core)."),
    "interest_arrears_amount": E(
        "payment_performance", ["payment_performance", "delinquency"], _ALL4,
        "higher_is_worse", "sum", "high",
        "Interest amount in arrears (MI core)."),
    "principal_arrears_amount": E(
        "payment_performance", ["payment_performance", "delinquency"], _ALL4,
        "higher_is_worse", "sum", "high",
        "Principal amount in arrears (MI core)."),
    "number_of_days_in_arrears": E(
        "payment_performance", ["payment_performance", "delinquency"], _ALL4,
        "higher_is_worse", "average", "high",
        "Days past due; standard delinquency severity measure with governed "
        "arrears buckets at the analytics layer."),
    "number_of_days_in_interest_arrears": E(
        "payment_performance", ["payment_performance", "delinquency"], _ALL4,
        "higher_is_worse", "average", "high",
        "Days in interest arrears (MI extended)."),
    "number_of_days_in_principal_arrears": E(
        "payment_performance", ["payment_performance", "delinquency"], _ALL4,
        "higher_is_worse", "average", "high",
        "Days in principal arrears (MI extended)."),
    "days_in_arrears_prior": E(
        "payment_performance", ["payment_performance", "delinquency"],
        ["period_change"], "higher_is_worse", "average", "medium",
        "Prior-period days past due; comparison basis for arrears "
        "deterioration flags (risk_monitor arrears_days_increase)."),
    "interest_in_arrears": E(
        "payment_performance", ["payment_performance", "delinquency"],
        _PC_PCMP_MON, "higher_is_worse", "share", "high",
        "Whether the loan is in arrears (MI core flag); arrears incidence "
        "share of the book."),
    "date_last_in_arrears": E(
        "payment_performance", ["payment_performance", "delinquency"],
        ["monitoring"], "neutral", "distribution", "medium",
        "Date last in arrears (MI extended); recency of arrears experience."),
    "deferred_interest": E(
        "payment_performance", ["payment_performance", "cashflow"],
        _PC_PCMP_MON, "higher_is_worse", "sum", "medium",
        "Deferred interest amount; unpaid interest carried forward."),
    "payment_due": E(
        "payment_performance", ["payment_performance", "cashflow"],
        ["period_change"], "neutral", "sum", "medium",
        "Scheduled payment due in the period; denominator for collection "
        "performance."),
    "total_scheduled_principal_interest_due": E(
        "cashflow", ["cashflow", "payment_performance"], ["period_change"],
        "neutral", "sum", "medium",
        "Total scheduled principal and interest due; scheduled cashflow "
        "expectation for the period."),
    "total_scheduled_principal_interest_paid": E(
        "cashflow", ["cashflow", "payment_performance"], _PC_PCMP,
        "higher_is_better", "sum", "medium",
        "Total scheduled principal and interest actually paid; collection "
        "performance versus schedule."),
    "total_shortfalls_in_principal_interest_outstanding": E(
        "payment_performance",
        ["payment_performance", "delinquency", "cashflow"], _ALL4,
        "higher_is_worse", "sum", "high",
        "Outstanding principal and interest shortfalls; unresolved payment "
        "underperformance."),
    "payment_in_kind": E(
        "payment_performance", ["payment_performance", "credit_quality"],
        _PCMP_MON, "higher_is_worse", "share", "medium",
        "Payment-in-kind flag; interest satisfied other than in cash, a "
        "credit-stress indicator."),
    "penalty_interest_balance": E(
        "payment_performance", ["payment_performance", "cashflow"],
        ["period_change", "monitoring"], "higher_is_worse", "sum", "medium",
        "Accrued penalty interest balance; consequence of payment "
        "underperformance."),
    "penalty_interest_rate": E(
        "pricing", ["pricing"], _PCMP, "neutral", "weighted_average",
        "medium",
        "Penalty interest rate applied to amounts in default."),
    "current_default_interest_rate": E(
        "pricing", ["pricing", "payment_performance"], _PCMP, "neutral",
        "weighted_average", "medium",
        "Current default interest rate charged on defaulted amounts."),
    "actual_default_interest": E(
        "cashflow", ["cashflow", "payment_performance"], ["period_change"],
        "neutral", "sum", "medium",
        "Default interest actually charged/collected in the period "
        "(ESMA CREL135)."),
    "account_status": E(
        "credit_quality", ["credit_quality", "payment_performance"],
        _PC_PCMP_MON, "context_dependent", "distribution", "high",
        "Current account/loan status (performing, arrears, default, "
        "redeemed); primary book-composition status dimension (MI core)."),
    "modification": E(
        "credit_quality", ["credit_quality", "payment_performance"],
        ["period_change", "monitoring"], "context_dependent", "distribution",
        "medium",
        "Loan modification type; forbearance and restructuring incidence."),
    "date_of_restructuring": E(
        "credit_quality", ["credit_quality", "payment_performance"],
        ["monitoring"], "neutral", "distribution", "medium",
        "Date the loan was restructured (MI extended); forbearance recency "
        "profile."),

    # ================= CREDIT QUALITY / RISK MODEL =================
    "internal_risk_grade": E(
        "credit_quality", ["credit_quality", "risk"], _ALL4,
        "higher_is_worse", "distribution", "high",
        "Internal obligor risk grade (Phase 0B risk model); ordered A "
        "(best) to G (worst) per risk_monitor deterioration orderings."),
    "internal_risk_score": E(
        "credit_quality", ["credit_quality", "risk"], _PC_PCMP_MON,
        "context_dependent", "average", "medium",
        "Numeric internal/behavioural risk score; score direction is not "
        "defined in the registry so directionality is context-dependent."),
    "internal_risk_stage": E(
        "credit_quality", ["credit_quality", "risk"], _PC_PCMP_MON,
        "context_dependent", "distribution", "high",
        "Internal monitoring/watchlist stage (lender taxonomy, unordered per "
        "risk_monitor; distinct from IFRS 9)."),
    "ifrs9_stage": E(
        "credit_quality", ["credit_quality", "loss"], _PC_PCMP_MON,
        "higher_is_worse", "distribution", "high",
        "IFRS 9 impairment stage; ordered Stage 1 to Stage 3 per "
        "risk_monitor deterioration orderings."),
    "probability_of_default": E(
        "credit_quality", ["credit_quality", "risk"], _ALL4,
        "higher_is_worse", "weighted_average", "high",
        "Probability of default (0-1) from the Phase 0B risk model; core "
        "credit-quality metric."),
    "pd_bucket": E(
        "credit_quality", ["credit_quality", "concentration"], _PC_PCMP_MON,
        "higher_is_worse", "distribution", "high",
        "Banded PD stratification dimension; ordered banding defined in "
        "risk_monitor deterioration orderings."),
    "loss_given_default": E(
        "loss", ["loss", "credit_quality", "collateral"], _ALL4,
        "higher_is_worse", "weighted_average", "high",
        "Loss given default (0-1) from the Phase 0B risk model; loss "
        "severity assumption."),
    "lgd_bucket": E(
        "loss", ["loss", "concentration"], _PC_PCMP_MON, "higher_is_worse",
        "distribution", "high",
        "Banded LGD stratification dimension; ordered banding defined in "
        "risk_monitor deterioration orderings."),
    "risk_grade_previous": E(
        "credit_quality", ["credit_quality", "risk"], ["period_change"],
        "higher_is_worse", "distribution", "medium",
        "Previous-snapshot risk grade; input to the risk-grade migration "
        "matrix (risk_monitor migration pairs)."),
    "risk_grade_current": E(
        "credit_quality", ["credit_quality", "risk"], ["period_change"],
        "higher_is_worse", "distribution", "medium",
        "Current-snapshot risk grade; input to the risk-grade migration "
        "matrix (risk_monitor migration pairs)."),
    "pd_previous": E(
        "credit_quality", ["credit_quality", "risk"], ["period_change"],
        "higher_is_worse", "weighted_average", "medium",
        "Previous-snapshot PD; input to PD migration and deterioration "
        "flags (risk_monitor)."),
    "pd_current": E(
        "credit_quality", ["credit_quality", "risk"], ["period_change"],
        "higher_is_worse", "weighted_average", "medium",
        "Current-snapshot PD; input to PD migration and deterioration flags "
        "(risk_monitor)."),
    "lgd_previous": E(
        "loss", ["loss", "credit_quality"], ["period_change"],
        "higher_is_worse", "weighted_average", "medium",
        "Previous-snapshot LGD; input to LGD migration analysis."),
    "lgd_current": E(
        "loss", ["loss", "credit_quality"], ["period_change"],
        "higher_is_worse", "weighted_average", "medium",
        "Current-snapshot LGD; input to LGD migration analysis."),
    "ifrs9_stage_previous": E(
        "credit_quality", ["credit_quality", "loss"], ["period_change"],
        "higher_is_worse", "distribution", "medium",
        "Previous-snapshot IFRS 9 stage; input to stage migration matrix "
        "(risk_monitor migration pairs)."),
    "ifrs9_stage_current": E(
        "credit_quality", ["credit_quality", "loss"], ["period_change"],
        "higher_is_worse", "distribution", "medium",
        "Current-snapshot IFRS 9 stage; input to stage migration matrix "
        "(risk_monitor migration pairs)."),
    "credit_impaired_obligor": E(
        "credit_quality", ["credit_quality", "risk"], _PC_PCMP_MON,
        "higher_is_worse", "share", "high",
        "Credit-impaired obligor flag (ESMA); share of book with impaired "
        "obligors."),
    "default_or_foreclosure_on_the_loan_per_basel_iii_definition": E(
        "credit_quality", ["credit_quality", "loss"], _PC_PCMP_MON,
        "higher_is_worse", "share", "medium",
        "Default/foreclosure flag under the Basel III definition; default "
        "incidence measure."),
    "default_or_foreclosure_on_the_loan_per_the_transaction_definition": E(
        "credit_quality", ["credit_quality", "loss"], _PC_PCMP_MON,
        "higher_is_worse", "share", "medium",
        "Default/foreclosure flag under the transaction definition; default "
        "incidence measure."),
    "default_amount": E(
        "loss", ["loss", "credit_quality", "exposure"], _ALL4,
        "higher_is_worse", "sum", "high",
        "Balance at the point of default (MI core); defaulted exposure "
        "measure."),
    "default_date": E(
        "credit_quality", ["credit_quality", "loss"], ["monitoring"],
        "neutral", "distribution", "medium",
        "Date of default (MI extended); default timing/vintage profile."),
    "bank_internal_rating": E(
        "credit_quality", ["credit_quality", "risk"], _PCMP_MON,
        "context_dependent", "distribution", "medium",
        "Bank internal obligor rating (SME); internal credit-quality scale "
        "without a registry-defined ordering."),
    "bank_internal_loss_given_default_lgd_estimate": E(
        "loss", ["loss", "credit_quality"], _PCMP_MON, "higher_is_worse",
        "weighted_average", "medium",
        "Bank internal LGD estimate (SME); loss severity assumption."),
    "bank_internal_loss_given_default_lgd_estimate_down_turn": E(
        "loss", ["loss", "credit_quality", "tail_risk"], _PCMP,
        "higher_is_worse", "weighted_average", "medium",
        "Bank internal downturn LGD estimate (SME); stressed loss severity."),
    "corporate_guarantor_bank_internal_1_year_probability_default": E(
        "credit_quality", ["credit_quality", "collateral"], _PCMP_MON,
        "higher_is_worse", "weighted_average", "medium",
        "Guarantor one-year PD; credit quality of guarantee support."),
    "dominion_bond_rating_service_dbrs_public_rating_equivalent": E(
        "credit_quality", ["credit_quality"], _PCMP_MON, "context_dependent",
        "distribution", "medium",
        "DBRS public-rating equivalent; external credit-quality scale.",
        display="DBRS Public Rating Equivalent"),
    "fitch_public_rating_equivalent": E(
        "credit_quality", ["credit_quality"], _PCMP_MON, "context_dependent",
        "distribution", "medium",
        "Fitch public-rating equivalent; external credit-quality scale."),
    "moody_s_public_rating_equivalent": E(
        "credit_quality", ["credit_quality"], _PCMP_MON, "context_dependent",
        "distribution", "medium",
        "Moody's public-rating equivalent; external credit-quality scale.",
        display="Moody's Public Rating Equivalent"),
    "s_p_public_rating_equivalent": E(
        "credit_quality", ["credit_quality"], _PCMP_MON, "context_dependent",
        "distribution", "medium",
        "S&P public-rating equivalent; external credit-quality scale.",
        display="S&P Public Rating Equivalent"),
    "other_public_rating": E(
        "credit_quality", ["credit_quality"], _PCMP, "context_dependent",
        "distribution", "medium",
        "Other public rating; external credit-quality scale."),
    "servicer_watchlist_code": E(
        "credit_quality", ["credit_quality", "operational_performance"],
        ["period_change", "monitoring"], "context_dependent", "distribution",
        "medium",
        "Servicer watchlist code; servicer-flagged credit concern status."),
    "special_servicing_status": E(
        "credit_quality", ["credit_quality", "loss"], _PC_PCMP_MON,
        "higher_is_worse", "share", "medium",
        "Whether the loan is in special servicing; distressed-management "
        "incidence."),
    "litigation": E(
        "credit_quality", ["credit_quality", "risk"], _PCMP_MON,
        "higher_is_worse", "share", "medium",
        "Litigation flag; share of book subject to legal action."),
    "obligor_reporting_breach": E(
        "credit_quality", ["credit_quality", "data_quality"], ["monitoring"],
        "higher_is_worse", "share", "medium",
        "Obligor breach of financial-reporting obligations; covenant-"
        "compliance signal (SME)."),
    "leveraged_transaction": E(
        "leverage", ["leverage", "credit_quality"], _PCMP,
        "higher_is_worse", "share", "medium",
        "Leveraged-transaction flag; share of highly leveraged exposures."),
    "seniority": E(
        "credit_quality", ["credit_quality", "collateral"], _PCMP,
        "context_dependent", "distribution", "medium",
        "Debt seniority ranking; claim priority mix of the book."),
    "non_recoverability_determined": E(
        "loss", ["loss"], ["period_change", "monitoring"], "higher_is_worse",
        "share", "medium",
        "Determination that amounts are non-recoverable; write-off pipeline "
        "indicator."),

    # ================= OBLIGOR FINANCIALS =================
    "revenue": E(
        "credit_quality", ["obligor_financials", "credit_quality", "income"],
        _PC_PCMP, "higher_is_better", "sum", "medium",
        "Obligor revenue; financial-strength input for corporate/SME/CRE "
        "credit assessment."),
    "most_recent_revenue": E(
        "credit_quality", ["obligor_financials", "income"], ["period_change"],
        "higher_is_better", "sum", "medium",
        "Most recent reported obligor revenue; updated financial-strength "
        "measure."),
    "ebitda": E(
        "credit_quality", ["obligor_financials", "income"], _PC_PCMP,
        "higher_is_better", "sum", "medium",
        "Obligor EBITDA; core debt-service capacity input."),
    "earnings_before_interest_taxes_depreciation_and_amortisation_ebitda": E(
        "credit_quality", ["obligor_financials", "income"], _PCMP,
        "higher_is_better", "sum", "medium",
        "Obligor EBITDA (SME financial statements)."),
    "earnings_before_interest_taxes_ebit": E(
        "credit_quality", ["obligor_financials", "income"], _PCMP,
        "higher_is_better", "sum", "medium",
        "Obligor EBIT (SME financial statements)."),
    "net_profit": E(
        "credit_quality", ["obligor_financials", "income"], _PCMP,
        "higher_is_better", "sum", "medium",
        "Obligor net profit (SME financial statements)."),
    "free_cashflow": E(
        "cashflow", ["cashflow", "obligor_financials"], _PCMP,
        "higher_is_better", "sum", "medium",
        "Obligor free cashflow; debt-service capacity measure."),
    "financial_expenses": E(
        "credit_quality", ["obligor_financials"], _PCMP, "context_dependent",
        "sum", "medium",
        "Obligor financial expenses (SME financial statements); debt-cost "
        "burden."),
    "total_debt": E(
        "leverage", ["leverage", "obligor_financials"], _PCMP,
        "higher_is_worse", "sum", "medium",
        "Obligor total debt; leverage input."),
    "long_term_debt": E(
        "leverage", ["leverage", "obligor_financials"], _PCMP,
        "higher_is_worse", "sum", "medium",
        "Obligor long-term debt (SME financial statements)."),
    "short_term_financial_debt": E(
        "leverage", ["leverage", "liquidity", "obligor_financials"], _PCMP,
        "higher_is_worse", "sum", "medium",
        "Obligor short-term financial debt; near-term refinancing burden."),
    "commercial_liabilities": E(
        "credit_quality", ["obligor_financials", "leverage"], _PCMP,
        "context_dependent", "sum", "medium",
        "Obligor commercial liabilities (SME financial statements)."),
    "total_liabilities_excluding_equity": E(
        "leverage", ["leverage", "obligor_financials"], _PCMP,
        "higher_is_worse", "sum", "medium",
        "Obligor total liabilities excluding equity (SME financial "
        "statements)."),
    "equity": E(
        "collateral", ["collateral", "obligor_financials", "leverage"],
        _PC_PCMP, "higher_is_better", "sum", "medium",
        "Equity cushion; the MI layer defines this as borrower equity in the "
        "property while the registry types it as an SME financials field — "
        "see the uncertain-entries review."),
    "enterprise_value": E(
        "valuation", ["valuation", "obligor_financials"], _PCMP,
        "higher_is_better", "sum", "medium",
        "Obligor enterprise value (SME); valuation of the obligor business."),
    "turnover_of_obligor": E(
        "credit_quality", ["obligor_financials", "income"], _PCMP,
        "higher_is_better", "sum", "medium",
        "Obligor turnover; business-scale and financial-strength measure."),
    "enterprise_size": E(
        "concentration", ["concentration", "obligor_financials"], _PCMP,
        "neutral", "distribution", "medium",
        "Obligor enterprise-size classification (ESMA enum); size-mix "
        "dimension for SME books."),
    "obligor_basel_iii_segment": E(
        "concentration", ["concentration", "credit_quality"], _PCMP,
        "neutral", "distribution", "medium",
        "Obligor Basel III segment; regulatory risk-segmentation mix."),
    "borrower_basel_iii_segment": E(
        "concentration", ["concentration", "credit_quality"], _PCMP,
        "neutral", "distribution", "medium",
        "Borrower Basel III segment (SME); regulatory risk-segmentation "
        "mix."),
    "nace_industry_code": E(
        "concentration", ["concentration"], _PCMP_MON, "neutral",
        "distribution", "medium",
        "NACE industry code (governed enum); industry concentration "
        "dimension."),
    "customer_type": E(
        "concentration", ["concentration"], _PCMP, "neutral", "distribution",
        "medium",
        "Customer type (ESMA enum); borrower-type mix dimension."),

    # ================= INCOME / AFFORDABILITY =================
    "primary_income": E(
        "credit_quality", ["affordability", "credit_quality", "income"],
        _PCMP, "higher_is_better", "average", "medium",
        "Primary borrower income (ESMA); affordability input."),
    "secondary_income": E(
        "credit_quality", ["affordability", "income"], _PCMP,
        "higher_is_better", "average", "medium",
        "Secondary borrower income (RRE); affordability input."),
    "primary_income_verification": E(
        "data_quality", ["data_quality", "credit_quality", "affordability"],
        _PCMP_MON, "context_dependent", "distribution", "medium",
        "Income verification basis (governed enum); underwriting evidence "
        "quality mix (e.g. verified vs self-certified)."),
    "debt_to_income_ratio": E(
        "leverage", ["leverage", "affordability", "risk"], _ALL4,
        "higher_is_worse", "weighted_average", "high",
        "Debt-to-income ratio (MI extended); borrower leverage and "
        "affordability metric."),
    "employment_status": E(
        "credit_quality", ["credit_quality", "affordability"], _PCMP,
        "neutral", "distribution", "medium",
        "Borrower employment status (governed enum, MI extended); income "
        "stability mix."),

    # ================= COVERAGE =================
    "current_debt_service_coverage_ratio": E(
        "coverage", ["coverage", "cashflow", "credit_quality"], _ALL4,
        "higher_is_better", "weighted_average", "high",
        "Current DSCR (MI extended); income cover of debt service, core "
        "CRE/income-property risk metric."),
    "debt_service_coverage_ratio_at_the_securitisation_date": E(
        "coverage", ["coverage", "cashflow"], _PCMP, "higher_is_better",
        "weighted_average", "medium",
        "DSCR at securitisation; static baseline for coverage deterioration "
        "comparison."),
    "current_interest_coverage_ratio": E(
        "coverage", ["coverage", "cashflow"], _ALL4, "higher_is_better",
        "weighted_average", "high",
        "Current interest coverage ratio; income cover of interest cost."),
    "interest_coverage_ratio_at_the_securitisation_date": E(
        "coverage", ["coverage", "cashflow"], _PCMP, "higher_is_better",
        "weighted_average", "medium",
        "ICR at securitisation; static baseline for coverage comparison."),

    # ================= LEVERAGE / LTV =================
    "current_loan_to_value": E(
        "leverage", ["leverage", "collateral", "risk"], _ALL4,
        "higher_is_worse", "weighted_average", "high",
        "Current LTV (MI core); weighted-average leverage metric for "
        "collateral and portfolio risk, with governed LTV buckets."),
    "indexed_loan_to_value": E(
        "leverage", ["leverage", "collateral", "risk"], _ALL4,
        "higher_is_worse", "weighted_average", "high",
        "Indexed LTV (MI core); leverage against index-updated valuation."),
    "original_loan_to_value": E(
        "leverage", ["leverage", "collateral"],
        ["portfolio_comparison", "ranking"], "higher_is_worse",
        "weighted_average", "high",
        "LTV at origination (MI core); static underwriting-leverage "
        "baseline."),
    "stressed_LTV": E(
        "leverage", ["leverage", "tail_risk", "collateral"],
        ["portfolio_comparison", "ranking", "monitoring"], "higher_is_worse",
        "weighted_average", "high",
        "Stressed LTV; leverage under a stressed valuation, a tail-risk "
        "measure.", display="Stressed LTV"),
    "ltv_cap": E(
        "eligibility", ["eligibility", "leverage"], _PCMP, "neutral",
        "weighted_average", "medium",
        "Product LTV cap (equity release); maximum permitted leverage "
        "constraint."),
    "collateralisation_ratio": E(
        "collateral", ["collateral", "leverage"], _PC_PCMP_MON,
        "higher_is_better", "weighted_average", "medium",
        "Collateralisation ratio (SME); collateral cover of the exposure."),
    "deposit_amount": E(
        "leverage", ["leverage", "collateral"], _PCMP, "higher_is_better",
        "sum", "medium",
        "Deposit amount; borrower stake reducing effective leverage."),
    "down_payment_amount": E(
        "leverage", ["leverage", "collateral"], _PCMP, "higher_is_better",
        "sum", "medium",
        "Down payment amount; borrower stake reducing effective leverage."),

    # ================= VALUATION =================
    "current_valuation_amount": E(
        "valuation", ["valuation", "collateral"], _ALL4, "higher_is_better",
        "sum", "high",
        "Most recent collateral valuation (MI core)."),
    "indexed_value": E(
        "valuation", ["valuation", "collateral"], _PC_PCMP_MON,
        "higher_is_better", "sum", "high",
        "Valuation indexed to the current period (MI core)."),
    "original_valuation_amount": E(
        "valuation", ["valuation", "collateral"], _PCMP, "neutral", "sum",
        "high",
        "Valuation at origination (MI extended); static valuation baseline."),
    "market_value": E(
        "valuation", ["valuation"], _PC_PCMP, "higher_is_better", "sum",
        "high",
        "Market value of the asset/collateral."),
    "collateral_value": E(
        "valuation", ["valuation", "collateral"], _ALL4, "higher_is_better",
        "sum", "high",
        "Collateral value; basis of collateral cover and LTV."),
    "valuation_at_securitisation": E(
        "valuation", ["valuation", "collateral"], _PCMP, "neutral", "sum",
        "medium",
        "Collateral valuation at securitisation (CRE); static baseline."),
    "vacant_possession_value_at_securitisation_date": E(
        "valuation", ["valuation", "collateral"], _PCMP, "neutral", "sum",
        "medium",
        "Vacant-possession value at securitisation (CRE); conservative "
        "valuation basis."),
    "property_portfolio_value_at_securitisation_date": E(
        "valuation", ["valuation", "collateral"], _PCMP, "neutral", "sum",
        "medium",
        "Property portfolio value at securitisation; static baseline."),
    "purchase_price": E(
        "valuation", ["valuation"], _PCMP, "neutral", "sum", "medium",
        "Purchase price of the asset; acquisition valuation basis."),
    "defaulted_underlying_exposure_purchase_price": E(
        "valuation", ["valuation", "loss"], _PCMP, "neutral", "sum",
        "medium",
        "Purchase price of defaulted exposures; NPL acquisition pricing "
        "measure."),
    "real_estate_sale_price": E(
        "loss", ["loss", "recovery", "valuation"], _PC_PCMP,
        "higher_is_better", "sum", "medium",
        "Realised real-estate sale price; recovery outcome measure."),
    "sale_price": E(
        "loss", ["loss", "recovery", "valuation"], _PC_PCMP,
        "higher_is_better", "sum", "medium",
        "Realised sale price of the asset; recovery outcome measure."),
    "option_to_buy_price": E(
        "valuation", ["valuation", "cashflow"], _PCMP, "neutral", "sum",
        "medium",
        "Lessee option-to-buy price (equipment); residual-value realisation "
        "reference."),
    "current_valuation_date": E(
        "valuation", ["valuation", "data_quality"], ["monitoring"],
        "neutral", "distribution", "medium",
        "Date of the most recent valuation (MI extended); valuation "
        "staleness profile."),
    "current_valuation_method": E(
        "valuation", ["valuation", "data_quality"], _PCMP,
        "context_dependent", "distribution", "medium",
        "Valuation method (governed enum); valuation reliability mix "
        "(e.g. full survey vs indexed/AVM)."),
    "valuation_type": E(
        "valuation", ["valuation", "data_quality"], _PCMP,
        "context_dependent", "distribution", "medium",
        "Valuation type/basis (MI core, RRE); valuation reliability mix."),
    "net_internal_floor_area_validated": E(
        "data_quality", ["data_quality", "collateral"], _PCMP_MON,
        "higher_is_better", "share", "medium",
        "Whether the floor area was validated; collateral data-quality "
        "share."),
    "current_residual_value_of_asset": E(
        "valuation", ["valuation", "collateral"], _PC_PCMP_MON,
        "higher_is_better", "sum", "medium",
        "Current residual value of the leased asset (equipment); residual "
        "value risk measure."),
    "original_residual_value_of_asset": E(
        "valuation", ["valuation", "collateral"], _PCMP, "neutral", "sum",
        "medium",
        "Original residual value of the leased asset (equipment); static "
        "baseline."),
    "securitised_residual_value": E(
        "exposure", ["exposure", "valuation"], _PCMP, "context_dependent",
        "sum", "medium",
        "Residual value securitised (equipment); residual-value exposure."),

    # ================= COLLATERAL CHARACTERISTICS =================
    "collateral_type": E(
        "collateral", ["collateral", "concentration"], _PCMP_MON, "neutral",
        "distribution", "high",
        "Collateral type (governed enum); collateral mix dimension."),
    "property_type": E(
        "collateral", ["collateral", "concentration"], _PCMP_MON, "neutral",
        "distribution", "high",
        "Property type (governed enum); collateral mix dimension."),
    "security_type": E(
        "collateral", ["collateral"], _PCMP, "neutral", "distribution",
        "medium",
        "Security type (governed enum); form of security mix."),
    "charge_type": E(
        "collateral", ["collateral"], _PCMP, "neutral", "distribution",
        "medium",
        "Charge type (governed enum); nature of the charge over collateral."),
    "lien": E(
        "collateral", ["collateral", "credit_quality"], _PCMP_MON,
        "higher_is_worse", "distribution", "high",
        "Lien/charge ranking (MI extended); first-charge positions carry "
        "stronger recovery claims than junior positions."),
    "tenure": E(
        "collateral", ["collateral", "concentration"], _PCMP,
        "context_dependent", "distribution", "high",
        "Property tenure (MI core, RRE); freehold/leasehold mix affecting "
        "collateral quality."),
    "property_leasehold_expiry": E(
        "collateral", ["collateral", "maturity"], _PCMP_MON,
        "lower_is_worse", "distribution", "medium",
        "Leasehold expiry date (CRE); short unexpired terms erode "
        "collateral value."),
    "occupancy_type": E(
        "collateral", ["collateral", "concentration"], _PCMP,
        "context_dependent", "distribution", "high",
        "Occupancy type (MI core); owner-occupied vs rental mix."),
    "property_status": E(
        "collateral", ["collateral"], _PCMP, "context_dependent",
        "distribution", "medium",
        "Property status (governed enum); collateral condition/status mix."),
    "status_of_properties": E(
        "collateral", ["collateral"], _PCMP, "context_dependent",
        "distribution", "medium",
        "Status of properties (CRE enum); collateral status mix."),
    "finished_property": E(
        "collateral", ["collateral", "risk"], _PCMP, "higher_is_better",
        "share", "medium",
        "Whether the property is finished; development/completion risk "
        "share."),
    "new_build": E(
        "collateral", ["collateral"], _PCMP, "context_dependent", "share",
        "medium",
        "New-build flag (RRE); new-build share of the book."),
    "year_built": E(
        "collateral", ["collateral"], _PCMP, "neutral", "distribution",
        "medium",
        "Year the property was built; property age profile."),
    "year_of_manufacture_construction": E(
        "collateral", ["collateral", "valuation"], _PCMP, "neutral",
        "distribution", "medium",
        "Year of manufacture/construction (equipment); asset age and "
        "depreciation profile."),
    "new_or_used": E(
        "collateral", ["collateral"], _PCMP, "context_dependent",
        "distribution", "medium",
        "New or used asset (governed enum); asset condition mix."),
    "energy_performance_certificate_value": E(
        "collateral", ["collateral", "eligibility"], _PCMP,
        "context_dependent", "distribution", "medium",
        "EPC rating (governed enum); collateral energy-efficiency mix with "
        "eligibility relevance."),
    "number_of_bedrooms": E(
        "collateral", ["collateral"], _PCMP, "neutral", "distribution",
        "medium",
        "Number of bedrooms (MI extended); property-size mix dimension."),
    "number_of_properties_at_data_cut_off_date": E(
        "collateral", ["collateral", "exposure"], _PCMP, "neutral",
        "average", "medium",
        "Number of properties securing the loan (MI extended); collateral "
        "pool breadth."),
    "number_of_units": E(
        "collateral", ["collateral"], _PCMP, "neutral", "average", "medium",
        "Number of units in the property; income-property scale measure."),
    "number_of_leased_objects": E(
        "collateral", ["collateral"], _PCMP, "neutral", "average", "medium",
        "Number of leased objects (equipment); collateral pool breadth per "
        "lease."),
    "guarantor_type": E(
        "collateral", ["collateral", "credit_quality"], _PCMP, "neutral",
        "distribution", "medium",
        "Guarantor type (governed enum); guarantee-support mix."),
    "guarantee_type": E(
        "collateral", ["collateral", "credit_quality"], _PCMP, "neutral",
        "distribution", "medium",
        "Guarantee type; form of credit support mix."),
    "recourse": E(
        "collateral", ["collateral", "credit_quality"], _PCMP,
        "higher_is_better", "share", "medium",
        "Recourse flag; share of exposures with recourse to the obligor."),
    "asset_insurance": E(
        "collateral", ["collateral", "risk"], _PCMP, "higher_is_better",
        "share", "medium",
        "Asset insurance flag (SME); insured share of collateral."),
    "total_proceeds_from_other_collateral_or_guarantees": E(
        "loss", ["recovery", "collateral"], _PC_PCMP, "higher_is_better",
        "sum", "medium",
        "Proceeds from other collateral or guarantees; recovery source "
        "measure."),

    # ================= EQUITY RELEASE SPECIFIC =================
    "negative_equity_guarantee": E(
        "tail_risk", ["tail_risk", "collateral", "eligibility"], _PCMP_MON,
        "higher_is_worse", "share", "high",
        "No-negative-equity guarantee flag (MI core); NNEG share drives "
        "lender tail risk from collateral shortfall at redemption.",
        assets=["equity_release"]),
    "protected_equity_flag": E(
        "collateral", ["collateral", "eligibility"], _PCMP,
        "context_dependent", "share", "medium",
        "Protected-equity flag (MI core, derived from the percentage); "
        "share of loans with ring-fenced borrower equity.",
        assets=["equity_release"]),
    "protected_equity_percentage": E(
        "collateral", ["collateral", "leverage"], _PCMP,
        "context_dependent", "weighted_average", "medium",
        "Protected-equity share of the property (MI core); reduces the "
        "collateral available to the lender.",
        assets=["equity_release"]),
    "youngest_borrower_age": E(
        "maturity", ["maturity", "tail_risk", "risk"], _PC_PCMP_MON,
        "context_dependent", "average", "high",
        "Age of the youngest borrower (MI core); drives expected loan "
        "duration and NNEG exposure on joint-life equity-release loans.",
        assets=["equity_release"]),
    "borrower_1_age": E(
        "maturity", ["maturity", "tail_risk"], _PCMP, "context_dependent",
        "average", "medium",
        "Primary borrower age (equity release); duration/longevity input.",
        assets=["equity_release"]),
    "borrower_2_age": E(
        "maturity", ["maturity", "tail_risk"], _PCMP, "context_dependent",
        "average", "medium",
        "Second borrower age (equity release); joint-life duration input.",
        assets=["equity_release"]),
    "health_impairment_flag": E(
        "eligibility", ["eligibility", "tail_risk"], _PCMP,
        "context_dependent", "share", "medium",
        "Health impairment flag; enhanced-terms eligibility and longevity "
        "assumption input."),
    "loan_redemption_flag": E(
        "cashflow", ["cashflow"], _PC_PCMP_MON, "context_dependent",
        "share", "high",
        "Whether the loan has redeemed (MI core); redemption/attrition "
        "incidence.",
        assets=["equity_release"]),

    # ================= PRICING / RATES =================
    "current_interest_rate": E(
        "pricing", ["pricing", "return"], _ALL4, "context_dependent",
        "weighted_average", "high",
        "Current interest rate (MI core); weighted-average portfolio "
        "coupon/yield measure."),
    "current_interest_rate_margin": E(
        "pricing", ["pricing", "return"], _ALL4, "context_dependent",
        "weighted_average", "high",
        "Current interest margin/spread (MI extended)."),
    "original_interest_rate": E(
        "pricing", ["pricing"], _PCMP, "neutral", "weighted_average",
        "medium",
        "Interest rate at origination; static pricing baseline."),
    "interest_rate_at_the_securitisation_date": E(
        "pricing", ["pricing"], _PCMP, "neutral", "weighted_average",
        "medium",
        "Interest rate at securitisation; static pricing baseline."),
    "current_index_rate": E(
        "pricing", ["pricing"], ["period_change", "monitoring"], "neutral",
        "weighted_average", "medium",
        "Current index/reference rate underlying floating pricing."),
    "interest_rate_type": E(
        "pricing", ["pricing", "concentration"], _PC_PCMP_MON,
        "context_dependent", "distribution", "high",
        "Interest rate type (MI core enum); fixed/floating mix and rate "
        "risk profile."),
    "current_interest_rate_index": E(
        "pricing", ["pricing", "concentration"], _PCMP, "neutral",
        "distribution", "medium",
        "Reference index (governed enum); basis-risk mix of floating-rate "
        "exposures."),
    "interest_rate_cap": E(
        "pricing", ["pricing", "risk"], _PCMP, "context_dependent",
        "weighted_average", "medium",
        "Contractual interest rate cap; bounds borrower payment shock and "
        "lender yield."),
    "interest_rate_floor": E(
        "pricing", ["pricing", "risk"], _PCMP, "context_dependent",
        "weighted_average", "medium",
        "Contractual interest rate floor; bounds yield downside."),
    "interest_rate_reset_interval": E(
        "pricing", ["pricing", "risk"], _PCMP, "context_dependent",
        "average", "medium",
        "Interest-rate reset interval; repricing frequency and rate-risk "
        "measure."),
    "loan_hedged": E(
        "pricing", ["pricing", "risk"], _PCMP_MON, "higher_is_better",
        "share", "medium",
        "Whether the loan is hedged; hedged share of rate risk."),
    "early_repayment_charge": E(
        "cashflow", ["cashflow", "pricing"], _PCMP, "context_dependent",
        "share", "medium",
        "Early repayment charge applies (MI extended); prepayment "
        "protection share."),
    "prepayment_fee": E(
        "cashflow", ["cashflow", "pricing"], ["period_change"], "neutral",
        "sum", "medium",
        "Prepayment fee amount; compensation income on early repayment."),

    # ================= CASHFLOW / COLLECTIONS =================
    "interest_collections_in_period": E(
        "cashflow", ["cashflow", "income"], _PC_PCMP, "higher_is_better",
        "sum", "high",
        "Interest collected in the period; core cash income measure."),
    "redemptions_received_in_period": E(
        "cashflow", ["cashflow"], _PC_PCMP_MON, "context_dependent", "sum",
        "high",
        "Redemption proceeds received in the period (MI core); attrition "
        "cashflow."),
    "principal_advances_in_period": E(
        "cashflow", ["cashflow", "exposure"], ["period_change"], "neutral",
        "sum", "high",
        "Principal advanced in the period; new-lending cash outflow."),
    "recoveries_in_period": E(
        "loss", ["recovery", "cashflow", "loss"], _PC_PCMP_MON,
        "higher_is_better", "sum", "high",
        "Recoveries received in the period (MI core)."),
    "cumulative_recoveries": E(
        "loss", ["recovery", "loss"], _PC_PCMP, "higher_is_better", "sum",
        "high",
        "Cumulative recoveries to date (ESMA); realised recovery "
        "performance."),
    "cumulative_prepayments": E(
        "cashflow", ["cashflow"], _PC_PCMP_MON, "context_dependent", "sum",
        "high",
        "Cumulative unscheduled prepayments (MI extended); prepayment "
        "behaviour measure."),
    "unscheduled_principal_collections": E(
        "cashflow", ["cashflow"], ["period_change"], "context_dependent",
        "sum", "medium",
        "Unscheduled principal collected in the period (CRE); prepayment "
        "cashflow."),
    "prepayment_interest_excess_shortfall": E(
        "cashflow", ["cashflow"], ["period_change"], "context_dependent",
        "sum", "medium",
        "Prepayment interest excess/shortfall; signed cashflow variance on "
        "prepayment."),
    "contractual_annual_rental_income": E(
        "cashflow", ["cashflow", "income"], _ALL4, "higher_is_better", "sum",
        "high",
        "Contractual annual rental income (MI extended); income-producing "
        "collateral cashflow."),
    "net_operating_income_at_securitisation": E(
        "cashflow", ["cashflow", "income"], _PCMP, "higher_is_better", "sum",
        "high",
        "NOI at securitisation (MI extended); static income baseline for "
        "coverage comparison."),
    "most_recent_operating_expenses": E(
        "operational_performance", ["operational_performance", "cashflow"],
        _PC_PCMP, "higher_is_worse", "sum", "medium",
        "Most recent operating expenses (CRE); property cost performance."),
    "most_recent_capital_expenditure": E(
        "operational_performance", ["operational_performance", "cashflow"],
        ["period_change"], "context_dependent", "sum", "medium",
        "Most recent capital expenditure (CRE); property investment level."),
    "income_expiring_1_12_months": E(
        "cashflow", ["cashflow", "income", "maturity"], _PCMP_MON,
        "context_dependent", "sum", "medium",
        "Lease income expiring within 12 months; near-term income rollover "
        "risk."),
    "income_expiring_13_24_months": E(
        "cashflow", ["cashflow", "income", "maturity"], _PCMP,
        "context_dependent", "sum", "medium",
        "Lease income expiring in 13-24 months; income rollover profile."),
    "income_expiring_25_36_months": E(
        "cashflow", ["cashflow", "income", "maturity"], _PCMP,
        "context_dependent", "sum", "medium",
        "Lease income expiring in 25-36 months; income rollover profile."),
    "income_expiring_37_48_months": E(
        "cashflow", ["cashflow", "income", "maturity"], _PCMP,
        "context_dependent", "sum", "medium",
        "Lease income expiring in 37-48 months; income rollover profile."),
    "income_expiring_49_months": E(
        "cashflow", ["cashflow", "income", "maturity"], _PCMP,
        "context_dependent", "sum", "medium",
        "Lease income expiring beyond 49 months; income rollover profile."),
    "economic_occupancy_at_securitisation": E(
        "operational_performance", ["operational_performance", "income"],
        _PCMP, "higher_is_better", "weighted_average", "medium",
        "Economic occupancy at securitisation (CRE); income-generation "
        "baseline."),
    "physical_occupancy_at_securitisation": E(
        "operational_performance", ["operational_performance", "income"],
        _PCMP, "higher_is_better", "weighted_average", "medium",
        "Physical occupancy at securitisation (CRE); utilisation baseline."),
    "amounts_held_in_escrow": E(
        "liquidity", ["liquidity", "collateral"],
        ["period_change", "monitoring"], "higher_is_better", "sum", "medium",
        "Amounts held in escrow (CRE); reserve liquidity supporting the "
        "exposure."),
    "amounts_added_to_escrows_in_current_period": E(
        "liquidity", ["liquidity", "cashflow"], ["period_change"], "neutral",
        "sum", "medium",
        "Escrow additions in the period (CRE); reserve funding flow."),
    "target_escrow_amounts_reserves": E(
        "liquidity", ["liquidity"], ["monitoring"], "neutral", "sum",
        "medium",
        "Target escrow/reserve amounts (CRE); required reserve level for "
        "adequacy monitoring."),
    "total_reserve_balance": E(
        "liquidity", ["liquidity"], ["period_change", "monitoring"],
        "higher_is_better", "sum", "medium",
        "Total reserve balance; liquidity support available to the "
        "structure."),

    # ================= LOSS / WORKOUT =================
    "allocated_losses": E(
        "loss", ["loss", "risk"], _ALL4, "higher_is_worse", "sum", "high",
        "Losses allocated to the loan (MI core); realised credit loss "
        "measure."),
    "liquidation_expense": E(
        "loss", ["loss", "recovery"], _PC_PCMP, "higher_is_worse", "sum",
        "medium",
        "Liquidation expenses; costs reducing net recoveries."),
    "foreclosure_cost": E(
        "loss", ["loss", "recovery"], _PCMP, "higher_is_worse", "sum",
        "medium",
        "Foreclosure costs; workout cost reducing net recovery."),
    "net_proceeds_received_on_liquidation": E(
        "loss", ["recovery", "loss"], _PC_PCMP, "higher_is_better", "sum",
        "medium",
        "Net liquidation proceeds; realised recovery outcome."),
    "expected_timing_of_recoveries": E(
        "forecast", ["forecast", "recovery", "loss"], ["monitoring"],
        "lower_is_better", "average", "medium",
        "Expected timing of recoveries (ESMA); forward-looking recovery "
        "horizon."),
    "recovery_source": E(
        "loss", ["recovery", "loss"], _PCMP, "neutral", "distribution",
        "medium",
        "Source of recoveries (governed enum); recovery-channel mix."),
    "reason_for_default_or_foreclosure": E(
        "loss", ["loss", "credit_quality"], _PCMP, "neutral",
        "distribution", "medium",
        "Reason for default or foreclosure (governed enum); default-driver "
        "mix."),
    "workout_strategy_code": E(
        "loss", ["loss", "operational_performance"], ["monitoring"],
        "neutral", "distribution", "medium",
        "Workout strategy code (CRE); resolution-approach mix."),
    "work_out_process_complete": E(
        "loss", ["loss", "operational_performance"],
        ["period_change", "monitoring"], "higher_is_better", "share",
        "medium",
        "Whether the workout process is complete (SME); resolution "
        "progress share."),

    # ================= MATURITY / TERM / SEASONING =================
    "maturity_date": E(
        "maturity", ["maturity"], _PCMP_MON, "neutral", "distribution",
        "high",
        "Scheduled maturity date (MI core); maturity/refinancing profile "
        "with governed maturity-year cohorts."),
    "origination_date": E(
        "maturity", ["maturity", "seasoning"], _PCMP, "neutral",
        "distribution", "high",
        "Origination date (MI core); vintage cohort basis and seasoning "
        "input."),
    "original_term": E(
        "maturity", ["maturity"], _PCMP, "neutral", "average", "high",
        "Original loan term (MI extended) with governed term buckets."),
    "weighted_average_life": E(
        "maturity", ["maturity", "cashflow"], _PC_PCMP,
        "context_dependent", "average", "high",
        "Weighted average life; expected duration of the exposure."),
    "duration_of_extension_option": E(
        "maturity", ["maturity"], _PCMP, "context_dependent", "average",
        "medium",
        "Duration of the extension option; potential maturity extension."),
    "date_of_lease_expiration": E(
        "maturity", ["maturity", "cashflow"], ["monitoring"], "neutral",
        "distribution", "medium",
        "Lease expiration date (equipment); income and asset return "
        "horizon."),
    "weighted_average_lease_terms": E(
        "maturity", ["maturity", "cashflow"], _PCMP, "higher_is_better",
        "average", "medium",
        "Weighted average lease term (equipment); contracted income "
        "duration."),
    "number_of_payments_before_securitisation": E(
        "maturity", ["seasoning", "maturity"], _PCMP, "context_dependent",
        "average", "medium",
        "Payments made before securitisation; seasoning measure."),

    # ================= CONCENTRATION / MIX DIMENSIONS =================
    "geographic_region_obligor": E(
        "concentration", ["concentration", "geography"], _PC_PCMP_MON,
        "neutral", "distribution", "high",
        "Obligor NUTS3 region (MI core); geographic concentration "
        "dimension monitored by the risk monitor."),
    "geographic_region_collateral": E(
        "concentration", ["concentration", "geography", "collateral"],
        _PC_PCMP_MON, "neutral", "distribution", "high",
        "Collateral NUTS3 region (MI extended); geographic concentration "
        "dimension."),
    "collateral_geography": E(
        "concentration", ["concentration", "geography", "collateral"],
        _PC_PCMP_MON, "neutral", "distribution", "high",
        "Readable collateral region label (MI core Region); geographic "
        "concentration dimension."),
    "postcode": E(
        "concentration", ["concentration", "geography"], _PCMP, "neutral",
        "distribution", "medium",
        "Property postcode (MI extended); granular geographic drilldown "
        "dimension."),
    "borrower_jurisdiction": E(
        "concentration", ["concentration", "geography"], _PCMP, "neutral",
        "distribution", "medium",
        "Borrower legal jurisdiction (MI core); country-mix dimension."),
    "broker_channel": E(
        "concentration", ["concentration"], _PC_PCMP_MON, "neutral",
        "distribution", "high",
        "Broker/origination channel (MI core); channel concentration "
        "monitored by the risk monitor."),
    "origination_channel": E(
        "concentration", ["concentration"], _PC_PCMP_MON, "neutral",
        "distribution", "high",
        "Origination channel (governed enum); channel concentration "
        "monitored by the risk monitor."),
    "originator_name": E(
        "concentration", ["concentration"], _PCMP_MON, "neutral",
        "distribution", "high",
        "Originator (MI core); originator concentration dimension."),
    "servicer_name": E(
        "concentration", ["concentration", "operational_performance"],
        _PCMP, "neutral", "distribution", "medium",
        "Servicer; servicer concentration and operational-dependency "
        "dimension."),
    "product_type": E(
        "concentration", ["concentration", "product_mix"], _PCMP_MON,
        "neutral", "distribution", "high",
        "Product type (governed enum); product-mix concentration "
        "dimension."),
    "erm_product_type": E(
        "concentration", ["concentration", "product_mix"], _PC_PCMP_MON,
        "neutral", "distribution", "high",
        "Equity-release product type (MI core); product-mix dimension "
        "monitored by the risk monitor.",
        assets=["equity_release"], display="ERM Product Type"),
    "erm_sub_product_type": E(
        "concentration", ["concentration", "product_mix"], _PCMP, "neutral",
        "distribution", "medium",
        "Equity-release sub-product type (MI core); product-variant mix.",
        assets=["equity_release"], display="ERM Sub Product Type"),
    "debt_type": E(
        "concentration", ["concentration", "product_mix"], _PCMP, "neutral",
        "distribution", "medium",
        "Debt type (governed enum); instrument-mix dimension."),
    "asset_type": E(
        "concentration", ["concentration", "product_mix"], _PCMP, "neutral",
        "distribution", "medium",
        "Asset type (SME, governed enum); financed-asset mix dimension."),
    "amortisation_type": E(
        "cashflow", ["cashflow", "concentration", "product_mix"],
        _PC_PCMP_MON, "context_dependent", "distribution", "high",
        "Amortisation/repayment profile (MI core enum); interest-only vs "
        "repayment mix monitored by the risk monitor."),
    "purpose": E(
        "concentration", ["concentration", "credit_quality"], _PCMP,
        "neutral", "distribution", "medium",
        "Loan purpose (governed enum); purpose-mix dimension."),
    "special_scheme": E(
        "eligibility", ["eligibility", "concentration"], _PCMP, "neutral",
        "distribution", "medium",
        "Special scheme participation; government/support-scheme mix."),
    "exposure_currency_denomination": E(
        "concentration", ["concentration", "risk"], _PCMP, "neutral",
        "distribution", "medium",
        "Exposure currency (core canonical); currency-mix and FX-risk "
        "dimension."),
    "source_portfolio_type": E(
        "concentration", ["concentration"], _PCMP, "neutral",
        "distribution", "medium",
        "Direct vs acquired book (MI segmentation dimension); origination-"
        "source mix."),
    "originator_affiliate": E(
        "eligibility", ["eligibility"], _PCMP, "context_dependent", "share",
        "medium",
        "Whether the originator is an affiliate (MI extended); risk-"
        "retention/eligibility relevant share."),
}

# --------------------------------------------------------------------------- #
# UNCERTAIN ENTRIES (tracked, not forced into the registry)
# --------------------------------------------------------------------------- #
# status: "excluded_pending_review" — NOT written to the registry; a human
#         decision is required to promote the field.
#         "included_ambiguous"      — written to the registry (also present in
#         CURATION) but flagged for review because its definition is ambiguous.

UNCERTAIN: Dict[str, Dict[str, str]] = {
    "maximum_balance": {
        "status": "excluded_pending_review",
        "definition": "analytics field, no format defined in the registry",
        "proposed": "exposure (facility maximum balance, sum)",
        "reason": "No format or business definition in the canonical "
                  "registry; overlaps total_credit_limit.",
        "decision": "Confirm definition vs total_credit_limit, then include "
                    "or retire.",
    },
    "loan_entered_arrears": {
        "status": "excluded_pending_review",
        "definition": "analytics decimal, performance layer",
        "proposed": "payment_performance (arrears-entry incidence, share)",
        "reason": "Decimal type for what reads as an event/flag; semantics "
                  "(count? flag? amount?) are not defined.",
        "decision": "Clarify whether this is a flag, count or amount, then "
                    "classify.",
    },
    "bank_internal_rating_prior_to_default": {
        "status": "excluded_pending_review",
        "definition": "analytics decimal, performance layer (SME)",
        "proposed": "credit_quality (pre-default rating, distribution)",
        "reason": "Niche loss-analysis field; rating scale and ordering are "
                  "not defined in the registry.",
        "decision": "Confirm scale/ordering before enabling comparisons.",
    },
    "number_of_employees": {
        "status": "excluded_pending_review",
        "definition": "analytics decimal, SME obligor size",
        "proposed": "credit_quality (obligor size, average)",
        "reason": "Firm-size fact with weak direct analytical use; "
                  "enterprise_size already provides the governed size mix.",
        "decision": "Include only if headcount-based analysis is required.",
    },
    "customer_segment": {
        "status": "excluded_pending_review",
        "definition": "analytics field, no format defined (SME)",
        "proposed": "concentration (segment mix, distribution)",
        "reason": "No format or value set defined in the registry.",
        "decision": "Define the segment taxonomy, then include as a mix "
                    "dimension.",
    },
    "borrower_2_income": {
        "status": "excluded_pending_review",
        "definition": "analytics decimal (equity release)",
        "proposed": "credit_quality (affordability, average)",
        "reason": "Income is not an underwriting driver for equity-release "
                  "lifetime mortgages; analytical relevance unclear.",
        "decision": "Include only if affordability analysis is extended to "
                    "ERM books.",
    },
    "borrower_deposit_amount": {
        "status": "excluded_pending_review",
        "definition": "analytics field, no format defined (SME)",
        "proposed": "leverage (borrower stake, sum)",
        "reason": "No format defined; overlaps deposit_amount / "
                  "down_payment_amount.",
        "decision": "Deduplicate against deposit_amount, then include one.",
    },
    "final_margin": {
        "status": "excluded_pending_review",
        "definition": "analytics field, no format defined",
        "proposed": "pricing (post-revision margin, weighted_average)",
        "reason": "No format or definition distinguishing it from "
                  "current_interest_rate_margin.",
        "decision": "Confirm definition vs current margin, then include or "
                    "retire.",
    },
    "ground_rent_payable": {
        "status": "excluded_pending_review",
        "definition": "regulatory decimal (ESMA)",
        "proposed": "cashflow (leasehold cost burden, sum)",
        "reason": "Analytically relevant for UK leasehold risk but no "
                  "existing MI usage or derived metric to anchor it.",
        "decision": "Include if leasehold-cost analysis is required.",
    },
    "rent_payable": {
        "status": "excluded_pending_review",
        "definition": "regulatory decimal (ESMA)",
        "proposed": "cashflow (rental obligation or income; unclear)",
        "reason": "Direction of the rent (payable by borrower vs received) "
                  "is ambiguous across annex contexts.",
        "decision": "Confirm whether this is an income or a cost field per "
                    "annex, then classify.",
    },
    "number_of_collateral_items_securing_the_loan": {
        "status": "excluded_pending_review",
        "definition": "analytics field, no format defined",
        "proposed": "collateral (pool breadth, average)",
        "reason": "No format defined; overlaps "
                  "number_of_properties_at_data_cut_off_date.",
        "decision": "Deduplicate against the properties count, then include "
                    "one.",
    },
    "unconditional_corporate_third_party_guarantee_amount": {
        "status": "excluded_pending_review",
        "definition": "analytics field, no format defined (SME)",
        "proposed": "collateral (guarantee support amount, sum)",
        "reason": "No format defined in the registry.",
        "decision": "Confirm the field is populated/typed, then include.",
    },
    "unconditional_personal_guarantee_amount": {
        "status": "excluded_pending_review",
        "definition": "analytics field, no format defined (SME)",
        "proposed": "collateral (guarantee support amount, sum)",
        "reason": "No format defined in the registry.",
        "decision": "Confirm the field is populated/typed, then include.",
    },
    "interest_reset_period": {
        "status": "excluded_pending_review",
        "definition": "analytics field, no format defined",
        "proposed": "pricing (repricing frequency, average)",
        "reason": "Duplicates interest_rate_reset_interval without a "
                  "defined format.",
        "decision": "Retire in favour of interest_rate_reset_interval or "
                    "define separately.",
    },
    "securitised_loan_amount": {
        "status": "excluded_pending_review",
        "definition": "analytics field, no format defined",
        "proposed": "exposure (securitised amount, sum)",
        "reason": "No format defined; overlaps current/original principal "
                  "balances at securitisation.",
        "decision": "Confirm definition, then include or retire.",
    },
    "total_credit_limit_used": {
        "status": "excluded_pending_review",
        "definition": "analytics field, no format defined",
        "proposed": "exposure (limit utilisation input, sum)",
        "reason": "No format defined; utilisation is better expressed as a "
                  "derived ratio of drawn/limit which does not yet exist.",
        "decision": "Define a governed utilisation metric instead of the "
                    "raw field.",
    },
    "prior_balances": {
        "status": "excluded_pending_review",
        "definition": "analytics field, no format defined",
        "proposed": "exposure (prior-period balance, sum)",
        "reason": "No format defined; prior_principal_balances (typed) "
                  "covers the same concept.",
        "decision": "Retire in favour of prior_principal_balances.",
    },
    "borrower_1_gender": {
        "status": "excluded_pending_review",
        "definition": "analytics string (equity release); MI core dimension",
        "proposed": "concentration (demographic mix for joint-life "
                    "mortality analysis)",
        "reason": "Demographic attribute, not a metric; relevant to "
                  "equity-release mortality/NNEG modelling but no governed "
                  "mortality analytics exist yet.",
        "decision": "Include when mortality/longevity analytics are built.",
    },
    "borrower_2_gender": {
        "status": "excluded_pending_review",
        "definition": "analytics string (equity release); MI core dimension",
        "proposed": "concentration (demographic mix for joint-life "
                    "mortality analysis)",
        "reason": "Same as borrower_1_gender.",
        "decision": "Include when mortality/longevity analytics are built.",
    },
    "borrower_1_date_of_death": {
        "status": "excluded_pending_review",
        "definition": "analytics date (equity release)",
        "proposed": "cashflow (mortality/redemption event input)",
        "reason": "Event date, not a metric; mortality-experience analysis "
                  "would consume a derived rate, which does not yet exist.",
        "decision": "Keep excluded until mortality-experience analytics are "
                    "defined.",
    },
    "borrower_2_date_of_death": {
        "status": "excluded_pending_review",
        "definition": "analytics date (equity release)",
        "proposed": "cashflow (mortality/redemption event input)",
        "reason": "Same as borrower_1_date_of_death.",
        "decision": "Keep excluded until mortality-experience analytics are "
                    "defined.",
    },
    "date_of_financials": {
        "status": "excluded_pending_review",
        "definition": "regulatory date (ESMA)",
        "proposed": "data_quality (financials staleness input)",
        "reason": "Raw date; the analytical concept is a derived staleness "
                  "measure that does not yet exist.",
        "decision": "Define a governed staleness metric if financials "
                    "recency monitoring is required.",
    },
    "most_recent_financials_as_of_end_date": {
        "status": "excluded_pending_review",
        "definition": "regulatory date (ESMA)",
        "proposed": "data_quality (financials staleness input)",
        "reason": "Same as date_of_financials.",
        "decision": "Define a governed staleness metric if required.",
    },
    "regular_interest_instalment": {
        "status": "excluded_pending_review",
        "definition": "analytics field, no format defined",
        "proposed": "cashflow (scheduled instalment, sum)",
        "reason": "No format defined; overlaps "
                  "total_scheduled_principal_interest_due.",
        "decision": "Confirm definition, then include or retire.",
    },
    "regular_principal_instalment": {
        "status": "excluded_pending_review",
        "definition": "analytics field, no format defined",
        "proposed": "cashflow (scheduled instalment, sum)",
        "reason": "No format defined; overlaps "
                  "total_scheduled_principal_interest_due.",
        "decision": "Confirm definition, then include or retire.",
    },
    "equity": {
        "status": "included_ambiguous",
        "definition": "analytics decimal, portfolio_type sme; MI layer "
                      "describes it as borrower equity in the property",
        "proposed": "collateral (equity cushion, sum) — included with "
                    "medium confidence",
        "reason": "The canonical registry types equity as an SME balance-"
                  "sheet field while the MI semantics layer uses it as "
                  "property equity for equity release.",
        "decision": "Confirm a single definition (or split into two "
                    "fields).",
    },
}

# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #


def load_canonical_fields(source: Path) -> Dict[str, dict]:
    data = yaml.safe_load(Path(source).read_text(encoding="utf-8")) or {}
    fields = data.get("fields")
    if not isinstance(fields, dict) or not fields:
        raise ValueError(f"No 'fields' mapping found in registry: {source}")
    return fields


def _asset_applicability(name: str, meta: dict,
                         override: Optional[List[str]]) -> List[str]:
    if override:
        return sorted(override)
    ptype = str(meta.get("portfolio_type", "") or "").strip().lower()
    mapped = ASSET_APPLICABILITY_MAP.get(ptype)
    if mapped is None:
        raise ValueError(
            f"{name}: unmapped portfolio_type {ptype!r}; extend "
            f"ASSET_APPLICABILITY_MAP")
    return [mapped]


def _validate_curated(name: str, cur: Dict[str, Any]) -> None:
    tx = TAXONOMY
    if cur["concept"] not in tx["analytical_concepts"]:
        raise ValueError(f"{name}: unknown analytical_concept {cur['concept']!r}")
    for c in cur["categories"]:
        if c not in tx["categories"]:
            raise ValueError(f"{name}: unknown category {c!r}")
    if not cur["categories"]:
        raise ValueError(f"{name}: at least one category required")
    for t in cur["tags"]:
        if t not in tx["workflow_tags"]:
            raise ValueError(f"{name}: unknown workflow_tag {t!r}")
    if cur["directionality"] not in tx["directionality"]:
        raise ValueError(f"{name}: invalid directionality "
                         f"{cur['directionality']!r}")
    if cur["aggregation"] not in tx["aggregation_types"]:
        raise ValueError(f"{name}: invalid aggregation_type "
                         f"{cur['aggregation']!r}")
    if cur["confidence"] not in tx["confidence"]:
        raise ValueError(f"{name}: invalid confidence {cur['confidence']!r}")
    if not str(cur["rationale"]).strip():
        raise ValueError(f"{name}: rationale required")


def build_registry(source: Path = DEFAULT_SOURCE) -> Dict[str, Any]:
    fields = load_canonical_fields(source)

    missing = sorted(set(CURATION) - set(fields))
    if missing:
        raise ValueError(
            "Curated fields not found in the canonical registry: "
            + ", ".join(missing))

    for name, info in UNCERTAIN.items():
        if name not in fields:
            raise ValueError(
                f"Uncertain field not found in the canonical registry: {name}")
        in_curation = name in CURATION
        if info["status"] == "excluded_pending_review" and in_curation:
            raise ValueError(
                f"{name}: excluded_pending_review but present in CURATION")
        if info["status"] == "included_ambiguous" and not in_curation:
            raise ValueError(
                f"{name}: included_ambiguous but missing from CURATION")

    out_fields: Dict[str, dict] = {}
    for name in sorted(CURATION):
        cur = CURATION[name]
        _validate_curated(name, cur)
        meta = fields[name] or {}
        assets = _asset_applicability(name, meta, cur["assets"])
        for a in assets:
            if a not in TAXONOMY["asset_applicability"]:
                raise ValueError(f"{name}: invalid asset_applicability {a!r}")
        materiality = cur["materiality"]
        if materiality is None:
            materiality = cur["aggregation"] in _MATERIALITY_AGGREGATIONS
        out_fields[name] = {
            "source_field": name,
            "display_name": cur["display"] or _display_name(name),
            "analytical_concept": cur["concept"],
            "categories": list(cur["categories"]),
            "workflow_tags": list(cur["tags"]),
            "directionality": cur["directionality"],
            "aggregation_type": cur["aggregation"],
            "comparable_across_portfolios": bool(cur["comparable"]),
            "supports_materiality_assessment": bool(materiality),
            "asset_applicability": assets,
            "confidence": cur["confidence"],
            "rationale": cur["rationale"],
        }

    metadata = {
        "version": VERSION,
        "source_registry": "config/system/fields_registry.yaml",
        "generator": "scripts/build_business_semantics_registry.py",
        "description": (
            "Governed Business Semantics Registry: the subset of canonical "
            "registry fields that represent meaningful analytical concepts, "
            "enriched with controlled metadata for period-change analysis, "
            "portfolio risk comparison, ranking and monitoring workflows. "
            "See docs/business_semantics_registry_review.md for the full "
            "review, exclusions and uncertain entries."),
        "source_field_count": len(fields),
        "included_field_count": len(out_fields),
        "uncertain_pending_review_count": sum(
            1 for u in UNCERTAIN.values()
            if u["status"] == "excluded_pending_review"),
        "taxonomy": {k: sorted(v) for k, v in TAXONOMY.items()},
    }
    return {"metadata": metadata, "fields": out_fields}


class _NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):  # pragma: no cover - trivial
        return True


HEADER = (
    "# AUTO-GENERATED by scripts/build_business_semantics_registry.py\n"
    "# Governed Business Semantics Registry over the canonical field "
    "registry.\n"
    "# Curated allowlist + controlled taxonomy — see "
    "docs/business_semantics_registry_review.md.\n"
    "# Do NOT edit by hand: change the CURATION in the build script and\n"
    "# regenerate with --force (the script refuses to overwrite otherwise).\n"
)


def dump_registry(registry: Dict[str, Any]) -> str:
    body = yaml.dump(registry, Dumper=_NoAliasDumper, sort_keys=False,
                     allow_unicode=True, width=100, default_flow_style=False)
    return HEADER + body


def write_registry(text: str, output: Path, force: bool) -> None:
    output = Path(output)
    if output.exists() and not force:
        existing = output.read_text(encoding="utf-8")
        if existing == text:
            print(f"Up to date: {output}")
            return
        raise SystemExit(
            f"REFUSING to overwrite reviewed registry {output} — it exists "
            f"and differs from the regenerated content. Re-run with --force "
            f"to overwrite, or use --check to see that it differs.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(f"Wrote {output}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the governed Business Semantics Registry.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing (reviewed) output file.")
    parser.add_argument("--check", action="store_true",
                        help="Verify the committed output matches "
                             "regeneration; do not write.")
    args = parser.parse_args(argv)

    if not args.source.exists():
        print(f"ERROR: source registry not found: {args.source}",
              file=sys.stderr)
        return 2

    registry = build_registry(args.source)
    text = dump_registry(registry)

    if args.check:
        if not args.output.exists():
            print(f"CHECK FAILED: {args.output} does not exist",
                  file=sys.stderr)
            return 1
        if args.output.read_text(encoding="utf-8") != text:
            print(f"CHECK FAILED: {args.output} differs from regeneration",
                  file=sys.stderr)
            return 1
        print(f"CHECK OK: {args.output} matches regeneration")
        return 0

    write_registry(text, args.output, force=args.force)
    meta = registry["metadata"]
    print(f"  source fields: {meta['source_field_count']}")
    print(f"  included:      {meta['included_field_count']}")
    print(f"  uncertain:     {meta['uncertain_pending_review_count']} "
          f"(excluded pending review)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
