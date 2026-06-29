#!/usr/bin/env python
"""
validate_business_rules.py

Level 2 validation: cross-field, business logic, ND rules, calculated field tolerances.

Runs after validate_canonical.py on the _typed.csv

Usage:
    python validate_business_rules.py path/to/file_ESMA_Annex2_typed.csv --regime ESMA_Annex2
"""

import argparse
from pathlib import Path
import sys  # needed later in main()

import pandas as pd
import numpy as np

OUT_DIR = Path("out_validation")
OUT_DIR.mkdir(exist_ok=True)

DATE_RULE_COLUMNS = {
    "origination_date",
    "reporting_date",
    "maturity_date",
    "default_date",
    "current_valuation_date",
    "securitisation_date",
    "repossession_date",
}

NUMERIC_RULE_COLUMNS = {
    "current_principal_balance",
    "arrears_balance",
    "current_loan_to_value",
    "current_valuation_amount",
    "collateral_value",
    "number_of_days_in_principal_arrears",
    "days_past_due",
    "allocated_losses",
    "default_amount",
    "cumulative_recoveries",
    "current_interest_rate",
    "annual_turnover_of_obligor",
    "number_of_employees",
    "rent_payable",
    "recovery_amount",
    "residual_value",
    "original_asset_value",
    "borrower_1_age",
    "allocated_percentage",
}


def _resolve_default_regime(config: dict) -> str | None:
    """Resolve client default regime with legacy compatibility."""
    if not isinstance(config, dict):
        return None
    return config.get("default_regime") or config.get("regime")


def _backfill_ltv(df: pd.DataFrame,
                  bal_col: str,
                  val_col: str,
                  ltv_col: str) -> pd.DataFrame:
    """
    Fill LTV% where missing using: LTV = balance / valuation * 100.

    - No effect if balance/valuation columns are missing.
    - Creates the LTV column if it doesn't exist.
    - Only computes where valuation > 0 and both inputs are present.
    """
    if bal_col not in df.columns or val_col not in df.columns:
        return df

    # Ensure target LTV column exists
    if ltv_col not in df.columns:
        df[ltv_col] = np.nan

    mask = df[ltv_col].isna() & df[bal_col].notna() & df[val_col].notna()
    if not mask.any():
        return df

    bal = pd.to_numeric(df.loc[mask, bal_col], errors="coerce")
    val = pd.to_numeric(df.loc[mask, val_col], errors="coerce")

    valid = val > 0
    if not valid.any():
        return df

    idx = df.loc[mask].index[valid]
    df.loc[idx, ltv_col] = (bal[valid] / val[valid]) * 100.0
    return df

def _dat101_test(df: pd.DataFrame) -> pd.Series:
    required = ["account_status", "maturity_date", "reporting_date"]
    for c in required:
        if c not in df.columns:
            # If the engine calls us anyway, pass everything
            return pd.Series(True, index=df.index)

    status = df["account_status"].astype("string").str.upper()
    is_active = status.eq("ACTIVE")

    mat = pd.to_datetime(df["maturity_date"], errors="coerce")
    rep = pd.to_datetime(df["reporting_date"], errors="coerce")

    ok = (~is_active) | mat.isna() | rep.isna() | (mat >= rep)
    return ok.fillna(True).astype(bool)


def _coerce_types_for_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce known rule columns to stable dtypes before rule evaluation.
    Prevents Python type exceptions when mixed string/float values appear in CSV input.
    """
    out = df.copy()

    for col in DATE_RULE_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")

    for col in NUMERIC_RULE_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in ("account_status", "interest_rate_type", "reference_rate_index"):
        if col in out.columns:
            out[col] = out[col].astype("string").str.strip()

    return out


# ------------------------------------------------------------------
# Source-portfolio provenance helpers (PROV* rules)
# ------------------------------------------------------------------
# These run with required_columns=[] so they fire even when the provenance
# column is ABSENT (a tape with no source tag must fail, not be skipped).
# Each helper returns either a row-level Series (True = pass) or a scalar
# bool (portfolio-level pass/fail).

def _col_blank_mask(df: pd.DataFrame, col: str) -> pd.Series:
    s = df[col]
    return s.isna() | (s.astype(str).str.strip().isin(["", "nan", "None"]))


def _prov_id_present(df: pd.DataFrame):
    """source_portfolio_id must exist and be non-null on every row."""
    if "source_portfolio_id" not in df.columns:
        return False  # column absent → portfolio-level failure
    return ~_col_blank_mask(df, "source_portfolio_id")


def _prov_cohort_present(df: pd.DataFrame):
    if "portfolio_cohort" not in df.columns:
        return False
    return ~_col_blank_mask(df, "portfolio_cohort")


def _prov_type_valid(df: pd.DataFrame):
    if "source_portfolio_type" not in df.columns:
        return False
    vals = df["source_portfolio_type"].astype(str).str.strip().str.lower()
    return vals.isin(["direct", "acquired"])


def _prov_acq_required_for_acquired(df: pd.DataFrame):
    """Acquired rows must carry an acquisition_date (blank allowed only via
    the onboarding override, which leaves the field genuinely empty — that is a
    deliberate, surfaced choice rather than a silent gap, so we warn there)."""
    if "source_portfolio_type" not in df.columns:
        return True  # presence handled by PROV002
    is_acq = df["source_portfolio_type"].astype(str).str.strip().str.lower() == "acquired"
    if "acquisition_date" not in df.columns:
        return ~is_acq  # acquired rows fail when the column is entirely absent
    has_date = ~_col_blank_mask(df, "acquisition_date")
    return ~is_acq | has_date


def _prov_acq_blank_for_direct(df: pd.DataFrame):
    if "source_portfolio_type" not in df.columns or "acquisition_date" not in df.columns:
        return True
    is_direct = df["source_portfolio_type"].astype(str).str.strip().str.lower() == "direct"
    has_date = ~_col_blank_mask(df, "acquisition_date")
    return ~(is_direct & has_date)


# ==================================================================
# RULE DEFINITIONS – this will grow to 200–300 rules
# ==================================================================
# Format:
#   rule_id: str
#   regimes: list[str] or ["all"]
#   severity: "error" | "warning"
#   description: str
#   required_columns: list[str]  (engine skips rule if any are missing)
#   test: callable(df) -> pd.Series[bool] OR scalar bool  (True = pass, False = fail)
#   fail_message: optional callable(row, col) -> str
#       - for row-level rules: row is a pandas.Series for the failing row
#       - for portfolio rules: row is None


RULES = [
    
    # ==========================================================
    # DATE LOGIC
    # ==========================================================

    {
        "rule_id": "DAT001",
        "regimes": ["all"],
        "severity": "error",
        "description": "Origination date must be on or before reporting_date.",
        "required_columns": ["origination_date", "reporting_date"],
        "test": lambda df: df["origination_date"].isna()
                           | (df["origination_date"] <= df["reporting_date"]),
        "fail_message": lambda row, col: f"Origination date {row.get('origination_date')} after reporting_date.",
    },
    {
        "rule_id": "DAT002",
        "regimes": ["all"],
        "severity": "error",
        "description": "Origination date must be on or before maturity date.",
        "required_columns": ["origination_date", "maturity_date"],
        "test": lambda df: df["origination_date"].isna()
                           | df["maturity_date"].isna()
                           | (df["origination_date"] <= df["maturity_date"]),
        "fail_message": lambda row, col: f"Origination date {row.get('origination_date')} > maturity date {row.get('maturity_date')}.",
    },
    {
        "rule_id": "DAT003",
        "regimes": ["all"],
        "severity": "error",
        "description": "If default_date is populated, it must be ≥ origination_date.",
        "required_columns": ["default_date", "origination_date"],
        "test": lambda df: df["default_date"].isna()
                           | (df["default_date"] >= df["origination_date"]),
        "fail_message": lambda row, col: f"Default date {row.get('default_date')} before origination_date {row.get('origination_date')}.",
    },
    {
        "rule_id": "DAT004",
        "regimes": ["ESMA_Annex2"],
        "severity": "error",
        "description": "Current valuation date must be on or before securitisation_date (RRE).",
        "required_columns": ["current_valuation_date", "securitisation_date"],
        "test": lambda df: df["current_valuation_date"].isna()
                           | df["securitisation_date"].isna()
                           | (df["current_valuation_date"] <= df["securitisation_date"]),
        "fail_message": lambda row, col: f"Valuation date {row.get('current_valuation_date')} > securitisation_date {row.get('securitisation_date')}.",
    },
    {
        "rule_id": "DAT005",
        "regimes": ["all"],
        "severity": "warning",
        "description": "Origination date in canonical YYYY-MM-DD string format (no time component).",
        "required_columns": ["origination_date"],
        "test": lambda df: df["origination_date"].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$"),
        "fail_message": lambda row, col: f"Origination date not in YYYY-MM-DD format: {row.get('origination_date')}",
    },

    # Extra date logic (new, non-conflicting IDs)
    {
        "rule_id": "DAT101",
        "regimes": ["all"],
        "severity": "warning",
        "description": "For ACTIVE loans, maturity_date should be >= reporting_date.",
        "required_columns": ["maturity_date", "account_status", "reporting_date"],
        "test": _dat101_test,
        "fail_message": lambda row, col: (
            f"ACTIVE loan with maturity_date {row.get('maturity_date')} "
            f"< reporting_date {row.get('reporting_date')}."
        ),
    },
    
    # Securitisation date
    {
        "rule_id": "SEC001",
        "regimes": ["all"],
        "severity": "error",
        "description": "securitisation_date must be identical for all loans in the pool.",
        "required_columns": ["securitisation_date"],
        "test": lambda df: df["securitisation_date"].nunique() <= 1 if len(df) > 0 else True,
        "fail_message": lambda row, col: "Multiple securitisation_date values detected in tape.",
    },

    # ==========================================================
    # BALANCE & NUMERIC SANITY
    # ==========================================================

    {
        "rule_id": "BAL001",
        "regimes": ["all"],
        "severity": "error",
        "description": "Current principal balance must be ≥ 0.",
        "required_columns": ["current_principal_balance"],
        "test": lambda df: df["current_principal_balance"] >= 0,
        "fail_message": lambda row, col: f"Current principal balance {row.get('current_principal_balance')} < 0.",
    },
    {
        "rule_id": "BAL002",
        "regimes": ["all"],
        "severity": "warning",
        "description": "Arrears balance must be ≤ current principal balance.",
        "required_columns": ["arrears_balance", "current_principal_balance"],
        "test": lambda df: df["arrears_balance"] <= df["current_principal_balance"],
        "fail_message": lambda row, col: f"Arrears {row.get('arrears_balance')} > current principal {row.get('current_principal_balance')}.",
    },

    # new: current ≤ original
    {
        "rule_id": "BAL101",
        "regimes": [],  # no regimes -> rule never runs
        "severity": "error",
        "description": "DISABLED: ERM portfolios can have current principal > original.",
        "required_columns": ["current_principal_balance", "original_principal_balance"],
        "test": lambda df: pd.Series(True, index=df.index),
        "fail_message": lambda row, col: "",
    }
    ,
    {
        "rule_id": "BAL102",
        "regimes": ["all"],
        "severity": "error",
        "description": "If account_status is REPAID/REDEEMED/CLOSED, current balance must be 0.",
        "required_columns": ["current_principal_balance", "account_status"],
        "test": lambda df: ~df["account_status"].isin(["REPAID", "REDEEMED", "CLOSED", "MATURED"])
                           | (df["current_principal_balance"] == 0),
        "fail_message": lambda row, col: f"Status {row.get('account_status')} but current balance {row.get('current_principal_balance')} != 0.",
    },
    {
        "rule_id": "BAL103",
        "regimes": ["all"],
        "severity": "error",
        "description": "If current balance is 0, status should be REPAID/REDEEMED/CLOSED.",
        "required_columns": ["current_principal_balance", "account_status"],
        "test": lambda df: (df["current_principal_balance"] != 0)
                           | df["account_status"].isin(["REPAID", "REDEEMED", "CLOSED", "MATURED"]),
        "fail_message": lambda row, col: f"Zero balance with non-closed status {row.get('account_status')}.",
    },

    # ==========================================================
    # LTV & VALUATION
    # ==========================================================

    {
        "rule_id": "LTV001",
        "regimes": ["ESMA_Annex2", "ESMA_Annex3"],
        "severity": "error",
        "description": "Current LTV > 0 and ≤ 500%.",
        "required_columns": ["current_loan_to_value"],
        "test": lambda df: (df["current_loan_to_value"] > 0) & (df["current_loan_to_value"] <= 500),
        "fail_message": lambda row, col: f"LTV {row.get('current_loan_to_value')} out of bounds (0–500%).",
    },
    {
        "rule_id": "LTV002",
        "regimes": ["all"],
        "severity": "warning",
        "description": "Reported LTV ≈ (balance / valuation) * 100 (within 1% tolerance).",
        "required_columns": ["current_loan_to_value", "current_principal_balance", "current_valuation_amount"],
        "test": lambda df: pd.Series(
            np.isclose(
                df["current_principal_balance"] / df["current_valuation_amount"] * 100,
                df["current_loan_to_value"],
                atol=1.0,
                rtol=0.01,
                equal_nan=True,
             ),
            index=df.index,
        ),
        "fail_message": lambda row, col: "current_loan_to_value not consistent with balance and valuation (outside 1% tolerance).",
    },

    {
        "rule_id": "VAL001",
        "regimes": ["ESMA_Annex3"],
        "severity": "error",
        "description": "Collateral value must be ≥ 0 in CRE.",
        "required_columns": ["collateral_value"],
        "test": lambda df: df["collateral_value"] >= 0,
        "fail_message": lambda row, col: f"Collateral value {row.get('collateral_value')} < 0.",
    },
    # new: valuation vs balance
    {
        "rule_id": "VAL101",
        "regimes": ["ESMA_Annex2", "ESMA_Annex3"],
        "severity": "warning",
        "description": "Current property value should generally exceed current balance (within 10% tolerance).",
        "required_columns": ["current_principal_balance", "current_valuation_amount"],
        "test": lambda df: df["current_valuation_amount"] * 0.9 >= df["current_principal_balance"],
        "fail_message": lambda row, col: f"Current valuation {row.get('current_valuation_amount')} is too low vs balance {row.get('current_principal_balance')}.",
    },

    # ==========================================================
    # ARREARS & DEFAULT LOGIC
    # ==========================================================

    {
        "rule_id": "ARR001",
        "regimes": ["all"],
        "severity": "error",
        "description": "If arrears_balance > 0 then account_status must be ARRE/DEFAULT/RESTRUCTURED.",
        "required_columns": ["arrears_balance", "account_status"],
        "test": lambda df: (df["arrears_balance"] <= 0)
                           | df["account_status"].isin(["ARRE", "DEFAULT", "RESTRUCTURED"]),
        "fail_message": lambda row, col: f"Arrears {row.get('arrears_balance')} with status {row.get('account_status')} not in ARRE/DEFAULT/RESTRUCTURED.",
    },
    {
        "rule_id": "ARR002",
        "regimes": ["all"],
        "severity": "warning",
        "description": "Days in arrears > 0 implies arrears balance > 0.",
        "required_columns": ["number_of_days_in_principal_arrears", "arrears_balance"],
        "test": lambda df: (df["number_of_days_in_principal_arrears"] <= 0)
                           | (df["arrears_balance"] > 0),
        "fail_message": lambda row, col: f"DPD {row.get('number_of_days_in_principal_arrears')} > 0 but arrears_balance {row.get('arrears_balance')} <= 0.",
    },

    {
        "rule_id": "DEF001",
        "regimes": ["all"],
        "severity": "error",
        "description": "If default_date is populated, account_status must be DEFAULT.",
        "required_columns": ["default_date", "account_status"],
        "test": lambda df: df["default_date"].isna()
                           | (df["account_status"] == "DEFAULT"),
        "fail_message": lambda row, col: f"default_date {row.get('default_date')} but account_status {row.get('account_status')} != DEFAULT.",
    },
    {
        "rule_id": "DEF002",
        "regimes": ["all"],
        "severity": "warning",
        "description": "If allocated_losses > 0, account_status should be DEFAULT.",
        "required_columns": ["allocated_losses", "account_status"],
        "test": lambda df: (df["allocated_losses"] <= 0)
                           | (df["account_status"] == "DEFAULT"),
        "fail_message": lambda row, col: f"allocated_losses {row.get('allocated_losses')} > 0 but account_status {row.get('account_status')} != DEFAULT.",
    },

    # new: status vs DPD
    {
        "rule_id": "STAT101",
        "regimes": ["all"],
        "severity": "warning",
        "description": "If days_past_due >= 90, status should be ARRE/DEFAULT/RESTRUCTURED.",
        "required_columns": ["days_past_due", "account_status"],
        "test": lambda df: (df["days_past_due"] < 90)
                           | df["account_status"].isin(["ARRE", "DEFAULT", "RESTRUCTURED"]),
        "fail_message": lambda row, col: f"days_past_due {row.get('days_past_due')} >= 90 but status {row.get('account_status')} not in ARRE/DEFAULT/RESTRUCTURED.",
    },

    # ==========================================================
    # RECOVERIES / LOSSES
    # ==========================================================

    {
        "rule_id": "REC001",
        "regimes": ["all"],
        "severity": "error",
        "description": "Cumulative recoveries ≤ allocated_losses + default_amount.",
        "required_columns": ["cumulative_recoveries", "allocated_losses", "default_amount"],
        "test": lambda df: df["cumulative_recoveries"] <= (df["allocated_losses"] + df["default_amount"]),
        "fail_message": lambda row, col: f"Recoveries {row.get('cumulative_recoveries')} > allocated_losses + default_amount ({row.get('allocated_losses')} + {row.get('default_amount')}).",
    },

    # ==========================================================
    # IDENTIFIERS / LEI
    # ==========================================================

    {
        "rule_id": "ID001",
        "regimes": ["all"],
        "severity": "error",
        "description": "Loan identifiers must be unique within the tape.",
        "required_columns": ["unique_identifier"],
        "test": lambda df: ~df["unique_identifier"].duplicated(keep="first"),
        "fail_message": lambda row, col: f"Duplicate unique_identifier: {row.get('unique_identifier')}",
    },
    {
        "rule_id": "LEI001",
        "regimes": ["all"],
        "severity": "error",
        "description": "LEI scheme must contain http://standards.iso.org/iso/17442.",
        "required_columns": ["legal_entity_identifier"],
        "test": lambda df: df["legal_entity_identifier"].astype(str).str.contains("http://standards.iso.org/iso/17442", na=False),
        "fail_message": lambda row, col: f"Invalid LEI scheme: {row.get('legal_entity_identifier')}",
    },

    # ==========================================================
    # INTEREST / RATE LOGIC (NEW)
    # ==========================================================

    {
        "rule_id": "INT001",
        "regimes": ["all"],
        "severity": "error",
        "description": "Current interest rate >= 0 (or -1 for legacy special flag).",
        "required_columns": ["current_interest_rate"],
        "test": lambda df: df["current_interest_rate"].isin([-1]) | (df["current_interest_rate"] >= 0),
        "fail_message": lambda row, col: f"Invalid current_interest_rate {row.get('current_interest_rate')} (must be >=0 or -1).",
    },
    {
        "rule_id": "INT101",
        "regimes": ["all"],
        "severity": "warning",
        "description": "If interest_rate_type is FIXED, reference_rate_index should be empty.",
        "required_columns": ["interest_rate_type", "reference_rate_index"],
        "test": lambda df: (df["interest_rate_type"] != "FIXED") | df["reference_rate_index"].isna(),
        "fail_message": lambda row, col: f"FIXED rate loan but reference_rate_index present: {row.get('reference_rate_index')}.",
    },

    # ==========================================================
    # SME / EQUIPMENT / ERM – EXAMPLE NEW RULES
    # (These will auto-skip if required fields are missing)
    # ==========================================================

    {
        "rule_id": "SME101",
        "regimes": ["all"],
        "severity": "error",
        "description": "Annual turnover of obligor must be >= 0.",
        "required_columns": ["annual_turnover_of_obligor"],
        "test": lambda df: df["annual_turnover_of_obligor"] >= 0,
        "fail_message": lambda row, col: f"Annual turnover {row.get('annual_turnover_of_obligor')} < 0.",
    },
    {
        "rule_id": "SME102",
        "regimes": ["all"],
        "severity": "error",
        "description": "Number of employees must be >= 0.",
        "required_columns": ["number_of_employees"],
        "test": lambda df: df["number_of_employees"] >= 0,
        "fail_message": lambda row, col: f"Number of employees {row.get('number_of_employees')} < 0.",
    },
    {
        "rule_id": "SME103",
        "regimes": ["all"],
        "severity": "warning",
        "description": "If enterprise_size == 'micro', turnover should not exceed 2m (example threshold).",
        "required_columns": ["enterprise_size", "annual_turnover_of_obligor"],
        "test": lambda df: (df["enterprise_size"] != "micro") | (df["annual_turnover_of_obligor"] <= 2_000_000),
        "fail_message": lambda row, col: f"enterprise_size 'micro' but turnover {row.get('annual_turnover_of_obligor')} > 2m.",
    },

    {
        "rule_id": "EQP101",
        "regimes": ["all"],
        "severity": "error",
        "description": "If lease_status is REPOSSESSED, repossession_date must be populated.",
        "required_columns": ["lease_status", "repossession_date"],
        "test": lambda df: (df["lease_status"] != "REPOSSESSED") | df["repossession_date"].notna(),
        "fail_message": lambda row, col: "lease_status REPOSSESSED but repossession_date is missing.",
    },
    {
        "rule_id": "EQP102",
        "regimes": ["all"],
        "severity": "warning",
        "description": "If lease_status is REPOSSESSED, recovery_amount should be >= 0.",
        "required_columns": ["lease_status", "recovery_amount"],
        "test": lambda df: (df["lease_status"] != "REPOSSESSED") | (df["recovery_amount"] >= 0),
        "fail_message": lambda row, col: f"REPOSSESSED lease with negative recovery_amount {row.get('recovery_amount')}.",
    },
    {
        "rule_id": "EQP103",
        "regimes": ["all"],
        "severity": "warning",
        "description": "Residual value should not exceed original asset value.",
        "required_columns": ["residual_value", "original_asset_value"],
        "test": lambda df: df["residual_value"] <= df["original_asset_value"],
        "fail_message": lambda row, col: f"Residual value {row.get('residual_value')} > original_asset_value {row.get('original_asset_value')}.",
    },

    # ERM examples
    {
        "rule_id": "ERM101",
        "regimes": ["ESMA_Annex2"],
        "severity": "warning",
        "description": "If negative_equity_guarantee = Y then ltv_cap should be populated.",
        "required_columns": ["negative_equity_guarantee", "ltv_cap"],
        "test": lambda df: df["negative_equity_guarantee"].ne("Y") | df["ltv_cap"].notna(),
        "fail_message": lambda row, col: "negative_equity_guarantee = Y but ltv_cap is missing.",
    },
    {
        "rule_id": "ERM102",
        "regimes": ["ESMA_Annex2"],
        "severity": "warning",
        "description": "Borrower 1 age for ERM should be between 50 and 110.",
        "required_columns": ["borrower_1_age"],
        "test": lambda df: df["borrower_1_age"].between(50, 110) | df["borrower_1_age"].isna(),
        "fail_message": lambda row, col: f"ERM borrower_1_age {row.get('borrower_1_age')} outside [50,110].",
    },

    # ==========================================================
    # PORTFOLIO-LEVEL CHECKS
    # ==========================================================

    {
        "rule_id": "POR001",
        "regimes": ["all"],
        "severity": "warning",
        "description": "Sum of allocated % of underlying exposure at securitisation date ≈ 100% (±0.01).",
        "required_columns": ["allocated_percentage_of_underlying_exposure_at_securitisation_date"],
        "test": lambda df: np.isclose(
            df["allocated_percentage_of_underlying_exposure_at_securitisation_date"].sum(),
            100.0,
            atol=0.05,
        ) if len(df) > 0 else True,
        "fail_message": lambda row, col: "Allocated percentage sum not ≈ 100%.",
    },
    {
        "rule_id": "POR002",
        "regimes": ["ESMA_Annex3"],
        "severity": "error",
        "description": "For occupied CRE tenants, rent_payable must be > 0.",
        "required_columns": ["occupancy_status", "rent_payable"],
        "test": lambda df: (df["occupancy_status"] != "OCCUPIED") | (df["rent_payable"] > 0),
        "fail_message": lambda row, col: f"Occupied tenant with non-positive rent_payable {row.get('rent_payable')}.",
    },
    {
        "rule_id": "PORT101",
        "regimes": ["all"],
        "severity": "error",
        "description": "Loan identifiers must be unique within the tape (portfolio-level sanity).",
        "required_columns": ["loan_identifier"],
        "test": lambda df: df["loan_identifier"].nunique() == len(df),
        "fail_message": lambda row, col: "Duplicate loan_identifier values present in tape.",
    },

    {
        "rule_id": "CUR001",
        "regimes": ["all"],
        "severity": "warning",
        "description": "All monetary fields should be in the same currency (single currency tape).",
        "required_columns": ["currency"],
        "test": lambda df: df["currency"].nunique() <= 1,
        "fail_message": lambda row, col: "Mixed currencies detected in 'currency' column.",
    },

    # Meta / completeness diagnostics
    {
        "rule_id": "META101",
        "regimes": ["all"],
        "severity": "warning",
        "description": "Proportion of missing origination_date should be ≤1%.",
        "required_columns": ["origination_date"],
        "test": lambda df: df["origination_date"].isna().mean() <= 0.01 if len(df) > 0 else True,
        "fail_message": lambda row, col: "Missing origination_date exceeds 1% of portfolio.",
    },
    {
        "rule_id": "META102",
        "regimes": ["all"],
        "severity": "warning",
        "description": "Proportion of missing current_principal_balance should be ≤1%.",
        "required_columns": ["current_principal_balance"],
        "test": lambda df: df["current_principal_balance"].isna().mean() <= 0.01 if len(df) > 0 else True,
        "fail_message": lambda row, col: "Missing current_principal_balance exceeds 1% of portfolio.",
    },

    # ==========================================================
    # SOURCE-PORTFOLIO PROVENANCE (run-level onboarding metadata)
    # required_columns=[] so a tape with NO provenance fails closed.
    # ==========================================================
    {
        "rule_id": "PROV001",
        "regimes": ["all"],
        "severity": "error",
        "description": "source_portfolio_id must be present and non-null on every loan "
                       "(direct_001 / acquired_001 / …). Provenance must be stamped at "
                       "onboarding; Trakt never assigns 'unknown'.",
        "required_columns": [],
        "test": _prov_id_present,
        "fail_message": lambda row, col: (
            "Missing source_portfolio_id — loan has no source-portfolio provenance. "
            "Re-run onboarding with --source-portfolio-id."
        ),
    },
    {
        "rule_id": "PROV002",
        "regimes": ["all"],
        "severity": "error",
        "description": "source_portfolio_type must be present and one of {direct, acquired}.",
        "required_columns": [],
        "test": _prov_type_valid,
        "fail_message": lambda row, col: (
            f"Invalid/missing source_portfolio_type: {None if row is None else row.get('source_portfolio_type')}."
        ),
    },
    {
        "rule_id": "PROV003",
        "regimes": ["all"],
        "severity": "warning",
        "description": "Acquired portfolios should carry an acquisition_date "
                       "(blank only when onboarded with --allow-unknown-acquisition-date).",
        "required_columns": [],
        "test": _prov_acq_required_for_acquired,
        "fail_message": lambda row, col: (
            f"Acquired loan {None if row is None else row.get('source_portfolio_id')} "
            "has no acquisition_date."
        ),
    },
    {
        "rule_id": "PROV004",
        "regimes": ["all"],
        "severity": "warning",
        "description": "Direct/originated books should not carry an acquisition_date.",
        "required_columns": [],
        "test": _prov_acq_blank_for_direct,
        "fail_message": lambda row, col: (
            f"Direct loan {None if row is None else row.get('source_portfolio_id')} "
            f"has an acquisition_date {None if row is None else row.get('acquisition_date')}."
        ),
    },
    {
        "rule_id": "PROV005",
        "regimes": ["all"],
        "severity": "error",
        "description": "portfolio_cohort must be present and non-null (defaults to source_portfolio_id).",
        "required_columns": [],
        "test": _prov_cohort_present,
        "fail_message": lambda row, col: "Missing portfolio_cohort — provenance not carried through transformation.",
    },
]
def run_rules(df: pd.DataFrame, regime: str) -> pd.DataFrame:
    """
    Apply all business rules relevant for the given regime to df.

    Returns a DataFrame of violations with columns:
        rule_id, severity, description, message, row_index
    """
    violations = []

    # Filter rules by regime
    active_rules = [
        r for r in RULES
        if regime in r.get("regimes", []) or "all" in r.get("regimes", [])
    ]

    for rule in active_rules:
        rule_id = rule["rule_id"]
        severity = rule["severity"]
        description = rule["description"]
        required_cols = rule.get("required_columns", [])

        # Skip if required columns are missing
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            # Optional: debug print
            # print(f"Skipping {rule_id}: missing columns {missing}")
            continue

        test_fn = rule["test"]
        msg_fn = rule.get("fail_message", None)

        try:
            result = test_fn(df)
        except Exception as e:
            # If the rule itself crashes, record a meta-violation and move on
            violations.append({
                "rule_id": rule_id,
                "severity": "error",
                "description": f"Rule execution failed: {description}",
                "message": f"Exception: {e}",
                "row_index": -1,
            })
            continue

        # Row-level rule: result is a Series[bool]
        if isinstance(result, pd.Series):
            fail_mask = ~result  # False = fail
            if fail_mask.any():
                for idx in df.index[fail_mask]:
                    row = df.loc[idx]
                    if msg_fn is not None:
                        msg = msg_fn(row, None)
                    else:
                        msg = description
                    violations.append({
                        "rule_id": rule_id,
                        "severity": severity,
                        "description": description,
                        "message": msg,
                        "row_index": int(idx),
                    })

        # Portfolio-level rule: result is a single bool
        else:
            if not bool(result):
                if msg_fn is not None:
                    msg = msg_fn(None, None)
                else:
                    msg = description
                violations.append({
                    "rule_id": rule_id,
                    "severity": severity,
                    "description": description,
                    "message": msg,
                    "row_index": -1,
                })

    if violations:
        return pd.DataFrame(violations)
    else:
        return pd.DataFrame(columns=["rule_id", "severity", "description", "message", "row_index"])


def main():
    parser = argparse.ArgumentParser(description="Level 2 ESMA business rule validation.")
    parser.add_argument("--config", default=None, help="Client config YAML")
    parser.add_argument("typed_canonical_csv", nargs="?", help="Path to *_typed.csv produced by canonical_transform.py")
    parser.add_argument("--input", help="Path to typed canonical CSV (alternative to positional)")
    parser.add_argument("--report", help="Optional path for business rule violations CSV")
    parser.add_argument("--regime", default=None, help="Regime name, e.g. ESMA_Annex2, ESMA_Annex3, ESMA_Annex4")
    args = parser.parse_args()

    import sys
    import yaml

    config = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    if not args.regime:
        args.regime = _resolve_default_regime(config)

    if not args.regime:
        print("ERROR: Missing regime. Provide --regime or set 'regime' in the config YAML.", file=sys.stderr)
        sys.exit(1)

    input_arg = args.input or args.typed_canonical_csv
    if not input_arg:
        print("ERROR: You must provide a typed canonical CSV (positional or --input).", file=sys.stderr)
        sys.exit(1)

    in_path = Path(input_arg)
    regime = args.regime

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    # Read typed canonical – let pandas infer dtypes (we rely on canonical_transform for most typing)
    df = pd.read_csv(in_path, low_memory=False)
    
    df = _backfill_ltv(df, "current_principal_balance", 
                   "current_valuation_amount", "current_loan_to_value")
    
    # --- Column normalisation for alignment with frozen canonical ---
    # Business rules were originally authored against legacy column names in some places.
    # To avoid brittle skip behaviour, we alias the frozen canonical names to the legacy names where needed.
    if 'reporting_date' not in df.columns and 'data_cut_off_date' in df.columns:
        df['reporting_date'] = df['data_cut_off_date']
    if 'currency' not in df.columns:
        if 'exposure_currency_denomination' in df.columns:
            df['currency'] = df['exposure_currency_denomination']
        elif 'loan_denomination_currency' in df.columns:
            df['currency'] = df['loan_denomination_currency']


    typed_df = _coerce_types_for_rules(df)
    violations_df = run_rules(typed_df, regime)

    # Build output path
    # e.g. myfile_ESMA_Annex2_typed.csv -> myfile_ESMA_Annex2_business_rules_violations.csv
    stem = in_path.stem
    if stem.endswith("_typed"):
        stem = stem[:-6]

    if args.report:
        out_path = Path(args.report)
        out_path.parent.mkdir(exist_ok=True)
    else:
        OUT_DIR.mkdir(exist_ok=True)
        out_path = OUT_DIR / f"{stem}_business_rules_violations.csv"

    if not violations_df.empty:
        violations_df.to_csv(out_path, index=False)
        print(f"✗ {len(violations_df)} business rule violations → {out_path}")
        print("Top failing rules:")
        print(violations_df["rule_id"].value_counts().head(10))
    else:
        violations_df.to_csv(out_path, index=False)
        print("✓ All business rules passed!")

if __name__ == "__main__":
    main()
