"""
scenario_engine.py

Production-ready ERM portfolio scenario engine with comprehensive error handling.

Projects loan balances and property values forward in time under a set of
assumptions, computing:

- Expected portfolio balance runoff
- Expected indexed property value growth
- Percentage of original balance remaining
- Expected NNEG (No Negative Equity Guarantee) losses per year

Uses simplified hazard-rate exit model for computational efficiency.
Appropriate for strategic planning and risk committee analysis.

Version: 2.0 (Production)
Enhanced: 2025-12-10
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Tuple, Union
import warnings

import numpy as np
import pandas as pd


@dataclass
class ScenarioAssumptions:
    """
    User-controlled scenario assumptions (all rates annualized).
    
    Attributes
    ----------
    hpi_rate : float
        House price inflation rate (e.g., 0.02 = +2% annual growth)
    interest_rate_spread : float
        Spread added to each loan's current rate (e.g., 0.01 = +1% on all loans)
    voluntary_prepay_rate : float
        Annual conditional probability of voluntary prepayment
    mortality_rate : float
        Annual conditional probability of borrower death
    move_to_care_rate : float
        Annual conditional probability of move to long-term care
    sale_cost_pct : float
        Property sale costs as % of value (e.g., 0.05 = 5%)
    n_years : int
        Projection horizon in years
    """
    hpi_rate: float = 0.02
    interest_rate_spread: float = 0.0
    voluntary_prepay_rate: float = 0.02
    mortality_rate: float = 0.03
    move_to_care_rate: float = 0.01
    sale_cost_pct: float = 0.05
    n_years: int = 25

    def __post_init__(self):
        """Validate assumptions on creation."""
        if self.n_years < 1:
            raise ValueError(f"n_years must be >= 1, got {self.n_years}")
        if self.n_years > 100:
            raise ValueError(f"n_years must be <= 100, got {self.n_years}")
        
        # Warn on extreme values
        if abs(self.hpi_rate) > 0.5:
            warnings.warn(f"HPI rate {self.hpi_rate:.1%} is extreme (>50% annually)")
        if self.voluntary_prepay_rate > 0.5:
            warnings.warn(f"Prepayment rate {self.voluntary_prepay_rate:.1%} is very high")
        if self.mortality_rate > 0.5:
            warnings.warn(f"Mortality rate {self.mortality_rate:.1%} is very high")

    def to_dict(self) -> Dict[str, float]:
        """Export as dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'ScenarioAssumptions':
        """Create from dictionary."""
        return cls(**d)


# =============================================================================
# PRESET SCENARIOS
# =============================================================================

PRESET_SCENARIOS = {
    "Base Case": ScenarioAssumptions(
        hpi_rate=0.02,
        interest_rate_spread=0.0,
        voluntary_prepay_rate=0.02,
        mortality_rate=0.03,
        move_to_care_rate=0.01,
        n_years=25
    ),
    "House Price Stress": ScenarioAssumptions(
        hpi_rate=-0.05,
        interest_rate_spread=0.0,
        voluntary_prepay_rate=0.04,  # Higher prepay in stress
        mortality_rate=0.03,
        move_to_care_rate=0.01,
        n_years=25
    ),
    "Severe House Price Stress": ScenarioAssumptions(
        hpi_rate=-0.10,
        interest_rate_spread=0.01,
        voluntary_prepay_rate=0.05,
        mortality_rate=0.04,
        move_to_care_rate=0.02,
        n_years=25
    ),
    "High Mortality": ScenarioAssumptions(
        hpi_rate=0.02,
        interest_rate_spread=0.0,
        voluntary_prepay_rate=0.02,
        mortality_rate=0.06,  # Double base rate
        move_to_care_rate=0.02,
        n_years=25
    ),
    "Rising Interest Rates": ScenarioAssumptions(
        hpi_rate=0.01,
        interest_rate_spread=0.02,  # +2% on all loans
        voluntary_prepay_rate=0.01,  # Lower prepay when rates high
        mortality_rate=0.03,
        move_to_care_rate=0.01,
        n_years=25
    ),
    "Benign Conditions": ScenarioAssumptions(
        hpi_rate=0.05,
        interest_rate_spread=0.0,
        voluntary_prepay_rate=0.03,
        mortality_rate=0.02,
        move_to_care_rate=0.005,
        n_years=25
    ),
}


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def _validate_and_extract_column(
    loans: pd.DataFrame,
    col_name: str,
    required: bool = True,
    default_value: float = 0.0
) -> np.ndarray:
    """
    Validate and extract numeric column from DataFrame.
    
    Parameters
    ----------
    loans : DataFrame
        Input loan data
    col_name : str
        Column name to extract
    required : bool
        If True, raise error if column missing. If False, return default.
    default_value : float
        Value to use if column missing and not required
        
    Returns
    -------
    np.ndarray
        Validated numeric array
        
    Raises
    ------
    ValueError
        If column missing (when required=True) or contains non-numeric data
    """
    if col_name not in loans.columns:
        if required:
            raise ValueError(
                f"Required column '{col_name}' not found. "
                f"Available columns: {list(loans.columns)}"
            )
        else:
            return np.full(len(loans), default_value, dtype=float)
    
    # Convert to numeric, coerce errors to NaN
    values = pd.to_numeric(loans[col_name], errors='coerce')
    
    # Check for NaN values
    n_missing = values.isna().sum()
    if n_missing > 0:
        if required:
            raise ValueError(
                f"Column '{col_name}' contains {n_missing} non-numeric or missing values. "
                f"Clean data before running scenario."
            )
        else:
            warnings.warn(
                f"Column '{col_name}' has {n_missing} missing values, filling with {default_value}"
            )
            values = values.fillna(default_value)
    
    return values.to_numpy(dtype=float)


def _normalize_interest_rates(rates: np.ndarray) -> np.ndarray:
    """
    Normalize interest rates to decimal form.
    
    If median rate > 1, assumes rates are in percentage form (e.g., 5.0 = 5%)
    and divides by 100.
    
    Parameters
    ----------
    rates : np.ndarray
        Raw interest rates
        
    Returns
    -------
    np.ndarray
        Normalized rates (0.05 = 5%)
    """
    median_rate = np.median(rates[rates > 0]) if (rates > 0).any() else 0
    
    if median_rate > 1:
        warnings.warn(
            f"Interest rates appear to be in percentage form (median={median_rate:.2f}). "
            f"Dividing by 100 to normalize."
        )
        return rates / 100.0
    
    return rates


# =============================================================================
# MAIN PROJECTION ENGINE
# =============================================================================

def project_portfolio(
    loans: pd.DataFrame,
    assumptions: ScenarioAssumptions,
    balance_col: str = "current_principal_balance",
    accrued_interest_col: Optional[str] = "accrued_interest",
    valuation_col: str = "current_valuation_amount",
    rate_col: str = "current_interest_rate",
    return_loan_level: bool = False,
) -> pd.DataFrame:
    """
    Project portfolio balances and property values using expected-value model
    with proper exit and removal of loans and properties.

    Exits (voluntary prepay, mortality, move to care) are treated as:
      - Full loan repayment (balance removed from portfolio)
      - Property sale (property removed from portfolio)
      - NNEG loss realised if balance > net sale proceeds

    Parameters
    ----------
    loans : DataFrame
        Current snapshot of loan tape. Required columns:
        - balance_col (principal outstanding)
        - valuation_col (current property value)
        - rate_col (current loan interest rate, annual)
        Optional columns:
        - accrued_interest_col (current accrued interest)
        - original_principal_balance (for remaining % calculation)
    assumptions : ScenarioAssumptions
        Scenario inputs (HPI, mortality, prepayment rates, etc.)
    balance_col : str
        Column name for principal balance
    accrued_interest_col : str or None
        Column name for accrued interest (optional, defaults to 0 if None)
    valuation_col : str
        Column name for property valuation
    rate_col : str
        Column name for interest rate
    return_loan_level : bool
        If True, return tuple (portfolio_df, loan_detail_df)
        If False, return only portfolio_df

    Returns
    -------
    DataFrame (or tuple of DataFrames if return_loan_level=True)
        Portfolio-level projections with columns:
        - year : int (0 to n_years)
        - portfolio_balance : float (expected outstanding, survivors only)
        - portfolio_property_value : float (expected property value of survivors)
        - remaining_balance_ratio : float (as % of original principal)
        - expected_nneg_loss : float (per year, expected)
        - cumulative_expected_nneg : float
        - underwater_exposure : float (expected, survivors only)
        - portfolio_ltv : float (balance / value * 100)
    """
    # ---- 1. Validate inputs ----
    if loans.empty:
        raise ValueError("Input loans DataFrame is empty")

    if not isinstance(assumptions, ScenarioAssumptions):
        raise TypeError(
            f"assumptions must be ScenarioAssumptions instance, got {type(assumptions)}"
        )

    n_loans = len(loans)
    a = assumptions  # shorthand

    # ---- 2. Extract and validate columns ----

    # Principal balance (required)
    principal = _validate_and_extract_column(loans, balance_col, required=True)

    # Accrued interest (optional, defaults to 0)
    if accrued_interest_col and accrued_interest_col in loans.columns:
        accrued = _validate_and_extract_column(
            loans, accrued_interest_col, required=False
        )
    else:
        accrued = np.zeros(n_loans, dtype=float)

    balance0 = principal + accrued

    # Property valuation (required)
    prop_val0 = _validate_and_extract_column(loans, valuation_col, required=True)

    # Sanity check: property values should be positive
    if (prop_val0 <= 0).any():
        n_invalid = (prop_val0 <= 0).sum()
        raise ValueError(
            f"{n_invalid} loans have non-positive property values. "
            f"Check '{valuation_col}' column."
        )

    # Interest rates (required, with normalization)
    base_rates = _validate_and_extract_column(loans, rate_col, required=True)
    base_rates = _normalize_interest_rates(base_rates)

    # Check for negative rates
    if (base_rates < 0).any():
        n_negative = (base_rates < 0).sum()
        warnings.warn(f"{n_negative} loans have negative interest rates, setting to 0")
        base_rates = np.maximum(base_rates, 0)

    eff_rates = base_rates + a.interest_rate_spread

    # Original principal (optional, used for remaining % calculation)
    if "original_principal_balance" in loans.columns:
        orig_principal = _validate_and_extract_column(
            loans, "original_principal_balance", required=False
        )
        # Fall back to current if original is missing/zero
        orig_principal = np.where(orig_principal > 0, orig_principal, principal)
    else:
        orig_principal = principal.copy()

    total_orig_principal = float(orig_principal.sum())
    if total_orig_principal <= 0:
        raise ValueError(
            "Total original principal balance is zero or negative. "
            "Cannot compute remaining balance ratio."
        )

    # ---- 3. Compute loan-level exit probabilities (age-based mortality) ----
    # Treat prepay, mortality, and move-to-care as independent hazards,
    # but let mortality vary by borrower age.

    if "youngest_borrower_age" in loans.columns:
        ages = pd.to_numeric(loans["youngest_borrower_age"], errors="coerce")
        # Fill missing ages with median or a sensible default (e.g. 75)
        if ages.isna().all():
            ages = pd.Series(75.0, index=loans.index)
        else:
            ages = ages.fillna(ages.median())
    else:
        # Fallback: assume flat age of 75 if we have no age column
        ages = pd.Series(75.0, index=loans.index)

    # Age factor around an 80-year "reference" age
    #  - at 80: factor = 1.0 (base mortality)
    #  - at 60: factor = 0.75  (lower mortality)
    #  - at 90: factor = 1.125 (higher mortality)
    age_factor = ages / 80.0
    age_factor = age_factor.clip(lower=0.5, upper=1.5)

    # Loan-level mortality rates
    mortality_i = a.mortality_rate * age_factor.to_numpy(dtype=float)
    mortality_i = np.clip(mortality_i, 0.0, 0.5)  # cap at 50% for safety

    # Combine hazards per loan:
    # P_stay_i = (1 - prepay) * (1 - mortality_i) * (1 - move_to_care)
    # P_exit_i = 1 - P_stay_i
    v = a.voluntary_prepay_rate
    c = a.move_to_care_rate

    if not (0.0 <= v <= 1.0 and 0.0 <= a.mortality_rate <= 1.0 and 0.0 <= c <= 1.0):
        raise ValueError(
            "Hazard rates must be between 0 and 1. "
            f"Got prepay={v}, mortality={a.mortality_rate}, care={c}"
        )

    p_stay = (1.0 - v) * (1.0 - mortality_i) * (1.0 - c)
    p_exit = 1.0 - p_stay

    # Sanity check: all probabilities in [0,1]
    if (p_exit < 0).any() or (p_exit > 1).any():
        raise ValueError(
            "Computed loan-level exit probabilities outside [0, 1]. "
            "Check hazard assumptions and age scaling."
        )

    # ---- 4. Initialize projection state arrays ----
    n_years = a.n_years

    # Conditional on being alive (per loan)
    balances_cond = np.zeros((n_years + 1, n_loans), dtype=float)
    prop_vals = np.zeros((n_years + 1, n_loans), dtype=float)

    # Survival probability per loan
    survival = np.zeros((n_years + 1, n_loans), dtype=float)

    # Year 0 state
    balances_cond[0, :] = balance0
    prop_vals[0, :] = prop_val0
    survival[0, :] = 1.0

    # Portfolio-level metrics per year
    expected_nneg_loss = np.zeros(n_years + 1, dtype=float)
    underwater_exposure = np.zeros(n_years + 1, dtype=float)

    # ---- 5. Project forward with proper exits ----
    for t in range(1, n_years + 1):
        # Grow property values by HPI (unconditional trajectory)
        prop_vals[t, :] = prop_vals[t - 1, :] * (1.0 + a.hpi_rate)

        # Grow balances (conditional on being still in force before exit)
        grown_balance_cond = balances_cond[t - 1, :] * (1.0 + eff_rates)

        # Survival and exit probabilities this year
        surv_prev = survival[t - 1, :]
        exit_flow = surv_prev * p_exit       # fraction exiting this year (per original loan)
        surv_curr = surv_prev * p_stay       # fraction surviving to next year

        # NNEG loss at exit (on exiting loans only)
        net_proceeds = prop_vals[t, :] * (1.0 - a.sale_cost_pct)
        nneg_per_loan = np.maximum(grown_balance_cond - net_proceeds, 0.0)

        # Expected NNEG loss this year: per-loan loss × exit probability
        expected_nneg_loss[t] = float(np.sum(nneg_per_loan * exit_flow))

        # Underwater exposure for survivors only (loans still on the book at year-end)
        underwater_per_loan = np.maximum(grown_balance_cond - prop_vals[t, :], 0.0)
        underwater_exposure[t] = float(np.sum(underwater_per_loan * surv_curr))

        # Update state for next year
        balances_cond[t, :] = grown_balance_cond
        survival[t, :] = surv_curr

    # ---- 6. Aggregate to portfolio level ----
    years = np.arange(n_years + 1)

    # Expected portfolio balance and property at each year-end
    portfolio_balance = np.sum(balances_cond * survival, axis=1)
    portfolio_property = np.sum(prop_vals * survival, axis=1)

    # Remaining balance as % of original principal
    remaining_balance_ratio = portfolio_balance / total_orig_principal

    # Cumulative NNEG losses
    cumulative_expected_nneg = np.cumsum(expected_nneg_loss)

    # Portfolio LTV (expected balance / expected property)
    portfolio_ltv = np.where(
        portfolio_property > 0,
        portfolio_balance / portfolio_property * 100.0,
        0.0,
    )

    result = pd.DataFrame(
        {
            "year": years,
            "portfolio_balance": portfolio_balance,
            "portfolio_property_value": portfolio_property,
            "remaining_balance_ratio": remaining_balance_ratio,
            "expected_nneg_loss": expected_nneg_loss,
            "cumulative_expected_nneg": cumulative_expected_nneg,
            "underwater_exposure": underwater_exposure,
            "portfolio_ltv": portfolio_ltv,
        }
    )

    # ---- 7. Optional: loan-level detail (expected exposures) ----
    if return_loan_level:
        # Expected exposures = conditional state × survival probability
        exp_balance_yr0 = balances_cond[0, :] * survival[0, :]
        exp_prop_yr0 = prop_vals[0, :] * survival[0, :]

        idx_yr5 = min(5, n_years)
        idx_yr10 = min(10, n_years)
        idx_final = n_years

        exp_balance_yr5 = balances_cond[idx_yr5, :] * survival[idx_yr5, :]
        exp_prop_yr5 = prop_vals[idx_yr5, :] * survival[idx_yr5, :]

        exp_balance_yr10 = balances_cond[idx_yr10, :] * survival[idx_yr10, :]
        exp_prop_yr10 = prop_vals[idx_yr10, :] * survival[idx_yr10, :]

        exp_balance_final = balances_cond[idx_final, :] * survival[idx_final, :]
        exp_prop_final = prop_vals[idx_final, :] * survival[idx_final, :]

        # Expected LTVs at loan level
        ltv_yr0 = np.where(
            exp_prop_yr0 > 0,
            exp_balance_yr0 / exp_prop_yr0 * 100.0,
            0.0,
        )
        ltv_final = np.where(
            exp_prop_final > 0,
            exp_balance_final / exp_prop_final * 100.0,
            0.0,
        )

        loan_detail = pd.DataFrame(
            {
                "loan_index": np.arange(n_loans),
                "balance_yr0": exp_balance_yr0,
                "property_yr0": exp_prop_yr0,
                "balance_yr5": exp_balance_yr5,
                "property_yr5": exp_prop_yr5,
                "balance_yr10": exp_balance_yr10,
                "property_yr10": exp_prop_yr10,
                "balance_final": exp_balance_final,
                "property_final": exp_prop_final,
                "ltv_yr0": ltv_yr0,
                "ltv_final": ltv_final,
            }
        )

        return result, loan_detail

    return result

# =============================================================================
# SCENARIO COMPARISON
# =============================================================================

def compare_scenarios(
    loans: pd.DataFrame,
    scenarios: Dict[str, ScenarioAssumptions],
    **kwargs
) -> pd.DataFrame:
    """
    Run multiple scenarios and return combined results for comparison.
    
    Parameters
    ----------
    loans : DataFrame
        Current loan tape
    scenarios : dict
        Dictionary of {scenario_name: ScenarioAssumptions}
    **kwargs
        Additional arguments passed to project_portfolio()
        
    Returns
    -------
    DataFrame
        Combined results with columns:
        - year
        - scenario_name
        - portfolio_balance
        - portfolio_property_value
        - expected_nneg_loss
        - cumulative_expected_nneg
        - portfolio_ltv
    """
    results = []
    
    for name, assumptions in scenarios.items():
        projection = project_portfolio(loans, assumptions, **kwargs)
        projection['scenario_name'] = name
        results.append(projection)
    
    combined = pd.concat(results, ignore_index=True)
    
    return combined


# =============================================================================
# SENSITIVITY ANALYSIS
# =============================================================================

def sensitivity_analysis(
    loans: pd.DataFrame,
    base_assumptions: ScenarioAssumptions,
    parameter: str,
    values: list,
    **kwargs
) -> pd.DataFrame:
    """
    Run sensitivity analysis on a single parameter.
    
    Parameters
    ----------
    loans : DataFrame
        Current loan tape
    base_assumptions : ScenarioAssumptions
        Base case assumptions
    parameter : str
        Parameter to vary (e.g., 'hpi_rate', 'mortality_rate')
    values : list
        List of values to test for the parameter
    **kwargs
        Additional arguments passed to project_portfolio()
        
    Returns
    -------
    DataFrame
        Results with scenario_name showing parameter values
    """
    scenarios = {}
    
    for value in values:
        # Create copy of base assumptions
        assumptions_dict = base_assumptions.to_dict()
        assumptions_dict[parameter] = value
        
        name = f"{parameter}={value:+.1%}" if isinstance(value, float) else f"{parameter}={value}"
        scenarios[name] = ScenarioAssumptions.from_dict(assumptions_dict)
    
    return compare_scenarios(loans, scenarios, **kwargs)


# =============================================================================
# EXPORT / REPORTING
# =============================================================================

def export_scenario_summary(
    projection: pd.DataFrame,
    assumptions: ScenarioAssumptions,
    output_path: str
) -> None:
    """
    Export scenario results to CSV with metadata header.
    
    Parameters
    ----------
    projection : DataFrame
        Scenario projection results
    assumptions : ScenarioAssumptions
        Scenario assumptions used
    output_path : str
        Path to save CSV file
    """
    import csv
    from datetime import datetime
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write metadata header
        writer.writerow(['# ERM Portfolio Scenario Projection'])
        writer.writerow(['# Generated:', datetime.now().isoformat()])
        writer.writerow(['#'])
        writer.writerow(['# Scenario Assumptions:'])
        for key, value in assumptions.to_dict().items():
            if isinstance(value, float):
                writer.writerow([f'# {key}:', f'{value:.4f}'])
            else:
                writer.writerow([f'# {key}:', str(value)])
        writer.writerow(['#'])
        writer.writerow([])  # Blank line
    
    # Append data
    projection.to_csv(output_path, mode='a', index=False)
    
    print(f"Scenario exported to: {output_path}")
