"""
risk_monitor.py

Risk monitoring engine for ERM portfolio dashboards.
Calculates current metrics and checks against configured limits.
"""

from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
from config.client.risk_limits_config import ALL_LIMITS, LIMIT_CATEGORIES


@dataclass
class LimitCheck:
    """Result of a single limit check"""
    limit_id: str
    category: str
    description: str
    limit_value: float
    current_value: float
    status: str  # 'green', 'amber', 'red', 'unknown'
    utilization_pct: float  # How much of limit is used (0-100+)
    breach_amount: Optional[float]  # Amount by which limit is breached (if any)
    severity: str  # 'critical' or 'high'

REGION_NAME_TO_CODE = {
    "LONDON": "UKI",
    "SOUTH EAST": "UKJ",
    "EAST ANGLIA": "UKH",
    "EAST OF ENGLAND": "UKH",
    "SOUTH WEST": "UKK",
    "WEST MIDLANDS": "UKG",
    "EAST MIDLANDS": "UKF",
    "NORTH WEST": "UKD",
    "YORKSHIRE AND HUMBERSIDE": "UKE",
    "YORKSHIRE & HUMBERSIDE": "UKE",
    "NORTH EAST": "UKC",
    "WALES": "UKL",
    "SCOTLAND": "UKM",
    "NORTHERN IRELAND": "UKN",
}

class RiskMonitor:
    """Calculate portfolio metrics and check against risk limits"""

    def __init__(self, df: pd.DataFrame, limits_config: Dict = None):
        """
        Args:
            df: Portfolio dataframe (canonical format)
            limits_config: Optional custom limits (defaults to ALL_LIMITS)
        """
        self.df = df
        self.limits = limits_config or ALL_LIMITS

        # Decide which balance column to use everywhere
        if "current_principal_balance" not in df.columns:
            raise ValueError(
                "RiskMonitor requires 'current_principal_balance' in the dataframe."
            )

        self.balance_col = "current_principal_balance"
        self.total_balance = float(df[self.balance_col].sum())

        self.total_balance = (
            float(df[self.balance_col].sum()) if self.balance_col is not None else 0.0
        )

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def weighted_average(self, series: pd.Series, weights: pd.Series) -> float:
        """Calculate weighted average, properly handling NaN."""
        mask = series.notna() & weights.notna()
        if not mask.any():
            return np.nan
        return float(np.average(series[mask], weights=weights[mask]))

    def _percent_of_balance(self, mask: pd.Series) -> float:
        """Generic helper: % of portfolio balance matching a boolean mask."""
        if self.balance_col is None or self.total_balance == 0:
            return 0.0
        return float(
            self.df.loc[mask, self.balance_col].sum() / self.total_balance * 100.0
        )

    def _get_borrower_col(self) -> Optional[str]:
        """Try to identify a borrower identifier column."""
        for candidate in ["borrower_id", "unique_borrower_id", "customer_id"]:
            if candidate in self.df.columns:
                return candidate
        return None


    # ------------------------------------------------------------------
    # Schedule 8 – Regional concentration metrics
    # ------------------------------------------------------------------

    def calc_region_pct(self, region_codes: List[str]) -> float:
        col = "geographic_region_classification"
        if col not in self.df.columns:
            return np.nan

        s = (
            self.df[col]
            .astype(str)
            .str.upper()
            .str.strip()
        )

    # Map region names → ONS codes
        mapped_codes = s.map(REGION_NAME_TO_CODE)

        mask = mapped_codes.isin([c.upper().strip() for c in region_codes])
        return self._percent_of_balance(mask)


    def calc_region_uki(self) -> float:
        """London % of portfolio."""
        return self.calc_region_pct(["UKI"])

    def calc_region_ukj(self) -> float:
        """South East % of portfolio."""
        return self.calc_region_pct(["UKJ"])

    def calc_region_ukc(self) -> float:
        """North East % of portfolio."""
        return self.calc_region_pct(["UKC"])

    def calc_region_ukd(self) -> float:
        """North West % of portfolio."""
        return self.calc_region_pct(["UKD"])

    def calc_region_uke(self) -> float:
        """Yorkshire and the Humberside % of portfolio."""
        return self.calc_region_pct(["UKE"])

    def calc_region_ukf(self) -> float:
        """East Midlands % of portfolio."""
        return self.calc_region_pct(["UKF"])

    def calc_region_ukg(self) -> float:
        """West Midlands % of portfolio."""
        return self.calc_region_pct(["UKG"])

    def calc_region_ukh(self) -> float:
        """East Anglia % of portfolio."""
        return self.calc_region_pct(["UKH"])

    def calc_region_ukk(self) -> float:
        """South West % of portfolio."""
        return self.calc_region_pct(["UKK"])

    def calc_region_ukl(self) -> float:
        """Wales % of portfolio."""
        return self.calc_region_pct(["UKL"])

    def calc_region_ukm(self) -> float:
        """Scotland % of portfolio."""
        return self.calc_region_pct(["UKM"])

    def calc_region_ukn(self) -> float:
        """Northern Ireland % of portfolio."""
        return self.calc_region_pct(["UKN"])

    def calc_region_uki_ukj_combined(self) -> float:
        """London + South East combined % of portfolio (if used as a backstop)."""
        return self.calc_region_pct(["UKI", "UKJ"])

    # ------------------------------------------------------------------
    # Schedule 8 – Property value buckets (original valuation)
    # ------------------------------------------------------------------

    def calc_low_value_property_pct(self, threshold: float = 150_000.0) -> float:
        """
        % of portfolio where original property valuation is below threshold.
        Schedule 8: <£150k cap as % of portfolio.
        """
        col = "original_valuation_amount"
        if col not in self.df.columns:
            return np.nan
        vals = pd.to_numeric(self.df[col], errors="coerce")
        mask = vals < threshold
        return self._percent_of_balance(mask)

    def calc_high_value_property_pct(self, threshold: float = 1_000_000.0) -> float:
        """
        % of portfolio where original property valuation exceeds threshold.
        Schedule 8: >£1m cap as % of portfolio.
        """
        col = "original_valuation_amount"
        if col not in self.df.columns:
            return np.nan
        vals = pd.to_numeric(self.df[col], errors="coerce")
        mask = vals > threshold
        return self._percent_of_balance(mask)

    # ------------------------------------------------------------------
    # Schedule 8 – Borrower-level concentration
    # ------------------------------------------------------------------

    def calc_max_single_borrower_balance_pct(self) -> float:
        """Max % of portfolio balance to any single borrower."""
        borrower_col = self._get_borrower_col()
        if borrower_col is None or self.total_balance == 0:
            return np.nan

        by_borrower = self.df.groupby(borrower_col)[self.balance_col].sum()
        if by_borrower.empty:
            return 0.0
        max_balance = by_borrower.max()
        return float(max_balance / self.total_balance * 100.0)

    def calc_max_loans_per_borrower(self) -> float:
        """Maximum number of loans to any single borrower."""
        borrower_col = self._get_borrower_col()
        if borrower_col is None:
            return np.nan

        counts = self.df.groupby(borrower_col).size()
        return float(counts.max()) if not counts.empty else 0.0

    def calc_balance_to_multi_loan_borrowers_pct(self, min_loans: int = 3) -> float:
        """
        % of portfolio balance to borrowers with at least `min_loans` loans.
        Schedule 8: constrain share of balance to borrowers with >2 loans.
        """
        borrower_col = self._get_borrower_col()
        if borrower_col is None or self.total_balance == 0:
            return np.nan

        counts = self.df.groupby(borrower_col).size()
        multi_ids = counts[counts >= min_loans].index
        if len(multi_ids) == 0:
            return 0.0

        mask = self.df[borrower_col].isin(multi_ids)
        return self._percent_of_balance(mask)

    # ------------------------------------------------------------------
    # Schedule 8 – Age-based concentration
    # ------------------------------------------------------------------

    def calc_age_over_85_pct(self) -> float:
        """
        % of portfolio balance to borrowers aged >85.
        Uses 'youngest_borrower_age' as proxy (if available).
        """
        col = "youngest_borrower_age"
        if col not in self.df.columns:
            return np.nan
        ages = pd.to_numeric(self.df[col], errors="coerce")
        mask = ages > 85
        return self._percent_of_balance(mask)

    # ------------------------------------------------------------------
    # Schedule 8 – Interest rate structure
    # ------------------------------------------------------------------

    def calc_variable_rate_pct(self) -> float:
        """
        % of portfolio with variable interest rates.
        Tries a few likely flag columns.
        """
        rate_type_col = None
        for c in ["interest_rate_type", "rate_type", "fixed_or_variable_flag"]:
            if c in self.df.columns:
                rate_type_col = c
                break

        if rate_type_col is None:
            return np.nan

        series = self.df[rate_type_col].astype(str).str.upper()
        # Basic heuristic: anything containing 'VAR' treated as variable
        mask = series.str.contains("VAR")
        return self._percent_of_balance(mask)

    # ------------------------------------------------------------------
    # Metric router
    # ------------------------------------------------------------------

    def calculate_metric(self, limit_id: str) -> float:
        """Route limit_id to appropriate calculator."""
        calculators = {

            # Schedule 8 – Regional
            "max_region_uki_pct": self.calc_region_uki,
            "max_region_ukj_pct": self.calc_region_ukj,
            "max_region_ukc_pct": self.calc_region_ukc,
            "max_region_ukd_pct": self.calc_region_ukd,
            "max_region_uke_pct": self.calc_region_uke,
            "max_region_ukf_pct": self.calc_region_ukf,
            "max_region_ukg_pct": self.calc_region_ukg,
            "max_region_ukh_pct": self.calc_region_ukh,
            "max_region_ukk_pct": self.calc_region_ukk,
            "max_region_ukl_pct": self.calc_region_ukl,
            "max_region_ukm_pct": self.calc_region_ukm,
            "max_region_ukn_pct": self.calc_region_ukn,
            "max_region_uki_ukj_combined": self.calc_region_uki_ukj_combined,

            # Schedule 8 – Property value buckets
            "max_low_value_property_pct": self.calc_low_value_property_pct,
            "max_high_value_property_pct": self.calc_high_value_property_pct,

            # Schedule 8 – Borrower concentration
            "max_single_borrower_balance_pct": self.calc_max_single_borrower_balance_pct,
            "max_loans_per_borrower": self.calc_max_loans_per_borrower,
            "max_balance_to_multi_loan_borrowers": self.calc_balance_to_multi_loan_borrowers_pct,

            # Schedule 8 – Age & rate-type
            "max_age_over_85_pct": self.calc_age_over_85_pct,
            "max_variable_rate_pct": self.calc_variable_rate_pct,
        }

        calculator = calculators.get(limit_id)
        if calculator is None:
            # No calculator defined for this limit_id – return NaN
            return np.nan

        try:
            return calculator()
        except Exception as e:
            print(f"Warning: Could not calculate {limit_id}: {e}")
            return np.nan

    # ------------------------------------------------------------------
    # Status determination
    # ------------------------------------------------------------------

    def determine_status(
        self,
        current_value: float,
        limit_value: float,
        direction: str,
        amber_threshold: float,
    ) -> Tuple[str, float, Optional[float]]:
        """
        Determine status (green/amber/red) and calculate utilization.

        Returns:
            (status, utilization_pct, breach_amount)
        """
        if np.isnan(current_value):
            return "unknown", 0.0, None

        if limit_value == 0 and direction == "max":
            # Edge case: zero cap (e.g. age > 85)
            if current_value > 0:
                return "red", 100.0, current_value
            else:
                return "green", 0.0, None

        if direction == "max":
            # Current should be ≤ limit
            utilization = (current_value / limit_value) * 100 if limit_value != 0 else 0.0
            amber_trigger = limit_value * (amber_threshold / 100.0)

            if current_value > limit_value:
                return "red", utilization, current_value - limit_value
            elif current_value > amber_trigger:
                return "amber", utilization, None
            else:
                return "green", utilization, None

        else:  # direction == "min"
            # Current should be ≥ limit
            utilization = (current_value / limit_value) * 100 if limit_value != 0 else 0.0
            amber_trigger = limit_value * (amber_threshold / 100.0)

            if current_value < limit_value:
                return "red", utilization, limit_value - current_value
            elif current_value < amber_trigger:
                return "amber", utilization, None
            else:
                return "green", utilization, None

    # ------------------------------------------------------------------
    # Main evaluation APIs
    # ------------------------------------------------------------------

    def check_all_limits(self) -> List[LimitCheck]:
        """Check all configured limits and return results."""
        results: List[LimitCheck] = []

        for limit_id, config in self.limits.items():
            # Calculate current value
            current_value = self.calculate_metric(limit_id)

            # Determine status
            status, utilization, breach = self.determine_status(
                current_value=current_value,
                limit_value=config["limit_value"],
                direction=config["direction"],
                amber_threshold=config["amber_threshold"],
            )

            # Find category
            category = "Other"
            for cat, ids in LIMIT_CATEGORIES.items():
                if limit_id in ids:
                    category = cat
                    break

            # Create result
            result = LimitCheck(
                limit_id=limit_id,
                category=category,
                description=config["description"],
                limit_value=config["limit_value"],
                current_value=current_value,
                status=status,
                utilization_pct=utilization,
                breach_amount=breach,
                severity=config["severity"],
            )

            results.append(result)

        return results

    def get_summary_stats(self, results: List[LimitCheck]) -> Dict:
        """Calculate summary statistics from limit checks."""
        total = len(results)
        red = sum(1 for r in results if r.status == "red")
        amber = sum(1 for r in results if r.status == "amber")
        green = sum(1 for r in results if r.status == "green")
        unknown = sum(1 for r in results if r.status == "unknown")

        critical_breaches = sum(
            1 for r in results if r.status == "red" and r.severity == "critical"
        )
        high_breaches = sum(
            1 for r in results if r.status == "red" and r.severity == "high"
        )

        overall_status = "green"
        if red > 0:
            overall_status = "red"
        elif amber > 0:
            overall_status = "amber"

        return {
            "total_limits": total,
            "breaches": red,
            "warnings": amber,
            "compliant": green,
            "unknown": unknown,
            "critical_breaches": critical_breaches,
            "high_breaches": high_breaches,
            "overall_status": overall_status,
        }
