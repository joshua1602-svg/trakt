"""
risk_limits_config.py

Configurable risk limits for the LEGACY Streamlit ERM dashboard and the client
PPTX pack (``analytics/risk_monitor.py``, ``analytics/streamlit_app_erm.py``,
``analytics/generate_pptx_client.py``).

WHERE THE GOVERNED SCHEDULE 8 LIVES — READ THIS BEFORE EDITING
==============================================================
This module is NOT the production Schedule 8 view. The governed, operator-
approved, versioned concentration configuration is:

    config/risk/concentration_test_library.yaml      the metric definitions
    mi_agent.concentration_tests.*                   extraction → approval →
                                                     activation → evaluation
    GET /mi/concentration-tests                      the Eligibility &
                                                     Concentrations workspace

That path measures Schedule 8 over ELIGIBLE MORTGAGE LOANS against the
contractual Concentration Limit Denominator (the greater of the £33m floor and
the eligible current balance). Nothing here does, and nothing here should try:
one governed definition per contractual test is the whole point.

What this module is for is the legacy dashboard, which predates that engine and
still has readers. Its Schedule 8 figures are therefore kept CONSISTENT WITH the
governed source — where the two disagreed, the values below were corrected to
the supplied Schedule 8 — but they remain a legacy monitor measured over the
whole funded book, and they are labelled as such.

THE TWO GROUPS BELOW ARE DIFFERENT THINGS
=========================================
``FACILITY_CONCENTRATION_LIMITS``
    Tests that come from the warehouse facility's Schedule 8. Contractual.
    Deprecated here in favour of the governed engine; kept so the legacy
    dashboard keeps rendering, and corrected so it does not render a stale
    threshold beside the governed one.

``PORTFOLIO_RISK_LIMITS``
    Trakt / mandate monitoring that is NOT in Schedule 8 — the maximum loans to
    one borrower, the share to multi-loan borrowers, the largest single-borrower
    exposure, over-85 borrowers, variable-rate share. These are not contractual
    facility tests and are NOT deleted merely because Schedule 8 is silent on
    them; they answer a different question.

Structure of an entry:
    - limit_value: The threshold value
    - amber_threshold: % of limit that triggers amber warning
    - direction: 'max' (must be <=) or 'min' (must be >=)
    - severity: 'critical' or 'high' (impacts alert priority)
    - basis: which rulebook the limit comes from
    - threshold_amount: (optional) the numeric parameter the metric compares
      against, where the metric takes one. Read by
      ``analytics.risk_monitor.RiskMonitor.calculate_metric`` so a contractual
      amount lives in configuration rather than in a Python default.
"""

from dataclasses import dataclass
from typing import Optional

#: Where a limit comes from. A facility test and a house monitoring rule must
#: never be shown to an operator as the same kind of thing.
BASIS_FACILITY = "facility_concentration"
BASIS_PORTFOLIO = "portfolio_risk_monitoring"


@dataclass
class LimitCheck:
    """Result of a single risk limit check."""
    limit_id: str
    category: str
    description: str
    limit_value: float
    current_value: float
    status: str  # 'green', 'amber', 'red', 'unknown'
    utilization_pct: float  # How much of limit is used (0-100+)
    breach_amount: Optional[float]  # Amount by which limit is breached (if any)
    severity: str  # 'critical' or 'high'


# =========================================================================
# FACILITY CONCENTRATION LIMITS — warehouse Schedule 8
#
# Values reconciled to the supplied Schedule 8. The corrections made, and why:
#
#   * London and South East are ONE combined test at 50%, not two separate
#     30% tests. Schedule 8 states "UKI + UKJ  London and South East  [50]%";
#     the two separate limits that used to sit here were a stricter test the
#     facility does not impose, and they would have shown a book as breaching
#     a limit that does not exist.
#   * UKD North West 15% -> 10%, UKE Yorkshire & The Humber 10% -> 15%,
#     UKF East Midlands 10% -> 15%, UKG West Midlands 10% -> 15%,
#     UKH East of England 10% -> 25%, UKK South West 10% -> 20%: each read off
#     the schedule's own limits table.
#   * UKH is "East of England", not "East Anglia" — the schedule's wording.
#   * High-value property is > £1.5m, not > £1m.
#   * UKN Northern Ireland is NOT in Schedule 8's table. It is retained as
#     portfolio monitoring below rather than presented as a facility limit.
#
# The remaining Schedule 8 tests — borrower aggregate initial principal,
# average initial principal, the under-55 age test, the two-borrower test and
# Portfolio Net WAC — are NOT restated here. The legacy monitor has no
# calculator for them, and writing one would create the second Schedule 8
# engine this file's header exists to prevent. They are evaluated by the
# governed engine, which is where the whole schedule lives.
# =========================================================================

FACILITY_CONCENTRATION_LIMITS = {
    # ------------------------------------------------------------------
    # Geographic concentration — % of the Concentration Limit Denominator
    # ------------------------------------------------------------------
    "max_region_uki_ukj_combined": {
        "limit_value": 50.0,
        "amber_threshold": 90,
        "direction": "max",
        "severity": "critical",
        "basis": BASIS_FACILITY,
        "description": "London + South East (UKI + UKJ) exposure must not exceed 50%"
    },
    "max_region_ukc_pct": {
        "limit_value": 10.0,
        "amber_threshold": 90,
        "direction": "max",
        "severity": "high",
        "basis": BASIS_FACILITY,
        "description": "North East (UKC) exposure must not exceed 10%"
    },
    "max_region_ukd_pct": {
        "limit_value": 10.0,
        "amber_threshold": 90,
        "direction": "max",
        "severity": "high",
        "basis": BASIS_FACILITY,
        "description": "North West (UKD) exposure must not exceed 10%"
    },
    "max_region_uke_pct": {
        "limit_value": 15.0,
        "amber_threshold": 90,
        "direction": "max",
        "severity": "high",
        "basis": BASIS_FACILITY,
        "description": "Yorkshire & The Humber (UKE) exposure must not exceed 15%"
    },
    "max_region_ukf_pct": {
        "limit_value": 15.0,
        "amber_threshold": 90,
        "direction": "max",
        "severity": "high",
        "basis": BASIS_FACILITY,
        "description": "East Midlands (UKF) exposure must not exceed 15%"
    },
    "max_region_ukg_pct": {
        "limit_value": 15.0,
        "amber_threshold": 90,
        "direction": "max",
        "severity": "high",
        "basis": BASIS_FACILITY,
        "description": "West Midlands (UKG) exposure must not exceed 15%"
    },
    "max_region_ukh_pct": {
        "limit_value": 25.0,
        "amber_threshold": 90,
        "direction": "max",
        "severity": "high",
        "basis": BASIS_FACILITY,
        "description": "East of England (UKH) exposure must not exceed 25%"
    },
    "max_region_ukk_pct": {
        "limit_value": 20.0,
        "amber_threshold": 90,
        "direction": "max",
        "severity": "high",
        "basis": BASIS_FACILITY,
        "description": "South West (UKK) exposure must not exceed 20%"
    },
    "max_region_ukm_pct": {
        "limit_value": 10.0,
        "amber_threshold": 90,
        "direction": "max",
        "severity": "high",
        "basis": BASIS_FACILITY,
        "description": "Scotland (UKM) exposure must not exceed 10%"
    },
    "max_region_ukl_pct": {
        "limit_value": 10.0,
        "amber_threshold": 90,
        "direction": "max",
        "severity": "high",
        "basis": BASIS_FACILITY,
        "description": "Wales (UKL) exposure must not exceed 10%"
    },

    # ------------------------------------------------------------------
    # Property value concentration (ORIGINAL valuation)
    # ------------------------------------------------------------------
    "max_low_value_property_pct": {
        "limit_value": 10.0,
        "threshold_amount": 150_000.0,
        "amber_threshold": 80,
        "direction": "max",
        "severity": "high",
        "basis": BASIS_FACILITY,
        "description": "Loans on properties with an Original Valuation < £150,000 "
                       "must not exceed 10%"
    },
    "max_high_value_property_pct": {
        "limit_value": 10.0,
        # Schedule 8 says £1,500,000. The £1,000,000 that used to be hard-coded
        # in the calculator's default understated this test's population.
        "threshold_amount": 1_500_000.0,
        "amber_threshold": 80,
        "direction": "max",
        "severity": "high",
        "basis": BASIS_FACILITY,
        "description": "Loans on properties with an Original Valuation > £1,500,000 "
                       "must not exceed 10%"
    },
}

# =========================================================================
# PORTFOLIO RISK MONITORING — NOT Schedule 8
#
# Retained deliberately. None of these is a facility concentration test, and
# none is deleted for being absent from Schedule 8: they monitor a different
# risk and answer to a different owner.
# =========================================================================

PORTFOLIO_RISK_LIMITS = {
    "max_region_ukn_pct": {
        "limit_value": 10.0,
        "amber_threshold": 90,
        "direction": "max",
        "severity": "high",
        "basis": BASIS_PORTFOLIO,
        "description": "Northern Ireland (UKN) exposure must not exceed 10% "
                       "(portfolio monitoring — not a Schedule 8 limit)"
    },
    "max_single_borrower_balance_pct": {
        "limit_value": 10.0,
        "amber_threshold": 80,
        "direction": "max",
        "severity": "critical",
        "basis": BASIS_PORTFOLIO,
        "description": "Current Balance to any single borrower/group must not "
                       "exceed 10% of portfolio (largest-borrower monitoring — "
                       "distinct from the Schedule 8 aggregate-initial-principal "
                       "test, which the governed engine evaluates)"
    },
    "max_loans_per_borrower": {
        "limit_value": 5.0,
        "amber_threshold": 80,
        "direction": "max",
        "severity": "critical",
        "basis": BASIS_PORTFOLIO,
        "description": "No single borrower may have more than 5 loans in the portfolio"
    },
    "max_balance_to_multi_loan_borrowers": {
        "limit_value": 20.0,
        "amber_threshold": 80,
        "direction": "max",
        "severity": "high",
        "basis": BASIS_PORTFOLIO,
        "description": "Borrowers with more than 2 loans must not account for more "
                       "than 20% of portfolio balance"
    },
    "max_age_over_85_pct": {
        "limit_value": 0.0,
        "amber_threshold": 0,  # any non-zero is a breach
        "direction": "max",
        "severity": "critical",
        "basis": BASIS_PORTFOLIO,
        "description": "Aggregate Current Balance to borrowers aged >85 at "
                       "origination must not exceed 0% of portfolio (distinct from "
                       "the Schedule 8 youngest-borrower-under-55 test)"
    },
    "max_variable_rate_pct": {
        "limit_value": 90.0,
        "amber_threshold": 80,
        "direction": "max",
        "severity": "high",
        "basis": BASIS_PORTFOLIO,
        "description": "Current Balance of loans with variable interest rates must "
                       "not exceed 90% of portfolio"
    },
}

# ==================================================================
# BACKWARD-COMPATIBLE VIEWS
#
# ``CONCENTRATION_LIMITS`` and ``ALL_LIMITS`` are what the legacy dashboard,
# the PPTX generator and the concentration-test migration import. They keep
# their names and their shape; what changed is that the facility tests and the
# portfolio monitoring rules are now separable, and that a reader can tell
# which is which from ``basis``.
# ==================================================================

CONCENTRATION_LIMITS = {
    **FACILITY_CONCENTRATION_LIMITS,
    **PORTFOLIO_RISK_LIMITS,
}

ALL_LIMITS = {
    **CONCENTRATION_LIMITS,
}

# ==================================================================
# LIMIT CATEGORIES (for organized display)
# ==================================================================

LIMIT_CATEGORIES = {
    "Facility concentration (Schedule 8)": list(FACILITY_CONCENTRATION_LIMITS.keys()),
    "Portfolio risk monitoring": list(PORTFOLIO_RISK_LIMITS.keys()),
}
