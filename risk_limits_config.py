"""
risk_limits_config.py

Configurable risk limits for ERM portfolio monitoring.
Update these values to match specific mandate/investment criteria.

Structure:
    - limit_value: The threshold value
    - amber_threshold: % of limit that triggers amber warning (default 80%)
    - direction: 'max' (must be ≤) or 'min' (must be ≥)
    - severity: 'critical' or 'high' (impacts alert priority)
"""

CONCENTRATION_LIMITS = {
    # ------------------------------------------------------------------
    # Regional concentration limits 
    # All are % of Concentration Limit Denominator (= effectively portfolio balance)
    # ------------------------------------------------------------------
    "max_region_uki_pct": {
        "limit_value": 30.0,
        "amber_threshold": 90,  # warn at 27%
        "direction": "max",
        "severity": "critical",
        "description": "London exposure must not exceed 30% of portfolio"
    },
    "max_region_ukj_pct": {
        "limit_value": 30.0,
        "amber_threshold": 90,
        "direction": "max",
        "severity": "critical",
        "description": "South East exposure must not exceed 30% of portfolio"
    },
    "max_region_ukc_pct": {
        "limit_value": 10.0,
        "amber_threshold": 90,
        "direction": "max",
        "severity": "high",
        "description": "North East exposure must not exceed 10% of portfolio"
    },
    "max_region_ukd_pct": {
        "limit_value": 15.0,
        "amber_threshold": 90,
        "direction": "max",
        "severity": "high",
        "description": "North West exposure must not exceed 15% of portfolio"
    },
    "max_region_uke_pct": {
        "limit_value": 10.0,
        "amber_threshold": 90,
        "direction": "max",
        "severity": "high",
        "description": "Yorkshire and the Humberside exposure must not exceed 10% of portfolio"
    },
    "max_region_ukf_pct": {
        "limit_value": 10.0,
        "amber_threshold": 90,
        "direction": "max",
        "severity": "high",
        "description": "East Midlands exposure must not exceed 10% of portfolio"
    },
    "max_region_ukg_pct": {
        "limit_value": 10.0,
        "amber_threshold": 90,
        "direction": "max",
        "severity": "high",
        "description": "West Midlands exposure must not exceed 10% of portfolio"
    },
    "max_region_ukh_pct": {
        "limit_value": 10.0,
        "amber_threshold": 90,
        "direction": "max",
        "severity": "high",
        "description": "East Anglia exposure must not exceed 10% of portfolio"
    },
    "max_region_ukk_pct": {
        "limit_value": 10.0,
        "amber_threshold": 90,
        "direction": "max",
        "severity": "high",
        "description": "South West exposure must not exceed 10% of portfolio"
    },
    "max_region_ukl_pct": {
        "limit_value": 10.0,
        "amber_threshold": 90,
        "direction": "max",
        "severity": "high",
        "description": "Wales exposure must not exceed 10% of portfolio"
    },
    "max_region_ukm_pct": {
        "limit_value": 10.0,
        "amber_threshold": 90,
        "direction": "max",
        "severity": "high",
        "description": "Scotland exposure must not exceed 10% of portfolio"
    },
    "max_region_ukn_pct": {
        "limit_value": 10.0,
        "amber_threshold": 90,
        "direction": "max",
        "severity": "high",
        "description": "Northern Ireland exposure must not exceed 10% of portfolio"
    },

    # You can optionally still keep a generic "max_region_concentration"
    # as a backstop (e.g. 30%) if you like:
    # "max_region_concentration": {...}

    # ------------------------------------------------------------------
    # Property value concentration (original valuation buckets)
    # ------------------------------------------------------------------
    "max_low_value_property_pct": {
        "limit_value": 10.0,   # < £150k
        "amber_threshold": 80,
        "direction": "max",
        "severity": "high",
        "description": "Loans secured on properties with original valuation < £150k must not exceed 10% of portfolio"
    },
    "max_high_value_property_pct": {
        "limit_value": 10.0,   # > £1m
        "amber_threshold": 80,
        "direction": "max",
        "severity": "high",
        "description": "Loans secured on properties with original valuation > £1m must not exceed 10% of portfolio"
    },

    # ------------------------------------------------------------------
    # Borrower-level concentration
    # ------------------------------------------------------------------
    "max_single_borrower_balance_pct": {
        "limit_value": 10.0,   # any individual borrower
        "amber_threshold": 80,
        "direction": "max",
        "severity": "critical",
        "description": "Current Balance to any single borrower/group must not exceed 10% of portfolio"
    },
    "max_loans_per_borrower": {
        "limit_value": 5.0,    # max 5 loans per borrower
        "amber_threshold": 80,
        "direction": "max",
        "severity": "critical",
        "description": "No single borrower may have more than 5 loans in the portfolio"
    },
    "max_balance_to_multi_loan_borrowers": {
        "limit_value": 20.0,   # >2 loans per borrower
        "amber_threshold": 80,
        "direction": "max",
        "severity": "high",
        "description": "Borrowers with more than 2 loans must not account for more than 20% of portfolio balance"
    },

    # ------------------------------------------------------------------
    # Age-based concentration
    # ------------------------------------------------------------------
    "max_age_over_85_pct": {
        "limit_value": 0.0,
        "amber_threshold": 0,  # any non-zero is a breach
        "direction": "max",
        "severity": "critical",
        "description": "Aggregate Current Balance to borrowers aged >85 at origination must not exceed 0% of portfolio"
    },

    # ------------------------------------------------------------------
    # Interest rate structure concentration
    # ------------------------------------------------------------------
    "max_variable_rate_pct": {
        "limit_value": 90.0,
        "amber_threshold": 80,
        "direction": "max",
        "severity": "high",
        "description": "Current Balance of loans with variable interest rates must not exceed 90% of portfolio"
    },
}

# ==================================================================
# MASTER LIMITS DICTIONARY
# ==================================================================

ALL_LIMITS = {
    **CONCENTRATION_LIMITS,
}

# ==================================================================
# LIMIT CATEGORIES (for organized display)
# ==================================================================

LIMIT_CATEGORIES = {
    "Concentration": list(CONCENTRATION_LIMITS.keys()),
}
