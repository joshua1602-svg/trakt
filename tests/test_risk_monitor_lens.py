from __future__ import annotations

import pandas as pd

from analytics.risk_monitor import RiskMonitor


def _base_limits():
    return {
        "max_variable_rate_pct": {
            "limit_value": 30.0,
            "direction": "max",
            "amber_threshold": 90.0,
            "description": "Variable rate cap",
            "severity": "high",
        },
        "max_age_over_85_pct": {
            "limit_value": 5.0,
            "direction": "max",
            "amber_threshold": 90.0,
            "description": "Age over 85 cap",
            "severity": "critical",
        },
    }


def test_funded_lens_behavior_unchanged_without_strict_missing():
    df = pd.DataFrame(
        {
            "current_principal_balance": [100.0, 200.0],
            "interest_rate_type": ["FIXED", None],
            "youngest_borrower_age": [70, 80],
        }
    )
    monitor = RiskMonitor(df, limits_config=_base_limits(), unknown_on_missing_required=False)
    results = {r.limit_id: r for r in monitor.check_all_limits()}
    assert results["max_variable_rate_pct"].status == "green"


def test_forward_combined_lens_marks_unknown_when_required_fields_missing():
    df = pd.DataFrame(
        {
            "current_principal_balance": [100.0, 200.0],
            "interest_rate_type": ["FIXED", None],
            "youngest_borrower_age": [70, None],
        }
    )
    monitor = RiskMonitor(df, limits_config=_base_limits(), unknown_on_missing_required=True)
    results = {r.limit_id: r for r in monitor.check_all_limits()}
    assert results["max_variable_rate_pct"].status == "unknown"
    assert results["max_age_over_85_pct"].status == "unknown"
