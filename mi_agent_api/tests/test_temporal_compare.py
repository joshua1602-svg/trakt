"""tests/test_temporal_compare.py

Execution coverage for the governed cross-period comparison helper (Part 3, bug
#2). Builds a small evolution-style ``periods`` list and asserts value A / value
B / absolute + % delta / source periods / reconciliation, plus the controlled
insufficient-data response when a period or metric is missing.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent_api import temporal_compare as tc


def _periods():
    return [
        {"period": "2025-10", "reporting_date": "2025-10-31",
         "metrics": {"funded_balance": 10_000_000.0, "loan_count": 40,
                     "wa_ltv": 0.42},
         "reconciliation": {"coverage_by_balance_pct": 100.0},
         "source_file": "/runs/mi_2025_10/18_central_lender_tape.csv"},
        {"period": "2025-11", "reporting_date": "2025-11-30",
         "metrics": {"funded_balance": 12_500_000.0, "loan_count": 45,
                     "wa_ltv": 0.44},
         "reconciliation": {"coverage_by_balance_pct": 100.0},
         "source_file": "/runs/mi_2025_11/18_central_lender_tape.csv"},
    ]


def test_resolve_metric_key_funded():
    assert tc.resolve_metric_key("funded", "current_outstanding_balance", "sum")[0] == "funded_balance"
    assert tc.resolve_metric_key("funded", None, "count")[0] == "loan_count"
    assert tc.resolve_metric_key("funded", "current_loan_to_value", "weighted_avg")[0] == "wa_ltv"
    assert tc.resolve_metric_key("pipeline", None, "sum")[0] == "pipeline_amount"
    assert tc.resolve_metric_key("pipeline", None, "count")[0] == "pipeline_case_count"


def test_compare_balance_october_november():
    out = tc.compare_periods(_periods(), metric_key="funded_balance",
                             period_a="October", period_b="November",
                             label="Funded balance", fmt="gbp")
    assert out["available"] is True and out["status"] == "ok"
    assert out["periodA"] == "2025-10" and out["periodB"] == "2025-11"
    assert out["valueA"] == 10_000_000.0 and out["valueB"] == 12_500_000.0
    assert out["absoluteDelta"] == 2_500_000.0
    assert out["percentageDelta"] == 25.0
    assert out["direction"] == "up"
    assert out["sourcePeriods"][0].endswith("mi_2025_10/18_central_lender_tape.csv")
    assert out["reconciliation"]["periodA"]["coverage_by_balance_pct"] == 100.0


def test_compare_loan_count_relative_tokens():
    out = tc.compare_periods(_periods(), metric_key="loan_count",
                             period_a="prior", period_b="latest", fmt="count")
    assert out["valueA"] == 40 and out["valueB"] == 45
    assert out["absoluteDelta"] == 5.0
    assert out["direction"] == "up"


def test_compare_insufficient_data_missing_period():
    out = tc.compare_periods(_periods(), metric_key="funded_balance",
                             period_a="August", period_b="November")
    assert out["available"] is False
    assert out["status"] == "insufficient_data"
    assert "August" in out["reason"]
    assert out["availablePeriods"] == ["2025-10", "2025-11"]


def test_compare_insufficient_data_missing_metric():
    out = tc.compare_periods(_periods(), metric_key="wa_interest_rate",
                             period_a="October", period_b="November")
    assert out["available"] is False
    assert out["status"] == "insufficient_data"


def test_compare_single_period_history_is_insufficient():
    one = _periods()[:1]
    out = tc.compare_periods(one, metric_key="funded_balance",
                             period_a="prior", period_b="latest")
    # 'prior' cannot resolve against a single-period history.
    assert out["available"] is False and out["status"] == "insufficient_data"
