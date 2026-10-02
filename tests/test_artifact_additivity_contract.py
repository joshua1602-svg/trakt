#!/usr/bin/env python3
"""tests/test_artifact_additivity_contract.py — the engine publishes what it knows.

Whether a column can be SUMMED across categories is a property of the
aggregation that produced it, and the engine is the only layer that knows.
It used to keep the answer to itself — using it to decide whether a capped bar
kept an aggregated "Other" bucket — while the browser re-derived it from the
DISPLAY FORMAT. A format cannot tell a sum from an average, so ten broker
averages were summed into a "portfolio total" and shares were reported of it.

The engine now states two facts on every chart artifact:

  displayHints[column].additive     may this column be added up?
  population                        are these rows the whole population?

Both are asserted here at the source. The consumer side is pinned in
``frontend/mi-agent-ui/src/lib/additivity.contract.test.ts`` against artifacts
captured from this same route.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mi_agent_api.adapters import is_additive_measure  # noqa: E402


# --------------------------------------------------------------------------- #
# The determination itself.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("column,additive", [
    ("current_outstanding_balance_sum", True),
    ("current_outstanding_balance_total", True),
    ("count", True),
    # Money, and NOT sum-able. This is the whole point.
    ("current_outstanding_balance_avg", False),
    ("current_outstanding_balance_weighted_avg", False),
    ("current_outstanding_balance_median", False),
    ("current_loan_to_value_avg", False),
    ("concentration_pct", False),
    ("loan_count", False),      # a per-row count column, not the count aggregate
    ("", False),
])
def test_additivity_is_decided_by_aggregation_not_format(column, additive):
    assert is_additive_measure(column) is additive, column


def test_two_columns_of_the_same_field_differ():
    """The pair that defeats a format-based rule: both render as money."""
    assert is_additive_measure("current_outstanding_balance_sum") is True
    assert is_additive_measure("current_outstanding_balance_avg") is False


# --------------------------------------------------------------------------- #
# It reaches the artifact, through the real route.
# --------------------------------------------------------------------------- #

BROKERS = [f"Broker {c}" for c in "ABCDEFGHIJKLM"]     # 13 > the top-10 cap


@pytest.fixture()
def book(tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd

    root = tmp_path / "runs"
    for run_id, date, n in (("mi_2026_05", "2026-05-31", 120),
                            ("mi_2026_06", "2026-06-30", 130)):
        d = root / "client_001" / run_id / "central"
        d.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(7)
        pd.DataFrame({
            "loan_identifier": [f"{run_id}_{i}" for i in range(n)],
            "unique_identifier": [f"{run_id}_{i}" for i in range(n)],
            "source_portfolio_id": "direct_001", "source_portfolio_type": "direct",
            "current_outstanding_balance": rng.uniform(120_000, 480_000, n).round(2),
            "current_valuation_amount": rng.uniform(500_000, 1_200_000, n).round(2),
            "current_loan_to_value": rng.uniform(20, 70, n).round(1),
            "current_interest_rate": rng.uniform(5, 8, n).round(2),
            "youngest_borrower_age": rng.integers(60, 88, n),
            "broker_channel": [BROKERS[i % len(BROKERS)] for i in range(n)],
            "geographic_region_collateral": "London",
            "origination_date": "2022-05-01", "data_cut_off_date": date,
        }).to_csv(d / "18_central_lender_tape.csv", index=False)
    monkeypatch.setenv("MI_AGENT_ONBOARDING_OUTPUT_ROOT", str(root))
    monkeypatch.setenv("MI_AGENT_AUTH_ENABLED", "false")
    monkeypatch.setenv("TRAKT_RUNTIME_MODE", "test")
    return root


def _chart(book, question):
    from fastapi.testclient import TestClient
    from mi_agent_api.app import app

    body = TestClient(app).post("/mi/query", json={
        "question": question, "portfolioId": "client_001/mi_2026_06",
        "asOfDate": "2026-06-30"}).json()
    assert body.get("ok"), body.get("error")
    chart = next((a for a in body["artifacts"] if a.get("type") == "chart"), None)
    assert chart is not None, "no chart artifact"
    return chart


def test_a_money_average_is_published_non_additive(book):
    """THE AUDIT CASE, at the source."""
    chart = _chart(book, "What is the average balance by broker channel?")
    key = chart["series"][0]["key"]
    assert key.endswith("_avg"), key
    hint = chart["displayHints"][key]
    assert hint["format"] == "gbp"        # money…
    assert hint["additive"] is False      # …and not sum-able


def test_a_money_sum_is_published_additive(book):
    chart = _chart(book, "What is the total balance by broker channel?")
    key = chart["series"][0]["key"]
    assert chart["displayHints"][key]["format"] == "gbp"
    assert chart["displayHints"][key]["additive"] is True


def test_a_truncated_non_additive_result_says_its_population_is_incomplete(book):
    """13 brokers in, 10 out, tail DROPPED — so the rows are not the whole."""
    chart = _chart(book, "What is the average balance by broker channel?")
    assert chart["population"] == {
        "returnedCount": 10, "totalCount": 13,
        "truncated": True, "populationComplete": False,
    }


def test_a_truncated_additive_result_is_still_complete(book):
    """13 in, 10 out — but the remainder is folded into "Other", so every
    category's VALUE is still represented and a share is still meaningful."""
    chart = _chart(book, "What is the total balance by broker channel?")
    population = chart["population"]
    assert population["truncated"] is True
    assert population["returnedCount"] == 10
    assert population["totalCount"] == 13
    assert population["populationComplete"] is True
    assert any(str(r.get("broker_channel")) == "Other" for r in chart["rows"])


def test_an_uncapped_result_is_complete_and_not_truncated(book):
    chart = _chart(book, "What is the total balance by region?")
    population = chart["population"]
    assert population["truncated"] is False
    assert population["populationComplete"] is True
    assert population["returnedCount"] == population["totalCount"]
