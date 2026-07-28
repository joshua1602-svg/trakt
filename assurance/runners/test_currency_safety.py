"""Currency safety on the default point-in-time route.

Defect ASSURE-006 (Critical): a mixed-currency population (e.g. a book carrying
both GBP and EUR) had its balances summed into a single figure formatted with the
modal currency symbol, silently. Adding amounts across currencies without
governed FX conversion is meaningless. The fix suppresses monetary totals on a
mixed-currency population and discloses the limitation; count measures remain.

Residual: dashboard GET routes and geo/cohort surfaces share the ungoverned sum
sites and are NOT covered by this guard — see the go-live report; launch is
recommended contained to a single currency until those are remediated.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

FIX = "assurance/fixtures/generated"


def _ask(fixture, monkeypatch, question="What is the total outstanding balance?"):
    monkeypatch.setenv("TRAKT_RUNTIME_MODE", "test")
    monkeypatch.setenv("MI_AGENT_PLATFORM_CANONICAL", os.path.abspath(f"{FIX}/{fixture}.csv"))
    monkeypatch.delenv("MI_AGENT_ONBOARDING_OUTPUT_ROOT", raising=False)
    monkeypatch.setenv("MI_AGENT_CLIENT_ID", "client_001")
    monkeypatch.setenv("MI_AGENT_LLM_ENABLED", "0")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from mi_agent_api import data_source, datasets
    data_source.reset_cache()
    datasets._CLIENT_CURRENCY_CACHE.clear()
    from trakt_core.context import ExecutionContext
    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query
    ctx = ExecutionContext.for_internal("client_001")
    result = execute_governed_mi_query(MiQueryRequest(question=question), ctx)
    data_source.reset_cache()
    datasets._CLIENT_CURRENCY_CACHE.clear()
    return result


def _artifact(result):
    arts = (result.result or {}).get("artifacts") or []
    assert arts
    return arts[0]


def test_mixed_currency_suppresses_monetary_total(monkeypatch):
    result = _ask("mixed_currency", monkeypatch)
    art = _artifact(result)
    balance_kpi = next((k for k in art.get("kpis") or []
                        if "balance" in str(k.get("label", "")).lower()), None)
    assert balance_kpi is not None
    assert balance_kpi.get("suppressed") is True
    assert "mixed currency" in str(balance_kpi.get("value")).lower()
    assert (art.get("reconciliation") or {}).get("total_balance") is None


def test_mixed_currency_is_disclosed(monkeypatch):
    result = _ask("mixed_currency", monkeypatch)
    warnings = (result.result or {}).get("warnings") or []
    assert any("more than one currency" in w for w in warnings)
    assert (result.result or {}).get("metadata", {}).get("currencyLimitation")


def test_mixed_currency_count_measure_still_answered(monkeypatch):
    result = _ask("mixed_currency", monkeypatch, question="How many loans are there?")
    art = _artifact(result)
    loan_kpi = next((k for k in art.get("kpis") or []
                     if "loan" in str(k.get("label", "")).lower()), None)
    assert loan_kpi is not None
    assert loan_kpi.get("suppressed") is not True
    assert str(loan_kpi.get("value")) not in ("", "None")


def test_single_currency_is_unchanged(monkeypatch):
    result = _ask("three_portfolios", monkeypatch)
    art = _artifact(result)
    expected = float(pd.read_csv(f"{FIX}/three_portfolios.csv")["current_outstanding_balance"].sum())
    assert (art.get("reconciliation") or {}).get("total_balance") == pytest.approx(expected)
    warnings = (result.result or {}).get("warnings") or []
    assert not any("currency" in w.lower() for w in warnings)
