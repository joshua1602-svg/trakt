"""Reporting-date safety on the default point-in-time route.

Defect ASSURE-004 (High): a combined tape carrying more than one cut-off date
was aggregated in full by the default `/mi/query` point-in-time path, so a KPI
such as total balance double-counted every loan present at two cuts and was
labelled with the first row's date, silently. The governed comparison and
concentration workflows already refuse a multi-date scope; the fix brings the
default route into line by resolving as-of-latest with a disclosed narrowing.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

MAY = Path("synthetic_demo/output/multibook/platform_2026-05-31_canonical_typed.csv")
JUN = Path("synthetic_demo/output/multibook/platform_2026-06-30_canonical_typed.csv")


@pytest.fixture()
def combined_tape(tmp_path):
    a = pd.read_csv(MAY)
    b = pd.read_csv(JUN)
    out = tmp_path / "combined_two_dates.csv"
    pd.concat([a, b], ignore_index=True).to_csv(out, index=False)
    return out, float(b["current_outstanding_balance"].sum()), len(b)


@pytest.fixture()
def ask(combined_tape, monkeypatch):
    tape, _, _ = combined_tape
    monkeypatch.setenv("TRAKT_RUNTIME_MODE", "test")
    monkeypatch.setenv("MI_AGENT_PLATFORM_CANONICAL", str(tape))
    monkeypatch.delenv("MI_AGENT_ONBOARDING_OUTPUT_ROOT", raising=False)
    monkeypatch.setenv("MI_AGENT_CLIENT_ID", "client_001")
    monkeypatch.setenv("MI_AGENT_LLM_ENABLED", "0")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from mi_agent_api import data_source, datasets
    data_source.reset_cache()
    datasets._CLIENT_CURRENCY_CACHE.clear()
    from trakt_core.context import ExecutionContext
    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query

    def _ask():
        ctx = ExecutionContext.for_internal("client_001")
        return execute_governed_mi_query(
            MiQueryRequest(question="What is the total outstanding balance?"), ctx)

    yield _ask, combined_tape
    data_source.reset_cache()
    datasets._CLIENT_CURRENCY_CACHE.clear()


def _reconciliation(result):
    arts = (result.result or {}).get("artifacts") or []
    assert arts
    return arts[0].get("reconciliation") or {}


def test_multi_date_tape_resolves_as_of_latest_not_aggregated(ask):
    run, (_, latest_balance, latest_rows) = ask
    result = run()
    rec = _reconciliation(result)
    assert rec.get("total_balance") == pytest.approx(latest_balance), (
        "point-in-time KPI must equal the latest cut, not the sum of both dates")
    assert rec.get("total_records") == latest_rows


def test_narrowing_is_disclosed(ask):
    run, _ = ask
    result = run()
    warnings = (result.result or {}).get("warnings") or []
    assert any("reporting date" in w.lower() and "latest" in w.lower() for w in warnings), (
        "a reporting-date narrowing must be disclosed, never silent")


def test_single_date_tape_is_unchanged_and_unwarned(monkeypatch):
    """The guard is a no-op for a single-date dataset."""
    monkeypatch.setenv("TRAKT_RUNTIME_MODE", "test")
    monkeypatch.setenv("MI_AGENT_PLATFORM_CANONICAL", str(JUN))
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
    result = execute_governed_mi_query(
        MiQueryRequest(question="What is the total outstanding balance?"), ctx)
    expected = float(pd.read_csv(JUN)["current_outstanding_balance"].sum())
    arts = (result.result or {}).get("artifacts") or []
    rec = arts[0].get("reconciliation") or {}
    assert rec.get("total_balance") == pytest.approx(expected)
    warnings = (result.result or {}).get("warnings") or []
    assert not any("reporting date" in w.lower() for w in warnings)
    data_source.reset_cache()
    datasets._CLIENT_CURRENCY_CACHE.clear()
