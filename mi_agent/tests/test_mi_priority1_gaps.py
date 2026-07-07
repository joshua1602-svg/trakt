#!/usr/bin/env python3
"""tests/test_mi_priority1_gaps.py

Focused regression for the two Priority-1 MI calibration gaps:

1. Filtered time-series — a value filter alongside a trend must be applied
   before the time series is built (or fail closed). Never a silently
   unfiltered trend.
2. Third dimension dropped at parse — a requested third dimension must never be
   silently truncated; all requested dimensions are preserved (as a table/pivot)
   and the dimension invariant sees them applied.

Plus fail-closed guards: the dimension invariant catches a simulated parse
truncation, and the filter invariant catches a simulated line-path filter
omission. An optional @pytest.mark.live_llm test runs the same cases through the
LLM parser (skipped without ANTHROPIC_API_KEY) and asserts the SAME contract.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent.mi_query_validator import load_mi_semantics
from mi_agent.mi_agent_workflow import run_mi_agent_query
from mi_agent.mi_query_harness import build_fixture
from mi_agent.mi_query_contract import (
    canonical_of, check_filter_invariant, check_dimension_invariant)
from mi_agent import mi_calibration as CAL

_SEMANTICS = _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"


@pytest.fixture(scope="module")
def semantics():
    return load_mi_semantics(_SEMANTICS)


@pytest.fixture(scope="module")
def df():
    return build_fixture()


def _cols(res):
    qr = res["query_result"]
    return [str(c) for c in qr.data.columns]


def _recon(res):
    qr = res.get("query_result")
    return (getattr(qr, "metadata", {}) or {}).get("reconciliation") or {} if qr else {}


def _chart_types(res):
    from mi_agent_api.adapters import adapt_workflow_result
    ad = adapt_workflow_result(res, portfolio_id="client_001", as_of=None)
    return [a.get("chartType") if a["type"] == "chart" else a["type"]
            for a in ad.get("artifacts") or []]


# --------------------------------------------------------------------------- #
# Gap 1 — filtered time-series applies the filter (never an unfiltered trend).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("q", [
    "balance trend where LTV above 50%",
    "funded balance by month where LTV > 50%",
])
def test_filtered_time_series_applies_the_filter(df, semantics, q):
    res = run_mi_agent_query(q, df, semantics)
    assert res.get("ok") is True, res.get("error")
    fi = res["filter_invariant"]
    # The LTV filter is parsed AND applied to the mask before the trend.
    assert fi["ok"] and fi["filters_applied"], fi
    assert "current_loan_to_value" in fi["applied_filters"], fi
    recon = _recon(res)
    assert recon["records_after_filters"] < recon["total_records"], recon
    # It is a BALANCE trend, not the filter field's trend.
    assert res["spec"]["metric"] == "current_outstanding_balance"
    # A line chart is rendered (filtered), not a silently unfiltered one.
    assert "line" in _chart_types(res)


def test_filtered_time_series_is_never_silently_unfiltered(df, semantics):
    """Whatever the outcome (applied filter OR fail-closed), the result must not
    be an unfiltered trend presented as if it were filtered."""
    res = run_mi_agent_query("balance trend where LTV above 50%", df, semantics)
    fi = res["filter_invariant"]
    recon = _recon(res)
    answered = res.get("ok") and res.get("query_result") is not None
    if answered:
        # applied → fewer records
        assert fi["filters_applied"] and recon["records_after_filters"] < recon["total_records"]
    else:
        # fail closed → no data + reason
        assert res.get("error") or res.get("warnings")


# --------------------------------------------------------------------------- #
# Gap 2 — third dimension preserved (never silently truncated at parse).
# --------------------------------------------------------------------------- #
def test_three_dimensions_are_all_preserved_as_a_table(df, semantics):
    res = run_mi_agent_query("balance by region by borrower type by LTV bucket",
                             df, semantics)
    assert res.get("ok") is True, res.get("error")
    di = res["dimension_invariant"]
    # All three requested dimensions applied — none dropped.
    for d in ("geographic_region_obligor", "borrower_type", "ltv_bucket"):
        assert d in di["applied"], (d, di)
    assert not di["dropped"], di
    cols = _cols(res)
    for d in ("geographic_region_obligor", "borrower_type", "ltv_bucket"):
        assert canonical_of(d, semantics) in cols, (d, cols)
    # Rendered as a table (a chart shows at most two) with a visible reason.
    assert "table" in _chart_types(res)
    assert any("table across 3 dimensions" in str(w).lower() for w in (res.get("warnings") or [])), \
        res.get("warnings")


def test_parser_does_not_truncate_third_dimension(df, semantics):
    from mi_agent.llm_query_parser import _deterministic_parse
    spec, _ = _deterministic_parse("balance by region by borrower type by LTV bucket",
                                   semantics, available_columns=set(df.columns))
    dims = list(spec.dimensions or ([spec.dimension] if spec.dimension else []))
    assert len(dims) == 3, dims  # was truncated to 2 before the fix


# --------------------------------------------------------------------------- #
# Fail-closed guards catch simulated omissions.
# --------------------------------------------------------------------------- #
def test_dimension_invariant_catches_simulated_parse_truncation(df, semantics):
    """If the third dimension were dropped (e.g. absent from group cols and not
    rejected), the dimension invariant must flag it."""
    res = run_mi_agent_query("balance by region by borrower type by LTV bucket",
                             df, semantics)
    qr = res["query_result"]

    class _FakeQR:
        pass
    fake = _FakeQR()
    meta = dict(qr.metadata or {})
    meta["group_field_keys"] = ["geographic_region_obligor", "borrower_type"]  # dropped 3rd
    fake.metadata = meta
    fake.data = qr.data.drop(columns=[canonical_of("ltv_bucket", semantics)])
    inv = check_dimension_invariant(res["spec"], fake, semantics)
    assert inv.ok is False
    assert any(d["dimension"] == "ltv_bucket" for d in inv.dropped)


def test_filter_invariant_catches_simulated_line_filter_omission(df, semantics):
    res = run_mi_agent_query("balance trend where LTV above 50%", df, semantics)
    qr = res["query_result"]

    class _FakeQR:
        pass
    fake = _FakeQR()
    meta = dict(qr.metadata or {})
    recon = dict(meta.get("reconciliation") or {})
    recon["filters_applied"] = False   # simulate the line path dropping the filter
    recon["filters"] = {}
    meta["reconciliation"] = recon
    fake.metadata = meta
    fake.data = qr.data
    finv = check_filter_invariant(res["spec"], fake, semantics)
    assert finv.ok is False
    assert any(d["filter"] == "current_loan_to_value" for d in finv.dropped)


# --------------------------------------------------------------------------- #
# Existing single-dim / two-dim behaviour still holds.
# --------------------------------------------------------------------------- #
def test_single_and_two_dim_unaffected(df, semantics):
    r1 = run_mi_agent_query("balance by region", df, semantics)
    assert r1["ok"] and "bar" in _chart_types(r1)
    r2 = run_mi_agent_query("balance by broker by product type", df, semantics)
    assert r2["ok"] and "heatmap" in _chart_types(r2)
    di = r2["dimension_invariant"]
    assert set(di["applied"]) == {"broker_channel", "erm_product_type"}


# --------------------------------------------------------------------------- #
# trace contract (requirement 7)
# --------------------------------------------------------------------------- #
def test_query_trace_exposes_requested_applied_rejected(df, semantics):
    from mi_agent_api.adapters import adapt_workflow_result
    res = run_mi_agent_query("balance by region where LTV above 50%", df, semantics)
    ad = adapt_workflow_result(res, portfolio_id="client_001", as_of=None)
    t = ad["queryTrace"]
    for key in ("requested_dimensions", "applied_dimensions", "rejected_dimensions",
                "requested_filters", "applied_filters", "rejected_filters",
                "artifact_type", "reconciliation", "invariant", "filterInvariant"):
        assert key in t, key
    assert "geographic_region_obligor" in t["applied_dimensions"]
    assert "current_loan_to_value" in t["applied_filters"]
    assert t["artifact_type"] == "bar"


# --------------------------------------------------------------------------- #
# Optional live-LLM contract (skipped without credentials).
# --------------------------------------------------------------------------- #
@pytest.mark.live_llm
@pytest.mark.skipif(not CAL.llm_available(), reason="ANTHROPIC_API_KEY not set")
def test_priority1_contract_under_live_llm(df, semantics):
    rows = CAL.run_live_llm_priority(df=df, semantics=semantics)
    for row in rows:
        assert row["contract_ok"], row
        assert row["dimension_invariant_ok"], row
        assert row["filter_invariant_ok"], row


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
