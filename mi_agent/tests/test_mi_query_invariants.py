#!/usr/bin/env python3
"""tests/test_mi_query_invariants.py

The fail-closed dimension-invariant regression, driven by the generative golden
harness (:mod:`mi_agent.mi_query_harness`). These tests run the REAL pipeline
(parse → validate → execute → adapt) over hundreds of registry-generated MI
questions and assert that no parsed dimension (or requested metric) is ever
silently dropped between the parser, the executor and the artifact renderer.

Deterministic and offline: the parser falls back to the deterministic grammar
when no LLM key is configured, so the suite is fully reproducible in CI.

Run just this file:
    python -m pytest mi_agent/tests/test_mi_query_invariants.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent.mi_query_validator import load_mi_semantics
from mi_agent.mi_query_contract import check_dimension_invariant, all_group_dims, canonical_of
from mi_agent.mi_agent_workflow import run_mi_agent_query
from mi_agent import mi_query_harness as H

_SEMANTICS = _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"


@pytest.fixture(scope="module")
def semantics():
    return load_mi_semantics(_SEMANTICS)


@pytest.fixture(scope="module")
def df():
    return H.build_fixture()


@pytest.fixture(scope="module")
def suite(df, semantics):
    results, summary = H.run_suite(df=df, semantics=semantics)
    return results, summary


# --------------------------------------------------------------------------- #
# The headline guarantee — the whole generated suite passes the invariant.
# --------------------------------------------------------------------------- #
def test_full_generated_suite_holds_the_invariant(suite):
    results, summary = suite
    failures = [r for r in results if not r.ok]
    # A readable failure report: class, reason, and the offending query.
    report = "\n".join(f"  [{r.failure_class}] {r.case.query} :: {r.detail}"
                       for r in failures)
    assert not failures, (
        f"{len(failures)}/{summary['total']} generated queries violated the "
        f"dimension/metric invariant:\n{report}")


def test_suite_is_large_enough_to_be_meaningful(suite):
    _, summary = suite
    # Hundreds of generated variants across all query classes.
    assert summary["total"] >= 200, summary["total"]
    for kind in ("single_dim", "two_dim", "three_dim", "filtered_kpi",
                 "grouped_filter", "filter_unsupported", "top_n", "ranking",
                 "weighted_avg", "count", "rejection"):
        assert kind in summary["by_kind"], kind


def test_both_invariants_hold_across_the_suite(suite):
    """Dimension AND filter invariants are tracked separately and both hold for
    every case where they apply."""
    _, summary = suite
    dim = summary["dimension_invariant"]
    filt = summary["filter_invariant"]
    assert dim["checked"] > 0 and dim["breached"] == 0, dim
    assert filt["checked"] > 0 and filt["breached"] == 0, filt
    # Filters were genuinely exercised (not a vacuous pass).
    assert summary["filters_exercised"] >= 10, summary["filters_exercised"]


def test_filters_reach_the_mask_on_supported_shape(suite):
    """Filtered KPIs AND grouped filters must genuinely apply the filter (parsed
    into the spec AND reflected in the reconciliation)."""
    results, _ = suite
    fcases = [r for r in results if r.case.kind in ("filtered_kpi", "grouped_filter")]
    assert fcases, "expected filtered cases"
    assert all(r.ok for r in fcases), [(r.case.query, r.detail) for r in fcases if not r.ok]


def test_grouped_filter_now_supported(suite):
    """Grouped query + value filter is now SUPPORTED: execution applies the
    filter to the mask before grouping. (Previously an evidenced limitation.)"""
    _, summary = suite
    sup = summary.get("grouped_filter_support")
    assert sup is not None
    assert sup["supported"] is True, sup
    assert sup["filter_parsed"] is True and sup["filters_applied"] is True, sup
    assert sup["records_after_filters"] < sup["total_records"], sup


def test_no_silent_filter_omission(suite):
    """No case may fail because a parsed filter was silently omitted, and no
    'ok' case may carry a breached filter invariant."""
    results, _ = suite
    silent = [r for r in results
              if not r.ok and r.failure_class == "filter"]
    assert not silent, [(r.case.query, r.detail) for r in silent]
    leaked = [r for r in results if r.ok and r.filter_invariant_ok is False]
    assert not leaked, [r.case.query for r in leaked]


def test_no_silent_drops_anywhere(suite):
    results, _ = suite
    # No case may fail specifically because a dimension was silently dropped.
    silent = [r for r in results
              if not r.ok and r.failure_class == "executor"
              and "silent drop" in r.detail]
    assert not silent, [r.case.query for r in silent]


# --------------------------------------------------------------------------- #
# The reported regression, pinned as an explicit named case.
# --------------------------------------------------------------------------- #
def test_balance_by_borrower_type_by_region_groups_by_both(df, semantics):
    """The exact reported failure: two dimensions must BOTH survive to the
    result — grouping by borrower type only is a silent drop and must not
    happen (and, were it to, the invariant would refuse the answer)."""
    res = run_mi_agent_query("Show balance by borrower type by region", df, semantics)
    assert res.get("ok") is True, res.get("error")
    inv = check_dimension_invariant_from_result(res, semantics)
    qr = res["query_result"]
    cols = [str(c) for c in qr.data.columns]
    assert "borrower_type" in cols
    assert canonical_of("geographic_region_obligor", semantics) in cols
    assert inv["ok"] is True
    assert set(inv["applied"]) >= {"borrower_type", "geographic_region_obligor"}
    assert not inv["dropped"]


def test_a_simulated_silent_drop_is_refused(df, semantics):
    """If the executor were to group by only one of two parsed dimensions, the
    fail-closed guard must refuse the query rather than answer misleadingly."""
    res = run_mi_agent_query("balance by broker by region", df, semantics)
    qr = res["query_result"]
    # Simulate a renderer/executor that dropped the second dimension by pruning
    # its group metadata + column, then re-check the invariant directly.
    import copy
    dropped_meta = dict(qr.metadata or {})
    dropped_meta["group_field_keys"] = ["broker_channel"]

    class _FakeQR:
        pass
    fake = _FakeQR()
    fake.metadata = dropped_meta
    fake.data = qr.data.drop(columns=[canonical_of("geographic_region_obligor", semantics)])
    spec = _spec_object(res)
    inv = check_dimension_invariant(spec, fake, semantics)
    assert inv.ok is False
    assert any(d["dimension"] == "geographic_region_obligor" for d in inv.dropped)


# --------------------------------------------------------------------------- #
# Filter invariant — the five required shapes.
# --------------------------------------------------------------------------- #
def _recon(res):
    qr = res.get("query_result")
    return (getattr(qr, "metadata", {}) or {}).get("reconciliation") or {} if qr else {}


def test_filter_simple_filtered_kpi(df, semantics):
    res = run_mi_agent_query("how many loans have LTV above 50%", df, semantics)
    assert res.get("ok") is True, res.get("error")
    fi = res["filter_invariant"]
    assert fi["ok"] and fi["filters_applied"]
    assert "current_loan_to_value" in fi["applied_filters"]
    assert _recon(res)["records_after_filters"] < _recon(res)["total_records"]


def test_filter_grouped_categorical(df, semantics):
    """Grouped query + categorical filter: both the grouping and the filter
    survive."""
    res = run_mi_agent_query("balance by broker for joint borrowers", df, semantics)
    assert res.get("ok") is True, res.get("error")
    fi = res["filter_invariant"]
    assert fi["ok"] and fi["filters_applied"]
    assert "borrower_type" in fi["applied_filters"]
    cols = [str(c) for c in res["query_result"].data.columns]
    assert "broker_channel" in cols  # grouping preserved
    assert _recon(res)["filters_applied"] is True


def test_filter_grouped_numeric_range(df, semantics):
    """Grouped query + range filter: filter applied to the mask before grouping."""
    res = run_mi_agent_query("balance by region where LTV between 40 and 60", df, semantics)
    assert res.get("ok") is True, res.get("error")
    fi = res["filter_invariant"]
    assert fi["ok"] and fi["filters_applied"]
    assert "current_loan_to_value" in fi["applied_filters"]
    recon = _recon(res)
    assert 0 < recon["records_after_filters"] < recon["total_records"]


def test_filter_unsupported_shape_is_refused(df, semantics):
    """A filter on a field ABSENT from the dataset is refused / surfaced, never
    silently answered with unfiltered data."""
    res = run_mi_agent_query("how many loans have Risk Grade above 700", df, semantics)
    # Refused (not answered) OR the predicate is surfaced as unavailable.
    fi = res.get("filter_invariant") or {}
    refused = res.get("ok") is False or bool(res.get("controlled_unsupported"))
    surfaced = bool(fi.get("unavailable_filters"))
    assert refused or surfaced, res.get("error")
    # Never silently returned filtered-looking data with the filter dropped.
    if res.get("ok"):
        assert not (fi.get("parsed_filters") and not fi.get("filters_applied"))


def test_no_silent_fall_through_to_unfiltered_data(df, semantics):
    """If a parsed filter is NOT applied to the mask, the workflow fails closed
    rather than returning unfiltered grouped data."""
    from mi_agent.mi_query_contract import check_filter_invariant
    from mi_agent.mi_query_spec import MIQuerySpec
    from mi_agent.mi_query_executor import execute_mi_query
    # A spec that carries a filter; simulate the executor not applying it by
    # blanking the reconciliation's applied filters, then check fail-closed.
    spec = MIQuerySpec(intent="chart", metric="current_outstanding_balance",
                       aggregation="sum", chart_type="bar",
                       dimension="geographic_region_obligor",
                       filters={"current_loan_to_value": {"op": "gt", "value": 50.0}})
    qr = execute_mi_query(spec, df, semantics)
    # Sanity: normally applied.
    assert check_filter_invariant(spec, qr, semantics).ok is True

    class _FakeQR:
        pass
    fake = _FakeQR()
    meta = dict(qr.metadata or {})
    recon = dict(meta.get("reconciliation") or {})
    recon["filters_applied"] = False
    recon["filters"] = {}
    meta["reconciliation"] = recon
    fake.metadata = meta
    fake.data = qr.data
    finv = check_filter_invariant(spec, fake, semantics)
    assert finv.ok is False
    assert any(d["filter"] == "current_loan_to_value" for d in finv.dropped)


# --------------------------------------------------------------------------- #
# Safe chart selection — two categorical dims never collapse onto a bar/line.
# --------------------------------------------------------------------------- #
def test_two_dimensions_do_not_collapse_onto_a_bar(df, semantics):
    pytest.importorskip("mi_agent_api.adapters")
    from mi_agent_api.adapters import adapt_workflow_result
    res = run_mi_agent_query("balance by broker by region", df, semantics)
    adapted = adapt_workflow_result(res, portfolio_id="client_001", as_of=None)
    chart = next((a for a in adapted["artifacts"] if a["type"] == "chart"), None)
    assert chart is not None
    # A matrix/heatmap (or a genuinely 2-axis chart) — never a single-axis bar
    # that shows only one of the two dimensions.
    if chart["chartType"] in ("bar", "line"):
        axes = {chart.get("xKey"), chart.get("yKey")}
        axes |= {s.get("key") for s in chart.get("series") or []}
        assert canonical_of("broker_channel", semantics) in axes
        assert canonical_of("geographic_region_obligor", semantics) in axes
    else:
        assert chart["chartType"] in ("heatmap", "treemap")


# --------------------------------------------------------------------------- #
# A rejected dimension is rejected WITH a reason (which the workflow surfaces
# as a user warning) — never silently absent.
# --------------------------------------------------------------------------- #
def test_extra_heatmap_dimension_is_rejected_with_a_reason(df, semantics):
    from mi_agent.mi_query_executor import execute_mi_query
    from mi_agent.mi_query_spec import MIQuerySpec
    spec = MIQuerySpec(intent="chart", metric="current_outstanding_balance",
                       aggregation="sum", chart_type="heatmap",
                       dimensions=["geographic_region_obligor", "borrower_type",
                                   "account_status"])
    res = execute_mi_query(spec, df, semantics)
    rejected = (res.metadata or {}).get("rejected_dimensions") or []
    names = {r["dimension"] for r in rejected}
    assert "account_status" in names
    assert all(r.get("reason") for r in rejected), rejected
    # group columns carry exactly the two applied dimensions.
    assert (res.metadata or {}).get("group_field_keys") == [
        "geographic_region_obligor", "borrower_type"]


# --------------------------------------------------------------------------- #
# Metric is never silently substituted with the default balance.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("query,expected_metric", [
    ("valuation by broker", "current_valuation_amount"),
    ("property value by region", "current_valuation_amount"),
    ("original balance by broker", "original_principal_balance"),
    ("original principal by region", "original_principal_balance"),
    ("balance by broker", "current_outstanding_balance"),  # default still holds
])
def test_requested_metric_resolves_to_its_own_field(df, semantics, query, expected_metric):
    from mi_agent.llm_query_parser import _deterministic_parse
    spec, _ = _deterministic_parse(query, semantics, available_columns=set(df.columns))
    assert spec.metric == expected_metric, (query, spec.metric)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def check_dimension_invariant_from_result(res, semantics):
    inv = res.get("dimension_invariant")
    assert inv is not None, "workflow must attach dimension_invariant"
    return inv


def _spec_object(res):
    """The invariant accepts the dict spec too (dict-aware ``all_group_dims``)."""
    return res.get("spec")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
