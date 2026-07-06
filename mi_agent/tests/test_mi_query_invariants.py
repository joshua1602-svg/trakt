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
    for kind in ("single_dim", "two_dim", "three_dim", "filter_group",
                 "top_n", "ranking", "weighted_avg", "count", "rejection"):
        assert kind in summary["by_kind"], kind


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
