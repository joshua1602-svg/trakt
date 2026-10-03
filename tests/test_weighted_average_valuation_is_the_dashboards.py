"""The weighted-average valuation the agent answers is the dashboard's tile.

The production bank (2026-09-29) asked "What is the weighted average current
valuation?" and the model refused, correctly, because the governed registry did
not permit a weighted average of valuation — while the dashboard shows exactly
that figure ("Weighted avg property value", balance-weighted) and uses current
valuation as the LTV denominator.

Owner decision D10 (2026-09-29): if available, it should answer. The registry
entry now permits `weighted_avg` weighted by current balance — the tile's own
definition — so the model is shown the statistic, the compiler binds it, and
both answer paths compute it the way the tile does.
"""
from __future__ import annotations

import pandas as pd
import pytest

from mi_agent import plan_runtime_adapter as adapter
from mi_agent.interpretation_v2.compiler import CompilerContext, DeterministicCompiler
from mi_agent.interpretation_v2.intent import parse_candidate_intent
from mi_agent.interpretation_v2.vocabulary import load_governed_vocabulary
from mi_agent.mi_query_executor import execute_mi_query
from mi_agent_api import snapshots

#: Balances deliberately uneven, so a balance-weighted average and a simple one
#: differ and the test can tell them apart.
_FRAME = pd.DataFrame({
    "loan_identifier": ["L1", "L2", "L3", "L4"],
    "current_outstanding_balance": [100_000.0, 300_000.0, 50_000.0, 550_000.0],
    "current_valuation_amount": [400_000.0, 900_000.0, 250_000.0, 1_200_000.0],
})


@pytest.fixture(scope="module")
def semantics():
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent_api.data_source import semantics_path
    return load_mi_semantics(semantics_path())


def _tile() -> float:
    """The dashboard tile's own computation, on the same rows."""
    return float(snapshots._weighted_average(
        _FRAME["current_valuation_amount"], _FRAME["current_outstanding_balance"]))


def test_the_weighted_and_simple_averages_differ_on_the_fixture():
    assert _tile() != pytest.approx(float(_FRAME["current_valuation_amount"].mean()))


def test_the_model_is_shown_the_statistic():
    concept = load_governed_vocabulary().resolve("current_valuation_amount")
    assert "weighted_average" in concept.allowed_statistics


def test_the_governed_path_computes_the_tile(semantics):
    result = DeterministicCompiler(CompilerContext()).compile(parse_candidate_intent({
        "schema_version": "candidate_intent/1.0",
        "capability": "generic_analysis", "operation": "point_in_time",
        "population": {"base": "funded"},
        "measures": [{"concept": "current_valuation_amount",
                      "statistic": "weighted_average"}],
        "time": {"form": "current"}}))
    assert result.plan is not None, [r.code for r in result.reasons]
    spec = adapter.spec_for_plan(result.plan.to_dict())
    out = execute_mi_query(spec, _FRAME, semantics, validate=False)
    value = float(out.data[adapter.value_column(spec)].iloc[0])
    assert value == pytest.approx(_tile())


def test_the_legacy_path_computes_the_tile(semantics):
    from mi_agent.mi_query_spec import MIQuerySpec
    out = execute_mi_query(
        MIQuerySpec(intent="summary", metric="current_valuation_amount",
                    aggregation="weighted_avg"),
        _FRAME, semantics, validate=True, dataset="funded")
    value = float(out.data["current_valuation_amount_weighted_avg"].iloc[0])
    assert value == pytest.approx(_tile())
