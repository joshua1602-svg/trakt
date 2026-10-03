"""The query agent and the Pipeline tab report one pipeline: the live cases.

ERE's weekly extract keeps completed and withdrawn cases with their balance. The
Pipeline tab counts KFI / Application / Offer only and discloses the rest
(#505). The agent kept summing the whole extract, so on 2026-09-29's production
bank it answered "What is the pipeline amount?" with £1.18bn while the tab showed
the live pipeline — two correct sums of two different things, under one name.

Owner decision, 2026-09-29: live cases are the default, and the dashboard and
the query agent use the same pipeline. These tests pin that on both answer
paths by computing the tab's own snapshot for the same file and requiring the
agent's figure to equal it — not a number written down here, the tab's.
"""
from __future__ import annotations

import glob

import pytest

from mi_agent import plan_pipeline_runtime as pipeline_rt
from mi_agent.interpretation_v2.compiler import CompilerContext, DeterministicCompiler
from mi_agent.interpretation_v2.intent import parse_candidate_intent
from mi_agent_api import pipeline_contract as pc
from mi_agent_api.pipeline_prep import OPEN_STAGES

_EXTRACT = sorted(glob.glob("tests/fixtures/client_001_mi_pack/pipeline/*/*"))[0]
_SOURCE = {"source_file": _EXTRACT, "pipeline_as_of_date": "2025-10-01"}


@pytest.fixture(scope="module")
def semantics():
    """The governed field registry the production request loads."""
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent_api.data_source import semantics_path
    return load_mi_semantics(semantics_path())


@pytest.fixture(scope="module")
def prepared():
    return pc.load_prepared_pipeline(_EXTRACT)


@pytest.fixture(scope="module")
def tab(prepared):
    """The Pipeline tab's snapshot for the fixture extract."""
    frame, report = prepared
    return pc.compute_pipeline_snapshot(frame, report, {}, client_id="client_001",
                                        run_id="fixture", source=_SOURCE)


def _plan(**over):
    body = {"schema_version": "candidate_intent/1.0", "capability": "pipeline",
            "operation": "summary", "population": {"base": "pipeline"},
            "measures": [{"concept": "pipeline_amount"}],
            "time": {"form": "current"}}
    body.update(over)
    return DeterministicCompiler(CompilerContext()).compile(
        parse_candidate_intent(body)).plan.to_dict()


def test_the_fixture_carries_closed_cases(prepared, tab):
    """Otherwise every equality below would hold for the wrong reason."""
    frame, report = prepared
    assert tab["excludedFromOpenPipeline"]["cases"] > 0
    assert tab["pipelineAmount"] < report["total_pipeline_amount"]
    assert tab["pipelineRowCount"] < report["row_count"]


# --------------------------------------------------------------------------- #
# the governed path
# --------------------------------------------------------------------------- #

def test_the_agent_pipeline_amount_is_the_tabs_tile(tab):
    out = pipeline_rt.execute_current(_plan(), source=_SOURCE)
    assert out.ok, out.detail
    assert out.value == pytest.approx(tab["pipelineAmount"])


def test_the_agent_case_count_is_the_tabs_tile(tab):
    out = pipeline_rt.execute_current(
        _plan(operation="point_in_time", measures=[{"concept": "loan"}]),
        source=_SOURCE)
    assert out.ok, out.detail
    assert out.value == pytest.approx(tab["pipelineRowCount"])


def test_the_agent_stage_breakdown_is_the_tabs_breakdown(tab, semantics):
    by_stage = {r["stage"]: r for r in tab["stageBreakdown"]}
    assert set(by_stage) <= set(OPEN_STAGES)

    amounts = pipeline_rt.execute_current(
        _plan(operation="breakdown", dimensions=["pipeline_stage"]),
        source=_SOURCE, semantics=semantics)
    assert amounts.ok, amounts.detail
    assert {c["pipeline_stage"]: c["value"] for c in amounts.cells} == pytest.approx(
        {k: r["pipelineAmount"] for k, r in by_stage.items()})

    counts = pipeline_rt.execute_current(
        _plan(operation="breakdown", measures=[{"concept": "pipeline_case_count"}],
              dimensions=["pipeline_stage"]),
        source=_SOURCE, semantics=semantics)
    assert counts.ok, counts.detail
    assert {c["pipeline_stage"]: c["value"] for c in counts.cells} == {
        k: float(r["caseCount"]) for k, r in by_stage.items()}


def test_the_agent_states_the_tabs_exclusion(tab):
    out = pipeline_rt.execute_current(_plan(), source=_SOURCE)
    scope = out.receipt["pipeline_scope"]
    assert scope["population"] == "open"
    assert scope["excluded"] == tab["excludedFromOpenPipeline"]
    note = scope["note"]
    assert note.startswith("Live pipeline (KFI, Application, Offer)")
    assert f"excludes {tab['excludedFromOpenPipeline']['cases']:,} " in note
    for row in tab["excludedFromOpenPipeline"]["stages"]:
        assert row["stage"].title() in note


# --------------------------------------------------------------------------- #
# the legacy path — the same pipeline, through the one query function
# --------------------------------------------------------------------------- #

def _legacy(spec_kw, prepared, semantics):
    from mi_agent.mi_query_executor import execute_mi_query
    from mi_agent.mi_query_spec import MIQuerySpec
    frame, _ = prepared
    return execute_mi_query(MIQuerySpec(**spec_kw), frame, semantics,
                            validate=False, dataset="pipeline")


def _total(result, column: str) -> float:
    return float(result.data[column].sum())


def test_the_legacy_pipeline_total_is_the_tabs_tile(prepared, semantics, tab):
    result = _legacy({"intent": "chart", "chart_type": "bar",
                      "metric": "current_outstanding_balance",
                      "dimension": "pipeline_stage", "aggregation": "sum"},
                     prepared, semantics)
    assert _total(result, "current_outstanding_balance_sum") == pytest.approx(
        tab["pipelineAmount"])
    assert set(result.data["pipeline_stage"]) <= set(OPEN_STAGES)
    assert result.metadata["pipeline_scope"]["population"] == "open"
    assert result.metadata["input_row_count"] == tab["pipelineRowCount"]


def test_a_question_that_names_a_stage_reads_the_whole_extract(prepared, semantics):
    """'How many withdrawn cases?' must still find them."""
    result = _legacy({"intent": "chart", "chart_type": "bar",
                      "metric": "current_outstanding_balance",
                      "dimension": "pipeline_stage", "aggregation": "count",
                      "filters": {"pipeline_stage": "WITHDRAWN"}},
                     prepared, semantics)
    assert result.metadata["pipeline_scope"]["population"] == "extract"
    assert list(result.data["pipeline_stage"]) == ["WITHDRAWN"]
    assert _total(result, "count") == 1


def test_a_funded_query_is_not_scoped(prepared, semantics):
    from mi_agent.mi_query_executor import execute_mi_query
    from mi_agent.mi_query_spec import MIQuerySpec
    frame, _ = prepared
    result = execute_mi_query(
        MIQuerySpec(intent="chart", chart_type="bar",
                    metric="current_outstanding_balance",
                    dimension="pipeline_stage", aggregation="sum"),
        frame, semantics, validate=False, dataset="funded")
    assert result.metadata["pipeline_scope"] is None
