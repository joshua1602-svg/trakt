"""Catalogue batch 1: the Pipeline tab's weighted value, expected-completion
view and breakdowns, answered by the agent as the tab computes them.

The 2026-09-29 production bank refused or misread these because the agent's
catalogue had no concept for them, while the Pipeline tab showed every one:

  - "What is the weighted expected funded amount?", "... by stage / broker"
  - "How much pipeline is overdue / current month / expected next month?" (D11)
  - "What is the pipeline amount by expected completion month?"
  - "Show pipeline amount by product" (refused as "not available")
  - "Show pipeline amount by LTV bucket / band"

Each figure here is compared with the tab's own snapshot for the same file —
never a number written down in the test.
"""
from __future__ import annotations

import glob

import pytest

from mi_agent import plan_pipeline_runtime as pipeline_rt
from mi_agent.interpretation_v2.compiler import CompilerContext, DeterministicCompiler
from mi_agent.interpretation_v2.intent import parse_candidate_intent
from mi_agent.interpretation_v2.vocabulary import (
    SPECIALIST_DIMENSION_VALUES, SPECIALIST_MEASURE_DEFINITIONS, VOCABULARY_VERSION,
    load_governed_vocabulary)
from mi_agent_api import pipeline_contract as pc

_EXTRACT = sorted(glob.glob("tests/fixtures/client_001_mi_pack/pipeline/*/*"))[0]
_AS_OF = "2025-10-01"
_OVERDUE = {"schema_version": "candidate_intent/1.0", "capability": "pipeline",
            "operation": "point_in_time", "population": {"base": "pipeline"},
            "measures": [{"concept": "pipeline_amount"}],
            "time": {"form": "current"},
            "filters": [{"concept": "expected_completion_timing",
                         "comparator": "eq", "value": "overdue"}]}


@pytest.fixture(scope="module")
def semantics():
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent_api.data_source import semantics_path
    return load_mi_semantics(semantics_path())


def _source():
    return {"source_file": _EXTRACT, "pipeline_as_of_date": _AS_OF}


def _tab():
    frame, report = pc.load_prepared_pipeline(_source())
    return pc.compute_pipeline_snapshot(frame, report, {}, client_id="client_001",
                                        run_id="fixture", source=_source())


@pytest.fixture
def overdue_case(monkeypatch):
    """The October extract with one of its forecast cases expected in
    SEPTEMBER — overdue as at the extract. The agent and the tab both read this
    frame, so the comparison is still the tab's figure against the agent's."""
    frame, report = pc.load_prepared_pipeline(_source())
    frame = frame.copy()
    first = pc.forecast_rows(frame).index[0]
    frame.loc[first, "expected_completion_month"] = "2025-09"
    monkeypatch.setattr(pc, "load_prepared_pipeline",
                        lambda *a, **k: (frame, report))
    return pc.compute_pipeline_snapshot(frame, report, {}, client_id="client_001",
                                        run_id="fixture", source=_source())


def _plan(**over):
    body = dict(_OVERDUE)
    body.pop("filters")
    body.update(over)
    result = DeterministicCompiler(CompilerContext()).compile(
        parse_candidate_intent(body))
    assert result.plan is not None, [r.code for r in result.reasons]
    return result.plan.to_dict()


def _run(plan, semantics):
    ok, why, detail = pipeline_rt.check_eligibility(plan)
    assert ok, (why, detail)
    out = pipeline_rt.execute_current(plan, source=_source(),
                                      semantics=semantics)
    assert out.ok, (out.reason, out.detail)
    return out


_WEIGHTED = [{"concept": "weighted_expected_funded_amount"}]
_TIMING = lambda v: [{"concept": "expected_completion_timing",  # noqa: E731
                      "comparator": "eq", "value": v}]


# --------------------------------------------------------------------------- #
# what the model is shown
# --------------------------------------------------------------------------- #

def test_the_model_is_shown_the_three_concepts():
    vocabulary = load_governed_vocabulary()
    weighted = vocabulary.resolve("weighted_expected_funded_amount")
    assert weighted.owning_capability == "pipeline"
    assert "NOT the pipeline amount" in SPECIALIST_MEASURE_DEFINITIONS[
        "weighted_expected_funded_amount"]
    timing = vocabulary.resolve("expected_completion_timing")
    assert timing.values == ("overdue", "current_month", "next_month")
    assert "arrears" in timing.description        # D11: loans are another question
    assert vocabulary.resolve("expected_completion_month").role == "dimension"
    assert SPECIALIST_DIMENSION_VALUES["expected_completion_timing"] == \
        pipeline_rt.TIMING_VALUES
    major, minor, _ = (int(x) for x in VOCABULARY_VERSION.split("."))
    assert (major, minor) >= (2, 6)


# --------------------------------------------------------------------------- #
# every figure is the tab's
# --------------------------------------------------------------------------- #

def test_the_weighted_pipeline_is_the_tabs_and_the_forecasts_pipeline_part(semantics):
    out = _run(_plan(measures=_WEIGHTED), semantics)
    tab = _tab()
    _frame, report = pc.load_prepared_pipeline(_EXTRACT)
    assert out.value == pytest.approx(tab["weightedExpectedFundedAmount"])
    # One weighted figure, not two: the forecast bridge reads the same total.
    assert out.value == pytest.approx(report["weighted_expected_funded_amount"])


def test_the_weighted_pipeline_by_stage_is_the_tabs(semantics):
    out = _run(_plan(operation="breakdown", measures=_WEIGHTED,
                     dimensions=["pipeline_stage"]), semantics)
    assert {c["pipeline_stage"]: c["value"] for c in out.cells} == pytest.approx(
        {r["stage"]: r["weightedExpectedFundedAmount"]
         for r in _tab()["stageBreakdown"]})


@pytest.mark.parametrize("dimension,tab_key", [
    ("broker_channel", "brokerBreakdownFull"),
    ("erm_product_type", "productBreakdownFull"),
    ("ltv_bucket", "ltvBreakdown")])
def test_a_breakdown_is_the_tabs(dimension, tab_key, semantics):
    out = _run(_plan(operation="breakdown", dimensions=[dimension]), semantics)
    tab = {r["key"]: r["pipelineAmount"] for r in _tab()[tab_key]}
    assert tab, "the fixture must carry the breakdown"
    assert {c[dimension]: c["value"] for c in out.cells} == pytest.approx(tab)
    assert out.receipt["execution_owner"] == pipeline_rt.OWNER_TAB_BREAKDOWN


@pytest.mark.parametrize("measure,tab_key", [
    ([{"concept": "pipeline_amount"}], "expectedFundedAmount"),
    ([{"concept": "pipeline_case_count"}], "caseCount"),
    (_WEIGHTED, "weightedExpectedFundedAmount")])
def test_by_expected_completion_month_is_the_tabs_chart(measure, tab_key, semantics):
    out = _run(_plan(operation="breakdown", measures=measure,
                     dimensions=["expected_completion_month"]), semantics)
    tab = {r["month"]: r[tab_key] for r in _tab()["expectedCompletionBreakdown"]}
    assert tab
    assert {c["expected_completion_month"]: c["value"] for c in out.cells} == \
        pytest.approx(tab)
    assert out.receipt["completion_basis"] == pipeline_rt.COMPLETION_BASIS


@pytest.mark.parametrize("timing,measure,tab_key", [
    ("overdue", [{"concept": "pipeline_amount"}], "overdueExpectedCompletionAmount"),
    ("overdue", [{"concept": "pipeline_case_count"}], "overdueExpectedCompletionCount"),
    ("overdue", _WEIGHTED, "overdueExpectedCompletionWeightedAmount"),
    ("current_month", [{"concept": "pipeline_amount"}],
     "currentMonthExpectedCompletionAmount"),
    ("current_month", [{"concept": "pipeline_case_count"}],
     "currentMonthExpectedCompletionCount"),
    ("next_month", _WEIGHTED, "nextExpectedCompletionWeightedAmount")])
def test_overdue_this_month_and_next_are_the_tabs(timing, measure, tab_key,
                                                  semantics, overdue_case):
    out = _run(_plan(measures=measure, filters=_TIMING(timing)), semantics)
    assert out.value == pytest.approx(float(
        overdue_case["expectedCompletionSummary"][tab_key]))
    assert out.receipt["applied_predicates"] == [
        {"field": "expected_completion_timing", "op": "eq", "values": [timing]}]


def test_the_fixture_makes_overdue_and_this_month_real_figures(semantics,
                                                               overdue_case):
    summary = overdue_case["expectedCompletionSummary"]
    assert summary["overdueExpectedCompletionCount"] >= 1
    assert summary["currentMonthExpectedCompletionCount"] >= 1


# --------------------------------------------------------------------------- #
# refused rather than approximated
# --------------------------------------------------------------------------- #

def test_an_ungoverned_timing_value_is_refused():
    ok, why, _ = pipeline_rt.check_eligibility(_plan(filters=_TIMING("soonish")))
    assert (ok, why) == (False, pipeline_rt.FILTERS_NOT_SUPPORTED)


def test_timing_with_a_breakdown_is_refused():
    ok, why, _ = pipeline_rt.check_eligibility(_plan(
        operation="breakdown", dimensions=["broker_channel"],
        filters=_TIMING("overdue")))
    assert (ok, why) == (False, pipeline_rt.FILTERS_NOT_SUPPORTED)


def test_any_other_filter_is_still_refused():
    ok, why, _ = pipeline_rt.check_eligibility(_plan(filters=[
        {"concept": "pipeline_stage", "comparator": "eq", "value": "OFFER"}]))
    assert (ok, why) == (False, pipeline_rt.FILTERS_NOT_SUPPORTED)


def test_region_is_not_the_tabs_obligor_region():
    """"By region" is the reporting taxonomy; the tab's region is the raw
    obligor field. Refused, not substituted."""
    plan = _plan(operation="breakdown",
                 geography={"requested": True, "group_by": True,
                            "level": "reporting"})
    ok, why, _ = pipeline_rt.check_eligibility(plan)
    assert (ok, why) == (False, pipeline_rt.GEOGRAPHY_NOT_SUPPORTED)
    assert "region" not in " ".join(pipeline_rt.TAB_COLUMN)


def test_the_weekly_history_breaks_down_by_stage_only():
    plan = _plan(operation="breakdown", dimensions=["broker_channel"],
                 time={"form": "series", "grain": "weekly"})
    ok, why, _ = pipeline_rt.check_eligibility(plan)
    assert (ok, why) == (False, pipeline_rt.DIMENSION_NOT_SUPPORTED)


def test_a_missing_column_is_refused_not_invented(semantics, monkeypatch):
    frame, report = pc.load_prepared_pipeline(_EXTRACT)
    stripped = frame.drop(columns=["product_type"])
    monkeypatch.setattr(pc, "load_prepared_pipeline",
                        lambda *a, **k: (stripped, report))
    plan = _plan(operation="breakdown", dimensions=["erm_product_type"])
    out = pipeline_rt.execute_current(plan, source=_source(), semantics=semantics)
    assert (out.ok, out.reason) == (False, pipeline_rt.FIELD_UNAVAILABLE)


# --------------------------------------------------------------------------- #
# the answer
# --------------------------------------------------------------------------- #

def test_the_overdue_answer_states_the_bucket_and_its_basis(monkeypatch,
                                                           overdue_case):
    from mi_agent_api.mi_service import _governed_plan_coverage
    from tests.interpretation_v2.test_specialist_runtime_pipeline import _served

    payload, record = _served(_OVERDUE, monkeypatch, pipeline_source=_source())
    assert payload is not None, record.get("execution")
    answer = payload["answer"]
    assert answer.startswith("The live pipeline amount overdue (expected to "
                             "complete before 2025-10) is £")
    assert "carrying a completion forecast" in answer
    assert _governed_plan_coverage(payload)["unaccounted"] == []


def test_a_broker_breakdown_names_the_leaders(monkeypatch):
    from tests.interpretation_v2.test_specialist_runtime_pipeline import _served

    intent = dict(_OVERDUE, operation="breakdown", dimensions=["broker_channel"])
    intent.pop("filters")
    payload, record = _served(intent, monkeypatch, pipeline_source=_source())
    assert payload is not None, record.get("execution")
    leader = _tab()["brokerBreakdownFull"][0]["key"]
    assert payload["answer"].startswith(f"The live pipeline amount by broker: {leader}")
