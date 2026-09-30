"""Several figures, one question — governed answer composition (P1, D4; §28).

"Is any of the pipeline overdue to complete, and how much?" was read exactly —
the case count AND the amount of the overdue pipeline — and declined because
every runtime serves one figure per plan (then, before D18, answered by the
legacy path with the whole pipeline). The design settled where this belongs
(§22.3): "two figures on one axis is answer composition, not a pipeline
branch". These tests pin the composition rule on the funded book and the
pipeline alike:

    each figure is served by its own runtime through every gate a one-figure
    question passes, the figures are the one-figure answers' figures, every
    figure is stated, the answer is all or nothing, the parts must declare the
    same data, and the coverage owner proves the composed answer as the
    conjunction of its parts.
"""
from __future__ import annotations

import ast
import inspect

import pytest

from mi_agent import plan_composition as composition
from mi_agent import plan_serving_canary as canary
from mi_agent_api.mi_service import _governed_plan_coverage
from tests.interpretation_v2 import test_funded_breadth as funded
from tests.interpretation_v2 import test_specialist_runtime_pipeline as pipeline
from tests.interpretation_v2.test_funded_breadth import (  # noqa: F401
    book, harmonised, semantics)

_OVERDUE = [{"concept": "expected_completion_timing", "comparator": "eq",
             "value": "overdue"}]


def _pipeline_intent(measures, *, filters=(), dimensions=()):
    return dict(pipeline._PIPELINE_INTENT,
                operation="breakdown" if dimensions else "point_in_time",
                measures=[{"concept": m} for m in measures],
                filters=list(filters), dimensions=list(dimensions))


def _served(intent, monkeypatch, semantics=None):
    # D21: the weighted figures rest on the served book's own history,
    # measured at test-book scale (the fixture is far below production's).
    from tests.measured_history import measured_history
    over = {"semantics": semantics} if semantics is not None else {}
    over["pipeline_history"] = measured_history("tests/fixtures/client_001_mi_pack")
    return pipeline._served(intent, monkeypatch, **over)


_REAL_SERVE = canary.serve


def _declining(monkeypatch):
    """Serve as production does (`respond`, D18): a decline is the answer."""
    monkeypatch.setattr(canary, "serve",
                        lambda **kw: _REAL_SERVE(**dict(kw, decline=True)))


def _kpis(payload):
    return {k["field"]: k["rawValue"] for a in payload["artifacts"]
            if a["type"] == "kpi" for k in a.get("kpis") or ()}


# --------------------------------------------------------------------------- #
# the pipeline: how many and how much
# --------------------------------------------------------------------------- #

def test_how_many_and_how_much_is_one_answer_with_both_figures(monkeypatch):
    """The 15:53 reading: the count and the amount of the overdue pipeline."""
    payload, record = _served(
        _pipeline_intent(["pipeline_case_count", "pipeline_amount"],
                         filters=_OVERDUE), monkeypatch)
    assert payload is not None, record.get("composition")
    assert record["serving"]["decision"] == "NEW"
    assert record["execution"]["runtime"] == "composition"
    assert [p["figure"] for p in record["composition"]["parts"]] == [
        "pipeline_case_count", "pipeline_amount"]

    # the figures are the one-figure answers' figures
    alone = {}
    for measure in ("pipeline_case_count", "pipeline_amount"):
        one, _ = _served(_pipeline_intent([measure], filters=_OVERDUE),
                         monkeypatch)
        alone[measure] = _kpis(one)[measure]
    assert _kpis(payload) == alone

    answer = payload["answer"]
    assert answer.index("case count overdue") < answer.index("amount overdue")
    ledger = _governed_plan_coverage(payload)
    assert ledger["unaccounted"] == []
    assert sorted(e["value"] for e in ledger["concepts"]
                  if e["kind"] == "governed_plan:figure") == [
        "pipeline_amount", "pipeline_case_count"]


def test_a_figure_one_part_states_is_not_restated_beside_another(monkeypatch):
    """D16 states the figures a month's headline is not; in a composed answer
    a figure another part states is not said twice."""
    payload, _ = _served(
        _pipeline_intent(["pipeline_amount", "pipeline_case_count",
                          "weighted_expected_funded_amount"],
                         filters=[dict(_OVERDUE[0], value="next_month")]),
        monkeypatch)
    assert payload is not None
    answer = payload["answer"]
    assert "At face value" not in answer
    assert "Weighted by each case's chance" not in answer
    assert set(_kpis(payload)) == {"pipeline_amount", "pipeline_case_count",
                                   "weighted_expected_funded_amount"}


def test_a_breakdown_of_two_figures_is_one_table_matched_by_member(monkeypatch):
    from mi_agent.mi_query_validator import load_mi_semantics
    sem = load_mi_semantics("mi_agent/mi_semantics_field_registry.yaml")
    payload, _ = _served(
        _pipeline_intent(["pipeline_case_count", "pipeline_amount"],
                         dimensions=["pipeline_stage"]), monkeypatch, sem)
    assert payload is not None
    tables = [a for a in payload["artifacts"] if a["type"] == "table"]
    assert len(tables) == 1
    table = tables[0]
    assert [c["label"] for c in table["columns"]] == [
        "Stage", "Pipeline case count", "Pipeline amount"]
    count_key, amount_key = (c["key"] for c in table["columns"][1:])
    for measure, key in (("pipeline_case_count", count_key),
                         ("pipeline_amount", amount_key)):
        one, _ = _served(_pipeline_intent([measure],
                                          dimensions=["pipeline_stage"]),
                         monkeypatch, sem)
        cells = {r["pipeline_stage"]: r["value"]
                 for r in one["artifacts"][0]["rows"]}
        assert {r["pipeline_stage"]: r[key] for r in table["rows"]} == cells
    assert _governed_plan_coverage(payload)["unaccounted"] == []


# --------------------------------------------------------------------------- #
# the funded book: the same rule
# --------------------------------------------------------------------------- #

def test_the_funded_book_composes_by_the_same_rule(monkeypatch, harmonised,
                                                   semantics):
    intent = funded._intent(
        geography=funded._REGION,
        measures=[{"concept": "loan", "statistic": "count"},
                  {"concept": "current_outstanding_balance"}])
    payload, record, coverage = funded._served(intent, monkeypatch, harmonised,
                                               semantics)
    assert payload is not None, record.get("composition")
    assert coverage["unaccounted"] == []
    assert "Number of loans by Region" in payload["answer"]
    assert "Balance by Region" in payload["answer"]
    tables = [a for a in payload["artifacts"] if a["type"] == "table"]
    assert len(tables) == 1
    keys = {c["key"] for c in tables[0]["columns"]}
    assert {"count", "current_outstanding_balance_sum"} <= keys


# --------------------------------------------------------------------------- #
# all or nothing; one data
# --------------------------------------------------------------------------- #

def test_one_unanswerable_figure_declines_the_whole_question(monkeypatch):
    from mi_agent import plan_pipeline_runtime as pipeline_rt
    real = pipeline_rt.check_eligibility

    def refuse_the_amount(plan):
        if composition.figures(plan) == ["pipeline_amount"]:
            return False, pipeline_rt.MEASURE_NOT_SUPPORTED, "forced"
        return real(plan)

    monkeypatch.setattr(pipeline_rt, "check_eligibility", refuse_the_amount)
    _declining(monkeypatch)
    payload, record = _served(
        _pipeline_intent(["pipeline_case_count", "pipeline_amount"],
                         filters=_OVERDUE), monkeypatch)
    assert payload is not None and payload["ok"] is False
    assert record["composition"]["failed_figure"] == "pipeline_amount"
    assert record["serving"]["decision"] == canary.SERVED_DECLINED
    assert "for the pipeline amount" in payload["answer"].lower()
    assert "£" not in payload["answer"]


def test_figures_from_different_data_are_not_put_side_by_side(monkeypatch):
    monkeypatch.setattr(composition, "aligned", lambda envelopes: False)
    _declining(monkeypatch)
    payload, record = _served(
        _pipeline_intent(["pipeline_case_count", "pipeline_amount"],
                         filters=_OVERDUE), monkeypatch)
    assert payload["ok"] is False
    assert record["serving"]["reason"] == composition.FIGURES_NOT_ALIGNED
    assert payload["metadata"]["governedDecline"]["kind"] == "failed"


def test_aligned_reads_the_data_each_part_declares():
    def envelope(as_of):
        return {"metadata": {"governedPlan": {"executed": {
            "dataset": {"as_of_date": as_of}, "measure_concept": "x"}}}}
    assert composition.aligned([envelope("2026-09-24"), envelope("2026-09-24")])
    assert not composition.aligned([envelope("2026-09-24"),
                                    envelope("2026-09-21")])


def test_a_figure_no_part_proves_is_unaccounted():
    part = {"requested": {"capability": "pipeline",
                          "measure_concept": "pipeline_case_count",
                          "population": {"base": "pipeline"}},
            "executed": {"capability": "pipeline", "population_base": "pipeline",
                         "measure_concept": "pipeline_case_count"}}
    block = {"requested": {"measure_concepts": ["pipeline_case_count",
                                                "pipeline_amount"]},
             "composed": [part]}
    ledger = _governed_plan_coverage(
        {"metadata": {"parserMode": "governed_plan", "governedPlan": block}})
    assert [e["value"] for e in ledger["unaccounted"]] == ["pipeline_amount"]


# --------------------------------------------------------------------------- #
# structure
# --------------------------------------------------------------------------- #

def test_every_part_passes_the_one_figure_gates():
    """A part is served by `_serve_plan` — the same function, and so the same
    perimeter, population proof, execution, reconciliation and rendering, as
    a one-figure question."""
    tree = ast.parse(inspect.getsource(canary._attempt_composed))
    calls = {n.func.id for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "_serve_plan" in calls
    source = inspect.getsource(canary._serve_plan)
    assert source.index("check_population_base") < source.index(
        "adapter.check_eligibility(plan)")


def test_a_runtime_that_states_several_figures_itself_is_not_split():
    plan = {"capability": "pipeline_stage_movement",
            "operation": "reconciliation",
            "outputs": [{"measures": [{"concept": "stage_opening"},
                                      {"concept": "stage_closing"}]}]}
    assert not composition.needs_composition(plan)
    assert composition.needs_composition({
        "capability": "pipeline", "operation": "point_in_time",
        "outputs": [{"measures": [{"concept": "pipeline_case_count"},
                                  {"concept": "pipeline_amount"}]}]})


def test_a_part_differs_from_the_plan_only_in_its_measure():
    plan = {"plan_id": "plan_x", "capability": "pipeline",
            "filters": _OVERDUE, "period": {"form": "current"},
            "outputs": [{"measures": [{"concept": "a"}, {"concept": "b"}],
                         "dimensions": [{"concept": "pipeline_stage"}]}]}
    parts = composition.split(plan)
    assert [composition.figures(p) for p in parts] == [["a"], ["b"]]
    for part in parts:
        assert part["filters"] == plan["filters"]
        assert part["period"] == plan["period"]
        assert part["outputs"][0]["dimensions"] == plan["outputs"][0]["dimensions"]
    assert composition.siblings(parts[0]) == ["b"]
