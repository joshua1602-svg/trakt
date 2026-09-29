"""The pipeline's change between two dated extracts (owner decision D13,
P0 design §20).

"Compare latest pipeline with prior pipeline" and "Show pipeline growth from
October to November" are MUST ANSWER. Both were read correctly on 2026-09-29
and fell: the pipeline runtime gave the pipeline AT each date but never the
change, and normalisation sent a metric delta on a pipeline measure to the
funded-book runtime, which refused the population.

Now the two figures are read exactly as the dated shape reads them — the
weekly owner's, at the extracts D7 or the extract order names — and the change
between them is the semantic engine's one `period_change`. Every figure here is
compared with what the weekly owner itself returns.
"""
from __future__ import annotations

import pytest

from mi_agent import plan_pipeline_runtime as pipeline_rt
from mi_agent import semantic_engine as engine
from mi_agent.interpretation_v2.compiler import CompilerContext, DeterministicCompiler
from mi_agent.interpretation_v2.intent import parse_candidate_intent
from tests.interpretation_v2.test_specialist_runtime_pipeline_dated import (  # noqa: F401
    _CLIENT, _owner, _served, history)

_OCT_NOV = {"form": "explicit_period", "grain": "monthly",
            "labels": ["October 2025", "November 2025"]}


def _intent(**over):
    payload = {"schema_version": "candidate_intent/1.0",
               "capability": "generic_analysis", "operation": "movement",
               "population": {"base": "pipeline"},
               "measures": [{"concept": "pipeline_amount"}],
               "time": dict(_OCT_NOV), "change_form": "metric_delta"}
    payload.update(over)
    return payload


def _plan(**over):
    result = DeterministicCompiler(CompilerContext()).compile(
        parse_candidate_intent(_intent(**over)))
    assert result.plan is not None, [(r.code, r.detail) for r in result.reasons]
    return result.plan.to_dict()


def _run(history, **over):
    plan = _plan(**over)
    assert pipeline_rt.check_eligibility(plan) == (True, "", ""), \
        pipeline_rt.check_eligibility(plan)
    return pipeline_rt.execute_dated(plan, root=history, client_id=_CLIENT)


# --------------------------------------------------------------------------- #
# routing: the measure's owner implements the change
# --------------------------------------------------------------------------- #

def test_a_metric_delta_on_a_pipeline_measure_stays_with_the_pipeline():
    plan = _plan()
    assert (plan["capability"], plan["operation"]) == ("pipeline", "movement")


def test_latest_against_prior_as_a_level_comparison_is_the_pipelines():
    plan = _plan(capability="pipeline", operation="compare",
                 time={"form": "relative_pair", "grain": "weekly",
                       "labels": ["latest", "prior"]},
                 change_form="level_comparison")
    assert (plan["capability"], plan["operation"]) == ("pipeline", "compare")
    assert pipeline_rt.check_eligibility(plan) == (True, "", "")


def test_a_funded_metric_delta_still_goes_to_the_funded_owner():
    plan = _plan(population={"base": "funded"},
                 measures=[{"concept": "current_outstanding_balance"}])
    assert plan["capability"] == "period_movement"


# --------------------------------------------------------------------------- #
# the figures are the owner's; the change is the engine's
# --------------------------------------------------------------------------- #

def test_growth_from_october_to_november_is_the_owners_two_figures(history):
    by_date, _ = _owner(history)
    outcome = _run(history)
    assert outcome.ok, outcome.detail
    earlier = by_date["2025-10-30"]["metrics"]["pipeline_amount"]
    later = by_date["2025-11-27"]["metrics"]["pipeline_amount"]
    assert outcome.receipt["change"] == engine.period_change(earlier, later)
    assert outcome.value == engine.period_change(earlier, later)["change"]
    assert outcome.receipt["change_owner"] == "semantic_engine.period_change"
    assert [r["rule"] for r in outcome.receipt["period_resolution"]] == \
        [pipeline_rt.MONTH_RULE] * 2


def test_latest_against_the_previous_week_is_the_last_two_extracts(history):
    by_date, _ = _owner(history)
    outcome = _run(history, time={"form": "relative_pair", "grain": "weekly",
                                  "periods_back": 1})
    assert outcome.ok, outcome.detail
    assert outcome.receipt["selected_periods"] == ["2025-11-27", "2026-10-29"]
    assert outcome.receipt["change"] == engine.period_change(
        by_date["2025-11-27"]["metrics"]["pipeline_amount"],
        by_date["2026-10-29"]["metrics"]["pipeline_amount"])


def test_the_change_by_stage_is_each_stages_own(history):
    outcome = _run(history, dimensions=["pipeline_stage"])
    assert outcome.ok, outcome.detail
    dated = _run(history, operation="breakdown", change_form=None,
                 capability="pipeline", dimensions=["pipeline_stage"])
    at = {(c["period"], c["pipeline_stage"]): c["value"] for c in dated.cells}
    assert outcome.cells
    for row in outcome.cells:
        assert {k: row[k] for k in ("from", "to", "change", "change_pct")} == \
            engine.period_change(at.get(("2025-10-30", row["pipeline_stage"])),
                                 at.get(("2025-11-27", row["pipeline_stage"])))


def test_a_change_needs_exactly_two_dates():
    plan = _plan(time={"form": "explicit_period",
                       "labels": ["October 2025", "November 2025", "October 2026"]})
    assert pipeline_rt.check_eligibility(plan)[1] == pipeline_rt.PERIOD_NOT_SUPPORTED


def test_a_change_with_no_dates_names_no_pair():
    """The compiler refuses a movement over one period; the runtime refuses it
    again, should a plan ever reach it."""
    result = DeterministicCompiler(CompilerContext()).compile(
        parse_candidate_intent(_intent(capability="pipeline",
                                       time={"form": "current"}, change_form=None)))
    assert result.plan is None
    plan = _plan()
    plan["period"] = {"form": "current"}
    assert pipeline_rt.check_eligibility(plan)[1] == pipeline_rt.PERIOD_NOT_SUPPORTED


# --------------------------------------------------------------------------- #
# the answer
# --------------------------------------------------------------------------- #

def test_the_answer_states_the_change_both_figures_and_both_extracts(monkeypatch,
                                                                    history):
    from mi_agent_api.mi_service import _governed_plan_coverage

    payload, record = _served(_intent(), monkeypatch, history)
    assert payload is not None, record.get("execution")
    answer = payload["answer"]
    moved = record["execution"]["receipt"]["change"]
    verb = "rose" if moved["change"] > 0 else "fell"
    assert answer.startswith(f"The live pipeline amount {verb} by ")
    assert "the weekly extract of 2025-10-30 (October 2025)" in answer
    assert "the weekly extract of 2025-11-27 (November 2025)" in answer
    assert _governed_plan_coverage(payload)["unaccounted"] == []


def test_the_pipeline_runtime_still_computes_no_figure():
    """The change is the engine's; the runtime only chooses the extracts."""
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(pipeline_rt))
    numeric = (ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Pow)
    source = inspect.getsource(pipeline_rt._dated_change)
    assert not [n for n in ast.walk(ast.parse(source))
                if isinstance(n, ast.BinOp) and isinstance(n.op, numeric)]
    assert "period_change" in source
    assert tree is not None
