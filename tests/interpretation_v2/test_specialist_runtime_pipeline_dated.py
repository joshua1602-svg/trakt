"""The pipeline AT named dates — P0 Change 3, step 1, with D7.

Offline. The weekly history is the committed five-extract fixture re-dated into
a temporary root, so it spans two months of 2025 and one of 2026:

    2025-10-02  2025-10-30  2025-11-06  2025-11-27  2026-10-29

which is what D7 needs: two extracts in each 2025 month (so "last in the month"
is a real choice), and an October in two different years (so a bare "October"
is genuinely ambiguous). Every figure asserted is compared with what the weekly
owner itself returns for the same extract.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from mi_agent import plan_pipeline_runtime as pipeline_rt
from mi_agent.interpretation_v2.compiler import CompilerContext, DeterministicCompiler
from mi_agent.interpretation_v2.intent import parse_candidate_intent
from tests.interpretation_v2.test_specialist_runtime_pipeline import _Scripted

_CLIENT = "client_001"
_DATES = ("2025-10-02", "2025-10-30", "2025-11-06", "2025-11-27", "2026-10-29")


@pytest.fixture(scope="module")
def history(tmp_path_factory):
    root = tmp_path_factory.mktemp("pipeline_history_dated")
    sources = sorted(Path("tests/fixtures/pipeline_history_5w").glob("*/*.csv"))
    assert len(sources) == len(_DATES)
    for date, source in zip(_DATES, sources):
        folder = root / date
        folder.mkdir()
        shutil.copy(source, folder / f"M2L_KFI_and_Pipeline_{date.replace('-', '_')}.csv")
    return str(root)


def _owner(history, model=None):
    from mi_agent_api import evolution
    series = evolution.pipeline_evolution(history, _CLIENT, None,
                                          historical_model=model)
    return {p["extract_date"]: p for p in series["periods"]}, series


def _intent(**over):
    payload = {"schema_version": "candidate_intent/1.0", "capability": "pipeline",
               "operation": "point_in_time", "population": {"base": "pipeline"},
               "measures": [{"concept": "pipeline_amount"}],
               "time": {"form": "explicit_period",
                        "labels": ["October 2025", "November 2025"]}}
    payload.update(over)
    return payload


def _plan(**over):
    result = DeterministicCompiler(CompilerContext()).compile(
        parse_candidate_intent(_intent(**over)))
    assert result.plan is not None, [r.code for r in result.reasons]
    return result.plan.to_dict()


def _run(history, **over):
    plan = _plan(**over)
    assert pipeline_rt.check_eligibility(plan) == (True, "", ""), \
        pipeline_rt.check_eligibility(plan)
    return pipeline_rt.execute_dated(plan, root=history, client_id=_CLIENT)


# --------------------------------------------------------------------------- #
# D7 — a named month is the last weekly extract in it
# --------------------------------------------------------------------------- #

def test_a_named_month_is_its_last_weekly_extract(history):
    outcome = _run(history)
    assert outcome.ok, outcome.detail
    assert outcome.receipt["selected_periods"] == ["2025-10-30", "2025-11-27"]
    assert [r["rule"] for r in outcome.receipt["period_resolution"]] == \
        [pipeline_rt.MONTH_RULE] * 2


def test_the_figures_are_the_weekly_owners_own(history):
    by_date, _ = _owner(history)
    outcome = _run(history)
    assert [c["value"] for c in outcome.cells] == [
        by_date[d]["metrics"]["pipeline_amount"] for d in ("2025-10-30", "2025-11-27")]
    counted = _run(history, measures=[{"concept": "pipeline_case_count"}])
    assert [c["value"] for c in counted.cells] == [
        float(by_date[d]["metrics"]["pipeline_case_count"])
        for d in ("2025-10-30", "2025-11-27")]


def test_no_change_between_the_dates_is_computed(history):
    """The plan asked for the pipeline AT two dates; a movement is another op."""
    receipt = _run(history).receipt
    assert not {"change", "delta", "absolute_delta", "difference"} & set(receipt)
    assert receipt["execution_owner"] == pipeline_rt.OWNER_EVOLUTION


def test_a_bare_month_in_one_year_resolves(history):
    outcome = _run(history, time={"form": "explicit_period", "labels": ["November"]})
    assert outcome.ok and outcome.receipt["selected_periods"] == ["2025-11-27"]


def test_a_bare_month_held_in_two_years_is_ambiguous(history):
    outcome = _run(history, time={"form": "explicit_period", "labels": ["October"]})
    assert outcome.reason == pipeline_rt.PERIOD_LABEL_AMBIGUOUS
    assert "2025" in outcome.detail and "2026" in outcome.detail


def test_a_month_the_history_does_not_hold_is_refused(history):
    outcome = _run(history, time={"form": "explicit_period",
                                  "labels": ["March 2025"]})
    assert outcome.reason == pipeline_rt.PERIOD_NOT_AVAILABLE


def test_the_month_rule_unit():
    dates = list(_DATES)
    assert pipeline_rt.last_extract_in_month(dates, month=10, year=2025)[0] == \
        "2025-10-30"
    assert pipeline_rt.last_extract_in_month(dates, month=10, year=2026)[0] == \
        "2026-10-29"
    assert pipeline_rt.last_extract_in_month(dates, month=10, year=None)[1] == \
        pipeline_rt.PERIOD_LABEL_AMBIGUOUS


# --------------------------------------------------------------------------- #
# latest against previous
# --------------------------------------------------------------------------- #

def test_latest_against_the_previous_week(history):
    outcome = _run(history, time={"form": "relative_pair", "grain": "weekly",
                                  "periods_back": 1})
    assert outcome.ok, outcome.detail
    assert outcome.receipt["selected_periods"] == ["2025-11-27", "2026-10-29"]


def test_latest_month_against_a_year_earlier_uses_d7_on_both(history):
    outcome = _run(history, time={"form": "relative_pair", "grain": "monthly",
                                  "periods_back": 12})
    assert outcome.ok, outcome.detail
    assert outcome.receipt["selected_periods"] == ["2025-10-30", "2026-10-29"]


def test_a_month_back_that_the_history_lacks_is_refused(history):
    outcome = _run(history, time={"form": "relative_pair", "grain": "monthly",
                                  "periods_back": 1})
    assert outcome.reason == pipeline_rt.PERIOD_NOT_AVAILABLE


def test_a_pair_with_no_grain_is_read_on_the_pipelines_own_extracts():
    """It was refused; [82] "latest against prior" (a must-answer) arrived in
    exactly this shape on the 2026-09-30 full bank. The pipeline's periods are
    its weekly extracts, so 'previous' with no grain is the previous extract
    (`test_pipeline_change.py` pins the figures and the stated rule)."""
    plan = _plan(time={"form": "relative_pair", "periods_back": 1})
    plan["period"]["grain"] = None
    assert pipeline_rt.check_eligibility(plan) == (True, "", "")


# --------------------------------------------------------------------------- #
# by stage
# --------------------------------------------------------------------------- #

def test_by_stage_at_named_months(history):
    _, series = _owner(history)
    outcome = _run(history, operation="breakdown", dimensions=["pipeline_stage"])
    assert outcome.ok, outcome.detail
    assert outcome.receipt["group_field_keys"] == ["pipeline_stage"]
    assert {c["period"] for c in outcome.cells} == {"2025-10-30", "2025-11-27"}
    # THE LIVE STAGES ONLY (owner decision 2026-09-29): the owner's figures,
    # untouched, for KFI / Application / Offer; the closed stages are named on
    # the receipt, not charted as pipeline.
    from mi_agent_api.pipeline_prep import OPEN_STAGES
    expected = {(str(r["period"]), str(r["stage"])): r["value"]
                for r in series["byStage"]
                if r["period"] in ("2025-10-30", "2025-11-27")
                and str(r["stage"]).upper() in OPEN_STAGES}
    assert {(c["period"], c["pipeline_stage"]): c["value"]
            for c in outcome.cells} == expected
    closed = {str(r["stage"]) for r in series["byStage"]
              if r["period"] in ("2025-10-30", "2025-11-27")
              and str(r["stage"]).upper() not in OPEN_STAGES}
    assert closed, "the fixture must carry a closed stage for this to prove anything"
    scope = outcome.receipt["pipeline_scope"]
    assert scope["population"] == "open"
    assert set(scope["excluded_stages"]) == closed
    assert "not counted" in scope["note"]


# --------------------------------------------------------------------------- #
# through serve()
# --------------------------------------------------------------------------- #

def _served(payload, monkeypatch, history):
    from mi_agent import plan_serving_canary as canary
    from mi_agent import plan_shadow_evidence as evidence
    from mi_agent import plan_shadow_wiring as wiring

    monkeypatch.setenv("MI_AGENT_PLAN_SERVE", "canary")
    monkeypatch.setenv("MI_AGENT_PLAN_SERVE_PRINCIPALS", "canary-principal")
    wiring.set_interpreter_factory(lambda: _Scripted(payload))
    written = []
    monkeypatch.setattr(evidence, "write", lambda body: written.append(body))

    class Principal:
        actor_id = "canary-principal"

    try:
        out = canary.serve(question="q", context=Principal(), client_id=_CLIENT,
                           run_id=None, legacy_result={"ok": True}, frame=None,
                           semantics={}, view="funded", pipeline_root=history,
                           pipeline_client_id=_CLIENT)
    finally:
        wiring.set_interpreter_factory(None)
    return out, (written[-1] if written else {})


def test_serve_answers_the_pipeline_at_two_named_months(monkeypatch, history):
    from mi_agent_api.mi_service import _governed_plan_coverage

    payload, record = _served(_intent(), monkeypatch, history)
    assert payload is not None, record.get("execution")
    assert record["execution"]["runtime"] == "pipeline_dated"
    answer = payload["answer"]
    # D4: the measure, and which pipeline it is, lead the sentence.
    assert answer.startswith("The live pipeline amount")
    assert any(n["field"] == "population" for n in payload["sourceNotes"])
    assert "2025-10-30" in answer and "2025-11-27" in answer
    assert any(pipeline_rt.MONTH_RULE in n["note"] for n in payload["sourceNotes"])
    assert _governed_plan_coverage(payload)["unaccounted"] == []


def test_serve_refuses_an_ambiguous_month_and_legacy_serves(monkeypatch, history):
    payload, record = _served(
        _intent(time={"form": "explicit_period", "labels": ["October"]}),
        monkeypatch, history)
    assert payload is None
    assert record["serving"]["reason"].endswith(pipeline_rt.PERIOD_LABEL_AMBIGUOUS)


# --------------------------------------------------------------------------- #
# a series BY MONTH is D7's extract for each month, never the weekly series
# --------------------------------------------------------------------------- #
#
# The 2026-09-29 production bank asked "Show pipeline amount evolution by month"
# and was served all 90 weekly extracts: the series path never read the grain.

_MONTHLY = ["2025-10-30", "2025-11-27", "2026-10-29"]


def _series(history, grain, operation="series", model=None, **over):
    plan = _plan(operation=operation,
                 time={"form": "series", "grain": grain}, **over)
    assert pipeline_rt.check_eligibility(plan) == (True, "", ""), \
        pipeline_rt.check_eligibility(plan)
    return pipeline_rt.execute_temporal(plan, root=history, client_id=_CLIENT,
                                        history_model=model)


def test_a_monthly_series_is_the_last_weekly_extract_of_each_month(history):
    by_date, _ = _owner(history)
    outcome = _series(history, "monthly")
    assert outcome.ok, outcome.detail
    assert [c["period"] for c in outcome.cells] == _MONTHLY
    assert [c["value"] for c in outcome.cells] == [
        by_date[d]["metrics"]["pipeline_amount"] for d in _MONTHLY]
    assert outcome.receipt["grain"] == "monthly"
    assert outcome.receipt["month_rule"] == pipeline_rt.MONTH_RULE
    assert outcome.receipt["selected_periods"] == _MONTHLY


def test_a_weekly_series_is_every_extract(history):
    outcome = _series(history, "weekly")
    assert [c["period"] for c in outcome.cells] == list(_DATES)
    assert outcome.receipt["grain"] == "weekly"
    assert "month_rule" not in outcome.receipt


def test_a_monthly_stage_series_keeps_only_those_extracts(history):
    outcome = _series(history, "monthly", operation="breakdown",
                      dimensions=["pipeline_stage"])
    assert outcome.ok, outcome.detail
    assert sorted({c["period"] for c in outcome.cells}) == _MONTHLY


def test_the_weighted_series_is_the_evolution_charts_weighted_line(history):
    """Catalogue batch 1: "weighted expected funded amount by month" is the
    weekly owner's own weighted figure for each month's extract — weighted by
    the book's own history, measured at test-book scale (D21)."""
    from tests.measured_history import measured_history
    model = measured_history(str(history), _CLIENT)
    by_date, _ = _owner(history, model)
    outcome = _series(history, "monthly", model=model,
                      measures=[{"concept": "weighted_expected_funded_amount"}])
    assert outcome.ok, outcome.detail
    assert [c["value"] for c in outcome.cells] == [
        by_date[d]["metrics"]["weighted_expected_funded_amount"] for d in _MONTHLY]
    assert all(c["value"] is not None for c in outcome.cells)


def test_the_weighted_value_by_stage_over_time_is_refused(history):
    plan = _plan(operation="breakdown", dimensions=["pipeline_stage"],
                 measures=[{"concept": "weighted_expected_funded_amount"}],
                 time={"form": "series", "grain": "weekly"})
    ok, why, _ = pipeline_rt.check_eligibility(plan)
    assert (ok, why) == (False, pipeline_rt.MEASURE_NOT_SUPPORTED)


def test_a_grain_the_weekly_history_cannot_state_is_refused(history):
    plan = _plan(operation="series", time={"form": "series", "grain": "quarterly"})
    ok, why, _ = pipeline_rt.check_eligibility(plan)
    assert (ok, why) == (False, pipeline_rt.PERIOD_NOT_SUPPORTED)


def test_the_monthly_answer_says_it_is_monthly(monkeypatch, history):
    payload, record = _served(
        _intent(operation="series", time={"form": "series", "grain": "monthly"}),
        monkeypatch, history)
    assert payload is not None, record.get("execution")
    assert "3 months, at the last weekly extract of each" in payload["answer"]
    assert any(n["field"] == "grain: monthly" for n in payload["sourceNotes"])



def test_the_weighted_series_is_not_stated_without_measured_rates(history):
    """D21: read with the production thresholds, the book's few cases measure
    no stage rate, so each extract's weighted amount is not stated — never 0."""
    by_date, _ = _owner(history)
    assert all(p["metrics"]["weighted_expected_funded_amount"] is None
               for p in by_date.values())


def test_by_stage_at_named_months_states_every_figure(monkeypatch, history):
    """2026-10-01 full bank [80]: "Show pipeline by stage for October and
    November" was answered with the stage NAMES only — the figures were in the
    table. The sentence states each stage's figure at each date."""
    payload, record = _served(_intent(operation="breakdown",
                                      dimensions=["pipeline_stage"]),
                              monkeypatch, history)
    assert payload is not None, record.get("execution")
    from mi_agent.plan_serving_canary import _stage_name

    answer = payload["answer"]
    for row in payload["artifacts"][0]["rows"]:
        stated = answer.split(f"at {row['period']}: ")[1].split(";")[0]
        for stage in (k for k in row if k != "period"):
            assert f"{_stage_name(stage)} £" in stated, (stage, stated)
