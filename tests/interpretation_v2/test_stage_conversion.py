"""Conversion and pull-through (D2a, P0 design §20) — the stage-movement
capability's measured rates, declared in its semantic model and served by the
one engine every declared figure uses.

Owner decision 2026-09-29: the pipeline conversion rate and the offer-to-
completion pull-through are MUST ANSWER. The owner already computed both — the
pipeline's case history, tracked case by case across the weekly extracts, is
what the Pipeline tab's conversion card and the forecast's stage rates read —
so these tests hold the agent to that owner's own figures, on the fixture
histories, and never to a number written here.
"""
from __future__ import annotations

import ast
import inspect

import pytest

from mi_agent import plan_stage_movement_runtime as stage_rt
from mi_agent import semantic_engine as engine
from mi_agent import semantic_model
from mi_agent.interpretation_v2.compiler import CompilerContext, DeterministicCompiler
from mi_agent.interpretation_v2.intent import parse_candidate_intent
from tests.interpretation_v2.test_specialist_runtime_pipeline import _Scripted

_CLIENT = "client_001"
_HISTORIES = ("tests/fixtures/pipeline_history_5w",
              "tests/fixtures/client_001_mi_pack")
_MODEL = semantic_model.load("pipeline_stage_movement")


@pytest.fixture(scope="module", params=_HISTORIES)
def history(request):
    """The owner's own case history for a fixture — the model the production
    request builds (`pipeline_contract.build_pipeline_history`)."""
    from mi_agent_api import pipeline_contract
    return pipeline_contract.build_pipeline_history(request.param, _CLIENT)


def _intent(measure, **over):
    payload = {"schema_version": "candidate_intent/1.0",
               "capability": "pipeline_stage_movement",
               "operation": "point_in_time", "population": {"base": "pipeline"},
               "measures": [{"concept": measure}], "time": {"form": "current"}}
    payload.update(over)
    return payload


def _plan(measure, **over):
    result = DeterministicCompiler(CompilerContext()).compile(
        parse_candidate_intent(_intent(measure, **over)))
    assert result.plan is not None, [(r.code, r.detail) for r in result.reasons]
    return result.plan.to_dict()


def _stage(axis, value):
    return [{"concept": axis, "comparator": "eq", "value": value}]


def _run(plan, history):
    ok, why, detail = stage_rt.check_eligibility(plan)
    assert ok, (why, detail)
    outcome = stage_rt.execute(plan, root=None, client_id=_CLIENT,
                               history_model=history)
    assert outcome.ok, (outcome.reason, outcome.detail)
    return outcome


# --------------------------------------------------------------------------- #
# the model file
# --------------------------------------------------------------------------- #

def test_every_declared_path_resolves_in_the_owners_output(history):
    for m in _MODEL.measures.values():
        if m.value:
            assert semantic_model.has(history, m.value), (m.name, m.value)
        for binding in m.by.values():
            for spec in (binding.get("members") or {}).values():
                assert semantic_model.has(history, spec["value"]), spec
            if "map" in binding:
                assert semantic_model.has(history, binding["map"]), binding
    view = _MODEL.views["case_history"]
    assert semantic_model.has(history, view.inputs["case_history"]["as_of"])


def test_the_rates_are_the_vocabularys_and_owned_by_stage_movement():
    from mi_agent.interpretation_v2.vocabulary import load_governed_vocabulary
    vocabulary = load_governed_vocabulary()
    for name in ("cohort_conversion", "stage_pull_through", "stage_completion_rate"):
        concept = vocabulary.concepts[name]
        assert concept.owning_capability == "pipeline_stage_movement"
        assert concept.description == _MODEL.measure(name).definition
    assert "NOT" in vocabulary.concepts["cohort_conversion"].description


# --------------------------------------------------------------------------- #
# every figure is the owner's
# --------------------------------------------------------------------------- #

def test_the_conversion_rate_is_the_owners_cohort_conversion(history):
    out = _run(_plan("cohort_conversion"), history)
    assert out.value == history["cumulativeCohortConversion"]
    assert out.receipt["inputs"]["case_history"]["as_of"] == \
        history["observationWindowEnd"]
    assert out.receipt["cohort_count"] == history["cohortProgression"]["cohortSize"]


def test_the_funnel_is_the_owners_latest_week(history):
    out = _run(_plan("cohort_conversion", operation="breakdown",
                     dimensions=["destination_stage"]), history)
    latest = history["cohortProgression"]["latest"]
    assert {c["destination_stage"]: c["value"] for c in out.cells} == {
        s: latest[s] for s in ("APPLICATION", "OFFER", "COMPLETED")}


def test_offer_to_completion_pull_through_is_the_run_off_models(history):
    out = _run(_plan("stage_pull_through",
                     filters=_stage("origin_stage", "OFFER")), history)
    stage = history["runoff"]["stages"]["OFFER"]
    assert out.value == stage["pullThrough"]
    assert out.receipt["provisional"] is (not stage["sufficient"])
    # D26: the lapsed cases (open past the stage's window) are stated beside
    # the recorded withdrawals — both are fall-outs of the pull-through.
    assert out.receipt["member_evidence"] == {"advanced": stage["advanced"],
                                              "fellOut": stage["fellOut"],
                                              "lapsed": stage["lapsed"]}


def test_the_assumed_kfi_to_completion_rate_is_the_forecasts_stage_rate(history):
    rate = history["historicalCompletionRateByStage"]["KFI"]["rate"]
    plan = _plan("stage_completion_rate", filters=_stage("origin_stage", "KFI"))
    if rate is None:
        # D27 with D21: no KFI was ever seen leaving the stage in this book,
        # so its way to completion is not measured — and the decline says so.
        out = stage_rt.execute(plan, root=None, client_id=_CLIENT,
                               history_model=history)
        assert not out.ok and out.reason == engine.FIGURE_WITHHELD
        assert "does not yet measure every step" in out.detail
        return
    out = _run(plan, history)
    assert out.value == rate
    # D27: the forecast weights no KFI, and the answer's owner says so.
    assert out.receipt["member_notes"] == [
        history["historicalCompletionRateByStage"]["KFI"]["note"]]


def test_pull_through_by_stage_names_the_stages_the_owner_flagged(history):
    out = _run(_plan("stage_pull_through", operation="breakdown",
                     dimensions=["origin_stage"]), history)
    stages = history["runoff"]["stages"]
    assert {c["origin_stage"]: c["value"] for c in out.cells} == {
        s: v["pullThrough"] for s, v in stages.items()}
    assert out.receipt["provisional_members"] == [
        s for s, v in stages.items() if not v["sufficient"]]


def test_a_rate_the_owner_did_not_publish_is_refused_not_invented(history):
    plan = _plan("stage_pull_through", filters=_stage("origin_stage", "WITHDRAWN"))
    out = stage_rt.execute(plan, root=None, client_id=_CLIENT, history_model=history)
    assert not out.ok and out.reason == engine.FIELD_UNAVAILABLE


def test_no_case_history_is_refused_by_name():
    plan = _plan("cohort_conversion")
    out = stage_rt.execute(plan, root=None, client_id=_CLIENT, history_model=None)
    assert not out.ok and out.reason == stage_rt.SOURCE_UNAVAILABLE


def test_a_pull_through_with_no_stage_asks_for_one():
    ok, why, _ = stage_rt.check_eligibility(_plan("stage_pull_through"))
    assert (ok, why) == (False, engine.DIMENSION_NOT_SUPPORTED)


# --------------------------------------------------------------------------- #
# the answer
# --------------------------------------------------------------------------- #

def _served(intent, monkeypatch, history):
    from mi_agent import plan_serving_canary as canary
    from mi_agent import plan_shadow_evidence as evidence
    from mi_agent import plan_shadow_wiring as wiring
    from mi_agent_api.mi_service import _governed_plan_coverage

    monkeypatch.setenv("MI_AGENT_PLAN_SERVE", "canary")
    monkeypatch.setenv("MI_AGENT_PLAN_SERVE_PRINCIPALS", "canary-principal")
    wiring.set_interpreter_factory(lambda: _Scripted(intent))
    written = []
    monkeypatch.setattr(evidence, "write", lambda body: written.append(body))

    class Principal:
        actor_id = "canary-principal"

    try:
        payload = canary.serve(question="q", context=Principal(), client_id=_CLIENT,
                               run_id=None, legacy_result={"ok": True}, frame=None,
                               semantics={}, view="funded",
                               pipeline_root="tests/fixtures/pipeline_history_5w",
                               pipeline_client_id=_CLIENT, pipeline_history=history)
    finally:
        wiring.set_interpreter_factory(None)
    record = written[-1] if written else {}
    assert payload is not None, record.get("execution")
    assert _governed_plan_coverage(payload)["unaccounted"] == []
    return payload


def test_the_conversion_answer_states_the_rate_the_cohort_and_the_date(
        monkeypatch, history):
    payload = _served(_intent("cohort_conversion"), monkeypatch, history)
    answer = payload["answer"]
    assert answer.startswith(
        f"Cumulative cohort conversion: {history['cumulativeCohortConversion']:.1f}% — "
        f"of the {history['cohortProgression']['cohortSize']} KFI cases tracked")
    assert f"As at pipeline history {history['observationWindowEnd']}." in answer


def test_the_pull_through_answer_names_the_stage_and_its_evidence(monkeypatch,
                                                                 history):
    payload = _served(_intent("stage_pull_through",
                              filters=_stage("origin_stage", "OFFER")),
                      monkeypatch, history)
    stage = history["runoff"]["stages"]["OFFER"]
    answer = payload["answer"]
    assert answer.startswith(
        f"Stage pull-through (from stage: Offer): {stage['pullThrough'] * 100:.1f}%.")
    if not stage["sufficient"]:
        assert "Provisional: the owner measured it on too few cases" in answer
    assert f"advanced {stage['advanced']}, fell out {stage['fellOut']}" in answer


def test_the_kfi_completion_rate_says_the_forecast_weights_no_kfi(monkeypatch,
                                                                history):
    """D27 (owner 2026-10-01): asked what the forecast assumes from KFI, the
    answer was the KFI rate alone; the forecast weights no KFI, and now says
    so. The count so far is the evidence, not the rate."""
    rates = history["historicalCompletionRateByStage"]
    if rates["KFI"]["rate"] is None:
        pytest.skip("no KFI leaves the stage in this book (declined, above)")
    payload = _served(_intent("stage_completion_rate",
                              filters=_stage("origin_stage", "KFI")),
                      monkeypatch, history)
    answer = payload["answer"]
    assert answer.startswith(
        f"Historical completion rate (from stage: KFI): "
        f"{rates['KFI']['rate'] * 100:.1f}%.")
    assert f"completed so far {rates['KFI']['completedSoFar']}" in answer
    assert answer.endswith("The forecast weights no KFI case: it is top of funnel.")


# --------------------------------------------------------------------------- #
# the engine
# --------------------------------------------------------------------------- #

def test_the_engine_computes_only_the_period_change():
    """Every arithmetic node in the engine is inside `period_change` — the one
    place a change between two governed figures is computed."""
    tree = ast.parse(inspect.getsource(engine))
    change = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "period_change")
    inside = {id(n) for n in ast.walk(change)}
    outside = [ast.unparse(n) for n in ast.walk(tree)
               if isinstance(n, ast.BinOp) and id(n) not in inside]
    assert outside == [], outside


@pytest.mark.parametrize("earlier, later, change, pct", [
    (100.0, 110.0, 10.0, 10.0), (200.0, 150.0, -50.0, -25.0),
    (0.0, 50.0, 50.0, None), (None, 5.0, None, None)])
def test_a_period_change_is_the_later_figure_less_the_earlier(earlier, later,
                                                              change, pct):
    out = engine.period_change(earlier, later)
    assert out["change"] == change and out["change_pct"] == pct


def test_the_runtimes_share_the_engine():
    from mi_agent import plan_forecast_runtime as forecast_rt
    assert forecast_rt._Refusal is engine.Refusal
    assert "serve_figure" in inspect.getsource(forecast_rt)
    assert "serve_figure" in inspect.getsource(stage_rt)
