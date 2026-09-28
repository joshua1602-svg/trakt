"""Forecast executes from a GovernedQueryPlan — P0 Change 2a.

Offline, no model call. The funded side is five small governed runs written to a
temporary onboarding root; the pipeline side is the committed fixtures, which
between them give both D1 cases for free:

    client_001_mi_pack     one extract, 2025-12-01 — one day after the funded
                           book's 2025-11-30, inside the 45-day ceiling
    pipeline_history_5w    five extracts ending 2026-05-29 — 180 days after it,
                           outside the ceiling

Every figure asserted here is compared with what the forecast owner itself
returns for the same inputs, never with a number written into the test.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mi_agent import plan_forecast_runtime as forecast_rt
from mi_agent import plan_runtime_registry as registry
from mi_agent.interpretation_v2.compiler import CompilerContext, DeterministicCompiler
from mi_agent.interpretation_v2.intent import parse_candidate_intent
from mi_agent_api import forecast_extrapolation as fx
from tests.interpretation_v2.test_specialist_runtime_pipeline import (
    _Scripted, _code_of)

_CLIENT = "client_001"
_NEAR = "tests/fixtures/client_001_mi_pack"
_STALE = "tests/fixtures/pipeline_history_5w"
_RUNS = (("mi_2025_07", "2025-07-31", 50), ("mi_2025_08", "2025-08-31", 55),
         ("mi_2025_09", "2025-09-30", 60), ("mi_2025_10", "2025-10-31", 66),
         ("mi_2025_11", "2025-11-30", 72))


@pytest.fixture(scope="module")
def funded_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("onboarding_output")
    rng = np.random.default_rng(7)
    for run_id, reporting_date, n in _RUNS:
        frame = pd.DataFrame({
            "loan_identifier": [f"{run_id}_{i}" for i in range(n)],
            "current_outstanding_balance": rng.uniform(120_000, 280_000, n).round(2),
            "current_loan_to_value": rng.uniform(20, 55, n).round(1),
            "current_interest_rate": rng.uniform(3, 8, n).round(2),
            "youngest_borrower_age": rng.integers(62, 88, n),
            "reporting_date": [reporting_date] * n,
        })
        out = root / _CLIENT / run_id / "output" / "central"
        out.mkdir(parents=True)
        frame.to_csv(out / "18_central_lender_tape.csv", index=False)
    return str(root)


def _intent(**over):
    payload = {
        "schema_version": "candidate_intent/1.0", "capability": "forecast",
        "operation": "forecast_milestone",
        "measures": [{"concept": "forecast_milestone_date"}],
        "time": {"form": "forward_looking"},
        "target": {"concept": "forecast_funded_balance", "comparator": "gte",
                   "value": 20_000_000},
    }
    payload.update(over)
    return payload


_RUN_RATE = {"operation": "point_in_time", "target": None,
             "measures": [{"concept": "forecast_completion_rate"}],
             "time": {"form": "current"}}


def _plan(**over):
    result = DeterministicCompiler(CompilerContext()).compile(
        parse_candidate_intent(_intent(**over)))
    assert result.plan is not None, [r.code for r in result.reasons]
    return result.plan.to_dict()


def _owner(funded_root, pipeline_root, threshold=None):
    """What the forecast owner itself says, for the same inputs."""
    return fx.build_extrapolation(
        funded_root, pipeline_root, _CLIENT, None, history_model=None,
        extra_thresholds=([threshold] if threshold else ()))


# --------------------------------------------------------------------------- #
# the declaration
# --------------------------------------------------------------------------- #

def test_the_runtime_declares_forecast_and_only_forecast():
    assert forecast_rt.CAPABILITY == "forecast"
    assert forecast_rt.EXECUTABLE_POPULATIONS == frozenset({"forecast"})
    assert forecast_rt.EXECUTION_POPULATION == "forecast"


def test_a_derived_population_declares_its_inputs():
    """P0 §4.1 and D2a: funded and pipeline feed it; conversion is consumed."""
    inputs = forecast_rt.POPULATION_INPUTS
    assert set(inputs) == {"funded", "pipeline", "conversion"}
    assert inputs["funded"]["required"] is True
    assert inputs["conversion"]["owner"] == "pipeline_stage_movement"
    assert inputs["conversion"]["consumed_not_computed"] is True


def test_the_runtime_is_registered_above_the_funded_gate():
    assert forecast_rt in registry.POPULATION_OWNING_RUNTIMES
    assert "forecast" not in registry.DELIBERATELY_UNEXECUTED
    assert "forecast" in registry.executable_populations()
    assert registry.FUNDED_GATE_POPULATIONS == frozenset({"funded"})


def test_the_ceiling_context_is_the_funded_routes_whole_book_context():
    """D1 reads the SAME ceiling for the same context — so no second 45."""
    from mi_agent import portfolio_lens
    assert forecast_rt.WHOLE_BOOK_CONTEXT_ID == portfolio_lens.LENS_TOTAL


# --------------------------------------------------------------------------- #
# the perimeter
# --------------------------------------------------------------------------- #

def test_the_milestone_and_balance_plans_are_eligible():
    for plan in (_plan(), _plan(population={"base": "funded"}),
                 _plan(**_BALANCE)):
        assert forecast_rt.check_eligibility(plan) == (True, "", "")


def test_the_shapes_production_used_for_several_questions_are_held():
    """The owner computes them; the plan cannot yet say which figure it means."""
    for plan in (_plan(**_RUN_RATE), _plan(**_PROJECTION)):
        ok, why, detail = forecast_rt.check_eligibility(plan)
        assert (ok, why) == (False, forecast_rt.AMBIGUOUS_READING)
        assert "vocabulary defines the measure" in detail


def test_a_grain_asks_for_a_series_and_is_refused():
    """[76], [77], [127]: 'by month' / 'over the next twelve months'."""
    plan = _plan(**_BALANCE)
    plan["period"]["grain"] = "monthly"
    assert forecast_rt.check_eligibility(plan)[1] == forecast_rt.PERIOD_NOT_SUPPORTED


_BALANCE = {"operation": "point_in_time", "target": None,
            "measures": [{"concept": "forecast_funded_balance"}],
            "time": {"form": "current"}}
_PROJECTION = {"operation": "forecast_projection", "target": None,
               "measures": [{"concept": "forecast_funded_balance"}],
               "time": {"form": "forward_looking"}}


@pytest.mark.parametrize("over, reason", [
    ({"operation": "series", "target": None,
      "measures": [{"concept": "forecast_funded_balance"}],
      "time": {"form": "series", "grain": "monthly"}},
     forecast_rt.OPERATION_NOT_SUPPORTED),
    ({"population": {"base": "pipeline"}}, forecast_rt.POPULATION_NOT_FORECAST),
    ({"population": {"base": "funded", "lens": "direct"}},
     forecast_rt.SCOPE_NOT_SUPPORTED),
    ({"target": {"concept": "forecast_funded_balance", "comparator": "gt",
                 "value": 20_000_000}}, forecast_rt.TARGET_NOT_SUPPORTED),
    (dict(_BALANCE, time={"form": "range", "grain": "weekly",
                          "periods_back": 8}), forecast_rt.PERIOD_NOT_SUPPORTED),
])
def test_what_2a_does_not_serve_is_refused_by_name(over, reason):
    result = DeterministicCompiler(CompilerContext()).compile(
        parse_candidate_intent(_intent(**over)))
    plan = result.plan.to_dict() if result.plan is not None else _intent(**over)
    ok, why, detail = forecast_rt.check_eligibility(plan)
    assert (ok, why) == (False, reason), detail


def test_a_balance_series_is_refused_naming_the_definition():
    """D6 defined the POINT figure; the only series is the month-joined one."""
    plan = _plan(operation="series", target=None,
                 measures=[{"concept": "forecast_funded_balance"}],
                 time={"form": "series", "grain": "monthly"})
    ok, why, detail = forecast_rt.check_eligibility(plan)
    assert (ok, why) == (False, forecast_rt.OPERATION_NOT_SUPPORTED)
    assert "D6" in detail


def test_a_filtered_or_grouped_forecast_is_refused():
    plan = _plan()
    plan["outputs"][0]["filters"] = [{"canonical_field": "broker_channel",
                                      "comparator": "eq", "value": "Alpha"}]
    assert forecast_rt.check_eligibility(plan)[1] == forecast_rt.FILTERS_NOT_SUPPORTED
    plan = _plan()
    plan["outputs"][0]["dimensions"] = [{"concept": "region",
                                         "canonical_field": "region"}]
    assert forecast_rt.check_eligibility(plan)[1] == forecast_rt.DIMENSION_NOT_SUPPORTED


def test_another_capability_is_not_claimed():
    plan = _plan()
    plan["capability"] = "pipeline"
    assert forecast_rt.claims(plan) is False
    assert forecast_rt.check_eligibility(plan)[1] == forecast_rt.CAPABILITY_NOT_FORECAST


# --------------------------------------------------------------------------- #
# execution — every figure is the owner's
# --------------------------------------------------------------------------- #

def test_a_projected_milestone_is_the_owners_answer(funded_root):
    outcome = forecast_rt.execute(_plan(), output_root=funded_root,
                                  pipeline_root=_NEAR, client_id=_CLIENT)
    assert outcome.ok, outcome.detail
    owner = _owner(funded_root, _NEAR, 20_000_000)
    rr = owner["completionRunRateForecast"]
    decided = fx.milestone_answer(rr["milestones"], 20_000_000,
                                  owner["currentFundedBalance"])
    receipt = outcome.receipt
    assert receipt["milestone_state"] == decided["state"] == fx.MILESTONE_PROJECTED
    assert receipt["milestone"]["baseDate"] == decided["milestone"]["baseDate"]
    assert outcome.value == decided["milestone"]["baseDate"]
    assert receipt["base_monthly_run_rate"] == rr["baseMonthlyRunRate"]
    assert receipt["current_funded_balance"] == owner["currentFundedBalance"]
    assert receipt["target"] == {"concept": "forecast_funded_balance",
                                 "comparator": "gte", "value": 20_000_000}
    assert receipt["decision_owner"] == forecast_rt.OWNER_MILESTONE_RULE


def test_a_projected_milestone_states_both_vintages_and_the_skew(funded_root):
    """D1 inside the ceiling: answer, with both inputs dated on the receipt."""
    receipt = forecast_rt.execute(_plan(), output_root=funded_root,
                                  pipeline_root=_NEAR, client_id=_CLIENT).receipt
    owner = _owner(funded_root, _NEAR, 20_000_000)
    assert receipt["inputs"]["funded"]["as_of"] == owner["fundedReportingDate"]
    assert receipt["inputs"]["pipeline"]["as_of"] == owner["completionFlowExtractDate"]
    assert receipt["input_vintage_skew_days"] == 1
    assert receipt["input_vintage_ceiling_days"] == 45
    assert receipt["completion_signal"]["kind"] == fx.SIGNAL_OBSERVED_COMPLETION_FLOW


def test_already_reached_reads_the_funded_book_alone(funded_root):
    """No pipeline input fed "already reached", so none is claimed."""
    plan = _plan(target={"concept": "forecast_funded_balance",
                         "comparator": "gte", "value": 5_000_000})
    outcome = forecast_rt.execute(plan, output_root=funded_root,
                                  pipeline_root=_NEAR, client_id=_CLIENT)
    assert outcome.ok, outcome.detail
    assert outcome.receipt["milestone_state"] == fx.MILESTONE_ALREADY_REACHED
    assert set(outcome.receipt["inputs"]) == {"funded"}
    assert outcome.receipt["input_vintage_skew_days"] is None


def test_the_run_rate_is_the_owners_and_is_as_at_its_signal(funded_root):
    outcome = forecast_rt.execute(_plan(**_RUN_RATE), output_root=funded_root,
                                  pipeline_root=_NEAR, client_id=_CLIENT)
    assert outcome.ok, outcome.detail
    rr = _owner(funded_root, _NEAR)["completionRunRateForecast"]
    assert outcome.value == rr["baseMonthlyRunRate"]
    assert outcome.receipt["annualised_run_rate"] == rr["annualisedRunRate"]
    assert set(outcome.receipt["inputs"]) == {"pipeline"}
    assert outcome.receipt["completion_signal"]["description"] == \
        rr["assumptions"]["completionSignal"]


def test_inputs_beyond_the_ceiling_are_refused_naming_both_dates(funded_root):
    """D1 outside the ceiling: 180 days, refused, and the operator can see why."""
    outcome = forecast_rt.execute(_plan(), output_root=funded_root,
                                  pipeline_root=_STALE, client_id=_CLIENT)
    assert not outcome.ok
    assert outcome.reason == forecast_rt.POPULATION_VINTAGE_SKEW
    for fragment in ("2025-11-30", "2026-05-29", "180 days", "45 days"):
        assert fragment in outcome.detail


def test_the_ceiling_is_the_governed_policy_for_the_whole_book(funded_root):
    """No constant here: the policy decides, for the context the funded route uses."""
    asked = []

    class Policy:
        def __init__(self, days):
            self.days = days

        def max_snapshot_gap_days(self, context_id=None):
            asked.append(context_id)
            return self.days

    wide = forecast_rt.execute(_plan(), output_root=funded_root,
                               pipeline_root=_STALE, client_id=_CLIENT,
                               policy=Policy(200))
    assert wide.ok and wide.receipt["input_vintage_ceiling_days"] == 200
    tight = forecast_rt.execute(_plan(), output_root=funded_root,
                                pipeline_root=_NEAR, client_id=_CLIENT,
                                policy=Policy(0))
    assert tight.reason == forecast_rt.POPULATION_VINTAGE_SKEW
    assert set(asked) == {forecast_rt.WHOLE_BOOK_CONTEXT_ID}
    unset = forecast_rt.execute(_plan(), output_root=funded_root,
                                pipeline_root=_NEAR, client_id=_CLIENT,
                                policy=Policy(None))
    assert unset.reason == forecast_rt.POPULATION_VINTAGE_SKEW


def test_a_used_input_without_a_date_is_unresolved():
    ok, why, _detail, _ = forecast_rt.vintage_skew(
        {"funded": {"as_of": "2026-08-31"}, "pipeline": {"as_of": None}},
        ceiling_days=45)
    assert (ok, why) == (False, forecast_rt.POPULATION_INPUT_UNRESOLVED)


def test_no_funded_root_refuses_rather_than_discovers():
    outcome = forecast_rt.execute(_plan(), output_root=None, pipeline_root=_NEAR,
                                  client_id=_CLIENT)
    assert outcome.reason == forecast_rt.INPUTS_UNAVAILABLE


# --------------------------------------------------------------------------- #
# the invariants
# --------------------------------------------------------------------------- #

def test_the_runtime_computes_no_forecast_figure():
    """§8.5: the one arithmetic node is the D1 distance between two DATES."""
    tree = ast.parse(Path(forecast_rt.__file__).read_text())
    arithmetic = [node for node in ast.walk(tree) if isinstance(node, ast.BinOp)]
    assert len(arithmetic) == 1, [ast.unparse(n) for n in arithmetic]
    assert ast.unparse(arithmetic[0]) == "latest - earliest"
    code = _code_of(forecast_rt)
    for forbidden in ("sum(", "math.", "ceil(", "_add_months", "_project_series"):
        assert forbidden not in code, forbidden


def test_the_forecast_runtime_never_reads_the_question():
    code = _code_of(forecast_rt)
    for forbidden in ("ParsedQuestion", "parse_with_repair", "resolve_dataset",
                      "try_route", "llm_query_parser", "chat_routing",
                      "recognise", "_scenario_multiplier"):
        assert forbidden not in code, forbidden
    # The one mention allowed: the composer's context takes a `question`, and
    # this module hands it an EMPTY one — it has none to give.
    tree = ast.parse(Path(forecast_rt.__file__).read_text())
    mentions = [node for node in ast.walk(tree)
                if (isinstance(node, ast.Name) and node.id == "question")
                or (isinstance(node, ast.Attribute) and node.attr == "question")
                or (isinstance(node, ast.keyword) and node.arg == "question")
                or (isinstance(node, ast.arg) and node.arg == "question")]
    assert len(mentions) == 1, [ast.unparse(m) if not isinstance(m, (ast.keyword, ast.arg))
                                else m.arg for m in mentions]
    assert isinstance(mentions[0], ast.keyword)
    assert isinstance(mentions[0].value, ast.Constant) and mentions[0].value.value == ""


# --------------------------------------------------------------------------- #
# end to end, through the real serve()
# --------------------------------------------------------------------------- #

def _served(payload, monkeypatch, funded_root, **over):
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

    kwargs = {"question": "q", "context": Principal(), "client_id": _CLIENT,
              "run_id": None, "legacy_result": {"ok": True}, "frame": None,
              "semantics": {}, "view": "funded", "output_root": funded_root,
              "pipeline_root": _NEAR, "pipeline_client_id": _CLIENT,
              "pipeline_history": None}
    kwargs.update(over)
    try:
        out = canary.serve(**kwargs)
    finally:
        wiring.set_interpreter_factory(None)
    return out, (written[-1] if written else {})


def test_serve_answers_a_milestone_with_its_measure_and_both_vintages(
        monkeypatch, funded_root):
    """D4 and D1 on the sentence itself, and the coverage owner accounts for all."""
    from mi_agent_api.mi_service import _governed_plan_coverage

    payload, record = _served(_intent(population={"base": "funded"}),
                              monkeypatch, funded_root)
    assert payload is not None, record.get("execution")
    assert record["serving"]["decision"] == "NEW"
    assert record["execution"]["runtime"] == "forecast"
    owner = _owner(funded_root, _NEAR, 20_000_000)
    answer = payload["answer"]
    assert "Forecast milestone date" in answer
    assert owner["fundedReportingDate"] in answer
    assert owner["completionFlowExtractDate"] in answer
    coverage = _governed_plan_coverage(payload)
    assert "governed_plan:target" in {e["kind"] for e in coverage["concepts"]}
    assert coverage["unaccounted"] == []


def test_serve_refuses_a_stale_pipeline_and_legacy_serves(monkeypatch, funded_root):
    payload, record = _served(_intent(), monkeypatch, funded_root,
                              pipeline_root=_STALE)
    assert payload is None
    assert record["serving"]["reason"].endswith(forecast_rt.POPULATION_VINTAGE_SKEW)


def test_a_different_threshold_on_the_receipt_is_unaccounted(monkeypatch,
                                                             funded_root):
    """The £250m defect, caught on the way out even if a runtime reintroduced it."""
    from mi_agent_api.mi_service import _governed_plan_coverage

    payload, _record = _served(_intent(), monkeypatch, funded_root)
    executed = payload["metadata"]["governedPlan"]["executed"]
    executed["target"] = dict(executed["target"], value=75_000_000)
    unaccounted = _governed_plan_coverage(payload)["unaccounted"]
    assert [e["kind"] for e in unaccounted] == ["governed_plan:target"]


# --------------------------------------------------------------------------- #
# D6 — the forecast funded balance is the composer's, which is the tab's
# --------------------------------------------------------------------------- #

@pytest.fixture
def estate(funded_root, monkeypatch):
    """The environment `mi_service` resolves a production request in."""
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    monkeypatch.setenv("MI_AGENT_ONBOARDING_OUTPUT_ROOT", funded_root)
    monkeypatch.setenv("MI_AGENT_PIPELINE_ROOT", _NEAR)
    # As `mi_agent_api/tests/conftest.py` does for every API test: the tab's
    # endpoint is called in-process, not through an authenticated client.
    monkeypatch.setenv("MI_AGENT_AUTH_ENABLED", "false")
    from mi_agent_api import datasets as ds

    def funded_frame(client_id, run_id):
        """What `mi_service._routed_frame` does, verbatim in effect."""
        pid = f"{client_id}/{run_id}" if run_id else (client_id or None)
        frame, err = ds._resolve_query_frame("funded", pid)
        return None if err else frame

    return funded_frame


def _tab(run_id="mi_2025_11"):
    """What the React Forecast tab is served for the same book."""
    from fastapi.testclient import TestClient
    from mi_agent_api.app import app
    body = TestClient(app).get(
        f"/mi/forecast/snapshot?portfolioId={_CLIENT}/{run_id}").json()
    return body.get("forecastBridge") or {}


def _semantics():
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent_api.data_source import semantics_path
    return load_mi_semantics(semantics_path())


def test_the_governed_balance_is_the_forecast_tabs_figure(funded_root, estate):
    for plan in (_plan(**_BALANCE), _plan(**_PROJECTION)):
        outcome = forecast_rt.execute(
            plan, output_root=funded_root, pipeline_root=_NEAR,
            client_id=_CLIENT, run_id="mi_2025_11",
            funded_frame_resolver=estate, semantics=_semantics())
        assert outcome.ok, outcome.detail
        tab = _tab()
        assert tab.get("forecastFundedBalance") is not None
        assert outcome.value == tab["forecastFundedBalance"]
        receipt = outcome.receipt
        assert receipt["current_funded_balance"] == tab["fundedBalance"]
        assert receipt["weighted_expected_funded_amount"] == \
            tab["weightedExpectedFundedAmount"]
        assert receipt["execution_owner"] == forecast_rt.OWNER_COMPOSER
        assert receipt["definition_decision"] == "D6"


def test_the_balance_states_both_vintages(funded_root, estate):
    outcome = forecast_rt.execute(
        _plan(**_BALANCE), output_root=funded_root, pipeline_root=_NEAR,
        client_id=_CLIENT, run_id="mi_2025_11",
        funded_frame_resolver=estate, semantics=_semantics())
    tab = _tab()
    inputs = outcome.receipt["inputs"]
    assert inputs["funded"]["as_of"] == tab["fundedReportingDate"]
    assert inputs["pipeline"]["as_of"] == tab["pipelineAsOfDate"]
    assert outcome.receipt["input_vintage_skew_days"] is not None


def test_the_balance_is_not_the_month_joined_series(funded_root, estate):
    """The other definition exists, and is not what is served."""
    outcome = forecast_rt.execute(
        _plan(**_BALANCE), output_root=funded_root, pipeline_root=_NEAR,
        client_id=_CLIENT, run_id="mi_2025_11",
        funded_frame_resolver=estate, semantics=_semantics())
    series = _owner(funded_root, _NEAR)["currentWeightedPipelineForecast"]
    assert outcome.receipt["execution_owner"] != forecast_rt.OWNER_EXTRAPOLATION
    assert outcome.receipt["calculation_owner"].endswith("compute_forecast_bridge")
    # On this fixture the funded month has no same-month extract, so the
    # month-joined figure has no weighted pipeline at all while the composer's
    # does: two definitions, visibly different.
    assert series["weightedExpectedPipeline"] != \
        outcome.receipt["weighted_expected_funded_amount"]


def test_the_balance_without_a_funded_resolver_refuses(funded_root, estate):
    outcome = forecast_rt.execute(
        _plan(**_BALANCE), output_root=funded_root, pipeline_root=_NEAR,
        client_id=_CLIENT, run_id="mi_2025_11", funded_frame_resolver=None,
        semantics=_semantics())
    assert outcome.reason == forecast_rt.INPUTS_UNAVAILABLE


def test_serve_answers_the_forecast_funded_balance(monkeypatch, funded_root,
                                                   estate):
    from mi_agent_api.mi_service import _governed_plan_coverage

    payload, record = _served(_intent(**_BALANCE), monkeypatch, funded_root,
                              run_id="mi_2025_11", funded_frame_resolver=estate,
                              semantics=_semantics())
    assert payload is not None, record.get("execution")
    tab = _tab()
    answer = payload["answer"]
    assert answer.startswith("Forecast funded balance: ")
    assert tab["fundedReportingDate"] in answer
    assert tab["pipelineAsOfDate"] in answer
    assert payload["artifacts"][0]["kpis"][0]["rawValue"] == \
        tab["forecastFundedBalance"]
    assert _governed_plan_coverage(payload)["unaccounted"] == []
