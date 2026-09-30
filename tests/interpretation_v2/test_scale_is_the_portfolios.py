""""Scale" is the portfolio's threshold for its stage — owner decision D9.

The 2026-09-29 production bank asked "When does the book reach scale?" and
"What is the expected time to securitisation scale?". The model, correctly,
asked for an amount; the legacy path answered with a fixed ladder of round
numbers and no date. D9: scale is specific to the portfolio — an established
client is at scale once its assets under management exceed £250MM (owner decision 2026-09-30; was £200MM); a new SPV
before securitisation has scale at £100MM — recorded as `portfolio.stage` in
the client configuration, never a figure the model invents.

The model names the word `scale`; the compiler binds it as a governed named
threshold; the forecast runtime resolves it from the portfolio's stage after
interpretation, and refuses when no stage is recorded.
"""
from __future__ import annotations

import pytest

from mi_agent import plan_forecast_runtime as forecast_rt
from mi_agent import portfolio_metadata
from mi_agent.interpretation_v2.vocabulary import (
    NAMED_THRESHOLDS, SPECIALIST_MEASURE_DEFINITIONS, VOCABULARY_VERSION)
from mi_agent_api import forecast_extrapolation as fx
from mi_agent_api import scale_policy
from tests.interpretation_v2.test_specialist_runtime_forecast import (
    _CLIENT, _NEAR, _intent, _owner, _plan, _served, funded_root)  # noqa: F401

_SCALE = {"concept": "forecast_funded_balance", "comparator": "gte",
          "value": "scale"}


@pytest.fixture
def stage(monkeypatch):
    """Record a portfolio stage in the client layer, and a test policy whose
    threshold the fixture book can reach or has passed."""
    def _set(name, threshold=None):
        monkeypatch.setattr(portfolio_metadata, "_client_config",
                            lambda cid: {"portfolio": {"stage": name}} if name else {})
        if threshold is not None:
            policy = dict(scale_policy.load_policy())
            policy[name] = dict(policy[name], threshold=threshold)
            monkeypatch.setattr(scale_policy, "load_policy", lambda: policy)
    return _set


# --------------------------------------------------------------------------- #
# the decision and what the model is shown
# --------------------------------------------------------------------------- #

def test_the_policy_is_the_owners_decision():
    policy = scale_policy.load_policy()
    assert policy["pre_securitisation_spv"]["threshold"] == 100_000_000
    # £250MM (owner decision 2026-09-30; was £200MM).
    assert policy["established"]["threshold"] == 250_000_000
    assert "all the client's portfolios" in policy["established"]["measured_on_label"]


def test_ere_is_recorded_as_a_pre_securitisation_spv():
    """Owner-confirmed 2026-09-29 and again 2026-09-30: ERE is a new SPV
    before securitisation, at scale at £100MM (an established client's
    £250MM is another stage's). Production reads the configuration OCC
    activated, not this file."""
    import yaml
    doc = yaml.safe_load(open("config/client/config_client_ERE.yaml"))
    threshold, why, _ = scale_policy.resolve("ERE", document=doc)
    assert why == "" and threshold.stage == "pre_securitisation_spv"
    assert threshold.threshold == 100_000_000


def test_the_model_is_told_to_name_scale_and_never_a_number():
    definition = SPECIALIST_MEASURE_DEFINITIONS["forecast_milestone_date"]
    assert "the word `scale`" in definition
    assert "never write a number" in definition
    assert set(NAMED_THRESHOLDS) == {"scale"}
    for figure in ("100", "200", "£"):
        assert figure not in definition
    major, minor, _ = (int(x) for x in VOCABULARY_VERSION.split("."))
    assert (major, minor) >= (2, 5)


def test_the_compiler_binds_scale_and_refuses_any_other_word():
    assert _plan(target=_SCALE)["target"]["value"] == "scale"
    assert _plan(target=dict(_SCALE, value="Scale"))["target"]["value"] == "scale"
    from mi_agent.interpretation_v2.compiler import CompilerContext, DeterministicCompiler
    from mi_agent.interpretation_v2.intent import parse_candidate_intent
    result = DeterministicCompiler(CompilerContext()).compile(
        parse_candidate_intent(_intent(target=dict(_SCALE, value="huge"))))
    assert result.plan is None


def test_a_scale_milestone_is_eligible():
    assert forecast_rt.check_eligibility(_plan(target=_SCALE))[0] is True


# --------------------------------------------------------------------------- #
# execution — the portfolio's threshold, the owner's answer
# --------------------------------------------------------------------------- #

def test_a_scale_milestone_is_the_owners_answer_at_the_portfolios_threshold(
        funded_root, stage):
    stage("pre_securitisation_spv", threshold=20_000_000)
    outcome = forecast_rt.execute(_plan(target=_SCALE), output_root=funded_root,
                                  pipeline_root=_NEAR, client_id=_CLIENT)
    assert outcome.ok, outcome.detail
    owner = _owner(funded_root, _NEAR, 20_000_000)
    decided = fx.milestone_answer(owner["completionRunRateForecast"]["milestones"],
                                  20_000_000, owner["currentFundedBalance"])
    receipt = outcome.receipt
    assert receipt["threshold_applied"] == 20_000_000
    assert receipt["scale"]["stage"] == "pre_securitisation_spv"
    assert receipt["scale"]["decision"] == "D9"
    assert receipt["milestone_state"] == decided["state"] == fx.MILESTONE_PROJECTED
    assert receipt["milestone"]["baseDate"] == decided["milestone"]["baseDate"]
    assert receipt["gap_to_threshold"] == decided["gap"] > 0
    assert receipt["target"]["value"] == "scale"


def test_a_portfolio_past_its_threshold_is_at_scale(funded_root, stage):
    stage("established", threshold=5_000_000)
    outcome = forecast_rt.execute(_plan(target=_SCALE), output_root=funded_root,
                                  pipeline_root=_NEAR, client_id=_CLIENT)
    assert outcome.ok, outcome.detail
    assert outcome.receipt["milestone_state"] == fx.MILESTONE_ALREADY_REACHED
    assert outcome.receipt["gap_to_threshold"] == 0
    assert outcome.receipt["scale"]["stage"] == "established"


def test_no_stage_recorded_is_refused_not_guessed(funded_root, stage):
    stage(None)
    outcome = forecast_rt.execute(_plan(target=_SCALE), output_root=funded_root,
                                  pipeline_root=_NEAR, client_id=_CLIENT)
    assert not outcome.ok
    assert outcome.reason == forecast_rt.SCALE_NOT_CONFIGURED
    assert "portfolio.stage" in outcome.detail


def test_a_named_amount_is_unchanged(funded_root, stage):
    stage(None)
    outcome = forecast_rt.execute(_plan(), output_root=funded_root,
                                  pipeline_root=_NEAR, client_id=_CLIENT)
    assert outcome.ok, outcome.detail
    assert outcome.receipt["threshold_applied"] == 20_000_000
    assert outcome.receipt["scale"] is None


# --------------------------------------------------------------------------- #
# the answer
# --------------------------------------------------------------------------- #

def test_the_answer_names_the_rule_the_stage_and_the_gap(monkeypatch, funded_root,
                                                         stage):
    from mi_agent_api.mi_service import _governed_plan_coverage

    stage("pre_securitisation_spv", threshold=20_000_000)
    payload, record = _served(_intent(population={"base": "funded"}, target=_SCALE),
                              monkeypatch, funded_root)
    assert payload is not None, record.get("execution")
    answer = payload["answer"]
    assert answer.startswith("Scale (D9): for a new SPV before securitisation, "
                             "scale is £20.0m")
    assert "to go" in answer
    assert _governed_plan_coverage(payload)["unaccounted"] == []


def test_the_answer_says_the_portfolio_is_at_scale(monkeypatch, funded_root, stage):
    stage("established", threshold=5_000_000)
    payload, _ = _served(_intent(population={"base": "funded"}, target=_SCALE),
                         monkeypatch, funded_root)
    assert "so the portfolio is at scale" in payload["answer"]
    assert "assets under management" in payload["answer"]
