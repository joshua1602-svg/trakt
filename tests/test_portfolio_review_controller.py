#!/usr/bin/env python3
"""The autonomous period review: one loop, a pinned period, evidence kept.

The model is scripted here. That is deliberate and not a shortcut: what needs
testing is the controller — that the period is resolved from governed discovery
and stated to the model, that a period with nothing to compare is an answer
rather than a crash, that the loop stays bounded, and that every governed call
lands in the transcript a finding can be traced to. None of that is a property
of the model, and testing it against a real one would make it untestable.

What the model may decide is not asserted anywhere. There is no test that it
calls a particular tool first, because the moment such a test exists the loop is
executing a checklist and the autonomy claim is dead.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from readiness_agent.session import GovernedSession
from trakt_core import config_cache
from trakt_core.context import SCOPE_LOAN_READ, SCOPE_RISK_READ
from trakt_tools.execution import ToolDependencies

import portfolio_review as review
from portfolio_review.controller import (
    DEFAULT_REVIEW_STEPS, ReviewOutcome, resolve_period, run_review,
)

from tests.planted_portfolio import SNAPSHOT_ID, planted_frame
from tests.test_agent_governed_execution import _Datasets, _Descriptor
from tests.test_agent_loan_retrieval import (
    PORTFOLIO_A, TENANT, _catalogue, _context, _CountingResolver,
)

BOTH = (SCOPE_LOAN_READ, SCOPE_RISK_READ)

MONTHLY_CONTEXT = {
    "available": True, "period": review.PERIOD_MONTHLY_FUNDED,
    "current_run_id": "mi_2026_07", "prior_run_id": "mi_2026_06",
    "current_reporting_date": "2026-07-31",
    "prior_reporting_date": "2026-06-30",
}


@pytest.fixture(autouse=True)
def _clean_cache():
    config_cache.reset()
    yield
    config_cache.reset()


def _session() -> GovernedSession:
    deps = ToolDependencies(
        datasets=_Datasets(_Descriptor(snapshot_id=SNAPSHOT_ID)),
        runtime_mode="test", catalogue=_catalogue(), output_root="/unused",
        loan_frame_resolver=_CountingResolver(planted_frame()),
        pipeline_root="/governed/pipeline")
    return GovernedSession(_context(capabilities=BOTH), PORTFOLIO_A.key,
                           dependencies=deps)


# --------------------------------------------------------------------------- #
# A scripted model — enough of the Anthropic shape for the loop to drive it
# --------------------------------------------------------------------------- #
class _Block:
    def __init__(self, type_, **kw):
        self.type = type_
        for k, v in kw.items():
            setattr(self, k, v)


class _Response:
    def __init__(self, content):
        self.content = content
        self.usage = type("U", (), {"input_tokens": 10, "output_tokens": 5})()


class _ScriptedModel:
    """Replays a fixed list of turns and records the prompts it was given."""

    def __init__(self, turns: List[List[Any]]):
        self._turns = list(turns)
        self.systems: List[str] = []
        self.tool_names: List[str] = []
        self.openings: List[str] = []
        self.calls = 0
        self.messages = type("M", (), {"create": self._create})()

    def _create(self, *, model, max_tokens, system, tools, messages):
        self.calls += 1
        self.systems.append(system)
        self.tool_names.append([t["name"] for t in tools])
        if self.calls == 1:
            self.openings.append(messages[0]["content"])
        if not self._turns:
            return _Response([_Block("text", text="done")])
        return _Response(self._turns.pop(0))


def _tool_turn(name: str, **args):
    return [_Block("tool_use", name=name, id=f"tu-{name}", input=args)]


REVIEW_PAYLOAD = {
    "period_verdict": "MATERIAL_DEVELOPMENTS",
    "headline": "Funded assets rose, driven by one added portfolio.",
    "summary": "Two or three sentences.",
    "findings": [{
        "title": "Portfolio addition dominates the month",
        "observation": "funded_composition reported portfolio_additions of £68m.",
        "why_it_matters": "The incumbent book must be read separately.",
        "severity": "high",
        "evidence_tools": ["funded_composition"],
    }],
    "period_explained_by": "portfolio_beta",
}


def _submit_turn():
    return [_Block("tool_use", name="submit_review", id="tu-submit",
                   input=REVIEW_PAYLOAD)]


# =========================================================================== #
# Reuse — one loop, not a second framework
# =========================================================================== #
def test_the_review_runs_on_the_readiness_loop():
    """A second loop would duplicate the property that the model cannot compute,
    and a duplicated safety property is how one copy quietly loses it."""
    import inspect

    from portfolio_review import controller

    source = inspect.getsource(controller)
    assert "run_assessment" in source
    # And it builds no client of its own.
    assert "anthropic" not in source


def test_the_review_uses_its_own_prompt_and_submission_tool():
    model = _ScriptedModel([_submit_turn()])
    run_review(_session(), period=review.PERIOD_MONTHLY_FUNDED,
               period_context=MONTHLY_CONTEXT, client=model)

    assert "MANAGEMENT INFORMATION" in model.systems[0]
    assert "submit_review" in model.tool_names[0]
    assert "submit_assessment" not in model.tool_names[0]


def test_the_review_is_offered_MI_tools_and_no_readiness_tools():
    """The mandate reaches the model as a narrower tool surface, not advice.

    Asserted on what the model was actually handed rather than on the prompt,
    because the prompt is the hint and the surface is the control.
    """
    from portfolio_review import mandate

    model = _ScriptedModel([_submit_turn()])
    run_review(_session(), period=review.PERIOD_MONTHLY_FUNDED,
               period_context=MONTHLY_CONTEXT, client=model)

    offered = set(model.tool_names[0]) - {"submit_review"}
    assert offered == set(mandate.ALLOWED)
    assert not (offered & mandate.EXCLUDED_NAMES)
    for readiness_tool in ("regulatory_readiness", "evaluate_rule_packs",
                           "readiness_metrics"):
        assert readiness_tool not in offered


def test_the_readiness_agent_still_uses_its_own(monkeypatch):
    """The parameterisation must not have changed the agent it was taken from."""
    from readiness_agent import agent as ra

    model = _ScriptedModel([[_Block("tool_use", name="submit_assessment",
                                    id="tu-1", input={"summary": "x"})]])
    ra.run_assessment(_session(), client=model, max_steps=2)

    assert "securitisation readiness" in model.systems[0]
    assert "submit_assessment" in model.tool_names[0]


# =========================================================================== #
# The period is pinned, and pinned from governed discovery
# =========================================================================== #
def test_the_period_is_stated_to_the_model():
    model = _ScriptedModel([_submit_turn()])
    run_review(_session(), period=review.PERIOD_MONTHLY_FUNDED,
               period_context=MONTHLY_CONTEXT, client=model)

    opening = model.openings[0]
    assert "2026-07-31" in opening
    assert "2026-06-30" in opening
    assert "mi_2026_07" in opening


def test_no_figure_about_the_position_is_put_in_the_prompt():
    """Every number the review states must have come from a tool call.

    That rule has to hold of the model's INPUT as well as its output: a balance
    in the opening message is a number it could quote without ever asking.
    """
    model = _ScriptedModel([_submit_turn()])
    run_review(_session(), period=review.PERIOD_MONTHLY_FUNDED,
               period_context=MONTHLY_CONTEXT, client=model)

    opening = model.openings[0]
    assert "Nothing about the position is stated here" in opening
    for token in ("£", "balance is", "loan count"):
        assert token not in opening


def test_the_monthly_period_comes_from_the_governed_snapshot_discovery(monkeypatch):
    from mi_agent_api import snapshots as snap

    monkeypatch.setattr(snap, "discover_snapshots", lambda root: {"portfolios": [{
        "client_id": TENANT,
        "runs": [{"run_id": "mi_2026_05", "reporting_date": "2026-05-31"},
                 {"run_id": "mi_2026_06", "reporting_date": "2026-06-30"},
                 {"run_id": "mi_2026_07", "reporting_date": "2026-07-31"}]}]})

    context = resolve_period(review.PERIOD_MONTHLY_FUNDED, client_id=TENANT,
                             output_root="/governed/funded")
    assert context["current_run_id"] == "mi_2026_07"
    assert context["prior_run_id"] == "mi_2026_06"


def test_an_explicit_run_id_reviews_that_period_not_the_latest(monkeypatch):
    """A review of an approved run must not silently drift to a later one."""
    from mi_agent_api import snapshots as snap

    monkeypatch.setattr(snap, "discover_snapshots", lambda root: {"portfolios": [{
        "client_id": TENANT,
        "runs": [{"run_id": "mi_2026_05", "reporting_date": "2026-05-31"},
                 {"run_id": "mi_2026_06", "reporting_date": "2026-06-30"},
                 {"run_id": "mi_2026_07", "reporting_date": "2026-07-31"}]}]})

    context = resolve_period(review.PERIOD_MONTHLY_FUNDED, client_id=TENANT,
                             output_root="/governed/funded",
                             to_run_id="mi_2026_06")
    assert context["current_run_id"] == "mi_2026_06"
    assert context["prior_run_id"] == "mi_2026_05"


def test_the_weekly_period_comes_from_the_governed_extract_inventory(monkeypatch):
    from mi_agent_api import pipeline_contract as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "weekly_extract_inventory",
                        lambda root, cid: {"extracts": [
                            {"pipeline_extract_date": "2026-07-31"},
                            {"pipeline_extract_date": "2026-08-07"}]})

    context = resolve_period(review.PERIOD_WEEKLY_PIPELINE, client_id=TENANT,
                             pipeline_root="/governed/pipeline")
    assert context["current_reporting_date"] == "2026-08-07"
    assert context["prior_reporting_date"] == "2026-07-31"


# =========================================================================== #
# Nothing to compare is an answer
# =========================================================================== #
def test_a_single_period_is_reported_not_raised(monkeypatch):
    from mi_agent_api import snapshots as snap

    monkeypatch.setattr(snap, "discover_snapshots", lambda root: {"portfolios": [{
        "client_id": TENANT,
        "runs": [{"run_id": "mi_2026_07", "reporting_date": "2026-07-31"}]}]})

    model = _ScriptedModel([_submit_turn()])
    outcome = run_review(_session(), period=review.PERIOD_MONTHLY_FUNDED,
                         client_id=TENANT, output_root="/governed/funded",
                         client=model)

    assert outcome.available is False
    assert "at least two governed funded reporting periods" in outcome.unavailable
    # And the model was never asked: there was nothing to review.
    assert model.calls == 0


def test_no_configured_root_is_reported_not_raised():
    model = _ScriptedModel([_submit_turn()])
    outcome = run_review(_session(), period=review.PERIOD_WEEKLY_PIPELINE,
                         client_id=TENANT, pipeline_root=None, client=model)

    assert outcome.available is False
    assert "no governed weekly pipeline root" in outcome.unavailable
    assert model.calls == 0


# =========================================================================== #
# The loop stays bounded, and the model never touches data
# =========================================================================== #
def test_a_model_that_never_submits_stops_at_the_ceiling():
    """A confused run costs a bounded amount, not an unbounded one."""
    model = _ScriptedModel([_tool_turn("portfolio_capabilities")
                            for _ in range(50)])
    outcome = run_review(_session(), period=review.PERIOD_MONTHLY_FUNDED,
                         period_context=MONTHLY_CONTEXT, client=model,
                         max_steps=3)

    assert outcome.available is False
    assert "step ceiling reached (3)" in outcome.stopped_reason
    assert outcome.steps == 3


def test_the_review_ceiling_is_tighter_than_the_readiness_one():
    from readiness_agent.agent import DEFAULT_MAX_STEPS

    assert DEFAULT_REVIEW_STEPS < DEFAULT_MAX_STEPS


def test_the_session_hands_the_model_no_frame():
    """The structural enforcement, asserted rather than trusted."""
    session = _session()
    assert not hasattr(session, "frame")
    assert not hasattr(session, "dataframe")
    public = {n for n in dir(session) if not n.startswith("_")}
    assert public == {"call", "capabilities", "available_metrics",
                      "unavailable_metrics", "transcript", "call_count",
                      "efficiency", "audit_record", "resource"}


# =========================================================================== #
# Evidence
# =========================================================================== #
def test_every_governed_call_lands_in_the_transcript():
    model = _ScriptedModel([
        _tool_turn("portfolio_capabilities"),
        _tool_turn("funded_composition", resource=PORTFOLIO_A.key),
        _submit_turn(),
    ])
    outcome = run_review(_session(), period=review.PERIOD_MONTHLY_FUNDED,
                         period_context=MONTHLY_CONTEXT, client=model)

    assert outcome.available is True
    tools = [c["tool"] for c in outcome.transcript]
    assert tools == ["portfolio_capabilities", "funded_composition"]
    assert outcome.efficiency["total_calls"] == 2


def test_a_finding_can_be_traced_to_the_calls_behind_it():
    model = _ScriptedModel([
        _tool_turn("funded_composition", resource=PORTFOLIO_A.key),
        _submit_turn(),
    ])
    outcome = run_review(_session(), period=review.PERIOD_MONTHLY_FUNDED,
                         period_context=MONTHLY_CONTEXT, client=model)

    finding = outcome.review["findings"][0]
    evidence = outcome.evidence_for(finding)
    assert [c["tool"] for c in evidence] == ["funded_composition"]


def test_a_finding_citing_a_tool_it_never_called_has_no_evidence():
    """Silence, not the whole transcript. Attaching every call would make an
    unsupported finding look as well evidenced as a supported one."""
    model = _ScriptedModel([_submit_turn()])
    outcome = run_review(_session(), period=review.PERIOD_MONTHLY_FUNDED,
                         period_context=MONTHLY_CONTEXT, client=model)

    assert outcome.evidence_for(outcome.review["findings"][0]) == []


def test_the_outcome_serialises_with_its_period_and_evidence():
    model = _ScriptedModel([_tool_turn("portfolio_capabilities"),
                            _submit_turn()])
    payload = run_review(_session(), period=review.PERIOD_MONTHLY_FUNDED,
                         period_context=MONTHLY_CONTEXT,
                         client=model).to_dict()

    assert payload["period"] == review.PERIOD_MONTHLY_FUNDED
    assert payload["period_context"]["current_reporting_date"] == "2026-07-31"
    assert payload["review"]["period_verdict"] == "MATERIAL_DEVELOPMENTS"
    assert payload["transcript"]


# =========================================================================== #
# The objective steers nothing
# =========================================================================== #
def test_no_objective_names_a_metric_or_an_ordering():
    """A prompt listing checks is workflow automation wearing an agent's
    clothes: it runs the same calls on a quiet month as on an acquisition
    month, and can never find the thing nobody listed."""
    for objective in (review.WEEKLY_PIPELINE_OBJECTIVE,
                      review.MONTHLY_FUNDED_OBJECTIVE):
        lowered = objective.lower()
        for metric in ("ltv", "borrower age", "vintage", "interest rate",
                       "first,", "then ", "step 1"):
            assert metric not in lowered, (objective, metric)


def test_the_prompt_forbids_inferring_an_acquisition_from_a_movement():
    """The rule the deterministic layer enforces, restated where the model reads."""
    prompt = review.SYSTEM_PROMPT
    assert "is NOT evidence that a portfolio was acquired" in prompt
    assert "unclassified" in prompt
    assert "You perform NO arithmetic" in prompt


def test_the_submission_schema_keeps_measurement_and_judgement_apart():
    properties = review.SUBMIT_REVIEW["input_schema"]["properties"]
    finding = properties["findings"]["items"]["properties"]
    assert "observation" in finding and "why_it_matters" in finding
    assert "could_not_assess" in properties
    # A routine period is a real verdict, not a fallback for having found nothing.
    assert "ROUTINE_PERIOD" in properties["period_verdict"]["enum"]


# =========================================================================== #
# The publication gate, end to end
# =========================================================================== #
def _turn(name, **args):
    return [_Block("tool_use", name=name, id=f"tu-{name}", input=args)]


def _review_turn(**over):
    payload = {**REVIEW_PAYLOAD, **over}
    return [_Block("tool_use", name="submit_review", id="tu-submit",
                   input=payload)]


def _run(model):
    return run_review(_session(), period=review.PERIOD_MONTHLY_FUNDED,
                      period_context=MONTHLY_CONTEXT, client=model)


def test_a_finding_stating_an_ungoverned_figure_never_reaches_the_card():
    """The gate runs on the production path, not only in the harness.

    `REVIEW_PAYLOAD` claims £68m of portfolio additions. The planted portfolio
    contains no such number, and no tool was called at all in this run, so the
    figure has no possible governed source.
    """
    outcome = _run(_ScriptedModel([_submit_turn()]))

    assert outcome.gate_status == "DEGRADED"
    assert outcome.card["findings"] == []
    assert outcome.unsupported_claims
    assert outcome.dropped_findings
    # The model's own words are preserved for the audit record even though
    # nothing was published from them.
    assert outcome.review["findings"][0]["title"].startswith("Portfolio addition")


def test_an_ungoverned_headline_withholds_the_whole_review():
    outcome = _run(_ScriptedModel([_review_turn(
        headline="Funded assets rose £999.9m this period.")]))

    assert outcome.gate_status == "BLOCKED"
    assert outcome.card is None
    assert outcome.available is False
    assert outcome.submitted is True
    assert "no governed result contains" in outcome.unavailable


def test_a_review_with_no_figures_is_published_intact():
    outcome = _run(_ScriptedModel([_review_turn(
        headline="Nothing material changed this period.",
        summary="The book was quiet.",
        findings=[{"title": "A routine month",
                   "observation": "No governed measure moved materially.",
                   "why_it_matters": "Nothing needs management attention.",
                   "severity": "low"}],
        period_explained_by=None)]))

    assert outcome.gate_status == "PUBLISHABLE"
    assert outcome.available is True
    assert len(outcome.card["findings"]) == 1
    assert outcome.unsupported_claims == []


def test_a_governed_figure_survives_and_lands_in_the_claim_ledger():
    """A number the session actually obtained is published and traceable."""
    model = _ScriptedModel([
        _turn("portfolio_summary"),
        _review_turn(headline="The book is steady.",
                     summary="One governed measure is quoted below.",
                     findings=[{"title": "Scale",
                                "observation": "The book totals £3.2m.",
                                "why_it_matters": "Unchanged scale.",
                                "severity": "low"}],
                     period_explained_by=None)])
    outcome = _run(model)

    assert outcome.gate_status == "PUBLISHABLE"
    row = next(r for r in outcome.claim_ledger if r["output_number"] == "3.2m")
    assert row["governed_source_tool"] == "portfolio_summary"
    assert row["source_field"] == "population.total_balance"
    assert row["exact_match"] == "yes"


def test_an_out_of_mandate_call_is_refused_and_audited():
    """The model may name a readiness tool; it may not reach one."""
    model = _ScriptedModel([
        _turn("regulatory_readiness"),
        _review_turn(headline="Nothing material changed.",
                     summary="Quiet.", findings=[], period_explained_by=None)])
    outcome = _run(model)

    refused = outcome.out_of_mandate_calls
    assert [c["tool"] for c in refused] == ["regulatory_readiness"]
    assert refused[0]["executed"] is False
    assert outcome.tool_call_audit[0]["in_allow_list"] is False
    # It never entered the governed transcript, because it never ran.
    assert all(c["tool"] != "regulatory_readiness" for c in outcome.transcript)


def test_the_card_caps_findings_at_five():
    many = [{"title": f"Finding {i}", "observation": "No figure here.",
             "why_it_matters": "Judgement.", "severity": "low"}
            for i in range(9)]
    outcome = _run(_ScriptedModel([_review_turn(
        headline="Several things moved.", summary="Nine findings submitted.",
        findings=many, period_explained_by=None)]))

    assert len(outcome.card["findings"]) == 5
    assert len(outcome.card["withheld_findings"]) == 4
    assert outcome.card["word_count"] <= 260
