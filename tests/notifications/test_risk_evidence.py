"""A Risk Review may only state a clear position over checks that actually ran.

These tests drive ``trakt_notifications.sources.resolve`` — the production
resolver — and stub only at the governed SERVICE boundary
(``compute_concentration_tests`` / ``period_movement``). That distinction is the
whole point of the file.

The defect these regressions exist for was invisible to the message-level tests
because those tests hand ``risk_review.build`` a ``GovernedInputs`` with
``concentration`` already populated. Production never populated it on a
funded-only approval: ``resolve`` skipped the pipeline side, which was the only
caller of the concentration service, so the domain was neither evaluated nor
recorded as unavailable — and the Risk Review reported the unqualified

    "No material portfolio risks were identified from this update."

on the strength of a check that did not run. A fixture that supplies the
evidence production withholds cannot catch that, so nothing here supplies it.
"""

from __future__ import annotations

import pytest

from trakt_notifications import risk_review, sources
from trakt_notifications.contract import (
    SEVERITY_ATTENTION, SEVERITY_CLEAR, SEVERITY_CONCERN,
)
from trakt_notifications.sources import CAP_CONCENTRATION

from .conftest import TENANT, concentration_risk

OUTPUT_ROOT = "/governed/funded"
PIPELINE_ROOT = "/governed/pipeline"


# --------------------------------------------------------------------------- #
# Governed service doubles — the SERVICE boundary, never the resolver
# --------------------------------------------------------------------------- #
def _clear_concentration() -> dict:
    """A concentration evaluation that ran and found nothing."""
    return {"available": True, "reportingDate": "2026-07-31",
            "states": {"available": True}, "tests": [], "emergingRisks": []}


def _warning_concentration() -> dict:
    """A test at warning: evaluated, and NOT clear."""
    return {
        "available": True, "reportingDate": "2026-07-31",
        "states": {"available": True},
        "tests": [{"testId": "region_ldn", "displayName": "London exposure",
                   "status": "warning", "utilization": 0.94}],
        "emergingRisks": [{
            "category": "expected_warning_low_headroom", "rank": 3,
            "testId": "region_ldn", "displayName": "London exposure",
            "statement": ("London exposure has low expected headroom: funded "
                          "28.4% vs 30.0%."),
            "expectedHeadroom": 1.6}],
    }


def _movement() -> dict:
    """A governed funded period movement, as ``period_movement`` returns it."""
    return {
        "available": True,
        "currentReportingDate": "2026-07-31",
        "priorReportingDate": "2026-06-30",
        "current": {"funded_balance": 184_000_000.0, "loan_count": 980,
                    "wa_ltv_points": 30.2},
        "prior": {"funded_balance": 180_000_000.0, "loan_count": 960,
                  "wa_ltv_points": 30.0},
        "delta": {"funded_balance": 4_000_000.0, "loan_count": 20,
                  "wa_ltv_points": 0.2},
        "cohortMovements": [],
        "primaryRegion": None,
    }


@pytest.fixture
def governed(monkeypatch):
    """Patch the two governed services; leave the resolver itself real.

    Returns a mutable dict the test sets to choose what each service returns, so
    a scenario is expressed as data rather than as another patch.
    """
    state = {"concentration": _clear_concentration(), "movement": _movement()}

    def _concentration(output_root, client_id, funded_run_id, scope=None):
        result = state["concentration"]
        if isinstance(result, Exception):
            raise result
        return result

    def _period_movement(output_root, client_id, **kwargs):
        return state["movement"]

    from mi_agent_api import concentration_tests_api, movement_summary
    monkeypatch.setattr(concentration_tests_api, "compute_concentration_tests",
                        _concentration)
    monkeypatch.setattr(movement_summary, "period_movement", _period_movement)
    return state


def _funded_inputs() -> sources.GovernedInputs:
    """A monthly funded approval, resolved exactly as the trigger resolves it."""
    return sources.resolve(
        tenant_id=TENANT, portfolio_id=TENANT, portfolio_context="total",
        pipeline_root=PIPELINE_ROOT, output_root=OUTPUT_ROOT,
        want_pipeline=False, want_funded=True)


# --------------------------------------------------------------------------- #
# 1. Funded monthly review + concentration available
# --------------------------------------------------------------------------- #
def test_a_funded_only_approval_evaluates_concentration(governed):
    """The regression itself: the funded path must reach the risk domain.

    Before the fix this returned ``None`` — the service was only ever called
    from the pipeline side, which a funded-only approval does not run.
    """
    inputs = _funded_inputs()

    assert inputs.concentration is not None
    assert inputs.concentration["available"] is True
    assert CAP_CONCENTRATION not in inputs.unavailable
    assert not sources.unevaluated_risk_domains(inputs)


# --------------------------------------------------------------------------- #
# 2. Funded monthly review + concentration unavailable
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("outcome, expected_reason", [
    # The controlled shape: a governed service saying so in its own words.
    ({"available": False, "reason": "no approved concentration configuration"},
     "no approved concentration configuration"),
    # The exceptional shape: _safe records it.
    (RuntimeError("the concentration engine is down"), None),
])
def test_unavailable_concentration_cannot_produce_an_unqualified_clear(
        governed, outcome, expected_reason):
    """Neither shape of absence may read as a pass.

    The controlled shape is the one that slipped through: ``_safe`` only records
    a capability when the call RAISES, so a service returning
    ``{"available": False}`` left ``unavailable`` empty and the message read the
    silence as a clear position.
    """
    governed["concentration"] = outcome
    inputs = _funded_inputs()

    assert CAP_CONCENTRATION in inputs.unavailable
    if expected_reason:
        assert inputs.unavailable[CAP_CONCENTRATION] == expected_reason
    assert CAP_CONCENTRATION in sources.unavailable_summary(inputs)

    message = risk_review.build(inputs)
    assert message.severity == SEVERITY_CLEAR
    # The weaker claim, about the checks — never the stronger one, about the book.
    assert risk_review.PARTIAL_STATEMENT in [i.text for i in message.items]
    assert risk_review.CLEAR_STATEMENT not in [i.text for i in message.items]
    assert CAP_CONCENTRATION in message.unavailable_checks
    # And no reassurance line for a check that did not run.
    assert not any("No current concentration breaches" in i.text
                   for i in message.items)


def test_no_funded_output_root_is_reported_not_assumed_clear(governed):
    """A deployment with no governed funded root evaluates nothing, and says so."""
    inputs = sources.resolve(
        tenant_id=TENANT, portfolio_id=TENANT, portfolio_context="total",
        pipeline_root=PIPELINE_ROOT, output_root="",
        want_pipeline=False, want_funded=True)

    assert CAP_CONCENTRATION in inputs.unavailable
    message = risk_review.build(inputs)
    assert risk_review.CLEAR_STATEMENT not in [i.text for i in message.items]


# --------------------------------------------------------------------------- #
# 3. Genuine all-clear
# --------------------------------------------------------------------------- #
def test_a_genuine_clear_month_still_states_the_strong_claim(governed):
    """The fix must not make every month read as incomplete.

    A concentration evaluation that ran and found nothing entitles the message
    to the claim about the PORTFOLIO, and to the reassurance lines with it.
    """
    inputs = _funded_inputs()
    message = risk_review.build(inputs)

    assert message.severity == SEVERITY_CLEAR
    assert risk_review.CLEAR_STATEMENT in [i.text for i in message.items]
    assert risk_review.PARTIAL_STATEMENT not in [i.text for i in message.items]
    assert message.unavailable_checks == []
    assert any("No current concentration breaches" in i.text
               for i in message.items)


# --------------------------------------------------------------------------- #
# 4. Partial evidence
# --------------------------------------------------------------------------- #
def test_partial_evidence_names_the_check_that_did_not_run(governed):
    """Concentration ran; the funded movement did not. Both facts survive."""
    governed["movement"] = {"available": False,
                            "reason": "only one funded period is available"}
    inputs = _funded_inputs()

    assert inputs.concentration["available"] is True
    assert sources.CAP_FUNDED_MOVEMENT in inputs.unavailable

    message = risk_review.build(inputs)
    assert risk_review.PARTIAL_STATEMENT in [i.text for i in message.items]
    assert sources.CAP_FUNDED_MOVEMENT in message.unavailable_checks
    # Concentration DID run, so its reassurance line is earned and kept.
    assert any("No current concentration breaches" in i.text
               for i in message.items)


# --------------------------------------------------------------------------- #
# 5 & 6. Breach and warning propagate through the real resolver
# --------------------------------------------------------------------------- #
def test_a_concentration_breach_reaches_the_funded_risk_review(governed):
    governed["concentration"] = concentration_risk(expected_breach=False)
    inputs = _funded_inputs()

    message = risk_review.build(inputs)
    assert message.severity == SEVERITY_CONCERN
    assert "South West" in message.headline
    assert risk_review.CLEAR_STATEMENT not in [i.text for i in message.items]


def test_a_concentration_warning_reaches_the_funded_risk_review(governed):
    governed["concentration"] = _warning_concentration()
    inputs = _funded_inputs()

    message = risk_review.build(inputs)
    assert message.severity == SEVERITY_ATTENTION
    assert "London" in message.headline
    assert risk_review.CLEAR_STATEMENT not in [i.text for i in message.items]


# --------------------------------------------------------------------------- #
# The invariant, stated once and tested directly
# --------------------------------------------------------------------------- #
def test_a_required_domain_with_no_evidence_is_always_reported():
    """The guard does not depend on the resolver having recorded anything.

    Constructed by hand precisely because a future resolver change might skip a
    domain WITHOUT recording it — which is the shape of the original defect. The
    guard asks for evidence that the domain ran, so an omission is self-
    reporting rather than silent.
    """
    bare = sources.GovernedInputs(tenant_id=TENANT, portfolio_id=TENANT)

    assert CAP_CONCENTRATION in sources.unevaluated_risk_domains(bare)
    assert CAP_CONCENTRATION in sources.unavailable_summary(bare)
    assert risk_review.CLEAR_STATEMENT not in [
        i.text for i in risk_review.build(bare).items]


def test_concentration_is_resolved_once_for_a_combined_update(governed):
    """A combined approval must not pay for the same evaluation twice."""
    calls = {"n": 0}
    inner = governed["concentration"]

    def _counting(output_root, client_id, funded_run_id, scope=None):
        calls["n"] += 1
        return inner

    from mi_agent_api import concentration_tests_api
    concentration_tests_api.compute_concentration_tests = _counting

    sources.resolve(
        tenant_id=TENANT, portfolio_id=TENANT, portfolio_context="total",
        pipeline_root=PIPELINE_ROOT, output_root=OUTPUT_ROOT,
        want_pipeline=True, want_funded=True)

    assert calls["n"] == 1
