"""Acceptance: an acquisition month, end to end, and a fourth book for free.

Runs the real pathway with two things replaced — the governed frame discovery
(which would need a portfolio on disk) and the Bot Framework network call.
Everything between them is production code: the concentration resolution, the
funded generators, the composition decomposition, the underlying lens, the
message contract, recipient authorisation, the outbox, card rendering, ordering
and delivery.

The scale claim is tested rather than asserted. ``portfolio_alpha`` /
``portfolio_beta`` / ``portfolio_gamma`` / ``portfolio_delta`` are names no
production module knows, and the last test adds a further acquired book by
changing data alone.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from mi_agent_api import funded_composition as comp
from trakt_notifications import cards, generate, sources, trigger
from trakt_notifications.contract import (
    MESSAGE_PORTFOLIO_UPDATE, MESSAGE_RISK_REVIEW, UPDATE_FUNDED,
)
from trakt_notifications.delivery import DeliveryWorker
from trakt_notifications.outbox import Outbox, STATE_SENT
from trakt_notifications.store import BatchStore
from trakt_notifications.teams_client import SendResult, TeamsClient

from .conftest import TENANT

BALANCE, LOAN, PORTFOLIO = ("current_outstanding_balance", "loan_identifier",
                            "source_portfolio_id")
LABEL, PTYPE = "source_portfolio_label", "source_portfolio_type"


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False)


def _frame(rows) -> pd.DataFrame:
    return pd.DataFrame([{LOAN: l, BALANCE: b, PORTFOLIO: p, LABEL: lab, **x}
                         for l, b, p, lab, x in rows])


# --------------------------------------------------------------------------- #
# The month: £112m incumbent book, a £68m book acquired, £4m of its own growth
# --------------------------------------------------------------------------- #
PRIOR = _frame([
    ("A1", 60_000_000.0, "portfolio_alpha", "Direct Book", {}),
    ("A2", 52_000_000.0, "portfolio_alpha", "Direct Book", {}),
])
CURRENT = _frame([
    ("A1", 61_000_000.0, "portfolio_alpha", "Direct Book", {}),
    ("A2", 52_000_000.0, "portfolio_alpha", "Direct Book", {}),
    ("A3", 3_000_000.0, "portfolio_alpha", "Direct Book", {}),
    ("B1", 40_000_000.0, "portfolio_beta", "Portfolio B",
     {PTYPE: "acquired", "acquisition_date": "2026-07-15"}),
    ("B2", 28_000_000.0, "portfolio_beta", "Portfolio B",
     {PTYPE: "acquired", "acquisition_date": "2026-07-15"}),
])


def _frames(current, prior):
    return [
        {"run_id": "mi_2026_06", "reporting_date": "2026-06-30", "df": prior,
         "source": "/p/prior.csv"},
        {"run_id": "mi_2026_07", "reporting_date": "2026-07-31", "df": current,
         "source": "/p/current.csv"},
    ]


def _movement(current, prior) -> dict:
    """A governed period movement consistent with the frames above."""
    def _bal(df):
        return round(float(df[BALANCE].sum()), 2)
    return {
        "available": True,
        "currentReportingDate": "2026-07-31",
        "priorReportingDate": "2026-06-30",
        "current": {"funded_balance": _bal(current), "loan_count": len(current),
                    "wa_ltv_points": 29.4},
        "prior": {"funded_balance": _bal(prior), "loan_count": len(prior),
                  "wa_ltv_points": 31.0},
        "delta": {"funded_balance": round(_bal(current) - _bal(prior), 2),
                  "loan_count": len(current) - len(prior),
                  "wa_ltv_points": -1.6},
        "regionContributions": [], "cohortMovements": [], "primaryRegion": None,
    }


LONDON_TEST = {
    "testId": "region_ldn", "displayName": "London exposure",
    "status": "warning", "priorStatus": "pass",
    "statusTransition": "pass -> warning", "deteriorated": True,
    "currentValue": 0.284, "priorValue": 0.240, "threshold": 0.30,
    "utilization": 0.947, "headroom": 1.6, "unit": "percent",
    "reportingDate": "2026-07-31", "priorReportingDate": "2026-06-30",
}


@pytest.fixture
def governed(monkeypatch):
    """The governed services, over the frames above. Everything else is real."""
    from mi_agent_api import concentration_tests_api as conc_mod
    from mi_agent_api import evolution as evolution_mod
    from mi_agent_api import movement_summary as movement_mod

    state = {"current": CURRENT, "prior": PRIOR,
             "concentration": {
                 "available": True, "reportingDate": "2026-07-31",
                 "source": "approved_configuration",
                 "tests": [LONDON_TEST], "emergingRisks": [],
                 "states": {"available": True},
                 "lineage": {"configurationVersion": "v3"}}}

    monkeypatch.setattr(evolution_mod, "funded_frames",
                        lambda *a, **k: _frames(state["current"], state["prior"]))
    monkeypatch.setattr(movement_mod, "period_movement",
                        lambda *a, **k: _movement(state["current"], state["prior"]))
    monkeypatch.setattr(conc_mod, "compute_concentration_tests",
                        lambda *a, **k: state["concentration"])
    # The trigger resolves its roots from the environment, as production does —
    # it takes no root parameter, which is what keeps a caller from naming one.
    # The frames above are still what the services return; this only lets the
    # resolver get past "no governed root is configured".
    monkeypatch.setenv("MI_AGENT_ONBOARDING_OUTPUT_ROOT", "/governed/funded")
    monkeypatch.setenv("MI_AGENT_PIPELINE_ROOT", "/governed/pipeline")
    return state


def _resolve() -> sources.GovernedInputs:
    return sources.resolve(
        tenant_id=TENANT, portfolio_id=TENANT, portfolio_context="total",
        pipeline_root="/governed/pipeline", output_root="/governed/funded",
        want_pipeline=False, want_funded=True)


# =========================================================================== #
# E. The acquisition month
# =========================================================================== #
def test_the_month_is_decomposed_before_it_is_narrated(governed):
    inputs = _resolve()
    insights = {i["insight_type"]: i for i in inputs.funded_insights()}

    composition = insights["FUNDED_COMPOSITION"]
    assert composition["metrics"]["portfolio_additions"] == 68_000_000.0
    assert composition["metrics"]["organic_new_lending"] == 3_000_000.0
    assert composition["metrics"]["existing_book_movement"] == 1_000_000.0
    assert composition["methodology"]["reconciliation"]["reconciles"] is True


def test_the_card_says_what_the_acquisition_accounted_for(governed):
    from trakt_notifications import portfolio_update

    message = portfolio_update.build(_resolve(), update_type=UPDATE_FUNDED)
    body = " ".join(i.text for i in message.items)

    assert "£68.0m of the £72.0m movement reflects the acquisition of " \
           "Portfolio B" in body
    assert "Funded balance is £184.0m" in (message.summary or "")


def test_the_underlying_book_is_reported_beside_the_headline(governed):
    from trakt_notifications import portfolio_update

    body = " ".join(i.text for i in
                    portfolio_update.build(_resolve(),
                                           update_type=UPDATE_FUNDED).items)
    # The incumbent book grew £4m on £112m while the headline says £72m.
    assert "the existing book increased by £4.0m (+3.6%)" in body


def test_the_limit_crossing_leads_the_risk_review(governed):
    from trakt_notifications import risk_review

    message = risk_review.build(_resolve())
    # An insight-derived finding's headline is already a sentence, so the
    # renderer lowercases its first character; a test-derived one is a name and
    # keeps its case.
    assert message.headline == "Risk Review — london exposure: pass → warning"
    assert message.severity == "attention"
    assert risk_review.CLEAR_STATEMENT not in [i.text for i in message.items]


def test_the_whole_pathway_reaches_a_teams_card(governed, storage, layout,
                                                recipient, enabled_config):
    """Approved run → MI → insights → outbox → Teams, with the real code."""
    class _RecordingClient(TeamsClient):
        """Stubbed at the send boundary, as the rest of this suite is.

        The transport underneath enforces the service-URL allowlist, which is a
        separate rule with its own tests; stubbing below it here would make this
        acceptance test depend on that rule holding.
        """

        def __init__(self):
            super().__init__(app_id="app", app_password="secret")
            self.sent = []

        def send_card(self, *, service_url, conversation_id, attachment,
                      summary):
            self.sent.append(attachment["content"])
            return SendResult(teams_message_id=f"msg-{len(self.sent)}",
                              status=201)

    outcome = trigger.on_publication_approved(
        tenant_id=TENANT, portfolio_id=TENANT, datasets=["funded"],
        approved_run_ids=["mi_2026_07"], run_status="published",
        storage=storage, layout=layout)

    assert outcome.sent_to_outbox is True
    assert outcome.recipients == 1
    assert len(outcome.outbox_item_ids) == 2       # update + risk review

    client = _RecordingClient()
    report = DeliveryWorker(storage=storage, layout=layout,
                            client=client).run(TENANT)
    assert report.sent == 2

    update = _json(client.sent[0])
    assert "Monthly Funded Update" in update
    assert "the acquisition of Portfolio B" in update
    assert "the existing book increased by £4.0m" in update

    risk = _json(client.sent[1])
    assert "London exposure deteriorated from pass to warning" in risk
    assert "Utilisation is 94.7% of the limit" in risk


def test_a_second_approval_of_the_same_run_sends_nothing(governed, storage,
                                                         layout, recipient,
                                                         enabled_config):
    """Deterministic batch identity: re-approving cannot re-notify."""
    first = trigger.on_publication_approved(
        tenant_id=TENANT, portfolio_id=TENANT, datasets=["funded"],
        approved_run_ids=["mi_2026_07"], run_status="published",
        storage=storage, layout=layout)
    second = trigger.on_publication_approved(
        tenant_id=TENANT, portfolio_id=TENANT, datasets=["funded"],
        approved_run_ids=["mi_2026_07"], run_status="published",
        storage=storage, layout=layout)

    assert first.sent_to_outbox is True
    assert second.sent_to_outbox is False
    assert second.suppressed_reason == trigger.SUPPRESS_DUPLICATE
    assert second.notification_batch_id == first.notification_batch_id


def test_nothing_is_sent_while_delivery_is_disabled(governed, storage, layout,
                                                    recipient, monkeypatch):
    """Configuration, not code, is the switch — and it is off by default."""
    monkeypatch.delenv("TRAKT_TEAMS_NOTIFICATIONS", raising=False)
    monkeypatch.delenv("TRAKT_TEAMS_NOTIFICATIONS_CONFIG", raising=False)

    outcome = trigger.on_publication_approved(
        tenant_id=TENANT, portfolio_id=TENANT, datasets=["funded"],
        approved_run_ids=["mi_2026_07"], run_status="published",
        storage=storage, layout=layout)

    assert outcome.sent_to_outbox is False
    assert outcome.suppressed_reason == trigger.SUPPRESS_DISABLED
    assert Outbox(storage, layout).due(TENANT) == []


# =========================================================================== #
# F. Scale — a fourth book is data, not a code change
# =========================================================================== #
def test_a_further_acquired_portfolio_needs_no_production_change(governed):
    """The only thing that changes is the frame.

    No module is edited, no id is registered, no branch is added. A fourth book
    arrives with a governed source_portfolio_id, is absent from the prior frame,
    and decomposes — which is the whole scalability claim, tested rather than
    asserted.
    """
    governed["current"] = _frame([
        ("A1", 61_000_000.0, "portfolio_alpha", "Direct Book", {}),
        ("A2", 52_000_000.0, "portfolio_alpha", "Direct Book", {}),
        ("A3", 3_000_000.0, "portfolio_alpha", "Direct Book", {}),
        ("B1", 40_000_000.0, "portfolio_beta", "Portfolio B",
         {PTYPE: "acquired"}),
        ("B2", 28_000_000.0, "portfolio_beta", "Portfolio B",
         {PTYPE: "acquired"}),
        ("C1", 25_000_000.0, "portfolio_gamma", "Portfolio C",
         {PTYPE: "acquired"}),
        ("D1", 11_000_000.0, "portfolio_delta", "Portfolio D",
         {PTYPE: "acquired"}),
    ])

    inputs = _resolve()
    composition = next(i for i in inputs.funded_insights()
                       if i["insight_type"] == "FUNDED_COMPOSITION")

    added = composition["contributors"]["portfolio_additions"]
    assert [p["source_portfolio_id"] for p in added] == [
        "portfolio_beta", "portfolio_delta", "portfolio_gamma"]
    assert composition["metrics"]["portfolio_additions"] == 104_000_000.0
    assert composition["methodology"]["reconciliation"]["reconciles"] is True


def test_the_underlying_lens_still_names_only_the_incumbent_books(governed):
    governed["current"] = _frame([
        ("A1", 61_000_000.0, "portfolio_alpha", "Direct Book", {}),
        ("E1", 9_000_000.0, "portfolio_epsilon", "Portfolio E",
         {PTYPE: "acquired"}),
    ])
    decomposition = comp.decompose(governed["current"], governed["prior"])

    assert comp.underlying_lens_filters(decomposition) == {
        "source_portfolio_id": ["portfolio_alpha"]}


def test_no_production_module_names_a_portfolio_id():
    """The scalability claim as a property of the source, not of one fixture."""
    import pathlib
    import re

    banned = re.compile(r"(acquired_00\d|direct_00\d|portfolio_alpha|SPV2)")
    roots = [pathlib.Path("mi_agent_api/funded_composition.py"),
             pathlib.Path("mi_agent_api/insight_generators_funded.py"),
             pathlib.Path("trakt_tools/handlers/portfolio_review.py"),
             pathlib.Path("portfolio_review/controller.py"),
             pathlib.Path("portfolio_review/objective.py")]
    for path in roots:
        assert not banned.search(path.read_text(encoding="utf-8")), path
