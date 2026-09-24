"""Onboard once; an unchanged month reports itself.

The operating model: onboarding takes less than a day, and every later
delivery whose source schema has not changed is reported without anyone
touching it. The publication screen offered "Yes — future deliveries for this
portfolio" and then only RECORDED the answer, so every month — and every
historic month of a backfill — waited for a click.

A "Yes" is now a standing approval, stored as a governed rule tied to the
schema fingerprint of the delivery a person approved, per dataset (funded and
pipeline are different sources). A later delivery publishes on it when its
schema matches, nothing is open and nothing was excepted — for management
reporting and, where the product includes it, the regulatory return. A first
delivery for a client or portfolio is always a person's sign-off.

And an earlier month never becomes "latest": publishing June after August
files June under its period and leaves MI reading August.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from operations_control.contracts import (RUN_AWAITING_PUBLICATION,
                                          RUN_PUBLISHED, WF_BACKFILL,
                                          WF_NEW_PORTFOLIO, WF_RECURRING)
from operations_control.engine import AUTO_PUBLICATION_ACTOR, OpsError

from .conftest import (make_client_config, make_engine, register_and_create,
                       start_and_wait, wait_for)


def _pack(root: Path, name: str, header: str = "loan_ref,balance,rate") -> Path:
    d = root / name
    d.mkdir()
    (d / "loans.csv").write_text(f"{header}\nL1,1000,0.05\nL2,2000,0.06\n",
                                 encoding="utf-8")
    return d


@pytest.fixture()
def engine(store, source_registry, tmp_path):
    return make_engine(store, source_registry, "happy",
                       client_config=make_client_config(tmp_path))


def _first_month(engine, tmp_path, *, scope="portfolio"):
    run = register_and_create(engine, _pack(tmp_path, "aug"),
                              period="2026-08-31",
                              workflow_type=WF_NEW_PORTFOLIO)
    start_and_wait(engine, run, statuses=(RUN_AWAITING_PUBLICATION,))
    engine.approve_publication(client_id="client_a",
                               workflow_id=run.workflow_id, actor="josh",
                               remember_scope=scope)
    return run


def _later_month(engine, tmp_path, name, period, *, header="loan_ref,balance,rate",
                 workflow_type=WF_RECURRING):
    run = register_and_create(engine, _pack(tmp_path, name, header),
                              period=period, workflow_type=workflow_type)
    start_and_wait(engine, run,
                   statuses=(RUN_PUBLISHED, RUN_AWAITING_PUBLICATION,
                             "needs_review", "blocked", "failed"))
    # "Awaiting publication" is a moment before the standing approval acts;
    # wait for the run's own thread to finish, then read where it settled.
    wait_for(lambda: not engine._is_executing(run.workflow_id))
    return engine.store.load_workflow(run.client_id, run.workflow_id)


class TestAnUnchangedMonthPublishesItself:

    def test_the_next_month_publishes_without_a_click(self, engine, tmp_path):
        _first_month(engine, tmp_path)
        sep = _later_month(engine, tmp_path, "sep", "2026-09-30")
        assert sep.status == RUN_PUBLISHED
        pub = engine.store.load_publication("client_a", "2026-09-30")
        assert pub["approved_by"] == AUTO_PUBLICATION_ACTOR
        audit = [a["event"] for a in engine.store.list_audit("client_a")]
        assert "publication_auto_approved" in audit

    def test_a_changed_schema_waits_for_a_person(self, engine, tmp_path):
        _first_month(engine, tmp_path)
        sep = _later_month(engine, tmp_path, "sep", "2026-09-30",
                           header="loan_ref,balance,rate,new_column")
        assert sep.status == RUN_AWAITING_PUBLICATION
        reasons = engine.auto_publication_refusals(sep)
        assert any("schema differs" in r for r in reasons)

    def test_no_standing_approval_means_a_person_approves(self, engine, tmp_path):
        _first_month(engine, tmp_path, scope="delivery")
        sep = _later_month(engine, tmp_path, "sep", "2026-09-30")
        assert sep.status == RUN_AWAITING_PUBLICATION

    def test_a_first_delivery_is_always_a_person_s_sign_off(self, engine,
                                                           tmp_path):
        _first_month(engine, tmp_path)
        again = _later_month(engine, tmp_path, "sep", "2026-09-30",
                             workflow_type=WF_NEW_PORTFOLIO)
        assert again.status == RUN_AWAITING_PUBLICATION


class TestABackfillPublishesAndLeavesLatestAlone:

    def test_an_older_month_publishes_but_is_not_latest(self, engine, tmp_path):
        _first_month(engine, tmp_path)
        jun = _later_month(engine, tmp_path, "jun", "2026-06-30",
                           workflow_type=WF_BACKFILL)
        assert jun.status == RUN_PUBLISHED
        pub = engine.store.load_publication("client_a", "2026-06-30")
        assert pub["published_artefacts"]["is_latest"] is False
        assert not pub["published_artefacts"].get("latest")
        aug = engine.store.load_publication("client_a", "2026-08-31")
        assert aug["published_artefacts"]["is_latest"] is True


class TestGrantingFromAPublishedDelivery:

    def test_a_published_delivery_can_stand_for_later_ones(self, engine,
                                                          tmp_path):
        aug = _first_month(engine, tmp_path, scope="delivery")
        granted = engine.grant_standing_publication(
            client_id="client_a", workflow_id=aug.workflow_id,
            scope="portfolio", actor="josh")
        assert granted["rule_id"]
        sep = _later_month(engine, tmp_path, "sep", "2026-09-30")
        assert sep.status == RUN_PUBLISHED

    def test_an_unpublished_delivery_cannot(self, engine, tmp_path):
        run = register_and_create(engine, _pack(tmp_path, "aug"),
                                  period="2026-08-31")
        start_and_wait(engine, run, statuses=(RUN_AWAITING_PUBLICATION,))
        with pytest.raises(OpsError):
            engine.grant_standing_publication(
                client_id="client_a", workflow_id=run.workflow_id,
                scope="portfolio", actor="josh")


class TestTheRegulatoryReturnToo:
    """Where the product includes Annex 2, an unchanged month's regulatory
    delivery publishes on the same standing approval."""

    def test_an_unchanged_regulatory_month_publishes(self, engine, tmp_path):
        from operations_control.contracts import OUTCOME_MI_ANNEX2
        aug = register_and_create(engine, _pack(tmp_path, "aug"),
                                  period="2026-08-31", outcome=OUTCOME_MI_ANNEX2,
                                  workflow_type=WF_NEW_PORTFOLIO)
        start_and_wait(engine, aug, statuses=(RUN_AWAITING_PUBLICATION,))
        engine.approve_publication(client_id="client_a",
                                   workflow_id=aug.workflow_id, actor="josh",
                                   remember_scope="portfolio")
        sep = register_and_create(engine, _pack(tmp_path, "sep"),
                                  period="2026-09-30", outcome=OUTCOME_MI_ANNEX2,
                                  workflow_type=WF_RECURRING)
        start_and_wait(engine, sep, statuses=(RUN_PUBLISHED,
                                              RUN_AWAITING_PUBLICATION,
                                              "needs_review", "blocked"))
        wait_for(lambda: not engine._is_executing(sep.workflow_id))
        sep = engine.store.load_workflow("client_a", sep.workflow_id)
        assert sep.status == RUN_PUBLISHED, engine.auto_publication_refusals(sep)

    def test_a_product_change_waits_for_a_person(self, engine, tmp_path):
        from operations_control.contracts import OUTCOME_MI_ANNEX2
        _first_month(engine, tmp_path)                   # MI only approved
        sep = register_and_create(engine, _pack(tmp_path, "sep"),
                                  period="2026-09-30", outcome=OUTCOME_MI_ANNEX2,
                                  workflow_type=WF_RECURRING)
        start_and_wait(engine, sep, statuses=(RUN_PUBLISHED,
                                              RUN_AWAITING_PUBLICATION,
                                              "needs_review", "blocked"))
        wait_for(lambda: not engine._is_executing(sep.workflow_id))
        sep = engine.store.load_workflow("client_a", sep.workflow_id)
        assert sep.status != RUN_PUBLISHED


class TestEveryApprovedSchemaStaysApproved:

    def test_approving_an_older_layout_keeps_the_current_one(self, engine,
                                                            tmp_path):
        _first_month(engine, tmp_path)                       # August layout
        old = _later_month(engine, tmp_path, "mar", "2026-03-31",
                           header="loan_ref,balance",         # older layout
                           workflow_type=WF_BACKFILL)
        assert old.status == RUN_AWAITING_PUBLICATION
        engine.approve_publication(client_id="client_a",
                                   workflow_id=old.workflow_id, actor="josh",
                                   remember_scope="portfolio")
        sep = _later_month(engine, tmp_path, "sep", "2026-09-30")
        assert sep.status == RUN_PUBLISHED
        feb = _later_month(engine, tmp_path, "feb", "2026-02-28",
                           header="loan_ref,balance", workflow_type=WF_BACKFILL)
        assert feb.status == RUN_PUBLISHED
