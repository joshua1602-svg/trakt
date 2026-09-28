"""Reversing a decision, which the rules surface could not do.

A standing rule — "treat 'Curr Bal' as the current outstanding balance" — is
read on every future delivery. Approving one is easy and well covered. Taking
one back was impossible from anywhere an operator can reach: ``RuleStore``
has had ``retire()`` all along, with no route and no button, and the Rules
screen is a reader — search, filter, expand to see history.

So a mapping confirmed in error could only be undone by chance: by a later
delivery happening to raise the same question again, answered differently, and
superseding it. That is not a reversal, it is a hope.

Two properties this must hold:

* **it is not a delete.** The rule keeps its versions and its history, and
  what it did while it was in force stays readable. Nothing about withdrawing
  a decision justifies losing the record of having made it.
* **the reason is not optional.** Withdrawing a rule changes what the platform
  will do with data nobody has sent yet, and in six months the only thing that
  explains that is what was written at the time.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from operations_control.api import app as app_module
from operations_control.contracts import KIND_FIELD_MAPPING
from operations_control.rules import (
    RULE_ACTIVE,
    RULE_RETIRED,
    RuleRecord,
)

from .conftest import OP_A, OP_ADMIN, make_engine

ADMIN = {"X-Operator-Token": OP_ADMIN}
OPERATOR = {"X-Operator-Token": OP_A}


@pytest.fixture()
def api(store, source_registry):
    engine = make_engine(store, source_registry)
    app_module.set_engine(engine)
    client = TestClient(app_module.app, raise_server_exceptions=False)
    try:
        yield client, engine
    finally:
        app_module.set_engine(None)


def approve_mapping(engine, client_id="client_a", column="Curr Bal",
                    canonical="current_outstanding_balance"):
    return engine.rules.approve(RuleRecord(
        rule_id="", version=0, kind=KIND_FIELD_MAPPING, scope="client",
        client_id=client_id,
        payload={"source_column": column, "canonical_field": canonical},
        description=f"Treat '{column}' as '{canonical.replace('_', ' ')}'.",
        approved_by="Alice", reason="confirmed on the first delivery"))


class TestWithdrawingARule:

    def test_a_rule_in_force_can_be_withdrawn(self, api):
        client, engine = api
        rule = approve_mapping(engine)

        response = client.post(f"/ops/rules/{rule.rule_id}/retire",
                               json={"reason": "The column was the ORIGINAL "
                                               "balance, not the current one.",
                                     "client": "client_a"},
                               headers=ADMIN)
        assert response.status_code == 200, response.text
        assert response.json()["rule"]["status"] == RULE_RETIRED

    def test_it_stops_being_applied(self, api):
        client, engine = api
        rule = approve_mapping(engine)
        assert any(r.rule_id == rule.rule_id
                   for r in engine.rules.list_current("client_a"))

        client.post(f"/ops/rules/{rule.rule_id}/retire",
                    json={"reason": "wrong column", "client": "client_a"},
                    headers=ADMIN)

        applicable = engine.rules.applicable(client_id="client_a")
        assert not any(r.rule_id == rule.rule_id for r in applicable), \
            "a withdrawn rule was still being applied"

    def test_it_is_not_a_delete(self, api):
        """The record of having made the decision outlives the decision."""
        client, engine = api
        rule = approve_mapping(engine)
        client.post(f"/ops/rules/{rule.rule_id}/retire",
                    json={"reason": "wrong column", "client": "client_a"},
                    headers=ADMIN)

        history = client.get(f"/ops/rules/{rule.rule_id}/history",
                             params={"client": "client_a"}, headers=ADMIN)
        assert history.status_code == 200
        assert history.json()["history"], "the rule's history was lost"

    def test_the_reason_is_kept_with_it(self, api):
        client, engine = api
        rule = approve_mapping(engine)
        why = "The column was the ORIGINAL balance, not the current one."

        client.post(f"/ops/rules/{rule.rule_id}/retire",
                    json={"reason": why, "client": "client_a"}, headers=ADMIN)

        assert engine.rules.get("client_a", rule.rule_id).reason == why

    def test_who_withdrew_it_is_recorded(self, api):
        client, engine = api
        rule = approve_mapping(engine)
        client.post(f"/ops/rules/{rule.rule_id}/retire",
                    json={"reason": "wrong column", "client": "client_a"},
                    headers=ADMIN)
        assert engine.rules.get("client_a", rule.rule_id).approved_by


class TestWhatItRefuses:

    def test_a_withdrawal_with_no_reason_is_refused(self, api):
        client, engine = api
        rule = approve_mapping(engine)

        response = client.post(f"/ops/rules/{rule.rule_id}/retire",
                               json={"reason": "   ", "client": "client_a"}, headers=ADMIN)
        assert response.status_code == 400
        assert engine.rules.get("client_a", rule.rule_id).status == RULE_ACTIVE

    def test_withdrawing_twice_says_there_is_nothing_to_withdraw(self, api):
        client, engine = api
        rule = approve_mapping(engine)
        client.post(f"/ops/rules/{rule.rule_id}/retire",
                    json={"reason": "wrong column", "client": "client_a"},
                    headers=ADMIN)

        again = client.post(f"/ops/rules/{rule.rule_id}/retire",
                            json={"reason": "wrong column", "client": "client_a"},
                    headers=ADMIN)
        assert again.status_code == 409

    def test_a_rule_that_does_not_exist_is_not_found(self, api):
        client, _engine = api
        response = client.post("/ops/rules/rule_nope/retire",
                               json={"reason": "whatever",
                                     "client": "client_a"}, headers=ADMIN)
        assert response.status_code == 404

    def test_another_tenants_rule_is_not_reachable(self, api):
        """The same tenancy rule the rest of the API keeps: not 403, which
        would confirm the rule exists, but not-found."""
        client, engine = api
        rule = approve_mapping(engine, client_id="client_b")

        response = client.post(f"/ops/rules/{rule.rule_id}/retire",
                               json={"reason": "not mine to touch",
                                     "client": "client_b"},
                               headers=OPERATOR)
        assert response.status_code == 404
        assert engine.rules.get("client_b", rule.rule_id).status == RULE_ACTIVE

    def test_an_unauthenticated_caller_is_refused(self, api):
        client, engine = api
        rule = approve_mapping(engine)
        response = client.post(f"/ops/rules/{rule.rule_id}/retire",
                               json={"reason": "wrong column",
                                     "client": "client_a"})
        assert response.status_code == 401
        assert engine.rules.get("client_a", rule.rule_id).status == RULE_ACTIVE
