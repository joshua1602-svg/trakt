"""Which file to believe, when a delivery's files disagree.

The rehearsal can now tell an operator that a pack disagrees with itself: the
loan tape says the rate is 5.10 and the property tape says 5.90 for the same
loan. What it could not do was let them answer. The remedy the central tape
builder names — an approved source-precedence rule — was a governed artefact
with no way in from the OCC at all, so the finding ended in a shrug.

This is not a mapping decision and must not be filed as one. Every source is
correctly mapped; they simply carry different values, and something has to say
which is authoritative. Without that the builder refuses the field and raises
a blocking gap, which is the right behaviour and exactly why the answer has to
be recordable.

The chain this has to complete, end to end:

    an operator's answer
      -> a governed rule, versioned and superseding, in the OCC's rule store
      -> a client-memory entry, via the projection the rules already use
      -> `13_source_precedence_rules.yaml`, which `central_tape_builder` reads

Each link already existed except the first, and the last two are tested here
against the ENGINE'S OWN readers rather than against a restatement of their
shape — a key spelled differently at either end is a rule the builder silently
ignores, which would be worse than no rule at all.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from operations_control.api import app as app_module
from operations_control.contracts import KIND_SOURCE_PRECEDENCE
from operations_control.rules import (
    RULE_ACTIVE,
    RULE_SUPERSEDED,
    project_rules_to_client_memory,
)

from .conftest import OP_A, OP_ADMIN, make_engine

ADMIN = {"X-Operator-Token": OP_ADMIN}
OPERATOR = {"X-Operator-Token": OP_A}

LOAN = "LoanExtract One - OMNI 2026_09_01.xlsx"
PROP = "PropertyExtract - Omni 2026_09_01.xlsx"


@pytest.fixture()
def api(store, source_registry):
    engine = make_engine(store, source_registry)
    app_module.set_engine(engine)
    client = TestClient(app_module.app, raise_server_exceptions=False)
    try:
        yield client, engine
    finally:
        app_module.set_engine(None)


def believe(client, *, primary=LOAN, primary_column="Loan Interest Rate",
            secondary=PROP, secondary_column="Interest Rate",
            field="current_interest_rate", client_id="client_a",
            reason="The loan tape is the system of record for pricing.",
            headers=ADMIN):
    return client.post("/ops/rules/precedence", headers=headers, json={
        "client_id": client_id, "canonical_field": field,
        "primary_source_file": primary, "primary_source_column": primary_column,
        "secondary_source_file": secondary,
        "secondary_source_column": secondary_column, "reason": reason})


class TestRecordingTheDecision:

    def test_a_precedence_decision_can_be_recorded(self, api):
        client, _engine = api
        response = believe(client)
        assert response.status_code == 201, response.text
        assert response.json()["rule"]["kind"] == KIND_SOURCE_PRECEDENCE

    def test_it_says_which_file_to_believe_in_plain_words(self, api):
        client, _engine = api
        rule = believe(client).json()["rule"]
        assert "current interest rate" in rule["description"]
        assert LOAN in rule["description"]

    def test_it_is_filed_as_its_own_kind_not_as_a_mapping(self, api):
        """Every source here is correctly mapped. Filing this as a mapping
        would say one of them was read wrongly, which is not what happened."""
        client, engine = api
        believe(client)
        kinds = {r.kind for r in engine.rules.list_current("client_a")}
        assert kinds == {KIND_SOURCE_PRECEDENCE}

    def test_the_reason_travels_with_it(self, api):
        client, engine = api
        why = "The loan tape is the system of record for pricing."
        believe(client, reason=why)
        rule = engine.rules.list_current("client_a")[0]
        assert rule.reason == why

    def test_changing_your_mind_supersedes_rather_than_stands_beside(self, api):
        """A field has one authoritative source, not a growing list of them."""
        client, engine = api
        first = believe(client).json()["rule"]
        second = believe(client, primary=PROP, primary_column="Interest Rate",
                         secondary=LOAN,
                         secondary_column="Loan Interest Rate").json()["rule"]

        assert second["version"] == first["version"] + 1
        current = engine.rules.list_current("client_a")
        assert len(current) == 1
        assert current[0].status == RULE_ACTIVE
        assert current[0].payload["primary_source_file"] == PROP

        history = engine.rules.history("client_a", first["rule_id"])
        assert any(r.status == RULE_SUPERSEDED for r in history), \
            "the earlier answer was overwritten rather than superseded"

    def test_a_decision_for_a_different_field_is_its_own_rule(self, api):
        client, engine = api
        believe(client)
        believe(client, field="current_outstanding_balance",
                primary_column="Current Outstanding Balance",
                secondary_column="Total OSBalance")
        assert len(engine.rules.list_current("client_a")) == 2


class TestWhatItRefuses:

    def test_a_decision_with_no_field_is_refused(self, api):
        client, engine = api
        assert believe(client, field="  ").status_code == 400
        assert engine.rules.list_current("client_a") == []

    def test_a_decision_naming_no_file_is_refused(self, api):
        client, engine = api
        assert believe(client, primary="   ").status_code == 400
        assert engine.rules.list_current("client_a") == []

    def test_preferring_a_file_over_itself_is_refused(self, api):
        """It reads as an answer and decides nothing."""
        client, engine = api
        response = believe(client, primary=LOAN, secondary=LOAN)
        assert response.status_code == 400
        assert engine.rules.list_current("client_a") == []

    def test_another_tenants_client_is_refused(self, api):
        client, engine = api
        response = believe(client, client_id="client_b", headers=OPERATOR)
        assert response.status_code in (403, 404)
        assert engine.rules.list_current("client_b") == []

    def test_an_unauthenticated_caller_is_refused(self, api):
        client, engine = api
        response = client.post("/ops/rules/precedence", json={
            "client_id": "client_a", "canonical_field": "current_interest_rate",
            "primary_source_file": LOAN})
        assert response.status_code == 401
        assert engine.rules.list_current("client_a") == []


class TestItReachesThePipeline:
    """The half that matters. A rule the builder never reads is not a rule."""

    def test_it_is_projected_into_the_clients_memory(self, api, tmp_path):
        client, engine = api
        believe(client)
        written = project_rules_to_client_memory(
            engine.rules.list_current("client_a"), "client_a",
            memory_dir=tmp_path / "memory")
        assert written == 1

    def test_the_engine_reads_it_back_as_a_precedence_rule(self, api, tmp_path):
        """Read back through the ENGINE'S OWN reader, not a restatement of
        its shape: a key spelled differently at either end is a rule the
        builder silently ignores."""
        from engine.onboarding_agent.mapping_memory import (
            MappingMemoryStore,
            precedence_rules_from_memory,
        )
        client, engine = api
        believe(client)
        memory_dir = tmp_path / "memory"
        project_rules_to_client_memory(engine.rules.list_current("client_a"),
                                       "client_a", memory_dir=memory_dir)

        rules = precedence_rules_from_memory(
            MappingMemoryStore(memory_dir, client_id="client_a"))

        assert "current_interest_rate" in rules, \
            "the engine's own reader did not find the decision"
        got = rules["current_interest_rate"]
        assert got["primary_source_file"] == LOAN
        assert got["primary_source_column"] == "Loan Interest Rate"
        assert got["secondary_source_file"] == PROP

    def test_the_builder_would_order_the_named_file_first(self, api, tmp_path):
        """The one thing a precedence rule has to cause. Asserted against the
        builder's own ordering function rather than by reading the YAML."""
        from engine.onboarding_agent.central_tape_builder import _order_sources
        from engine.onboarding_agent.mapping_memory import (
            MappingMemoryStore,
            precedence_rules_from_memory,
        )
        client, engine = api
        believe(client)
        memory_dir = tmp_path / "memory"
        project_rules_to_client_memory(engine.rules.list_current("client_a"),
                                       "client_a", memory_dir=memory_dir)
        precedence = precedence_rules_from_memory(
            MappingMemoryStore(memory_dir, client_id="client_a"))

        class _Src:
            def __init__(self, file_name):
                self.file_name = file_name
                self.classification = ""

        ordered = _order_sources("current_interest_rate",
                                 [_Src(PROP), _Src(LOAN)],
                                 precedence, {}, {})
        assert ordered[0].file_name == LOAN, \
            "the file the operator named was not preferred"

    def test_a_withdrawn_decision_stops_being_projected(self, api, tmp_path):
        client, engine = api
        rule = believe(client).json()["rule"]
        client.post(f"/ops/rules/{rule['rule_id']}/retire",
                    json={"reason": "the property tape is authoritative after "
                                    "all", "client": "client_a"},
                    headers=ADMIN)
        written = project_rules_to_client_memory(
            engine.rules.list_current("client_a"), "client_a",
            memory_dir=tmp_path / "memory")
        assert written == 0
