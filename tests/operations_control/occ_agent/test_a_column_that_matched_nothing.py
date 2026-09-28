"""A column nothing matched used to be a dead end.

THE GAP, IN THE OPERATOR'S WORDS

    "For unmapped fields, there should still be an option to i) add to field
    registry as a new entry, or ii) add to an existing field in the field
    registry as an alias."

The table labelled such a column "Not used" and stopped there. On a first
delivery that is most of the tape — eighty-nine of a hundred and fifty three
for the first client through this — and an operator who KNEW what a column was
had nowhere to write it down. The knowledge that would have fixed the mapping
was in the room and the screen had no way to take it.

THE TWO ACTS ARE NOT ONE ACT, AND THESE TESTS HOLD THEM APART

* Naming a field Trakt already has is an ALIAS. It is this client's, it is
  settled here and now, and it promotes at activation into a governed rule
  scoped to the portfolio like any other mapping an operator approves.

* Asking for a field Trakt does NOT have is a change to the platform's
  canonical vocabulary — which every client, every regime projection and every
  validation rule reads. ``operations_control.rules`` is explicit that "the
  core field registry is never written" from this container, and the registry
  has its own governed route: a versioned system config package an
  administrator drafts and activates.

So the second is recorded as a REQUEST. It maps nothing, the column stays
unused, and the ask travels with the case. An onboarding operator adding a
canonical field with one click would be one client's first delivery changing
what every other client's report means.
"""

from __future__ import annotations

import pytest

from operations_control.engine import OpsError
from operations_control.occ_agent import field_registry, mapping_promotion
from operations_control.occ_agent import states as _states
from operations_control.occ_agent.scenarios import run_scenario

from .conftest import ACTOR, TENANT_A


#: The column in the prepared example that matches nothing.
#:
#: A REAL column of a real file — ``fixtures.property_tape`` carries it — not a
#: row pushed into the report by this test. That matters here more than
#: elsewhere: committing the set reruns the onboarding, which rebuilds the
#: report from the files, and a fabricated row would vanish at exactly the
#: moment these tests are asserting on it.
UNPLACEABLE = "Broker Code"


@pytest.fixture()
def halted(service):
    """A case stopped where the mapping table is the operator's screen, with
    one column in its delivery that nothing matched."""
    run = run_scenario(service, "scenario_b_ambiguous_mapping",
                       tenant=TENANT_A, actor=ACTOR, resolve_decisions=False)
    assert run.case.run.state == _states.EXCEPTIONS_REQUIRE_INPUT
    return run.case


def _unused(service, agent_case):
    """One column the mapper could not place, from the run's own report."""
    rows = service.status(agent_case)["mapping"]["rows"]
    unused = [r for r in rows if r["state"] == "unused"]
    assert unused, "the fixture no longer contains an unmapped column"
    return unused[0]


def _commit(service, agent_case, actor=ACTOR):
    """Answer whatever is still genuinely open, then confirm the set.

    The commit refuses while a weak match or an ambiguity has no answer, which
    is the point of it — so a test about what a COMMITTED mapping does has to
    answer them first, exactly as an operator would. Where two columns claim
    one field the first wins and the rest are set aside, because "confirm
    both" is not an answer to "which of these is it?".
    """
    taken = set()
    for row in service.status(agent_case)["mapping"]["rows"]:
        if row["state"] != "needs_you":
            continue
        field = row["canonical_field"]
        action = "confirm" if field and field not in taken else "not_used"
        taken.add(field)
        agent_case = service.stage_mapping(
            agent_case, source_file=row["source_file"],
            source_column=row["source_column"], action=action, actor=actor)
    return service.confirm_mappings(agent_case, actor=actor)


class TestTheFieldsAnOperatorMayChooseFrom:
    def test_the_picker_is_the_mappers_own_selection(self, service, halted):
        """A field offered on the screen must be one the run will accept.

        Two lists would drift, and the drift shows up as an operator choosing
        a field and being refused for reasons the screen cannot explain.
        """
        from engine.gate_1_alignment.semantic_alignment import (
            load_field_registry,
            select_registry_fields,
        )
        facts = service.facts(halted)
        registry = load_field_registry(field_registry.REGISTRY_PATH)
        expected = set(select_registry_fields(registry, facts.asset_class))
        offered = {f["name"] for f in service.field_catalogue(halted)}
        assert offered == expected

    def test_each_field_says_which_obligation_it_answers(self, service,
                                                         halted):
        """Choosing between two plausible fields is choosing between two
        regulatory obligations, and a picker that hides that invites a
        guess."""
        catalogue = service.field_catalogue(halted)
        annexed = [f for f in catalogue if f["regimes"]]
        assert annexed, "no field in the catalogue carries a regime code"
        assert all(isinstance(f["label"], str) and f["label"]
                   for f in catalogue)


class TestNamingAFieldTraktAlreadyHas:
    def test_naming_it_is_a_draft_until_the_set_is_confirmed(self, service,
                                                             halted):
        """It reads as answered and it has not been applied.

        Nothing about the column is resolved, promoted or rerun until the
        operator commits — which is what lets them change their mind about it.
        """
        row = _unused(service, halted)
        updated = service.map_unmapped_column(
            halted, source_file=row["source_file"],
            source_column=row["source_column"],
            target_field="current_principal_balance", actor=ACTOR,
            reason="their name for the outstanding balance")
        after = {(r["source_file"], r["source_column"]): r
                 for r in service.status(updated)["mapping"]["rows"]}
        settled = after[(row["source_file"], row["source_column"])]
        assert settled["state"] == "staged"
        assert settled["staged_field"] == "current_principal_balance"
        # Not applied: the report still says nothing matched it, and no
        # decision asserts a mapping.
        assert settled["canonical_field"] == ""
        assert not [d for d in updated.run.open_decisions
                    if (d.get("subject") or {}).get("artefact")
                    == "unmapped_column"]

    def test_the_column_stops_being_unused_once_confirmed(self, service,
                                                          halted):
        row = _unused(service, halted)
        updated = service.map_unmapped_column(
            halted, source_file=row["source_file"],
            source_column=row["source_column"],
            target_field="current_principal_balance", actor=ACTOR,
            reason="their name for the outstanding balance")
        updated = _commit(service, updated)
        after = {(r["source_file"], r["source_column"]): r
                 for r in service.status(updated)["mapping"]["rows"]}
        settled = after[(row["source_file"], row["source_column"])]
        assert settled["state"] == "confirmed"
        assert settled["canonical_field"] == "current_principal_balance"

    def test_it_is_recorded_as_the_operators_own_answer(self, service,
                                                        halted):
        row = _unused(service, halted)
        updated = service.map_unmapped_column(
            halted, source_file=row["source_file"],
            source_column=row["source_column"],
            target_field="current_principal_balance", actor=ACTOR)
        updated = _commit(service, updated)
        decision = next(d for d in updated.run.open_decisions
                        if (d.get("subject") or {}).get("artefact")
                        == "unmapped_column")
        assert decision["status"] == "approved"
        assert decision["resolved_by"] == ACTOR
        assert decision["subject"]["source_file"] == row["source_file"]

    def test_it_promotes_into_a_governed_rule(self, service, halted):
        """The point of recording it as a decision at all.

        An alias that lived only on the run would be re-derived from scratch
        next month and the same column would come back unmapped — which is the
        gap promotion was built to close.
        """
        row = _unused(service, halted)
        updated = service.map_unmapped_column(
            halted, source_file=row["source_file"],
            source_column=row["source_column"],
            target_field="current_principal_balance", actor=ACTOR)
        updated = _commit(service, updated)
        decision = next(d for d in updated.run.open_decisions
                        if (d.get("subject") or {}).get("artefact")
                        == "unmapped_column")
        mapping = mapping_promotion.mapping_of(decision)
        assert mapping == {"source_column": row["source_column"],
                           "canonical_field": "current_principal_balance"}

    def test_a_field_trakt_does_not_report_on_is_refused(self, service,
                                                         halted):
        """A free-text target would promote into a governed rule that silently
        matches nothing, every month, without saying so."""
        row = _unused(service, halted)
        with pytest.raises(OpsError) as caught:
            service.map_unmapped_column(
                halted, source_file=row["source_file"],
                source_column=row["source_column"],
                target_field="made_up_field", actor=ACTOR)
        assert caught.value.code == "OCC_AGENT_FIELD_NOT_REGISTERED"

    def test_a_column_that_is_not_in_the_delivery_is_refused(self, service,
                                                             halted):
        row = _unused(service, halted)
        with pytest.raises(OpsError) as caught:
            service.map_unmapped_column(
                halted, source_file=row["source_file"],
                source_column="No Such Column",
                target_field="current_principal_balance", actor=ACTOR)
        assert caught.value.code == "OCC_AGENT_COLUMN_NOT_FOUND"


class TestAskingForAFieldTraktDoesNotHave:
    def test_it_records_a_request_and_maps_nothing(self, service, halted):
        """The distinction the whole design turns on.

        The row reads STAGED rather than unused, because the ask is an answer
        about the column — "Trakt has no field for this" — and is held as a
        draft like every other answer. What matters is unchanged and asserted
        here: nothing is mapped, and the ask is on the row. See
        :mod:`tests.operations_control.occ_agent
        .test_a_requested_column_is_not_also_mapped` for why the set-aside is
        the load-bearing half.
        """
        row = _unused(service, halted)
        updated = service.request_registry_field(
            halted, source_file=row["source_file"],
            source_column=row["source_column"], field_name="broker_code",
            description="The intermediary who introduced the case.",
            data_type="string", actor=ACTOR)
        assert [r["field_name"] for r in updated.run.field_requests] \
            == ["broker_code"]
        after = {(r["source_file"], r["source_column"]): r
                 for r in service.status(updated)["mapping"]["rows"]}
        still = after[(row["source_file"], row["source_column"])]
        assert still["state"] == "staged"
        assert still["staged_action"] == "not_used"
        assert still["canonical_field"] == ""
        assert still["requested_field"] == "broker_code"

    def test_the_core_registry_is_not_written(self, service, halted):
        """``operations_control.rules``: the core field registry is never
        written from here. A first delivery from one client must not change
        what every other client's report is written in."""
        before = field_registry.REGISTRY_PATH.read_bytes()
        row = _unused(service, halted)
        service.request_registry_field(
            halted, source_file=row["source_file"],
            source_column=row["source_column"], field_name="broker_code",
            actor=ACTOR)
        assert field_registry.REGISTRY_PATH.read_bytes() == before

    def test_the_request_says_who_can_act_on_it(self, service, halted):
        row = _unused(service, halted)
        updated = service.request_registry_field(
            halted, source_file=row["source_file"],
            source_column=row["source_column"], field_name="broker_code",
            description="The intermediary who introduced the case.",
            actor=ACTOR)
        request = updated.run.field_requests[0]
        assert request["requested_by"] == ACTOR
        assert request["source_column"] == row["source_column"]
        assert "fields_registry.yaml" in request["route"]

    def test_a_name_the_registry_could_not_accept_is_refused(self, service,
                                                             halted):
        row = _unused(service, halted)
        with pytest.raises(OpsError) as caught:
            service.request_registry_field(
                halted, source_file=row["source_file"],
                source_column=row["source_column"],
                field_name="Curr Bal (GBP)", actor=ACTOR)
        assert caught.value.code == "OCC_AGENT_FIELD_NAME_INVALID"

    def test_asking_for_a_field_that_exists_points_at_the_other_act(
            self, service, halted):
        row = _unused(service, halted)
        with pytest.raises(OpsError) as caught:
            service.request_registry_field(
                halted, source_file=row["source_file"],
                source_column=row["source_column"],
                field_name="current_principal_balance", actor=ACTOR)
        assert caught.value.code == "OCC_AGENT_FIELD_ALREADY_REGISTERED"

    def test_asking_twice_about_one_column_leaves_one_ask(self, service,
                                                          halted):
        """A request IS its (file, column): asking again restates it.

        It used to be superseded by its ID, which was true only while the id
        scheme held still. The scheme has since changed, so a request recorded
        under the old one would not have been replaced by the same ask under
        the new one and the case would carry the column twice — a live case
        mid-onboarding is exactly where that would have shown up.
        """
        row = _unused(service, halted)
        updated = service.request_registry_field(
            halted, source_file=row["source_file"],
            source_column=row["source_column"], field_name="broker_code",
            actor=ACTOR)
        # Rewrite the stored id to the shape an earlier release would have
        # written, which is what a case opened before the change carries.
        updated.run.field_requests[0]["request_id"] = "fieldreq_old-scheme-id"
        service.store.save(updated.run)
        updated = service.request_registry_field(
            updated, source_file=row["source_file"],
            source_column=row["source_column"], field_name="introducer_code",
            actor=ACTOR)
        assert len(updated.run.field_requests) == 1
        assert updated.run.field_requests[0]["field_name"] == "introducer_code"

    def test_a_request_can_be_taken_back(self, service, halted):
        """A mistaken ask would otherwise sit in the activation pack for ever.

        Withdrawn, not deleted: an ask that was made and taken back is part of
        the record of what happened on this case.
        """
        row = _unused(service, halted)
        updated = service.request_registry_field(
            halted, source_file=row["source_file"],
            source_column=row["source_column"], field_name="broker_code",
            actor=ACTOR)
        updated = service.withdraw_registry_field_request(
            updated, source_file=row["source_file"],
            source_column=row["source_column"], actor=ACTOR,
            reason="asked in error")
        assert updated.run.field_requests[0]["status"] == "withdrawn"
        after = {(r["source_file"], r["source_column"]): r
                 for r in service.status(updated)["mapping"]["rows"]}
        assert after[(row["source_file"],
                      row["source_column"])]["requested_field"] == ""

    def test_mapping_the_column_withdraws_the_ask(self, service, halted):
        """Two answers to one question. Leaving the request open would have the
        case asking for a field it no longer needs."""
        row = _unused(service, halted)
        updated = service.request_registry_field(
            halted, source_file=row["source_file"],
            source_column=row["source_column"], field_name="broker_code",
            actor=ACTOR)
        updated = service.map_unmapped_column(
            updated, source_file=row["source_file"],
            source_column=row["source_column"],
            target_field="current_principal_balance", actor=ACTOR)
        assert updated.run.field_requests[0]["status"] == "withdrawn"

    def test_it_reaches_the_readiness_package(self, service, halted):
        """An approver signing readiness off should see what this client's data
        needs that Trakt has no field for."""
        from operations_control.occ_agent import readiness as _readiness
        row = _unused(service, halted)
        updated = service.request_registry_field(
            halted, source_file=row["source_file"],
            source_column=row["source_column"], field_name="broker_code",
            actor=ACTOR)
        facts = service.facts(updated)
        package = _readiness.build_package(
            updated.run, updated.case, facts,
            _readiness.evaluate(updated.run, updated.case, facts,
                                service.policy),
            [], service.policy)
        assert [r["field_name"]
                for r in package["field_registry_requests"]] == ["broker_code"]


class TestTheRecordSaysWhatHappened:
    def test_both_acts_are_audited_under_their_own_names(self, service,
                                                         halted):
        row = _unused(service, halted)
        updated = service.map_unmapped_column(
            halted, source_file=row["source_file"],
            source_column=row["source_column"],
            target_field="current_principal_balance", actor=ACTOR)
        updated = _commit(service, updated)
        events = [e["action"] for e in
                  service.store.list_audit(TENANT_A, updated.run.case_ref)]
        assert "mappings_confirmed" in events

    def test_an_ask_is_audited_as_an_ask(self, service, halted):
        row = _unused(service, halted)
        updated = service.request_registry_field(
            halted, source_file=row["source_file"],
            source_column=row["source_column"], field_name="broker_code",
            actor=ACTOR)
        events = [e["action"] for e in
                  service.store.list_audit(TENANT_A, updated.run.case_ref)]
        assert "field_registry_requested" in events


# --------------------------------------------------------------------------- #
# Over the API, which is what the screen actually calls
# --------------------------------------------------------------------------- #

HEADERS = {"X-Operator-Token": "tok-a"}


@pytest.fixture()
def api_client(agent_env, service):
    import importlib

    from fastapi.testclient import TestClient

    import operations_control.api.app as app_module
    from operations_control.occ_agent import api as agent_api
    importlib.reload(app_module)
    agent_api.configure(service)
    return TestClient(app_module.app)


class TestTheRoutesTheScreenCalls:
    def test_the_picker_is_served(self, api_client, halted):
        body = api_client.get(
            f"/ops/agent/cases/{halted.case_ref}/field-registry",
            headers=HEADERS).json()
        assert body["ok"] is True
        assert any(f["name"] == "current_principal_balance"
                   for f in body["fields"])

    def test_naming_a_field_settles_the_column(self, api_client, service,
                                               halted):
        row = _unused(service, halted)
        body = api_client.post(
            f"/ops/agent/cases/{halted.case_ref}/mappings/unmapped",
            headers=HEADERS,
            json={"source_file": row["source_file"],
                  "source_column": row["source_column"],
                  "action": "use_existing",
                  "target_field": "current_principal_balance"}).json()
        assert body["ok"] is True
        after = {(r["source_file"], r["source_column"]): r
                 for r in body["mapping"]["rows"]}
        settled = after[(row["source_file"], row["source_column"])]
        assert settled["state"] == "staged"
        assert settled["staged_field"] == "current_principal_balance"

    def test_asking_for_a_field_comes_back_as_a_request(self, api_client,
                                                        service, halted):
        row = _unused(service, halted)
        body = api_client.post(
            f"/ops/agent/cases/{halted.case_ref}/mappings/unmapped",
            headers=HEADERS,
            json={"source_file": row["source_file"],
                  "source_column": row["source_column"],
                  "action": "request_field",
                  "field_name": "broker_code"}).json()
        assert [r["field_name"] for r in body["run"]["field_requests"]] \
            == ["broker_code"]

    def test_an_action_trakt_does_not_have_is_refused(self, api_client,
                                                      service, halted):
        row = _unused(service, halted)
        response = api_client.post(
            f"/ops/agent/cases/{halted.case_ref}/mappings/unmapped",
            headers=HEADERS,
            json={"source_file": row["source_file"],
                  "source_column": row["source_column"],
                  "action": "invent_a_field"})
        assert response.status_code == 400
