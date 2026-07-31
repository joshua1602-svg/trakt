"""The client pack, the way it is issued, and the package a human approves.

Three properties, each of which the brief asks for by name.

**The pack is a projection of the governed catalogue.** Every question traces
to a field ``config/onboarding/field_catalogue.yaml`` declares, and every field
the catalogue asks a client for appears. There is no second questionnaire, and
adding a field to the catalogue adds it to the pack with no change here.

**Nothing leaves Trakt without a human, and nothing claims to have left when it
has not.** The pack workflow is ``DRAFTED → HUMAN_REVIEW_REQUIRED →
APPROVED_TO_SEND → SENT``, and the receipt says plainly whether anything was
actually sent.

**The review package is complete, and states what it does not cover.** Field
mappings are learned at first ingestion and the package says so; user access is
a requirement rather than a grant, and the package emits operator actions
instead of claiming anything was provisioned.
"""

from __future__ import annotations

import pytest

from operations_control.engine import OpsError
from operations_control.occ_agent import communication as _comms
from operations_control.occ_agent import pack as _pack
from operations_control.occ_agent import review as _review
from operations_control.occ_agent import states as _states
from operations_control.occ_agent.scenarios import run_scenario
from operations_control.onboarding.catalogue import catalogue

from .conftest import ACTOR, TENANT_A

OPENING = ("Onboard Northstar Lending. UK equity release. Monthly portfolio "
           "MI. Portfolio id direct_101.")


@pytest.fixture()
def opened(service):
    return service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                               instruction=OPENING)


@pytest.fixture()
def drafted(service, opened):
    return service.draft_pack(opened, actor=ACTOR)


# --------------------------------------------------------------------------- #
# The pack IS the catalogue
# --------------------------------------------------------------------------- #

def test_every_question_traces_to_a_catalogue_field(service, opened):
    cat = catalogue()
    built = service.build_pack(opened)
    assert built.question_count
    for section in built.sections:
        assert cat.section(section.key) is not None, section.key
        for question in section.questions:
            f = cat.field(section.key, question.field)
            assert f is not None, f"{section.key}.{question.field}"
            assert f.collected is True
            assert question.label == f.label
            assert question.writes_to == f.writes_to


def test_every_outstanding_client_field_appears_in_the_pack(service, opened):
    """Coverage in the other direction, over what a CLIENT is asked.

    Not over every collected field: the pack deliberately no longer projects
    values Trakt mints, defaults, derives, or reads from the first delivery.
    What must not be dropped is a question the client alone can answer and has
    not yet — and that is exactly the classification's category 2.
    """
    from operations_control.occ_agent import classification as _classification

    cat = service.onboarding.catalogue
    built = service.build_pack(opened)
    present = {(s.key, q.field, q.index) for s in built.sections
               for q in s.questions}
    rows = _classification.classify_all(opened.case.answers, cat=cat)
    for row in rows:
        if row.client_facing:
            assert (row.section, row.field, row.index) in present, \
                f"{row.section}.{row.field}"


def test_the_pack_covers_the_areas_the_brief_names(service, opened):
    """Every named area has a home — asked now, or reported as deferred.

    ``client``, ``reporting`` and ``sources`` do not appear as questions for
    this case because the opening instruction already answered them or Trakt
    derives them; they appear on the confirmation list and in ``not_asked``.
    What matters is that no area is silently absent.
    """
    built = service.build_pack(opened)
    asked = {s.key for s in built.sections}
    deferred = {row["key"].split(".")[0] for row in built.not_asked}
    confirmed = {q.section for q in built.confirmations}
    accounted = asked | deferred | confirmed
    for expected in ("client", "entities", "contacts", "portfolios", "sources",
                     "reporting", "data_definitions", "access"):
        assert expected in accounted, expected
    # And the data definitions are deferred rather than dropped.
    assert "data_definitions" in deferred


def test_the_pack_states_that_mappings_are_not_collected(service, opened):
    built = service.build_pack(opened)
    assert built.mapping_statement == _pack.MAPPING_STATEMENT
    assert "does not ask you to map" in built.document()
    # And no question asks for one.
    fields = {q.field for s in built.sections for q in s.questions}
    assert "file_role_schemas" not in fields
    assert "expected_schema_fingerprint" not in fields


def test_the_catalogue_still_records_mappings_as_not_collected():
    """The governed decision this feature must not quietly reverse."""
    not_collected = {row["key"] for row in catalogue().not_collected}
    assert "file_role_schemas" in not_collected
    assert "expected_schema_fingerprint" in not_collected


def test_the_pack_asks_what_the_numbers_mean_rather_than_how_to_map_them():
    """Decision 1's permitted scope, as catalogue fields."""
    cat = catalogue()
    section = cat.section("data_definitions")
    assert section is not None
    keys = {f.key for f in section.fields}
    for expected in ("source_file", "description", "proprietary_fields",
                     "units_and_currency", "balance_definition",
                     "date_conventions", "measure_basis", "known_limitations"):
        assert expected in keys, expected


def test_the_pack_asks_for_files_from_the_governed_input_requirements(service,
                                                                      opened):
    from operations_control.occ_agent.input_roles import artefact_vocabulary
    built = service.build_pack(opened)
    vocab = artefact_vocabulary()
    outcome = service.facts(opened).outcome
    assert {r["role"] for r in built.artefacts.required} == \
        set(vocab.required_roles(outcome))


def test_an_answered_question_is_not_asked_again(service, opened):
    """The instruction named the client, so the pack confirms rather than asks."""
    built = service.build_pack(opened)
    asked = {(s.key, q.field) for s in built.sections for q in s.questions}
    assert ("client", "client_name") not in asked
    name = next(q for q in built.confirmations if q.field == "client_name")
    assert name.status == _pack.ANSWERED
    assert name.value == "Northstar Lending"


def test_what_trakt_supplies_is_never_asked_of_the_client(service, opened):
    built = service.build_pack(opened)
    for section in built.sections:
        for question in section.questions:
            f = catalogue().field(section.key, question.field)
            if f.answered_by_trakt and not question.value:
                assert question.status == _pack.DERIVED


# --------------------------------------------------------------------------- #
# The four-state workflow
# --------------------------------------------------------------------------- #

def test_drafting_puts_the_pack_in_front_of_a_human(service, drafted):
    assert drafted.run.pack_status == _comms.HUMAN_REVIEW_REQUIRED
    assert drafted.run.state == _states.PACK_REVIEW_REQUIRED
    assert [h["status"] for h in drafted.run.pack_history] == [
        _comms.DRAFTED, _comms.HUMAN_REVIEW_REQUIRED]


def test_a_pack_cannot_be_sent_before_it_is_approved(service, drafted):
    from operations_control.occ_agent.service import ActionNotAllowed
    with pytest.raises(ActionNotAllowed):
        service.send_pack(drafted, actor=ACTOR, to=["someone@example.com"])


def test_the_workflow_cannot_skip_the_review():
    with pytest.raises(_comms.PackWorkflowError):
        _comms.assert_pack_transition(_comms.DRAFTED, _comms.APPROVED_TO_SEND)
    with pytest.raises(_comms.PackWorkflowError):
        _comms.assert_pack_transition("", _comms.SENT)


def test_approving_the_pack_is_attributed_and_audited(service, drafted):
    approved = service.approve_pack(drafted, actor=ACTOR, reason="Read it.")
    approval = approved.run.approval("client_pack")
    assert approval["actor"] == ACTOR
    assert approval["decision"] == "approved"
    assert approval["content_hash"]
    actions = [e["action"] for e in
               service.store.list_audit(TENANT_A, approved.case_ref)]
    assert "onboarding_pack_approved" in actions


def test_issuing_the_pack_never_claims_it_was_sent(service, drafted):
    approved = service.approve_pack(drafted, actor=ACTOR)
    sent = service.send_pack(approved, actor=ACTOR,
                             to=["reporting@northstar.example"])
    receipt = sent.run.pack_receipt
    assert receipt["sent"] is False
    assert receipt["adapter"] == "record_only"
    assert "did not send" in receipt["statement"]
    assert sent.run.state == _states.PACK_SENT


def test_issuing_with_no_recipient_is_refused(service, drafted):
    approved = service.approve_pack(drafted, actor=ACTOR)
    with pytest.raises(OpsError) as excinfo:
        service.send_pack(approved, actor=ACTOR, to=[])
    assert excinfo.value.code == "OCC_AGENT_NO_RECIPIENT"


def test_the_default_adapter_sends_nothing():
    adapter = _comms.default_adapter()
    receipt = adapter.deliver(to=["a@b.example"], subject="s", body="b",
                              artefacts=[], content_hash="h", actor=ACTOR)
    assert receipt.sent is False


def test_the_exact_approved_artefacts_are_retrievable(service, drafted):
    """What an operator sends by hand is what was approved."""
    stored = service.store.read_package(TENANT_A, drafted.case_ref,
                                        "onboarding_pack.md")
    assert stored
    assert stored == service.build_pack(drafted).document()
    email = service.store.read_package(TENANT_A, drafted.case_ref,
                                       "covering_email.txt")
    assert "Subject:" in email


# --------------------------------------------------------------------------- #
# The review package
# --------------------------------------------------------------------------- #

@pytest.fixture()
def reviewed(service):
    return run_scenario(service, "scenario_a_clean", tenant=TENANT_A,
                        actor=ACTOR).case


def test_the_review_package_is_assembled_and_stored(service, reviewed):
    assert reviewed.run.review_package_ref
    stored = service.store.read_package(TENANT_A, reviewed.case_ref,
                                        "review_package.json")
    assert stored


def test_the_review_package_carries_every_required_part(service, reviewed):
    package = service.build_review_package(reviewed).to_dict()
    for part in ("sections", "outstanding", "data_definitions",
                 "access_requirements", "operator_actions", "artefacts",
                 "communication", "configuration_preview", "rehearsal",
                 "readiness", "activation", "approvals", "audit_trail",
                 "mapping_note", "access_note", "content_hash"):
        assert part in package, part


def test_every_answer_in_the_review_package_says_where_it_came_from(service,
                                                                    reviewed):
    package = service.build_review_package(reviewed)
    rows = [r for s in package.sections for r in s["rows"]]
    assert rows
    for row in rows:
        assert row["provenance_label"]
    assert any(r["provenance"] == "client_supplied" for r in rows), \
        "nothing is recorded as having come from the client"


def test_the_review_package_states_that_mappings_are_learned_later(service,
                                                                   reviewed):
    package = service.build_review_package(reviewed)
    assert package.mapping_note == _review.MAPPING_NOTE
    assert "were not collected" in package.mapping_note
    assert "first representative delivery" in package.mapping_note
    assert "Field mappings" in package.document()


def test_the_review_package_shows_what_activation_would_do(service, reviewed):
    package = service.build_review_package(reviewed)
    assert package.activation["files"]
    assert package.activation["actions"]
    assert package.configuration_preview["written"] is False


def test_the_review_package_reports_the_pack_honestly(service, reviewed):
    package = service.build_review_package(reviewed)
    assert package.communication["pack_status"] == _comms.SENT
    assert package.communication["sent"] is False
    assert "did not send" in package.communication["statement"]


# --------------------------------------------------------------------------- #
# User access is a requirement, not a grant
# --------------------------------------------------------------------------- #

def test_access_requirements_become_operator_actions():
    actions = _review.access_actions([
        {"user_name": "Dana Fox", "user_email": "dana@northstar.example",
         "user_role": "approver", "scope_note": "direct_101",
         "occ_access_required": True, "dashboard_access_required": True,
         "report_recipient": True},
    ])
    kinds = {a.kind for a in actions}
    assert kinds == {"occ_access", "dashboard_access", "report_distribution",
                     "approval_role"}
    assert all(a.status == "not_provisioned" for a in actions)


def test_nothing_claims_access_was_provisioned():
    actions = _review.access_actions([
        {"user_name": "Dana Fox", "occ_access_required": True}])
    text = " ".join(a.detail for a in actions).lower()
    assert "not done by activating" in text or "not provisioned" in text
    assert "granted" not in text
    assert _review.ACCESS_NOTE.startswith(
        "User access recorded during onboarding is a REQUIREMENT")


def test_access_is_collected_into_the_governed_onboarding_record():
    """Decision 2: the existing artefact, not a new identity framework."""
    section = catalogue().section("access")
    assert section is not None
    for f in section.fields:
        assert f.writes_to == "onboarding_record", f.key


def test_recorded_access_reaches_the_review_package(service, opened):
    case = service.onboarding.save_step(
        case_id=opened.case_ref, step="access", by=ACTOR,
        payload={"access": [{"user_name": "Dana Fox",
                             "user_email": "dana@northstar.example",
                             "user_role": "approver",
                             "occ_access_required": True}]})
    opened.case = case
    package = service.build_review_package(opened)
    assert package.access_requirements
    assert package.operator_actions
    assert "administrator" in package.document().lower()
