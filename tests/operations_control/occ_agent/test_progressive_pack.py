"""The two-phase onboarding request.

The pack a client receives first used to mix six different kinds of question
together — blocking detail, facts Trakt already held, optional preferences,
generic financial semantics, file instructions and per-field definitions. It
ran to 33 questions and asked the client to explain what a balance is.

The principle the pack already stated for MAPPINGS is now applied to MEANING:
Trakt analyses the file first, and comes back only where something genuinely
needs clarifying. These tests hold that shape in place.

Phase 1 — three sections, in this order:
    what we need now · what we already understand · optional context
Phase 2 — after a file arrives, and only then, the deferred questions.
"""

from __future__ import annotations

import pytest

from operations_control.engine import OpsError
from operations_control.occ_agent import clarification as _clarification
from operations_control.occ_agent import communication as _comms
from operations_control.occ_agent import pack as _pack
from operations_control.onboarding.catalogue import catalogue

from .conftest import ACTOR, TENANT_A

OPENING = ("Onboard Mortgage Capital. UK residential mortgages. Monthly "
           "portfolio MI. Portfolio id direct_201.")

#: The questions a client should never be asked before a file has arrived.
#: Named individually because these are the ones a real client saw.
GENERIC_SEMANTICS = (
    "What a balance means", "Gross or net amounts",
    "Point in time or cumulative", "How accrued interest is treated",
    "When a loan counts as redeemed", "What your account statuses mean",
)


@pytest.fixture()
def opened(service):
    return service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                               instruction=OPENING)


@pytest.fixture()
def built(service, opened):
    return service.build_pack(opened)


@pytest.fixture()
def document(built):
    return built.document()


def heading_order(document: str) -> list:
    return [line[3:].strip() for line in document.split("\n")
            if line.startswith("## ")]


# --------------------------------------------------------------------------- #
# 1 — Phase 1 asks for what is genuinely needed
# --------------------------------------------------------------------------- #

def test_the_pack_has_three_client_facing_sections_in_order(document):
    assert heading_order(document)[:3] == [
        "What we need from you now",
        "Please confirm what we already understand",
        "Optional context"]


def test_what_is_needed_now_carries_every_required_question(built, document):
    """Nothing blocking may be demoted into optional context."""
    needed = document.split("## Please confirm")[0]
    required = [q.label for s in built.sections for q in s.questions
                if q.required]
    assert required
    for label in required:
        assert label in needed, label


def test_the_essentials_are_there(built):
    """Entity, contacts, portfolio identity and access — the minimum set."""
    asked = {s.key for s in built.sections if not s.optional_context}
    for section in ("entities", "contacts", "portfolios", "access"):
        assert section in asked, section


def test_the_representative_file_is_requested_up_front(document):
    needed = document.split("## Please confirm")[0]
    assert "Files to send" in needed
    assert "Primary loan tape" in needed


# --------------------------------------------------------------------------- #
# 2 — confirm-only facts are shown, never re-asked
# --------------------------------------------------------------------------- #

def test_what_trakt_already_knows_is_shown_for_correction_not_re_entry(
        built, document):
    confirm = document.split("## Please confirm what we already understand")[1]
    confirm = confirm.split("## Optional context")[0]
    assert "tell us if any are incorrect" in confirm
    assert "nothing to re-enter" in confirm
    assert built.confirmations
    for question in built.confirmations:
        assert question.label in confirm, question.label


def test_a_confirmed_fact_is_not_also_a_question(built):
    confirmed = {(q.section, q.field) for q in built.confirmations}
    asked = {(s.key, q.field) for s in built.sections for q in s.questions}
    assert confirmed and not (confirmed & asked)


# --------------------------------------------------------------------------- #
# 3 — optional context is unmistakably optional
# --------------------------------------------------------------------------- #

def test_optional_context_says_it_may_be_left_blank(document):
    optional = document.split("## Optional context")[1]
    optional = optional.split("## What we are NOT")[0]
    assert "optional" in optional.lower()
    assert "Leave it blank" in optional


def test_nothing_required_hides_in_optional_context(built):
    for section in built.sections:
        if section.optional_context:
            assert not [q for q in section.questions if q.required], section.key


def test_which_sections_are_optional_is_the_catalogues_decision():
    """Config, not a hard-coded list in the pack builder."""
    cat = catalogue()
    assert cat.section("presentation").optional_context is True
    assert cat.section("risk_limits").optional_context is True
    assert cat.section("contacts").optional_context is False
    source = (_pack.__file__)
    text = open(source, encoding="utf-8").read()
    assert '"presentation"' not in text and '"risk_limits"' not in text, \
        "the pack must not name sections; the catalogue declares them"


# --------------------------------------------------------------------------- #
# 4 — no generic field-semantic question before ingestion
# --------------------------------------------------------------------------- #

def test_no_generic_semantic_question_is_asked_before_a_file_arrives(document):
    asked = document.split("## What we are NOT asking you for")[0]
    for label in GENERIC_SEMANTICS:
        assert label not in asked, label


def test_the_client_document_does_not_inventory_what_is_not_asked(document):
    """A list of questions you are NOT being asked adds nothing and makes a
    short request look heavy. It is gone from the client's copy."""
    assert "What we are NOT asking you for" not in document
    for label in GENERIC_SEMANTICS:
        assert label not in document, label


def test_what_is_not_asked_stays_visible_to_the_operator(built):
    """Removed from the CLIENT's document, not from the record.

    ``not_asked`` still carries every deferred field with its reason, which is
    what the review package and the operator surfaces read.
    """
    reported = {row["key"]: row for row in built.not_asked}
    assert reported
    deferred = [k for k, row in reported.items()
                if row["category"] == "first_delivery"]
    assert deferred
    assert all(reported[k]["reason"] for k in deferred)
    assert any(k.startswith("data_semantics.") for k in deferred)


def test_the_pack_does_not_count_questions_at_the_client(document):
    """"There are 33 questions outstanding" measures Trakt, not the client."""
    assert "questions outstanding" not in document
    assert "question for you" not in document and "questions for you" not in \
        document
    assert _pack.EFFORT_STATEMENT in document
    assert "representative portfolio data artefacts" in document


def test_the_covering_email_leads_with_effort_not_arithmetic(built):
    body = built.email.body
    assert "questions outstanding" not in body
    assert "representative portfolio data artefacts" in body
    assert "only where something needs clarifying" in body
    assert body.startswith("Dear ")
    assert body.rstrip().endswith("Trakt Operations")


# --------------------------------------------------------------------------- #
# 5 — phase two: clarification after ingestion
# --------------------------------------------------------------------------- #

def test_there_is_nothing_to_clarify_before_a_file_arrives(service, opened):
    available = service.clarifications_available(opened)
    assert available["items"] == []
    assert available["summary"]["files_analysed"] == 0
    with pytest.raises(OpsError) as caught:
        service.request_clarifications(opened, actor=ACTOR)
    assert caught.value.code == "OCC_AGENT_NOTHING_TO_CLARIFY"


def test_once_a_file_arrives_the_deferred_semantics_can_be_asked(service,
                                                                 opened):
    delivered = service.generate_synthetic_response(opened, actor=ACTOR)
    available = service.clarifications_available(delivered)
    labels = {item["label"] for item in available["items"]}
    assert labels, "a delivered file must make the deferred questions askable"
    assert labels & set(GENERIC_SEMANTICS), \
        "the questions deferred from phase 1 are the ones asked in phase 2"
    assert available["summary"]["files_analysed"] >= 1


def test_a_clarification_carries_the_evidence_it_is_asked_against(service,
                                                                  opened):
    """Evidence-led: the question names what was actually found."""
    delivered = service.generate_synthetic_response(opened, actor=ACTOR)
    items = service.clarifications_available(delivered)["items"]
    assert items
    for item in items:
        assert item["evidence"]["files"], item["label"]
        assert item["evidence"]["columns"], item["label"]
        assert item["deferred_reason"]


def test_a_clarification_becomes_an_ordinary_information_request(service,
                                                                 opened):
    """No second channel: the governed request path, unchanged."""
    delivered = service.generate_synthetic_response(opened, actor=ACTOR)
    before = len(delivered.case.requests())
    asked = service.request_clarifications(delivered, actor=ACTOR)
    requests = asked.case.requests()
    assert len(requests) == before + 1
    assert requests[-1].responsible_party == "client"
    assert requests[-1].items


# --------------------------------------------------------------------------- #
# 6 — internal completeness is not weakened
# --------------------------------------------------------------------------- #

def test_every_deferred_field_is_still_declared_and_still_writes_somewhere():
    cat = catalogue()
    section = cat.section("data_semantics")
    assert len(section.fields) == 10, "deferring must not delete fields"
    for f in section.fields:
        assert f.writes_to, f.key
        assert f.consumers, f.key


def test_the_catalogue_still_declares_the_same_field_count():
    """A simplification of the ASK, not of the record."""
    cat = catalogue()
    assert sum(len(s.fields) for s in cat.sections) >= 70


# --------------------------------------------------------------------------- #
# 7 — the approval gates are untouched
# --------------------------------------------------------------------------- #

def test_the_pack_still_cannot_be_sent_without_a_human(service, opened):
    drafted = service.draft_pack(opened, actor=ACTOR)
    assert drafted.run.pack_status == _comms.HUMAN_REVIEW_REQUIRED
    with pytest.raises(OpsError):
        service.send_pack(drafted, actor=ACTOR, to=["client@example.com"])
    approved = service.approve_pack(drafted, actor=ACTOR)
    assert approved.run.pack_status == _comms.APPROVED_TO_SEND
    sent = service.send_pack(approved, actor=ACTOR, to=["client@example.com"])
    assert sent.run.pack_status == _comms.SENT


def test_the_pack_workflow_transitions_are_unchanged():
    assert _comms.PACK_WORKFLOW == (
        _comms.DRAFTED, _comms.HUMAN_REVIEW_REQUIRED,
        _comms.APPROVED_TO_SEND, _comms.SENT)
    assert _comms.PACK_TRANSITIONS[_comms.DRAFTED] == (
        _comms.HUMAN_REVIEW_REQUIRED, _comms.DRAFTED)


def test_requesting_clarifications_does_not_activate_anything(service, opened):
    delivered = service.generate_synthetic_response(opened, actor=ACTOR)
    state_before = delivered.run.state
    asked = service.request_clarifications(delivered, actor=ACTOR)
    assert asked.run.state == state_before
    assert asked.case.status != "activated"


# --------------------------------------------------------------------------- #
# Generating a client response — answers, not files
# --------------------------------------------------------------------------- #

def test_generating_answers_clears_the_outstanding_checklist(service, opened):
    """The defect this exists for.

    At "receive client responses" an operator sees a checklist of unanswered
    questions. The button beside it used to generate a loan TAPE — the call
    succeeded, the checklist was untouched, and it read as a broken button.
    """
    sent = service.send_pack(
        service.approve_pack(service.draft_pack(opened, actor=ACTOR),
                             actor=ACTOR),
        actor=ACTOR, to=["client@example.com"])
    assert service.onboarding.client_checklist(sent.case)

    answered = service.generate_synthetic_answers(sent, actor=ACTOR)
    assert service.onboarding.client_checklist(answered.case) == []


def test_generated_answers_go_through_the_ordinary_validated_path(service,
                                                                  opened):
    """No back door: the same validation a real client's answers face."""
    sent = service.send_pack(
        service.approve_pack(service.draft_pack(opened, actor=ACTOR),
                             actor=ACTOR),
        actor=ACTOR, to=["client@example.com"])
    answered = service.generate_synthetic_answers(sent, actor=ACTOR)

    contacts = answered.case.answers.get("contacts") or {}
    assert "@" in str(contacts.get("reporting_contact_email", ""))
    # Every value is one the form would have accepted — an enum takes a
    # DECLARED option rather than a plausible-looking string.
    portfolios = answered.case.answers.get("portfolios") or []
    if portfolios:
        cat = service.onboarding.catalogue
        declared = {o["value"] for o
                    in cat.field("portfolios", "portfolio_type").options}
        assert portfolios[0].get("portfolio_type") in declared


def test_generating_answers_is_audited_as_synthetic(service, opened):
    sent = service.send_pack(
        service.approve_pack(service.draft_pack(opened, actor=ACTOR),
                             actor=ACTOR),
        actor=ACTOR, to=["client@example.com"])
    answered = service.generate_synthetic_answers(sent, actor=ACTOR)
    events = service.store.list_audit(TENANT_A, answered.case_ref)
    generated = [e for e in events
                 if e["action"] == "synthetic_answers_generated"]
    assert len(generated) == 1
    assert generated[0]["execution_classification"] == "synthetically_executed"
    assert generated[0]["detail"]["answers"] > 0


def test_answers_and_artefacts_are_two_different_acts(service, opened):
    """Generating answers must not register files, and vice versa."""
    sent = service.send_pack(
        service.approve_pack(service.draft_pack(opened, actor=ACTOR),
                             actor=ACTOR),
        actor=ACTOR, to=["client@example.com"])

    answered = service.generate_synthetic_answers(sent, actor=ACTOR)
    assert answered.run.received_artefacts == [], \
        "answering questions must not invent a delivery"

    delivered = service.generate_synthetic_response(answered, actor=ACTOR)
    assert delivered.run.received_artefacts, \
        "the artefact button still makes up files"


# --------------------------------------------------------------------------- #
# The confirm list is something a client can actually check
# --------------------------------------------------------------------------- #

def test_two_registrations_do_not_print_the_same_line_twice(service):
    """A weekly pipeline and a monthly funded book are two registrations.

    Dropping the qualifier made both read "Expected cadence: Weekly" — the same
    line twice, which reads as a bug and gives the client nothing to correct.
    """
    case = service.create_case(
        tenant=TENANT_A, initiating_user=ACTOR,
        instruction="Onboard ERM Capital Mgt. It is a UK equity release "
                    "lender. The need weekly pipeline, monthly MI and Annex 2 "
                    "regime reporting.")
    lines = [line for line in service.build_pack(case).document().split("\n")
             if line.startswith("- **Expected cadence**")]
    assert len(lines) == len(set(lines)), lines
    assert len(lines) > 1, "this instruction registers two streams"


def test_a_qualifier_is_never_an_internal_identifier(service):
    """``direct_001/funded`` is a source key. A client cannot confirm a path."""
    case = service.create_case(
        tenant=TENANT_A, initiating_user=ACTOR,
        instruction="Onboard ERM Capital Mgt. It is a UK equity release "
                    "lender. The need weekly pipeline, monthly MI and Annex 2 "
                    "regime reporting.")
    confirm = service.build_pack(case).document()
    confirm = confirm.split("## Please confirm what we already understand")[1]
    confirm = confirm.split("## Optional context")[0]
    # Only the QUALIFIER is checked: a VALUE may legitimately carry a slash
    # (the reporting time zone is "Europe/London").
    qualifiers = [line.split("**", 2)[2].split(":")[0]
                  for line in confirm.split("\n")
                  if line.startswith("- **") and " — " in line]
    assert qualifiers, "this instruction needs at least one qualifier"
    for qualifier in qualifiers:
        assert "/" not in qualifier, qualifier
        assert "direct_001" not in qualifier, qualifier
    # …but the meaningful part of it survives, so the rows stay distinct.
    assert "funded" in confirm and "pipeline" in confirm


def test_a_single_registration_needs_no_qualifier(service, opened):
    """The qualifier is restored only where a row would otherwise collide."""
    document = service.build_pack(opened).document()
    for line in document.split("\n"):
        if line.startswith("- **Client name**") or \
                line.startswith("- **Jurisdiction**"):
            assert " — " not in line, line
