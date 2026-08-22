"""What the client is actually asked, and how their answers get in.

Catalogue coverage was already proven. This is the other half: that the pack is
an *appropriate* thing to send someone. Two properties:

**Only questions the client can answer.** Every catalogue field falls into
exactly one of five categories and only category 2 becomes a question. The other
four are pre-populated, derived, deferred to the first delivery, or internal —
and each is reported, so "why is that not being asked?" always has an answer.

**Structured answers are persisted verbatim.** A response is keyed by
authoritative catalogue keys, checked against the form the client was served,
and written through ``OnboardingService.save_step`` exactly as submitted. No
natural-language parsing sits between a control and the case.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from operations_control.engine import OpsError
from operations_control.occ_agent import classification as _classification
from operations_control.occ_agent import client_form as _client_form
from operations_control.onboarding.catalogue import catalogue

from .conftest import ACTOR, TENANT_A

OPENING = ("Onboard Northstar Lending. UK equity release. Monthly portfolio "
           "MI. Portfolio id direct_101. First reporting date 2026-06-30.")


@pytest.fixture()
def opened(service):
    return service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                               instruction=OPENING)


# --------------------------------------------------------------------------- #
# Every field lands in exactly one category
# --------------------------------------------------------------------------- #

def test_every_catalogue_field_is_classified(opened, service):
    rows = _classification.classify_all(opened.case.answers,
                                        cat=service.onboarding.catalogue)
    assert rows
    for row in rows:
        assert row.category in _classification.CATEGORIES, row.field
        assert row.reason, f"{row.section}.{row.field} has no reason"


def test_a_field_is_in_exactly_one_category(opened, service):
    rows = _classification.classify_all(opened.case.answers,
                                        cat=service.onboarding.catalogue)
    seen = {}
    for row in rows:
        key = (row.section, row.field, row.index)
        assert key not in seen, f"{key} classified twice"
        seen[key] = row.category


def test_the_counts_add_up(opened, service):
    rows = _classification.classify_all(opened.case.answers,
                                        cat=service.onboarding.catalogue)
    summary = _classification.summarise(rows)
    assert sum(summary["counts"].values()) == summary["total"] == len(rows)


def test_every_override_names_a_real_field():
    cat = catalogue()
    for path, category in _classification.OVERRIDES.items():
        section_key, field_key = path.split(".", 1)
        assert cat.field(section_key, field_key) is not None, path
        assert category in _classification.CATEGORIES, path


def test_every_deferred_section_is_a_real_section():
    cat = catalogue()
    for key in _classification.deferred_sections(cat):
        assert cat.section(key) is not None, key


# --------------------------------------------------------------------------- #
# Only category 2 reaches a client
# --------------------------------------------------------------------------- #

def test_the_form_contains_only_client_questions(opened, service):
    cat = service.onboarding.catalogue
    rows = _classification.classify_all(opened.case.answers, cat=cat)
    client = {(r.section, r.field, r.index) for r in rows if r.client_facing}
    form = _client_form.build(opened.case, cat=cat)
    for step in form.steps:
        for group in step.groups:
            for f in group.fields:
                assert (f.section, f.field, f.index) in client, f.key


def test_nothing_trakt_mints_is_asked_of_a_client(opened, service):
    form = _client_form.build(opened.case, cat=service.onboarding.catalogue)
    keys = set(form.keys())
    for internal in ("client.client_id", "portfolios[0].portfolio_id",
                     "sources[0].dataset", "sources[0].sample_provided",
                     "sources[0].mapping_complete"):
        assert internal not in keys, internal


def test_nothing_learned_from_a_delivery_is_asked_of_a_client(opened, service):
    """Category 4: the Onboarding Agent reads these, so nobody asks."""
    form = _client_form.build(opened.case, cat=service.onboarding.catalogue)
    keys = set(form.keys())
    for deferred in ("sources[0].file_format", "sources[0].expected_files",
                     "portfolios[0].asset_class"):
        assert deferred not in keys, deferred
    # Nor is any data-definition question in the INITIAL pack.
    assert not [k for k in keys if k.startswith("data_definitions")]
    assert any(row["step"] == "data" for row in form.locked)


def test_what_is_not_asked_is_reported_rather_than_hidden(opened, service):
    pack = service.build_pack(opened)
    assert pack.not_asked, "nothing was reported as not asked"
    for row in pack.not_asked:
        assert row["reason"], row["label"]
    kinds = {row["category"] for row in pack.not_asked}
    assert _classification.FIRST_DELIVERY in kinds
    assert _classification.INTERNAL in kinds


def test_the_pack_is_materially_smaller_than_the_catalogue(opened, service):
    """The point of the exercise, stated as a property rather than a number."""
    cat = service.onboarding.catalogue
    declared = sum(len(s.fields) for s in cat.sections)
    pack = service.build_pack(opened)
    assert pack.question_count < declared / 2, (
        f"{pack.question_count} questions from {declared} declared fields")
    assert pack.required_count < pack.question_count


# --------------------------------------------------------------------------- #
# Pre-population and confirmation
# --------------------------------------------------------------------------- #

def test_what_the_instruction_said_is_pre_populated_not_re_asked(opened,
                                                                 service):
    form = _client_form.build(opened.case, cat=service.onboarding.catalogue)
    keys = set(form.keys())
    # The instruction named all of these; none is asked again.
    for known in ("client.client_name", "client.jurisdiction",
                  "portfolios[0].asset_class", "reporting.products"):
        assert known not in keys, known


def test_material_known_values_are_offered_for_confirmation(opened, service):
    pack = service.build_pack(opened)
    labels = {q.label for q in pack.confirmations}
    assert "Client name" in labels
    assert "Jurisdiction" in labels
    for question in pack.confirmations:
        assert question.value not in (None, ""), question.label


def test_an_identifier_the_client_never_saw_is_not_put_up_for_confirmation(
        opened, service):
    pack = service.build_pack(opened)
    fields = {q.field for q in pack.confirmations}
    assert "client_id" not in fields
    assert "owning_entity" not in fields


# --------------------------------------------------------------------------- #
# Conditional and progressive
# --------------------------------------------------------------------------- #

def test_a_product_specific_question_is_not_asked_without_the_product(opened,
                                                                      service):
    form = _client_form.build(opened.case, cat=service.onboarding.catalogue)
    # This case selected `mi` only.
    assert "contacts.investor_report_recipients" not in set(form.keys())


def test_selecting_a_product_asks_its_questions(service):
    case = service.create_case(
        tenant=TENANT_A, initiating_user=ACTOR,
        instruction="Onboard Aldermere Advances. UK equity release. They need "
                    "monthly portfolio MI and investor reporting. Portfolio id "
                    "direct_104.")
    form = _client_form.build(case.case, cat=service.onboarding.catalogue)
    assert "investor_reporting" in case.case.products
    assert "contacts.investor_report_recipients" in set(form.keys())


def test_a_follow_up_appears_only_once_its_trigger_is_answered(service, opened):
    cat = service.onboarding.catalogue
    before = set(_client_form.build(opened.case, cat=cat).keys())
    assert not [k for k in before if k.endswith(".source_party")]

    answered = service.submit_client_response(
        opened, actor=ACTOR,
        response={"portfolios[0].portfolio_type": "acquired"})
    after = set(_client_form.build(answered.case, cat=cat).keys())
    assert [k for k in after if k.endswith(".source_party")], (
        "answering how the book was acquired did not unlock who sends it")


def test_a_question_that_does_not_apply_is_reported_as_such(opened, service):
    rows = _classification.classify_all(opened.case.answers,
                                        cat=service.onboarding.catalogue)
    excluded = [r for r in rows
                if r.category == _classification.NOT_APPLICABLE]
    assert excluded, "nothing was excluded by a catalogue condition"
    for row in excluded:
        assert "does not apply" in row.reason


# --------------------------------------------------------------------------- #
# Grouped, progressive, and not repetitive
# --------------------------------------------------------------------------- #

def test_the_form_is_grouped_into_client_facing_steps(opened, service):
    form = _client_form.build(opened.case, cat=service.onboarding.catalogue)
    assert len(form.steps) > 1
    declared = {s["key"] for s in _client_form.STEPS}
    for step in form.steps:
        assert step.key in declared
        assert step.label and step.help
        assert step.question_count > 0, f"{step.key} is an empty step"


def test_mandatory_and_optional_are_distinguished(opened, service):
    form = _client_form.build(opened.case, cat=service.onboarding.catalogue)
    assert 0 < form.required_count < form.question_count


def test_a_second_portfolio_does_not_repeat_client_level_questions(service,
                                                                   opened):
    cat = service.onboarding.catalogue
    one = _client_form.build(opened.case, cat=cat)
    two = _client_form.build(
        service.instruct(opened,
                         text="They also have portfolio id direct_102, "
                              "equity release.",
                         actor=ACTOR, confirm=True).case.case, cat=cat)

    def contacts(form):
        return [k for k in form.keys() if k.startswith("contacts.")]

    assert contacts(one) == contacts(two), "contacts were asked twice"
    # The extra questions are the book's, and only the book's.
    extra = set(two.keys()) - set(one.keys())
    assert extra
    assert all(k.startswith("portfolios[") for k in extra), extra


def test_each_portfolio_gets_its_own_group(service, opened):
    case = service.instruct(
        opened, text="They also have portfolio id direct_102, equity release.",
        actor=ACTOR, confirm=True).case
    form = _client_form.build(case.case, cat=service.onboarding.catalogue)
    groups = [g for s in form.steps for g in s.groups
              if g.key == "portfolios"]
    assert len(groups) == 2
    assert {g.index for g in groups} == {0, 1}


# --------------------------------------------------------------------------- #
# Structured answers are persisted verbatim
# --------------------------------------------------------------------------- #

def test_every_form_key_is_an_authoritative_catalogue_key(opened, service):
    cat = service.onboarding.catalogue
    form = _client_form.build(opened.case, cat=cat)
    for key in form.keys():
        section_key, _index, field_key = _client_form.parse_key(key)
        assert cat.field(section_key, field_key) is not None, key


def test_a_structured_answer_is_written_exactly_as_submitted(opened, service):
    case = service.submit_client_response(
        opened, actor=ACTOR,
        response={"contacts.reporting_contact_name": "Dana Fox",
                  "entities[0].lei": "894500SYNTHETIC00042"})
    assert case.case.answers["contacts"]["reporting_contact_name"] == "Dana Fox"
    assert case.case.items("entities")[0]["lei"] == "894500SYNTHETIC00042"


def test_a_structured_answer_records_the_client_as_its_source(opened, service):
    case = service.submit_client_response(
        opened, actor=ACTOR,
        response={"contacts.reporting_contact_name": "Dana Fox"})
    assert case.case.provenance_class[
        "contacts.reporting_contact_name"] == "client_supplied"


def test_a_key_the_catalogue_does_not_declare_is_refused(opened, service):
    with pytest.raises(OpsError) as excinfo:
        service.submit_client_response(
            opened, actor=ACTOR, response={"contacts.favourite_colour": "blue"})
    assert excinfo.value.code == "OCC_AGENT_UNKNOWN_ANSWER_KEY"


def test_a_field_the_client_was_not_asked_is_refused(opened, service):
    """A real field is not the same as a question this client was put."""
    with pytest.raises(OpsError) as excinfo:
        service.submit_client_response(
            opened, actor=ACTOR, response={"client.client_id": "HIJACKED"})
    assert excinfo.value.code == "OCC_AGENT_NOT_A_CLIENT_QUESTION"
    reloaded = service.load(TENANT_A, opened.case_ref)
    assert reloaded.case.answers["client"]["client_id"] != "HIJACKED"


def test_a_value_outside_the_declared_options_is_refused(opened, service):
    with pytest.raises(OpsError) as excinfo:
        service.submit_client_response(
            opened, actor=ACTOR,
            response={"portfolios[0].portfolio_type": "nonsense"})
    assert excinfo.value.code == "OCC_AGENT_RESPONSE_INVALID"


def test_a_malformed_key_is_refused(opened, service):
    for key in ("contacts", "contacts.", ".name", "contacts[x].name",
                "Contacts.Name"):
        with pytest.raises(OpsError):
            service.submit_client_response(opened, actor=ACTOR,
                                           response={key: "x"})


def test_nothing_is_written_when_any_key_is_refused(opened, service):
    """All or nothing: a client is entitled to know their answer did not land."""
    with pytest.raises(OpsError):
        service.submit_client_response(
            opened, actor=ACTOR,
            response={"contacts.reporting_contact_name": "Dana Fox",
                      "contacts.favourite_colour": "blue"})
    reloaded = service.load(TENANT_A, opened.case_ref)
    assert not (reloaded.case.answers.get("contacts") or {}).get(
        "reporting_contact_name")


# --------------------------------------------------------------------------- #
# No interpretation on the structured lane
# --------------------------------------------------------------------------- #

def test_the_form_module_cannot_reach_an_interpreter():
    """The structural guarantee behind "persisted without reinterpretation"."""
    source = Path(_client_form.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    for forbidden in ("interpretation", "extraction", "planning",
                      ".interpretation", ".extraction"):
        assert not any(forbidden in name for name in imported), forbidden


def test_submitting_a_response_never_calls_the_interpreter(opened, service):
    class Exploding:
        def interpret_instruction(self, text):        # pragma: no cover
            raise AssertionError("the structured lane interpreted an answer")

        def interpret_action(self, text, run, case):  # pragma: no cover
            raise AssertionError("the structured lane interpreted an answer")

    service.interpreter = Exploding()
    case = service.submit_client_response(
        opened, actor=ACTOR,
        response={"contacts.reporting_contact_name": "map balance to nonsense"})
    # Even an answer that reads like an instruction is stored as a value.
    assert case.case.answers["contacts"]["reporting_contact_name"] == \
        "map balance to nonsense"


def test_planning_a_response_is_a_pure_function(opened, service):
    """Same case, same response, same plan — twice, and in either order."""
    cat = service.onboarding.catalogue
    response = {"contacts.reporting_contact_name": "Dana Fox",
                "entities[0].lei": "894500SYNTHETIC00042"}
    first = _client_form.plan_response(opened.case, response, cat=cat)
    second = _client_form.plan_response(opened.case, dict(
        reversed(list(response.items()))), cat=cat)
    assert first.steps == second.steps
    assert first.accepted == second.accepted


def test_the_form_is_deterministic(opened, service):
    cat = service.onboarding.catalogue
    first = _client_form.build(opened.case, cat=cat)
    second = _client_form.build(opened.case, cat=cat)
    assert first.content_hash == second.content_hash
    assert first.to_dict() == second.to_dict()


def test_the_classification_is_pure(opened, service):
    """Classifying writes nothing to the case."""
    before = inspect.getsource(_classification)
    assert "save_step" not in before
    assert "save_case" not in before
    snapshot = dict(opened.case.answers)
    _classification.classify_all(opened.case.answers,
                                 cat=service.onboarding.catalogue)
    assert opened.case.answers == snapshot


# --------------------------------------------------------------------------- #
# Business meaning is asked; technical file detail is deferred
# --------------------------------------------------------------------------- #

#: What only the client can tell Trakt, because two lenders can send the same
#: column name and mean different things by it.
BUSINESS_DEFINITIONS = (
    "balance_definition", "valuation_basis", "cashflow_basis",
    "units_and_currency", "cut_off_convention", "accrued_interest_treatment",
    "redemption_definition", "status_definitions", "gross_net_convention",
    "measure_basis",
)

#: What the Onboarding Agent reads out of the file itself.
TECHNICAL_DEFINITIONS = (
    "source_file", "description", "proprietary_fields", "date_conventions",
    "known_limitations",
)


def test_both_kinds_of_definition_are_deferred_by_the_catalogue():
    """Meaning and file detail both wait for a file — and the catalogue says so.

    Trakt does not ask a client to map their fields; asking them in the
    abstract what a balance means is the same request wearing a different hat.
    Both are answered from the file, then confirmed. Neither section is
    deleted: every field is still declared, and still collected.
    """
    cat = catalogue()
    business = cat.section("data_semantics")
    technical = cat.section("data_definitions")
    assert business is not None and technical is not None
    assert business.deferred_until, \
        "business meaning must wait for a file, so it can be asked about one"
    assert technical.deferred_until, \
        "file-specific detail must be deferred by the catalogue"
    # Deferred, not dropped.
    assert {f.key for f in business.fields} == set(BUSINESS_DEFINITIONS)
    assert {f.key for f in technical.fields} == set(TECHNICAL_DEFINITIONS)


def test_the_deferral_is_not_hardcoded():
    """``deferred_sections`` reads the catalogue rather than holding a list.

    This is the property that made the change a configuration edit: no code
    names ``data_semantics``, so which questions wait is a catalogue decision.
    """
    cat = catalogue()
    deferred = _classification.deferred_sections(cat)
    assert deferred == {s.key: s.deferred_until for s in cat.sections
                        if s.deferred_until}
    assert "data_definitions" in deferred
    assert "data_semantics" in deferred


def test_no_generic_semantic_question_reaches_the_first_communication(opened,
                                                                     service):
    """1 — the six a client should never be asked cold, and the rest with them.

    "What does a balance mean?" asks the client to write a data dictionary for
    a file Trakt has not looked at. None of these may appear before one has
    arrived.
    """
    form = _client_form.build(opened.case, cat=service.onboarding.catalogue)
    keys = set(form.keys())
    for never in BUSINESS_DEFINITIONS:
        assert f"data_semantics.{never}" not in keys, never
    assert not [s for s in form.steps if s.key == "meaning"], \
        "the meaning step has no place in the initial client form"


def test_the_deferred_questions_are_reported_not_silently_dropped(opened,
                                                                  service):
    """6 — internal completeness survives the simplification.

    Every deferred field is still classified, still carries the reason it is
    absent, and is still named to the client — so "why am I not being asked
    that?" has an answer, and nothing falls out of the catalogue unnoticed.
    """
    built = service.build_pack(opened)
    reported = {row["key"]: row for row in built.not_asked}
    for key in BUSINESS_DEFINITIONS:
        path = f"data_semantics.{key}"
        assert path in reported, f"{key} vanished instead of being reported"
        assert reported[path]["reason"], f"{key} must say why it is not asked"
    # Those that apply to this client are deferred to the first file; the rest
    # are out of scope for its products, which is a different answer and is
    # given as one.
    deferred = [k for k in BUSINESS_DEFINITIONS
                if reported[f"data_semantics.{k}"]["category"]
                == "first_delivery"]
    assert deferred, "the applicable semantics must be deferred, not dropped"
    assert "representative file" in \
        reported[f"data_semantics.{deferred[0]}"]["reason"]


def test_the_conditions_on_deferred_questions_are_kept_for_later(service):
    """1 — deferring is not deleting: the applicability rules still stand.

    ``asked_when`` decides whether a question applies at all. It has to survive
    the deferral, because the post-ingestion clarification needs it to know
    which questions are even relevant to this client.
    """
    cat = service.onboarding.catalogue
    conditional = ("valuation_basis", "cashflow_basis",
                   "accrued_interest_treatment")
    for key in conditional:
        assert cat.field("data_semantics", key).asked_when, key

    # And no product selection brings them forward into the first ask.
    with_mi = service.create_case(
        tenant=TENANT_A, initiating_user=ACTOR,
        instruction="Onboard Northstar Lending. UK equity release. Monthly "
                    "portfolio MI. Portfolio id direct_101.")
    asked = set(_client_form.build(with_mi.case, cat=cat).keys())
    for key in conditional:
        assert f"data_semantics.{key}" not in asked, key


def test_technical_file_detail_stays_deferred(opened, service):
    """2 — mappings, schema and file formatting are never in the pack."""
    cat = service.onboarding.catalogue
    form = _client_form.build(opened.case, cat=cat)
    keys = set(form.keys())
    for technical in TECHNICAL_DEFINITIONS:
        assert f"data_definitions.{technical}" not in keys, technical
    # And it is reported as waiting, not as absent.
    locked = {row["step"] for row in form.locked}
    assert "data" in locked

    rows = _classification.classify_all(opened.case.answers, cat=cat)
    deferred = {r.field for r in rows
                if r.section == "data_definitions"}
    assert deferred == set(TECHNICAL_DEFINITIONS)
    for row in rows:
        if row.section == "data_definitions":
            assert row.category == _classification.FIRST_DELIVERY


def test_mappings_and_fingerprints_are_still_not_collected_at_all():
    """2 — the governed `not_collected` decision is untouched."""
    not_collected = {row["key"] for row in catalogue().not_collected}
    assert "file_role_schemas" in not_collected
    assert "expected_schema_fingerprint" in not_collected
    # And no section asks for one under another name.
    for section in catalogue().sections:
        for f in section.fields:
            assert "fingerprint" not in f.key, f"{section.key}.{f.key}"
            assert "canonical_field" not in f.key, f"{section.key}.{f.key}"


def test_the_pack_stays_progressive_rather_than_flat(opened, service):
    """3 — adding business questions must not revert to a flat projection."""
    cat = service.onboarding.catalogue
    declared = sum(len(s.fields) for s in cat.sections)
    rows = _classification.classify_all(opened.case.answers, cat=cat)
    form = _client_form.build(opened.case, cat=cat)

    # Still materially fewer questions than fields.
    assert form.question_count < declared / 2, (
        f"{form.question_count} questions from {declared} declared fields")
    # And the reduction is because of the OTHER categories, not truncation.
    asked = len([r for r in rows if r.client_facing])
    assert form.question_count == asked
    for category in (_classification.KNOWN, _classification.FIRST_DELIVERY,
                     _classification.INTERNAL):
        assert [r for r in rows if r.category == category], category


def test_answering_a_question_removes_it_from_the_pack(opened, service):
    """3 — progressive in the other direction: answered questions go away."""
    cat = service.onboarding.catalogue
    before = _client_form.build(opened.case, cat=cat).question_count
    answered = service.submit_client_response(
        opened, actor=ACTOR,
        response={"contacts.reporting_contact_name": "Dana Fox",
                  "contacts.reporting_contact_email": "dana@northstar.example"})
    after = _client_form.build(answered.case, cat=cat)
    assert after.question_count == before - 2
    assert "contacts.reporting_contact_email" not in set(after.keys())


def test_a_deferred_definition_can_still_be_recorded_when_it_is_answered(
        opened, service):
    """6 — the requirement is deferred, never removed.

    The onboarding record still has somewhere for every one of these to go, so
    the post-ingestion clarification writes through the SAME governed path the
    initial pack would have used.
    """
    cat = service.onboarding.catalogue
    for key in BUSINESS_DEFINITIONS:
        declared = cat.field("data_semantics", key)
        assert declared is not None, key
        assert declared.writes_to, f"{key} must still have a destination"


# --------------------------------------------------------------------------- #
# Provenance in the operator review
# --------------------------------------------------------------------------- #

def test_the_operator_review_distinguishes_where_a_value_came_from(service):
    """4 — five kinds of "already known", told apart."""
    from operations_control.occ_agent.scenarios import run_scenario

    run = run_scenario(service, "scenario_a_clean", tenant=TENANT_A,
                       actor=ACTOR)
    package = service.build_review_package(run.case)
    rows = [r for s in package.sections for r in s["rows"]]
    assert rows

    seen = {r["provenance"] for r in rows}
    for expected in ("human_supplied", "client_supplied", "trakt_derived",
                     "inherited_default", "artefact_derived"):
        assert expected in seen, f"{expected} not distinguished; saw {seen}"


def test_no_reviewed_value_is_left_unattributed(service):
    """4 — every value says something; none reads "not recorded"."""
    from operations_control.occ_agent.scenarios import run_scenario

    run = run_scenario(service, "scenario_a_clean", tenant=TENANT_A,
                       actor=ACTOR)
    package = service.build_review_package(run.case)
    for section in package.sections:
        for row in section["rows"]:
            assert row["provenance"], f"{row['label']} has no provenance"
            assert row["provenance_label"] != "not recorded", row["label"]


def test_every_provenance_value_has_a_human_label():
    from operations_control.occ_agent import review as _review
    from operations_control.occ_agent.interpretation import PROVENANCE_SOURCES

    for source in PROVENANCE_SOURCES:
        assert source in _review.PROVENANCE_LABELS, source


def test_a_default_and_a_derivation_read_differently(service, opened):
    """The distinction that matters: what Trakt chose vs what it computed."""
    from operations_control.occ_agent import review as _review

    package = service.build_review_package(opened)
    # Keyed by SECTION and field: two sections declare a reporting currency,
    # and they come from different places.
    by_path = {f"{s['key']}.{r['field']}": r
               for s in package.sections for r in s["rows"]}
    # A governed default Trakt applied.
    assert by_path["client.reporting_currency"]["provenance"] == \
        "inherited_default"
    # Something Trakt worked out from another answer.
    assert by_path["entities.entity_id"]["provenance"] == "trakt_derived"
    # And the labels are not the same sentence.
    assert (_review.PROVENANCE_LABELS["inherited_default"]
            != _review.PROVENANCE_LABELS["trakt_derived"])
