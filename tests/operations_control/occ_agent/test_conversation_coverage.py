"""Can the conversation answer every question the pack asks?

The gap this closes: the agent generated a checklist it could not itself take
answers for. An operator reading "the LEI is still needed" then typed "the LEI
is 894500..." and was told Trakt did not understand.

So the coverage target is the catalogue, not a list written here. For every
field the catalogue says is collected, a sentence built from that field's own
label must be read back to that field. Add a field to
``config/onboarding/field_catalogue.yaml`` and this test starts asking about it
immediately — which is the point.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import pytest

from operations_control.occ_agent import extraction as _extraction
from operations_control.onboarding.catalogue import catalogue

CAT = catalogue()
COLLECTED = _extraction.collected_fields(CAT)

#: A plausible value for each declared shape. Deliberately not per-field: the
#: point is that the FIELD's own declaration drives recognition.
_BY_RULE = {
    "lei": "894500SYNTHETIC00042",
    "email": "dana@northstar.example",
    "currency": "GBP",
    "country": "GB",
    "identifier": "direct_101",
    "colour": "#112233",
    "date": "2026-06-30",
    "date_or_nd": "2026-06-30",
    "yes_no": "yes",
    "boolean": "yes",
}


def _value_for(ref: _extraction.FieldRef) -> Any:
    if ref.options:
        return ref.options[0]
    return _BY_RULE.get(ref.validation or ref.type, "Practice Value")


def _sentence(ref: _extraction.FieldRef, value: Any) -> str:
    return f"The {ref.label.lower()} is {value}."


def _ids(pair: Tuple[Any, ...]) -> str:
    return pair[0].path


# --------------------------------------------------------------------------- #
# Coverage
# --------------------------------------------------------------------------- #

def test_there_is_something_to_cover():
    assert len(COLLECTED) > 30, len(COLLECTED)


@pytest.mark.parametrize("ref", COLLECTED, ids=lambda r: r.path)
def test_every_collected_field_can_be_answered_in_words(ref):
    """A sentence naming the field, in the catalogue's own words, is read."""
    value = _value_for(ref)
    reading = _extraction.read(_sentence(ref, value), CAT)
    hit = next((f for f in reading.found if f.ref.path == ref.path), None)
    assert hit is not None, (
        f"'{_sentence(ref, value)}' did not answer {ref.path}; "
        f"found {[f.ref.path for f in reading.found]}")


@pytest.mark.parametrize("ref", COLLECTED, ids=lambda r: r.path)
def test_the_value_read_back_is_the_value_given(ref):
    """The value survives the round trip, in the shape the FIELD declares.

    A ``boolean`` field turns "yes" into ``True`` and a ``yes_no`` field into
    "Y" — so the comparison is against the field's own coercion, not against
    the raw text.
    """
    value = _value_for(ref)
    reading = _extraction.read(_sentence(ref, value), CAT)
    hit = next(f for f in reading.found if f.ref.path == ref.path)
    read_back = hit.value[0] if isinstance(hit.value, list) else hit.value
    expected = _extraction._coerce(ref, str(value))
    assert str(read_back).lower() == str(expected).lower(), ref.path


def test_every_field_has_at_least_one_cue():
    for section in CAT.sections:
        for f in section.fields:
            if f.collected:
                assert _extraction.cues_for(section, f), \
                    f"{section.key}.{f.key} cannot be named"


def test_every_checklist_item_can_be_answered(service):
    """The specific failure: the agent's own checklist, answered back to it."""
    from .conftest import ACTOR, TENANT_A

    agent_case = service.create_case(
        tenant=TENANT_A, initiating_user=ACTOR,
        instruction="Onboard Northstar Lending. UK equity release. Monthly "
                    "portfolio MI. Portfolio id direct_101.")
    checklist = service.onboarding.client_checklist(agent_case.case)
    assert checklist, "there was nothing outstanding to answer"

    unanswerable: List[str] = []
    for row in checklist:
        ref = next((r for r in COLLECTED
                    if r.section == row["section"] and r.key == row["field"]),
                   None)
        if ref is None:
            unanswerable.append(f"{row['section']}.{row['field']} (not readable)")
            continue
        reading = _extraction.read(_sentence(ref, _value_for(ref)), CAT)
        if not any(f.ref.path == ref.path for f in reading.found):
            unanswerable.append(ref.path)
    assert unanswerable == [], unanswerable


# --------------------------------------------------------------------------- #
# The phrasings an operator actually uses
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("text,path,value", [
    ("The LEI is 894500SYNTHETIC00042", "entities.lei",
     "894500SYNTHETIC00042"),
    ("Their reporting contact is dana@northstar.example",
     "contacts.reporting_contact_email", "dana@northstar.example"),
    ("The reporting contact is Dana Fox", "contacts.reporting_contact_name",
     "Dana Fox"),
    ("They are established in Ireland", "entities.country_of_establishment",
     "IE"),
    ("They report in sterling", "client.reporting_currency", "GBP"),
    ("Portfolio id direct_102", "portfolios.portfolio_id", "direct_102"),
    ("Files arrive by SFTP", "sources.delivery_channel", "sftp"),
    ("The file format is csv", "sources.file_format", "csv"),
    ("They deliver monthly", "sources.cadence", "monthly"),
    ("The book was acquired", "portfolios.portfolio_type", "acquired"),
    ("The portfolio is an acquired book", "portfolios.portfolio_type",
     "acquired"),
])
def test_the_phrasings_an_operator_uses_are_understood(text, path, value):
    reading = _extraction.read(text, CAT)
    found = {f.ref.path: f.value for f in reading.found}
    assert path in found, f"{text!r} -> {sorted(found)}"
    read_back = found[path]
    read_back = read_back[0] if isinstance(read_back, list) else read_back
    assert str(read_back) == value


def test_two_answers_in_one_sentence_both_land():
    reading = _extraction.read(
        "The operational contact is Sam Patel and the operational email is "
        "sam@northstar.example", CAT)
    found = {f.ref.path: f.value for f in reading.found}
    assert found.get("contacts.operational_contact_name") == "Sam Patel"
    assert found.get("contacts.operational_contact_email") == \
        "sam@northstar.example"


def test_a_sentence_that_answers_nothing_is_reported_not_dropped():
    reading = _extraction.read("Chase them for the outstanding items", CAT)
    assert reading.found == []
    assert reading.unrecognised == ["Chase them for the outstanding items"]


def test_a_half_understood_sentence_reports_the_half_it_missed():
    reading = _extraction.read(
        "The LEI is 894500SYNTHETIC00042 and sort out the other thing", CAT)
    assert any(f.ref.path == "entities.lei" for f in reading.found)
    assert reading.unrecognised, "the trailing clause was dropped silently"


def test_recognition_is_deterministic():
    text = ("Onboard Northstar Lending. UK equity release. Monthly portfolio "
            "MI. Portfolio id direct_101.")
    first = [(f.ref.path, f.value, f.confidence)
             for f in _extraction.read(text, CAT).found]
    _extraction.reset_cache()
    second = [(f.ref.path, f.value, f.confidence)
              for f in _extraction.read(text, catalogue()).found]
    assert first == second


# --------------------------------------------------------------------------- #
# An identifier is not a sentence
# --------------------------------------------------------------------------- #

def test_a_portfolio_id_is_not_read_as_the_words_inside_it():
    """``direct_001`` is an identifier, not the enum value "direct".

    Separators were flattened before matching so that "equity-release" and
    "equity_release" read as one phrase. That also turned ``direct_001`` into
    the words "direct 001", so the bare-option matcher found ``portfolio_type``
    inside a portfolio id — and naming a source in order to answer a question
    about it proposed changing the book's type instead. The answer was dropped
    and an unrelated field on another section was changed.
    """
    reading = _extraction.read(
        "The file format for direct_001/funded is csv", CAT)
    found = {f.ref.path: f.value for f in reading.found}
    assert "portfolios.portfolio_type" not in found
    # And the sentence still reads for what it actually says.
    assert found.get("sources.file_format") == "csv"
    assert found.get("sources.dataset") == "funded"


def test_the_phrase_flattening_it_existed_for_still_works():
    for text in ("equity_release book", "equity-release book",
                 "equity release book"):
        found = {f.ref.path: f.value for f in _extraction.read(text, CAT).found}
        assert found.get("portfolios.asset_class") == "equity_release", text


def test_an_option_word_standing_alone_is_still_read():
    found = {f.ref.path: f.value
             for f in _extraction.read("the direct book", CAT).found}
    assert found.get("portfolios.portfolio_type") == "direct"
