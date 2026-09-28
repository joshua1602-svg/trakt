"""Every catalogue section is a step, and every step is a catalogue section.

WHY THIS IS A TEST AND NOT A CONVENTION

``STEPS`` is not a list of screens. It is the set of sections whose answers can
be WRITTEN:

* ``OnboardingService.save_step`` raises ``OPS_BAD_STEP`` for anything absent
  from it;
* ``OccAgentService.submit_client_response`` iterates ``STEPS`` and skips
  whatever it does not find, so a section missing from the tuple is never even
  offered to the writer.

A section can therefore be fully declared in the catalogue — asked of the
client, rendered on the pack, rendered in the client form, validated on the way
in — and still have nowhere to land. ``funding_facility`` and
``additional_context`` were in exactly that state: the client typed, the
operator submitted, and the value was dropped.

It is the quietest failure the onboarding can have. Both sections are
``optional_context``, so nothing goes red when an answer disappears: a section
nobody must answer looks identical when it is answered and lost as when it is
left blank. Nothing would have surfaced it except someone noticing a note they
knew they had recorded was no longer there.

So the invariant is asserted in BOTH directions. One direction alone is not
enough: sections-in-steps catches the hole these two fell into, and
steps-in-sections catches a step naming a section that does not exist, which
fails at ``catalogue.section(step)`` returning ``None`` and then silently
writing nothing at all.
"""

from __future__ import annotations

import pytest

from operations_control.onboarding.catalogue import catalogue
from operations_control.onboarding.service import STEP_LABELS, STEPS

#: The only step that is not a catalogue section. It is the wizard's closing
#: screen — review and activate — and it carries no fields of its own.
NOT_A_SECTION = {"review"}


@pytest.fixture(scope="module")
def section_keys() -> set:
    return {section.key for section in catalogue().sections}


def test_every_catalogue_section_can_be_written(section_keys):
    """A declared section with no step has nowhere to put an answer."""
    missing = sorted(section_keys - set(STEPS))
    assert missing == [], (
        "these catalogue sections are asked but cannot be saved — "
        f"save_step refuses them and submit_client_response skips them: {missing}")


def test_every_step_is_a_catalogue_section(section_keys):
    """A step naming no section writes nothing, silently."""
    unknown = sorted(set(STEPS) - section_keys - NOT_A_SECTION)
    assert unknown == [], (
        f"these steps name no catalogue section: {unknown}")


def test_every_step_has_a_label():
    """``reference()`` indexes STEP_LABELS by step; a gap is a KeyError in the
    browser rather than a missing caption."""
    unlabelled = sorted(set(STEPS) - set(STEP_LABELS))
    assert unlabelled == [], f"steps with no label: {unlabelled}"


def test_the_open_question_asks_about_output_not_about_reading_the_data():
    """``additional_context`` is the one open question, and it asks what the
    client wants OUT of Trakt.

    Asking here what would help Trakt READ the data duplicated
    ``data_semantics`` and ``data_definitions``, which ask the same thing far
    better — deferred until there is a file, against the columns actually
    found in it. This section is the only place the client is asked which
    numbers they run their book on, and nothing else in the catalogue can
    answer that for them.
    """
    section = catalogue().section("additional_context")
    assert section is not None
    field = section.field("client_context_note")
    assert field is not None, "the answer's destination key must not be renamed"

    prompt = f"{section.label} {section.help} {field.label} {field.help}".lower()
    for word in ("metric", "question", "report"):
        assert word in prompt, f"the open question no longer asks about {word}s"
    # Offered, never required: a client with nothing to add is not held up.
    assert field.required is False
    assert section.optional_context is True
