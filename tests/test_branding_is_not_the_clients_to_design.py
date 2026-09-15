"""Brand colour, logo and disclaimer are Trakt's to set, not the client's.

WHY THIS SECTION LEFT THE CLIENT PACK

Every pack ended with a step headed "Your reports" that asked for a hex colour,
a logo, a disclaimer and a note about the reporting calendar. All four optional,
all four with a Trakt default the section itself describes as "fine".

Under a MANAGED SERVICE none of it is the client's to design at onboarding, for
the same reason the access section stopped asking them to design their own
entitlements: the answer is Trakt's, the operator will set it, and a question
that reads as consequential while changing nothing is the worst kind to ask.

The alternative on offer was worse. Because the fields are optional, an
operator could make the step disappear by answering them — and those answers do
not vanish into the form, they are written to the client's live configuration:

    brand_colour              -> client_config:mi.branding.theme.primary_color
    logo_uri                  -> client_config:mi.branding.logo_uri
    disclaimer                -> client_config:mi.branding.disclaimer
    reporting_calendar_note   -> onboarding record

and `mi.branding.disclaimer` is read by the MI surface and the Annex delivery
publication path. Answering "none" to silence the question would have printed
the word "none" as the disclaimer on delivered output. Silencing a question by
writing junk into production configuration is not an improvement on the
question.

WHAT CHANGED, AND WHAT DID NOT

Only `source`. The fields still exist, still write where they wrote, and are
still `amendable`, so an operator sets any of them whenever the client actually
expresses a preference. What stops is asking a client to invent one during
onboarding.

`report_title`, `day_count_convention` and `payment_frequency` are untouched:
they were never client-facing, because Trakt infers them.
"""

from __future__ import annotations

import pytest

from operations_control.onboarding.catalogue import catalogue
from operations_control.occ_agent import classification as _classification

#: The four questions that used to end every client pack.
RETIRED_FROM_THE_PACK = ("brand_colour", "logo_uri", "disclaimer",
                         "reporting_calendar_note")

#: What each still writes. Removing the QUESTION must not remove the plumbing —
#: an operator setting a brand colour later must still land in the same place.
STILL_WRITES_TO = {
    "brand_colour": "client_config:mi.branding.theme.primary_color",
    "logo_uri": "client_config:mi.branding.logo_uri",
    "disclaimer": "client_config:mi.branding.disclaimer",
    "reporting_calendar_note": "onboarding_record",
}


@pytest.fixture(scope="module")
def section():
    found = catalogue().section("presentation")
    assert found is not None
    return found


class TestTheClientIsNoLongerAskedToDesignTheReports:
    @pytest.mark.parametrize("key", RETIRED_FROM_THE_PACK)
    def test_it_is_not_client_supplied(self, section, key):
        field = section.field(key)
        assert field is not None, f"{key} should still exist"
        assert field.source != "client_supplied", (
            f"{key} is the client being asked to design a report Trakt "
            "already has a default for")

    @pytest.mark.parametrize("key", RETIRED_FROM_THE_PACK)
    def test_it_is_not_classified_as_a_client_question(self, section, key):
        """The property the client pack actually filters on.

        MI is selected deliberately. All four are gated
        ``asked_when: "reporting.products contains mi"``, so classifying them
        against empty answers returns "not applicable" and the assertion passes
        without testing anything. MI is also ``always_applies`` — derived for
        every delivery — so this is not a corner case, it is every client.
        """
        field = section.field(key)
        answers = {"reporting": {"products": ["mi", "esma_annex2"]}}
        row = _classification.classify(section, field, holder={},
                                       answers=answers)
        assert not row.client_facing, (
            f"{key} still reaches the client pack")

    @pytest.mark.parametrize("key", RETIRED_FROM_THE_PACK)
    def test_it_still_exists_and_still_writes_where_it_wrote(self, section, key):
        """The question goes; the field and its destination stay."""
        field = section.field(key)
        assert field.writes_to == STILL_WRITES_TO[key]
        assert field.amendable is True, (
            f"{key} must stay amendable — an operator sets it later")

    @pytest.mark.parametrize("key", RETIRED_FROM_THE_PACK)
    def test_it_is_still_optional(self, section, key):
        """Nothing becomes mandatory by ceasing to be asked."""
        assert section.field(key).required is False


class TestTheFieldsTrarktAlreadyInferredAreUntouched:
    @pytest.mark.parametrize("key", ["report_title", "day_count_convention",
                                     "payment_frequency"])
    def test_they_are_still_inferred(self, section, key):
        assert section.field(key).source == "inferred"


class TestTheStepDisappearsFromTheClientForm:
    """The point of the change, asserted where a client would see it."""

    def test_no_presentation_questions_reach_the_pack(self):
        from operations_control.onboarding.case import OnboardingCase
        from operations_control.occ_agent import client_form

        case = OnboardingCase(case_id="PROBE",
                              client_name="ERE Funding Limited")
        # MI selected: the condition that used to make all four appear.
        case.answers["reporting"] = {"products": ["mi", "esma_annex2"]}

        form = client_form.build(case).to_dict()
        presentation = [
            field["label"]
            for step in form["steps"]
            for group in step["groups"]
            if group["key"] == "presentation"
            for field in group["fields"]]
        assert presentation == [], (
            f"the client is still asked: {presentation}")
