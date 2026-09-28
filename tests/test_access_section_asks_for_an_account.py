"""Who needs Trakt access: a name and a work email, and nothing else.

WHY THIS SHRANK

The section asked a client to design their own entitlements — a role from a
four-way enum ("Approves deliveries and configuration"), which portfolios each
person may see, and three booleans for Operations Control Centre access,
dashboard access and report delivery by email. Seven questions to collect two
facts.

Under a MANAGED SERVICE none of that is the client's to decide. The Operations
Control Centre is operated by Trakt, not by the lender. Reports reach people
through the platform rather than an email distribution list the client
maintains. Everyone named needs the same thing: an account.

Asking anyway is not thoroughness. It invites someone to stall over what
"Approves deliveries and configuration" might commit them to, and it collects
answers the operator will override — the worst kind of question, because it
reads as consequential and is not.

What remains is exactly what creating the account requires.
"""

from __future__ import annotations

import pytest

from operations_control.onboarding.catalogue import catalogue

#: Everything the section used to ask beyond identity. Named rather than
#: implied, so a reinstatement fails loudly instead of quietly regrowing the
#: form.
RETIRED = ("user_role", "scope_note", "occ_access_required",
           "dashboard_access_required", "report_recipient")


@pytest.fixture(scope="module")
def section():
    found = catalogue().section("access")
    assert found is not None
    return found


class TestItAsksForAnAccountAndNothingElse:
    def test_exactly_two_questions(self, section):
        assert [f.key for f in section.fields] == ["user_name", "user_email"]

    def test_both_are_required(self, section):
        assert all(f.required for f in section.fields)

    @pytest.mark.parametrize("key", RETIRED)
    def test_the_entitlement_questions_are_gone(self, section, key):
        assert section.field(key) is None, (
            f"{key} is the client being asked to design entitlements a "
            "managed service decides for them")

    def test_the_email_is_validated_as_one(self, section):
        """It becomes a sign-in identity, so a typo is not a cosmetic problem."""
        email = section.field("user_email")
        assert email.type == "email"
        assert email.validation == "email"

    def test_nobody_is_forced_to_name_anyone(self, section):
        """The section stays optional in aggregate.

        Its fields are required PER PERSON, but the section has no minimum, so
        a client with nothing to add is not held up — and an operator can add
        users later without the onboarding having refused to complete.
        """
        assert section.repeatable is True
        assert not getattr(section, "min_items", None)


class TestTheOperatorGetsOneActionPerPerson:
    """Not four, derived from booleans nobody is asked for any more."""

    def test_one_account_to_create(self):
        from operations_control.occ_agent.review import access_actions
        actions = access_actions([
            {"user_name": "A Person", "user_email": "a.person@ere.example"}])
        assert len(actions) == 1
        assert actions[0].kind == "user_account"
        assert "a.person@ere.example" in actions[0].detail

    def test_two_people_two_actions(self):
        from operations_control.occ_agent.review import access_actions
        actions = access_actions([
            {"user_name": "A", "user_email": "a@ere.example"},
            {"user_name": "B", "user_email": "b@ere.example"}])
        assert [a.kind for a in actions] == ["user_account", "user_account"]

    def test_a_row_with_no_email_still_produces_a_readable_action(self):
        """Incomplete input is reported, never silently dropped."""
        from operations_control.occ_agent.review import access_actions
        actions = access_actions([{"user_name": "A Person"}])
        assert len(actions) == 1
        assert "has not been given" in actions[0].detail
