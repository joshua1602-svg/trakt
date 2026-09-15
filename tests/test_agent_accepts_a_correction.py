"""A message that says only the field being corrected must still be applied.

THE SHAPE OF A CORRECTION

The agent reads the opening instruction and proposes a portfolio name and an
entity's roles from it. When it reads either wrongly — and a three-word
instruction gives it every chance to — the operator's reply says ONLY that
field:

    Portfolio name = "ERE Direct Origination Book"
    ERE Funding Limited has one role: reporting entity.

``_says_something`` used to discard exactly that shape. A portfolios item whose
only key was ``display_name`` was dropped; so was an entities item whose only
keys were ``legal_name`` and ``roles``. The extractor read both messages
correctly and the guard threw the reading away, so the operator got "Trakt
could not tell what to do with that" — a reply that names no field, and so
gives no hint that the message was understood and then discarded. Two operators
in a row concluded the agent could not persist anything.

WHY THE GUARD CAN GO RATHER THAN BE NARROWED AGAIN

It existed to stop a bare proper-noun run binding a value: "send it to
Northstar" proposing a client called Northstar. That protection now lives in
``extraction.Candidate.names``, which will not bind a free-text value on a bare
topic cue at all — so the phrase yields nothing to propose long before this is
consulted.

The client's own name was carved out of the guard once already for this reason.
These were the same defect one field over, which is why the guard is removed
rather than given a third exception.

Both halves are asserted here. The corrections must apply; the false positives
must still bind nothing — and must do so at EXTRACTION, so the protection
cannot quietly migrate back into a guess about which fields count.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from operations_control.occ_agent.interpretation import DeterministicInterpreter

OPENING = ("Onboard ERE Funding Limited, a UK equity release lender. "
           "Client identifier: ERE. Portfolio direct_001.")


@pytest.fixture()
def agent(tmp_path, monkeypatch):
    """A real agent service over a file-backed practice container."""
    monkeypatch.setenv("TRAKT_STORAGE_BACKEND", "file")
    monkeypatch.setenv("TRAKT_LOCAL_BLOB_ROOT", str(tmp_path / "blob"))
    monkeypatch.setenv("OCC_AGENT_SYNTHETIC_ENABLED", "1")
    from apps.blob_trigger_app.storage import Storage
    from operations_control.occ_agent.service import OccAgentService
    return OccAgentService(Storage(tmp_path / "blob"),
                           container="operations-control-synthetic",
                           sandbox=Path(tmp_path) / "sandbox")


def _case(agent):
    return agent.create_case(tenant="t1", initiating_user="operator",
                             instruction=OPENING)


def _reload(agent, agent_case):
    return agent.onboarding.load_case(agent_case.case.case_id)


# --------------------------------------------------------------------------- #
# The corrections, end to end through the real service
# --------------------------------------------------------------------------- #

class TestACorrectionIsApplied:
    def test_a_message_saying_only_the_portfolio_name(self, agent):
        agent_case = _case(agent)
        before = _reload(agent, agent_case).items("portfolios")[0]["display_name"]

        result = agent.instruct(agent_case,
                                text='Portfolio name = "ERE Direct Origination Book"',
                                actor="operator", confirm=True)

        assert result.applied is True
        after = _reload(agent, result.case).items("portfolios")[0]["display_name"]
        assert after == "ERE Direct Origination Book"
        assert after != before

    def test_a_message_saying_only_the_entity_roles(self, agent):
        agent_case = _case(agent)
        assert _reload(agent, agent_case).items("entities")[0]["roles"] \
            == ["originator"], "the opening instruction proposes originator"

        result = agent.instruct(
            agent_case, text="ERE Funding Limited has one role: reporting entity.",
            actor="operator", confirm=True)

        assert result.applied is True
        assert _reload(agent, result.case).items("entities")[0]["roles"] \
            == ["reporting_entity"]

    def test_the_correction_replaces_rather_than_accumulates(self, agent):
        """A role stated afresh is the whole answer, not another entry.

        ``roles`` is a multi-enum, and the field carries no
        ``repeated_mentions``, so a new mention replaces. Asserted because the
        opposite — accumulating — would leave the originator in place and with
        it the LEI requirement the correction exists to remove.
        """
        agent_case = _case(agent)
        result = agent.instruct(
            agent_case, text="ERE Funding Limited has one role: reporting entity.",
            actor="operator", confirm=True)
        roles = _reload(agent, result.case).items("entities")[0]["roles"]
        assert "originator" not in roles


# --------------------------------------------------------------------------- #
# The false positives the guard existed for
# --------------------------------------------------------------------------- #

class TestTheFalsePositivesAreStoppedAtExtraction:
    """Where the protection actually lives, now that the guard is gone.

    These are asserted against the EXTRACTOR rather than the service: the point
    is not merely that they are refused, but that they are refused before any
    judgement about which fields count is reached.
    """

    @pytest.mark.parametrize("phrase", [
        "send it to Northstar",
        "email the pack to ERE Funding Limited",
        "chase Northstar Lending for the tape",
        "forward this to Harbour Point Capital",
        "I spoke to ERE Funding Limited yesterday",
        "the portfolio looks fine",
    ])
    def test_a_bare_topic_cue_binds_nothing(self, phrase):
        assert DeterministicInterpreter().interpret_instruction(phrase).steps == {}

    def test_such_a_message_is_still_refused_by_the_service(self, agent):
        from operations_control.occ_agent.interpretation import (
            InterpretationError,
        )
        agent_case = _case(agent)
        with pytest.raises(InterpretationError):
            agent.instruct(agent_case, text="send it to Northstar",
                           actor="operator", confirm=True)


# --------------------------------------------------------------------------- #
# The reading itself, kept separate from what is done with it
# --------------------------------------------------------------------------- #

class TestTheExtractorReadThemAllAlong:
    """The defect was never in the reading, and this records that.

    Both messages extracted correctly before this change; the guard discarded
    the result afterwards. Keeping the two apart means a future regression
    reports which half broke.
    """

    def test_the_portfolio_name_is_read(self):
        steps = DeterministicInterpreter().interpret_instruction(
            'Portfolio name = "ERE Direct Origination Book"').steps
        assert steps["portfolios"]["portfolios"][0]["display_name"] \
            == "ERE Direct Origination Book"

    def test_the_role_is_read_as_a_single_replacing_value(self):
        steps = DeterministicInterpreter().interpret_instruction(
            "ERE Funding Limited has one role: reporting entity.").steps
        assert steps["entities"]["entities"][0]["roles"] == ["reporting_entity"]


# --------------------------------------------------------------------------- #
# A multi-valued field mentioned twice holds both
# --------------------------------------------------------------------------- #

class TestAMultiEnumKeepsEveryValue:
    """"A and B" must not quietly become "B".

    The extractor emits one hit per span it can claim, and how many spans one
    sentence produces turns on wording nobody chooses deliberately:

        "...the originator and reporting entity"      one hit, both values
        "...the originator and THE reporting entity"  two hits, one value each

    Assembling those with a plain assignment made the second overwrite the
    first, so a definite article dropped a role. Silently: the reply confirmed
    the roles it kept, an operator read their own sentence back in it, and the
    missing one surfaced later as a field that had gone optional — in the real
    case, an LEI that stopped being asked for because nobody held the
    originator role any more.
    """

    @pytest.mark.parametrize("sentence,expected", [
        ("ERE Funding Limited is the originator and the reporting entity.",
         {"originator", "reporting_entity"}),
        ("ERE Funding Limited is the originator and reporting entity.",
         {"originator", "reporting_entity"}),
        ("ERE Funding Limited is the reporting entity and the originator.",
         {"originator", "reporting_entity"}),
        ("ERE Funding Limited has two roles: originator and reporting entity.",
         {"originator", "reporting_entity"}),
        ("ERE Funding Limited is the originator, the servicer and the "
         "reporting entity.",
         {"originator", "servicer", "reporting_entity"}),
    ])
    def test_every_role_named_survives(self, sentence, expected):
        steps = DeterministicInterpreter().interpret_instruction(sentence).steps
        assert set(steps["entities"]["entities"][0]["roles"]) == expected

    def test_a_single_role_is_still_a_single_role(self):
        steps = DeterministicInterpreter().interpret_instruction(
            "ERE Funding Limited is the originator.").steps
        assert steps["entities"]["entities"][0]["roles"] == ["originator"]

    def test_stating_the_roles_afresh_still_REPLACES_what_is_stored(self, agent):
        """Merging is within one sentence, never against the case.

        Otherwise a correction could only ever add, and "one role: reporting
        entity" would leave the originator in place — the exact failure the
        merge was introduced to fix, in the other direction.
        """
        agent_case = _case(agent)
        assert _reload(agent, agent_case).items("entities")[0]["roles"] \
            == ["originator"]
        result = agent.instruct(
            agent_case, text="ERE Funding Limited has one role: reporting entity.",
            actor="operator", confirm=True)
        assert _reload(agent, result.case).items("entities")[0]["roles"] \
            == ["reporting_entity"]

    def test_and_the_correction_reaches_the_case(self, agent):
        agent_case = _case(agent)
        result = agent.instruct(
            agent_case,
            text="ERE Funding Limited is the originator and the reporting entity.",
            actor="operator", confirm=True)
        assert result.applied is True
        roles = _reload(agent, result.case).items("entities")[0]["roles"]
        assert set(roles) == {"originator", "reporting_entity"}
