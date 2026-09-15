"""Cancelling a case: the words the screen teaches, and the reason you gave.

TWO DEFECTS, ONE MOMENT.

Abandoning a case is deliberately not offered as a next action — ``cancel`` and
``withdraw`` sit in ``_NOT_A_WAY_FORWARD`` because suggesting them as the way
forward is noise. So the only way to reach them is to type the instruction, and
that puts the whole weight of the feature on two things being right: the words
being the ones an operator would actually reach for, and what they typed
surviving into the record.

Neither was.

1. THE WORDS THE SCREEN ITSELF USES DID NOT WORK.

   The Agent tab heads its list "Practice cases". The pattern was
   ``cancel (this )?(case|run)``, so "cancel this practice case" — the phrasing
   the screen teaches — fell through to an unrecognised instruction, as did
   "cancel this onboarding". An operator who reads the screen and says what it
   says is the one person the reader should never fail.

2. THE REASON WAS THROWN AWAY.

   Dispatch called ``cancel(agent_case, actor=actor)`` with no reason, so the
   audit recorded "cancelled by the operator" and the withdrawal reason became
   "The practice case was cancelled." — whatever the operator had typed. For a
   record whose entire purpose is to answer, months later, why this client was
   started and never finished, a default sentence is the one answer that is
   worthless. The governed dialog requires a reason and refuses a blank one;
   the chat path collected one and discarded it.
"""

from __future__ import annotations

import pytest

from operations_control.occ_agent import states as _states
from operations_control.occ_agent.interpretation import (
    DeterministicInterpreter, InterpretationError,
)


@pytest.fixture(scope="module")
def read():
    """Read an instruction the way the Agent does, with no case context.

    Action rules are matched before anything that needs a case, so a bare
    interpreter is enough and the test stays about the words.
    """
    interpreter = DeterministicInterpreter()

    def _read(text: str):
        return interpreter.interpret_action(text, None, None)

    return _read


class TestTheWordsTheScreenTeaches:
    """Every phrasing here is one an operator plausibly types."""

    @pytest.mark.parametrize("sentence", [
        "Cancel this case.",
        "Cancel this run.",
        "cancel case",
        # The screen's own vocabulary. This is the regression.
        "Cancel this practice case.",
        "Cancel the practice case.",
        # What the governed dialog calls the same act, said to the Agent.
        "Cancel this onboarding.",
    ])
    def test_it_is_read_as_cancelling(self, read, sentence):
        assert read(sentence).action == _states.ACTION_CANCEL

    def test_withdraw_still_withdraws(self, read):
        assert read("Withdraw this onboarding.").action == \
            _states.ACTION_WITHDRAW

    def test_cancelling_still_needs_confirming(self, read):
        """Abandoning a case is material. It is never done on one sentence."""
        change = read("Cancel this practice case.")
        assert change.material is True
        assert change.requires_confirmation is True

    @pytest.mark.parametrize("sentence", [
        "Cancel the pack before it goes out.",
        "Do not cancel this case.",
    ])
    def test_it_does_not_swallow_neighbouring_sentences(self, read, sentence):
        """Widening the pattern must not make it match everything.

        "cancel the pack" is a different act, and a negation is not an
        instruction to do the thing negated. Neither may reach cancel.

        What they are read as instead is not this test's business — today one
        refuses as unrecognised and the other is caught upstream as a question.
        Both outcomes are safe; being read as cancel is the only one that is
        not, so that is the whole assertion.
        """
        try:
            assert read(sentence).action != _states.ACTION_CANCEL
        except InterpretationError:
            pass                    # refused outright, which is also not cancel


class TestTheReasonSurvives:
    """What the operator typed, on the record, verbatim."""

    def test_the_sentence_reaches_the_payload(self, read):
        said = ("Cancel this case. Superseded by a fresh onboarding after "
                "platform remediation.")
        assert read(said).payload.get("reason") == said

    def test_withdrawing_carries_it_too(self, read):
        said = "Withdraw this onboarding. The client did not proceed."
        assert read(said).payload.get("reason") == said

    def test_an_unrelated_action_carries_no_reason(self, read):
        """Only the two acts that end a case collect one."""
        change = read("Draft the onboarding pack.")
        assert "reason" not in change.payload

    def test_a_very_long_sentence_is_still_valid(self, read):
        """`validate()` caps payload strings; the reason must respect that."""
        said = "Cancel this case. " + ("Superseded. " * 400)
        change = read(said)
        change.validate()          # must not raise
        assert change.payload["reason"]
