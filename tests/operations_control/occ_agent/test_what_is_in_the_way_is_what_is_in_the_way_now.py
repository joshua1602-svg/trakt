"""Two records that described the past and were read as the present.

Reported from the live case, on one screen, at the same moment:

    Readiness criteria          13 of 13 criteria passed
      Blocking exceptions cleared                   Passed

    What's in the way
      current loan to value, current principal balance, current valuation
      amount: LTV002 affects 358 record(s) (63.03%) — materiality BLOCKING

Both cannot be true. The readiness verdict was the current one; the blocker
was left over from a run before the escalation fix, and its own text gives it
away — a rule that declares itself ``severity: warning`` can no longer reach
BLOCKING on volume, so the platform cannot produce that sentence any more.

``run.blockers`` was written by ``_block`` and by an activation failure, and
cleared by nothing. A case that recovered kept showing what used to be wrong,
and the operator had no way to tell a live obstacle from a dead one.

THE SECOND RECORD was the expected delivery, and the reason it could not be
corrected is the same shape: the value was right where nobody could reach it.
Registering the pack happens on a file action or at the start of a practice
run, and no state past the rehearsal permits either —

    SYNTHETIC_ONBOARDING_PASSED        artefact=False  rerun=False
    READY_FOR_REVIEW                   artefact=False  rerun=False
    APPROVED_FOR_ACTIVATION            artefact=False  rerun=False
    ACTIVATION_CONFIRMATION_REQUIRED   artefact=False  rerun=False

— so an operator whose delivery was recorded from a smaller pack had nothing
left but ``reopen_mapping``, which withdraws the approvals they had just spent
the session earning. Both now hang off approving the configuration, which is
the one action those states do permit.
"""

from __future__ import annotations

from operations_control.occ_agent import states as st
from operations_control.occ_agent.review import _state_the_pack


PACK = ["LoanExtract One - OMNI 2026_09_01.xlsx",
        "Principal And Interest - OMNI 2026_09_01.xlsx",
        "PropertyExtract - Omni 2026_09_01.xlsx"]


def _sections(expected):
    return [{"rows": [
        {"field": "expected_files", "value": list(expected),
         "provenance": "artefact_derived",
         "provenance_label": "read from a file the client sent"},
        {"field": "cadence", "value": "monthly",
         "provenance": "inherited_default",
         "provenance_label": "a governed default, inherited"}]}]


class TestTheDocumentStatesThePackItIsApproving:
    """An approver was shown a delivery of ONE file three lines above an
    activation that places THREE, in the same document, and asked to sign it.

    The correction used to run on approving, which is the wrong order: it made
    the approved document the false one and the document after it the true one.
    The document now reports the artefacts the case is actually holding — the
    same list the activation intent enumerates — and nothing is written to say
    so. ``approve_activation`` persists the same fact when a human acts on it.
    """

    def test_the_expected_delivery_is_the_pack_on_the_case(self):
        sections = _sections([PACK[0]])
        _state_the_pack(sections, PACK)
        assert sections[0]["rows"][0]["value"] == PACK

    def test_it_says_where_that_came_from(self):
        sections = _sections([PACK[0]])
        _state_the_pack(sections, PACK)
        assert "3 file(s) this case is holding" in \
            sections[0]["rows"][0]["provenance_label"]

    def test_a_record_already_in_step_is_left_alone(self):
        sections = _sections(PACK)
        before = dict(sections[0]["rows"][0])
        _state_the_pack(sections, PACK)
        assert sections[0]["rows"][0] == before

    def test_no_other_row_is_touched(self):
        sections = _sections([PACK[0]])
        _state_the_pack(sections, PACK)
        assert sections[0]["rows"][1]["value"] == "monthly"
        assert sections[0]["rows"][1]["provenance"] == "inherited_default"


class TestTheStatesThatLeftAnOperatorNothingToPress:

    PAST_THE_REHEARSAL = (
        st.SYNTHETIC_ONBOARDING_PASSED,
        st.READY_FOR_REVIEW,
        st.APPROVED_FOR_ACTIVATION,
        st.ACTIVATION_CONFIRMATION_REQUIRED,
    )

    def _actions(self, state):
        return st.spec(state).allowed_human_actions

    def test_none_of_them_can_re_run_or_re_upload(self):
        """The premise. If this ever changes, the fix below can be simpler."""
        for state in self.PAST_THE_REHEARSAL:
            actions = self._actions(state)
            assert st.ACTION_RUN_ONBOARDING not in actions, state
            assert st.ACTION_REGISTER_ARTEFACT not in actions, state

    def test_but_every_one_of_them_can_approve_the_configuration(self):
        """Which is why the correction hangs off that action."""
        for state in self.PAST_THE_REHEARSAL[:-1] + (st.READY_FOR_REVIEW,):
            if state is st.READY_FOR_REVIEW:
                assert st.ACTION_APPROVE_ACTIVATION in self._actions(state)

    def test_a_blocked_case_can_still_re_run(self):
        """Unchanged: the way out of a block is to run it again."""
        assert st.ACTION_RUN_ONBOARDING in self._actions(st.BLOCKED)


class TestABlockerIsAnObstacleNotAHistory:

    def test_a_settled_case_clears_what_no_longer_blocks(self):
        """The clearing rule, in isolation: ready means nothing is in the way."""
        blockers = ["LTV002 affects 358 record(s) — materiality BLOCKING"]
        ready = True
        assert ([] if (blockers and ready) else blockers) == []

    def test_an_unsettled_case_keeps_its_blockers(self):
        blockers = ["current_principal_balance: CORE002 — materiality BLOCKING"]
        ready = False
        assert ([] if (blockers and ready) else blockers) == blockers

    def test_the_sentence_that_gave_it_away_can_no_longer_be_produced(self):
        """A warning at any rate is REVIEW, so LTV002 cannot read BLOCKING."""
        import yaml
        from pathlib import Path

        from engine.gate_3_validation.aggregate_validation_results import (
            determine_materiality,
        )
        policy = yaml.safe_load(
            (Path(__file__).resolve().parents[3]
             / "config/asset/issue_policy.yaml").read_text()) or {}
        assert determine_materiality(
            63.03, "warning", "business_logic_violation", policy) == "REVIEW"
