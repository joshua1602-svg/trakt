"""Reading the mapping table, and the one act that commits what you read.

WHAT WAS WRONG WITH APPROVING AS YOU GO

Every answer applied itself. Confirming one column resolved its decision and —
the moment the last blocking decision cleared — reran the whole onboarding. On
a hundred-and-fifty-column tape that makes reading the table an irreversible
walk: an operator who confirms forty columns and then realises the fortieth was
wrong has already committed thirty-nine, and the run may have moved on.

    "Workflow should be operator confirms each field and then there is a final
    'confirm' all changes button which is what persists. Before that point the
    user can change their confirmed fields."

WHAT THESE TESTS PIN

Two properties, and they are the whole design:

  * a staged answer CHANGES THE SCREEN AND NOTHING ELSE. No decision resolved,
    no rule promoted, no control rerun, nothing for promotion to read. It
    survives a reload, because the reading is the work and a closed tab must
    not cost an afternoon of it — persisted is not applied.

  * the commit is ONE ACT AND MANY RECORDS. Each column resolves in its own
    right, carrying the operator who STAGED it rather than whoever pressed the
    button: the reading is the judgement, and crediting the presser would put
    the wrong name against a hundred and fifty approvals.

And the one refusal that keeps it honest: a proposal carries a field, so
committing it as proposed is accepting a reading; a weak match or an ambiguity
carries no answer at all, so the commit refuses rather than inventing one.
"""

from __future__ import annotations

import pytest

from operations_control.engine import OpsError
from operations_control.occ_agent import staging as _staging
from operations_control.occ_agent import states as _states
from operations_control.occ_agent.scenarios import run_scenario

from .conftest import ACTOR, TENANT_A

OTHER = "Bob"


@pytest.fixture()
def halted(service):
    """A first delivery stopped at the mapping table, nothing settled."""
    run = run_scenario(service, "scenario_b_ambiguous_mapping",
                       tenant=TENANT_A, actor=ACTOR, resolve_decisions=False)
    assert run.case.run.state == _states.EXCEPTIONS_REQUIRE_INPUT
    return run.case


def _rows(service, agent_case, state=None):
    rows = service.status(agent_case)["mapping"]["rows"]
    return [r for r in rows if state is None or r["state"] == state]


def _one(service, agent_case, state):
    rows = _rows(service, agent_case, state)
    assert rows, f"the fixture has no {state} row to work with"
    return rows[0]


def _answer_the_questions(service, agent_case, actor=ACTOR):
    """Stage an answer for everything that genuinely has none.

    Where two columns claim one field the first wins and the rest are set
    aside: "confirm both" is not an answer to "which of these is it?".
    """
    taken = set()
    for row in _rows(service, agent_case, "needs_you"):
        field = row["canonical_field"]
        action = "confirm" if field and field not in taken else "not_used"
        taken.add(field)
        agent_case = service.stage_mapping(
            agent_case, source_file=row["source_file"],
            source_column=row["source_column"], action=action, actor=actor)
    return agent_case


class TestStagingChangesTheScreenAndNothingElse:
    def test_a_staged_row_says_it_is_ready_not_that_it_is_done(self, service,
                                                               halted):
        row = _one(service, halted, "proposed")
        updated = service.stage_mapping(
            halted, source_file=row["source_file"],
            source_column=row["source_column"], action="confirm", actor=ACTOR)
        after = next(r for r in _rows(service, updated)
                     if r["source_column"] == row["source_column"]
                     and r["source_file"] == row["source_file"])
        assert after["state"] == "staged"
        assert after["staged_action"] == "confirm"
        assert after["staged_field"] == row["canonical_field"]

    def test_nothing_is_resolved(self, service, halted):
        row = _one(service, halted, "proposed")
        before = [d["decision_id"] for d in halted.run.open_decisions
                  if d.get("status") == "approved"]
        updated = service.stage_mapping(
            halted, source_file=row["source_file"],
            source_column=row["source_column"], action="confirm", actor=ACTOR)
        after = [d["decision_id"] for d in updated.run.open_decisions
                 if d.get("status") == "approved"]
        assert after == before

    def test_nothing_reruns(self, service, halted):
        """Even staging the LAST outstanding answer leaves the run where it
        is. The commit is the only thing that moves it."""
        updated = _answer_the_questions(service, halted)
        for row in _rows(service, updated, "proposed"):
            updated = service.stage_mapping(
                updated, source_file=row["source_file"],
                source_column=row["source_column"], action="confirm",
                actor=ACTOR)
        assert updated.run.state == _states.EXCEPTIONS_REQUIRE_INPUT
        assert "validate" not in updated.run.stage_outcomes

    def test_promotion_has_nothing_to_read(self, service, halted):
        """A case abandoned mid-draft must leave no mapping behind."""
        from operations_control.occ_agent import mapping_promotion
        row = _one(service, halted, "proposed")
        updated = service.stage_mapping(
            halted, source_file=row["source_file"],
            source_column=row["source_column"], action="confirm", actor=ACTOR)
        promotable = [d for d in updated.run.open_decisions
                      if mapping_promotion.mapping_of(d) is not None]
        assert promotable == []

    def test_it_survives_a_reload(self, service, halted):
        """The reading is the work. A closed tab must not cost an afternoon of
        it, which is why the draft is on the run and not in the browser."""
        row = _one(service, halted, "proposed")
        service.stage_mapping(
            halted, source_file=row["source_file"],
            source_column=row["source_column"], action="confirm", actor=ACTOR)
        reloaded = service.load(TENANT_A, halted.case_ref)
        assert [e["source_column"] for e in reloaded.run.staged_mappings] \
            == [row["source_column"]]

    def test_an_answer_can_be_changed(self, service, halted):
        row = _one(service, halted, "proposed")
        updated = service.stage_mapping(
            halted, source_file=row["source_file"],
            source_column=row["source_column"], action="confirm", actor=ACTOR)
        updated = service.stage_mapping(
            updated, source_file=row["source_file"],
            source_column=row["source_column"], action="amend",
            target_field="borrower_1_age", actor=ACTOR)
        assert len(updated.run.staged_mappings) == 1
        assert updated.run.staged_mappings[0]["target_field"] \
            == "borrower_1_age"

    def test_an_answer_can_be_taken_back(self, service, halted):
        row = _one(service, halted, "proposed")
        updated = service.stage_mapping(
            halted, source_file=row["source_file"],
            source_column=row["source_column"], action="confirm", actor=ACTOR)
        updated = service.stage_mapping(
            updated, source_file=row["source_file"],
            source_column=row["source_column"], action="clear", actor=ACTOR)
        assert updated.run.staged_mappings == []
        after = next(r for r in _rows(service, updated)
                     if r["source_column"] == row["source_column"]
                     and r["source_file"] == row["source_file"])
        assert after["state"] == "proposed"

    def test_a_field_trakt_does_not_report_on_cannot_be_staged(self, service,
                                                               halted):
        """Caught here rather than at the commit. An operator who typed a name
        an hour ago should be told then, not when they press the button."""
        row = _one(service, halted, "proposed")
        with pytest.raises(OpsError) as caught:
            service.stage_mapping(
                halted, source_file=row["source_file"],
                source_column=row["source_column"], action="amend",
                target_field="made_up_field", actor=ACTOR)
        assert caught.value.code == "OCC_AGENT_FIELD_NOT_REGISTERED"

    def test_a_column_that_is_not_in_the_delivery_cannot_be_staged(
            self, service, halted):
        with pytest.raises(OpsError) as caught:
            service.stage_mapping(
                halted, source_file="harbourpoint_loan_extract_202606.csv",
                source_column="No Such Column", action="confirm", actor=ACTOR)
        assert caught.value.code == "OCC_AGENT_COLUMN_NOT_FOUND"

    def test_confirming_a_column_trakt_could_not_read_is_refused(self, service,
                                                                 halted):
        """"Confirm" means "what is on the row is right". On a row with no
        field there is nothing to be right, and accepting it would stage a
        mapping onto an empty field."""
        row = _one(service, halted, "unused")
        with pytest.raises(OpsError) as caught:
            service.stage_mapping(
                halted, source_file=row["source_file"],
                source_column=row["source_column"], action="confirm",
                actor=ACTOR)
        assert caught.value.code == "OCC_AGENT_NOTHING_TO_CONFIRM"


class TestTheCommit:
    def test_it_refuses_while_a_real_question_has_no_answer(self, service,
                                                            halted):
        """Scenario B halts on a genuine ambiguity. Committing the set around
        it would be the button answering a question nobody answered."""
        with pytest.raises(OpsError) as caught:
            service.confirm_mappings(halted, actor=ACTOR)
        assert caught.value.code == "OCC_AGENT_QUESTIONS_UNANSWERED"

    def test_it_applies_the_staged_and_the_untouched_together(self, service,
                                                             halted):
        updated = _answer_the_questions(service, halted)
        proposed = _rows(service, updated, "proposed")
        assert proposed, "the fixture no longer proposes anything"
        changed = proposed[0]
        updated = service.stage_mapping(
            updated, source_file=changed["source_file"],
            source_column=changed["source_column"], action="amend",
            target_field="borrower_1_age", actor=ACTOR)
        updated = service.confirm_mappings(updated, actor=ACTOR)
        rows = {(r["source_file"], r["source_column"]): r
                for r in _rows(service, updated)}
        # The one they changed took their field, and the ones they left alone
        # took the field Trakt read them as.
        assert rows[(changed["source_file"],
                     changed["source_column"])]["canonical_field"] \
            == "borrower_1_age"
        untouched = proposed[1]
        assert rows[(untouched["source_file"],
                     untouched["source_column"])]["canonical_field"] \
            == untouched["canonical_field"]

    def test_the_draft_is_emptied(self, service, halted):
        updated = _answer_the_questions(service, halted)
        updated = service.confirm_mappings(updated, actor=ACTOR)
        assert updated.run.staged_mappings == []

    def test_each_column_keeps_the_operator_who_read_it(self, service,
                                                        halted):
        """The reading is the judgement. A second operator pressing the button
        has not read the column, and a record crediting them with it is
        wrong."""
        updated = _answer_the_questions(service, halted)
        row = _one(service, updated, "proposed")
        updated = service.stage_mapping(
            updated, source_file=row["source_file"],
            source_column=row["source_column"], action="amend",
            target_field="borrower_1_age", actor=OTHER)
        updated = service.confirm_mappings(updated, actor=ACTOR)
        theirs = next(d for d in updated.run.open_decisions
                      if (d.get("subject") or {}).get("source_column")
                      == row["source_column"]
                      and (d.get("subject") or {}).get("source_file")
                      == row["source_file"])
        assert theirs["resolved_by"] == OTHER

    def test_it_promotes_what_was_committed(self, service, halted):
        from operations_control.occ_agent import mapping_promotion
        updated = _answer_the_questions(service, halted)
        updated = service.confirm_mappings(updated, actor=ACTOR)
        promoted = [mapping_promotion.mapping_of(d)
                    for d in updated.run.open_decisions]
        assert [m for m in promoted if m], "nothing reached promotion"

    def test_the_run_goes_on(self, service, halted):
        updated = _answer_the_questions(service, halted)
        updated = service.confirm_mappings(updated, actor=ACTOR)
        assert updated.run.stage_outcomes.get("validate") == \
            "deterministic_execution_completed"

    def test_a_set_aside_column_feeds_nothing_and_is_not_asked_again(
            self, service, halted):
        updated = _answer_the_questions(service, halted)
        row = _one(service, updated, "proposed")
        updated = service.stage_mapping(
            updated, source_file=row["source_file"],
            source_column=row["source_column"], action="not_used", actor=ACTOR)
        updated = service.confirm_mappings(updated, actor=ACTOR)
        after = next(r for r in _rows(service, updated)
                     if r["source_column"] == row["source_column"]
                     and r["source_file"] == row["source_file"])
        assert after["canonical_field"] == ""
        assert after["state"] == "confirmed"

    def test_there_is_nothing_to_commit_twice(self, service, halted):
        updated = _answer_the_questions(service, halted)
        updated = service.confirm_mappings(updated, actor=ACTOR)
        with pytest.raises(OpsError) as caught:
            service.confirm_mappings(updated, actor=ACTOR)
        assert caught.value.code in ("OCC_AGENT_NOTHING_TO_CONFIRM",
                                     "OCC_AGENT_ACTION_NOT_ALLOWED")

    def test_setting_the_losing_column_aside_does_not_drop_the_winner(
            self, service, halted):
        """An ambiguity is ONE decision about SEVERAL columns, and the operator
        answers per column.

        Writing each answer onto that shared decision made the last one
        overwrite the rest: setting the losing column aside stamped "not used"
        on the decision, and ``_approved_mappings`` reads that as BOTH columns
        unused — including the one just confirmed. The clash would then be
        re-raised on the rerun, for ever.
        """
        contested = [r for r in _rows(service, halted, "needs_you")
                     if any(c["same_file"] for c in r["also_claimed_by"])]
        assert len(contested) >= 2, "the fixture no longer has an ambiguity"
        winner, loser = contested[0], contested[1]
        assert winner["canonical_field"] == loser["canonical_field"]
        updated = halted
        for row in _rows(service, halted, "needs_you"):
            action = "confirm" if row is not loser and (
                row["source_column"] != loser["source_column"]) else "not_used"
            updated = service.stage_mapping(
                updated, source_file=row["source_file"],
                source_column=row["source_column"], action=action, actor=ACTOR)
        updated = service.confirm_mappings(updated, actor=ACTOR)
        rows = {(r["source_file"], r["source_column"]): r
                for r in _rows(service, updated)}
        kept = rows[(winner["source_file"], winner["source_column"])]
        dropped = rows[(loser["source_file"], loser["source_column"])]
        assert kept["canonical_field"] == winner["canonical_field"]
        assert dropped["canonical_field"] == ""
        # And it is settled: the rerun does not ask again.
        assert kept["state"] == "confirmed"
        assert dropped["state"] == "confirmed"

    def test_the_ambiguity_records_which_column_won(self, service, halted):
        """The shape ``_approved_mappings`` and ``mapping_promotion`` both read
        for an ambiguity — the answer names the COLUMN, not the field, which is
        the opposite of a confirmation."""
        contested = [r for r in _rows(service, halted, "needs_you")
                     if any(c["same_file"] for c in r["also_claimed_by"])]
        winner, loser = contested[0], contested[1]
        updated = halted
        for row in _rows(service, halted, "needs_you"):
            action = ("not_used" if row["source_column"] == loser["source_column"]
                      else "confirm")
            updated = service.stage_mapping(
                updated, source_file=row["source_file"],
                source_column=row["source_column"], action=action, actor=ACTOR)
        updated = service.confirm_mappings(updated, actor=ACTOR)
        shared = next(d for d in updated.run.open_decisions
                      if len((d.get("subject") or {}).get("source_columns")
                             or []) > 1)
        assert shared["status"] == "approved"
        assert shared["resolved_value"] == winner["source_column"]

    def test_the_record_says_what_was_read_and_what_was_taken_as_shown(
            self, service, halted):
        updated = _answer_the_questions(service, halted)
        row = _one(service, updated, "proposed")
        updated = service.stage_mapping(
            updated, source_file=row["source_file"],
            source_column=row["source_column"], action="amend",
            target_field="borrower_1_age", actor=ACTOR)
        updated = service.confirm_mappings(updated, actor=ACTOR)
        event = next(e for e in
                     service.store.list_audit(TENANT_A, updated.run.case_ref)
                     if e["action"] == "mappings_confirmed")
        detail = event["detail"]
        assert detail["changed"] == 1
        assert detail["as_proposed"] >= 1
        # Keyed by file AND column: every file in a pack carries a loan
        # identifier, and a record keyed on the name alone cannot say which.
        assert all("::" in k for k in detail["mappings"])


class TestTheButtonCanNameItsOwnConsequence:
    def test_the_overview_counts_what_a_commit_would_do(self, service,
                                                        halted):
        updated = _answer_the_questions(service, halted)
        row = _one(service, updated, "proposed")
        updated = service.stage_mapping(
            updated, source_file=row["source_file"],
            source_column=row["source_column"], action="confirm", actor=ACTOR)
        mapping = service.status(updated)["mapping"]
        assert mapping["unanswered_questions"] == 0
        assert mapping["staged"] >= 1
        assert mapping["to_confirm"] == mapping["staged"] + mapping["proposed"]

    def test_an_unanswered_question_is_counted_apart_from_a_proposal(
            self, service, halted):
        mapping = service.status(halted)["mapping"]
        assert mapping["unanswered_questions"] > 0
        assert mapping["proposed"] > 0


class TestTheStagingVocabulary:
    def test_setting_a_column_aside_resolves_as_an_empty_target(self):
        """The same string ``_approved_mappings`` reads as "do not use this
        column", so a set-aside column reaches the adapter as answered rather
        than as a question nobody answered."""
        assert _staging.resolved_value(
            {"action": _staging.ACTION_NOT_USED,
             "target_field": "anything"}) == "mark_unavailable"

    def test_a_change_resolves_as_an_amendment(self):
        """Promotion reads the word to know the answer names the FIELD rather
        than the column — the two directions a mapping question is asked are
        not symmetrical."""
        assert _staging.RESOLUTION[_staging.ACTION_AMEND] == "amend"
