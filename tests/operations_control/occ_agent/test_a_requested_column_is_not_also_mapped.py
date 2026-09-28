"""A column cannot be requested as a new field AND mapped to an old one.

WHAT THE OPERATOR SAW

A row of the live ERE case read:

    ERCs Paid in the period | Proposed | early repayment charge | Known alias | 100%

while the panel below the table read:

    early_repayment_charge_amount_in_period — ERCs Paid in the period ...

    "Why is this not updating?"

Two separate faults, and the display one was the lesser.

THE ASK DID NOT SET THE COLUMN ASIDE. ``request_registry_field`` recorded the
request and touched nothing else. Its own docstring promised otherwise — "the
column stays unmapped in this delivery, visibly, so nobody reads a request as a
mapping" — and that was true only while a request could only come from a column
that matched nothing. It cannot: "Change" opens the same dialog on ANY row. So
a column with a live proposal could carry an ask and keep its proposal, and
:meth:`confirm_mappings` approves *every untouched proposal*:

    untouched = [d for d in run.open_decisions
                 if ... _staging.key(...) not in staged]

A request stages nothing, so the column was untouched, so the commit mapped it.
One column, mapped and requested at once — and in this case the mapping was a
money column onto ``early_repayment_charge``, which is a ``Y/N`` field. A
decimal into a boolean parser, with a governed rule promoted for it at
activation. That is the ``protected_equity`` failure arriving by a new door.

THE CHIP WAS GATED ON THE SAME WRONG ASSUMPTION. ``MappingFieldCell`` read
``const requested = unmapped ? row.requested_field : ""``, so on any row that
was not ``unused`` the ask was thrown away before it reached the cell. The
operator's ask was recorded, correct, and invisible exactly where they were
looking.

WHAT THESE TESTS HOLD

An ask is an answer about the column — "Trakt has no field for this" — so it is
staged like any other answer: reversible until the set is committed, released
when the ask is withdrawn, and never overruling an answer the operator gave by
hand.
"""

from __future__ import annotations

import pytest

from operations_control.occ_agent import staging as _staging
from operations_control.occ_agent import states as _states
from operations_control.occ_agent.scenarios import run_scenario

from .conftest import ACTOR, TENANT_A


@pytest.fixture()
def halted(service):
    """A case stopped at the mapping table, mid-rehearsal."""
    run = run_scenario(service, "scenario_b_ambiguous_mapping",
                       tenant=TENANT_A, actor=ACTOR, resolve_decisions=False)
    assert run.case.run.state == _states.EXCEPTIONS_REQUIRE_INPUT
    return run.case


def _rows(service, agent_case):
    return {(r["source_file"], r["source_column"]): r
            for r in service.status(agent_case)["mapping"]["rows"]}


def _proposed(service, agent_case):
    """A column Trakt DID place — the shape the defect needed.

    The original request path was only ever exercised against a column that
    matched nothing, which is why nothing caught this: the failure needs a row
    that already carries a field.
    """
    rows = service.status(agent_case)["mapping"]["rows"]
    proposed = [r for r in rows
                if r["state"] == "proposed" and r["canonical_field"]]
    assert proposed, "the fixture no longer contains a proposed column"
    return proposed[0]


class TestAskingForAFieldOnAColumnThatAlreadyMatched:
    def test_the_ask_sets_the_column_aside(self, service, halted):
        """The docstring's promise, now kept by the code."""
        row = _proposed(service, halted)
        updated = service.request_registry_field(
            halted, source_file=row["source_file"],
            source_column=row["source_column"],
            field_name="early_repayment_charge_amount_in_period",
            description="ERCs paid in the reporting period.",
            data_type="decimal", actor=ACTOR)
        after = _rows(service, updated)[(row["source_file"],
                                         row["source_column"])]
        assert after["state"] == "staged"
        assert after["staged_action"] == _staging.ACTION_NOT_USED
        assert after["staged_origin"] == _staging.ORIGIN_REQUEST
        assert after["requested_field"] == \
            "early_repayment_charge_amount_in_period"

    def test_the_commit_does_not_map_it(self, service, halted):
        """THE DEFECT. The commit approves every untouched proposal, and a
        request used to leave the column untouched — so the operator's ask and
        a governed mapping onto the field they had just overruled went forward
        together."""
        row = _proposed(service, halted)
        field = row["canonical_field"]
        updated = service.request_registry_field(
            halted, source_file=row["source_file"],
            source_column=row["source_column"],
            field_name="early_repayment_charge_amount_in_period",
            actor=ACTOR)
        # Answer whatever is still genuinely open, as an operator would, then
        # commit the set.
        taken = set()
        for other in service.status(updated)["mapping"]["rows"]:
            if other["state"] != "needs_you":
                continue
            other_field = other["canonical_field"]
            action = ("confirm" if other_field and other_field not in taken
                      else "not_used")
            taken.add(other_field)
            updated = service.stage_mapping(
                updated, source_file=other["source_file"],
                source_column=other["source_column"], action=action,
                actor=ACTOR)
        updated = service.confirm_mappings(updated, actor=ACTOR)
        approved = service._approved_mappings(updated.run)
        key = f"{row['source_file']}::{row['source_column']}"
        assert approved.get(key) != field
        assert approved.get(key) in ("", _staging.NOT_USED_VALUE, None)

    def test_the_row_still_says_what_the_ask_displaced(self, service, halted):
        """The reading the operator overruled is the one thing they need to see
        when they come back to the row. It stays on the row — what changes is
        that it is no longer what the column feeds."""
        row = _proposed(service, halted)
        updated = service.request_registry_field(
            halted, source_file=row["source_file"],
            source_column=row["source_column"], field_name="broker_code",
            actor=ACTOR)
        after = _rows(service, updated)[(row["source_file"],
                                         row["source_column"])]
        assert after["canonical_field"] == row["canonical_field"]


class TestTakingTheAskBack:
    def test_withdrawing_releases_the_column(self, service, halted):
        """The ask is what put the column out, so taking it back puts it back.

        Otherwise withdrawing a mistaken request would silently leave the
        column unused — the opposite of what the operator meant, and invisible
        until the report came back short.
        """
        row = _proposed(service, halted)
        updated = service.request_registry_field(
            halted, source_file=row["source_file"],
            source_column=row["source_column"], field_name="broker_code",
            actor=ACTOR)
        updated = service.withdraw_registry_field_request(
            updated, source_file=row["source_file"],
            source_column=row["source_column"], actor=ACTOR,
            reason="asked in error")
        after = _rows(service, updated)[(row["source_file"],
                                         row["source_column"])]
        assert after["state"] == "proposed"
        assert after["staged_action"] == ""
        assert after["requested_field"] == ""

    def test_undoing_the_set_aside_takes_the_ask_back_too(self, service,
                                                          halted):
        """The same state, reached from the other end. Clearing the staged
        answer without withdrawing the ask would put the column back to
        proposed with the request still standing."""
        row = _proposed(service, halted)
        updated = service.request_registry_field(
            halted, source_file=row["source_file"],
            source_column=row["source_column"], field_name="broker_code",
            actor=ACTOR)
        updated = service.stage_mapping(
            updated, source_file=row["source_file"],
            source_column=row["source_column"],
            action=_staging.ACTION_CLEAR, actor=ACTOR)
        assert updated.run.field_requests[0]["status"] == "withdrawn"
        after = _rows(service, updated)[(row["source_file"],
                                         row["source_column"])]
        assert after["state"] == "proposed"

    def test_an_answer_the_operator_gave_by_hand_stands(self, service, halted):
        """A set-aside the OPERATOR made is not the request's to release.

        They read the column and said it feeds nothing. An unrelated ask being
        withdrawn is not them changing their mind, and a column quietly
        starting to feed a field again because of it is the worst direction
        this could fail in.
        """
        row = _proposed(service, halted)
        updated = service.stage_mapping(
            halted, source_file=row["source_file"],
            source_column=row["source_column"],
            action=_staging.ACTION_NOT_USED, actor=ACTOR,
            reason="we do not report this")
        updated = service.request_registry_field(
            updated, source_file=row["source_file"],
            source_column=row["source_column"], field_name="broker_code",
            actor=ACTOR)
        held = _staging.find(updated.run.staged_mappings, row["source_file"],
                             row["source_column"])
        assert held["origin"] == _staging.ORIGIN_OPERATOR
        assert held["reason"] == "we do not report this"

        updated = service.withdraw_registry_field_request(
            updated, source_file=row["source_file"],
            source_column=row["source_column"], actor=ACTOR)
        after = _rows(service, updated)[(row["source_file"],
                                         row["source_column"])]
        assert after["state"] == "staged"
        assert after["staged_action"] == _staging.ACTION_NOT_USED


class TestGivingTheColumnAFieldInstead:
    def test_it_replaces_the_ask_and_its_set_aside(self, service, halted):
        """Two answers to one question; the later one wins outright."""
        row = _proposed(service, halted)
        updated = service.request_registry_field(
            halted, source_file=row["source_file"],
            source_column=row["source_column"], field_name="broker_code",
            actor=ACTOR)
        updated = service.map_unmapped_column(
            updated, source_file=row["source_file"],
            source_column=row["source_column"],
            target_field="current_principal_balance", actor=ACTOR)
        assert updated.run.field_requests[0]["status"] == "withdrawn"
        held = _staging.find(updated.run.staged_mappings, row["source_file"],
                             row["source_column"])
        assert held["action"] == _staging.ACTION_AMEND
        assert held["target_field"] == "current_principal_balance"
        assert held["origin"] == _staging.ORIGIN_OPERATOR


class TestAColumnTheRunHadAlreadySettled:
    def test_asking_for_a_field_reopens_the_case(self, service, halted):
        """Requesting a field for a COMMITTED column is a mapping change, and
        costs what every other mapping change costs: the case goes back to this
        step and the approvals that rested on the old reading are withdrawn.

        It used to be refused outright — ``resolve_decision`` is not permitted
        once the set is committed — so the operator met a wall rather than a
        governed way back.
        """
        row = _proposed(service, halted)
        taken = set()
        committed = halted
        for other in service.status(committed)["mapping"]["rows"]:
            if other["state"] != "needs_you":
                continue
            other_field = other["canonical_field"]
            action = ("confirm" if other_field and other_field not in taken
                      else "not_used")
            taken.add(other_field)
            committed = service.stage_mapping(
                committed, source_file=other["source_file"],
                source_column=other["source_column"], action=action,
                actor=ACTOR)
        committed = service.confirm_mappings(committed, actor=ACTOR)
        assert committed.run.state != _states.EXCEPTIONS_REQUIRE_INPUT

        reopened = service.request_registry_field(
            committed, source_file=row["source_file"],
            source_column=row["source_column"],
            field_name="early_repayment_charge_amount_in_period",
            actor=ACTOR)
        assert reopened.run.state == _states.EXCEPTIONS_REQUIRE_INPUT
        event = next(e for e in
                     service.store.list_audit(TENANT_A, reopened.run.case_ref)
                     if e["action"] == "mapping_reopened")
        assert row["source_column"] in event["detail"]["columns"]
