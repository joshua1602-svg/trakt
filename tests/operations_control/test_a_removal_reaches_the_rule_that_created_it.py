"""The operator removed the mapping. The governed store never heard.

From the live rule store, days after the operator took these away::

    'Latest Property Value' -> current_valuation_amount  v1 active 2026-09-18
       history (1 version(s))
    'Loan Interest Rate'    -> current_interest_rate     v1 active 2026-09-16
       history (1 version(s))

One version each. Nothing from the evening the operator removed them. The
Operations Control Centre showed both columns with no target and the badge
"You confirmed it · 100%", the governed store still mapped them, the engine
reads the store — so the removals did nothing at all, and the operator was
asked to confirm the same mappings again on the next run.

WHY, EXACTLY. ``mapping_of`` returns ``None`` for four different situations:
not a mapping decision, never resolved, rejected, or asserting no mapping.
``promote`` treated all four the same way — write no rule. For three of them
that is right. For the fourth it is the bug, because ``RuleStore.approve`` is
what supersedes a rule, and writing nothing leaves the previous version in
force.

Setting a column aside is a POSITIVE statement that it feeds nothing, and it
usually lands on a column that was mapped before. Both directions now reach
the store by the same governed path: ``retire``, which marks the rule
withdrawn and keeps its history, exactly as ``approve`` supersedes rather than
overwrites.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from operations_control.contracts import KIND_FIELD_MAPPING
from operations_control.occ_agent import mapping_promotion as mp
from operations_control.occ_agent.staging import NOT_USED_VALUE


class _Rule:
    def __init__(self, rule_id, column, canonical, kind=KIND_FIELD_MAPPING):
        self.rule_id, self.kind, self.version = rule_id, kind, 1
        self.payload = {"source_column": column, "canonical_field": canonical}


class _Store:
    """Enough of RuleStore to see which way each decision moved a rule."""

    def __init__(self, current: List[_Rule] | None = None):
        self.current = list(current or [])
        self.approved: List[Any] = []
        self.retired: List[Dict[str, str]] = []

    def approve(self, rule):
        rule.rule_id = rule.rule_id or f"rule_{len(self.approved)}"
        rule.version = 1
        self.approved.append(rule)
        return rule

    def list_current(self, _client_id):
        return list(self.current)

    def retire(self, _client_id, rule_id, *, by, reason):
        match = next((r for r in self.current if r.rule_id == rule_id), None)
        if match is None:
            return None
        self.retired.append({"rule_id": rule_id, "by": by, "reason": reason})
        match.version += 1
        return match


def _decision(column, target, *, resolution="approve", resolved_value="",
              status="approved", decision_type="mapping_confirmation"):
    return {"decision_type": decision_type, "status": status,
            "resolution": resolution, "source_column": column,
            "target_field": target, "resolved_value": resolved_value,
            "decision_id": f"d_{column}"}


def _set_aside(column, previous_target):
    """What "Do not use" settles to: an answered decision with no target."""
    return _decision(column, previous_target, resolution="amend",
                     resolved_value=NOT_USED_VALUE)


def _promote(store, decisions):
    return mp.promote(store, decisions, client_id="ERE",
                      portfolio_id="direct_001", workflow_id="wf_1")


class TestTheLiveRemoval:

    def test_setting_a_column_aside_retires_its_rule(self):
        store = _Store([_Rule("rule_f4c9cdbd93e4", "Latest Property Value",
                              "current_valuation_amount")])
        _promote(store, [_set_aside("Latest Property Value",
                                    "current_valuation_amount")])
        assert [r["rule_id"] for r in store.retired] == ["rule_f4c9cdbd93e4"]

    def test_the_withdrawal_says_why_and_who(self):
        """A rule leaving force is an audited act, like the one that made it."""
        store = _Store([_Rule("r1", "Loan Interest Rate", "current_interest_rate")])
        _promote(store, [_set_aside("Loan Interest Rate", "current_interest_rate")])
        assert store.retired[0]["by"] == mp.WITHDRAWN_BY
        assert "Loan Interest Rate" in store.retired[0]["reason"]
        assert "feeds nothing" in store.retired[0]["reason"]

    def test_the_withdrawal_is_reported_to_the_caller(self):
        store = _Store([_Rule("r1", "Latest Property Value",
                              "current_valuation_amount")])
        out = _promote(store, [_set_aside("Latest Property Value",
                                          "current_valuation_amount")])
        withdrawn = [o for o in out if o.get("withdrawn") == "true"]
        assert [o["source_column"] for o in withdrawn] == ["Latest Property Value"]

    def test_a_removal_and_a_confirmation_in_one_commit(self):
        """The pair the operator actually made: keep one, drop the other."""
        store = _Store([_Rule("r_drop", "Latest Property Value",
                              "current_valuation_amount")])
        _promote(store, [
            _set_aside("Latest Property Value", "current_valuation_amount"),
            _decision("Latest Valuation", "current_valuation_amount")])
        assert [r["rule_id"] for r in store.retired] == ["r_drop"]
        assert [str(r.payload["source_column"]) for r in store.approved] == [
            "Latest Valuation"]


class TestNothingIsWithdrawnByAccident:
    """Three of the four cases that reach here must still write nothing."""

    @pytest.mark.parametrize("decision", [
        _decision("Post Code", "postcode", status="open"),
        _decision("Region", "geographic_region", status="rejected"),
        _decision("X", "y", decision_type="something_else"),
        _decision("", ""),
    ])
    def test_an_unsettled_or_unrelated_decision_withdraws_nothing(self, decision):
        store = _Store([_Rule("r1", "Post Code", "postcode"),
                        _Rule("r2", "Region", "geographic_region")])
        _promote(store, [decision])
        assert store.retired == []

    def test_a_settled_mapping_withdraws_nothing(self):
        store = _Store([_Rule("r1", "Post Code", "postcode")])
        _promote(store, [_decision("Post Code", "postcode")])
        assert store.retired == []

    def test_a_column_that_was_never_mapped_has_nothing_to_withdraw(self):
        store = _Store([_Rule("r1", "Post Code", "postcode")])
        _promote(store, [_set_aside("A Column Nobody Mapped", "")])
        assert store.retired == []

    def test_only_the_named_column_s_rule_moves(self):
        store = _Store([_Rule("r_keep", "Post Code", "postcode"),
                        _Rule("r_drop", "Loan Interest Rate", "current_interest_rate")])
        _promote(store, [_set_aside("Loan Interest Rate", "current_interest_rate")])
        assert [r["rule_id"] for r in store.retired] == ["r_drop"]

    def test_a_rule_of_another_kind_is_left_alone(self):
        store = _Store([_Rule("r_alias", "Loan Interest Rate", "x", kind="alias")])
        _promote(store, [_set_aside("Loan Interest Rate", "current_interest_rate")])
        assert store.retired == []


class TestTheColumnIsMatchedHoweverItWasSpelled:

    @pytest.mark.parametrize("stored,decided", [
        ("Latest Property Value", "latest property value"),
        ("Latest  Property  Value", "Latest Property Value"),
        (" Latest Property Value ", "Latest Property Value"),
    ])
    def test_spacing_and_case_do_not_hide_the_rule(self, stored, decided):
        store = _Store([_Rule("r1", stored, "current_valuation_amount")])
        _promote(store, [_set_aside(decided, "current_valuation_amount")])
        assert [r["rule_id"] for r in store.retired] == ["r1"]
