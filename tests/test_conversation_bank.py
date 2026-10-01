"""The conversation bank is well formed (P0 design §34, D24).

The bank is the conversational agent's specification before it is built, so
its shape is checked now: the groups and counts its header states, the labels
it scores by, a stand-alone twin on every follow-up that is answered, and every
twin that names a production-bank question quoting it word for word — the
twin's own outcome is then already measured. The model-view guard
(`test_no_bank_question_is_in_the_models_view.py`) finds the file by its
directory, as it does every bank.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
import yaml

from mi_agent_api.question_bank import DEFAULT_BANKS, load_bank

_BANK = (Path(__file__).resolve().parents[1]
         / "config/mi/golden_questions/conversation_bank_20261001.yaml")

EXPECTS = {"answer", "carry", "ask_back", "fill", "fresh", "decline"}
RUNS = {"live", "code"}
CHANGES = {"amount_to_count", "statistic", "breakdown", "narrow", "widen_back",
           "measure", "ranking", "window", "date", "date_compare",
           "date_series", "change_form", "scenario", "target"}
#: Conversations and follow-ups per group, as the header states them.
GROUPS = {"A": (14, 42), "B": (5, 20), "C": (6, 8), "D": (4, 6),
          "E": (5, 10), "F": (6, 11)}


@pytest.fixture(scope="module")
def bank():
    return yaml.safe_load(_BANK.read_text(encoding="utf-8"))


def _turns(bank):
    for conversation in bank["conversations"]:
        for i, turn in enumerate(conversation["turns"]):
            yield conversation, i, turn


def test_the_groups_and_counts_are_as_stated(bank):
    conversations = bank["conversations"]
    assert len(conversations) == 40
    ids = [c["id"] for c in conversations]
    assert len(ids) == len(set(ids))
    for group, (n_conversations, n_follow_ups) in GROUPS.items():
        mine = [c for c in conversations if c["group"] == group]
        assert len(mine) == n_conversations, group
        assert sum(len(c["turns"]) - 1 for c in mine) == n_follow_ups, group
    assert bank["memory_minutes"] == 5                       # D24


def test_every_turn_is_labelled_to_be_scored(bank):
    for conversation, i, turn in _turns(bank):
        where = f"{conversation['id']}[{i}]"
        assert turn["expect"] in EXPECTS, where
        assert turn["run"] in RUNS, where
        assert turn["question"].strip(), where
        if turn["expect"] == "carry":
            assert turn.get("change") in CHANGES, where
        if i == 0:
            assert turn["expect"] in {"answer", "ask_back"}, where


def test_every_answered_follow_up_has_a_stand_alone_twin(bank):
    for conversation, i, turn in _turns(bank):
        if i and turn["expect"] in {"carry", "fill"}:
            assert turn.get("twin", {}).get("question"), (
                f"{conversation['id']}[{i}] has no twin")
        if turn["expect"] in {"ask_back", "decline"}:
            assert turn.get("note") or turn.get("slot"), (
                f"{conversation['id']}[{i}] says nothing of what is asked or why")


def test_a_fresh_turn_after_a_complete_question_is_its_own_twin(bank):
    """D24: a complete question never inherits, so its twin is itself."""
    for conversation, i, turn in _turns(bank):
        twin = turn.get("twin")
        if turn["expect"] == "fresh" and turn["run"] == "live":
            assert twin and twin["question"] == turn["question"], conversation["id"]


def test_every_bank_twin_quotes_the_production_bank(bank):
    production = {row["id"]: row["question"] for row in load_bank(DEFAULT_BANKS)}
    for conversation, i, turn in _turns(bank):
        for named in (turn, turn.get("twin") or {}):
            bank_id = named.get("bank_id")
            if bank_id:
                assert bank_id in production, f"{conversation['id']}[{i}]: {bank_id}"
                assert named["question"] == production[bank_id], (
                    f"{conversation['id']}[{i}] does not quote {bank_id} word for "
                    f"word: {named['question']!r} vs {production[bank_id]!r}")


def test_each_kind_of_change_is_tested_more_than_once(bank):
    changes = Counter(turn.get("change") for _, i, turn in _turns(bank)
                      if turn["expect"] == "carry")
    for change in CHANGES:
        assert changes[change] >= 2, (change, changes[change])


def test_the_memory_mechanics_are_tested_in_code(bank):
    setups = Counter()
    for _, _, turn in _turns(bank):
        for key, value in (turn.get("setup") or {}).items():
            setups[f"{key}={value}"] += 1
        if turn.get("setup"):
            assert turn["run"] == "code"
    for needed in ("idle_minutes_before=6", "book_changed_before=True",
                   "chat_cleared_before=True",
                   "memory_token=edited_to_another_book",
                   "memory_token=presented_by_another_user"):
        assert setups[needed], needed
