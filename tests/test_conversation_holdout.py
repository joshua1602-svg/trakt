"""The held-out conversations are well formed and held out (P0 design §39.1).

The reader's reading conventions were drawn from the first conversation bank
(the 14:49 run of 2026-10-02). These 21 conversations test the same patterns
in other words, so they are kept in their own file, scored as their own group,
and checked here: labelled as the first set is, every answered follow-up with
a stand-alone twin, every production-bank twin quoted word for word, and no
follow-up that repeats one of the first set's. The model-view guard
(`test_no_bank_question_is_in_the_models_view.py`) finds the file by its
directory, as it does every bank.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
import yaml

from mi_agent_api.question_bank import (CONVERSATION_BANK, CONVERSATION_HOLDOUT,
                                        DEFAULT_BANKS, load_bank)

from tests.test_conversation_bank import CHANGES, EXPECTS, RUNS

_HOLDOUT = (Path(__file__).resolve().parents[1]
            / "config/mi/golden_questions/conversation_holdout_20261002.yaml")


@pytest.fixture(scope="module")
def holdout():
    return yaml.safe_load(_HOLDOUT.read_text(encoding="utf-8"))


def _turns(bank):
    for conversation in bank["conversations"]:
        for i, turn in enumerate(conversation["turns"]):
            yield conversation, i, turn


def _norm(text: str) -> str:
    return " ".join(str(text).lower().rstrip("?.!").split())


def test_the_runner_reads_this_file():
    assert CONVERSATION_HOLDOUT == _HOLDOUT


def test_the_set_is_as_its_header_states(holdout):
    conversations = holdout["conversations"]
    assert len(conversations) == 21
    assert sum(len(c["turns"]) - 1 for c in conversations) == 24
    assert {c["group"] for c in conversations} == {"H"}
    ids = [c["id"] for c in conversations]
    assert len(ids) == len(set(ids))
    assert holdout["memory_minutes"] == 5                    # D24


def test_no_conversation_id_is_the_first_sets():
    first = yaml.safe_load(CONVERSATION_BANK.read_text(encoding="utf-8"))
    held = yaml.safe_load(_HOLDOUT.read_text(encoding="utf-8"))
    assert not ({c["id"] for c in first["conversations"]}
                & {c["id"] for c in held["conversations"]})


def test_every_turn_is_labelled_to_be_scored(holdout):
    for conversation, i, turn in _turns(holdout):
        where = f"{conversation['id']}[{i}]"
        assert turn["expect"] in EXPECTS, where
        assert turn["run"] == "live" and turn["run"] in RUNS, where
        if turn["expect"] == "carry":
            assert turn.get("change") in CHANGES, where
        if i == 0:
            assert turn["expect"] == "answer", where
        if i and turn["expect"] in {"carry", "fresh"}:
            assert turn.get("twin", {}).get("question"), where
        if turn["expect"] in {"ask_back", "decline"}:
            assert turn.get("note"), where
        if turn["expect"] == "fresh":
            assert turn["twin"]["question"] == turn["question"], where


def test_every_bank_twin_quotes_the_production_bank(holdout):
    production = {row["id"]: row["question"] for row in load_bank(DEFAULT_BANKS)}
    for conversation, i, turn in _turns(holdout):
        for named in (turn, turn.get("twin") or {}):
            bank_id = named.get("bank_id")
            if bank_id:
                assert named["question"] == production[bank_id], (
                    f"{conversation['id']}[{i}]: {bank_id}")


def test_no_follow_up_repeats_the_first_sets(holdout):
    """Held out means other words: no follow-up is one of the first set's."""
    first = yaml.safe_load(CONVERSATION_BANK.read_text(encoding="utf-8"))
    asked = {_norm(turn["question"]) for _, i, turn in _turns(first) if i}
    repeats = [turn["question"] for _, i, turn in _turns(holdout)
               if i and _norm(turn["question"]) in asked]
    assert repeats == []


def test_the_patterns_the_rules_state_are_each_tested(holdout):
    """A grouping switched and added, a change taken back, a value narrowed
    to, a governed phrase kept, a complete question, a decline and an ask."""
    changes = Counter(turn.get("change") for _, i, turn in _turns(holdout) if i)
    for change in ("breakdown", "widen_back", "narrow", "measure", "window",
                   "date_series", "ranking", "scenario", "statistic",
                   "amount_to_count", "date"):
        assert changes[change], change
    expects = Counter(turn["expect"] for _, i, turn in _turns(holdout) if i)
    assert expects["fresh"] >= 2 and expects["decline"] >= 2
    assert expects["ask_back"] >= 1
