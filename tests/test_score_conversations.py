"""The conversation scorer reads a run the way D25 scores it (§34, §39)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_PATH = (Path(__file__).resolve().parents[1]
         / "due_diligence/evidence/qb_plan_readback/score_conversations.py")
_spec = importlib.util.spec_from_file_location("score_conversations", _PATH)
sc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sc)

_TWIN = {"outcome": "ANSWERED", "plan_id": "p1",
         "answer": "Balance by Region — largest: South East £19.5m."}


def _turn(expect, outcome="ANSWERED", plan_id="p1", read_as="Show balance by region.",
          answer=None, reason=""):
    lead = f"Following on from your previous question, I read this as “{read_as}”. " \
        if read_as else ""
    return {"id": "cv_X01_t1", "category": "conversation_A", "expect": expect,
            "question": "By region?", "outcome": outcome, "plan_id": plan_id,
            "serving_reason": reason,
            "conversation": {"read_as": read_as} if read_as else {},
            "answer": answer or lead + _TWIN["answer"]}


def test_a_follow_up_with_its_twins_plan_and_a_stated_reading_passes():
    assert sc.score(_turn("carry"), _TWIN)[0] == "PASS"


def test_the_same_answer_with_another_plan_id_still_passes():
    assert sc.score(_turn("carry", plan_id="p2"), _TWIN)[0] == "PASS"


def test_a_different_figure_is_wrong():
    turn = _turn("carry", plan_id="p2", answer="Balance — £1.0m.")
    assert sc.score(turn, _TWIN)[0] == "WRONG"


def test_a_carry_into_a_complete_question_is_wrong():
    assert sc.score(_turn("fresh"), _TWIN)[0] == "WRONG"
    assert sc.score(_turn("fresh", read_as=None), _TWIN)[0] == "PASS"


def test_an_ask_back_is_a_miss_not_a_wrong_answer():
    turn = _turn("carry", outcome="REFUSED", plan_id=None, read_as=None,
                 answer="I need one more detail…", reason="CLARIFY_CONVERSATION")
    assert sc.score(turn, _TWIN) == ("MISS", "asked back")


def test_a_figure_where_a_decline_was_expected_is_wrong():
    assert sc.score(_turn("decline"), None)[0] == "WRONG"


def test_the_reading_and_the_lapse_are_set_aside_before_comparing():
    text = ("More than 5 minutes passed since I asked, so the earlier question "
            "has lapsed and I have read this on its own. X £1m.")
    assert sc.body(text) == "X £1m."
