"""The conversation bank's memory mechanics, enforced in code (§34, D24, D25).

Fifteen turns of `conversation_bank_20261001.yaml` are marked `run: code`:
expiry, a change of book, a cleared chat, an edited, widened, borrowed or
replayed memory, and access checked on every turn. They need no model — what
they test is what the memory token accepts — so they run on every build and
must pass 100%. Each setup the bank uses is mapped to what the token must do
with it; a setup this file does not know fails, so a mechanic added to the
bank cannot go untested.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
import yaml

from mi_agent import conversation as convo

_BANK = (Path(__file__).resolve().parents[2]
         / "config/mi/golden_questions/conversation_bank_20261001.yaml")
_KEY = "k" * 40
_T0 = 1_800_000_000.0
_USER, _BOOK, _CHAT = "user-a", "ERE||", "chat-1"


def _code_turns():
    data = yaml.safe_load(_BANK.read_text(encoding="utf-8"))
    for conversation in data["conversations"]:
        turns = conversation["turns"]
        if any(t.get("run") == "code" for t in turns):
            yield conversation["id"], turns


@pytest.fixture(autouse=True)
def _on(monkeypatch):
    monkeypatch.setenv(convo.SWITCH_ENV, "on")
    monkeypatch.setenv(convo.KEY_ENV, _KEY)


def _rewrite(token, **changes):
    encoded, signature = token.split(".", 1)
    body = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
    body.update(changes)
    return (base64.urlsafe_b64encode(json.dumps(
        body, sort_keys=True, separators=(",", ":")).encode()).decode().rstrip("=")
        + "." + signature)


def _outcome(setup, *, answered_at, answer_took=0.0):
    """What the token does with the next message under `setup`: the memory
    it hands back (`ok`), or why it is not used. The token is issued at
    DELIVERY — `answered_at + answer_took` — as the canary issues it."""
    delivered = answered_at + answer_took
    token = convo.issue(principal=_USER, book=_BOOK, chat=_CHAT,
                        memory=convo.Memory(last="the opening question"),
                        now=delivered)
    now = delivered + 60.0 * float(setup.get("idle_minutes_before", 1))
    principal, book, chat = _USER, _BOOK, _CHAT
    if setup.get("book_changed_before"):
        book = "ERE|ERE/2026-07-31|"
    if setup.get("chat_cleared_before"):
        chat = "chat-2"
    edit = setup.get("memory_token")
    if edit == "edited_to_another_book":
        token, book = _rewrite(token, c="OTHER||"), "OTHER||"
    elif edit == "scope_widened":
        token = _rewrite(token, c="ERE||*")
    elif edit == "presented_by_another_user":
        principal = "user-b"
    elif edit == "replayed_after_chat_cleared":
        chat = "chat-2"
    elif edit is not None:
        raise AssertionError(f"a memory_token setup this file does not know: {edit}")
    return convo.read(token, principal=principal, book=book, chat=chat, now=now)


_KNOWN = {"idle_minutes_before", "previous_answer_took_minutes",
          "book_changed_before", "chat_cleared_before", "memory_token"}


@pytest.mark.parametrize("cid, turns", list(_code_turns()), ids=lambda x: x
                         if isinstance(x, str) else "")
def test_every_code_turn_does_what_the_bank_says(cid, turns):
    for i, turn in enumerate(turns[1:], start=1):
        if turn.get("run") != "code":
            continue
        setup = turn.get("setup") or {}
        assert set(setup) <= _KNOWN, (cid, i, sorted(set(setup) - _KNOWN))
        returned = _outcome(setup, answered_at=_T0,
                            answer_took=60.0 * float(
                                setup.get("previous_answer_took_minutes", 0)))
        expect = turn["expect"]
        if expect == "carry":
            # Read against the opening answer: the memory is handed back.
            assert returned.ok, (cid, i, returned.lapsed)
            assert returned.memory.last == "the opening question"
        elif expect == "fresh":
            # Nothing carries: the memory is refused, and a stated lapse is
            # available to the answer (D24: never silent).
            assert not returned.ok, (cid, i)
            assert convo.lapsed_notice(returned.lapsed)
            if turn.get("notice") == "lapsed":
                assert returned.lapsed in (convo.LAPSED_EXPIRED,
                                           convo.LAPSED_OTHER_BOOK), (cid, i)
        elif expect == "decline":
            # ACCESS ON EVERY TURN (F02): a follow-up asks in the book the
            # request is authorised for. A memory carries no scope of its
            # own — presented against any other book it is refused, so it can
            # never widen what the user may see.
            other = convo.read(convo.issue(principal=_USER, book=_BOOK, chat=_CHAT,
                                           memory=convo.Memory(last="q"), now=_T0),
                               principal=_USER, book="OTHER||", chat=_CHAT,
                               now=_T0 + 60)
            assert other.lapsed == convo.LAPSED_OTHER_BOOK
        else:
            raise AssertionError(f"{cid} turn {i}: expect {expect!r} in code")


def test_the_bank_has_fifteen_code_turns():
    count = sum(1 for _, turns in _code_turns() for t in turns
                if t.get("run") == "code")
    assert count == 15
